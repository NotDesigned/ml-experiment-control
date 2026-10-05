"""Read-only collection rejects ambiguous identities and bounds derived metrics."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ml_exp_server.actions.service import ActionService
from ml_exp_server.collectord import Collector, CollectorConfig
from ml_exp_server.controller_gateway import ProjectControllerGateway
from ml_exp_server.ingest import runscan
from ml_exp_server.ingest.indexer import RunIndex
from ml_exp_server.schemas import (ActionRuntimeConfig, CampaignBinding, CampaignRef,
    CampaignRelationship, ControllerConfig, EvidenceLayer, EvidenceLayers, ResearchProject, RunIndexRow)
from tests.test_collectord import _verified_action_store


@pytest.fixture
def canonical(tmp_path):
    authored = tmp_path / "project/campaign.yml"
    authored.parent.mkdir()
    payload = {"project": "elf", "campaign": "camp", "local_root": "outputs", "runs": [{"run_id": "run-a"}]}
    authored.write_text(yaml.safe_dump(payload))
    revision = "campaign." + hashlib.sha256(authored.read_bytes()).hexdigest()
    root = tmp_path / "daemon/runs"
    run_dir = root / "camp/run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.yaml").write_text("project: elf\ncampaign: camp\nrun_id: run-a\n")
    project = ResearchProject(project="elf", title="ELF", run_roots=[str(root)], daemon_run_root=root,
        controller=ControllerConfig(python="python3", experimentctl="ctl.py", workdir="."),
        base_dir=authored.parent, campaigns=[CampaignRef(name="camp", file=str(authored))])
    row = RunIndexRow(project="elf", run_id="run-a", campaign="camp", run_dir=str(run_dir), scheduler_state="RUNNING",
        campaign_binding=CampaignBinding(relationship=CampaignRelationship.MATCHED,
            origin_project="elf", origin_campaign="camp", origin_revision=revision, current_revision=revision))
    index = RunIndex(tmp_path / "index.sqlite")
    index.upsert_run(row)
    collector = Collector(index=index, projects=[project], config=CollectorConfig(execution_campaign_root=tmp_path / "campaigns"))
    yield collector, project, row, authored, payload
    index.close()


@pytest.mark.parametrize("fault", ["no-cache-root", "no-daemon-root", "missing-run", "wrong-path", "wrong-binding", "wrong-revision", "source-removed", "encoding", "cache-tamper", "cache-write"])
def test_canonical_execution_copy_fails_closed(canonical, monkeypatch, fault, tmp_path):
    from ml_exp_server import collectord as module
    collector, project, row, authored, payload = canonical
    if fault == "no-cache-root":
        collector.config.execution_campaign_root = None
    elif fault == "no-daemon-root":
        project.daemon_run_root = None
    elif fault == "missing-run":
        (Path(row.run_dir) / "manifest.yaml").unlink()
        Path(row.run_dir).rmdir()
    elif fault == "wrong-path":
        row.run_dir = str(tmp_path)
    elif fault == "wrong-binding":
        row.campaign_binding.origin_project = "other"
    elif fault == "wrong-revision":
        row.campaign_binding.current_revision = "campaign.wrong"
    elif fault == "source-removed":
        authored.unlink()
    elif fault == "encoding":
        def fail(*a, **kw):
            raise yaml.YAMLError("encoding")
        monkeypatch.setattr(module.yaml, "safe_dump", fail)
    elif fault == "cache-tamper":
        path, _ = collector._materialize_canonical_campaign(project, row, authored, payload)
        path.write_text("project: other")
    else:
        def fail(*a, **kw):
            raise OSError("read-only cache")
        monkeypatch.setattr(module, "atomic_text", fail)
    assert collector._materialize_canonical_campaign(project, row, authored, payload) is None


@pytest.mark.parametrize("content", [b'{"runs":[]}', b'[', b'[]', b'\xff'])
def test_campaign_json_reader_preserves_only_valid_mapping(tmp_path, content):
    path = tmp_path / "campaign.json"
    path.write_bytes(content)
    assert Collector._campaign_payload(path) == ({"runs": []} if content == b'{"runs":[]}' else None)


def test_authored_path_uses_exact_controller_root(canonical, tmp_path):
    collector, project, row, authored, payload = canonical
    assert collector._authored_run_dir(project, {**payload, "local_root": str(tmp_path)}, row) == tmp_path / "camp/run-a"
    assert collector._contains_exact_run({"runs": {}}, "run-a") is False
    assert collector._contains_exact_run({"runs": [{"run_id": "run-a"}, {"run_id": "run-a"}]}, "run-a") is False
    project.controller = None
    assert collector._authored_run_dir(project, payload, row) is None


@pytest.mark.parametrize("fault", ["bad-campaign", "missing-run", "authored-path", "empty-resolution"])
def test_poll_plan_requires_readable_unambiguous_files(canonical, monkeypatch, fault):
    collector, project, row, authored, payload = canonical
    if fault == "bad-campaign":
        authored.write_text("[")
    elif fault == "missing-run":
        (Path(row.run_dir) / "manifest.yaml").unlink()
        Path(row.run_dir).rmdir()
    elif fault == "authored-path":
        payload["local_root"] = str(project.daemon_run_root)
        authored.write_text(yaml.safe_dump(payload))
        assert collector._poll_campaign(project, row, authored) == (authored, None)
        return
    else:
        monkeypatch.setattr(collector, "_poll_campaign", lambda *a: (None, None))
    assert collector.plan_cycle() == []


def metric_row(root):
    return RunIndexRow(project="elf", run_id="run-a", run_dir=str(root),
        evidence=EvidenceLayers(scheduler=EvidenceLayer(state="RUNNING", attempt_id="attempt-001")))


@pytest.mark.parametrize("fault", ["missing", "nested-symlink", "nonobject", "mismatched", "invalid-step", "no-point"])
def test_metric_history_never_fabricates_or_escapes_attempt_identity(tmp_path, fault):
    root = tmp_path / "run"
    attempt = root / "attempts/attempt-001"
    attempt.mkdir(parents=True)
    if fault == "missing":
        attempt.rmdir()
    elif fault == "nested-symlink":
        nested = root / "attempts/nested/deeper"
        nested.mkdir(parents=True)
        attempt.rmdir()
        attempt.symlink_to(nested, target_is_directory=True)
    else:
        content = {
            "nonobject": [],
            "mismatched": {"attempt_id": "attempt-other", "latest_metric": {"step": 1}},
            "invalid-step": {"latest_metric": {"step": True}},
            "no-point": {"latest_metric": {}},
        }[fault]
        (attempt / "collection.json").write_text(json.dumps(content))
    Collector._persist_observed_metric(metric_row(root))
    assert not (attempt / "observed_train_metrics.jsonl").exists()


def test_metric_history_ignores_bad_old_steps_without_losing_valid_records(tmp_path):
    root = tmp_path / "run"
    attempt = root / "attempts/attempt-001"
    attempt.mkdir(parents=True)
    history = attempt / "observed_train_metrics.jsonl"
    history.write_text('{"step":true}\n{"step":"unknown"}\n')
    (attempt / "collection.json").write_text('{"latest_metric":{"step":2,"loss":0.5}}')
    Collector._persist_observed_metric(metric_row(root))
    unchanged = history.read_bytes()
    Collector._persist_observed_metric(metric_row(root))
    assert history.read_bytes() == unchanged
    assert json.loads(history.read_text().splitlines()[-1]) == {"step": 2, "loss": 0.5}


@pytest.mark.parametrize("missing_row", [False, True])
def test_metric_storage_failure_is_recorded_and_skips_decision(canonical, monkeypatch, missing_row):
    collector, project, row, authored, payload = canonical
    monkeypatch.setattr(collector, "_poll_campaign", lambda *a: (authored, None))
    collector.controller = ProjectControllerGateway(lambda *a, **kw: {"returncode": 0})
    if missing_row:
        monkeypatch.setattr(collector.index, "get_run", lambda *a: None)
    else:
        def fail(*a):
            raise OSError("disk full")
        monkeypatch.setattr(collector, "_persist_observed_metric", fail)
    calls = collector.run_cycle()
    assert [call.verb for call in calls] == ["observe", "decide"]
    status = collector.index.collector_statuses("elf")[0]
    if not missing_row:
        assert "observe failed" in status.last_error


@pytest.mark.parametrize("fault", ["unverified", "wrong-job", "symlink", "outside-root", "directory", "changed", "valid"])
def test_cancellation_uses_only_verified_immutable_execution_campaign(tmp_path, fault):
    source = tmp_path / "campaign.yml"
    source.write_text("campaign: camp\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    store, path, action = _verified_action_store(tmp_path, campaign=source, campaign_sha=digest)
    service = ActionService(store, ActionRuntimeConfig())
    if fault == "unverified":
        action["execution"]["status"] = "EXECUTING"
    elif fault == "wrong-job":
        action["execution"]["result"] = {"observation": {"backend_job_id": "other"}}
    elif fault == "symlink":
        path.unlink()
        path.symlink_to(source)
    elif fault == "outside-root":
        action["execution_campaign_file"] = str(source)
    elif fault == "directory":
        path.unlink()
        path.mkdir()
    elif fault == "changed":
        path.write_text("campaign: modified\n")
    else:
        action["execution"]["result"] = {"observation": {"backend_job_id": "job-123"}}
    result = service._verified_submission_campaign(project="elf", run_id="drifted-run", attempt_id="attempt-001", backend_job_id="job-123")
    assert result == (path if fault == "valid" else None)


def test_live_execution_reconciliation_leaves_claim_owner_untouched(tmp_path):
    from ml_exp_server.actions.store import ActionStore
    from tests.test_action_service_coverage import synthetic_plan
    store = ActionStore(tmp_path / "actions")
    aid = synthetic_plan(store, "live")
    execution = store.execution(aid)
    store.set_execution(aid, {**execution, "status": "EXECUTING"}, event="claimed")
    before = store.snapshot(aid)
    service = ActionService(store, ActionRuntimeConfig())
    assert service.reconcile(aid) == before
    assert store.snapshot(aid) == before


def test_restart_skips_legacy_records_without_execution_state():
    service = ActionService(SimpleNamespace(list_all=lambda: [{"execution": None}]), ActionRuntimeConfig())
    assert service.recover_interrupted_executions() == []


def test_metric_projection_bounds_keys_values_and_integral_steps():
    raw = {None: 1, "": 2, "x" * 129: 3, "flag": True, "count": 2, "step": 2.0, "loss": float("nan"), "object": [], "large": "x" * 1025}
    assert runscan.collection_latest_metric({"latest_metric": raw}) == {"flag": True, "count": 2, "step": 2}
    assert runscan._truthy(True) and not runscan._truthy(False)
    assert runscan._truthy("yes") and not runscan._truthy("disabled")


def test_train_fallback_skips_wrong_attempt_collection(tmp_path):
    attempt = tmp_path / "attempts/attempt-001"
    attempt.mkdir(parents=True)
    (attempt / "collection.json").write_text('{"attempt_id":"attempt-002","latest_metric":{"step":2}}')
    records, path, selected = runscan.train_metric_records(tmp_path, attempt_id="attempt-001", exact_attempt=True)
    assert records == [] and path is None and selected == "attempt-001"


def test_scan_exposes_training_metrics_without_promoting_timestamp(tmp_path):
    attempt = tmp_path / "attempts/attempt-001"
    attempt.mkdir(parents=True)
    (tmp_path / "manifest.yaml").write_text("project: elf\nrun_id: run-a\n")
    (attempt / "collection.json").write_text(json.dumps({"attempt_id": "attempt-001", "latest_metric": {"step": 2, "train_loss": 0.5, "timestamp": "now", "val_bpb": 1.0}}))
    (tmp_path / "collection.json").write_bytes((attempt / "collection.json").read_bytes())
    row = runscan.scan_run_dir(tmp_path, project="elf")
    assert row.latest_metrics["train_loss"] == 0.5
    assert "timestamp" not in row.latest_metrics
