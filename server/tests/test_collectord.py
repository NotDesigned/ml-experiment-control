"""Collector safety boundary: observation verbs only, terminal runs skipped."""

import errno
import hashlib
import os
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_exp_server.collectord import (
    Collector, CollectorConfig, CollectorLease, ForbiddenVerbError, OBSERVATION_VERBS,
)
from ml_exp_server.ingest.indexer import RunIndex, index_project
from ml_exp_server.schemas import (
    CampaignBinding,
    CampaignRef,
    CampaignRelationship,
    ControllerConfig,
    ResearchProject,
    AttemptSummary,
    EvidenceLayer,
    EvidenceLayers,
    RunIndexRow,
)
from tests.conftest import FIXTURES


def _project(tmp_path: Path, campaign_name: str) -> ResearchProject:
    campaign_file = tmp_path / "campaign.yml"
    campaign_file.write_text("schema_version: 1\ncampaign: " + campaign_name + "\n")
    return ResearchProject(
        project="elf", title="t", run_roots=[str(FIXTURES / "runs")],
        controller=ControllerConfig(python="/usr/bin/python3",
                                    experimentctl="tools/experimentctl.py",
                                    workdir="."),
        base_dir=tmp_path,
        campaigns=[CampaignRef(name=campaign_name, file=str(campaign_file))],
    )


@pytest.fixture
def collector(tmp_path):
    project = _project(tmp_path, "fusion-len256-gate-h100-20260711")
    index = RunIndex(tmp_path / "index.sqlite")
    index_project(index, project)
    return Collector(index=index, projects=[project],
                     config=CollectorConfig(dry_run=True))


@pytest.mark.parametrize("verb", ["submit", "cancel", "stage", "prepare", "assets-verify"])
def test_mutation_verbs_are_rejected(collector, tmp_path, verb):
    project = collector.projects[0]
    with pytest.raises(ForbiddenVerbError):
        collector.build_command(project, tmp_path / "campaign.yml", "run-x", verb)


def test_allowlist_is_exactly_observation_verbs():
    assert OBSERVATION_VERBS == {"observe", "decide", "status", "collect"}


def test_collector_defaults_to_twenty_second_polling():
    assert CollectorConfig().poll_interval_seconds == 20


def test_plan_cycle_targets_only_declared_nonterminal_runs(collector):
    # Execution discovery comes from the Project campaign catalog; this fixture
    # deliberately has no ResearchQuestion objects.
    assert collector.projects[0].research_questions == []
    calls = collector.plan_cycle()
    # A1 (RUNNING, campaign declared) gets observe+decide; the smoke run's
    # campaign is not declared in the project catalog → passive only.
    run_ids = {c.run_id for c in calls}
    assert run_ids == {"elf-a1-frozen-t5-l256-s42-h100-v1"}
    assert [c.verb for c in calls] == ["observe", "decide"]
    for call in calls:
        assert call.argv[3] in OBSERVATION_VERBS
        assert "--run" in call.argv


def test_terminal_runs_are_not_polled(tmp_path):
    project = _project(tmp_path, "camp")
    run_dir = tmp_path / "runs" / "camp" / "done-run"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.yaml").write_text("run_id: done-run\nproject: elf\ncampaign: camp\n")
    (run_dir / "status.json").write_text('{"state": "SUCCEEDED"}')
    project.run_roots = [str(tmp_path / "runs")]
    index = RunIndex(tmp_path / "index.sqlite")
    index_project(index, project)
    collector = Collector(index=index, projects=[project],
                          config=CollectorConfig(dry_run=True))
    assert collector.plan_cycle() == []


def test_revision_drift_is_not_polled_through_current_authored_campaign(tmp_path):
    project = _project(tmp_path, "camp")
    index = RunIndex(tmp_path / "index.sqlite")
    index.upsert_run(RunIndexRow(
        project="elf", campaign="camp", campaign_source="manifest",
        campaign_binding=CampaignBinding(
            relationship=CampaignRelationship.CAMPAIGN_REVISION_DRIFT,
            issues=[CampaignRelationship.CAMPAIGN_REVISION_DRIFT],
            origin_campaign="camp", origin_revision="campaign.old",
            current_revision="campaign.current",
        ),
        run_id="drifted-run", run_dir=str(tmp_path / "runs" / "camp" / "drifted-run"),
        scheduler_state="RUNNING",
    ))
    collector = Collector(index=index, projects=[project],
                          config=CollectorConfig(dry_run=True))
    assert collector.plan_cycle() == []


@pytest.mark.parametrize("scheduler_attempt", ["attempt-002", None])
def test_normal_observation_uses_exact_current_attempt(tmp_path, scheduler_attempt):
    project = _project(tmp_path, "camp")
    index = RunIndex(tmp_path / "index.sqlite")
    index.upsert_run(RunIndexRow(
        project="elf", campaign="camp", campaign_source="manifest",
        campaign_binding=CampaignBinding(
            relationship=CampaignRelationship.MATCHED,
            origin_project="elf", origin_campaign="camp",
        ),
        run_id="retry-run", run_dir=str(tmp_path / "runs" / "camp" / "retry-run"),
        scheduler_state="RUNNING",
        evidence=EvidenceLayers(scheduler=EvidenceLayer(
            state="RUNNING", attempt_id=scheduler_attempt,
        )),
        attempts=[
            AttemptSummary(
                attempt_id="attempt-001", state="FAILED", backend="slurm",
                backend_job_id="job-123", has_submission=True,
            ),
            AttemptSummary(
                attempt_id="attempt-002", state="RUNNING", backend="slurm",
                backend_job_id="job-456", has_submission=True,
            ),
        ],
    ))
    collector = Collector(
        index=index, projects=[project], config=CollectorConfig(dry_run=True),
    )

    calls = collector.plan_cycle()

    assert [call.verb for call in calls] == ["observe", "decide"]
    assert all(call.argv[-2:] == ["--attempt-id", "attempt-002"] for call in calls)


def _verified_action_store(
    tmp_path: Path, *, campaign: Path, campaign_sha: str,
    attempt_id: str = "attempt-001", backend_job_id: str = "job-123",
):
    root = tmp_path / "actions"
    action_id = "action-verified"
    action_root = root / action_id
    action_root.mkdir(parents=True)
    execution_campaign = action_root / "campaign.execution.yml"
    execution_campaign.write_bytes(campaign.read_bytes())
    action = {
        "action_id": action_id,
        "operation": "SUBMIT_RUN",
        "scope": {"project": "elf", "scope_type": "run", "object_id": "drifted-run"},
        "run_id": "drifted-run",
        "attempt_id": attempt_id,
        "created_at": "2026-07-15T00:00:00Z",
        "execution_campaign_file": str(execution_campaign),
        "execution_campaign_sha256": f"sha256:{campaign_sha}",
        "execution": {
            "status": "VERIFIED",
            "result": {"submission": {"backend_job_id": backend_job_id}},
        },
    }
    return SimpleNamespace(
        list_all=lambda: [action],
        directory=lambda value: root / value,
    ), execution_campaign, action


def _drifted_row(tmp_path: Path, campaign_sha: str) -> RunIndexRow:
    return RunIndexRow(
        project="elf", campaign="camp", campaign_source="manifest",
        campaign_binding=CampaignBinding(
            relationship=CampaignRelationship.CAMPAIGN_REVISION_DRIFT,
            issues=[CampaignRelationship.CAMPAIGN_REVISION_DRIFT],
            origin_project="elf", origin_campaign="camp",
            origin_revision=f"campaign.{campaign_sha}",
            current_revision="campaign.current",
        ),
        run_id="drifted-run", run_dir=str(tmp_path / "runs" / "camp" / "drifted-run"),
        scheduler_state="RUNNING",
        evidence=EvidenceLayers(scheduler=EvidenceLayer(
            state="RUNNING", attempt_id="attempt-001",
        )),
        attempts=[AttemptSummary(
            attempt_id="attempt-001", state="RUNNING", backend="slurm",
            backend_job_id="job-123", has_submission=True,
        )],
    )


def test_revision_drift_polls_exact_verified_execution_campaign(tmp_path):
    project = _project(tmp_path, "camp")
    execution_source = tmp_path / "execution-source.yml"
    execution_source.write_text(
        "schema_version: 1\ncampaign: camp\nlocal_root: /daemon/runs\n"
    )
    campaign_sha = hashlib.sha256(execution_source.read_bytes()).hexdigest()
    action_store, execution_campaign, _ = _verified_action_store(
        tmp_path, campaign=execution_source, campaign_sha=campaign_sha,
    )
    index = RunIndex(tmp_path / "index.sqlite")
    index.upsert_run(_drifted_row(tmp_path, campaign_sha))
    collector = Collector(
        index=index, projects=[project], action_store=action_store,
        config=CollectorConfig(dry_run=True),
    )

    calls = collector.plan_cycle()

    assert [call.verb for call in calls] == ["observe", "decide"]
    assert all(call.argv[2] == str(execution_campaign) for call in calls)
    assert all(call.argv[-2:] == ["--attempt-id", "attempt-001"] for call in calls)


def test_run_scoped_evaluation_can_bind_execution_campaign(tmp_path):
    project = _project(tmp_path, "camp")
    execution_source = tmp_path / "execution-source.yml"
    execution_source.write_text(
        "schema_version: 1\ncampaign: camp\nlocal_root: /daemon/runs\n"
    )
    campaign_sha = hashlib.sha256(execution_source.read_bytes()).hexdigest()
    action_store, execution_campaign, action = _verified_action_store(
        tmp_path, campaign=execution_source, campaign_sha=campaign_sha,
    )
    action["operation"] = "RUN_EVALUATION"
    index = RunIndex(tmp_path / "index.sqlite")
    index.upsert_run(_drifted_row(tmp_path, campaign_sha))
    collector = Collector(
        index=index, projects=[project], action_store=action_store,
        config=CollectorConfig(dry_run=True),
    )

    calls = collector.plan_cycle()

    assert [call.verb for call in calls] == ["observe", "decide"]
    assert all(call.argv[2] == str(execution_campaign) for call in calls)


def test_reconciled_submission_observation_can_bind_execution_campaign(tmp_path):
    project = _project(tmp_path, "camp")
    execution_source = tmp_path / "execution-source.yml"
    execution_source.write_text(
        "schema_version: 1\ncampaign: camp\nlocal_root: /daemon/runs\n"
    )
    campaign_sha = hashlib.sha256(execution_source.read_bytes()).hexdigest()
    action_store, execution_campaign, action = _verified_action_store(
        tmp_path, campaign=execution_source, campaign_sha=campaign_sha,
    )
    action["execution"]["result"] = {
        "submission": None,
        "observation": {"backend_job_id": "job-123", "state": "QUEUED"},
    }
    index = RunIndex(tmp_path / "index.sqlite")
    index.upsert_run(_drifted_row(tmp_path, campaign_sha))
    collector = Collector(
        index=index, projects=[project], action_store=action_store,
        config=CollectorConfig(dry_run=True),
    )

    calls = collector.plan_cycle()

    assert [call.verb for call in calls] == ["observe", "decide"]
    assert all(call.argv[2] == str(execution_campaign) for call in calls)


def test_revision_drift_requires_current_submitted_attempt(tmp_path):
    project = _project(tmp_path, "camp")
    execution_source = tmp_path / "execution-source.yml"
    execution_source.write_text("schema_version: 1\ncampaign: camp\n")
    campaign_sha = hashlib.sha256(execution_source.read_bytes()).hexdigest()
    action_store, _, _ = _verified_action_store(
        tmp_path, campaign=execution_source, campaign_sha=campaign_sha,
    )
    row = _drifted_row(tmp_path, campaign_sha)
    row.attempts = []
    index = RunIndex(tmp_path / "index.sqlite")
    index.upsert_run(row)
    collector = Collector(
        index=index, projects=[project], action_store=action_store,
        config=CollectorConfig(dry_run=True),
    )

    assert collector.plan_cycle() == []


@pytest.mark.parametrize("failure", ["directory", "read_error"])
def test_revision_drift_rejects_non_file_or_unreadable_campaign(
    tmp_path, monkeypatch, failure,
):
    project = _project(tmp_path, "camp")
    execution_source = tmp_path / "execution-source.yml"
    execution_source.write_text("schema_version: 1\ncampaign: camp\n")
    campaign_sha = hashlib.sha256(execution_source.read_bytes()).hexdigest()
    action_store, execution_campaign, action = _verified_action_store(
        tmp_path, campaign=execution_source, campaign_sha=campaign_sha,
    )
    if failure == "directory":
        action["execution_campaign_file"] = str(execution_campaign.parent)
    index = RunIndex(tmp_path / "index.sqlite")
    index.upsert_run(_drifted_row(tmp_path, campaign_sha))
    collector = Collector(
        index=index, projects=[project], action_store=action_store,
        config=CollectorConfig(dry_run=True),
    )
    if failure == "read_error":
        monkeypatch.setattr(
            collector, "_file_sha256",
            lambda _path: (_ for _ in ()).throw(OSError("unreadable")),
        )

    assert collector.plan_cycle() == []


def test_plan_cycle_ignores_action_without_mapping_scope(tmp_path):
    index = RunIndex(tmp_path / "index.sqlite")
    action_store = SimpleNamespace(list_all=lambda: [{"scope": "invalid"}])
    collector = Collector(
        index=index, projects=[], action_store=action_store,
        config=CollectorConfig(dry_run=True),
    )

    assert collector.plan_cycle() == []


@pytest.mark.parametrize(
    "tamper",
    [
        "campaign", "digest", "job", "job_conflict", "attempt", "status",
        "scope", "path", "origin", "action_id", "symlink",
    ],
)
def test_revision_drift_execution_campaign_binding_fails_closed(tmp_path, tamper):
    project = _project(tmp_path, "camp")
    execution_source = tmp_path / "execution-source.yml"
    execution_source.write_text(
        "schema_version: 1\ncampaign: camp\nlocal_root: /daemon/runs\n"
    )
    campaign_sha = hashlib.sha256(execution_source.read_bytes()).hexdigest()
    action_store, execution_campaign, action = _verified_action_store(
        tmp_path, campaign=execution_source, campaign_sha=campaign_sha,
    )
    row = _drifted_row(tmp_path, campaign_sha)
    if tamper == "campaign":
        execution_campaign.write_text("schema_version: 1\ncampaign: tampered\n")
    elif tamper == "digest":
        action["execution_campaign_sha256"] = "sha256:" + "0" * 64
    elif tamper == "job":
        action["execution"]["result"]["submission"]["backend_job_id"] = "job-other"
    elif tamper == "job_conflict":
        action["execution"]["result"]["observation"] = {
            "backend_job_id": "job-other",
        }
    elif tamper == "attempt":
        action["attempt_id"] = "attempt-002"
    elif tamper == "scope":
        action["scope"]["object_id"] = "other-run"
    elif tamper == "path":
        action["execution_campaign_file"] = str(execution_source)
    elif tamper == "origin":
        row.campaign_binding.origin_project = "other-project"
    elif tamper == "action_id":
        action["action_id"] = "invalid"
    elif tamper == "symlink":
        execution_campaign.unlink()
        execution_campaign.symlink_to(execution_source)
    else:
        action["execution"]["status"] = "AUTHORIZED"
    index = RunIndex(tmp_path / "index.sqlite")
    index.upsert_run(row)
    collector = Collector(
        index=index, projects=[project], action_store=action_store,
        config=CollectorConfig(dry_run=True),
    )

    assert collector.plan_cycle() == []


def test_run_cycle_executes_and_records_errors(tmp_path):
    """A fake experimentctl proves subprocess wiring + error capture."""
    project = _project(tmp_path, "fusion-len256-gate-h100-20260711")
    fake_ctl = tmp_path / "tools" / "experimentctl.py"
    fake_ctl.parent.mkdir(parents=True)
    log = tmp_path / "calls.log"
    fake_ctl.write_text(textwrap.dedent(f"""\
        import sys
        open({str(log)!r}, "a").write(" ".join(sys.argv[1:]) + "\\n")
        if sys.argv[2] == "decide":
            print("boom", file=sys.stderr); sys.exit(1)
    """))
    project.controller = ControllerConfig(python=sys_python(),
                                          experimentctl=str(fake_ctl), workdir=".")
    index = RunIndex(tmp_path / "index.sqlite")
    index_project(index, project)
    collector = Collector(index=index, projects=[project],
                          config=CollectorConfig(dry_run=False))
    calls = collector.run_cycle()
    assert len(calls) == 2
    logged = log.read_text().splitlines()
    assert len(logged) == 2 and "observe" in logged[0] and "decide" in logged[1]
    statuses = {s.last_verb: s for s in index.collector_statuses("elf")}
    # decide ran last and failed; its error must be recorded.
    assert statuses["decide"].last_error == "boom"
    assert index.get_meta("collector_last_cycle_at") is not None


def test_collector_lease_allows_exactly_one_owner(tmp_path):
    first = CollectorLease(tmp_path / "index.sqlite")
    second = CollectorLease(tmp_path / "index.sqlite")
    assert first.acquire() is True
    assert second.acquire() is False
    first.release()
    assert second.acquire() is True
    second.release()


def test_collector_lease_is_idempotent_and_release_without_owner(tmp_path):
    lease = CollectorLease(tmp_path / "index.sqlite")
    lease.release()
    assert lease.acquire() is True
    assert lease.acquire() is True
    lease.release()


@pytest.mark.parametrize(("error_number", "expected"), [
    (errno.EACCES, False),
    (errno.EAGAIN, False),
    (errno.EIO, None),
])
def test_collector_lease_maps_flock_os_errors(
    monkeypatch, tmp_path, error_number, expected,
):
    import fcntl

    lease = CollectorLease(tmp_path / "index.sqlite")
    monkeypatch.setattr(
        fcntl, "flock",
        lambda *_args: (_ for _ in ()).throw(OSError(error_number, "lock")),
    )
    if expected is False:
        assert lease.acquire() is False
    else:
        with pytest.raises(OSError) as caught:
            lease.acquire()
        assert caught.value.errno == error_number


def test_collector_lease_rejects_zero_length_write_and_preserves_primary_error(
    monkeypatch, tmp_path,
):
    lease = CollectorLease(tmp_path / "index.sqlite")
    real_close = os.close
    monkeypatch.setattr(os, "write", lambda *_args: 0)

    def close(fd):
        real_close(fd)
        raise OSError("close")

    monkeypatch.setattr(os, "close", close)
    with pytest.raises(OSError, match="could not write workspace lease metadata"):
        lease.acquire()


def test_lease_metadata_write_failure_releases_fd_for_retry(monkeypatch, tmp_path):
    lease = CollectorLease(tmp_path / "index.sqlite")
    real_ftruncate = os.ftruncate

    def fail_metadata_write(fd, length):
        raise OSError("lease metadata disk failure")

    monkeypatch.setattr(os, "ftruncate", fail_metadata_write)
    with pytest.raises(OSError, match="lease metadata disk failure"):
        lease.acquire()
    assert lease._fd is None

    monkeypatch.setattr(os, "ftruncate", real_ftruncate)
    assert lease.acquire() is True
    lease.release()


def test_campaign_catalog_handles_empty_relative_absolute_and_missing_entries(
    tmp_path,
):
    relative = tmp_path / "relative.yml"
    absolute = tmp_path / "absolute.yml"
    relative.write_text("campaign: relative\n")
    absolute.write_text("campaign: absolute\n")
    project = ResearchProject(
        project="demo", title="Demo", run_roots=[], base_dir=tmp_path,
        campaigns=[
            CampaignRef(name="empty", file=None),
            CampaignRef(name="relative", file="relative.yml"),
            CampaignRef(name="absolute", file=str(absolute)),
            CampaignRef(name="missing", file="missing.yml"),
        ],
    )
    value = Collector(RunIndex(tmp_path / "index.sqlite"), [project])
    assert value._campaign_files(project) == {
        "relative": relative.resolve(), "absolute": absolute,
    }
    project.campaigns = []
    assert value._campaign_files(project) == {}


def test_dry_run_cycle_and_inactive_campaign_skip(monkeypatch, collector):
    calls = collector.run_cycle()
    assert [call.verb for call in calls] == ["observe", "decide"]
    monkeypatch.setattr(
        "ml_exp_server.collectord.campaign_snapshot",
        lambda *_args: {"lifecycle_state": "COMPLETED"},
    )
    assert collector.plan_cycle() == []


def test_observe_failure_is_preserved_and_skips_decide(tmp_path):
    project = _project(tmp_path, "fusion-len256-gate-h100-20260711")
    fake_ctl = tmp_path / "tools" / "experimentctl.py"
    fake_ctl.parent.mkdir(parents=True)
    log = tmp_path / "calls.log"
    fake_ctl.write_text(textwrap.dedent(f"""\
        import sys
        open({str(log)!r}, "a").write(" ".join(sys.argv[1:]) + "\\n")
        if sys.argv[2] == "observe":
            print("ssh timeout", file=sys.stderr); sys.exit(1)
    """))
    project.controller = ControllerConfig(
        python=sys_python(), experimentctl=str(fake_ctl), workdir="."
    )
    index = RunIndex(tmp_path / "index.sqlite")
    index_project(index, project)
    collector = Collector(index=index, projects=[project])

    calls = collector.run_cycle()

    assert len(calls) == 2
    assert len(log.read_text().splitlines()) == 1
    status = index.collector_statuses("elf")[0]
    assert "ssh timeout" in status.last_error
    assert status.verb_results["observe"]["outcome"] == "failed"
    assert status.verb_results["decide"]["outcome"] == "skipped"
    assert "observe failed" in status.verb_results["decide"]["error"]


def sys_python() -> str:
    import sys
    return sys.executable
