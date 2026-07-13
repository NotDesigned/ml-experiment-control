"""Deep branch coverage for application read models and adapter error mapping."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import ml_exp_server.application as module
from ml_exp_server.application import (
    ApplicationError,
    ExperimentServerApplication,
    compact_evidence,
    structured_failure_summary,
)
from ml_exp_server.schemas import AgentScope, AgentScopeType, CampaignRelationship


class Dump:
    def __init__(self, **values):
        self.__dict__.update(values)

    def model_dump(self, **kwargs):
        return dict(self.__dict__)


def scope(kind=AgentScopeType.RUN, object_id="run-a"):
    return AgentScope(project="demo", scope_type=kind, object_id=object_id)


def application(**runtime_values):
    runtime = SimpleNamespace(**runtime_values)
    return ExperimentServerApplication(runtime)


def raises(error):
    raise error


def test_compact_evidence_depth_collection_limit_and_string_limit():
    nested = value = {}
    for _ in range(9):
        value["child"] = {}
        value = value["child"]
    payload = compact_evidence({
        "stdout_tail": ["secret"], "items": list(range(45)),
        "long": "x" * 2001, "nested": nested,
    })
    assert "stdout_tail" not in payload
    assert payload["items"][-1] == "[5 additional records omitted]"
    assert payload["long"].endswith("…[truncated]")
    assert "[nested evidence omitted]" in str(payload["nested"])


@pytest.mark.parametrize(
    ("collection", "decision", "signature", "failure_class"),
    [
        ({"process_evidence": {"stderr_tail": ["ModuleNotFoundError: x"]}}, None,
         "MISSING_PYTHON_MODULE", "configuration"),
        ({"process_evidence": {"stderr_tail": ["no kernel image is available"]}}, None,
         "UNSUPPORTED_CUDA_KERNEL", "configuration"),
        ({"process_evidence": {"stdout_tail": ["TIMEOUT"]}}, None,
         "TIMEOUT", "timeout"),
        ({"process_state": "FAILED"}, None,
         "UNCLASSIFIED_PROCESS_FAILURE", "unknown"),
        ({"failure_class": "none"}, {"failure_class": "data"},
         None, None),
    ],
)
def test_structured_failure_fallback_signatures(collection, decision, signature, failure_class):
    result = structured_failure_summary(collection, decision)
    if signature is None:
        assert result is None
    else:
        assert result["failure_signature"] == signature
        assert result["failure_class"] == failure_class


@pytest.mark.parametrize(
    ("kind", "object_id", "code"),
    [
        (AgentScopeType.PROJECT, "other", "UNKNOWN_PROJECT"),
        (AgentScopeType.RESEARCH_QUESTION, "missing", "UNKNOWN_RESEARCH_QUESTION"),
        (AgentScopeType.CAMPAIGN, "missing", "UNKNOWN_CAMPAIGN"),
        (AgentScopeType.RUN, "missing", "UNKNOWN_RUN"),
        (AgentScopeType.ATTEMPT, "invalid", "INVALID_ATTEMPT_ID"),
        (AgentScopeType.ATTEMPT, "missing::a1", "UNKNOWN_RUN"),
        (AgentScopeType.ATTEMPT, "run-a::missing", "UNKNOWN_ATTEMPT"),
    ],
)
def test_resolve_scope_maps_every_missing_identity(kind, object_id, code):
    run = SimpleNamespace(attempts=[])
    configured = SimpleNamespace(
        project="demo", research_questions=[], campaigns=[],
    )
    index = SimpleNamespace(get_run=lambda project, run_id: run if run_id == "run-a" else None)
    app = application(project=lambda name: configured, index=index)
    with pytest.raises(ApplicationError) as caught:
        app.resolve_scope("demo", kind, object_id)
    assert caught.value.code == code


def test_resolve_scope_unknown_project_maps_key_error():
    app = application(project=lambda name: raises(KeyError("unknown demo")))
    with pytest.raises(ApplicationError) as caught:
        app.resolve_scope("demo", AgentScopeType.PROJECT, "demo")
    assert caught.value.status_code == 404


def test_campaign_context_skips_excluded_peer_and_preserves_unbound_peer(monkeypatch):
    membership = Dump(included_in_analysis=True)
    binding = Dump(campaign="study", revision_id="r1", membership=membership)
    excluded = SimpleNamespace(
        run_id="excluded", campaign_memberships=[Dump(
            campaign="study", membership=Dump(included_in_analysis=False),
        )], scheduler_state="DONE", latest_metrics={}, eval_metrics={}, provenance={},
    )
    peer = SimpleNamespace(
        run_id="peer", campaign_memberships=[], scheduler_state="DONE",
        latest_metrics={}, eval_metrics={}, provenance={},
    )
    index = SimpleNamespace(list_runs=lambda *args: [excluded, peer])
    app = application(index=index)
    revision = SimpleNamespace(revision_id="r1", research_contract={"question": "q"})
    configured = SimpleNamespace(
        project="demo", campaigns=[SimpleNamespace(name="study", current_revision=revision)],
    )
    monkeypatch.setattr(module, "campaign_snapshot", lambda *args: {"lifecycle_state": "ACTIVE"})
    result = app.campaign_contexts(configured, SimpleNamespace(campaign_memberships=[binding]))
    assert [item["run_id"] for item in result[0]["comparator_runs"]] == ["peer"]
    assert result[0]["comparator_runs"][0]["membership"] is None


def test_bounded_evidence_all_scope_shapes(monkeypatch):
    layer = SimpleNamespace(stale=False)
    evidence = Dump(**{name: layer for name in (
        "scheduler", "worker", "process", "model", "evaluation",
    )})
    binding = Dump(relationship=CampaignRelationship.MATCHED)
    row = SimpleNamespace(
        run_id="run-a", campaign="study", role="arm", campaign_binding=binding,
        campaign_memberships=[], scheduler_state="DONE", decision={}, evidence=evidence,
        latest_metrics={}, eval_metrics={}, eval_variants=[], canonical_eval_variant_id=None,
        checkpoint={}, artifacts={}, provenance={}, warnings=[], evidence_conflicts=[], attempts=[],
    )
    index = SimpleNamespace(list_runs=lambda *args, **kwargs: [row], get_run=lambda *args: row)
    app = application(index=index)
    app.campaign_contexts = lambda *args: []
    question = Dump(id="q1", title="Q", status="OPEN", links=SimpleNamespace(campaigns=["study"]))
    campaign = Dump(name="study")
    configured = SimpleNamespace(
        project="demo", title="Demo", research_questions=[question], campaigns=[campaign],
    )
    monkeypatch.setattr(module, "campaign_snapshot", lambda *args: {"lifecycle_state": "ACTIVE"})
    assert app.bounded_evidence(scope(AgentScopeType.PROJECT, "demo"), configured, configured)["runs"]
    assert app.bounded_evidence(scope(AgentScopeType.RESEARCH_QUESTION, "q1"), configured, question)["runs"]
    assert app.bounded_evidence(scope(AgentScopeType.CAMPAIGN, "study"), configured, campaign)["runs"]
    assert app.bounded_evidence(scope(), configured, row)["run"]["run_id"] == "run-a"
    attempt = Dump(attempt_id="a1")
    assert app.bounded_evidence(
        scope(AgentScopeType.ATTEMPT, "run-a::a1"), configured, attempt,
    )["attempt"]["attempt_id"] == "a1"


def test_snapshot_and_object_show_plain_object():
    target = scope()
    stores = SimpleNamespace(
        snapshot=lambda value: {"state": "IDLE"},
    )
    runtime = SimpleNamespace(
        agent_store=stores,
        action_store=SimpleNamespace(list_for_scope=lambda value: [{"action_id": "a"}]),
    )
    app = ExperimentServerApplication(runtime)
    app.bounded_evidence = lambda *args: {"bounded": True}
    result = app._snapshot(target, object(), object())
    assert result["actions"][0]["action_id"] == "a"
    app.resolve_scope = lambda *args: (target, object(), {"raw": "value"})
    shown = app.object_show("demo", AgentScopeType.RUN, "run-a")
    assert shown["object"] == {"raw": "value"}


def test_campaign_list_status_and_status_error(monkeypatch):
    configured = SimpleNamespace(campaigns=[SimpleNamespace(name="one")])
    app = application(project=lambda name: configured, index=object())
    monkeypatch.setattr(module, "campaign_snapshot", lambda *args: {"campaign": args[-1]})
    assert app.campaign_list("demo")["campaigns"] == [{"campaign": "one"}]
    assert app.campaign_status("demo", "one") == {"campaign": "one"}
    monkeypatch.setattr(module, "campaign_snapshot", lambda *args: raises(KeyError("missing")))
    with pytest.raises(ApplicationError) as caught:
        app.campaign_status("demo", "missing")
    assert caught.value.code == "UNKNOWN_CAMPAIGN"


def test_campaign_and_object_proposal_validation_errors(monkeypatch):
    app = application(index=object())
    app._require_operation_available = lambda *args: None
    target = scope(AgentScopeType.CAMPAIGN, "study")
    configured = SimpleNamespace()
    app.resolve_scope = lambda *args: (target, configured, object())

    monkeypatch.setattr(module, "campaign_snapshot", lambda *args: {"lifecycle_state": "ACTIVE"})
    with pytest.raises(ApplicationError, match="not completable"):
        app.propose_campaign_completion("demo", "study", outcome="SUPPORTED", assessment="x")
    monkeypatch.setattr(module, "campaign_snapshot", lambda *args: {
        "lifecycle_state": "COMPLETABLE", "revision_id": "r1",
        "completion": {"evidence_digest": "d", "membership_run_ids": []},
    })
    with pytest.raises(ApplicationError, match="outcome is required"):
        app.propose_campaign_completion("demo", "study", outcome=" ", assessment="x")
    monkeypatch.setattr(module, "campaign_snapshot", lambda *args: {
        "lifecycle_state": "ARCHIVED",
    })
    with pytest.raises(ApplicationError, match="already archived"):
        app.propose_campaign_archive("demo", "study", reason="x")
    monkeypatch.setattr(module, "campaign_snapshot", lambda *args: {
        "lifecycle_state": "ACTIVE", "revision_id": "r1",
    })
    with pytest.raises(ApplicationError, match="reason is required"):
        app.propose_campaign_archive("demo", "study", reason=" ")

    app.resolve_scope = lambda *args: (target, configured, object())
    with pytest.raises(ApplicationError, match="only Run or Attempt"):
        app.propose_object_archive("demo", AgentScopeType.CAMPAIGN, "study", reason="x")
    run_target = scope()
    app.resolve_scope = lambda *args: (run_target, configured, object())
    with pytest.raises(ApplicationError, match="reason is required"):
        app.propose_object_archive("demo", AgentScopeType.RUN, "run-a", reason=" ")


def test_mapping_readers_manifest_lookup_and_validation_helpers(tmp_path):
    missing = tmp_path / "missing.json"
    assert ExperimentServerApplication._read_mapping(missing) == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{")
    assert ExperimentServerApplication._read_mapping(bad_json) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]")
    assert ExperimentServerApplication._read_mapping(list_json) == {}
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("[unterminated")
    assert ExperimentServerApplication._read_yaml_mapping(bad_yaml) == {}
    scalar_yaml = tmp_path / "scalar.yaml"
    scalar_yaml.write_text("text")
    assert ExperimentServerApplication._read_yaml_mapping(scalar_yaml) == {}
    assert ExperimentServerApplication._manifest_at(tmp_path, ("missing.json",)) == ({}, None)
    valid = tmp_path / "manifest.yaml"
    valid.write_text("project: demo\n")
    assert ExperimentServerApplication._manifest_at(tmp_path, ("missing.json", "manifest.yaml"))[0] == {
        "project": "demo",
    }

    missing_gate = ExperimentServerApplication._identity_gate(
        gate_id="run.identity", label="Run", payload={}, expected={"run_id": "r"},
    )
    mismatch = ExperimentServerApplication._identity_gate(
        gate_id="run.identity", label="Run", payload={"run_id": "x"},
        expected={"run_id": "r"},
    )
    incomplete = ExperimentServerApplication._identity_gate(
        gate_id="run.identity", label="Run", payload={"project": "demo"},
        expected={"project": "demo", "run_id": "r"},
    )
    assert (missing_gate["status"], mismatch["status"], incomplete["status"]) == (
        "UNKNOWN", "BLOCKED", "UNKNOWN",
    )
    payload = ExperimentServerApplication._validation_payload(
        object_type="run", identity="r", gates=[mismatch, incomplete],
    )
    assert payload["result"] == "BLOCKED"
    assert payload["execution_evidence_result"] == "UNKNOWN"


def test_attempt_local_read_models_cover_files_fallbacks_and_event_dedup(tmp_path, monkeypatch):
    attempt_dir = tmp_path / "attempts/a1"
    collected = attempt_dir / "collected_run"
    collected.mkdir(parents=True)
    (attempt_dir / "stdout.log").write_text("one\ntwo\n")
    (attempt_dir / "collection.json").write_text(json.dumps({
        "attempt_id": "a1",
        "latest_completed_checkpoint": "checkpoint-2",
        "latest_completed_checkpoint_step": 2,
        "artifacts": [],
        "process_evidence": {"stderr_tail": ["archived"], "sources": {"stderr": "remote"}},
    }))
    (collected / "checkpoint-2").mkdir()
    (collected / "samples").mkdir()
    event = {"attempt_id": "a1", "timestamp": "2026-01-01T00:00:00Z", "event": "done"}
    (collected / "events.jsonl").write_text(json.dumps(event) + "\n")
    (attempt_dir / "events.jsonl").write_text(json.dumps(event) + "\n")
    (tmp_path / "events.jsonl").write_text(
        json.dumps({**event, "attempt_id": "other"}) + "\n" +
        json.dumps({"attempt_id": "a1", "timestamp": "2025-01-01T00:00:00Z",
                    "event": "started"}) + "\n"
    )
    row = SimpleNamespace(
        run_id="run-a", run_dir=str(tmp_path), checkpoint={"latest_completed_checkpoint": "fallback"},
        artifacts={"fallback": True},
    )
    attempt = SimpleNamespace(attempt_id="a1")
    target = scope(AgentScopeType.ATTEMPT, "run-a::a1")
    app = application()
    app._attempt_context = lambda *args: (target, object(), row, attempt, attempt_dir)
    logs = app.attempt_logs("demo", "run-a::a1", lines=1)
    assert logs["streams"]["stdout"]["mode"] == "local_file"
    assert logs["streams"]["stderr"]["lines"] == ["archived"]
    assert logs["follow_supported"] is True
    with pytest.raises(ApplicationError, match="between 1 and 10000"):
        app.attempt_logs("demo", "run-a::a1", lines=0)
    checkpoints = app.attempt_checkpoints("demo", "run-a::a1")
    assert checkpoints["local_entries"] == ["collected_run/checkpoint-2"]
    artifacts = app.attempt_artifacts("demo", "run-a::a1")
    assert artifacts["summary"] == row.artifacts
    assert artifacts["local_roots"] == ["collected_run/checkpoint-2", "collected_run/samples"]
    events = app.attempt_events("demo", "run-a::a1")
    assert [item["event"] for item in events["events"]] == ["started", "done"]
    assert len(events["sources"]) == 2


def test_metric_payload_one_point_missing_key_and_invalid_limit():
    records = [{"step": 1, "loss": 3.0}, {"step": 2, "loss": 2.0}]
    payload = ExperimentServerApplication._metric_payload(
        records, keys="loss, absent", max_points=1, source=None, source_attempt_id=None,
    )
    assert payload["points"] == [{"step": 2, "timestamp": None, "loss": 2.0}]
    assert payload["missing_keys"] == ["absent"]
    with pytest.raises(ApplicationError, match="max_points must be positive"):
        ExperimentServerApplication._metric_payload(
            records, keys=None, max_points=0, source=None, source_attempt_id=None,
        )


def test_action_adapter_error_mapping_and_completion_drift(monkeypatch):
    target = scope(AgentScopeType.CAMPAIGN, "study")
    configured = object()
    agent_store = SimpleNamespace(proposal=lambda *args: raises(FileNotFoundError()))
    action_service = SimpleNamespace(
        prepare=lambda *args: {}, authorize=lambda *args: raises(FileNotFoundError()),
        execute=lambda *args: raises(RuntimeError("blocked")),
    )
    runtime = SimpleNamespace(
        agent_store=agent_store, action_service=action_service,
        action_store=SimpleNamespace(snapshot=lambda action_id: {"proposal_kind": "OTHER"}),
        index=object(), projects=[],
    )
    app = ExperimentServerApplication(runtime)
    app.resolve_scope = lambda *args: (target, configured, object())
    with pytest.raises(ApplicationError) as caught:
        app.prepare_action("demo", AgentScopeType.CAMPAIGN, "study", "missing")
    assert caught.value.code == "UNKNOWN_PROPOSAL"
    with pytest.raises(ApplicationError) as caught:
        app.authorize_action("missing")
    assert caught.value.code == "UNKNOWN_ACTION"
    with pytest.raises(ApplicationError) as caught:
        app.execute_action("a", "confirm")
    assert caught.value.code == "ACTION_BLOCKED"

    runtime.action_store.snapshot = lambda action_id: {
        "proposal_kind": "COMPLETE_CAMPAIGN",
        "scope": {"project": "demo", "object_id": "study"},
        "proposed_content": yaml.safe_dump({"evidence_digest": "old"}),
    }
    runtime.project = lambda name: configured
    monkeypatch.setattr(module, "index_project", lambda *args: None)
    monkeypatch.setattr(module, "campaign_snapshot", lambda *args: {
        "lifecycle_state": "COMPLETABLE", "completion": {"evidence_digest": "new"},
    })
    with pytest.raises(ApplicationError) as caught:
        app.execute_action("a", "confirm")
    assert caught.value.code == "CAMPAIGN_COMPLETION_DRIFT"


def test_proposal_lookup_and_decision_store_errors_are_mapped():
    target = scope()
    configured = object()
    resolved = object()
    store = SimpleNamespace(
        ensure=lambda *args, **kwargs: None,
        proposal=lambda *args: raises(FileNotFoundError()),
    )
    app = application(agent_store=store)
    app.resolve_scope = lambda *args: (target, configured, resolved)
    with pytest.raises(ApplicationError) as caught:
        app.decide_proposal("demo", AgentScopeType.RUN, "run-a", "missing", "REJECTED")
    assert caught.value.code == "UNKNOWN_PROPOSAL"
    with pytest.raises(ApplicationError) as caught:
        app.proposal_show("demo", AgentScopeType.RUN, "run-a", "missing")
    assert caught.value.code == "UNKNOWN_PROPOSAL"

    store.proposal = lambda *args: {"evidence_digest": "same"}
    store.decide_proposal = lambda *args, **kwargs: raises(ValueError("invalid transition"))
    app.bounded_evidence = lambda *args: {}
    with pytest.raises(ApplicationError) as caught:
        app.decide_proposal("demo", AgentScopeType.RUN, "run-a", "p", "REJECTED")
    assert caught.value.code == "INVALID_PROPOSAL"


def test_action_additional_error_and_verified_success_paths(monkeypatch):
    indexed = []
    runtime = SimpleNamespace(
        action_service=SimpleNamespace(
            authorize=lambda *args: raises(RuntimeError("not approved")),
            execute=lambda *args: {"execution": {"status": "VERIFIED"}},
        ),
        action_store=SimpleNamespace(snapshot=lambda action_id: {
            "proposal_kind": "OTHER", "scope": {"project": "one"},
        }),
        index=object(), projects=[SimpleNamespace(project="one"), SimpleNamespace(project="two")],
    )
    runtime.project = lambda name: next(
        project for project in runtime.projects if project.project == name
    )
    app = ExperimentServerApplication(runtime)
    with pytest.raises(ApplicationError) as caught:
        app.authorize_action("a")
    assert caught.value.code == "ACTION_BLOCKED"
    monkeypatch.setattr(module, "index_project", lambda index, project: indexed.append(project.project))
    result = app.execute_action("a", "confirm")
    assert result["execution"]["status"] == "VERIFIED"
    assert indexed == ["one"]
    runtime.action_store.snapshot = lambda action_id: raises(FileNotFoundError())
    with pytest.raises(ApplicationError) as caught:
        app.execute_action("missing", "confirm")
    assert caught.value.code == "UNKNOWN_ACTION"
