"""Deep branch coverage for application read models and adapter error mapping."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import ml_exp_server.application as module
import ml_exp_server.ingest.runscan as runscan
from ml_exp_server.application import (
    ApplicationError,
    ExperimentServerApplication,
    attempt_failure_evidence_assessment,
    compact_evidence,
    structured_failure_summary,
)
from ml_exp_server.schemas import OperationScope, OperationScopeType, CampaignRelationship


class Dump:
    def __init__(self, **values):
        self.__dict__.update(values)

    def model_dump(self, **kwargs):
        return dict(self.__dict__)


def scope(kind=OperationScopeType.RUN, object_id="run-a"):
    return OperationScope(project="demo", scope_type=kind, object_id=object_id)


def application(**runtime_values):
    runtime = SimpleNamespace(**runtime_values)
    return ExperimentServerApplication(runtime)


def raises(error):
    raise error


def application_error_code(call):
    with pytest.raises(ApplicationError) as caught:
        call()
    return caught.value.code


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
    ("relationship", "manifest", "current", "indexed", "expected"),
    [
        (
            CampaignRelationship.MATCHED,
            {
                "project": "demo", "run_id": "run-a", "campaign": "study",
                "source_id": "source-1", "image_id": "image-1",
                "config_path": "config.yml", "seed": 7,
            },
            "attempt-001", ["attempt-001"], "PASS",
        ),
        (
            CampaignRelationship.PROJECT_MISMATCH,
            {"project": "other", "run_id": "run-a", "campaign": "study"},
            None, [], "BLOCKED",
        ),
        (
            CampaignRelationship.UNRESOLVED,
            {
                "project": "demo", "run_id": "run-a", "campaign": "study",
                "git_commit": "commit-1", "image_id": "image-1",
                "config_path": "config.yml", "resolved_config": {"seed": 9},
            },
            "attempt-002", ["attempt-001"], "UNKNOWN",
        ),
    ],
)
def test_run_validate_covers_binding_provenance_and_current_attempt(
    monkeypatch, tmp_path, relationship, manifest, current, indexed, expected,
):
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    attempts = [SimpleNamespace(attempt_id=item) for item in indexed]
    binding = Dump(relationship=relationship)
    row = SimpleNamespace(
        run_id="run-a", run_dir=str(tmp_path), campaign="study",
        campaign_binding=binding, attempts=attempts,
    )
    app = application()
    app.resolve_scope = lambda *_args: (scope(), SimpleNamespace(), row)
    app._attempt_validation_gates = lambda *_args, **_kwargs: [
        app._gate("attempt.extra", "PASS", "checked")
    ]
    monkeypatch.setattr(module, "preferred_attempt_id", lambda _path: current)

    payload = app.run_validate("demo", "run-a")
    gates = {item["id"]: item for item in payload["gates"]}

    assert gates["run.campaign_binding"]["status"] == expected
    if current in indexed:
        assert gates["run.current_attempt"]["status"] == "PASS"
        assert gates["attempt.extra"]["status"] == "PASS"
    else:
        assert gates["run.current_attempt"]["status"] == "UNKNOWN"


def retry_context(
    *, state="FAILED", decision=None, attempts=("attempt-001",), job=None,
    run_dir=None,
):
    attempt = SimpleNamespace(
        attempt_id=attempts[0], state=state, decision=decision,
        backend_job_id=job,
    )
    row = SimpleNamespace(
        run_id="run-a", campaign="study",
        attempts=[SimpleNamespace(attempt_id=item) for item in attempts],
        run_dir=run_dir,
    )
    configured = SimpleNamespace(project="demo")
    return (
        scope(OperationScopeType.ATTEMPT, f"run-a::{attempt.attempt_id}"),
        configured, row, attempt, Path("/attempt"),
    )


def retry_application(context):
    app = application()
    app._attempt_context = lambda *_args: context
    app._require_operation_available = lambda *_args: None
    app._campaign_file = lambda *_args: Path("/campaign.yml")
    app._prepare_attempt_action = lambda *_args, **kwargs: {
        "action": {"kind": kwargs["kind"], "draft": kwargs["draft"]},
    }
    return app


def test_prepare_attempt_retry_rejects_invalid_budget_before_lookup():
    app = application()
    assert application_error_code(lambda: app.prepare_attempt_retry(
        "demo", "run-a::attempt-001", new_attempt_id=None,
        max_gpu_hours=0, reason="test",
    )) == "INVALID_GPU_BUDGET"


def test_prepare_attempt_retry_rejects_invalid_resource_approval_before_lookup():
    app = application()
    assert application_error_code(lambda: app.prepare_attempt_retry(
        "demo", "run-a::attempt-001", new_attempt_id=None,
        max_gpu_hours=None, reason="test", resource_approval="unexpected",
    )) == "INVALID_GPU_BUDGET"


@pytest.mark.parametrize(("context", "expected"), [
    (retry_context(state="RUNNING"), "ATTEMPT_NOT_RETRYABLE"),
    (retry_context(decision={"action": "DO_NOT_RETRY"}), "ATTEMPT_RETRY_FORBIDDEN"),
    (retry_context(decision={"retries_allowed": 1, "retries_used": 1}),
     "ATTEMPT_RETRY_BUDGET_EXHAUSTED"),
])
def test_prepare_attempt_retry_fail_closed_decisions(context, expected):
    app = retry_application(context)
    assert application_error_code(lambda: app.prepare_attempt_retry(
        "demo", "run-a::attempt-001", new_attempt_id=None,
        max_gpu_hours=1, reason="test",
    )) == expected


def test_prepare_attempt_retry_generates_valid_identity_and_prepares_action():
    context = retry_context(
        decision="legacy", attempts=("attempt-001", "custom", "attempt-009"),
    )
    app = retry_application(context)

    result = app.prepare_attempt_retry(
        "demo", "run-a::attempt-001", new_attempt_id=None,
        max_gpu_hours=1.5, reason="retry", wandb_cloud_sync=True,
    )

    draft = yaml.safe_load(result["action"]["draft"])
    assert result["action"]["kind"] == "RETRY_ATTEMPT"
    assert draft["attempt_id"] == "attempt-010"
    assert draft["wandb_cloud_sync"] is True


def test_prepare_attempt_retry_review_exact_preserves_legacy_root_without_budget():
    app = retry_application(retry_context(run_dir="/legacy/runs/run-a"))

    result = app.prepare_attempt_retry(
        "demo", "run-a::attempt-001", new_attempt_id="attempt-002",
        max_gpu_hours=None, reason="retry", resource_approval="review_exact",
    )

    draft = yaml.safe_load(result["action"]["draft"])
    assert draft["local_root"] == "/legacy"
    assert "max_gpu_hours" not in draft


@pytest.mark.parametrize(("new_attempt_id", "expected"), [
    ("bad", "INVALID_ATTEMPT_ID"),
    ("attempt-001", "DUPLICATE_ATTEMPT_ID"),
])
def test_prepare_attempt_retry_rejects_invalid_or_duplicate_identity(
    new_attempt_id, expected,
):
    app = retry_application(retry_context())
    assert application_error_code(lambda: app.prepare_attempt_retry(
        "demo", "run-a::attempt-001", new_attempt_id=new_attempt_id,
        max_gpu_hours=1, reason="test",
    )) == expected


def test_prepare_attempt_cancel_rejects_state_and_missing_backend_job():
    stopped = retry_application(retry_context(state="FAILED", job="job-1"))
    assert application_error_code(lambda: stopped.prepare_attempt_cancel(
        "demo", "run-a::attempt-001", reason="test",
    )) == "ATTEMPT_NOT_CANCELLABLE"

    missing = retry_application(retry_context(state="RUNNING", job=None))
    assert application_error_code(lambda: missing.prepare_attempt_cancel(
        "demo", "run-a::attempt-001", reason="test",
    )) == "BACKEND_JOB_ID_MISSING"


def test_prepare_attempt_cancel_binds_exact_backend_job():
    app = retry_application(retry_context(state="RUNNING", job="job-7"))
    result = app.prepare_attempt_cancel(
        "demo", "run-a::attempt-001", reason="stop",
    )
    draft = yaml.safe_load(result["action"]["draft"])
    assert result["action"]["kind"] == "CANCEL_RUN"
    assert draft["backend_job_id"] == "job-7"


def test_prepare_attempt_action_binds_bounded_evidence():
    app = application()
    app.bounded_evidence = lambda *_args: {"evidence": "exact"}
    app._prepare_action_intent = lambda _scope, _project, intent: intent
    context = retry_context()
    operation_scope, configured, row, attempt, _ = context

    result = app._prepare_attempt_action(
        operation_scope, configured, row, attempt,
        kind="RETRY_ATTEMPT", draft="draft", title="Retry",
        change_summary="retry", resource_estimate="1 GPU-hour", risk="scheduler",
    )

    assert result["action"]["kind"] == "RETRY_ATTEMPT"
    assert result["action"]["evidence_digest"].startswith("sha256:")


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
    ],
)
def test_structured_failure_fallback_signatures(collection, decision, signature, failure_class):
    collection = {"attempt_id": "a1", "process_state": "FAILED", **collection}
    result = structured_failure_summary(
        collection, decision, attempt_id="a1", attempt_state="FAILED",
        attempt_started_at=100.0, observed_at=101.0,
    )
    if signature is None:
        assert result is None
    else:
        assert result["failure_signature"] == signature
        assert result["failure_class"] == failure_class


def test_structured_oom_overrides_transport_and_reads_combined_stream():
    result = structured_failure_summary(
        {
            "attempt_id": "a1",
            "process_state": "FAILED",
            "failure_class": "transport",
            "process_evidence": {
                "stdout_tail": [
                    "Performing initial training step",
                    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate "
                    "536.00 MiB. GPU 0 has a total capacity of 79.18 GiB of which "
                    "128.00 MiB is free. Of the allocated memory 76.37 GiB is allocated",
                ],
                "stderr_tail": [],
            },
        },
        {"failure_class": "transport"},
        attempt_id="a1", attempt_state="FAILED",
        attempt_started_at=100.0, observed_at=101.0,
    )
    assert result["failure_signature"] == "CUDA_OOM"
    assert result["failure_class"] == "resource"
    assert result["requested_bytes"] == 536 * 1024 ** 2


def test_running_attempt_keeps_unclassified_decision_as_non_applicable_diagnostic():
    assessment = attempt_failure_evidence_assessment(
        {
            "attempt_id": "a1", "scheduler_state": "RUNNING",
            "worker_state": "ALLOCATED", "process_state": "RUNNING",
            "model_state": "OBSERVED",
        },
        {"action": "OBSERVE", "failure_class": "unknown"},
        attempt_id="a1", attempt_state="RUNNING",
        attempt_started_at=100.0, observed_at=110.0,
        decision_observed_at=111.0,
    )
    assert assessment["failure_summary"] is None
    assert assessment["diagnostic_evidence"] == [{
        "kind": "preliminary_failure_classification",
        "failure_class": "unknown",
        "source": "decision.failure_class",
        "attempt_id": "a1",
        "attempt_state": "RUNNING",
        "observed_at": 111.0,
        "applicability": "NON_APPLICABLE",
        "source_binding": "EXACT_ATTEMPT",
        "evidence_source": None,
        "reason": (
            "decision metadata is contextual only and is not exact, fresh, "
            "terminal failure evidence"
        ),
    }]


def test_failure_evidence_requires_exact_fresh_terminal_attempt():
    collection = {
        "attempt_id": "a1", "process_state": "FAILED",
        "process_evidence": {"stderr_tail": ["ModuleNotFoundError: pkg"]},
    }
    exact = attempt_failure_evidence_assessment(
        collection, attempt_id="a1", attempt_state="FAILED",
        attempt_started_at=100.0, observed_at=101.0,
    )
    assert exact["failure_summary"]["failure_signature"] == "MISSING_PYTHON_MODULE"
    assert exact["failure_summary"]["attempt_id"] == "a1"
    assert exact["failure_summary"]["applicability"] == "APPLICABLE"

    stale = attempt_failure_evidence_assessment(
        collection, attempt_id="a1", attempt_state="FAILED",
        attempt_started_at=100.0, observed_at=99.0,
    )
    assert stale["failure_summary"] is None
    assert stale["diagnostic_evidence"][0]["applicability"] == "STALE"

    mismatch = attempt_failure_evidence_assessment(
        {**collection, "attempt_id": "old"},
        attempt_id="a1", attempt_state="FAILED",
        attempt_started_at=100.0, observed_at=101.0,
    )
    assert mismatch["failure_summary"] is None
    assert mismatch["diagnostic_evidence"][0]["applicability"] == "ATTEMPT_MISMATCH"

    running = attempt_failure_evidence_assessment(
        collection, attempt_id="a1", attempt_state="RUNNING",
        attempt_started_at=100.0, observed_at=101.0,
    )
    assert running["failure_summary"] is None
    assert running["diagnostic_evidence"][0]["applicability"] == "NON_APPLICABLE"


def test_failure_evidence_requires_actual_start_not_manifest_creation_time():
    assessment = attempt_failure_evidence_assessment(
        {"attempt_id": "a1", "process_state": "FAILED"},
        attempt_id="a1", attempt_state="FAILED",
        attempt_started_at=None, observed_at=101.0,
    )
    assert assessment["failure_summary"] is None
    assert assessment["diagnostic_evidence"][0]["applicability"] \
        == "UNKNOWN_APPLICABILITY"
    assert "actual Attempt start" in assessment["diagnostic_evidence"][0]["reason"]


def test_failure_domains_use_independent_identity_and_observation_times():
    collection = {
        "attempt_id": "a1", "process_state": "FAILED",
        "scheduler_state": "TIMEOUT",
        "process_evidence": {"stderr_tail": ["ModuleNotFoundError: pkg"]},
    }
    scheduler_fresh = attempt_failure_evidence_assessment(
        collection, attempt_id="a1", attempt_state="FAILED",
        attempt_started_at=100.0,
        domain_observed_at={"process": 99.0, "scheduler": 101.0},
        domain_attempt_ids={"process": "a1", "scheduler": "a1"},
    )
    assert scheduler_fresh["failure_summary"]["failure_domain"] == "scheduler"
    assert scheduler_fresh["failure_summary"]["failure_signature"] \
        == "SCHEDULER_TERMINAL_FAILURE"
    assert scheduler_fresh["diagnostic_evidence"][0]["failure_domain"] == "process"
    assert scheduler_fresh["diagnostic_evidence"][0]["applicability"] == "STALE"

    process_fresh = attempt_failure_evidence_assessment(
        collection, attempt_id="a1", attempt_state="FAILED",
        attempt_started_at=100.0,
        domain_observed_at={"process": 101.0, "scheduler": 99.0},
        domain_attempt_ids={"process": "a1", "scheduler": "a1"},
    )
    assert process_fresh["failure_summary"]["failure_domain"] == "process"
    assert process_fresh["failure_summary"]["failure_signature"] \
        == "MISSING_PYTHON_MODULE"
    assert process_fresh["diagnostic_evidence"][0]["failure_domain"] == "scheduler"
    assert process_fresh["diagnostic_evidence"][0]["applicability"] == "STALE"

    wrong_scheduler = attempt_failure_evidence_assessment(
        {"attempt_id": "a1", "scheduler_state": "TIMEOUT"},
        attempt_id="a1", attempt_state="FAILED", attempt_started_at=100.0,
        domain_observed_at={"scheduler": 101.0},
        domain_attempt_ids={"scheduler": "old"},
    )
    assert wrong_scheduler["failure_summary"] is None
    assert wrong_scheduler["diagnostic_evidence"][0]["applicability"] \
        == "ATTEMPT_MISMATCH"


@pytest.mark.parametrize(
    ("states", "domain", "source"),
    [
        ({"scheduler_state": "FAILED", "process_state": "RUNNING"},
         "scheduler", "collection.scheduler_state"),
        ({"worker_state": "LOST", "process_state": "RUNNING"},
         "worker", "collection.worker_state"),
    ],
)
def test_terminal_infrastructure_causes_are_not_process_failures(states, domain, source):
    assessment = attempt_failure_evidence_assessment(
        {"attempt_id": "a1", **states},
        attempt_id="a1", attempt_state="FAILED",
        attempt_started_at=100.0, observed_at=101.0,
    )
    failure = assessment["failure_summary"]
    assert failure["failure_domain"] == domain
    assert failure["source"] == source
    assert failure["failure_signature"] != "UNCLASSIFIED_PROCESS_FAILURE"


@pytest.mark.parametrize(
    ("kind", "object_id", "code"),
    [
        (OperationScopeType.PROJECT, "other", "UNKNOWN_PROJECT"),
        (OperationScopeType.RESEARCH_QUESTION, "missing", "UNKNOWN_RESEARCH_QUESTION"),
        (OperationScopeType.CAMPAIGN, "missing", "UNKNOWN_CAMPAIGN"),
        (OperationScopeType.RUN, "missing", "UNKNOWN_RUN"),
        (OperationScopeType.ATTEMPT, "invalid", "INVALID_ATTEMPT_ID"),
        (OperationScopeType.ATTEMPT, "missing::a1", "UNKNOWN_RUN"),
        (OperationScopeType.ATTEMPT, "run-a::missing", "UNKNOWN_ATTEMPT"),
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
        app.resolve_scope("demo", OperationScopeType.PROJECT, "demo")
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
        run_dir="/missing",
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
    assert app.bounded_evidence(scope(OperationScopeType.PROJECT, "demo"), configured, configured)["runs"]
    assert app.bounded_evidence(scope(OperationScopeType.RESEARCH_QUESTION, "q1"), configured, question)["runs"]
    assert app.bounded_evidence(scope(OperationScopeType.CAMPAIGN, "study"), configured, campaign)["runs"]
    assert app.bounded_evidence(scope(), configured, row)["run"]["run_id"] == "run-a"
    attempt = Dump(attempt_id="a1", state=None)
    assert app.bounded_evidence(
        scope(OperationScopeType.ATTEMPT, "run-a::a1"), configured, attempt,
    )["attempt"]["attempt_id"] == "a1"


def test_object_show_plain_object():
    target = scope()
    app = ExperimentServerApplication(SimpleNamespace())
    app.bounded_evidence = lambda *args: {"bounded": True}
    app.resolve_scope = lambda *args: (
        target, SimpleNamespace(base_dir=None, authored_file=None), {"raw": "value"},
    )
    shown = app.object_show("demo", OperationScopeType.RUN, "run-a")
    assert shown["object"] == {"raw": "value"}
    assert shown["evidence_digest"].startswith("sha256:")


def test_object_show_run_does_not_duplicate_raw_failure_class():
    target = scope()
    row = SimpleNamespace(
        run_id="run-a",
        decision={"action": "OBSERVE", "failure_class": "unknown"},
    )
    app = ExperimentServerApplication(SimpleNamespace())
    app.resolve_scope = lambda *args: (
        target, SimpleNamespace(base_dir=None, authored_file=None), row,
    )
    app.row_evidence = lambda value: {
        "run_id": value.run_id,
        "decision": app._operational_decision(value.decision),
    }
    app.bounded_evidence = lambda *args: {
        "failure_assessment": {
            "failure_summary": None,
            "diagnostic_evidence": [{"failure_class": "unknown"}],
        },
    }

    shown = app.object_show("demo", OperationScopeType.RUN, "run-a")

    assert shown["object"]["decision"] == {"action": "OBSERVE"}
    assert shown["evidence"]["failure_assessment"]["diagnostic_evidence"]


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


def test_campaign_and_object_action_validation_errors(monkeypatch):
    app = application(index=object())
    app._require_operation_available = lambda *args: None
    target = scope(OperationScopeType.CAMPAIGN, "study")
    configured = SimpleNamespace()
    app.resolve_scope = lambda *args: (target, configured, object())

    monkeypatch.setattr(module, "campaign_snapshot", lambda *args: {
        "lifecycle_state": "ARCHIVED",
    })
    with pytest.raises(ApplicationError, match="already archived"):
        app.prepare_campaign_archive("demo", "study", reason="x")
    monkeypatch.setattr(module, "campaign_snapshot", lambda *args: {
        "lifecycle_state": "ACTIVE", "revision_id": "r1",
    })
    with pytest.raises(ApplicationError, match="reason is required"):
        app.prepare_campaign_archive("demo", "study", reason=" ")

    app.resolve_scope = lambda *args: (target, configured, object())
    with pytest.raises(ApplicationError, match="only Run or Attempt"):
        app.prepare_object_archive("demo", OperationScopeType.CAMPAIGN, "study", reason="x")
    run_target = scope()
    app.resolve_scope = lambda *args: (run_target, configured, object())
    with pytest.raises(ApplicationError, match="reason is required"):
        app.prepare_object_archive("demo", OperationScopeType.RUN, "run-a", reason=" ")


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
    target = scope(OperationScopeType.ATTEMPT, "run-a::a1")
    app = application(index=SimpleNamespace())
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


def test_running_attempt_show_and_checkpoint_validation_are_orthogonal(tmp_path):
    attempt_dir = tmp_path / "attempts/a1"
    attempt_dir.mkdir(parents=True)
    (tmp_path / "manifest.yaml").write_text(
        "project: demo\nrun_id: run-a\ncampaign: c1\n"
    )
    (attempt_dir / "attempt.yaml").write_text(
        "project: demo\nrun_id: run-a\nattempt_id: a1\n"
        "created_at: '2026-01-01T00:00:00Z'\n"
    )
    (attempt_dir / "status.json").write_text(json.dumps({
        "attempt_id": "a1", "state": "RUNNING",
    }))
    (attempt_dir / "collection.json").write_text(json.dumps({
        "attempt_id": "a1", "scheduler_state": "RUNNING",
        "worker_state": "ALLOCATED", "process_state": "RUNNING",
        "model_state": "OBSERVED", "failure_class": None,
    }))
    (attempt_dir / "decision.json").write_text(json.dumps({
        "attempt_id": "a1", "action": "OBSERVE", "failure_class": "unknown",
        "reason": "run is nonterminal",
    }))
    (attempt_dir / "train_metrics.jsonl").write_text(
        json.dumps({"step": 200, "loss": 3.0}) + "\n"
        + json.dumps({"step": 400, "loss": 2.5}) + "\n"
    )
    row = SimpleNamespace(
        run_id="run-a", run_dir=str(tmp_path), checkpoint={}, artifacts={},
    )
    attempt = Dump(
        attempt_id="a1", state="RUNNING", backend_job_id=None,
        decision={"action": "OBSERVE", "failure_class": "unknown"},
    )
    target = scope(OperationScopeType.ATTEMPT, "run-a::a1")
    app = application()
    app._attempt_context = lambda *args: (target, object(), row, attempt, attempt_dir)

    shown = app.attempt_show("demo", "run-a::a1")
    assessment = shown["failure_assessment"]
    assert assessment["failure_summary"] is None
    assert assessment["diagnostic_evidence"][0]["applicability"] == "NON_APPLICABLE"
    assert "failure_class" not in shown["attempt"]["decision"]
    assert "failure_class" not in shown["collection"]
    assert shown["collection"]["process_state"] == "RUNNING"

    validation = app.attempt_validate("demo", "run-a::a1")
    checkpoint_gate = next(
        gate for gate in validation["gates"]
        if gate["id"] == "attempt.checkpoint_evidence"
    )
    assert checkpoint_gate["status"] == "UNKNOWN"
    assert "missing" in checkpoint_gate["message"]


def test_attempt_agent_evidence_marks_diagnostics_non_applicable(tmp_path):
    attempt_dir = tmp_path / "attempts/a1"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "attempt.yaml").write_text(
        "attempt_id: a1\ncreated_at: '2026-01-01T00:00:00Z'\n"
    )
    (attempt_dir / "collection.json").write_text(json.dumps({
        "attempt_id": "a1", "scheduler_state": "RUNNING",
        "process_state": "RUNNING", "model_state": "OBSERVED",
    }))
    (attempt_dir / "decision.json").write_text(json.dumps({
        "failure_class": "unknown", "action": "OBSERVE",
    }))
    attempt = Dump(
        attempt_id="a1", state="RUNNING",
        decision={"failure_class": "unknown", "action": "OBSERVE"},
    )
    row = SimpleNamespace(run_id="run-a", run_dir=str(tmp_path))
    app = application(index=SimpleNamespace(get_run=lambda *args: row))
    app.row_evidence = lambda value: {"run_id": value.run_id, "scheduler_state": "RUNNING"}
    app.campaign_contexts = lambda *args: []

    evidence = app.bounded_evidence(
        scope(OperationScopeType.ATTEMPT, "run-a::a1"),
        SimpleNamespace(project="demo"), attempt,
    )
    assessment = evidence["failure_assessment"]
    assert assessment["failure_summary"] is None
    assert assessment["diagnostic_evidence"][0]["applicability"] == "NON_APPLICABLE"
    assert "MUST NOT" in assessment["agent_instruction"]
    assert "retry" in assessment["agent_instruction"]
    assert evidence["attempt"]["decision"] == {"action": "OBSERVE"}


def test_terminal_local_fallback_uses_domain_records_for_applicability(tmp_path):
    attempt_dir = tmp_path / "attempts/a1"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "attempt.json").write_text(json.dumps({
        "attempt_id": "a1", "state": "FAILED", "failure_class": "resource",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:01:00Z",
    }))
    (tmp_path / "status.json").write_text(json.dumps({
        "attempt_id": "a1", "state": "FAILED",
        "observed_at": "2026-01-01T00:01:01Z",
    }))
    row = SimpleNamespace(run_id="run-a", run_dir=str(tmp_path))
    attempt = Dump(
        attempt_id="a1", state="FAILED", backend_job_id=None, decision={},
    )
    target = scope(OperationScopeType.ATTEMPT, "run-a::a1")
    app = application()
    app._attempt_context = lambda *args: (target, object(), row, attempt, attempt_dir)

    shown = app.attempt_show("demo", "run-a::a1")

    failure = shown["failure_assessment"]["failure_summary"]
    assert failure["failure_domain"] == "process"
    assert failure["failure_class"] == "resource"
    assert failure["applicability"] == "APPLICABLE"
    assert failure["evidence_source"].endswith("attempts/a1/attempt.json")


def test_run_agent_evidence_sanitizes_raw_failure_class_and_uses_assessment(tmp_path):
    attempt_dir = tmp_path / "attempts/a1"
    attempt_dir.mkdir(parents=True)
    (tmp_path / "collection.json").write_text(json.dumps({
        "attempt_id": "a1", "scheduler_state": "RUNNING",
        "process_state": "RUNNING", "model_state": "OBSERVED",
    }))
    (tmp_path / "decision.json").write_text(json.dumps({
        "action": "OBSERVE", "failure_class": "unknown",
        "reason": "run is nonterminal",
    }))
    attempt = Dump(
        attempt_id="a1", state="RUNNING",
        decision={"action": "OBSERVE", "failure_class": "unknown"},
    )
    row = SimpleNamespace(
        run_id="run-a", run_dir=str(tmp_path), attempts=[attempt],
        decision={"action": "OBSERVE", "failure_class": "unknown",
                  "reason": "run is nonterminal"},
    )
    app = application(index=SimpleNamespace())
    app.row_evidence = lambda value: {
        "run_id": value.run_id,
        "decision": app._operational_decision(value.decision),
    }
    app.campaign_contexts = lambda *args: []

    evidence = app.bounded_evidence(
        scope(OperationScopeType.RUN, "run-a"),
        SimpleNamespace(project="demo"), row,
    )

    assert "failure_class" not in evidence["run"]["decision"]
    assert evidence["attempts"][0]["decision"] == {"action": "OBSERVE"}
    assessment = evidence["failure_assessment"]
    assert assessment["failure_summary"] is None
    assert assessment["diagnostic_evidence"][0]["failure_class"] == "unknown"
    assert assessment["diagnostic_evidence"][0]["source_binding"] \
        == "BOUND_BY_EXACT_ROOT_COLLECTION"
    assert "MUST NOT" in assessment["agent_instruction"]

    row.campaign = "study"
    row.campaign_memberships = []
    app.runtime.index.list_runs = lambda *args, **kwargs: [row]
    question = Dump(links=SimpleNamespace(campaigns=["study"]))
    rq = app.bounded_evidence(
        scope(OperationScopeType.RESEARCH_QUESTION, "q1"),
        SimpleNamespace(project="demo"), question,
    )
    assert rq["runs"][0]["decision"] == {
        "action": "OBSERVE", "reason": "run is nonterminal",
    }
    assert rq["runs"][0]["failure_assessment"]["failure_summary"] is None


def test_attempt_eval_uses_indexed_exact_snapshot_without_rescan(tmp_path, monkeypatch):
    variants = [{"variant": "oracle", "history": []}]
    snapshot = {"schema_version": 1, "family_state": "UNRESOLVED"}
    row = SimpleNamespace(
        run_id="run-a", run_dir=str(tmp_path), eval_variants=variants,
        evaluation_snapshot=snapshot,
        evidence=SimpleNamespace(
            evaluation=SimpleNamespace(attempt_id="a1"),
        ),
    )
    attempt = SimpleNamespace(attempt_id="a1")
    target = scope(OperationScopeType.ATTEMPT, "run-a::a1")
    app = application()
    app._attempt_context = lambda *args: (target, object(), row, attempt, tmp_path)
    monkeypatch.setattr(
        runscan, "evaluation_variants",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("rescanned")),
    )

    payload = app.attempt_eval("demo", "run-a::a1")

    assert payload["variants"] is variants
    assert payload["evaluation_snapshot"] is snapshot
    assert payload["evidence_status"] == "INDEXED_EXACT_ATTEMPT"

    row.evidence.evaluation.attempt_id = "other"
    missing = app.attempt_eval("demo", "run-a::a1")
    assert missing["variants"] == []
    assert missing["evidence_status"] == "EXACT_ATTEMPT_NOT_INDEXED"


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


def test_action_adapter_error_mapping(monkeypatch):
    target = scope(OperationScopeType.CAMPAIGN, "study")
    configured = object()
    action_service = SimpleNamespace(
        prepare=lambda *args: raises(RuntimeError("blocked")),
        authorize=lambda *args: raises(FileNotFoundError()),
        execute=lambda *args: raises(RuntimeError("blocked")),
    )
    runtime = SimpleNamespace(
        action_service=action_service,
        action_store=SimpleNamespace(snapshot=lambda action_id: {"intent_kind": "OTHER"}),
        index=object(), projects=[],
    )
    app = ExperimentServerApplication(runtime)
    app.resolve_scope = lambda *args: (target, configured, object())
    app.bounded_evidence = lambda *args: {"bounded": True}
    digest = module.evidence_digest({"bounded": True})
    with pytest.raises(ApplicationError) as caught:
        app.prepare_action(
            "demo", OperationScopeType.CAMPAIGN, "study",
            {"evidence_digest": digest},
        )
    assert caught.value.code == "ACTION_BLOCKED"
    with pytest.raises(ApplicationError) as caught:
        app.prepare_action(
            "demo", OperationScopeType.CAMPAIGN, "study",
            {"evidence_digest": "sha256:stale"},
        )
    assert caught.value.code == "STALE_EVIDENCE"
    with pytest.raises(ApplicationError) as caught:
        app.authorize_action("missing")
    assert caught.value.code == "UNKNOWN_ACTION"
    with pytest.raises(ApplicationError) as caught:
        app.execute_action("a", "confirm")
    assert caught.value.code == "ACTION_BLOCKED"

def test_action_additional_error_and_verified_success_paths(monkeypatch):
    indexed = []
    runtime = SimpleNamespace(
        action_service=SimpleNamespace(
            authorize=lambda *args: raises(RuntimeError("not approved")),
            execute=lambda *args: {"execution": {"status": "VERIFIED"}},
        ),
        action_store=SimpleNamespace(snapshot=lambda action_id: {
            "intent_kind": "OTHER", "scope": {"project": "one"},
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
