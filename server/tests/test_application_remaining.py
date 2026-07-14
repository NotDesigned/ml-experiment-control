"""Remaining domain branches for the transport-neutral application layer."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ml_exp_server import application as module
from ml_exp_server.application import (
    ApplicationError, ExperimentServerApplication,
    attempt_failure_evidence_assessment, structured_failure_summary,
)
from ml_exp_server.schemas import (
    CampaignRef, ControllerConfig, OperationScope, OperationScopeType,
    ProjectLifecycleState, ResearchProject,
)


def app(**runtime_values):
    return ExperimentServerApplication(SimpleNamespace(**runtime_values))


def scope(kind=OperationScopeType.RUN, object_id="run-a"):
    return OperationScope(project="demo", scope_type=kind, object_id=object_id)


def error_code(call):
    with pytest.raises(ApplicationError) as caught:
        call()
    return caught.value.code


def test_failure_helpers_cover_nonterminal_domain_and_missing_context():
    assessment = attempt_failure_evidence_assessment(
        {
            "attempt_id": "a1", "process_state": "RUNNING",
            "process_evidence": {"stderr_tail": ["ModuleNotFoundError: missing"]},
        },
        attempt_id="a1", attempt_state="FAILED",
        attempt_started_at=1, observed_at=2,
    )
    assert assessment["failure_summary"] is None
    assert assessment["diagnostic_evidence"][0]["applicability"] == "NON_APPLICABLE"
    assert structured_failure_summary({}) is None


def test_campaign_context_preserves_orphaned_membership():
    binding = SimpleNamespace(
        campaign="removed", revision_id="revision",
        membership=SimpleNamespace(model_dump=lambda **_kwargs: {"kind": "reuse"}),
    )
    value = app(index=SimpleNamespace())
    result = value.campaign_contexts(
        SimpleNamespace(campaigns=[]),
        SimpleNamespace(campaign_memberships=[binding]),
    )
    assert result[0]["orphaned_campaign"] is True


def test_operation_availability_fails_closed_on_unavailable_evidence(monkeypatch):
    operation = module.OPERATIONS_BY_ID["run.submit"]
    value = app()
    value.resolve_scope = lambda *_args: (scope(), SimpleNamespace(), object())
    value._publication_targets_available = lambda: ()
    value._cloud_publication_available = lambda: False
    value._operation_blockers = lambda *_args: (_ for _ in ()).throw(
        ValueError("unavailable"),
    )
    monkeypatch.setattr(module, "operations_for_scope", lambda _kind: (operation,))
    result = value.operation_availability("demo", OperationScopeType.RUN, "run-a")
    assert result[0].status == "BLOCKED"
    assert "unavailable" in result[0].reasons[0]


def test_operation_blocker_matrix(monkeypatch, tmp_path):
    runtime = SimpleNamespace(
        index=object(),
        config=SimpleNamespace(
            action_runtime=SimpleNamespace(allow_observability_mutations=True),
        ),
    )
    value = ExperimentServerApplication(runtime)
    project = SimpleNamespace(
        project="demo", research_questions_dir=None, authored_file=None,
        base_dir=tmp_path, controller=None,
    )
    assert value._operation_blockers(
        "question.create", scope(OperationScopeType.PROJECT, "demo"), project, object(),
    )
    assert value._operation_blockers(
        "campaign.create", scope(OperationScopeType.PROJECT, "demo"), project, object(),
    )
    assert value._operation_blockers(
        "campaign.update", scope(OperationScopeType.CAMPAIGN, "study"),
        project, SimpleNamespace(current_revision=None),
    )
    assert value._operation_blockers(
        "run.derive", scope(), project,
        SimpleNamespace(campaign_memberships=[], campaign=None),
    )

    monkeypatch.setattr(
        module, "campaign_snapshot", lambda *_args: {"lifecycle_state": "ARCHIVED"},
    )
    assert "already archived" in value._operation_blockers(
        "object.archive", scope(OperationScopeType.CAMPAIGN, "study"), project, object(),
    )[0]
    record = tmp_path / "experiments/archive_records/runs/run-a.yml"
    record.parent.mkdir(parents=True)
    record.write_text("archived: true\n")
    assert "already exists" in value._operation_blockers(
        "object.archive", scope(), project, object(),
    )[0]

    value._publication_targets_available = lambda: ("local",)
    value._observability_attempts = lambda *_args: (_ for _ in ()).throw(ValueError())
    assert "no observed Attempts" in value._operation_blockers(
        "observability.backfill", scope(), project, object(),
    )[0]
    value._observability_attempts = lambda *_args: [
        (f"run-{index}", "a1") for index in range(501)
    ]
    assert "500-Attempt" in value._operation_blockers(
        "observability.backfill", scope(), project, object(),
    )[0]

    submitted = SimpleNamespace(has_submission=True)
    blockers = value._operation_blockers(
        "run.submit", scope(), project,
        SimpleNamespace(
            scheduler_state="RUNNING", attempts=[submitted], campaign_memberships=[],
        ),
    )
    assert any("controller" in reason.lower() for reason in blockers)
    retry = SimpleNamespace(
        state="RUNNING", decision={
            "action": "DO_NOT_RETRY", "retries_allowed": 0, "retries_used": 0,
        },
    )
    retry_blockers = value._operation_blockers(
        "attempt.retry", scope(OperationScopeType.ATTEMPT, "run-a::a1"), project, retry,
    )
    assert len(retry_blockers) == 3
    cancel_blockers = value._operation_blockers(
        "attempt.cancel", scope(OperationScopeType.ATTEMPT, "run-a::a1"), project,
        SimpleNamespace(state="FAILED", decision={}, backend_job_id=None),
    )
    assert len(cancel_blockers) == 2

    run = SimpleNamespace(run_dir=str(tmp_path), run_id="run-a")
    assert value._operation_blockers("run.evaluate", scope(), project, run)
    project.controller = SimpleNamespace(capabilities={})
    monkeypatch.setattr(module, "preferred_attempt_id", lambda _path: None)
    assert any("evaluation_as_run" in reason for reason in value._operation_blockers(
        "run.evaluate", scope(), project, run,
    ))
    project.controller.capabilities["evaluation_as_run"] = True
    monkeypatch.setattr(module, "preferred_attempt_id", lambda _path: "a1")
    value.attempt_checkpoints = lambda *_args: {
        "latest_completed_checkpoint": "checkpoint-1",
    }
    assert value._operation_blockers("run.evaluate", scope(), project, run) == []
    assert value._operation_blockers("unknown", scope(), project, run) == []

    retry.state = "FAILED"
    retry.decision = {}
    assert value._operation_blockers(
        "attempt.retry", scope(OperationScopeType.ATTEMPT, "run-a::a1"),
        project, retry,
    ) == []


def test_require_and_direct_operation_dispatch_edges(monkeypatch):
    value = app()
    value.operation_availability = lambda *_args: []
    assert error_code(lambda: value._require_operation_available(
        "missing", "demo", OperationScopeType.RUN, "run-a",
    )) == "OPERATION_BLOCKED"

    value.resolve_scope = lambda *_args: (scope(), SimpleNamespace(), object())
    value._require_operation_available = lambda *_args: None
    assert error_code(lambda: value.invoke_direct_operation(
        "missing", "demo", OperationScopeType.RUN, "run-a",
    )) == "INVALID_OPERATION"

    def availability(operation_id, parameters=()):
        definition = module.OPERATIONS_BY_ID[operation_id]
        operation = SimpleNamespace(
            operation_id=operation_id,
            parameters=tuple(SimpleNamespace(key=key) for key in parameters),
        )
        return [SimpleNamespace(operation=operation)]

    value.operation_availability = lambda *_args: availability(
        "run.submit", ("wandb_cloud_sync", "max_gpu_hours"),
    )
    assert error_code(lambda: value.invoke_direct_operation(
        "run.submit", "demo", OperationScopeType.RUN, "run-a",
        {"wandb_cloud_sync": "maybe"},
    )) == "INVALID_OPERATION"
    assert error_code(lambda: value.invoke_direct_operation(
        "run.submit", "demo", OperationScopeType.RUN, "run-a",
        {"max_gpu_hours": "bad"},
    )) == "INVALID_OPERATION"
    assert error_code(lambda: value.invoke_direct_operation(
        "run.submit", "demo", OperationScopeType.RUN, "run-a",
        {"max_gpu_hours": 0},
    )) == "INVALID_OPERATION"

    value.prepare_campaign_archive = lambda *_args, **_kwargs: "campaign"
    value.prepare_object_archive = lambda *_args, **_kwargs: "object"
    value.prepare_attempt_retry = lambda *_args, **_kwargs: "retry"
    value.prepare_attempt_cancel = lambda *_args, **_kwargs: "cancel"
    for operation_id, kind, expected in (
        ("object.archive", OperationScopeType.CAMPAIGN, "campaign"),
        ("object.archive", OperationScopeType.RUN, "object"),
        ("attempt.retry", OperationScopeType.ATTEMPT, "retry"),
        ("attempt.cancel", OperationScopeType.ATTEMPT, "cancel"),
    ):
        local_scope = scope(kind, "run-a::a1" if kind is OperationScopeType.ATTEMPT else "x")
        value.resolve_scope = lambda *_args, local_scope=local_scope: (
            local_scope, SimpleNamespace(), object(),
        )
        params = ("max_gpu_hours",) if operation_id == "attempt.retry" else ()
        value.operation_availability = (
            lambda *_args, operation_id=operation_id, params=params:
            availability(operation_id, params)
        )
        payload = {"max_gpu_hours": 1} if params else {}
        assert value.invoke_direct_operation(
            operation_id, "demo", kind, local_scope.object_id, payload,
        ) == expected

    value.resolve_scope = lambda *_args: (scope(), SimpleNamespace(), object())
    value.operation_availability = lambda *_args: availability("question.create")
    assert error_code(lambda: value.invoke_direct_operation(
        "question.create", "demo", OperationScopeType.RUN, "run-a",
    )) == "INVALID_OPERATION"


def test_archive_and_backfill_prepare_success_and_validation(tmp_path):
    value = app()
    configured = SimpleNamespace()
    value._require_operation_available = lambda *_args: None
    value.bounded_evidence = lambda *_args: {"bounded": True}
    value._prepare_action_intent = lambda _scope, _configured, intent: intent

    for kind, object_id, expected in (
        (OperationScopeType.ATTEMPT, "run-a::a1", "ARCHIVE_ATTEMPT"),
        (OperationScopeType.RUN, "run-a", "ARCHIVE_RUN"),
    ):
        target_scope = scope(kind, object_id)
        value.resolve_scope = lambda *_args, target_scope=target_scope: (
            target_scope, configured, object(),
        )
        result = value.prepare_object_archive(
            "demo", kind, object_id, reason="retired",
        )
        assert result["action"]["kind"] == expected

    value.resolve_scope = lambda *_args: (scope(), configured, object())
    value._publication_targets_available = lambda: ("local",)
    assert error_code(lambda: value.prepare_observability_backfill(
        "demo", OperationScopeType.RUN, "run-a", target="cloud", reason="why",
    )) == "PUBLISHER_UNAVAILABLE"
    assert error_code(lambda: value.prepare_observability_backfill(
        "demo", OperationScopeType.RUN, "run-a", target="local", reason=" ",
    )) == "INVALID_BACKFILL_REASON"


def test_run_attempts_and_failure_assessment_missing_current_attempt(monkeypatch, tmp_path):
    attempts = [SimpleNamespace(
        attempt_id="a1", backend="local", backend_job_id=None, state="RUNNING",
        decision={}, has_submission=False,
    )]
    row = SimpleNamespace(run_id="run-a", run_dir=str(tmp_path), attempts=attempts)
    value = app()
    value.resolve_scope = lambda *_args: (scope(), object(), row)
    monkeypatch.setattr(module, "preferred_attempt_id", lambda _path: "a1")
    assert value.run_attempts("demo", "run-a")["attempts"][0]["current"] is True
    monkeypatch.setattr(module, "preferred_attempt_id", lambda _path: "missing")
    assert value.run_failure_assessment(row)["failure_summary"] is None


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def validation_context(tmp_path: Path):
    run_dir = tmp_path / "run"
    attempt_dir = run_dir / "attempts" / "a1"
    attempt_dir.mkdir(parents=True)
    immutable = {
        "project": "demo", "run_id": "run-a", "attempt_id": "a1",
        "source_id": "source", "image_id": "image", "config_path": "config.yml",
        "seed": 7, "campaign": "study",
    }
    (run_dir / "manifest.yaml").write_text(yaml.safe_dump({
        key: value for key, value in immutable.items() if key != "attempt_id"
    }))
    (attempt_dir / "attempt.yaml").write_text(yaml.safe_dump(immutable))
    write_json(attempt_dir / "status.json", {
        "attempt_id": "a1", "backend_job_id": "job-1", "state": "SUCCEEDED",
    })
    write_json(attempt_dir / "backend.json", {
        "attempt_id": "a1", "backend_job_id": "job-1",
    })
    write_json(attempt_dir / "decision.json", {"attempt_id": "a1"})
    write_json(attempt_dir / "collection.json", {
        "attempt_id": "a1", "backend_job_id": "job-1",
        "process_state": "SUCCEEDED", "model_state": "OBSERVED",
        "latest_completed_checkpoint": "checkpoint-1",
        "latest_completed_checkpoint_step": 1,
        "artifacts": {"checkpoints": {"records": 1}},
    })
    row = SimpleNamespace(run_id="run-a", run_dir=str(run_dir))
    attempt = SimpleNamespace(attempt_id="a1", backend_job_id="job-1", state="SUCCEEDED")
    return run_dir, attempt_dir, row, attempt


def gates_by_id(value):
    return {item["id"]: item for item in value}


def test_attempt_validation_complete_and_conflicting_paths(monkeypatch, tmp_path):
    _run_dir, attempt_dir, row, attempt = validation_context(tmp_path)
    value = app()
    monkeypatch.setattr(module, "preferred_attempt_id", lambda _root: "a1")
    monkeypatch.setattr(
        module, "train_metric_records",
        lambda *_args, **_kwargs: ([{"step": 1}], Path("metrics"), "a1"),
    )
    gates = gates_by_id(value._attempt_validation_gates(
        "demo", row, attempt, attempt_dir, require_current=True,
    ))
    assert gates["attempt.immutable_provenance"]["status"] == "PASS"
    assert gates["attempt.current"]["status"] == "PASS"
    assert gates["attempt.evidence_identity"]["status"] == "PASS"
    assert gates["attempt.backend_job_id"]["status"] == "PASS"
    assert gates["attempt.execution_layers"]["status"] == "PASS"
    assert gates["attempt.model_evidence"]["status"] == "PASS"
    assert gates["attempt.checkpoint_evidence"]["status"] == "PASS"
    assert gates["attempt.artifact_evidence"]["status"] == "PASS"

    manifest = yaml.safe_load((attempt_dir / "attempt.yaml").read_text())
    manifest["source_id"] = "different"
    (attempt_dir / "attempt.yaml").write_text(yaml.safe_dump(manifest))
    write_json(attempt_dir / "status.json", {
        "attempt_id": "other", "backend_job_id": "job-2", "state": "SUCCEEDED",
    })
    write_json(attempt_dir / "backend.json", {
        "attempt_id": "a1", "backend_job_id": "job-3",
    })
    write_json(attempt_dir / "collection.json", {
        "attempt_id": "a1", "backend_job_id": "job-4",
        "process_state": "FAILED", "model_state": "OBSERVED",
        "artifacts": {"empty": {"records": 0}},
    })
    monkeypatch.setattr(module, "preferred_attempt_id", lambda _root: "other")
    monkeypatch.setattr(
        module, "train_metric_records",
        lambda *_args, **_kwargs: ([{"step": 1}], Path("metrics"), "other"),
    )
    gates = gates_by_id(value._attempt_validation_gates(
        "demo", row, attempt, attempt_dir, require_current=True,
    ))
    assert gates["attempt.immutable_provenance"]["status"] == "BLOCKED"
    assert gates["attempt.current"]["status"] == "BLOCKED"
    assert gates["attempt.evidence_identity"]["status"] == "BLOCKED"
    assert gates["attempt.backend_job_id"]["status"] == "BLOCKED"
    assert gates["attempt.execution_layers"]["status"] == "BLOCKED"
    assert gates["attempt.model_evidence"]["status"] == "BLOCKED"
    assert gates["attempt.artifact_evidence"]["status"] == "UNKNOWN"

    monkeypatch.setattr(module, "preferred_attempt_id", lambda _root: None)
    gates = gates_by_id(value._attempt_validation_gates(
        "demo", row, attempt, attempt_dir, require_current=True,
    ))
    assert gates["attempt.current"]["status"] == "UNKNOWN"

    write_json(attempt_dir / "status.json", {"state": "SUCCEEDED"})
    write_json(attempt_dir / "backend.json", {
        "attempt_id": "a1", "backend_job_id": "job-1",
    })
    write_json(attempt_dir / "decision.json", {"attempt_id": "a1"})
    write_json(attempt_dir / "collection.json", {
        "attempt_id": "a1", "process_state": "SUCCEEDED",
        "model_state": "OBSERVED",
    })
    gates = gates_by_id(value._attempt_validation_gates(
        "demo", row, attempt, attempt_dir, require_current=False,
    ))
    assert gates["attempt.evidence_identity"]["status"] == "UNKNOWN"


@pytest.mark.parametrize(("integrity", "expected"), [
    ({"source": False}, "BLOCKED"),
    ({"source": True}, "PASS"),
])
def test_harness_attempt_validation_integrity_paths(
    monkeypatch, tmp_path, integrity, expected,
):
    run_dir = tmp_path / expected
    attempt_dir = run_dir / "attempts" / "a1"
    attempt_dir.mkdir(parents=True)
    write_json(run_dir / "manifest.json", {
        "project": "demo", "run_id": "run-a",
    })
    write_json(run_dir / "status.json", {
        "attempt_id": "a1", "state": "SUCCEEDED",
    })
    write_json(attempt_dir / "attempt.json", {
        "run_id": "run-a", "attempt_id": "a1", "state": "SUCCEEDED",
    })
    write_json(attempt_dir / "submission.json", {
        "attempt_id": "a1", "gpu": 0,
    })
    write_json(attempt_dir / "summary.json", {
        "attempt_id": "a1", "integrity": integrity, "metrics": {"loss": 1},
    })
    row = SimpleNamespace(run_id="run-a", run_dir=str(run_dir))
    attempt = SimpleNamespace(attempt_id="a1", backend_job_id=None, state="SUCCEEDED")
    monkeypatch.setattr(module, "preferred_attempt_id", lambda _root: "a1")
    monkeypatch.setattr(
        module, "train_metric_records",
        lambda *_args, **_kwargs: ([{"step": 1}], Path("metrics"), "a1"),
    )
    gates = gates_by_id(app()._attempt_validation_gates(
        "demo", row, attempt, attempt_dir, require_current=False,
    ))
    assert gates["attempt.identity"]["status"] == "PASS"
    assert gates["attempt.immutable_provenance"]["status"] == expected
    assert gates["attempt.backend_job_id"]["status"] == "PASS"


def test_attempt_events_with_no_sources_and_campaign_file_edges(tmp_path):
    attempt_dir = tmp_path / "attempts/a1"
    attempt_dir.mkdir(parents=True)
    row = SimpleNamespace(run_id="run-a", run_dir=str(tmp_path))
    attempt = SimpleNamespace(attempt_id="a1")
    value = app()
    value._attempt_context = lambda *_args: (scope(), object(), row, attempt, attempt_dir)
    assert value.attempt_events("demo", "run-a::a1")["events"] == []

    project = ResearchProject(project="demo", title="Demo", run_roots=[], base_dir=tmp_path)
    assert error_code(lambda: value._campaign_file(project, "missing")) == (
        "CAMPAIGN_FILE_MISSING"
    )
    project.campaigns = [CampaignRef(name="study", file="study.yml")]
    assert value._campaign_file(project, "study") == (tmp_path / "study.yml").resolve()


@pytest.mark.parametrize(("budget", "controller", "state", "submitted", "code"), [
    (0, True, "NOT_SUBMITTED", False, "INVALID_GPU_BUDGET"),
    (1, False, "NOT_SUBMITTED", False, "CONTROLLER_UNAVAILABLE"),
    (1, True, "RUNNING", False, "RUN_NOT_SUBMITTABLE"),
    (1, True, "NOT_SUBMITTED", True, "RUN_ALREADY_SUBMITTED"),
])
def test_prepare_run_submit_validation_matrix(
    tmp_path, budget, controller, state, submitted, code,
):
    configured = SimpleNamespace(
        controller=(SimpleNamespace() if controller else None),
    )
    row = SimpleNamespace(
        scheduler_state=state, attempts=[SimpleNamespace(has_submission=submitted)],
    )
    value = app()
    value._require_operation_available = lambda *_args: None
    value.resolve_scope = lambda *_args: (scope(), configured, row)
    assert error_code(lambda: value.prepare_run_submit(
        "demo", "run-a", max_gpu_hours=budget,
    )) == code


def test_misc_read_model_and_observability_scope_edges(tmp_path):
    value = app()
    attempt = SimpleNamespace(
        model_dump=lambda **_kwargs: {"attempt_id": "a1"},
        attempt_id="a1", state="RUNNING", backend=None, backend_job_id=None,
        decision={}, has_submission=False,
    )
    attempt_scope = scope(OperationScopeType.ATTEMPT, "run-a::a1")
    configured = SimpleNamespace(base_dir=None, authored_file=None)
    value.resolve_scope = lambda *_args: (attempt_scope, configured, attempt)
    value.bounded_evidence = lambda *_args: {}
    assert value.object_show(
        "demo", OperationScopeType.ATTEMPT, "run-a::a1",
    )["object"]["attempt_id"] == "a1"

    assert value._observability_attempts(
        scope(OperationScopeType.RESEARCH_QUESTION, "q1"),
        SimpleNamespace(project="demo"), object(),
    ) == []

    attempt_dir = tmp_path / "attempts/a1"
    attempt_dir.mkdir(parents=True)
    write_json(attempt_dir / "collection.json", {"artifacts": {"files": 1}})
    row = SimpleNamespace(run_id="run-a", run_dir=str(tmp_path), artifacts={})
    value._attempt_context = lambda *_args: (
        attempt_scope, configured, row, attempt, attempt_dir,
    )
    assert value.attempt_artifacts(
        "demo", "run-a::a1",
    )["summary"] == {"files": 1}


def test_resolve_scope_returns_existing_research_question():
    question = SimpleNamespace(id="q1")
    project = SimpleNamespace(
        project="demo", research_questions=[question], campaigns=[],
    )
    value = app(
        project=lambda _name: project,
        index=SimpleNamespace(get_run=lambda *_args: None),
    )
    resolved_scope, _, resolved = value.resolve_scope(
        "demo", OperationScopeType.RESEARCH_QUESTION, "q1",
    )
    assert resolved_scope.object_id == "q1"
    assert resolved is question


def test_action_reconcile_refresh_policy_and_project_adapter_edges(monkeypatch):
    def runtime_with(snapshot, reconcile):
        return SimpleNamespace(
            action_store=SimpleNamespace(snapshot=snapshot),
            action_service=SimpleNamespace(reconcile=reconcile, execute=lambda *_args: {}),
            observability=SimpleNamespace(enable_cloud=lambda *_args: None),
            index=object(),
        )

    missing = ExperimentServerApplication(runtime_with(
        lambda _id: (_ for _ in ()).throw(FileNotFoundError()), lambda _id: {},
    ))
    assert error_code(lambda: missing.reconcile_action("missing")) == "UNKNOWN_ACTION"
    blocked = ExperimentServerApplication(runtime_with(
        lambda _id: {},
        lambda _id: (_ for _ in ()).throw(RuntimeError("blocked")),
    ))
    assert error_code(lambda: blocked.reconcile_action("blocked")) == "ACTION_BLOCKED"

    runtime = runtime_with(lambda _id: {}, lambda _id: {})
    runtime.project = lambda name: (_ for _ in ()).throw(KeyError(name))
    value = ExperimentServerApplication(runtime)
    value._refresh_action_project({})
    value._refresh_action_project({"scope": {}})
    value._refresh_action_project({"scope": {"project": "missing"}})

    enabled = []
    value.runtime.observability.enable_cloud = lambda *args: enabled.append(args)
    verified = {"execution": {"status": "VERIFIED"}}
    value._activate_observability_policy(
        {"preflight_summary": {"wandb_cloud_sync": True}}, verified,
    )
    value._activate_observability_policy({
        "scope": {"project": "demo"},
        "preflight_summary": {"wandb_cloud_sync": True, "run_id": "run-a"},
    }, verified)
    value._activate_observability_policy({
        "scope": {"project": "demo"},
        "preflight_summary": {
            "wandb_cloud_sync": True, "run_id": "run-a", "attempt_id": "a1",
        },
    }, verified)
    assert enabled == [("demo", "run-a", "a1")]

    application_error = ApplicationError("already mapped")
    value.runtime.action_store.snapshot = lambda _id: {}
    value.runtime.action_service.execute = lambda *_args: (_ for _ in ()).throw(
        application_error,
    )
    with pytest.raises(ApplicationError) as caught:
        value._execute_action_local("a", "confirm")
    assert caught.value is application_error

    value.runtime.action_service.execute = lambda *_args: {
        "execution": {"status": "PENDING"},
    }
    value._refresh_action_project = lambda _action: (_ for _ in ()).throw(
        AssertionError("must not refresh"),
    )
    assert value._execute_action_local("a", "confirm")["execution"]["status"] == (
        "PENDING"
    )

    project_service = SimpleNamespace(unregister=lambda *args, **kwargs: "removed")
    value.project_service = project_service
    assert value.project_unregister("demo") == "removed"
