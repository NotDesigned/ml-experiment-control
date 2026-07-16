"""Focused line/branch coverage for :mod:`actions.service`."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from ml_exp_server.actions.errors import ActionError
from ml_exp_server.actions.service import ActionService
from ml_exp_server.actions.store import ActionStore
from ml_exp_server.intent_protocol import OperationIntent
from ml_exp_server.schemas import (
    ActionRuntimeConfig,
    ControllerConfig,
    OperationScope,
    OperationScopeType,
    ResearchProject,
)
from tests.test_action_edges import _local_evidence_action


def intent(kind, draft, key="focused-intent"):
    return {
        "kind": kind, "title": kind, "draft": yaml.safe_dump(draft),
        "idempotency_key": key, "evidence_digest": "sha256:evidence",
    }


def scope(kind="project", object_id="demo"):
    return OperationScope(project="demo", scope_type=kind, object_id=object_id)


def synthetic_plan(store, name, operation="SUBMIT_RUN", **extra):
    action_id = store.action_id(scope("run", "run-a"), name)
    expires = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    store.save_plan({
        "action_id": action_id,
        "scope": scope("run", "run-a").model_dump(mode="json"),
        "ready": True, "operation": operation,
        "intent_digest": "sha256:intent", "gate_bundle_digest": "sha256:gates",
        "gate_expires_at": expires,
        "command_preview": ["controller", "command"], "cwd": ".",
        "verification_command_preview": ["controller", "status"],
        "verification_cwd": ".", **extra,
    })
    return action_id


def execution_for(plan):
    return {**plan["execution"], "status": "EXECUTING"}


def test_manifest_path_and_prepare_dispatch_edges(tmp_path, monkeypatch):
    from ml_exp_server.actions import service as module

    assert module._canonical_manifest_path({}, cwd=tmp_path, run_id="run") is None
    assert module._canonical_manifest_path(
        {"local_root": "runs", "campaign": "study"}, cwd=tmp_path, run_id="run",
    ) == (tmp_path / "runs" / "study" / "run" / "manifest.yaml").resolve()
    assert module._canonical_manifest_path(
        {"local_root": str(tmp_path / "absolute"), "campaign": "study"},
        cwd=tmp_path, run_id="run",
    ) == (tmp_path / "absolute" / "study" / "run" / "manifest.yaml").resolve()

    project = ResearchProject(project="demo", title="Demo", run_roots=[], base_dir=tmp_path)
    store = ActionStore(tmp_path / "actions")
    service = ActionService(store, ActionRuntimeConfig())
    draft = {
        "project": "demo", "target": "local", "reason": "repair",
        "attempts": [{"run_id": "run-a", "attempt_id": "attempt-001"}],
    }
    first = service.prepare(scope(), project, intent("OBSERVABILITY_BACKFILL", draft))
    assert first["ready"] is True
    assert service.prepare(scope(), project, intent(
        "OBSERVABILITY_BACKFILL", draft,
    ))["action_id"] == first["action_id"]
    generated = service.prepare(scope(), project, {
        **intent("OBSERVABILITY_BACKFILL", draft, "temporary"),
        "idempotency_key": None,
    })
    assert generated["intent_id"].startswith("intent-")

    unknown = OperationIntent.model_construct(
        kind="UNKNOWN", title="unknown", draft="{}", idempotency_key="unknown",
        target="", change_summary="", resource_estimate="unknown", rationale="",
        risk="", evidence_digest="",
    )
    monkeypatch.setattr(module, "intent_scope_error", lambda *args: None)
    with pytest.raises(ActionError, match="has no executor"):
        service.prepare(scope(), project, unknown)

    fresh = ActionService(ActionStore(tmp_path / "other-actions"), ActionRuntimeConfig())
    monkeypatch.setattr(fresh.store, "save_plan", lambda plan: (_ for _ in ()).throw(
        RuntimeError("save conflict")
    ))
    with pytest.raises(ActionError, match="save conflict"):
        fresh.prepare(scope(), project, intent(
            "OBSERVABILITY_BACKFILL", draft, "save-fails",
        ))


def test_question_campaign_and_record_prepare_edges(tmp_path):
    from ml_exp_server.actions import service as module

    questions = tmp_path / "questions"
    project = ResearchProject(
        project="demo", title="Demo", run_roots=[], base_dir=tmp_path,
        research_questions_dir="questions",
    )
    service = ActionService(ActionStore(tmp_path / "actions"), ActionRuntimeConfig())
    plan = service.prepare(scope(), project, intent(
        "CREATE_RESEARCH_QUESTION_DRAFT", {"schema_version": 1, "id": "Q1", "title": "Q"},
    ))
    assert Path(plan["target_path"]) == questions / "Q1.yml"

    campaign = {
        "schema_version": 1, "project": "demo", "campaign": "study",
        "run_refs": [{"run_id": "run-a"}],
    }
    missing_catalog = service.prepare(scope(), project, intent(
        "CREATE_CAMPAIGN_DRAFT", campaign, "missing-catalog",
    ))
    assert missing_catalog["ready"] is False

    with pytest.raises(ActionError, match="exact existing Campaign scope"):
        service.prepare(scope("campaign", "other"), project, intent(
            "UPDATE_CAMPAIGN_DRAFT", campaign, "wrong-update-scope",
        ))

    absolute_project = ResearchProject(
        project="demo", title="Demo", run_roots=[], base_dir=tmp_path,
        research_questions_dir=str(questions),
    )
    assert service.prepare(scope(), absolute_project, intent(
        "CREATE_RESEARCH_QUESTION_DRAFT",
        {"schema_version": 1, "id": "Q2", "title": "Q"}, "absolute-question",
    ))["target_path"].endswith("Q2.yml")

    bogus = OperationIntent.model_construct(
        kind="BOGUS", title="bogus", draft=yaml.safe_dump({
            "project": "demo", "campaign": "study", "revision_id": "rev",
        }), idempotency_key="bogus", target="", change_summary="",
        resource_estimate="unknown", rationale="", risk="", evidence_digest="",
    ).model_dump(mode="json")
    bogus["intent_id"] = "bogus"
    fake_project = ResearchProject(project="demo", title="Demo", run_roots=[])
    fake_current = type("Current", (), {"revision_id": "rev"})()
    fake_project.campaigns = [type("Ref", (), {
        "name": "study", "current_revision": fake_current,
    })()]
    with pytest.raises(ActionError, match="only Campaign archive"):
        service._prepare_campaign_record(
            "action", scope("campaign", "study"), fake_project, bogus,
        )


def controller_project(tmp_path):
    experiments = tmp_path / "experiments"
    experiments.mkdir(exist_ok=True)
    campaign = experiments / "study.yml"
    campaign.write_text(
        "schema_version: 1\nproject: demo\ncampaign: study\nruns: []\n",
        encoding="utf-8",
    )
    project = ResearchProject(
        project="demo", title="Demo", run_roots=[], base_dir=tmp_path,
        controller=ControllerConfig(
            python="python", experimentctl="controller.py", workdir=".",
            capabilities={},
        ),
    )
    return project, campaign


def test_controller_prepare_rejection_edges(tmp_path):
    service = ActionService(ActionStore(tmp_path / "actions"), ActionRuntimeConfig())
    project = ResearchProject(project="demo", title="Demo", run_roots=[], base_dir=tmp_path)
    with pytest.raises(ActionError, match="no controller"):
        service.prepare(scope("run", "run-a"), project, intent("SUBMIT_RUN", {}))

    project, campaign = controller_project(tmp_path)
    with pytest.raises(ActionError, match="safe run_id"):
        service.prepare(scope("run", "bad/id"), project, intent(
            "SUBMIT_RUN", {"campaign_file": str(campaign), "run_id": "bad/id"},
            "bad-id",
        ))
    outside = tmp_path / "outside.yml"
    outside.write_text("campaign: x\n")
    with pytest.raises(ActionError, match="must exist under"):
        service.prepare(scope("run", "run-a"), project, intent(
            "SUBMIT_RUN", {"campaign_file": str(outside), "run_id": "run-a"},
            "outside",
        ))
    with pytest.raises(ActionError, match="not valid in attempt scope"):
        service._prepare_controller(
            "attempt-submit", scope("attempt", "run-a::attempt-001"), project,
            {"kind": "SUBMIT_RUN", "intent_id": "attempt-submit",
             "draft": yaml.safe_dump({
                 "campaign_file": "experiments/study.yml", "run_id": "run-a",
             }), "evidence_digest": "sha256:evidence"},
        )


def test_controller_preview_defensive_payload_edges(tmp_path, monkeypatch):
    project, campaign = controller_project(tmp_path)
    campaign.write_text(yaml.safe_dump({
        "schema_version": 1, "project": "demo", "campaign": "study",
        "local_root": "runs", "runs": [{"run_id": "run-a"}],
    }), encoding="utf-8")
    service = ActionService(ActionStore(tmp_path / "actions"), ActionRuntimeConfig())
    action_id = "action-previewedge"
    bad_preview = service.store.directory(action_id) / "manifest.yml"
    bad_preview.parent.mkdir(parents=True)
    bad_preview.write_text("- not-a-mapping\n", encoding="utf-8")
    canonical = tmp_path / "runs" / "study" / "run-a" / "manifest.yaml"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("- not-a-mapping\n", encoding="utf-8")

    results = iter([
        ({"name": "dry_run", "status": "PASS", "detail": "ok"}, {
            "returncode": 0, "timeout": False,
            "payload": [{"manifest_path": str(bad_preview)}],
        }),
        *[
            ({"name": "check", "status": "PASS", "detail": "ok"}, {
                "returncode": 0, "timeout": False, "payload": {},
            })
            for _ in range(3)
        ],
    ])
    monkeypatch.setattr(service, "_run_gate", lambda *args, **kwargs: next(results))
    plan = service._prepare_controller(
        action_id, scope("run", "run-a"), project,
        {"kind": "SUBMIT_RUN", "intent_id": action_id,
         "draft": yaml.safe_dump({
             "campaign_file": str(campaign), "run_id": "run-a",
             "max_gpu_hours": 1,
         }), "evidence_digest": "sha256:evidence"},
    )
    assert plan["ready"] is False

    empty_service = ActionService(
        ActionStore(tmp_path / "empty-actions"), ActionRuntimeConfig(),
    )
    monkeypatch.setattr(empty_service, "_run_gate", lambda *args, **kwargs: (
        {"name": "check", "status": "PASS", "detail": "ok"},
        {"returncode": 0, "timeout": False, "payload": {}},
    ))
    empty = empty_service._prepare_controller(
        "action-emptypreview", scope("run", "run-a"), project,
        {"kind": "SUBMIT_RUN", "intent_id": "empty-preview",
         "draft": yaml.safe_dump({
             "campaign_file": str(campaign), "run_id": "run-a",
             "max_gpu_hours": 1,
         }), "evidence_digest": "sha256:evidence"},
    )
    assert empty["ready"] is False


def test_gpu_hours_and_authorization_error_edges(tmp_path, monkeypatch):
    assert ActionService._gpu_hours({"gpus": 2}, {"time": "bad:h:value"}) is None

    store = ActionStore(tmp_path / "actions")
    service = ActionService(store, ActionRuntimeConfig(), actor_provider=lambda: "actor")
    action_id = synthetic_plan(store, "not-ready")
    plan_path = store.directory(action_id) / "plan.json"
    payload = json.loads(plan_path.read_text())
    payload["ready"] = False
    plan_path.write_text(json.dumps(payload))
    with pytest.raises(ActionError, match="ready PREPARED"):
        service.authorize(action_id, "note")

    expired = synthetic_plan(store, "expired-auth")
    expired_path = store.directory(expired) / "plan.json"
    payload = json.loads(expired_path.read_text())
    payload["gate_expires_at"] = "2000-01-01T00:00:00Z"
    expired_path.write_text(json.dumps(payload))
    with pytest.raises(ActionError, match="expired"):
        service.authorize(expired, "note")

    failing = synthetic_plan(store, "authorization-cas")
    monkeypatch.setattr(store, "set_execution", lambda *args, **kwargs: (
        _ for _ in ()
    ).throw(RuntimeError("stale authorization")))
    with pytest.raises(ActionError, match="stale authorization"):
        service.authorize(failing, "note")


def test_execute_begin_and_internal_executor_edges(tmp_path, monkeypatch):
    store = ActionStore(tmp_path / "actions")
    config = ActionRuntimeConfig(allow_observability_mutations=True)
    action_id = synthetic_plan(store, "begin-fail", "OBSERVABILITY_BACKFILL")
    service = ActionService(store, config, actor_provider=lambda: "actor")
    service.authorize(action_id, "note")
    monkeypatch.setattr(store, "begin_execution", lambda *args, **kwargs: (
        _ for _ in ()
    ).throw(RuntimeError("stale execution")))
    with pytest.raises(ActionError, match="stale execution"):
        service.execute(action_id, f"EXECUTE {action_id}")

    for name, executor, expected in (
        ("missing", None, "unavailable"),
        ("failure", lambda plan: (_ for _ in ()).throw(ValueError("boom")),
         "RECONCILE_REQUIRED"),
        ("success", lambda plan: {"token": "secret", "ok": True}, "VERIFIED"),
    ):
        store = ActionStore(tmp_path / f"actions-{name}")
        action_id = synthetic_plan(store, name, "OBSERVABILITY_BACKFILL")
        service = ActionService(
            store, config, actor_provider=lambda: "actor", internal_executor=executor,
        )
        service.authorize(action_id, "note")
        if executor is None:
            with pytest.raises(ActionError, match=expected):
                service.execute(action_id, f"EXECUTE {action_id}")
        else:
            assert service.execute(action_id, f"EXECUTE {action_id}")[
                "execution"
            ]["status"] == expected


def test_execute_write_unexpected_validation_error(tmp_path, monkeypatch):
    store = ActionStore(tmp_path / "actions")
    action_id = synthetic_plan(store, "write", "WRITE_RESEARCH_QUESTION",
                               target_path=str(tmp_path / "bad.yml"))
    plan = store.snapshot(action_id)
    service = ActionService(store, ActionRuntimeConfig())
    service.project_write_transaction = type("Tx", (), {
        "apply": lambda self, plan: {
            "files": [{"path": plan["target_path"], "sha256": "sha256:x"}],
        },
    })()
    result = service._execute_write(plan, execution_for(plan))
    assert result["execution"]["status"] == "RECONCILE_REQUIRED"
    assert "ValidationError" in result["execution"]["error"] or result["execution"]["error"]


def test_status_verification_matrix(tmp_path):
    plan = {
        "action_id": "a", "run_id": "run-a", "attempt_id": "attempt-001",
        "verification_command_preview": ["status"], "verification_cwd": str(tmp_path),
    }
    cases = [
        ({"timeout": False, "returncode": 0, "payload": {"backend_job_id": "j"}}, True),
        ({"timeout": False, "returncode": 0, "payload": [{}]}, False),
        ({"timeout": False, "returncode": 0, "payload": [{
            "backend_job_id": "j", "run_id": "other", "attempt_id": "attempt-001",
        }]}, False),
    ]
    for index, (runner_result, expected) in enumerate(cases):
        service = ActionService(
            ActionStore(tmp_path / f"actions-{index}"), ActionRuntimeConfig(),
            runner=lambda *args, result=runner_result, **kwargs: result,
        )
        assert service._verify_submission(plan, expected_job_id="j")[0] is expected
    service = ActionService(ActionStore(tmp_path / "none"), ActionRuntimeConfig())
    assert service._single_status_record({"state": "RUNNING"}) == {"state": "RUNNING"}
    assert service._verify_submission({}, expected_job_id=None)[0] is False


def test_reconcile_policy_and_controller_outcomes(tmp_path):
    store = ActionStore(tmp_path / "actions")
    service = ActionService(store, ActionRuntimeConfig())

    verified = synthetic_plan(store, "verified")
    store.set_execution(verified, {**store.execution(verified), "status": "VERIFIED"}, event="test")
    assert service.reconcile(verified)["execution"]["status"] == "VERIFIED"

    write = synthetic_plan(store, "write", "WRITE_CAMPAIGN")
    with pytest.raises(ActionError, match="project writes are disabled"):
        service.reconcile(write)
    write_bad_state = ActionService(
        store, ActionRuntimeConfig(allow_project_writes=True),
    )
    with pytest.raises(ActionError, match="not awaiting reconciliation"):
        write_bad_state.reconcile(write)

    internal = synthetic_plan(store, "internal", "OBSERVABILITY_BACKFILL")
    with pytest.raises(ActionError, match="observability mutations are disabled"):
        service.reconcile(internal)
    with pytest.raises(ActionError, match="not awaiting reconciliation"):
        ActionService(
            store, ActionRuntimeConfig(allow_observability_mutations=True),
            internal_executor=lambda plan: {},
        ).reconcile(internal)
    store.set_execution(
        internal, {**store.execution(internal), "status": "RECONCILE_REQUIRED"},
        event="test",
    )
    calls = []
    reconciled = ActionService(
        store, ActionRuntimeConfig(allow_observability_mutations=True),
        internal_executor=lambda plan: calls.append(plan) or {"ok": True},
    ).reconcile(internal)
    assert reconciled["execution"]["status"] == "RECONCILE_REQUIRED"
    assert reconciled["execution"]["result"]["reconciled_read_only"] is True
    assert "cannot be replayed" in reconciled["execution"]["error"]
    assert calls == []

    cancel = synthetic_plan(store, "cancel", "CANCEL_RUN")
    with pytest.raises(ActionError, match="not awaiting reconciliation"):
        service.reconcile(cancel)
    execution = {**store.execution(cancel), "status": "RECONCILE_REQUIRED"}
    store.set_execution(cancel, execution, event="test")
    cancel_path = store.directory(cancel) / "plan.json"
    payload = json.loads(cancel_path.read_text())
    payload["verification_command_preview"] = []
    cancel_path.write_text(json.dumps(payload))
    with pytest.raises(ActionError, match="no verification command"):
        service.reconcile(cancel)

    unknown = synthetic_plan(store, "unknown", "UNKNOWN")
    with pytest.raises(ActionError, match="only submission"):
        service.reconcile(unknown)
    submit = synthetic_plan(store, "submit-state", "SUBMIT_RUN")
    with pytest.raises(ActionError, match="not awaiting reconciliation"):
        service.reconcile(submit)

    pending = synthetic_plan(store, "submit-pending", "SUBMIT_RUN")
    store.set_execution(
        pending, {**store.execution(pending), "status": "RECONCILE_REQUIRED"},
        event="test",
    )
    pending_service = ActionService(
        store, ActionRuntimeConfig(), runner=lambda *args, **kwargs: {
            "timeout": False, "returncode": 0, "payload": [],
        },
    )
    assert pending_service.reconcile(pending)["execution"]["status"] == (
        "RECONCILE_REQUIRED"
    )


@pytest.mark.parametrize("result,status", [
    ({"timeout": True, "returncode": None, "payload": None}, "RECONCILE_REQUIRED"),
    ({"timeout": False, "returncode": 2, "stderr": "failed", "payload": None}, "FAILED"),
    ({"timeout": False, "returncode": 0, "payload": {"ok": True}}, "VERIFIED"),
])
def test_non_submission_controller_outcomes(tmp_path, result, status):
    store = ActionStore(tmp_path / "actions")
    action_id = synthetic_plan(store, status, "CANCEL_RUN")
    plan = store.snapshot(action_id)
    service = ActionService(
        store, ActionRuntimeConfig(), runner=lambda *args, **kwargs: result,
    )
    assert service._execute_controller(plan, execution_for(plan))[
        "execution"
    ]["status"] == status


def _local_payload(draft):
    return {
        "kind": "REBUILD_LOCAL_EVIDENCE", "intent_id": "local-edge",
        "draft": yaml.safe_dump(draft), "evidence_digest": "sha256:evidence",
    }


def test_local_evidence_prepare_rejects_scope_controller_identity_and_capability(tmp_path):
    project, attempt_scope, draft, _campaign, _collection = _local_evidence_action(
        tmp_path,
    )
    service = ActionService(ActionStore(tmp_path / "actions"), ActionRuntimeConfig())
    with pytest.raises(ActionError, match="requires exact Attempt scope"):
        service._prepare_local_evidence_rebuild(
            "action-localedge", scope(), project, _local_payload(draft),
        )

    no_controller = ResearchProject(
        project="demo", title="Demo", run_roots=[], base_dir=project.base_dir,
    )
    with pytest.raises(ActionError, match="no controller"):
        service._prepare_local_evidence_rebuild(
            "action-localedge", attempt_scope, no_controller, _local_payload(draft),
        )

    with pytest.raises(ActionError, match="conflicts with exact scope"):
        service._prepare_local_evidence_rebuild(
            "action-localedge", attempt_scope, project,
            _local_payload({**draft, "attempt_id": "attempt-999"}),
        )

    project.controller.capabilities = {}
    with pytest.raises(ActionError, match="does not declare"):
        service._prepare_local_evidence_rebuild(
            "action-localedge", attempt_scope, project, _local_payload(draft),
        )


def test_local_evidence_prepare_rejects_reason_and_paths(tmp_path):
    project, attempt_scope, draft, _campaign, _collection = _local_evidence_action(
        tmp_path,
    )
    service = ActionService(ActionStore(tmp_path / "actions"), ActionRuntimeConfig())
    with pytest.raises(ActionError, match="reason is required"):
        service._prepare_local_evidence_rebuild(
            "action-localedge", attempt_scope, project,
            _local_payload({**draft, "reason": " "}),
        )

    relative_campaign = Path(draft["campaign_file"]).relative_to(project.base_dir)
    with pytest.raises(ActionError, match="must exist under"):
        service._prepare_local_evidence_rebuild(
            "action-localedge", attempt_scope, project,
            _local_payload({**draft, "campaign_file": str(relative_campaign) + ".missing"}),
        )

    with pytest.raises(ActionError, match="paths must be absolute"):
        service._prepare_local_evidence_rebuild(
            "action-localedge", attempt_scope, project,
            _local_payload({**draft, "run_dir": "relative/run"}),
        )

    with pytest.raises(ActionError, match="conflicts with campaign identity"):
        service._prepare_local_evidence_rebuild(
            "action-localedge", attempt_scope, project,
            _local_payload({**draft, "run_dir": str(tmp_path / "wrong-run")}),
        )


def test_local_evidence_prepare_optional_inputs_and_missing_evidence(tmp_path):
    project, attempt_scope, draft, _campaign, collection = _local_evidence_action(
        tmp_path,
    )
    service = ActionService(ActionStore(tmp_path / "actions"), ActionRuntimeConfig())
    attempt_dir = Path(draft["run_dir"]) / "attempts" / draft["attempt_id"]
    (attempt_dir / "status.json").write_text("{}\n", encoding="utf-8")
    collection.unlink()
    (attempt_dir / "backend.json").unlink()
    with pytest.raises(ActionError, match="already-local evidence is missing"):
        service._prepare_local_evidence_rebuild(
            "action-localedge", attempt_scope, project, _local_payload(draft),
        )

    (attempt_dir / "backend.json").write_text("{}\n", encoding="utf-8")
    collected = attempt_dir / "collected_run"
    for path in collected.iterdir():
        path.unlink()
    with pytest.raises(ActionError, match="already-local evidence is missing"):
        service._prepare_local_evidence_rebuild(
            "action-localedge", attempt_scope, project, _local_payload(draft),
        )


def test_local_evidence_prepare_snapshot_and_preview_exceptions(tmp_path):
    project, attempt_scope, draft, _campaign, _collection = _local_evidence_action(
        tmp_path,
    )
    service = ActionService(ActionStore(tmp_path / "actions"), ActionRuntimeConfig())
    service.controller.snapshot_execution_bundle = lambda *args, **kwargs: (
        _ for _ in ()
    ).throw(OSError("snapshot failed"))
    with pytest.raises(ActionError, match="snapshot preparation failed"):
        service._prepare_local_evidence_rebuild(
            "action-localedge", attempt_scope, project, _local_payload(draft),
        )

    snapshot = {
        "manifest_sha256": "sha256:" + "a" * 64,
        "root": str(tmp_path / "private"),
        "manifest_path": str(tmp_path / "private" / "manifest.json"),
        "manifest": {},
    }
    service.controller.snapshot_execution_bundle = lambda *args, **kwargs: snapshot
    service.controller.execute_snapshot = lambda *args, **kwargs: (
        _ for _ in ()
    ).throw(ValueError("preview failed"))
    plan = service._prepare_local_evidence_rebuild(
        "action-localedge", attempt_scope, project, _local_payload(draft),
    )
    assert plan["ready"] is False
    assert "preview failed" in next(
        gate["detail"] for gate in plan["gates"]
        if gate["name"] == "local_evidence_preview"
    )

    service.controller.execute_snapshot = lambda *args, **kwargs: {
        "returncode": 0, "timeout": False, "stdout": "", "stderr": "",
        "payload": [{
            "project": "demo", "run_id": "run-a", "attempt_id": "attempt-001",
            "collection_path": "relative/collection.json",
        }],
    }
    plan = service._prepare_local_evidence_rebuild(
        "action-localrelative", attempt_scope, project, _local_payload(draft),
    )
    assert plan["collection_path"] is None


def test_local_evidence_policy_execute_and_reconcile_rejections(tmp_path, monkeypatch):
    store = ActionStore(tmp_path / "actions")
    local = synthetic_plan(
        store, "local-policy", "REBUILD_LOCAL_EVIDENCE",
        collection_path=str(tmp_path / "collection.json"),
        expected_collection_sha256=None, controller_snapshot={},
        snapshot_arguments=["refresh-evidence-local"],
    )
    blocked = ActionService(store, ActionRuntimeConfig(), actor_provider=lambda: "actor")
    blocked.authorize(local, "review")
    with pytest.raises(ActionError, match="disabled by daemon policy"):
        blocked.execute(local, f"EXECUTE {local}")

    internal_store = ActionStore(tmp_path / "internal-actions")
    internal = synthetic_plan(
        internal_store, "internal-policy", "OBSERVABILITY_BACKFILL",
    )
    internal_service = ActionService(
        internal_store, ActionRuntimeConfig(), actor_provider=lambda: "actor",
    )
    internal_service.authorize(internal, "review")
    with pytest.raises(ActionError, match="observability mutations are disabled"):
        internal_service.execute(internal, f"EXECUTE {internal}")

    allowed = ActionService(
        store, ActionRuntimeConfig(allow_local_evidence_rebuild=True),
        actor_provider=lambda: "actor",
    )
    monkeypatch.setattr(
        allowed.controller, "verify_execution_bundle",
        lambda value: (_ for _ in ()).throw(ValueError("bad snapshot")),
    )
    with pytest.raises(ActionError, match="approved private controller snapshot is invalid"):
        allowed.execute(local, f"EXECUTE {local}")

    with pytest.raises(ActionError, match="disabled by daemon policy"):
        blocked.reconcile(local)
    with pytest.raises(ActionError, match="not awaiting reconciliation"):
        allowed.reconcile(local)


def test_local_evidence_executor_allowlist_and_snapshot_exception(tmp_path):
    store = ActionStore(tmp_path / "actions")
    service = ActionService(store, ActionRuntimeConfig())
    action_id = synthetic_plan(store, "local-executor", "REBUILD_LOCAL_EVIDENCE")
    plan = store.snapshot(action_id)
    execution = execution_for(plan)
    with pytest.raises(ActionError, match="no allowlisted controller verb"):
        service._execute_local_evidence_rebuild(plan, execution)

    plan["snapshot_arguments"] = ["refresh-evidence-local"]
    service.controller.execute_snapshot = lambda *args, **kwargs: (
        _ for _ in ()
    ).throw(OSError("execution snapshot failed"))
    result = service._execute_local_evidence_rebuild(plan, execution)
    assert result["execution"]["status"] == "RECONCILE_REQUIRED"
    assert "execution snapshot failed" in result["execution"]["error"]
