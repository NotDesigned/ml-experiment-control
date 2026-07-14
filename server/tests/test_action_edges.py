"""Focused failure-path coverage for mutation safety boundaries."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from ml_exp_server.actions import service as actions
from ml_exp_server.actions.service import ActionError, ActionService
from ml_exp_server.actions.store import ActionStore
from ml_exp_server.schemas import (
    ActionRuntimeConfig, OperationScope, OperationScopeType, CampaignRef, CampaignRevision,
    ControllerConfig, ResearchProject,
)


def operation_intent(kind: str, payload: dict, idempotency_key: str = "intent-coverage") -> dict:
    return {
        "idempotency_key": idempotency_key, "kind": kind,
        "title": f"Prepare {kind}",
        "target": "target", "risk": "risk", "evidence_digest": "sha256:evidence",
        "draft": yaml.safe_dump(payload, sort_keys=False),
    }


def campaign_project(tmp_path: Path) -> tuple[ResearchProject, OperationScope]:
    revision = CampaignRevision(
        campaign="study", project="demo", revision_id="campaign.revision",
        file=str(tmp_path / "experiments" / "study.yml"), memberships=[],
    )
    project = ResearchProject(
        project="demo", title="Demo", run_roots=[], base_dir=tmp_path,
        campaigns=[CampaignRef(name="study", current_revision=revision)],
    )
    return project, OperationScope(project="demo", scope_type="campaign", object_id="study")


def test_action_campaign_lifecycle_records_cover_binding_and_immutability(tmp_path):
    project, operation_scope = campaign_project(tmp_path)
    service = ActionService(ActionStore(tmp_path / "actions"), ActionRuntimeConfig())
    archive_payload = {
        "project": "demo", "campaign": "study", "revision_id": "campaign.revision",
        "reason": "retired",
    }
    plan = service.prepare(operation_scope, project, operation_intent("ARCHIVE_CAMPAIGN", archive_payload))
    assert plan["ready"] is True and plan["operation"] == "WRITE_CAMPAIGN_ARCHIVE"

    Path(plan["target_path"]).parent.mkdir(parents=True)
    Path(plan["target_path"]).write_text("already: recorded\n")
    blocked = service.prepare(
        operation_scope, project,
        operation_intent("ARCHIVE_CAMPAIGN", archive_payload, "intent-blocked"),
    )
    assert blocked["ready"] is False
    assert {gate["name"]: gate["status"] for gate in blocked["gates"]} == {
        "exact_scope": "PASS", "record_absent": "FAIL", "archive_binding": "PASS",
    }


def test_explicit_idempotency_key_cannot_be_reused_for_different_intent(tmp_path):
    project, operation_scope = campaign_project(tmp_path)
    service = ActionService(ActionStore(tmp_path / "actions"), ActionRuntimeConfig())
    payload = {
        "project": "demo", "campaign": "study", "revision_id": "campaign.revision",
        "reason": "retired",
    }
    service.prepare(
        operation_scope, project,
        operation_intent("ARCHIVE_CAMPAIGN", payload, "intent-fixed"),
    )
    with pytest.raises(ActionError, match="already bound to a different"):
        service.prepare(
            operation_scope, project,
            operation_intent(
                "ARCHIVE_CAMPAIGN", {**payload, "reason": "different"}, "intent-fixed",
            ),
        )


@pytest.mark.parametrize(("payload", "scope", "message"), [
    ({"project": "other", "campaign": "study", "revision_id": "campaign.revision"},
     OperationScope(project="demo", scope_type="campaign", object_id="study"), "project does not match"),
    ({"project": "demo", "campaign": "study", "revision_id": "campaign.revision"},
     OperationScope(project="demo", scope_type="campaign", object_id="other"), "exact Campaign scope"),
    ({"project": "demo", "campaign": "study", "revision_id": "old"},
     OperationScope(project="demo", scope_type="campaign", object_id="study"), "current authored revision"),
])
def test_action_campaign_record_rejects_stale_or_wrong_binding(tmp_path, payload, scope, message):
    project, _ = campaign_project(tmp_path)
    service = ActionService(ActionStore(tmp_path / "actions"), ActionRuntimeConfig())
    with pytest.raises(ActionError, match=message):
        service.prepare(scope, project, operation_intent("ARCHIVE_CAMPAIGN", {**payload, "reason": "done"}))


@pytest.mark.parametrize(("kind", "payload", "scope", "message"), [
    ("ARCHIVE_RUN", {"project": "other", "run_id": "run-a"},
     OperationScope(project="demo", scope_type="run", object_id="run-a"), "project does not match"),
    ("ARCHIVE_RUN", {"project": "demo", "run_id": "run-a"},
     OperationScope(project="demo", scope_type="run", object_id="other"), "exact object scope"),
    ("ARCHIVE_ATTEMPT", {"project": "demo", "run_id": "bad/id", "attempt_id": "bad/id"},
     OperationScope(project="demo", scope_type="attempt", object_id="bad/id::bad/id"), "safe Run/Attempt"),
])
def test_action_archive_rejects_wrong_identity(tmp_path, kind, payload, scope, message):
    project = ResearchProject(project="demo", title="Demo", run_roots=[], base_dir=tmp_path)
    with pytest.raises(ActionError, match=message):
        ActionService(ActionStore(tmp_path / "actions"), ActionRuntimeConfig()).prepare(
            scope, project, operation_intent(kind, payload),
        )


def synthetic_plan(store: ActionStore, intent: str, *, operation: str = "SUBMIT_RUN") -> str:
    operation_scope = OperationScope(project="demo", scope_type="run", object_id="run-a")
    action_id = store.action_id(operation_scope, intent)
    expires = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    store.save_plan({
        "action_id": action_id, "scope": operation_scope.model_dump(mode="json"),
        "ready": True, "operation": operation, "intent_digest": "sha256:intent",
        "gate_bundle_digest": "sha256:gates", "gate_expires_at": expires,
        "command_preview": ["controller", "submit"], "cwd": ".",
        "verification_command_preview": ["controller", "status"],
        "verification_cwd": ".",
    })
    return action_id


def test_action_authorization_and_execution_fail_closed_on_policy_and_tampering(tmp_path):
    store = ActionStore(tmp_path / "actions")
    blank_actor = ActionService(store, ActionRuntimeConfig(), actor_provider=lambda: " ")
    action_id = synthetic_plan(store, "blank")
    with pytest.raises(ActionError, match="trusted actor"):
        blank_actor.authorize(action_id, "review")

    blocked = ActionService(store, ActionRuntimeConfig(), actor_provider=lambda: "tester")
    blocked.authorize(action_id, "review")
    with pytest.raises(ActionError, match="scheduler mutations are disabled"):
        blocked.execute(action_id, f"EXECUTE {action_id}")

    for intent, mutation, message in (
        ("expired", {"gate_expires_at": "2000-01-01T00:00:00Z"}, "expired"),
        ("intent", {"intent_digest": "sha256:changed"}, "immutable action intent"),
        ("gates", {"gate_bundle_digest": "sha256:changed"}, "gate bundle"),
    ):
        action_id = synthetic_plan(store, intent)
        service = ActionService(
            store, ActionRuntimeConfig(allow_scheduler_mutations=True), actor_provider=lambda: "tester",
        )
        service.authorize(action_id, "review")
        plan_path = store.directory(action_id) / "plan.json"
        payload = json.loads(plan_path.read_text())
        payload.update(mutation)
        plan_path.write_text(json.dumps(payload))
        with pytest.raises(ActionError, match=message):
            service.execute(action_id, f"EXECUTE {action_id}")

    action_id = synthetic_plan(store, "executing")
    store.set_execution(
        action_id, {**store.execution(action_id), "status": "EXECUTING"},
        event="test",
    )
    with pytest.raises(ActionError, match="reconcile"):
        blocked.execute(action_id, f"EXECUTE {action_id}")


@pytest.mark.parametrize(("result", "status"), [
    ({"timeout": True, "returncode": None, "stderr": "secret", "payload": None}, "RECONCILE_REQUIRED"),
    ({"timeout": False, "returncode": 2, "stderr": "failed", "payload": None}, "RECONCILE_REQUIRED"),
    ({"timeout": False, "returncode": 0, "stderr": "", "payload": []}, "RECONCILE_REQUIRED"),
    ({"timeout": False, "returncode": 0, "stderr": "", "payload": [{"backend_job_id": "job-1"}]}, "VERIFIED"),
])
def test_controller_execution_result_contract(tmp_path, result, status):
    store = ActionStore(tmp_path / "actions")
    action_id = synthetic_plan(store, status)
    plan = store.snapshot(action_id)
    execution = plan["execution"]
    service = ActionService(store, ActionRuntimeConfig(), runner=lambda *a, **k: result)
    assert service._execute_controller(plan, execution)["execution"]["status"] == status


def test_cancel_preparation_binds_live_backend_job_and_capability(tmp_path):
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    campaign = experiments / "study.yml"
    campaign.write_text("schema_version: 1\nproject: demo\ncampaign: study\nruns: [{run_id: run-a}]\n")
    project = ResearchProject(
        project="demo", title="Demo", run_roots=[], base_dir=tmp_path,
        controller=ControllerConfig(
            python="python", experimentctl="controller.py", workdir=".",
            capabilities={"cancel_outbox": True},
        ),
    )
    runner = lambda *a, **k: {
        "returncode": 0, "timeout": False, "stderr": "", "stdout": "",
        "payload": [{"backend_job_id": "job-7", "state": "RUNNING"}],
    }
    service = ActionService(ActionStore(tmp_path / "actions"), ActionRuntimeConfig(), runner=runner)
    operation_scope = OperationScope(project="demo", scope_type="attempt", object_id="run-a::attempt-001")
    payload = {
        "campaign_file": str(campaign), "run_id": "run-a", "attempt_id": "attempt-001",
        "backend_job_id": "job-7",
    }
    plan = service.prepare(operation_scope, project, operation_intent("CANCEL_RUN", payload))
    assert plan["ready"] is True
    assert {gate["name"]: gate["status"] for gate in plan["gates"]}[
        "backend_job_identity"
    ] == "PASS"

    project.controller.capabilities = {}
    blocked = service.prepare(
        operation_scope, project, operation_intent("CANCEL_RUN", {**payload, "backend_job_id": "wrong"}, "intent-wrong"),
    )
    statuses = {gate["name"]: gate["status"] for gate in blocked["gates"]}
    assert statuses["backend_job_identity"] == "FAIL"
    assert statuses["cancel_outbox_capability"] == "FAIL"


def test_cancel_timeout_can_reconcile_exact_terminal_job(tmp_path):
    store = ActionStore(tmp_path / "actions")
    operation_scope = OperationScope(
        project="demo", scope_type="attempt", object_id="run-a::attempt-001",
    )
    action_id = store.action_id(operation_scope, "cancel-timeout")
    store.save_plan({
        "action_id": action_id, "scope": operation_scope.model_dump(mode="json"),
        "ready": True, "operation": "CANCEL_RUN", "backend_job_id": "job-7",
        "verification_command_preview": ["controller", "status"],
        "verification_cwd": str(tmp_path), "request_digest": "sha256:cancel",
    })
    store.set_execution(
        action_id, {
            **store.execution(action_id),
            "status": "RECONCILE_REQUIRED", "result": {},
        }, event="timeout",
    )
    service = ActionService(
        store, ActionRuntimeConfig(),
        runner=lambda *a, **k: {
            "returncode": 0, "timeout": False,
            "payload": [{"backend_job_id": "job-7", "state": "CANCELLED"}],
        },
    )

    result = service.reconcile(action_id)

    assert result["execution"]["status"] == "VERIFIED"


def test_multi_write_detects_changed_target_and_rolls_back_partial_write(monkeypatch, tmp_path):
    store = ActionStore(tmp_path / "actions")
    service = ActionService(store, ActionRuntimeConfig())
    action_id = synthetic_plan(store, "multi", operation="WRITE_CAMPAIGN")
    target = tmp_path / "new.yml"
    plan = {
        **store.snapshot(action_id),
        "files": [{"path": str(target), "expected_sha256": "sha256:wrong", "content": "new: true\n"}],
    }
    result = service._execute_multi_write(plan, plan["execution"])
    assert result["execution"]["status"] == "FAILED"
    assert "targets changed" in result["execution"]["error"]

    action_id = synthetic_plan(store, "rollback", operation="WRITE_CAMPAIGN")
    plan = {
        **store.snapshot(action_id),
        "files": [{"path": str(target), "expected_sha256": None, "content": "new: true\n"}],
    }
    real_replace = actions.os.replace
    calls = 0

    def fail_first(source, destination):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("simulated atomic replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(actions.os, "replace", fail_first)
    result = service._execute_multi_write(plan, plan["execution"])
    assert result["execution"]["status"] == "FAILED"
    assert "rolled back" in result["execution"]["error"]
    assert not target.exists()


def test_action_helper_and_prepare_error_branches(tmp_path):
    assert actions._parse_mapping("```yaml\nwhen: 2026-07-13\n```") == {"when": "2026-07-13"}
    assert ActionService._gpu_hours({"gpus": 2}, {"time": "1.5h"}) == 3.0
    assert ActionService._gpu_hours({"gpus": 2}, {"time": "bad"}) is None
    assert ActionService._gpu_hours({}, {}) is None
    assert ActionService._gpu_hours([], {}) is None
    store = ActionStore(tmp_path / "actions")
    project = ResearchProject(project="demo", title="Demo", run_roots=[], base_dir=tmp_path)
    service = ActionService(store, ActionRuntimeConfig())
    project_scope = OperationScope(project="demo", scope_type="project", object_id="demo")
    with pytest.raises(ActionError, match="invalid operation intent"):
        service.prepare(project_scope, project, {"status": "PENDING"})
    with pytest.raises(ActionError, match="invalid operation intent"):
        service.prepare(project_scope, project, operation_intent("ANALYSIS_ONLY", {}))
    with pytest.raises(ActionError, match="safe file identity"):
        service.prepare(project_scope, project, operation_intent("CREATE_RESEARCH_QUESTION_DRAFT", {"id": "bad/id"}))
    with pytest.raises(ActionError, match="no research_questions_dir"):
        service.prepare(project_scope, project, operation_intent("CREATE_RESEARCH_QUESTION_DRAFT", {"id": "Q1"}))
    for payload, message in (
        ({"campaign": "bad/id", "project": "demo", "run_refs": [{}]}, "safe campaign"),
        ({"campaign": "study", "project": "other", "run_refs": [{}]}, "project does not match"),
        ({"campaign": "study", "project": "demo", "runs": []}, "non-empty"),
    ):
        with pytest.raises(ActionError, match=message):
            service.prepare(project_scope, project, operation_intent("CREATE_CAMPAIGN_DRAFT", payload, "intent-" + message[:4]))
