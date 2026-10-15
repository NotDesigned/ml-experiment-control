"""Deletion fences cover journal repair, Action claims, and late HTTP writers."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import yaml

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.deletion_state import RunFence
from ml_exp_server.runs.deletion_guards import require_action_runs
from ml_exp_server.runs.experiment_preparation import ExperimentPreparationService
from ml_exp_server.storage import DurableJsonState
from tests.test_run_deletion_api import deletion_api
from tests.test_run_deletion_service import deletion_env

PREPARATION = "preparation-" + "a" * 32


def preparation(runtime, run="run-a", identity=PREPARATION):
    service = ExperimentPreparationService.__new__(ExperimentPreparationService)
    service.runtime = runtime
    service.root = runtime.config.project_registry_root_path() / "experiment-preparations"
    service.containers = SimpleNamespace(require_enabled=lambda: None)
    value = {"project": "demo", "run_id": run, "phase": "ACCEPTED", "status": "EXECUTING",
             "tracking": {"enabled": False}, "matching": {"profile": {}}, "updated_at": "2026-10-15T00:00:00Z"}
    with service.state("demo", identity) as (store, snapshot):
        store.commit(value, expected_revision=snapshot.revision, event={"event": "isolated-preparation"})
    return service


def busy(root):
    with pytest.raises(ApplicationError) as error:
        with RunFence(root).exclusive():
            pytest.fail("delete crossed an active writer")
    assert error.value.code == "RUN_DELETION_BUSY"


def action(runtime, number=1, *, operation="CANCEL_RUN", run="run-a", resume_run=None):
    value = {"action_id": "action-" + f"{number:016x}", "scope": {"project": "demo", "scope_type": "run", "object_id": run},
             "run_id": run, "operation": operation, "ready": True, "intent_digest": "sha256:reviewed",
             "gate_bundle_digest": "sha256:reviewed-gates", "request_digest": "sha256:request",
             "gate_expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()}
    if operation == "WRITE_CAMPAIGN":
        value["scope"] = {"project": "demo", "scope_type": "campaign", "object_id": "candidate"}
        value.pop("run_id")
        definition = {"run_id": run}
        if resume_run is not None:
            definition["resume_from"] = {"run_id": resume_run}
        value["proposed_content"] = yaml.safe_dump({"runs": [definition], "run_refs": [{"run_id": "other"}]})
    return runtime.action_store.save_plan(value)


def test_preparation_shared_fence_precedes_snapshot_repair_and_covers_commit(deletion_env, monkeypatch):
    env = deletion_env
    service = preparation(env.runtime)
    snapshot = DurableJsonState.snapshot
    repair = DurableJsonState.repair_journal
    def guarded_snapshot(self, *args):
        busy(env.root)
        return snapshot(self, *args)
    def guarded_repair(self, *args):
        busy(env.root)
        return repair(self, *args)
    monkeypatch.setattr(DurableJsonState, "snapshot", guarded_snapshot)
    monkeypatch.setattr(DurableJsonState, "repair_journal", guarded_repair)
    with service.state("demo", PREPARATION) as (store, current):
        busy(env.root)
        store.commit({**current.value, "phase": "READY"}, expected_revision=current.revision,
                     event={"event": "isolated-state-transition"})
    assert service.update("demo", PREPARATION, status="READY")["status"] == "READY"


def test_deleted_preparation_is_checked_before_journal_repair_and_missing_update_is_404(deletion_env, monkeypatch):
    env = deletion_env
    service = preparation(env.runtime)
    path = service.root / "demo" / (PREPARATION + ".json")
    journal = path.with_suffix(".jsonl")
    original = path.read_bytes()
    journal.unlink()
    RunFence(env.root).mark("demo", "run-a")
    repair = DurableJsonState.repair_journal
    monkeypatch.setattr(DurableJsonState, "repair_journal", lambda *args: pytest.fail("deleted preparation repaired its journal"))
    for call in [lambda: service.read("demo", PREPARATION),
                 lambda: service.update("demo", PREPARATION, status="READY"),
                 lambda: service.continue_preparation("demo", PREPARATION)]:
        with pytest.raises(ApplicationError) as error:
            call()
        assert error.value.code == "RUN_DELETED"
    assert path.read_bytes() == original and not journal.exists()
    monkeypatch.setattr(DurableJsonState, "repair_journal", repair)
    path.unlink()
    with pytest.raises(ApplicationError) as error:
        service.update("demo", PREPARATION, phase="READY", status="READY")
    assert (error.value.status_code, error.value.code) == (404, "UNKNOWN_PREPARATION")
    assert not path.exists() and not journal.exists()


def test_startup_preparation_recovery_skips_deleted_run_without_repair(deletion_env):
    env = deletion_env
    service = preparation(env.runtime)
    preparation(env.runtime, run="other", identity="preparation-" + "b" * 32)
    path = service.root / "demo" / (PREPARATION + ".json")
    original = path.read_bytes()
    RunFence(env.root).mark("demo", "run-a")
    assert service.recover_interrupted() == 1
    assert path.read_bytes() == original


def test_old_preparation_callback_requires_executing_state_without_replay(deletion_env, monkeypatch):
    env = deletion_env
    service = preparation(env.runtime)
    monkeypatch.setattr(service, "check_profile", lambda *args: pytest.fail("completed or uncertain preparation replayed"))
    for status in ("READY", "RECONCILE_REQUIRED"):
        service.update("demo", PREPARATION, status=status)
        assert service.finish(("demo", PREPARATION))["status"] == status
    path = service.root / "demo" / (PREPARATION + ".json")
    path.unlink(); path.with_suffix(".jsonl").unlink()
    with pytest.raises(ApplicationError) as error:
        service.finish(("demo", PREPARATION))
    assert (error.value.status_code, error.value.code) == (404, "UNKNOWN_PREPARATION")


@pytest.mark.parametrize("operation", ["CANCEL_RUN", "WRITE_CAMPAIGN", "WRITE_CAMPAIGN_RESUME"])
def test_action_claims_hold_fence_and_late_callbacks_cannot_execute(deletion_env, monkeypatch, operation):
    env = deletion_env
    service = env.runtime.action_service
    service.config.allow_scheduler_mutations = True
    service.config.allow_project_writes = True
    if operation == "WRITE_CAMPAIGN_RESUME":
        plan = action(env.runtime, operation="WRITE_CAMPAIGN", run="consumer", resume_run="run-a")
    else:
        plan = action(env.runtime, operation=operation)
    transition, claim = service.store.set_execution, service.store.begin_execution
    def guarded_transition(*args, **kwargs):
        busy(env.root)
        return transition(*args, **kwargs)
    def guarded_claim(*args, **kwargs):
        busy(env.root)
        return claim(*args, **kwargs)
    monkeypatch.setattr(service.store, "set_execution", guarded_transition)
    monkeypatch.setattr(service.store, "begin_execution", guarded_claim)
    service.authorize(plan["action_id"], "isolated review")
    _, pending = service.begin_execute(plan["action_id"], "EXECUTE " + plan["action_id"])
    calls = []
    def finish(pending):
        busy(env.root)
        calls.append(pending)
        return {"isolated": True}
    monkeypatch.setattr(service, "_finish_execute_live", finish)
    assert service.finish_execute(pending) == {"isolated": True}
    RunFence(env.root).mark("demo", "run-a")
    for callback in [lambda: service.authorize(plan["action_id"], "late"),
                     lambda: service.begin_execute(plan["action_id"], "EXECUTE " + plan["action_id"]),
                     lambda: service.finish_execute(pending), lambda: service.reconcile(plan["action_id"])]:
        with pytest.raises(ApplicationError) as error:
            callback()
        assert error.value.code == "RUN_DELETED"
    assert calls == [pending]


@pytest.mark.parametrize("runs", [[{"run_id": "run-a"}], [{"run_id": "consumer", "resume_from": {"run_id": "run-a"}}]])
def test_campaign_plan_refuses_deleted_reused_identity_before_preview_write(deletion_env, runs):
    env = deletion_env
    from ml_exp_server.schemas import OperationScope
    scope = OperationScope(project="demo", scope_type="project", object_id="demo")
    RunFence(env.root).mark("demo", "run-a")
    intent = {"kind": "CREATE_CAMPAIGN_DRAFT", "title": "late reuse", "target": "candidate", "risk": "review",
              "evidence_digest": "sha256:review", "idempotency_key": "late-reuse",
              "draft": yaml.safe_dump({"schema_version": 1, "project": "demo", "campaign": "candidate",
                                        "runs": runs, "budget": {"gpu_hours": 1}})}
    with pytest.raises(ApplicationError) as error:
        env.runtime.action_service.prepare(scope, env.runtime.project("demo"), intent)
    assert error.value.code == "RUN_DELETED"
    assert not (env.root / "managed/demo/experiments/campaigns/candidate.yml").exists()


def test_late_worker_and_action_http_calls_keep_410_and_preparation_absence_404(deletion_api):
    api, env = deletion_api
    service = api.app.state.preparation_service
    preparation(api.app.state.runtime)
    objects = api.app.state.runtime.storage_service.objects
    _, token, _ = objects.issue("demo", "run-a", "attempt-001", env.local["run-a"], ["**/*"], record_stream=True)
    first = action(api.app.state.runtime, 1)
    second = action(api.app.state.runtime, 2)
    api.app.state.runtime.action_service.config.allow_scheduler_mutations = True
    api.app.state.application.authorize_action(second["action_id"], "isolated review")
    RunFence(env.root).mark("demo", "run-a")
    worker = {"Authorization": "Bearer " + token}
    requests = [
        lambda: api.get("/api/asset-transfers/demo/run-a/attempt-001/asset." + "a" * 64, headers=worker),
        lambda: api.put("/api/snapshot-transfers/demo/run-a/attempt-001", content=b"late", headers=worker),
        lambda: api.put("/api/checkpoint-transfers/demo/run-a/attempt-001", json={}, headers=worker),
        lambda: api.post("/api/attempt-uploads/demo/run-a/attempt-001/artifacts", json={"sha256": "a" * 64, "bytes": 10}, headers=worker),
        lambda: api.post("/api/record-transfers/demo/run-a/attempt-001", json={"records": []}, headers=worker),
        lambda: api.post("/api/actions/authorize", json={"action_id": first["action_id"]}),
        lambda: api.post("/api/actions/execute", json={"action_id": second["action_id"], "confirmation": "EXECUTE " + second["action_id"]}),
        lambda: api.post("/api/actions/reconcile", json={"action_id": first["action_id"]}),
        lambda: api.post("/api/experiments/demo/run-a/submissions/prepare", json={"max_gpu_hours": 1}),
        lambda: api.get("/api/projects/demo/experiment-preparations/" + PREPARATION),
        lambda: api.post("/api/projects/demo/experiment-preparations/" + PREPARATION + "/continue"),
    ]
    for request in requests:
        response = request()
        assert response.status_code == 410, response.text
        assert response.headers["X-ML-Expd-Error-Code"] == "RUN_DELETED"
    path = service.root / "demo" / (PREPARATION + ".json")
    path.unlink(); path.with_suffix(".jsonl").unlink()
    assert api.get("/api/projects/demo/experiment-preparations/" + PREPARATION).status_code == 404
    assert api.post("/api/projects/demo/experiment-preparations/" + PREPARATION + "/continue").status_code == 404


def test_record_http_race_after_authorization_keeps_410_without_orphan_events(deletion_api, monkeypatch):
    from ml_exp_server.records.store import RecordStore
    api, env = deletion_api
    objects = api.app.state.runtime.storage_service.objects
    _, token, _ = objects.issue("demo", "run-a", "attempt-001", env.local["run-a"], ["**/*"], record_stream=True)
    original = RecordStore.scope
    def deleted_before_scope(self, *identity):
        RunFence(env.root).mark("demo", "run-a")
        return original(self, *identity)
    monkeypatch.setattr(RecordStore, "scope", deleted_before_scope)
    body = {"records": [{"event_id": "late-event", "kind": "lifecycle", "data": {"state": "SUCCEEDED"}}]}
    response = api.post("/api/record-transfers/demo/run-a/attempt-001", json=body,
                        headers={"Authorization": "Bearer " + token})
    assert response.status_code == 410 and response.headers["X-ML-Expd-Error-Code"] == "RUN_DELETED"
    with env.records.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM events WHERE event_id='late-event'").fetchone()[0] == 0


def test_generic_campaign_prepare_http_preserves_deleted_run_error(deletion_api):
    from ml_exp_server.runs.failures import evidence_digest
    api, env = deletion_api
    RunFence(env.root).mark("demo", "run-a")
    application = api.app.state.application
    scope, project, resolved = application.resolve_scope("demo", "project", "demo")
    intent = {"kind": "CREATE_CAMPAIGN_DRAFT", "title": "late generic campaign", "target": "candidate",
              "risk": "review", "idempotency_key": "http-late-campaign",
              "evidence_digest": evidence_digest(application.bounded_evidence(scope, project, resolved)),
              "draft": yaml.safe_dump({"schema_version": 1, "project": "demo", "campaign": "candidate",
                                        "runs": [{"run_id": "run-a"}], "budget": {"gpu_hours": 1}})}
    response = api.post("/api/actions/prepare", json={"project": "demo", "scope_type": "project", "object_id": "demo", "intent": intent})
    assert response.status_code == 410 and response.headers["X-ML-Expd-Error-Code"] == "RUN_DELETED"
    assert not (env.root / "managed/demo/experiments/campaigns/candidate.yml").exists()


def test_startup_campaign_write_cannot_restore_deleted_run_or_hide_other_failure(deletion_env, monkeypatch):
    env = deletion_env
    service = env.runtime.action_service
    service.config.allow_project_writes = True
    plan = action(env.runtime, operation="WRITE_CAMPAIGN")
    execution = dict(plan["execution"], status="RECONCILE_REQUIRED")
    service.store.set_execution(plan["action_id"], execution, event="isolated-interruption")
    RunFence(env.root).mark("demo", "run-a")
    monkeypatch.setattr(service, "_execute_write", lambda *args: pytest.fail("startup recreated deleted Run definitions"))
    recovered = service.recover_pending_project_writes()
    assert len(recovered) == 1 and recovered[0]["execution"]["error"] == "RUN_DELETED"
    require_action_runs(SimpleNamespace(), "demo", ["run-a"])
    require_action_runs(service, "demo", [])
    failure = ApplicationError("independent fault", code="OTHER_FAULT")
    from contextlib import contextmanager
    @contextmanager
    def broken_guard(*args):
        raise failure
        yield
    monkeypatch.setattr("ml_exp_server.runs.deletion_guards.action_guard", broken_guard)
    with pytest.raises(ApplicationError) as error:
        service.recover_pending_project_writes()
    assert error.value is failure


def test_preparation_recovery_keeps_unrelated_failure(deletion_env, monkeypatch):
    env = deletion_env
    service = preparation(env.runtime)
    failure = ApplicationError("independent fault", code="OTHER_FAULT")
    from contextlib import contextmanager
    @contextmanager
    def broken_state(*args):
        raise failure
        yield
    monkeypatch.setattr(service, "state", broken_state)
    with pytest.raises(ApplicationError) as error:
        service.recover_interrupted()
    assert error.value is failure
