"""Integration checks for immutable submissions, async ownership, and Run derivation."""

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import threading
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
import yaml

from ml_exp_server.actions.service import ActionService, _missing_frozen_match_fields
from ml_exp_server.actions.store import ActionStore
from ml_exp_server.api.app import _shutdown, create_app
from ml_exp_server.application import ApplicationError, ExperimentServerApplication
from ml_exp_server.container_execution import RunRequest
from ml_exp_server.project_service import ProjectApplicationService
from ml_exp_server.schemas import ActionRuntimeConfig, ProjectLifecycleState
from ml_exp_server.submissions import ExperimentSubmissionService, _prepared_matches_current_authored_state
from ml_exp_server import storage as storage_module
from tests.test_app_edges import config
from tests.test_image_builder_boundary import builder
from tests.test_submissions import _app


@pytest.mark.parametrize("executor_enabled", [False, True])
def test_shutdown_retains_runtime_and_lease_until_failed_action_finishes(executor_enabled):
    app = FastAPI()
    closed, released = threading.Event(), threading.Event()
    pending = Future()
    app.state.runtime = SimpleNamespace(close=closed.set)
    app.state.collector_lease = SimpleNamespace(release=released.set)
    app.state._action_futures = {pending}
    app.state._action_futures_lock = threading.Lock()
    if executor_enabled:
        app.state.action_executor = ThreadPoolExecutor(max_workers=1)
    asyncio.run(_shutdown(app))
    assert not closed.is_set() and not released.is_set()
    pending.set_exception(RuntimeError("durably recorded worker failure"))
    assert closed.wait(2) and released.wait(2)


def test_startup_failure_without_collector_ownership_releases_no_foreign_lease(tmp_path, monkeypatch):
    from ml_exp_server.api import app as module
    class Lease:
        def __init__(self, path):
            pass
        def acquire(self):
            return False
        def release(self):
            pytest.fail("must not release another daemon's lease")
    monkeypatch.setattr(module, "CollectorLease", Lease)
    def failure(*args, **kwargs):
        raise RuntimeError("startup validation failed")
    monkeypatch.setattr(module.ExperimentServerRuntime, "create", failure)
    app = create_app(config(tmp_path), poll=False)
    with pytest.raises(RuntimeError, match="another ml-expd process owns"):
        with TestClient(app):
            pass


@pytest.mark.parametrize("parameters", [
    {"profiles": "h100,h200"}, {"config_overrides": "["},
    {"config_overrides": "{}"}, {"config_overrides": "[1]"},
    {"new_run_id": "run-a"}, {"new_run_id": "../run"},
])
def test_clone_rejects_ambiguous_or_invalid_input_before_writing(tmp_path, parameters):
    app, _ = _app(tmp_path)
    with TestClient(app) as client:
        result = client.post("/api/operations/direct", json={"project": "demo", "scope_type": "run",
            "object_id": "run-a", "operation_id": "run.clone",
            "parameters": {"new_run_id": "derived", **parameters}})
        assert result.status_code == 409
        assert result.headers["X-ML-Expd-Error-Code"] == "INVALID_OPERATION"
        assert client.app.state.runtime.action_store.list_all() == []


@pytest.mark.parametrize("fault,code", [
    ("ambiguous", "RUN_NOT_CLONEABLE"), ("revision", "CAMPAIGN_REVISION_MISSING"),
    ("missing", "CAMPAIGN_FILE_MISSING"), ("invalid-yaml", "CAMPAIGN_FILE_MISSING"),
    ("no-runs", "RUN_NOT_CLONEABLE"), ("removed-run", "RUN_NOT_CLONEABLE"),
])
def test_clone_detects_authored_state_disappearing_after_selection(tmp_path, monkeypatch, fault, code):
    app, _ = _app(tmp_path)
    with TestClient(app) as client:
        application = client.app.state.application
        scope, project, row = application.resolve_scope("demo", "run", "run-a")
        monkeypatch.setattr(application, "_require_operation_available", lambda *a: None)
        monkeypatch.setattr(application, "resolve_scope", lambda *a: (scope, project, row))
        revision = project.campaigns[0].current_revision
        path = Path(revision.file)
        if fault == "ambiguous":
            row.campaign_memberships = []
        elif fault == "revision":
            project.campaigns[0].current_revision = None
        elif fault == "missing":
            path.unlink()
        elif fault == "invalid-yaml":
            path.write_text("[")
        elif fault == "no-runs":
            path.write_text("[]")
        else:
            path.write_text("runs: [null, {run_id: other}]")
        with pytest.raises(ApplicationError) as error:
            application.prepare_run_clone("demo", "run-a", new_run_id="derived")
        assert error.value.code == code
        assert client.app.state.runtime.action_store.list_all() == []


def test_clone_relative_revision_preserves_unspecified_fields_and_replaces_overrides(tmp_path, monkeypatch):
    app, _ = _app(tmp_path)
    with TestClient(app) as client:
        application = client.app.state.application
        scope, project, row = application.resolve_scope("demo", "run", "run-a")
        revision = project.campaigns[0].current_revision
        path = Path(revision.file)
        payload = yaml.safe_load(path.read_text())
        payload["runs"][0]["config_overrides"] = ["batch_size=32", "seed=1"]
        payload["runs"].insert(0, None)
        path.write_text(yaml.safe_dump(payload))
        revision.file = str(path.relative_to(project.base_dir))
        monkeypatch.setattr(application, "_require_operation_available", lambda *a: None)
        monkeypatch.setattr(application, "resolve_scope", lambda *a: (scope, project, row))
        captured = []
        monkeypatch.setattr(application, "_prepare_action_intent", lambda *args: captured.append(args[-1]) or {})
        result = application.prepare_run_clone("demo", "run-a", new_run_id="derived", config_overrides='["seed=2"]')
        assert result["derived_run_ids"] == ["derived"]
        clone = yaml.safe_load(captured[0]["draft"])["runs"][-1]
        assert clone["config_overrides"] == ["batch_size=32", "seed=2"]
        assert clone["research_role"] == "candidate"


@pytest.mark.parametrize("error", [FileNotFoundError("missing"), RuntimeError("blocked")])
def test_begin_background_action_maps_claim_errors(error):
    def fail(*args):
        raise error
    application = ExperimentServerApplication(SimpleNamespace(action_service=SimpleNamespace(begin_execute=fail)))
    with pytest.raises(ApplicationError) as result:
        application.begin_action_execution("action-missing", "EXECUTE missing")
    assert result.value.code == ("UNKNOWN_ACTION" if isinstance(error, FileNotFoundError) else "ACTION_BLOCKED")


@pytest.mark.parametrize("failure", [False, True])
def test_background_non_scheduler_failure_preserves_uncertainty_without_refresh(tmp_path, failure):
    from tests.test_action_service_coverage import synthetic_plan
    store = ActionStore(tmp_path / "actions")
    action_id = synthetic_plan(store, "background", operation="REBUILD_LOCAL_EVIDENCE")
    plan = store.snapshot(action_id)
    store.set_execution(action_id, {**plan["execution"], "status": "EXECUTING"}, event="claimed")
    def finish(pending):
        if failure:
            raise OSError("disk offline")
        return {"execution": {"status": "BLOCKED"}}
    application = ExperimentServerApplication(SimpleNamespace(action_service=SimpleNamespace(finish_execute=finish), action_store=store))
    application._refresh_action_project = lambda action: pytest.fail("no terminal success to index")
    result = application.finish_action_execution(SimpleNamespace(plan=plan))
    assert result["execution"]["status"] == ("RECONCILE_REQUIRED" if failure else "BLOCKED")
    if failure:
        assert result["execution"]["safe_to_retry"] is False
        assert "disk offline" in result["execution"]["error"]


def test_prepared_reuse_requires_same_campaign_and_valid_summary():
    revision = SimpleNamespace(revision_id="campaign.new")
    assert _prepared_matches_current_authored_state({"status": "VERIFIED", "git_commit": "old"}, revision, {})
    assert not _prepared_matches_current_authored_state({"status": "PREPARED", "preflight_summary": {"campaign_id": "campaign.old"}}, revision, {})
    assert _prepared_matches_current_authored_state({"status": "PREPARED", "preflight_summary": None}, revision, {})
    view = ExperimentSubmissionService._view({"execution": {"status": "PREPARED"}, "semantic_changes": [None, {"path": "git_commit", "after": "a" * 40}]})
    assert view["git_commit"] == "a" * 40


def test_submission_sync_facade_preserves_verification_result():
    service = ExperimentSubmissionService(SimpleNamespace(execute_action=lambda *a: {"execution": {"status": "VERIFIED"}}), None)
    service._snapshot = lambda *a: {}
    assert service.execute("action-a", "EXECUTE action-a")["status"] == "VERIFIED"


def test_invalid_resource_policy_and_empty_outputs_are_explicit():
    with pytest.raises(ValueError, match="requires max_gpu_hours"):
        ActionRuntimeConfig(max_gpu_hours_per_action=None)
    definition = RunRequest(run_id="run-a", executor="wyd", runtime_id="runtime." + "a" * 64, outputs=[])
    assert definition.outputs == []
    assert RunRequest(run_id="run-a", executor="wyd", runtime_id="runtime." + "a" * 64, outputs=["checkpoints/*.pt", "metrics.json"]).outputs == ["checkpoints/*.pt", "metrics.json"]


def test_unknown_or_illegal_project_lifecycle_cannot_mutate_registry():
    service = ProjectApplicationService(SimpleNamespace(project_records=lambda: []))
    with pytest.raises(ApplicationError, match="unknown registered"):
        service.transition("missing", "pause", ProjectLifecycleState.PAUSED)
    service.runtime.project_records = lambda: [SimpleNamespace(project="demo", state=ProjectLifecycleState.ARCHIVED)]
    with pytest.raises(ApplicationError, match="cannot pause"):
        service.transition("demo", "pause", ProjectLifecycleState.PAUSED)


def test_atomic_text_closes_descriptor_and_keeps_previous_file_on_stream_failure(tmp_path, monkeypatch):
    path = tmp_path / "state.txt"
    path.write_text("previous")
    real_close = storage_module.os.close
    closed = []
    def close(fd):
        closed.append(fd)
        return real_close(fd)
    def fail(*args, **kwargs):
        raise OSError("cannot open stream")
    monkeypatch.setattr(storage_module.os, "fdopen", fail)
    monkeypatch.setattr(storage_module.os, "close", close)
    with pytest.raises(OSError, match="cannot open stream"):
        storage_module.atomic_text(path, "new")
    assert len(closed) == 1 and path.read_text() == "previous"
    assert list(tmp_path.iterdir()) == [path]


def test_publisher_rejects_missing_remote_digest(builder, tmp_path):
    value, _, _ = builder
    value._docker = lambda *args: None
    def skopeo(args, **kwargs):
        if args[0] == "copy":
            Path(args[args.index("--digestfile") + 1]).write_text("not-a-digest")
        return '{}'
    value._skopeo = skopeo
    with pytest.raises(ValueError, match="digest is unavailable"):
        value._publish("registry.example/runtime:tag", tmp_path)


def test_frozen_comparison_ignores_empty_nonstring_fields():
    assert _missing_frozen_match_fields({"research_contract": {"comparison": {"match_fields": [None, "", "source_id"]}}}) == ["source_id"]


def test_expired_preparation_is_replaced_without_reusing_authorization(tmp_path):
    from tests.test_action_service_coverage import synthetic_plan
    app, _ = _app(tmp_path)
    with TestClient(app) as client:
        expired = synthetic_plan(client.app.state.runtime.action_store, "expired", run_id="run-a",
            attempt_id="attempt-001", gate_expires_at="2000-01-01T00:00:00Z")
        response = client.post("/api/experiments/demo/run-a/submissions/prepare", json={"max_gpu_hours": 2})
        assert response.status_code == 200, response.text
        assert response.json()["submission_id"] != expired
        assert response.json()["reused"] is False
        assert client.get(f"/api/actions/{expired}").json()["execution"]["status"] == "PREPARED"


@pytest.mark.parametrize("container", [False, True])
def test_retry_without_git_commit_requires_container_source_identity(tmp_path, container):
    from ml_exp_server.actions.errors import ActionError
    from ml_exp_server.schemas import OperationScope
    from tests.test_actions import controller_project, FakeController, operation_intent
    project, campaign = controller_project(tmp_path)
    project.controller.capabilities["authored_campaign_revision"] = True
    project.controller.capabilities["container_execution"] = container
    canonical = project.base_dir / "outputs/runs/demo-campaign/run-a/manifest.yaml"
    canonical.parent.mkdir(parents=True)
    canonical.write_text(yaml.safe_dump({"project": "demo", "campaign": "demo-campaign", "run_id": "run-a", "campaign_id": "campaign." + "a" * 64}))
    service = ActionService(ActionStore(tmp_path / "actions"), ActionRuntimeConfig(), FakeController())
    scope = OperationScope(project="demo", scope_type="attempt", object_id="run-a::attempt-001")
    intent = operation_intent("RETRY_ATTEMPT", yaml.safe_dump({"campaign_file": str(campaign), "run_id": "run-a", "source_attempt_id": "attempt-001", "attempt_id": "attempt-002", "max_gpu_hours": 8}))
    if not container:
        with pytest.raises(ActionError, match="no valid immutable git_commit"):
            service.prepare(scope, project, intent)
    else:
        plan = service.prepare(scope, project, intent)
        assert "git_commit" not in yaml.safe_load(Path(plan["execution_campaign_file"]).read_text())
