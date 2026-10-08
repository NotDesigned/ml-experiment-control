"""Read-only staging diagnostics, including historical command redaction."""
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from experiment_control.backends.wyd import WydSlurmBackend
from ml_exp_server.api.app import create_app
from ml_exp_server.actions.store import ActionStore
from ml_exp_server.schemas import ServerConfig
from tests.test_action_edges import synthetic_plan
from tests.test_container_api import client, import_source, runtime


def _managed_action(api, *, executor="gpu", revision=False, embedded=False):
    source = import_source(api)
    bundle = runtime(api, source)
    response = api.post("/api/projects/demo/runs", json={
        "run_id": "run-a", "runtime_id": bundle["runtime_id"], "executor": executor,
    })
    assert response.status_code == 200, response.text
    daemon = api.app.state.runtime
    project = daemon.project("demo")
    original = Path(project.campaigns[0].current_revision.file)
    campaign = yaml.safe_load(original.read_text())
    campaign["local_root"] = str(daemon.config.project_run_root_path("demo"))
    store = daemon.action_store
    action_id = synthetic_plan(store, "diagnostics")
    path = store.directory(action_id) / "campaign.execution.yml"
    path.write_text(yaml.safe_dump(campaign))
    extra = ([] if embedded else ["--source-root", str(daemon.config.project_registry_root_path() / "source-revisions" / "sources" / "demo" / source["source_id"] / "tree"), "--source-id", source["source_id"]])
    if revision:
        extra += ["--campaign-id", "campaign.frozen"]
    attempt_id = "attempt-001" if embedded else None
    call = daemon.action_service.controller.build(project, path, "stage", "run-a", attempt_id=attempt_id, extra=extra)
    plan_path = store.directory(action_id) / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan.update(execution_campaign_file=str(path), execution_campaign_sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                expected_source_id="" if embedded else source["source_id"], run_id="run-a", attempt_id=attempt_id,
                stage_command_preview=call.argv, stage_cwd=str(call.cwd))
    if revision:
        plan["campaign_revision"] = "campaign.frozen"
    plan_path.write_text(json.dumps(plan))
    store.set_execution(action_id, {
        **store.execution(action_id), "status": "FAILED",
        "started_at": "2026-10-08T12:00:00Z", "finished_at": "2026-10-08T12:20:00Z",
        "resolution": "FAILED_BEFORE_SUBMISSION", "safe_to_retry": True,
        "next_action": "PREPARE_NEW_ACTION",
        "result": {"stage_command": {"timeout": True, "returncode": None,
                    "stdout": "image conversion began", "stderr": str(subprocess.TimeoutExpired(["private-controller", "private-capsule"], 1200))}},
    }, event="synthetic_staging_failure")
    return action_id, path, campaign


def test_saved_diagnostics_redact_legacy_timeout_and_preserve_failed_history(client, monkeypatch):
    action_id, _, _ = _managed_action(client)
    store = client.app.state.runtime.action_store
    before = store.snapshot(action_id)
    monkeypatch.setattr(WydSlurmBackend, "stage_progress", lambda *_args: pytest.fail("default GET must not query the backend"))
    response = client.get(f"/api/actions/{action_id}/diagnostics")
    assert response.status_code == 200, response.text
    value = response.json()
    assert value["status"] == "FAILED" and value["remote"]["status"] == "NOT_REQUESTED"
    assert value["stage_diagnostic"]["phase"] == "UNKNOWN"
    assert value["stage_command"]["returncode"] is None
    assert value["stage_command"]["stdout"] == "image conversion began"
    assert "private-controller" not in response.text and "private-capsule" not in response.text
    assert store.snapshot(action_id) == before
    assert client.get("/api/actions/action-deadbeefdeadbeef/diagnostics").status_code == 404
    assert client.get("/api/actions/invalid/diagnostics").status_code == 404


@pytest.mark.parametrize("after", [False, True])
def test_refresh_observes_ready_metadata_without_rewriting_failure(client, monkeypatch, after):
    action_id, _, _ = _managed_action(client, revision=after)
    store = client.app.state.runtime.action_store
    before = store.snapshot(action_id)
    event = {"phase": "SIF_PUBLISH", "status": "READY", "timestamp": "2026-10-08T12:21:00Z" if after else "2026-10-08T12:19:00Z"}
    monkeypatch.setattr(WydSlurmBackend, "stage_progress", lambda _self, _run: {**event, "argv": "private"})
    response = client.get(f"/api/actions/{action_id}/diagnostics?refresh=true")
    assert response.status_code == 200, response.text
    remote = response.json()["remote"]
    assert remote["status"] == "OBSERVED" and remote["event"] == event
    assert remote["observed_after_action"] is after
    assert remote["image_ready_reported"] is True and remote["cache_reverified"] is False
    assert remote["historical_failure_cause"] is False and remote["next_action"] == "PREPARE_NEW_ACTION"
    assert response.json()["status"] == "FAILED" and response.json()["stage_diagnostic"]["phase"] == "UNKNOWN"
    assert store.snapshot(action_id) == before


def test_refresh_reports_in_progress_and_stale_or_missing_evidence(client, monkeypatch):
    action_id, _, _ = _managed_action(client)
    for event, status in (
        ({"phase": "IMAGE_PULL_CONVERT", "status": "START", "timestamp": "2026-10-08T12:21:00Z"}, "OBSERVED"),
        ({"phase": "SIF_PUBLISH", "status": "READY", "timestamp": "2026-10-08T11:59:59Z"}, "UNAVAILABLE"),
        (None, "UNAVAILABLE"),
    ):
        monkeypatch.setattr(WydSlurmBackend, "stage_progress", lambda _self, _run: event)
        remote = client.get(f"/api/actions/{action_id}/diagnostics?refresh=true").json()["remote"]
        assert remote["status"] == status
        if status == "OBSERVED":
            assert remote["next_action"] == "OBSERVE_PREPARATION" and remote["image_ready_reported"] is False


def test_refresh_reports_verified_cache_reuse_as_ready_without_hashing_on_get(client, monkeypatch):
    action_id, _, _ = _managed_action(client)
    store = client.app.state.runtime.action_store
    before = store.snapshot(action_id)
    event = {"phase": "CACHE_VERIFY", "status": "READY", "timestamp": "2026-10-08T12:21:00Z"}
    monkeypatch.setattr(WydSlurmBackend, "stage_progress", lambda _self, _run: event)
    response = client.get(f"/api/actions/{action_id}/diagnostics?refresh=true")
    assert response.status_code == 200, response.text
    remote = response.json()["remote"]
    assert remote["event"] == event and remote["image_ready_reported"] is True
    assert remote["next_action"] == "PREPARE_NEW_ACTION" and remote["observed_after_action"] is True
    assert remote["cache_reverified"] is False and remote["historical_failure_cause"] is False
    assert store.snapshot(action_id) == before


def test_refresh_accepts_embedded_oci_source_provenance_without_source_flags(client, monkeypatch):
    action_id, _, _ = _managed_action(client, embedded=True, revision=True)
    store = client.app.state.runtime.action_store
    before = store.snapshot(action_id)
    assert before["expected_source_id"] == ""
    assert "--source-root" not in before["stage_command_preview"]
    assert "--source-id" not in before["stage_command_preview"]
    assert "--attempt-id" in before["stage_command_preview"] and "--campaign-id" in before["stage_command_preview"]
    event = {"phase": "CACHE_VERIFY", "status": "READY", "timestamp": "2026-10-08T12:21:00Z"}
    monkeypatch.setattr(WydSlurmBackend, "stage_progress", lambda _self, _run: event)
    response = client.get(f"/api/actions/{action_id}/diagnostics?refresh=true")
    assert response.status_code == 200, response.text
    remote = response.json()["remote"]
    assert remote["status"] == "OBSERVED" and remote["event"] == event
    assert remote["image_ready_reported"] is True and remote["next_action"] == "PREPARE_NEW_ACTION"
    assert store.snapshot(action_id) == before


@pytest.mark.parametrize("source", [None, "../private", "source.invalid", [], "source." + "f" * 64])
def test_embedded_oci_requires_valid_frozen_source_provenance(client, monkeypatch, source):
    action_id, path, campaign = _managed_action(client, embedded=True)
    store = client.app.state.runtime.action_store
    campaign["runs"][0]["source_id"] = source
    path.write_text(yaml.safe_dump(campaign))
    plan_path = store.directory(action_id) / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["execution_campaign_sha256"] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(WydSlurmBackend, "stage_progress", lambda *_args: pytest.fail("invalid embedded source must not query the backend"))
    remote = client.get(f"/api/actions/{action_id}/diagnostics?refresh=true").json()["remote"]
    assert remote["status"] == "UNAVAILABLE" and remote["error_class"] == "ValueError"


def test_nonempty_expected_source_must_match_the_frozen_run(client, monkeypatch):
    action_id, _, _ = _managed_action(client)
    store = client.app.state.runtime.action_store
    plan_path = store.directory(action_id) / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["expected_source_id"] = "source." + "f" * 64
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(WydSlurmBackend, "stage_progress", lambda *_args: pytest.fail("mismatched source must not query the backend"))
    remote = client.get(f"/api/actions/{action_id}/diagnostics?refresh=true").json()["remote"]
    assert remote["status"] == "UNAVAILABLE" and remote["error_class"] == "ValueError"


@pytest.mark.parametrize("tamper", ["hash", "project", "source", "run", "stage_command", "stage_cwd", "path", "symlink", "directory_symlink", "source_shape", "source_store"])
def test_refresh_rejects_frozen_binding_tamper_before_remote_access(client, monkeypatch, tamper):
    action_id, path, campaign = _managed_action(client)
    store = client.app.state.runtime.action_store
    plan_path = store.directory(action_id) / "plan.json"
    plan = json.loads(plan_path.read_text())
    if tamper == "hash":
        path.write_text(path.read_text() + "# tampered\n")
    elif tamper in {"project", "source", "run", "source_store"}:
        if tamper == "project": campaign["project"] = "other"
        elif tamper == "source": campaign["runs"][0]["source_id"] = "source." + "f" * 64
        elif tamper == "run": campaign["runs"][0]["run_id"] = "other"
        else: campaign["source_store"] = "/private/untrusted"
        path.write_text(yaml.safe_dump(campaign))
        plan["execution_campaign_sha256"] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    elif tamper == "stage_command": plan["stage_command_preview"] = ["private-command"]
    elif tamper == "stage_cwd": plan["stage_cwd"] = "/untrusted"
    elif tamper == "path": plan["execution_campaign_file"] = "/untrusted/campaign.yml"
    elif tamper == "source_shape": plan["expected_source_id"] = "../private"
    elif tamper == "symlink":
        actual = path.with_suffix(".real")
        path.rename(actual)
        path.symlink_to(actual)
    else:
        original = path.parent
        target = original.with_name(original.name + "-real")
        original.rename(target)
        original.symlink_to(target, target_is_directory=True)
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(WydSlurmBackend, "stage_progress", lambda *_args: pytest.fail("tampered binding must not reach backend"))
    remote = client.get(f"/api/actions/{action_id}/diagnostics?refresh=true").json()["remote"]
    assert remote["status"] == "UNAVAILABLE" and remote["error_class"] == "ValueError"


@pytest.mark.parametrize("unsupported", ["external", "base_missing", "controller_missing", "capability_missing", "entry_changed", "sensecore", "oci_missing"])
def test_refresh_never_invokes_external_or_non_wyd_controller(client, monkeypatch, unsupported):
    action_id, path, campaign = _managed_action(client, executor="cloud" if unsupported == "sensecore" else "gpu")
    daemon = client.app.state.runtime
    project = daemon.project("demo")
    if unsupported == "external": project.base_dir = Path("/external/project")
    elif unsupported == "base_missing": project.base_dir = None
    elif unsupported == "controller_missing": project.controller = None
    elif unsupported == "capability_missing": project.controller.capabilities = {}
    elif unsupported == "entry_changed": project.controller.experimentctl = "untrusted.py"
    elif unsupported == "oci_missing":
        campaign["runs"][0]["backend"].pop("oci_image")
        path.write_text(yaml.safe_dump(campaign))
        plan_path = daemon.action_store.directory(action_id) / "plan.json"
        plan = json.loads(plan_path.read_text())
        plan["execution_campaign_sha256"] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(WydSlurmBackend, "stage_progress", lambda *_args: pytest.fail("unsupported controller must not query"))
    assert client.get(f"/api/actions/{action_id}/diagnostics?refresh=true").json()["remote"]["status"] == "UNSUPPORTED"


def test_diagnostics_handle_unstarted_or_non_submission_actions(client):
    store = client.app.state.runtime.action_store
    action_id = synthetic_plan(store, "unstarted")
    store.set_execution(action_id, {**store.execution(action_id), "result": ["not-stage"]}, event="synthetic")
    value = client.get(f"/api/actions/{action_id}/diagnostics?refresh=true").json()
    assert value["stage_command"]["stdout"] == value["stage_command"]["stderr"] == ""
    assert value["remote"]["status"] == "NOT_STARTED"


@pytest.mark.parametrize("resolution,safe_retry,next_action,submitted,retry_scope,hint", [
    ("SUBMITTED", False, "MONITOR_RUN", True, None, "MONITOR_RUN"),
    ("UNKNOWN_DO_NOT_RETRY", False, "RECONCILE", None, None, "OBSERVE_ACTION"),
    (None, False, "WAIT", None, None, "OBSERVE_ACTION"),
    ("FAILED_BEFORE_SUBMISSION", True, "PREPARE_NEW_ACTION", False, "NEW_ACTION_BEFORE_SCHEDULER_SUBMISSION", "PREPARE_NEW_ACTION"),
    ("FAILED_BEFORE_SUBMISSION", False, "RECONCILE", False, None, "OBSERVE_ACTION"),
])
def test_saved_stage_diagnostic_does_not_claim_submitted_actions_are_safe_to_retry(
    client, resolution, safe_retry, next_action, submitted, retry_scope, hint,
):
    store = client.app.state.runtime.action_store
    action_id = synthetic_plan(store, "submission-facts")
    store.set_execution(action_id, {**store.execution(action_id), "status": "VERIFIED" if submitted else "FAILED",
                        "resolution": resolution, "safe_to_retry": safe_retry, "next_action": next_action,
                        "result": {"stage_command": {"returncode": 0, "timeout": False, "stdout": "", "stderr": ""}}}, event="synthetic")
    before = store.snapshot(action_id)
    diagnostic = client.get(f"/api/actions/{action_id}/diagnostics").json()["stage_diagnostic"]
    assert diagnostic["scheduler_submitted"] is submitted
    assert diagnostic["retry_scope"] == retry_scope and diagnostic["next_action"] == hint
    assert store.snapshot(action_id) == before


def test_action_diagnostics_require_normal_api_authentication(tmp_path):
    secret = tmp_path / "token"
    token = "diagnostic-test-token-private-12345"
    secret.write_text(token)
    secret.chmod(0o600)
    config = ServerConfig(index_db=str(tmp_path / "index.sqlite"), action_root=str(tmp_path / "actions"),
                          project_registry_root=str(tmp_path / "projects"), telemetry={"enabled": False},
                          collector_enabled=False, http_auth={"bearer_token_file": str(secret)})
    store = ActionStore(config.action_root_path())
    action_id = synthetic_plan(store, "auth")
    with TestClient(create_app(config, poll=False)) as api:
        path = f"/api/actions/{action_id}/diagnostics"
        assert api.get(path).status_code == 401
        assert api.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert api.get(path, headers={"Authorization": "Bearer " + token}).status_code == 200
