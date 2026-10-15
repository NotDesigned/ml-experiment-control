"""HTTP Action and receipt contracts; all data and backend effects are temporary."""
import json
import time

from fastapi.testclient import TestClient
import pytest

from ml_exp_server.api.app import create_app
from ml_exp_server.backends.run_deletion import digest as backend_digest
from ml_exp_server.projects.project_config import load_research_project
from tests.test_run_deletion_service import deletion_env, publish


@pytest.fixture
def deletion_api(deletion_env, tmp_path):
    env = deletion_env
    token = tmp_path / "admin-token"
    token.write_text("a" * 40)
    token.chmod(0o600)
    env.config.http_auth.bearer_token_file = str(token)
    app = create_app(env.config, projects=[load_research_project(env.project_file)], poll=False)
    app.state.runtime_initializers.append(env.install)
    with TestClient(app, headers={"Authorization": "Bearer " + "a" * 40}) as api:
        yield api, env


def test_prepare_execute_get_and_reconcile_exact_action_over_http(deletion_api):
    api, env = deletion_api
    receipt, original = publish(env)
    request = {"reason": "retire this isolated Run", "idempotency_key": "http-review"}
    response = api.post("/api/runs/demo/run-a/deletion", json=request)
    assert response.status_code == 200, response.text
    plan = response.json()
    action_id = plan["action_id"]
    assert plan["deletion_request"] == request and plan["intent_id"] == request["idempotency_key"]
    assert plan["target_manifest"]["run_id"] == "run-a" and not env.deleted
    assert plan["target_manifest"]["retained"]["metric_definitions"] is True
    assert api.post("/api/runs/demo/run-a/deletion", json=request).json() == plan
    listing = api.get("/api/actions", params={"project": "demo", "scope_type": "run", "object_id": "run-a"})
    assert listing.status_code == 200
    assert request["idempotency_key"] in listing.text
    assert api.post("/api/actions/execute", json={"action_id": action_id, "confirmation": "EXECUTE " + action_id}).status_code == 409
    authorized = api.post("/api/actions/authorize", json={"action_id": action_id, "note": "reviewed"})
    assert authorized.status_code == 200 and authorized.json()["execution"]["status"] == "AUTHORIZED"
    dispatch = api.post("/api/actions/execute", json={"action_id": action_id, "confirmation": "EXECUTE " + action_id})
    assert dispatch.status_code == 200 and dispatch.json()["execution"]["status"] == "EXECUTING"
    deadline = time.monotonic() + 5
    while True:
        observed = api.get("/api/actions/" + action_id).json()
        if observed["execution"]["status"] != "EXECUTING": break
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert observed["execution"]["status"] == "VERIFIED", observed
    assert observed["execution"]["result"]["target_digest"] == plan["target_digest"]
    assert observed["execution"]["result"]["retained"]["metric_definitions"] is True
    assert len(env.deleted) == 2 and receipt.read_bytes() == original
    assert api.post("/api/actions/reconcile", json={"action_id": action_id}).json() == observed
    late = api.put("/api/artifact-transfers/demo/run-a/attempt-001", content=b"late", headers={
        "Authorization": "Bearer " + env.tokens["run-a"]})
    assert late.status_code == 410 and late.headers["X-ML-Expd-Error-Code"] == "RUN_DELETED"
    assert api.get("/api/projects/demo/overview").status_code == 200


def test_administrator_auth_validation_and_blocked_policy_are_explicit(deletion_api):
    api, env = deletion_api
    endpoint = "/api/runs/demo/run-a/deletion"
    request = {"reason": "isolated cleanup", "idempotency_key": "http-review"}
    assert api.post(endpoint, json=request, headers={"Authorization": "Bearer wrong"}).status_code == 401
    for invalid in [{"reason": "", "idempotency_key": "key"}, {"reason": "x", "idempotency_key": "bad key"},
                    {**request, "target": "/arbitrary/path"}]:
        assert api.post(endpoint, json=invalid).status_code == 422
    whitespace = api.post(endpoint, json={"reason": " ", "idempotency_key": "space"})
    assert whitespace.status_code == 422 and whitespace.headers["X-ML-Expd-Error-Code"] == "DELETION_REASON_REQUIRED"
    assert api.post("/api/runs/demo/missing/deletion", json=request).status_code == 404
    env.config.action_runtime.allow_resource_deletions = False
    blocked = api.post(endpoint, json=request)
    assert blocked.status_code == 200 and blocked.json()["execution"]["status"] == "BLOCKED"
    assert not blocked.json()["ready"] and not env.deleted
    assert api.post("/api/actions/authorize", json={"action_id": blocked.json()["action_id"]}).status_code == 409


def test_worker_receipt_is_bounded_authenticated_idempotent_and_identity_bound(deletion_api):
    api, env = deletion_api
    env.backend.callback = True
    plan = api.post("/api/runs/demo/run-a/deletion", json={"reason": "isolate callback", "idempotency_key": "receipt-key"}).json()
    action_id = plan["action_id"]
    capability = "exact-worker-capability"
    state = {"deletion_id": action_id, "plan_sha256": backend_digest(plan["target_manifest"]["backend"]),
             "capability": capability, "receipt_url": "https://example.test/receipt"}
    env.service.update(action_id, status="DELETING", target_manifest=plan["target_manifest"], target_digest=plan["target_digest"], backend=state)
    payload = env.backend.payload(plan["target_manifest"]["backend"], state)
    receipt = {key: payload[key] for key in ("deletion_id", "project", "run_id", "plan_sha256", "target")}
    receipt.update(contract="run-deletion.v1", status="DELETED", target_absent=True, files_deleted=3, bytes_deleted=19)
    endpoint = "/api/run-deletion-transfers/" + action_id + "/receipt"
    headers = {"Authorization": "Bearer " + capability}
    for authorization in ("", "wrong scheme", "Bearer wrong", "Bearer " + "a" * 40):
        denied = api.post(endpoint, json=receipt, headers={"Authorization": authorization})
        assert denied.status_code == 403 and denied.headers["X-ML-Expd-Error-Code"] == "DELETION_CAPABILITY_REQUIRED"
    assert api.post(endpoint, content="{", headers=headers).status_code == 422
    assert api.post(endpoint, content=b"x" * 65537, headers=headers).status_code == 413
    assert api.post(endpoint, json={**receipt, "run_id": "other"}, headers=headers).status_code == 409
    assert api.post(endpoint, json=receipt, headers=headers).json() == {"accepted": True}
    assert api.post(endpoint, json=receipt, headers=headers).json() == {"accepted": True}
    conflict = api.post(endpoint, json={**receipt, "bytes_deleted": 20}, headers=headers)
    assert conflict.status_code == 409 and conflict.headers["X-ML-Expd-Error-Code"] == "DELETION_RECEIPT_CONFLICT"
    assert api.post("/api/runs/demo/run-a/deletion", json={"reason": "worker cannot delete", "idempotency_key": "bad"}, headers=headers).status_code == 401
    assert api.get(endpoint, headers=headers).status_code == 401
    assert api.post(endpoint.replace(action_id, "invalid"), json=receipt, headers=headers).status_code == 401
    assert not env.deleted and env.local["run-a"].exists()


def test_callback_only_accepts_in_progress_or_completed_original_state(deletion_api):
    api, env = deletion_api
    action_id = "action-" + "f" * 16
    env.service.update(action_id, status="PREPARED", backend={"capability": "fixture-capability"})
    response = api.post("/api/run-deletion-transfers/" + action_id + "/receipt", json={}, headers={"Authorization": "Bearer fixture-capability"})
    assert response.status_code == 403


def test_standalone_cli_accepts_real_action_contract_and_only_reports_verified_deletion(deletion_api, tmp_path, monkeypatch, capsys):
    from ml_exp_client.api import ClientError
    from ml_exp_client.cli import main
    api, env = deletion_api
    class Bridge:
        base = "https://isolated.example"
        def negotiate(self):
            return api.get("/api/health").json()
        def call(self, path, *, data=None):
            response = api.request("GET" if data is None else "POST", path, json=data)
            if response.status_code >= 400:
                raise ClientError("isolated HTTP failure", status=response.status_code)
            return response.json()
    monkeypatch.setattr("ml_exp_client.cli.Client", lambda *args: Bridge())
    state = tmp_path / "client-delete.json"
    local_result = tmp_path / "client-downloaded-result.bin"
    local_result.write_bytes(b"independent local download")
    assert main(["delete", "--project", "demo", "--run", "run-a", "--reason", "review real response",
                 "--state", str(state)]) == 0
    reviewed = json.loads(capsys.readouterr().out)
    assert reviewed["next_action"] == "EXECUTE_SAVED_PLAN" and not reviewed["deleted"]
    assert main(["delete", "--state", str(state), "--execute"]) == 2
    assert not json.loads(capsys.readouterr().out)["deleted"]
    deadline = time.monotonic() + 5
    while True:
        assert main(["delete", "--state", str(state)]) == 0
        observed = json.loads(capsys.readouterr().out)
        if observed["deleted"]: break
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert observed["action"]["execution"]["result"]["target_digest"] == reviewed["action"]["target_digest"]
    assert local_result.read_bytes() == b"independent local download"
