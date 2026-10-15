"""Administrative deletion binds reviewed targets and never replays unknown effects."""
from copy import deepcopy
import json

import pytest

from ml_exp_client.api import ClientError
from ml_exp_client.cli import main, parser
from ml_exp_client.run_deletion import run_deletion


ACTION = "action-" + "a" * 16
TARGET = "sha256:" + "b" * 64


class API:
    base = "https://api.example"

    def __init__(self, path):
        self.path = path
        self.calls = []
        self.action = None
        self.fail = None
        self.after_effect = False
        self.complete_execute = False

    def negotiate(self):
        return {"capabilities": ["resource-deletion.v1"]}

    def complete(self):
        self.action["execution"] = {"status": "VERIFIED", "result": {
            "status": "DELETED", "project": "demo", "run_id": "trial", "target_digest": TARGET,
            "retained": {"action_audit": True, "tombstone": True, "shared_assets": True,
                         "wandb_cloud": True, "provider_history": "unsupported"}}}

    def call(self, path, **kwargs):
        self.calls.append((path, deepcopy(kwargs)))
        saved = json.loads(self.path.read_text())
        if path.endswith("/deletion"):
            assert saved["request"] == kwargs["data"] and saved["plan"] is None
        if path == "/api/actions/authorize":
            assert saved["authorization_requested"]
        if path == "/api/actions/execute":
            assert saved["execution_requested"]
        if path == self.fail and not self.after_effect:
            raise ClientError("reply lost")
        if path.endswith("/deletion"):
            request = kwargs["data"]
            self.action = {"action_id": ACTION, "operation": "DELETE_RUN_RESOURCES",
                "scope": {"project": "demo", "scope_type": "run", "object_id": "trial"},
                "run_id": "trial", "intent_id": request["idempotency_key"],
                "reason": request["reason"], "deletion_request": deepcopy(request),
                "target_manifest": {"attempts": ["attempt-001"], "objects": [{"key": "exact-archive", "sha256": "c" * 64}]},
                "target_digest": TARGET, "intent_digest": "sha256:" + "d" * 64,
                "gate_bundle_digest": "sha256:" + "e" * 64,
                "gates": [{"name": "terminal", "status": "PASS"}], "ready": True,
                "execution": {"status": "PREPARED"}}
        elif path == "/api/actions/authorize":
            self.action["execution"]["status"] = "AUTHORIZED"
        elif path == "/api/actions/execute":
            self.action["execution"]["status"] = "EXECUTING"
            if self.complete_execute:
                self.complete()
        elif path == "/api/actions/reconcile":
            self.complete()
        else:
            assert path.startswith("/api/actions/action-")
        if path == self.fail:
            raise ClientError("reply lost")
        return deepcopy(self.action)


def prepare(api, **kwargs):
    return run_deletion(api, api.path, project="demo", run="trial", reason="retire completed experiment", **kwargs)


def test_prepare_review_observe_execute_and_verified_reconcile(tmp_path):
    api = API(tmp_path / "delete.json")
    review = prepare(api)
    saved = json.loads(api.path.read_text())
    assert api.path.stat().st_mode & 0o777 == 0o600
    assert review["next_action"] == "EXECUTE_SAVED_PLAN" and not review["deleted"]
    assert review["execute_argv"] == ["ml-exp", "--url", api.base, "delete", "--state", str(api.path), "--execute"]
    assert saved["request"]["idempotency_key"].startswith("deletion-")
    assert saved["plan"] == review["action"] == saved["action"]
    api.calls.clear()
    assert run_deletion(api, api.path)["action"] == review["action"]
    assert api.calls == [("/api/actions/" + ACTION, {})]
    result = run_deletion(api, api.path, execute=True)
    assert not result["deleted"] and result["next_action"] == "OBSERVE_ACTION"
    assert api.calls[-2:] == [("/api/actions/authorize", {"data": {
        "action_id": ACTION, "note": "explicit CLI execution of the saved Run deletion plan"}}),
        ("/api/actions/execute", {"data": {"action_id": ACTION, "confirmation": "EXECUTE " + ACTION}})]
    api.calls.clear()
    run_deletion(api, api.path, execute=True)
    assert api.calls == [("/api/actions/" + ACTION, {})]
    api.action["execution"]["status"] = "RECONCILE_REQUIRED"
    result = run_deletion(api, api.path, reconcile=True)
    assert result["deleted"] and result["next_action"] == "DELETION_COMPLETE"
    assert result["action"]["execution"]["result"]["retained"]["wandb_cloud"]
    api.calls.clear()
    run_deletion(api, api.path, execute=True, action_id=ACTION)
    assert api.calls == [("/api/actions/" + ACTION, {})]


@pytest.mark.parametrize("failure", ["prepare", "authorize", "execute"])
@pytest.mark.parametrize("after_effect", [False, True])
def test_lost_reply_preserves_markers_and_does_not_repeat_request(tmp_path, failure, after_effect):
    api = API(tmp_path / "delete.json")
    if failure != "prepare":
        prepare(api)
    endpoint = {"prepare": "/api/runs/demo/trial/deletion", "authorize": "/api/actions/authorize",
                "execute": "/api/actions/execute"}[failure]
    api.fail, api.after_effect = endpoint, after_effect
    with pytest.raises(ClientError, match="reply lost"):
        prepare(api) if failure == "prepare" else run_deletion(api, api.path, execute=True)
    saved = json.loads(api.path.read_text())
    assert saved["request"]["idempotency_key"]
    api.fail = None
    api.calls.clear()
    if failure == "prepare":
        for kwargs in ({}, {"execute": True}, {"reconcile": True}, {"action_id": ACTION, "execute": True}):
            with pytest.raises(ClientError, match="preparation response is unknown"):
                run_deletion(api, api.path, **kwargs)
        assert not api.calls
        if api.action is None:
            # The operator found the original matching Action; the client still
            # attaches with GET rather than replaying its preparation request.
            api.call(endpoint, data=saved["request"])
        api.calls.clear()
        result = run_deletion(api, api.path, action_id=ACTION)
        assert result["next_action"] == "EXECUTE_SAVED_PLAN"
        assert api.calls == [("/api/actions/" + ACTION, {})]
    else:
        result = run_deletion(api, api.path, execute=True)
        assert endpoint not in [path for path, _ in api.calls]
        if failure == "authorize" and after_effect:
            assert api.calls[-1][0] == "/api/actions/execute"
        else:
            assert api.calls == [("/api/actions/" + ACTION, {})]
            assert not result["deleted"] and result["next_action"] == "OBSERVE_ACTION"


@pytest.mark.parametrize("change", [
    {"operation": "WRITE_RUN_ARCHIVE"}, {"scope": {"project": "other", "scope_type": "run", "object_id": "trial"}},
    {"run_id": "other"}, {"intent_id": "other-key"}, {"deletion_request": {}}, {"reason": "changed"},
    {"action_id": "bad"}, {"action_id": "action-" + "f" * 16}, {"execution": None},
    {"execution": {"status": "UNKNOWN"}}, {"target_manifest": None}, {"gates": None}, {"ready": 1},
    {"target_digest": "unsealed"}, {"intent_digest": "sha256:" + "f" * 64},
    {"gate_bundle_digest": "sha256:" + "f" * 64}, {"target_digest": "sha256:" + "f" * 64},
    {"target_manifest": {"objects": ["unreviewed-object"]}},
])
def test_changed_identity_or_targets_never_reach_authorization(tmp_path, change):
    api = API(tmp_path / "delete.json")
    prepare(api)
    api.action.update(change)
    api.calls.clear()
    with pytest.raises(ClientError):
        run_deletion(api, api.path, execute=True)
    assert api.calls == [("/api/actions/" + ACTION, {})]


@pytest.mark.parametrize("change", [None, {}, {"status": "DELETED"},
    {"status": "PENDING", "project": "demo", "run_id": "trial", "target_digest": TARGET},
    {"status": "DELETED", "project": "other", "run_id": "trial", "target_digest": TARGET},
    {"status": "DELETED", "project": "demo", "run_id": "other", "target_digest": TARGET},
    {"status": "DELETED", "project": "demo", "run_id": "trial", "target_digest": "sha256:" + "c" * 64}])
def test_verified_requires_exact_deleted_result(tmp_path, change):
    api = API(tmp_path / "delete.json")
    prepare(api)
    api.action["execution"] = {"status": "VERIFIED", "result": change}
    with pytest.raises(ClientError, match="completion lacks"):
        run_deletion(api, api.path)


@pytest.mark.parametrize("status", ["BLOCKED", "FAILED", "EXECUTING", "RECONCILE_REQUIRED", "AUTHORIZED"])
def test_nonverified_status_is_never_completion_and_reconcile_is_explicit(tmp_path, status):
    api = API(tmp_path / "delete.json")
    prepare(api)
    api.action["execution"] = {"status": status, "result": {
        "status": "DELETED", "project": "demo", "run_id": "trial", "target_digest": TARGET}}
    result = run_deletion(api, api.path)
    assert not result["deleted"]
    assert result["next_action"] == {"EXECUTING": "OBSERVE_ACTION", "RECONCILE_REQUIRED": "RECONCILE",
        "AUTHORIZED": "EXECUTE_SAVED_PLAN"}.get(status, "INSPECT_ACTION")
    api.calls.clear()
    run_deletion(api, api.path, reconcile=True)
    assert any(path.endswith("/reconcile") for path, _ in api.calls) == (status in {"EXECUTING", "RECONCILE_REQUIRED"})
    assert not any(path.endswith("/execute") for path, _ in api.calls)


def test_blocked_plan_preserves_reasons_and_never_executes(tmp_path):
    api = API(tmp_path / "delete.json")
    prepare(api)
    api.action.update(ready=False, gates=[{"name": "checkpoint_references", "status": "FAIL", "detail": "another Run uses it"}])
    result = run_deletion(api, api.path, execute=True)
    assert result["next_action"] == "INSPECT_ACTION" and "execute_argv" not in result
    assert result["action"]["gates"][0]["detail"] == "another Run uses it"
    assert not any(path.endswith("/execute") for path, _ in api.calls)


@pytest.mark.parametrize("kwargs", [{"execute": True}, {"reconcile": True}, {"action_id": ACTION},
    {"reason": None}, {"reason": ""}, {"reason": "  "}, {"reason": "x" * 4001}, {"reason": 1},
    {"project": None}, {"run": None}, {"project": "../escape"}, {"run": "../escape"},
    {"execute": True, "reconcile": True}])
def test_invalid_preparation_does_not_write_state_or_call_http(tmp_path, kwargs):
    api = API(tmp_path / "delete.json")
    arguments = {"project": "demo", "run": "trial", "reason": "remove completed run"}
    arguments.update(kwargs)
    with pytest.raises(ClientError):
        run_deletion(api, api.path, **arguments)
    assert not api.path.exists() and not api.calls


@pytest.mark.parametrize("kwargs", [{"project": "other"}, {"run": "other"}, {"reason": "changed"},
    {"action_id": "action-" + "f" * 16}])
def test_saved_request_cannot_be_changed(tmp_path, kwargs):
    api = API(tmp_path / "delete.json")
    prepare(api)
    api.calls.clear()
    with pytest.raises(ClientError):
        run_deletion(api, api.path, execute=True, **kwargs)
    assert not api.calls


@pytest.mark.parametrize("change", [{"contract": "other"}, {"server": "https://other"}, {"operation": "retry"},
    {"request": []}, {"request": {"reason": "", "idempotency_key": "key"}},
    {"request": {"reason": "remove", "idempotency_key": None}},
    {"request": {"reason": "remove", "idempotency_key": "bad key"}},
    {"request": {"reason": "remove", "idempotency_key": "key", "extra": 1}},
    {"execution_requested": "false"}, {"authorization_requested": None}, {"project": None}, {"run_id": None},
    {"plan": []}, {"recovery_url": None}])
def test_malformed_saved_state_is_rejected_before_http(tmp_path, change):
    api = API(tmp_path / "delete.json")
    prepare(api)
    saved = json.loads(api.path.read_text())
    saved.update(change)
    api.path.write_text(json.dumps(saved))
    api.calls.clear()
    with pytest.raises(ClientError):
        run_deletion(api, api.path, execute=True)
    assert not api.calls


def test_attachment_requires_matching_original_id_and_request(tmp_path):
    api = API(tmp_path / "delete.json")
    prepare(api)
    saved = json.loads(api.path.read_text())
    saved.update(plan=None, action=None)
    api.path.write_text(json.dumps(saved))
    api.calls.clear()
    with pytest.raises(ClientError, match="invalid deletion Action identity"):
        run_deletion(api, api.path, action_id="bad")
    assert not api.calls
    with pytest.raises(ClientError, match="different deletion Action identity"):
        run_deletion(api, api.path, action_id="action-" + "f" * 16)
    api.action["intent_id"] = "unrelated-key"
    with pytest.raises(ClientError, match="identity or request differs"):
        run_deletion(api, api.path, action_id=ACTION)
    assert json.loads(api.path.read_text())["plan"] is None


def test_cli_prepares_reports_pending_and_completes(tmp_path, monkeypatch, capsys):
    api = API(tmp_path / "delete.json")
    monkeypatch.setattr("ml_exp_client.cli.Client", lambda *args: api)
    assert main(["delete", "--project", "demo", "--run", "trial", "--reason", "remove completed experiment",
                 "--state", str(api.path)]) == 0
    assert not json.loads(capsys.readouterr().out)["deleted"]
    assert main(["delete", "--state", str(api.path), "--execute"]) == 2
    assert not json.loads(capsys.readouterr().out)["deleted"]
    assert main(["delete", "--state", str(api.path), "--reconcile"]) == 0
    assert json.loads(capsys.readouterr().out)["deleted"]
    assert main(["delete", "--state", str(api.path), "--execute"]) == 0
    assert json.loads(capsys.readouterr().out)["deleted"]


def test_cli_requires_capability_before_saving_or_preparing(tmp_path, monkeypatch, capsys):
    api = API(tmp_path / "delete.json")
    api.negotiate = lambda: {"capabilities": []}
    monkeypatch.setattr("ml_exp_client.cli.Client", lambda *args: api)
    assert main(["delete", "--project", "demo", "--run", "trial", "--reason", "cleanup", "--state", str(api.path)]) == 2
    assert "resource-deletion.v1" in capsys.readouterr().err
    assert not api.calls and not api.path.exists()


def test_cli_execute_without_review_and_conflicting_flags_are_rejected(tmp_path, monkeypatch, capsys):
    api = API(tmp_path / "delete.json")
    monkeypatch.setattr("ml_exp_client.cli.Client", lambda *args: api)
    assert main(["delete", "--state", str(api.path), "--execute"]) == 2
    assert "prepare and review" in capsys.readouterr().err
    assert not api.calls
    with pytest.raises(SystemExit):
        parser().parse_args(["delete", "--state", str(api.path), "--execute", "--reconcile"])


def test_save_failure_prevents_authorization_request(tmp_path, monkeypatch):
    api = API(tmp_path / "delete.json")
    prepare(api)
    api.calls.clear()
    original = __import__("ml_exp_client.run_deletion", fromlist=["save"]).save
    def fail_marker(path, value):
        if value["authorization_requested"]:
            raise OSError("cannot persist marker")
        original(path, value)
    monkeypatch.setattr("ml_exp_client.run_deletion.save", fail_marker)
    with pytest.raises(OSError, match="cannot persist marker"):
        run_deletion(api, api.path, execute=True)
    assert api.calls == [("/api/actions/" + ACTION, {})]
