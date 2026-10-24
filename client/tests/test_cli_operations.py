"""Training reads retain scientific dimensions; scheduler Actions are never replayed."""
from copy import deepcopy
import json

import pytest

from ml_exp_client.api import ClientError
from ml_exp_client.cli import main, parser
from ml_exp_client.operations import training_operation
from ml_exp_client.status import read_attempt


ACTION_ID = "action-" + "a" * 16


def snapshot(operation="cancel", status="PREPARED", **changes):
    value = {"action_id": ACTION_ID, "operation": "CANCEL_RUN" if operation == "cancel" else "RETRY_ATTEMPT",
             "scope": {"project": "demo", "scope_type": "attempt", "object_id": "trial::attempt-001"},
             "run_id": "trial", "attempt_id": "attempt-001" if operation == "cancel" else "attempt-002",
             "intent_digest": "sha256:" + "b" * 64, "gate_bundle_digest": "sha256:" + "c" * 64,
             "ready": True, "gates": [{"name": "preflight", "status": "PASS"}],
             "preflight_summary": {"resources": {"gpus": 1}, "requested_gpu_hours": 1,
                                   "max_gpu_hours": 2, "resource_approval": "budget_cap"},
             "execution": {"status": status}}
    value.update(changes)
    return value


class API:
    def __init__(self, operation="cancel", *, action=None):
        self.base = "https://api.example"
        self.calls = []
        self.action = snapshot(operation) if action is None else action
        self.fail = None
        self.fail_after_effect = False

    def negotiate(self):
        return {"capabilities": []}

    def call(self, path, **kwargs):
        self.calls.append((path, deepcopy(kwargs)))
        if self.fail == path and not self.fail_after_effect:
            raise ClientError("reply lost")
        if path.endswith("/cancel") or path.endswith("/retry"):
            result = {"action": self.action}
        elif path == "/api/actions/authorize":
            self.action["execution"]["status"] = "AUTHORIZED"
            result = self.action
        elif path == "/api/actions/execute":
            self.action["execution"]["status"] = "EXECUTING"
            result = self.action
        elif path == "/api/actions/reconcile":
            self.action["execution"]["status"] = "VERIFIED"
            result = self.action
        elif path == "/api/actions/" + ACTION_ID:
            result = self.action
        else:
            raise AssertionError(path)
        if self.fail == path:
            raise ClientError("reply lost")
        return deepcopy(result)


def prepare(api, path, operation="cancel", **kwargs):
    if operation == "retry":
        kwargs.setdefault("max_gpu_hours", 2)
    return training_operation(api, operation, path, project="demo", run="trial", attempt="attempt-001", **kwargs)


@pytest.mark.parametrize("view", ["logs", "metrics"])
@pytest.mark.parametrize("explicit", [False, True])
def test_read_exact_attempt_and_preserve_metric_dimensions(view, explicit):
    calls = []
    payload = {"project": "demo", "run_id": "trial", "attempt_id": "attempt-001", "records": [{"step": 3, "loss": 1.5, "unit": "nats/token", "protocol": "validation", "variant": "heldout"}]}
    class Reader:
        def call(self, path, **kwargs):
            calls.append((path, kwargs))
            if path.endswith("/attempts"):
                return {"project": "demo", "run_id": "trial", "current_attempt_id": "attempt-001",
                        "attempts": [{"identity": "trial::attempt-001", "attempt_id": "attempt-001"}]}
            return payload
    result = read_attempt(Reader(), view, "demo", "trial", "attempt-001" if explicit else None,
                          stream="stderr", lines=17, keys="loss,accuracy", max_points=20)
    assert result is payload
    assert len(calls) == (1 if explicit else 2)
    suffix = "logs?stream=stderr&lines=17" if view == "logs" else "metrics?max_points=20&keys=loss%2Caccuracy"
    assert calls[-1] == ("/api/attempts/demo/trial%3A%3Aattempt-001/" + suffix, {})


@pytest.mark.parametrize("listing", [[], {}, {"project": "other", "run_id": "trial"},
    {"project": "demo", "run_id": "trial", "current_attempt_id": "latest"},
    {"project": "demo", "run_id": "trial", "current_attempt_id": "attempt-001", "attempts": []},
    {"project": "demo", "run_id": "trial", "current_attempt_id": "attempt-001", "attempts": None},
    {"project": "demo", "run_id": "trial", "current_attempt_id": "attempt-001",
     "attempts": [{"identity": "other::attempt-001", "attempt_id": "attempt-001"}]}])
def test_missing_or_changed_current_attempt_is_not_guessed(listing):
    class Reader:
        def call(self, path):
            assert path.endswith("/attempts")
            return listing
    with pytest.raises(ClientError):
        read_attempt(Reader(), "logs", "demo", "trial")


@pytest.mark.parametrize("view,kwargs", [("unknown", {}), ("logs", {"lines": 0}), ("logs", {"lines": 10001}),
    ("logs", {"stream": "secret"}), ("metrics", {"max_points": 0}), ("metrics", {"attempt": "latest"})])
def test_bad_read_arguments_do_not_make_http_requests(view, kwargs):
    api = API()
    with pytest.raises(ClientError):
        read_attempt(api, view, "demo", "trial", **kwargs)
    assert not api.calls


@pytest.mark.parametrize("operation", ["cancel", "retry"])
def test_prepare_saves_reviewable_plan_and_requires_second_explicit_execute(tmp_path, operation):
    api, path = API(operation), tmp_path / "plan.json"
    result = prepare(api, path, operation, reason="operator reviewed failure")
    assert len(api.calls) == 1 and api.calls[0][0].endswith("/" + operation)
    assert result["next_action"] == "EXECUTE_SAVED_PLAN"
    assert result["action"]["gates"] and result["action"]["preflight_summary"]
    assert result["execute_argv"][-3:] == ["--state", str(path), "--execute"]
    saved = json.loads(path.read_text())
    assert not saved["execution_requested"] and saved["plan"] == result["action"]
    assert path.stat().st_mode & 0o777 == 0o600
    if operation == "retry":
        assert "no checkpoint is selected or changed" in result["retry_semantics"]
        assert api.calls[0][1]["data"] == {"max_gpu_hours": 2, "reason": "operator reviewed failure"}
    api.calls.clear()
    observed = training_operation(api, operation, path)
    assert observed["next_action"] == "EXECUTE_SAVED_PLAN"
    assert api.calls == [("/api/actions/" + ACTION_ID, {})]
    executed = training_operation(api, operation, path, execute=True)
    assert executed["action"]["execution"]["status"] == "EXECUTING"
    assert api.calls[-2:] == [("/api/actions/authorize", {"data": {"action_id": ACTION_ID,
            "note": "explicit CLI execution of the saved training plan"}}),
        ("/api/actions/execute", {"data": {"action_id": ACTION_ID, "confirmation": "EXECUTE " + ACTION_ID}})]
    assert json.loads(path.read_text())["execution_requested"]
    api.calls.clear()
    training_operation(api, operation, path, execute=True)
    assert api.calls == [("/api/actions/" + ACTION_ID, {})]


@pytest.mark.parametrize("failure", ["prepare", "authorize", "execute"])
@pytest.mark.parametrize("after_effect", [False, True])
def test_lost_reply_never_replays_scheduler_requests(tmp_path, failure, after_effect):
    api, path = API(), tmp_path / "plan.json"
    if failure != "prepare":
        prepare(api, path)
    endpoint = {"prepare": "/api/attempts/demo/trial%3A%3Aattempt-001/cancel",
                "authorize": "/api/actions/authorize", "execute": "/api/actions/execute"}[failure]
    api.fail, api.fail_after_effect = endpoint, after_effect
    with pytest.raises(ClientError, match="reply lost"):
        if failure == "prepare":
            prepare(api, path)
        else:
            training_operation(api, "cancel", path, execute=True)
    assert path.exists()
    api.fail = None
    api.calls.clear()
    if failure == "prepare":
        with pytest.raises(ClientError, match="preparation response is unknown"):
            training_operation(api, "cancel", path, execute=True)
        assert not api.calls
        review = training_operation(api, "cancel", path, action_id=ACTION_ID)
        assert review["action"]["action_id"] == ACTION_ID
        assert api.calls == [("/api/actions/" + ACTION_ID, {})]
    else:
        result = training_operation(api, "cancel", path, execute=True)
        assert not any(p == endpoint for p, _ in api.calls)
        if failure == "authorize" and after_effect:
            assert api.calls[-1][0] == "/api/actions/execute"
        else:
            assert api.calls == [("/api/actions/" + ACTION_ID, {})]
            assert result["next_action"] == "OBSERVE_ACTION"


@pytest.mark.parametrize("change", [{"scope": {"project": "other", "scope_type": "attempt", "object_id": "trial::attempt-001"}},
    {"operation": "SUBMIT_RUN"}, {"action_id": "action-" + "d" * 16}, {"run_id": "other"},
    {"attempt_id": "attempt-009"}, {"intent_digest": "sha256:" + "d" * 64},
    {"gate_bundle_digest": "sha256:" + "d" * 64}, {"intent_digest": "unsealed"},
    {"execution": None}, {"execution": {"status": "SURPRISE"}}])
def test_changed_action_is_rejected_before_authorization(tmp_path, change):
    api, path = API(), tmp_path / "plan.json"
    prepare(api, path)
    api.action.update(change)
    api.calls.clear()
    with pytest.raises(ClientError):
        training_operation(api, "cancel", path, execute=True)
    assert api.calls == [("/api/actions/" + ACTION_ID, {})]


@pytest.mark.parametrize("kwargs", [{"execute": True}, {"reconcile": True}, {"action_id": ACTION_ID},
    {"project": None}, {"attempt": "latest"}, {"max_gpu_hours": float("nan")},
    {"max_gpu_hours": float("inf")}, {"max_gpu_hours": 0}, {"max_gpu_hours": -1},
    {"max_gpu_hours": None}, {"new_attempt": "attempt-001"}, {"reason": "x" * 4001}])
def test_invalid_first_retry_has_no_http_or_state(tmp_path, kwargs):
    api, path = API("retry"), tmp_path / "plan.json"
    arguments = {"project": "demo", "run": "trial", "attempt": "attempt-001", "max_gpu_hours": 2}
    arguments.update(kwargs)
    with pytest.raises(ClientError):
        training_operation(api, "retry", path, **arguments)
    assert not api.calls and not path.exists()


@pytest.mark.parametrize("kwargs", [{"project": "other"}, {"run": "other"}, {"attempt": "attempt-002"},
    {"reason": "changed"}, {"max_gpu_hours": 3}, {"new_attempt": "attempt-009"}, {"action_id": "action-" + "d" * 16}])
def test_saved_scope_and_request_cannot_be_changed(tmp_path, kwargs):
    api, path = API("retry"), tmp_path / "plan.json"
    prepare(api, path, "retry")
    api.calls.clear()
    with pytest.raises(ClientError):
        training_operation(api, "retry", path, execute=True, **kwargs)
    assert not api.calls


@pytest.mark.parametrize("status", ["BLOCKED", "FAILED", "VERIFIED", "AUTHORIZED", "EXECUTING", "RECONCILE_REQUIRED"])
def test_existing_action_status_controls_execution_and_reconciliation(tmp_path, status):
    api, path = API(), tmp_path / "plan.json"
    prepare(api, path)
    api.action["execution"]["status"] = status
    api.calls.clear()
    result = training_operation(api, "cancel", path, execute=True)
    assert any(p == "/api/actions/execute" for p, _ in api.calls) == (status == "AUTHORIZED")
    assert not any(p == "/api/actions/authorize" for p, _ in api.calls)
    api.calls.clear()
    training_operation(api, "cancel", path, reconcile=True)
    expected = result["action"]["execution"]["status"] in {"EXECUTING", "RECONCILE_REQUIRED"}
    assert any(p == "/api/actions/reconcile" for p, _ in api.calls) == expected
    assert not any(p == "/api/actions/execute" for p, _ in api.calls)


def test_not_ready_plan_cannot_execute(tmp_path):
    api, path = API(action=snapshot(ready=False)), tmp_path / "plan.json"
    result = prepare(api, path)
    assert result["next_action"] == "INSPECT_ACTION" and "execute_argv" not in result
    api.calls.clear()
    training_operation(api, "cancel", path, execute=True)
    assert api.calls == [("/api/actions/" + ACTION_ID, {})]


def test_retry_exact_new_attempt_and_changed_target(tmp_path):
    api, path = API("retry", action=snapshot("retry", attempt_id="attempt-005")), tmp_path / "plan.json"
    prepare(api, path, "retry", new_attempt="attempt-005")
    assert api.calls[0][1]["data"]["new_attempt_id"] == "attempt-005"
    api.action["attempt_id"] = "attempt-006"
    with pytest.raises(ClientError, match="different Attempt"):
        training_operation(api, "retry", path, execute=True)


def test_retry_automatically_assigned_target_remains_pinned(tmp_path):
    api, path = API("retry"), tmp_path / "plan.json"
    prepare(api, path, "retry")
    api.action["attempt_id"] = "attempt-009"
    with pytest.raises(ClientError, match="changed after review"):
        training_operation(api, "retry", path, execute=True)


@pytest.mark.parametrize("requested", [None, True, float("nan"), float("inf"), -1, 3])
@pytest.mark.parametrize("approval", ["budget_cap", "review_exact"])
def test_retry_actual_resources_must_fit_explicit_cli_budget(tmp_path, requested, approval):
    api, path = API("retry"), tmp_path / "plan.json"
    api.action["preflight_summary"].update(requested_gpu_hours=requested, resource_approval=approval)
    with pytest.raises(ClientError, match="resources exceed or lack"):
        prepare(api, path, "retry")
    assert not any(p in {"/api/actions/authorize", "/api/actions/execute"} for p, _ in api.calls)
    assert json.loads(path.read_text())["plan"] is None


@pytest.mark.parametrize("cap", [None, True, float("nan"), float("inf"), 0, 3])
def test_retry_attached_action_cannot_raise_cli_budget_cap(tmp_path, cap):
    api, path = API("retry"), tmp_path / "plan.json"
    api.fail = "/api/attempts/demo/trial%3A%3Aattempt-001/retry"
    with pytest.raises(ClientError):
        prepare(api, path, "retry")
    api.fail = None
    api.action["preflight_summary"]["max_gpu_hours"] = cap
    with pytest.raises(ClientError, match="raises or lacks"):
        training_operation(api, "retry", path, action_id=ACTION_ID)
    assert json.loads(path.read_text())["plan"] is None


@pytest.mark.parametrize("approval,cap", [("budget_cap", 1.5), ("review_exact", None)])
def test_retry_allows_smaller_server_cap_or_exact_review_within_cli_budget(tmp_path, approval, cap):
    api, path = API("retry"), tmp_path / "plan.json"
    api.action["preflight_summary"].update(resource_approval=approval, max_gpu_hours=cap)
    result = prepare(api, path, "retry")
    assert result["next_action"] == "EXECUTE_SAVED_PLAN"


@pytest.mark.parametrize("cap", [None, True, 0, float("nan")])
def test_saved_retry_budget_cannot_be_invalid(tmp_path, cap):
    api, path = API("retry"), tmp_path / "plan.json"
    prepare(api, path, "retry")
    value = json.loads(path.read_text())
    value["request"]["max_gpu_hours"] = cap
    path.write_text(json.dumps(value))
    api.calls.clear()
    with pytest.raises(ClientError, match="invalid saved retry budget"):
        training_operation(api, "retry", path, execute=True)
    assert not api.calls


@pytest.mark.parametrize("case", ["server", "operation", "contract", "request", "marker", "bad-json"])
def test_wrong_or_malformed_saved_state_never_calls_http(tmp_path, case):
    api, path = API(), tmp_path / "plan.json"
    prepare(api, path)
    saved = json.loads(path.read_text())
    if case == "server": saved["server"] = "https://other.example"
    if case == "operation": saved["operation"] = "retry"
    if case == "contract": saved["contract"] = "different"
    if case == "request": saved["request"] = []
    if case == "marker": saved["execution_requested"] = "false"
    path.write_text("{" if case == "bad-json" else json.dumps(saved))
    api.calls.clear()
    with pytest.raises((ClientError, ValueError)):
        training_operation(api, "cancel", path, execute=True)
    assert not api.calls


@pytest.mark.parametrize("command", ["cancel", "retry"])
def test_cli_prepares_and_explicitly_executes_saved_action(tmp_path, monkeypatch, capsys, command):
    api, path = API(command), tmp_path / "plan.json"
    monkeypatch.setattr("ml_exp_client.cli.Client", lambda *args: api)
    args = [command, "--project", "demo", "--run", "trial", "--attempt", "attempt-001", "--state", str(path), "--json"]
    if command == "retry": args += ["--max-gpu-hours", "2"]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["next_action"] == "EXECUTE_SAVED_PLAN"
    assert main([command, "--state", str(path), "--execute", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["action"]["execution"]["status"] == "EXECUTING"


@pytest.mark.parametrize("command", ["logs", "metrics"])
def test_cli_read_flags_and_no_dimension_rewriting(monkeypatch, capsys, command):
    calls = []
    class Reader:
        def negotiate(self): return {}
        def call(self, path, **kwargs):
            calls.append((path, kwargs))
            return {"project": "demo", "run_id": "trial", "attempt_id": "attempt-001", "unit": "nats/token", "protocol": "validation", "variant": "test"}
    monkeypatch.setattr("ml_exp_client.cli.Client", lambda *args: Reader())
    args = [command, "--project", "demo", "--run", "trial", "--attempt", "attempt-001"]
    args += ["--stream", "stdout", "--lines", "2"] if command == "logs" else ["--keys", "loss", "--max-points", "10"]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out) == {"project": "demo", "run_id": "trial", "attempt_id": "attempt-001", "unit": "nats/token", "protocol": "validation", "variant": "test"}
    assert len(calls) == 1 and not calls[0][1]


def test_cli_execute_first_and_blocked_execution_report_error(tmp_path, monkeypatch, capsys):
    api, path = API(action=snapshot(ready=False)), tmp_path / "plan.json"
    monkeypatch.setattr("ml_exp_client.cli.Client", lambda *args: api)
    assert main(["cancel", "--state", str(path), "--execute"]) == 2
    assert "prepare and review" in capsys.readouterr().err
    prepare(api, path)
    assert main(["cancel", "--state", str(path), "--execute", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["next_action"] == "INSPECT_ACTION"
    with pytest.raises(SystemExit):
        parser().parse_args(["retry", "--state", str(path), "--execute", "--reconcile"])


@pytest.mark.parametrize("phase", ["prepare", "authorize", "execute"])
def test_failed_durable_marker_prevents_corresponding_http_mutation(tmp_path, monkeypatch, phase):
    from ml_exp_client.state import save as durable_save
    api, path = API(), tmp_path / "plan.json"
    if phase != "prepare":
        prepare(api, path)
    api.calls.clear()
    def fail_marker(path, value):
        failing = (phase == "prepare" or (phase == "authorize" and value["authorization_requested"])
                   or (phase == "execute" and value["execution_requested"]))
        if failing:
            raise OSError("fsync failed")
        durable_save(path, value)
    monkeypatch.setattr("ml_exp_client.operations.save", fail_marker)
    with pytest.raises(OSError, match="fsync failed"):
        if phase == "prepare":
            prepare(api, path)
        else:
            training_operation(api, "cancel", path, execute=True)
    forbidden = {"prepare": "/api/attempts/demo/trial%3A%3Aattempt-001/cancel",
                 "authorize": "/api/actions/authorize", "execute": "/api/actions/execute"}[phase]
    assert not any(p == forbidden for p, _ in api.calls)


@pytest.mark.parametrize("view", ["logs", "metrics"])
@pytest.mark.parametrize("field", ["project", "run_id", "attempt_id"])
def test_attempt_read_reply_must_match_exact_requested_identity(view, field):
    class Wrong:
        def call(self, path, **kwargs):
            value = {"project": "demo", "run_id": "trial", "attempt_id": "attempt-001"}
            value[field] = "other"
            return value
    with pytest.raises(ClientError, match="read response differs"):
        read_attempt(Wrong(), view, "demo", "trial", "attempt-001")
