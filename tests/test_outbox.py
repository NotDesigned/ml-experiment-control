import json

import pytest

from experiment_control.outbox import cancel_intent_path, execute_cancel_outbox


def execute(tmp_path, status_call, cancel_call):
    return execute_cancel_outbox(
        run_dir=tmp_path,
        project="project",
        run_id="run",
        attempt_id="attempt-001",
        backend="test-backend",
        backend_job_id="job-123",
        status_call=status_call,
        cancel_call=cancel_call,
        now=lambda: "2026-07-12T00:00:00Z",
    )


def test_cancel_outbox_records_intent_before_exact_cancel_and_verifies(tmp_path):
    calls = []

    def cancel():
        path = cancel_intent_path(tmp_path, "attempt-001")
        assert json.loads(path.read_text())["state"] == "REQUESTED"
        calls.append("cancel")
        return {"state": "CANCELLED", "backend_job_id": "job-123"}

    result = execute(tmp_path, lambda: {}, cancel)
    assert result["state"] == "CANCELLED"
    assert calls == ["cancel"]
    intent = json.loads(
        cancel_intent_path(tmp_path, "attempt-001").read_text()
    )
    assert intent["state"] == "VERIFIED"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "cancel_requested", "cancel_verified",
    ]


def test_verified_cancel_is_idempotent_without_backend_calls(tmp_path):
    execute(
        tmp_path,
        lambda: (_ for _ in ()).throw(AssertionError("status must not run")),
        lambda: {"state": "CANCELLED", "backend_job_id": "job-123"},
    )
    result = execute(
        tmp_path,
        lambda: (_ for _ in ()).throw(AssertionError("status must not run")),
        lambda: (_ for _ in ()).throw(AssertionError("cancel must not run")),
    )
    assert result["state"] == "CANCELLED"


def test_unresolved_cancel_reconciles_terminal_without_second_mutation(tmp_path):
    path = cancel_intent_path(tmp_path, "attempt-001")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "schema_version": 1,
        "state": "REQUESTED",
        "requested_at": "earlier",
        "project": "project",
        "run_id": "run",
        "attempt_id": "attempt-001",
        "backend": "test-backend",
        "backend_job_id": "job-123",
    }))
    result = execute(
        tmp_path,
        lambda: {"state": "CANCELLED", "backend_job_id": "job-123"},
        lambda: (_ for _ in ()).throw(AssertionError("cancel must not run")),
    )
    assert result["state"] == "CANCELLED"


def test_unresolved_nonterminal_cancel_fails_closed(tmp_path):
    path = cancel_intent_path(tmp_path, "attempt-001")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "schema_version": 1,
        "state": "REQUESTED",
        "requested_at": "earlier",
        "project": "project",
        "run_id": "run",
        "attempt_id": "attempt-001",
        "backend": "test-backend",
        "backend_job_id": "job-123",
    }))
    with pytest.raises(RuntimeError, match="do not issue a second cancel"):
        execute(
            tmp_path,
            lambda: {"state": "RUNNING", "backend_job_id": "job-123"},
            lambda: (_ for _ in ()).throw(AssertionError("cancel must not run")),
        )


def test_cancel_requires_explicit_attempt_identity(tmp_path):
    with pytest.raises(ValueError, match="explicit attempt identity"):
        cancel_intent_path(tmp_path, "")


def test_existing_cancel_intent_rejects_identity_drift(tmp_path):
    path = cancel_intent_path(tmp_path, "attempt-001")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "state": "REQUESTED",
        "project": "other",
        "run_id": "run",
        "attempt_id": "attempt-001",
        "backend": "test-backend",
        "backend_job_id": "job-123",
    }))
    with pytest.raises(ValueError, match="conflicts.*project"):
        execute(tmp_path, lambda: {}, lambda: {})


@pytest.mark.parametrize("phase", ["reconciliation", "cancel"])
def test_cancel_outbox_rejects_a_different_scheduler_identity(tmp_path, phase):
    if phase == "reconciliation":
        path = cancel_intent_path(tmp_path, "attempt-001")
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "state": "REQUESTED",
            "project": "project",
            "run_id": "run",
            "attempt_id": "attempt-001",
            "backend": "test-backend",
            "backend_job_id": "job-123",
        }))
        status_call = lambda: {"state": "CANCELLED", "backend_job_id": "job-other"}
        cancel_call = lambda: (_ for _ in ()).throw(AssertionError("must not cancel"))
    else:
        status_call = lambda: {}
        cancel_call = lambda: {"state": "CANCELLED", "backend_job_id": "job-other"}

    with pytest.raises(RuntimeError, match="different backend job identity"):
        execute(tmp_path, status_call, cancel_call)
