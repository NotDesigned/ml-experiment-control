"""Durable ActionStore command, CAS, and recovery edges."""

from __future__ import annotations

import json

import pytest

from ml_exp_server.actions import store as store_module
from ml_exp_server.actions.store import ActionStore
from ml_exp_server.schemas import OperationScope, OperationScopeType
from ml_exp_server.storage import DurableSnapshot, StorageError, TransitionConflict, atomic_json


def _scope() -> OperationScope:
    return OperationScope(
        project="demo", scope_type=OperationScopeType.PROJECT, object_id="demo",
    )


def _prepared(tmp_path):
    store = ActionStore(tmp_path / "actions")
    action_id = store.action_id(_scope(), "intent")
    store.save_plan({
        "action_id": action_id,
        "scope": _scope().model_dump(mode="json"),
        "ready": True,
        "operation": "SUBMIT_RUN",
    })
    return store, action_id


def test_execution_snapshot_without_plan_remains_empty(tmp_path):
    store = ActionStore(tmp_path / "actions")
    action_id = store.action_id(_scope(), "missing")
    assert store._execution_snapshot(action_id).value == {}


def test_execution_snapshot_rejects_embedded_revision_drift(tmp_path):
    store, action_id = _prepared(tmp_path)
    path = store.directory(action_id) / "execution.json"
    raw = json.loads(path.read_text())
    raw["revision"] = 99
    path.write_text(json.dumps(raw))
    with pytest.raises(StorageError, match="execution revision does not match"):
        store.execution(action_id)


@pytest.mark.parametrize("phase", ["prepare", "invalid"])
def test_action_commands_reject_unsupported_phase(tmp_path, phase):
    store, action_id = _prepared(tmp_path)
    with pytest.raises(ValueError, match="unsupported action command phase"):
        store.write_command(action_id, phase, {})
    with pytest.raises(ValueError, match="unsupported action command phase"):
        store.read_command(action_id, phase)


def test_action_command_is_idempotent_and_payload_bound(tmp_path):
    store, action_id = _prepared(tmp_path)
    payload = {"note": "reviewed"}
    first = store.write_command(action_id, "authorize", payload)
    second = store.write_command(action_id, "authorize", payload)

    assert second == first
    assert store.read_command(action_id, "authorize") == payload
    with pytest.raises(ValueError, match="already recorded"):
        store.write_command(action_id, "authorize", {"note": "changed"})


def test_legacy_command_without_event_id_gets_stable_journal_identity(tmp_path):
    store, action_id = _prepared(tmp_path)
    path = store.directory(action_id) / "commands" / "execute.json"
    atomic_json(path, {
        "action_id": action_id, "phase": "execute",
        "payload": {"confirmation": "yes"}, "created_at": "before-ids",
    })

    record = store.write_command(
        action_id, "execute", {"confirmation": "yes"},
    )

    assert "journal_event_id" not in record
    assert store.snapshot(action_id)["journal"][-1]["event"] == (
        "execute_command_recorded"
    )


def test_read_command_missing_or_non_mapping_payload(tmp_path):
    store, action_id = _prepared(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.read_command(action_id, "execute")

    path = store.directory(action_id) / "commands" / "execute.json"
    atomic_json(path, {"payload": "legacy"})
    assert store.read_command(action_id, "execute") == {}


def test_activity_errors_validate_phase_and_round_trip(tmp_path):
    store, action_id = _prepared(tmp_path)
    with pytest.raises(ValueError, match="unsupported action activity phase"):
        store.write_activity_error(action_id, "invalid", "boom", "test")
    store.write_activity_error(action_id, "reconcile", "x" * 1200, "test")
    error = store.activity_error(action_id, "reconcile")
    assert len(error["message"]) == 1000
    assert error["category"] == "test"


def test_set_execution_maps_transition_conflict(monkeypatch, tmp_path):
    store, action_id = _prepared(tmp_path)
    current = store.execution(action_id)
    state = store._execution_state(action_id)
    monkeypatch.setattr(store, "_execution_state", lambda _action_id: state)
    monkeypatch.setattr(
        state,
        "commit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            TransitionConflict("lost CAS")
        ),
    )
    with pytest.raises(RuntimeError, match="lost CAS"):
        store.set_execution(action_id, current, event="test")


def test_begin_execution_rejects_status_and_stale_revision(tmp_path):
    store, action_id = _prepared(tmp_path)
    with pytest.raises(RuntimeError, match="expected AUTHORIZED"):
        store.begin_execution(
            action_id, store.execution(action_id), intent_digest="sha256:intent",
        )

    prepared = store.execution(action_id)
    authorized = store.set_execution(
        action_id, {**prepared, "status": "AUTHORIZED"}, event="authorized",
    )["execution"]
    with pytest.raises(RuntimeError, match="expected revision"):
        store.begin_execution(
            action_id, {**authorized, "revision": 0},
            intent_digest="sha256:intent",
        )


def test_scope_and_global_listing_skip_broken_action_identity(tmp_path):
    store, _ = _prepared(tmp_path)
    broken = store.root / "action-broken"
    broken.mkdir()
    atomic_json(broken / "plan.json", {
        "scope": _scope().model_dump(mode="json"),
        "action_id": "invalid/path",
    })
    assert len(store.list_for_scope(_scope())) == 1
    assert len(store.list_all()) == 1
