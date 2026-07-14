"""Shared durable transition/CAS behavior used by server-owned stores."""

from __future__ import annotations

import json

import pytest

from ml_exp_server import storage as storage_module
from ml_exp_server.storage import DurableJsonState, StorageError, TransitionConflict


def test_transition_is_cas_versioned_and_repairs_a_missing_journal(tmp_path):
    path = tmp_path / "state.json"
    journal = tmp_path / "events.jsonl"
    state = DurableJsonState(path, journal)

    first = state.commit(
        {"status": "AUTHORIZED"}, expected_revision=0,
        event={"event": "authorized", "at": "now"},
    )
    assert first.revision == 1
    assert state.snapshot({}).value == {"status": "AUTHORIZED"}
    assert json.loads(journal.read_text().splitlines()[-1])["revision"] == 1

    with pytest.raises(TransitionConflict, match="expected revision 0"):
        state.commit(
            {"status": "EXECUTING"}, expected_revision=0,
            event={"event": "started", "at": "later"},
        )

    # Model a crash after the atomic state replace but before its JSONL append.
    journal.unlink()
    current = state.snapshot({})
    state.repair_journal(current)
    repaired = json.loads(journal.read_text().strip())
    assert repaired["transition_id"] == current.last_transition["transition_id"]
    assert repaired["event"] == "authorized"


def test_transition_metadata_corruption_fails_closed(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"_durability": {"revision": "bad"}}\n', encoding="utf-8")
    state = DurableJsonState(path, tmp_path / "events.jsonl")
    with pytest.raises(StorageError, match="revision is invalid"):
        state.snapshot({})

    path.write_text(
        json.dumps({
            "_durability": {
                "revision": 2,
                "last_transition": {
                    "transition_id": "transition-2", "revision": 1,
                },
            },
        }) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(StorageError, match="does not match durable state"):
        state.snapshot({})


def test_committed_state_survives_interrupted_journal_append(monkeypatch, tmp_path):
    path = tmp_path / "state.json"
    journal = tmp_path / "events.jsonl"
    state = DurableJsonState(path, journal)
    real_append = storage_module.append_jsonl
    monkeypatch.setattr(
        storage_module, "append_jsonl",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk unavailable")),
    )

    committed = state.commit(
        {"status": "EXECUTING"},
        event={"event": "started"}, expected_revision=0,
    )
    assert committed.journal_pending is True
    assert state.snapshot({}).value["status"] == "EXECUTING"

    monkeypatch.setattr(storage_module, "append_jsonl", real_append)
    state.repair_journal(state.snapshot({}))
    assert json.loads(journal.read_text())["event"] == "started"


def test_plain_event_repairs_pending_transition_before_append(monkeypatch, tmp_path):
    path = tmp_path / "state.json"
    journal = tmp_path / "events.jsonl"
    state = DurableJsonState(path, journal)
    real_append = storage_module.append_jsonl
    monkeypatch.setattr(
        storage_module, "append_jsonl",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("crash")),
    )
    committed = state.commit(
        {"status": "PREPARED"}, event={"event": "action_prepared"},
        expected_revision=0,
    )
    assert committed.journal_pending is True

    monkeypatch.setattr(storage_module, "append_jsonl", real_append)
    state.append_event(
        {"event": "authorize_command_recorded"}, event_id="command-1",
    )
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "action_prepared", "authorize_command_recorded",
    ]
    assert events[1]["state_revision"] == 1


def test_partial_jsonl_tail_is_truncated_before_transition_replay(
    monkeypatch, tmp_path,
):
    path = tmp_path / "state.json"
    journal = tmp_path / "events.jsonl"
    state = DurableJsonState(path, journal)
    first = state.commit(
        {"status": "PREPARED"}, event={"event": "prepared"},
        expected_revision=0,
    )
    real_append = storage_module.append_jsonl
    monkeypatch.setattr(
        storage_module, "append_jsonl",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("crash")),
    )
    second = state.commit(
        {"status": "AUTHORIZED"}, event={"event": "authorized"},
        expected_revision=first.revision,
    )
    assert second.journal_pending is True
    with journal.open("ab") as stream:
        stream.write(b'{"event":"truncated"')

    monkeypatch.setattr(storage_module, "append_jsonl", real_append)
    state.repair_journal(state.snapshot({}))
    third = state.commit(
        {"status": "EXECUTING"}, event={"event": "started"},
        expected_revision=second.revision,
    )
    assert third.revision == 3
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    transitions = [event for event in events if event.get("transition_id")]
    assert [event["revision"] for event in transitions] == [1, 2, 3]


def test_complete_jsonl_corruption_fails_closed_before_next_commit(tmp_path):
    path = tmp_path / "state.json"
    journal = tmp_path / "events.jsonl"
    state = DurableJsonState(path, journal)
    first = state.commit(
        {"status": "PREPARED"}, event={"event": "prepared"},
        expected_revision=0,
    )
    second = state.commit(
        {"status": "AUTHORIZED"}, event={"event": "authorized"},
        expected_revision=first.revision,
    )
    lines = journal.read_text(encoding="utf-8").splitlines()
    journal.write_text(
        "not-json\n" + lines[1] + "\n", encoding="utf-8",
    )

    with pytest.raises(StorageError, match="invalid complete record"):
        state.commit(
            {"status": "EXECUTING"}, event={"event": "started"},
            expected_revision=second.revision,
        )
    assert state.snapshot({}).revision == 2


def test_transition_revision_gap_and_complete_non_mapping_tail_fail_closed(tmp_path):
    path = tmp_path / "state.json"
    journal = tmp_path / "events.jsonl"
    state = DurableJsonState(path, journal)
    first = state.commit(
        {"status": "PREPARED"}, event={"event": "prepared"},
        expected_revision=0,
    )
    state.commit(
        {"status": "AUTHORIZED"}, event={"event": "authorized"},
        expected_revision=first.revision,
    )
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    events[1]["revision"] = 3
    journal.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8",
    )
    with pytest.raises(StorageError, match="expected 2"):
        state.repair_journal(state.snapshot({}))

    journal.write_text("[]", encoding="utf-8")
    with pytest.raises(StorageError, match="tail is not a mapping"):
        state.repair_journal(state.snapshot({}))


def test_journal_divergence_and_missing_history_fail_closed(tmp_path):
    path = tmp_path / "state.json"
    journal = tmp_path / "events.jsonl"
    state = DurableJsonState(path, journal)
    first = state.commit(
        {"status": "PREPARED"}, event={"event": "prepared"},
        expected_revision=0,
    )
    state.commit(
        {"status": "AUTHORIZED"}, event={"event": "authorized"},
        expected_revision=first.revision,
    )
    snapshot = state.snapshot({})
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    events[-1]["event"] = "tampered"
    journal.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8",
    )
    with pytest.raises(StorageError, match="disagrees with authoritative state"):
        state.repair_journal(snapshot)

    journal.unlink()
    with pytest.raises(StorageError, match="history is missing"):
        state.repair_journal(snapshot)
