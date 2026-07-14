"""Migration, validation, and rollback edges for the observability store."""

from __future__ import annotations

import sqlite3

import pytest

from ml_exp_server.observability_store import (
    ArchiveRejection,
    AttemptRef,
    LeaseConflict,
    ObservabilityStore,
    OutboxRecord,
    SourceRef,
)


ATTEMPT = AttemptRef("workspace", "demo", "run-a", "attempt-001")
SOURCE = SourceRef(ATTEMPT, "metrics")
RECORD = OutboxRecord("record", "metrics", {"step": 1})


def test_store_migrates_legacy_cursor_columns(tmp_path):
    path = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE source_cursors ("
        "workspace_id TEXT, project TEXT, run_id TEXT, attempt_id TEXT, "
        "source_key TEXT, generation TEXT, byte_offset INTEGER, updated_at REAL, "
        "PRIMARY KEY (workspace_id, project, run_id, attempt_id, source_key))"
    )
    connection.commit()
    connection.close()

    store = ObservabilityStore(path)
    columns = {
        row[1] for row in store._conn.execute(
            "PRAGMA table_info(source_cursors)"
        ).fetchall()
    }
    assert {"file_identity", "anchor_start", "anchor_digest"} <= columns
    store.close()
    store.close()


@pytest.mark.parametrize(("updates", "message"), [
    ({"generation": ""}, "generation"),
    ({"byte_offset": -1}, "byte_offset"),
    ({"byte_offset": 1, "anchor_start": 2}, "cursor anchor"),
])
def test_enqueue_rejects_invalid_cursor(updates, message, tmp_path):
    values = {
        "expected": None, "generation": "g", "byte_offset": 1,
        "records": [], "targets": [],
    }
    values.update(updates)
    with pytest.raises(ValueError, match=message):
        ObservabilityStore(tmp_path / "store.sqlite").enqueue_and_advance(
            SOURCE, **values,
        )


def test_enqueue_rejects_invalid_archive_rejection(tmp_path):
    store = ObservabilityStore(tmp_path / "store.sqlite")
    with pytest.raises(ValueError, match="rejection identity"):
        store.enqueue_and_advance(
            SOURCE, expected=None, generation="g", byte_offset=1,
            records=[], targets=[],
            rejections=[ArchiveRejection("", 1, 0, "invalid")],
        )


def test_archive_error_is_bounded_and_visible_in_summary(tmp_path):
    store = ObservabilityStore(tmp_path / "store.sqlite")
    store.record_archive_error(SOURCE, "x" * 1200, now=1)
    summary = store.archive_summary()
    assert summary["sources"] == 1
    assert summary["degraded_sources"] == 1


def test_backfill_target_inserts_records_and_rewinds_cursor(tmp_path):
    store = ObservabilityStore(tmp_path / "store.sqlite")
    store.enqueue_and_advance(
        SOURCE, expected=None, generation="g", byte_offset=1,
        records=[], targets=[], now=1,
    )
    store.backfill_target(ATTEMPT, "cloud", records=[RECORD], now=2)
    assert store.get_cursor(SOURCE) is None
    assert store.claim("cloud", "worker", now=3)[0].record_key == "record"


def test_retry_and_revive_validate_bounds_and_lease(tmp_path):
    store = ObservabilityStore(tmp_path / "store.sqlite")
    with pytest.raises(ValueError, match="retry delay"):
        store.retry(1, "worker", "error", base_delay=-1)
    with pytest.raises(ValueError, match="max_attempts"):
        store.retry(1, "worker", "error", max_attempts=0)
    with pytest.raises(LeaseConflict):
        store.retry(1, "worker", "error")
    with pytest.raises(ValueError, match="cooldown_seconds"):
        store.revive_terminal("local", cooldown_seconds=float("inf"))


@pytest.mark.parametrize("url", [
    "ftp://example.com", "https://user@example.com", "https://example.com?token=x",
])
def test_target_state_rejects_unsafe_dashboard_url(tmp_path, url):
    store = ObservabilityStore(tmp_path / "store.sqlite")
    with pytest.raises(ValueError, match="credential-free"):
        store.set_target_state(ATTEMPT, "local", "READY", dashboard_url=url)


def test_target_state_rejects_invalid_dashboard_port(tmp_path):
    store = ObservabilityStore(tmp_path / "store.sqlite")
    with pytest.raises(ValueError, match="invalid port"):
        store.set_target_state(
            ATTEMPT, "local", "READY", dashboard_url="https://example.com:bad",
        )


class BrokenConnection:
    def __init__(self):
        self.rolled_back = False

    def execute(self, *_args, **_kwargs):
        raise sqlite3.OperationalError("database unavailable")

    def rollback(self):
        self.rolled_back = True


@pytest.mark.parametrize("operation", [
    lambda store: store.activate_target_and_rewind(ATTEMPT, "local"),
    lambda store: store.backfill_target(ATTEMPT, "local"),
    lambda store: store.claim("local", "worker"),
    lambda store: store.retry(1, "worker", "error"),
    lambda store: store.revive_terminal("local"),
])
def test_transactional_operations_roll_back_database_errors(tmp_path, operation):
    store = ObservabilityStore(tmp_path / "store.sqlite")
    real = store._conn
    broken = BrokenConnection()
    store._conn = broken
    try:
        with pytest.raises(sqlite3.OperationalError):
            operation(store)
        assert broken.rolled_back is True
    finally:
        store._conn = real
        store.close()
