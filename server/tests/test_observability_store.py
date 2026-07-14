"""Durability and concurrency contract for the observability outbox."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from ml_exp_server.observability_store import (
    ArchiveRejection, AttemptRef,
    CursorConflict,
    LeaseConflict,
    ObservabilityStore,
    OutboxRecord,
    SourceRef,
    stable_record_key,
)


ATTEMPT = AttemptRef("workspace-a", "elf", "run-a", "attempt-001")
SOURCE = SourceRef(ATTEMPT, "stdout")


def _record(
    source: SourceRef = SOURCE,
    *,
    generation: str = "dev:1",
    start: int = 0,
    end: int = 8,
    value: int = 1,
) -> OutboxRecord:
    return OutboxRecord(
        stable_record_key(
            source, generation=generation, start_offset=start,
            end_offset=end, kind="metric",
        ),
        "metric",
        {"loss": value, "step": value},
        observed_at=123.0 + value,
    )


def test_record_key_is_stable_and_bound_to_source_generation_and_range():
    first = _record().record_key
    assert first == _record(value=9).record_key
    assert first.startswith("obs-") and len(first) == 36
    assert first != _record(generation="dev:2").record_key
    assert first != _record(start=1).record_key
    assert first != _record(source=SourceRef(ATTEMPT, "stderr")).record_key
    with pytest.raises(ValueError, match="byte range"):
        stable_record_key(
            SOURCE, generation="g", start_offset=3, end_offset=2, kind="log",
        )


def test_enqueue_and_cursor_advance_are_atomic_and_target_independent(tmp_path):
    store = ObservabilityStore(tmp_path / "observability.sqlite")
    assert store.enqueue_and_advance(
        SOURCE, expected=None, generation="dev:1", byte_offset=8,
        records=[_record()], targets=["local", "cloud", "local"], now=10,
    ) == 2
    cursor = store.get_cursor(SOURCE)
    assert cursor is not None
    assert (cursor.generation, cursor.byte_offset, cursor.updated_at) == ("dev:1", 8, 10)

    local = store.claim("local", "local-worker", now=11)
    cloud = store.claim("cloud", "cloud-worker", now=11)
    assert [item.record_key for item in local] == [_record().record_key]
    assert [item.record_key for item in cloud] == [_record().record_key]
    store.acknowledge(local[0].id, "local-worker", now=12)
    store.retry(
        cloud[0].id, "cloud-worker", "cloud offline", now=12,
        base_delay=5, max_delay=20,
    )

    statuses = {item.target: item for item in store.statuses(attempt=ATTEMPT, now=13)}
    assert (statuses["local"].pending, statuses["local"].delivered) == (0, 1)
    assert (statuses["cloud"].pending, statuses["cloud"].delivered) == (1, 0)
    assert store.claim("cloud", "other", now=16) == []
    assert len(store.claim("cloud", "other", now=17)) == 1


def test_cursor_conflict_rolls_back_outbox_and_cursor(tmp_path):
    store = ObservabilityStore(tmp_path / "observability.sqlite")
    store.enqueue_and_advance(
        SOURCE, expected=None, generation="one", byte_offset=4,
        records=[_record(generation="one", end=4)], targets=["local"], now=1,
    )
    stale = store.get_cursor(SOURCE)
    assert stale is not None
    store.enqueue_and_advance(
        SOURCE, expected=stale, generation="one", byte_offset=8,
        records=[_record(generation="one", start=4, end=8)],
        targets=["local"], now=2,
    )
    with pytest.raises(CursorConflict):
        store.enqueue_and_advance(
            SOURCE, expected=stale, generation="one", byte_offset=12,
            records=[_record(generation="one", start=8, end=12)],
            targets=["local"], now=3,
        )

    current = store.get_cursor(SOURCE)
    assert current is not None and current.byte_offset == 8
    first = store.claim("local", "worker", now=4)
    assert len(first) == 1
    store.acknowledge(first[0].id, "worker", now=5)
    assert len(store.claim("local", "worker", now=6)) == 1


def test_ambiguous_replay_is_idempotent_when_cursor_is_reset_for_test(tmp_path):
    """The unique record identity prevents duplicate target deliveries."""
    store = ObservabilityStore(tmp_path / "observability.sqlite")
    assert store.enqueue_and_advance(
        SOURCE, expected=None, generation="g", byte_offset=8,
        records=[_record(generation="g")], targets=["local"], now=1,
    ) == 1
    cursor = store.get_cursor(SOURCE)
    assert cursor is not None
    assert store.enqueue_and_advance(
        SOURCE, expected=cursor, generation="g", byte_offset=8,
        records=[_record(generation="g")], targets=["local"], now=2,
    ) == 0
    assert len(store.claim("local", "worker", now=3)) == 1


def test_payload_must_be_finite_json_before_transaction(tmp_path):
    store = ObservabilityStore(tmp_path / "observability.sqlite")
    invalid = OutboxRecord("record", "metric", {"loss": float("nan")})
    with pytest.raises(ValueError, match="finite JSON"):
        store.enqueue_and_advance(
            SOURCE, expected=None, generation="g", byte_offset=4,
            records=[invalid], targets=["local"],
        )
    assert store.get_cursor(SOURCE) is None
    assert store.statuses() == []


def test_archive_summary_exposes_only_aggregate_rejection_reasons(tmp_path):
    store = ObservabilityStore(tmp_path / "observability.sqlite")
    store.enqueue_and_advance(
        SOURCE, expected=None, generation="g", byte_offset=4,
        records=[], targets=[],
        rejections=[
            ArchiveRejection("g", 0, 1, "secret_key"),
            ArchiveRejection("g", 1, 2, "secret_key"),
            ArchiveRejection("g", 2, 3, "nonfinite"),
        ], now=1,
    )
    summary = store.archive_summary()
    assert summary["rejected_records"] == 3
    assert summary["rejected_by_reason"] == {
        "nonfinite": 1, "secret_key": 2,
    }
    assert "source_key" not in summary

    # Replaying the same scan cannot inflate aggregate rejection counts.
    cursor = store.get_cursor(SOURCE)
    assert cursor is not None
    store.enqueue_and_advance(
        SOURCE, expected=cursor, generation="g", byte_offset=4,
        records=[], targets=[],
        rejections=[ArchiveRejection("g", 0, 1, "secret_key")], now=2,
    )
    assert store.archive_summary()["rejected_records"] == 3

    # A later clean batch must not erase an earlier identified rejection.
    cursor = store.get_cursor(SOURCE)
    assert cursor is not None
    store.enqueue_and_advance(
        SOURCE, expected=cursor, generation="g", byte_offset=8,
        records=[], targets=[], now=3,
    )
    assert store.archive_summary() == {
        "sources": 1,
        "degraded_sources": 1,
        "rejected_records": 3,
        "rejected_by_reason": {"nonfinite": 1, "secret_key": 2},
    }


def test_lease_expiry_retry_backoff_terminal_and_ack_ownership(tmp_path):
    store = ObservabilityStore(tmp_path / "observability.sqlite")
    records = [
        _record(start=offset, end=offset + 4, value=index)
        for index, offset in enumerate((0, 4), start=1)
    ]
    store.enqueue_and_advance(
        SOURCE, expected=None, generation="dev:1", byte_offset=8,
        records=records, targets=["local"], now=0,
    )
    first = store.claim("local", "worker-a", limit=1, lease_seconds=10, now=1)[0]
    # A higher sequence for this Attempt is blocked while the first is leased.
    assert store.claim("local", "worker-b", limit=2, now=2) == []
    with pytest.raises(LeaseConflict):
        store.acknowledge(first.id, "worker-b", now=3)
    reclaimed = store.claim("local", "worker-b", limit=2, now=11)
    assert [item.id for item in reclaimed] == [first.id]
    assert store.retry(
        first.id, "worker-b", "x" * 2000, now=12,
        base_delay=2, max_delay=5, max_attempts=2,
    ) is False
    assert store.claim("local", "worker-c", now=13) == []
    second_try = store.claim("local", "worker-c", now=14)[0]
    assert second_try.attempt_count == 1
    assert store.retry(
        first.id, "worker-c", "still down", now=15,
        base_delay=2, max_delay=5, max_attempts=2,
    ) is True
    status = store.statuses(attempt=ATTEMPT, now=16)[0]
    assert status.terminal == 1
    # Terminal failure is fail-closed: later W&B steps cannot leapfrog it.
    assert store.claim("local", "worker-d", now=16) == []


def test_target_state_and_outbox_survive_reopen(tmp_path):
    path = tmp_path / "observability.sqlite"
    with ObservabilityStore(path) as store:
        store.enqueue_and_advance(
            SOURCE, expected=None, generation="g", byte_offset=8,
            records=[_record(generation="g")], targets=["local"], now=1,
        )
        store.set_target_state(
            ATTEMPT, "local", "READY", dashboard_url="http://127.0.0.1:8080/r",
            error="old warning", now=2,
        )
    with ObservabilityStore(path) as reopened:
        cursor = reopened.get_cursor(SOURCE)
        assert cursor is not None and cursor.byte_offset == 8
        status = reopened.statuses(attempt=ATTEMPT)[0]
        assert status.state == "READY"
        assert status.dashboard_url == "http://127.0.0.1:8080/r"
        assert status.last_error == "old warning"
        assert len(reopened.claim("local", "worker", now=3)) == 1


def test_target_activation_is_idempotent_preserves_ready_and_rewinds_once(tmp_path):
    store = ObservabilityStore(tmp_path / "observability.sqlite")
    store.enqueue_and_advance(
        SOURCE, expected=None, generation="g", byte_offset=8,
        records=[_record(generation="g")], targets=[], now=1,
    )
    assert store.activate_target_and_rewind(ATTEMPT, "cloud", now=2) is True
    assert store.get_cursor(SOURCE) is None
    store.set_target_state(
        ATTEMPT, "cloud", "READY", dashboard_url="https://wandb.ai/team/p/runs/x", now=3,
    )
    assert store.activate_target_and_rewind(ATTEMPT, "cloud", now=4) is False
    status = store.statuses(attempt=ATTEMPT)[0]
    assert status.state == "READY"
    assert status.dashboard_url == "https://wandb.ai/team/p/runs/x"


def test_terminal_outbox_reopens_after_circuit_breaker_cooldown(tmp_path):
    store = ObservabilityStore(tmp_path / "observability.sqlite")
    store.enqueue_and_advance(
        SOURCE, expected=None, generation="g", byte_offset=4,
        records=[_record(generation="g")], targets=["local"], now=0,
    )
    item = store.claim("local", "worker", now=1)[0]
    assert store.retry(item.id, "worker", "down", now=2, max_attempts=1) is True
    assert store.revive_terminal("local", cooldown_seconds=10, now=11) == 0
    assert store.revive_terminal("local", cooldown_seconds=10, now=12) == 1
    assert store.claim("local", "worker", now=12)[0].id == item.id


def test_status_query_is_bounded_and_does_not_return_payload(tmp_path):
    store = ObservabilityStore(tmp_path / "observability.sqlite")
    for index in range(510):
        attempt = AttemptRef("w", "p", f"run-{index:03}", "a1")
        store.set_target_state(attempt, "local", "PENDING", now=float(index))
    statuses = store.statuses(limit=10_000)
    assert len(statuses) == 500
    assert statuses[0].attempt.run_id == "run-509"
    assert not hasattr(statuses[0], "payload")
    with pytest.raises(ValueError, match="positive integer"):
        store.statuses(limit=0)


def test_shared_connection_is_thread_safe(tmp_path):
    store = ObservabilityStore(tmp_path / "observability.sqlite")

    def enqueue(index: int) -> int:
        attempt = AttemptRef("w", "p", f"run-{index}", "a")
        source = SourceRef(attempt, "metrics")
        return store.enqueue_and_advance(
            source, expected=None, generation="g", byte_offset=4,
            records=[_record(source, generation="g", end=4, value=index)],
            targets=["local"], now=float(index),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(enqueue, range(40))) == 40
    assert len(store.claim("local", "worker", limit=100, now=100)) == 40


def test_validation_rejects_invalid_states_limits_and_names(tmp_path):
    store = ObservabilityStore(tmp_path / "observability.sqlite")
    with pytest.raises(ValueError, match="publication state"):
        store.set_target_state(ATTEMPT, "local", "UNKNOWN")
    with pytest.raises(ValueError, match="lease_seconds"):
        store.claim("local", "worker", lease_seconds=0)
    with pytest.raises(ValueError, match="target"):
        store.claim("", "worker")
