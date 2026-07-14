"""Durable cursors and publication outbox for observability mirrors.

Backend files remain canonical.  This database only records how far the daemon
has archived them and which sanitized records still need to be mirrored to an
observability target.  A record is queued once per target so Local and Cloud
publishers can fail and recover independently.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit


_SCHEMA = """
CREATE TABLE IF NOT EXISTS source_cursors (
    workspace_id TEXT NOT NULL,
    project TEXT NOT NULL,
    run_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    source_key TEXT NOT NULL,
    generation TEXT NOT NULL,
    byte_offset INTEGER NOT NULL CHECK (byte_offset >= 0),
    file_identity TEXT NOT NULL DEFAULT '',
    anchor_start INTEGER NOT NULL DEFAULT 0 CHECK (anchor_start >= 0),
    anchor_digest TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL,
    PRIMARY KEY (workspace_id, project, run_id, attempt_id, source_key)
);
CREATE TABLE IF NOT EXISTS publication_targets (
    workspace_id TEXT NOT NULL,
    project TEXT NOT NULL,
    run_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    target TEXT NOT NULL,
    state TEXT NOT NULL,
    dashboard_url TEXT,
    last_error TEXT,
    updated_at REAL NOT NULL,
    PRIMARY KEY (workspace_id, project, run_id, attempt_id, target)
);
CREATE TABLE IF NOT EXISTS archive_source_status (
    workspace_id TEXT NOT NULL,
    project TEXT NOT NULL,
    run_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    source_key TEXT NOT NULL,
    state TEXT NOT NULL,
    rejected_records INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at REAL NOT NULL,
    PRIMARY KEY (workspace_id, project, run_id, attempt_id, source_key)
);
CREATE TABLE IF NOT EXISTS archive_rejections (
    workspace_id TEXT NOT NULL,
    project TEXT NOT NULL,
    run_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    source_key TEXT NOT NULL,
    generation TEXT NOT NULL,
    byte_start INTEGER NOT NULL CHECK (byte_start >= 0),
    byte_end INTEGER NOT NULL CHECK (byte_end >= byte_start),
    reason TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (
        workspace_id, project, run_id, attempt_id, source_key,
        generation, byte_start, byte_end, reason
    )
);
CREATE TABLE IF NOT EXISTS publication_outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL,
    project TEXT NOT NULL,
    run_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    target TEXT NOT NULL,
    record_key TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    observed_at REAL,
    created_at REAL NOT NULL,
    available_at REAL NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    lease_owner TEXT,
    lease_until REAL,
    delivered_at REAL,
    terminal_at REAL,
    last_error TEXT,
    UNIQUE (workspace_id, project, run_id, attempt_id, target, record_key)
);
CREATE INDEX IF NOT EXISTS publication_outbox_claim
ON publication_outbox (target, delivered_at, terminal_at, available_at, lease_until, id);
CREATE INDEX IF NOT EXISTS publication_outbox_attempt
ON publication_outbox (workspace_id, project, run_id, attempt_id, target, id);
"""

_TARGET_STATES = {
    "DISABLED", "UNAVAILABLE", "PENDING", "SYNCING",
    "READY", "DEGRADED", "FAILED",
}
_MAX_ERROR_LENGTH = 1000
_MAX_QUERY_LIMIT = 500


class CursorConflict(RuntimeError):
    """The source cursor changed after it was read by the caller."""


class LeaseConflict(RuntimeError):
    """An outbox item is not leased by the publisher attempting to finish it."""


@dataclass(frozen=True)
class AttemptRef:
    workspace_id: str
    project: str
    run_id: str
    attempt_id: str

    def values(self) -> tuple[str, str, str, str]:
        return (self.workspace_id, self.project, self.run_id, self.attempt_id)


@dataclass(frozen=True)
class SourceRef:
    attempt: AttemptRef
    source_key: str


@dataclass(frozen=True)
class SourceCursor:
    generation: str
    byte_offset: int
    updated_at: float
    file_identity: str = ""
    anchor_start: int = 0
    anchor_digest: str = ""


@dataclass(frozen=True)
class OutboxRecord:
    record_key: str
    kind: str
    payload: Mapping[str, Any]
    observed_at: Optional[float] = None


@dataclass(frozen=True)
class ArchiveRejection:
    generation: str
    byte_start: int
    byte_end: int
    reason: str


@dataclass(frozen=True)
class OutboxItem:
    id: int
    attempt: AttemptRef
    target: str
    record_key: str
    kind: str
    payload: dict[str, Any]
    observed_at: Optional[float]
    created_at: float
    available_at: float
    attempt_count: int
    lease_owner: Optional[str]
    lease_until: Optional[float]


@dataclass(frozen=True)
class TargetStatus:
    attempt: AttemptRef
    target: str
    state: str
    dashboard_url: Optional[str]
    last_error: Optional[str]
    updated_at: float
    pending: int
    leased: int
    delivered: int
    terminal: int


def stable_record_key(
    source: SourceRef,
    *,
    generation: str,
    start_offset: int,
    end_offset: int,
    kind: str,
) -> str:
    """Return the stable identity of one source record.

    Payload content is deliberately excluded: identity follows the immutable
    source byte range.  A rewritten file must receive a different generation.
    """
    if start_offset < 0 or end_offset < start_offset:
        raise ValueError("invalid source byte range")
    identity = [
        *source.attempt.values(), source.source_key, generation,
        str(start_offset), str(end_offset), kind,
    ]
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return "obs-" + digest[:32]


class ObservabilityStore:
    """Thread-safe SQLite store for archive cursors and publisher work."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(db_path), check_same_thread=False, timeout=30.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=30000")
        had_rejection_identity = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='archive_rejections'"
        ).fetchone() is not None
        self._conn.executescript(_SCHEMA)
        if not had_rejection_identity:
            # Legacy totals lacked record identities and inflated on cursor
            # rewind. Reset them once; subsequent counts are replay-idempotent.
            self._conn.execute(
                "UPDATE archive_source_status SET rejected_records=0, "
                "last_error=CASE WHEN last_error='RejectedRecords' THEN NULL ELSE last_error END"
            )
        columns = {
            row[1] for row in self._conn.execute(
                "PRAGMA table_info(source_cursors)"
            ).fetchall()
        }
        for name, declaration in (
            ("file_identity", "TEXT NOT NULL DEFAULT ''"),
            ("anchor_start", "INTEGER NOT NULL DEFAULT 0"),
            ("anchor_digest", "TEXT NOT NULL DEFAULT ''"),
        ):
            if name not in columns:
                self._conn.execute(
                    f"ALTER TABLE source_cursors ADD COLUMN {name} {declaration}"
                )
        self._conn.commit()
        self._closed = False

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True

    def __enter__(self) -> "ObservabilityStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def get_cursor(self, source: SourceRef) -> Optional[SourceCursor]:
        with self._lock:
            row = self._conn.execute(
                "SELECT generation, byte_offset, updated_at, file_identity, "
                "anchor_start, anchor_digest FROM source_cursors "
                "WHERE workspace_id=? AND project=? AND run_id=? AND attempt_id=? "
                "AND source_key=?",
                (*source.attempt.values(), source.source_key),
            ).fetchone()
        if row is None:
            return None
        return SourceCursor(
            generation=row["generation"],
            byte_offset=row["byte_offset"],
            updated_at=row["updated_at"],
            file_identity=row["file_identity"],
            anchor_start=row["anchor_start"],
            anchor_digest=row["anchor_digest"],
        )

    def enqueue_and_advance(
        self,
        source: SourceRef,
        *,
        expected: Optional[SourceCursor],
        generation: str,
        byte_offset: int,
        file_identity: str = "",
        anchor_start: int = 0,
        anchor_digest: str = "",
        records: Iterable[OutboxRecord],
        targets: Sequence[str],
        rejections: Sequence[ArchiveRejection] = (),
        now: Optional[float] = None,
    ) -> int:
        """Atomically queue records for every target and advance a source cursor.

        ``expected`` provides optimistic concurrency.  Pass ``None`` only for a
        source that has no stored cursor.  Duplicate record keys are ignored,
        making a caller retry after an ambiguous commit safe.
        """
        if not generation:
            raise ValueError("generation must not be empty")
        if byte_offset < 0:
            raise ValueError("byte_offset must be non-negative")
        if anchor_start < 0 or anchor_start > byte_offset:
            raise ValueError("invalid cursor anchor")
        normalized_targets = tuple(dict.fromkeys(_validate_name(v, "target") for v in targets))
        prepared = [self._prepare_record(record) for record in records]
        timestamp = time.time() if now is None else now
        prepared_rejections = []
        for rejection in rejections:
            if (
                not rejection.generation or rejection.byte_start < 0
                or rejection.byte_end < rejection.byte_start
            ):
                raise ValueError("invalid archive rejection identity")
            prepared_rejections.append((
                rejection.generation, rejection.byte_start, rejection.byte_end,
                _validate_name(rejection.reason, "rejection reason"),
            ))
        inserted = 0
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                current = self._conn.execute(
                    "SELECT generation, byte_offset, updated_at, file_identity, "
                    "anchor_start, anchor_digest FROM source_cursors "
                    "WHERE workspace_id=? AND project=? AND run_id=? AND attempt_id=? "
                    "AND source_key=?",
                    (*source.attempt.values(), source.source_key),
                ).fetchone()
                if not _cursor_matches(current, expected):
                    raise CursorConflict("source cursor changed before commit")
                for target in normalized_targets:
                    self._conn.execute(
                        "INSERT INTO publication_targets "
                        "(workspace_id, project, run_id, attempt_id, target, state, updated_at) "
                        "VALUES (?,?,?,?,?,'PENDING',?) "
                        "ON CONFLICT(workspace_id, project, run_id, attempt_id, target) "
                        "DO NOTHING",
                        (*source.attempt.values(), target, timestamp),
                    )
                    for record, payload_json in prepared:
                        result = self._conn.execute(
                            "INSERT OR IGNORE INTO publication_outbox "
                            "(workspace_id, project, run_id, attempt_id, target, "
                            "record_key, kind, payload_json, observed_at, created_at, available_at) "
                            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                            (
                                *source.attempt.values(), target, record.record_key,
                                record.kind, payload_json, record.observed_at,
                                timestamp, timestamp,
                            ),
                        )
                        inserted += result.rowcount
                self._conn.execute(
                    "INSERT INTO source_cursors "
                    "(workspace_id, project, run_id, attempt_id, source_key, generation, "
                    "byte_offset, file_identity, anchor_start, anchor_digest, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(workspace_id, project, run_id, attempt_id, source_key) "
                    "DO UPDATE SET generation=excluded.generation, "
                    "byte_offset=excluded.byte_offset, "
                    "file_identity=excluded.file_identity, "
                    "anchor_start=excluded.anchor_start, "
                    "anchor_digest=excluded.anchor_digest, "
                    "updated_at=excluded.updated_at",
                    (
                        *source.attempt.values(), source.source_key, generation,
                        byte_offset, file_identity, anchor_start, anchor_digest,
                        timestamp,
                    ),
                )
                for rejection_generation, byte_start, byte_end, reason in prepared_rejections:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO archive_rejections "
                        "(workspace_id, project, run_id, attempt_id, source_key, generation, "
                        "byte_start, byte_end, reason, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (*source.attempt.values(), source.source_key, rejection_generation,
                         byte_start, byte_end, reason, timestamp),
                    )
                rejected_total = int(self._conn.execute(
                    "SELECT COUNT(*) AS count FROM archive_rejections WHERE "
                    "workspace_id=? AND project=? AND run_id=? AND attempt_id=? AND source_key=?",
                    (*source.attempt.values(), source.source_key),
                ).fetchone()["count"])
                self._conn.execute(
                    "INSERT INTO archive_source_status "
                    "(workspace_id, project, run_id, attempt_id, source_key, state, "
                    "rejected_records, last_error, updated_at) VALUES (?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(workspace_id, project, run_id, attempt_id, source_key) "
                    "DO UPDATE SET state=excluded.state, "
                    "rejected_records=excluded.rejected_records, "
                    "last_error=excluded.last_error, updated_at=excluded.updated_at",
                    (
                        *source.attempt.values(), source.source_key,
                        "DEGRADED" if rejected_total else "READY", rejected_total,
                        "RejectedRecords" if rejected_total else None, timestamp,
                    ),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return inserted

    def record_archive_error(
        self, source: SourceRef, error_class: str, *, now: Optional[float] = None,
    ) -> None:
        timestamp = time.time() if now is None else now
        with self._lock:
            self._conn.execute(
                "INSERT INTO archive_source_status "
                "(workspace_id, project, run_id, attempt_id, source_key, state, "
                "rejected_records, last_error, updated_at) VALUES (?,?,?,?,?,'DEGRADED',0,?,?) "
                "ON CONFLICT(workspace_id, project, run_id, attempt_id, source_key) "
                "DO UPDATE SET state='DEGRADED', last_error=excluded.last_error, "
                "updated_at=excluded.updated_at",
                (*source.attempt.values(), source.source_key,
                 _bounded_error(error_class), timestamp),
            )
            self._conn.commit()

    def archive_summary(self) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS sources, "
                "COALESCE(SUM(CASE WHEN state='DEGRADED' THEN 1 ELSE 0 END),0) AS degraded "
                "FROM archive_source_status"
            ).fetchone()
            rejected = self._conn.execute(
                "SELECT COUNT(*) AS rejected FROM archive_rejections"
            ).fetchone()
            reasons = self._conn.execute(
                "SELECT reason, COUNT(*) AS rejected "
                "FROM archive_rejections GROUP BY reason ORDER BY reason"
            ).fetchall()
        return {
            "sources": int(row["sources"]),
            "degraded_sources": int(row["degraded"]),
            "rejected_records": int(rejected["rejected"]),
            "rejected_by_reason": {
                item["reason"]: int(item["rejected"]) for item in reasons
            },
        }

    def set_target_state(
        self,
        attempt: AttemptRef,
        target: str,
        state: str,
        *,
        dashboard_url: Optional[str] = None,
        error: Optional[str] = None,
        now: Optional[float] = None,
    ) -> None:
        target = _validate_name(target, "target")
        state = state.upper()
        if state not in _TARGET_STATES:
            raise ValueError(f"invalid publication state: {state}")
        dashboard_url = _safe_dashboard_url(dashboard_url)
        timestamp = time.time() if now is None else now
        with self._lock:
            self._conn.execute(
                "INSERT INTO publication_targets "
                "(workspace_id, project, run_id, attempt_id, target, state, dashboard_url, "
                "last_error, updated_at) VALUES (?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(workspace_id, project, run_id, attempt_id, target) "
                "DO UPDATE SET state=excluded.state, dashboard_url=excluded.dashboard_url, "
                "last_error=excluded.last_error, updated_at=excluded.updated_at",
                (
                    *attempt.values(), target, state, dashboard_url,
                    _bounded_error(error), timestamp,
                ),
            )
            self._conn.commit()

    def activate_target_and_rewind(
        self, attempt: AttemptRef, target: str, *, now: Optional[float] = None,
    ) -> bool:
        """Activate a target once and rewind sources for complete backfill.

        Existing target state and dashboard URLs are preserved. Cursor rewind
        occurs in the same transaction as first activation, so replay after a
        crash is both idempotent and gap-free; record keys deduplicate mirrors
        that were already delivered to another target.
        """
        target = _validate_name(target, "target")
        timestamp = time.time() if now is None else now
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                inserted = self._conn.execute(
                    "INSERT OR IGNORE INTO publication_targets "
                    "(workspace_id, project, run_id, attempt_id, target, state, "
                    "dashboard_url, last_error, updated_at) VALUES (?,?,?,?,?,'PENDING',NULL,NULL,?)",
                    (*attempt.values(), target, timestamp),
                ).rowcount
                if inserted:
                    self._conn.execute(
                        "DELETE FROM source_cursors WHERE workspace_id=? AND project=? "
                        "AND run_id=? AND attempt_id=?",
                        attempt.values(),
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return bool(inserted)

    def backfill_target(
        self, attempt: AttemptRef, target: str, *, now: Optional[float] = None,
    ) -> None:
        """Explicitly request a complete idempotent replay for one target."""
        target = _validate_name(target, "target")
        timestamp = time.time() if now is None else now
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute(
                    "INSERT INTO publication_targets "
                    "(workspace_id, project, run_id, attempt_id, target, state, "
                    "dashboard_url, last_error, updated_at) "
                    "VALUES (?,?,?,?,?,'PENDING',NULL,NULL,?) "
                    "ON CONFLICT(workspace_id, project, run_id, attempt_id, target) "
                    "DO UPDATE SET state='PENDING', last_error=NULL, updated_at=excluded.updated_at",
                    (*attempt.values(), target, timestamp),
                )
                self._conn.execute(
                    "DELETE FROM source_cursors WHERE workspace_id=? AND project=? "
                    "AND run_id=? AND attempt_id=?",
                    attempt.values(),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def claim(
        self,
        target: str,
        worker_id: str,
        *,
        limit: int = 100,
        lease_seconds: float = 60.0,
        now: Optional[float] = None,
    ) -> list[OutboxItem]:
        """Lease available target work in FIFO order, including expired leases."""
        target = _validate_name(target, "target")
        worker_id = _validate_name(worker_id, "worker_id")
        bounded_limit = _bounded_limit(limit)
        if lease_seconds <= 0 or not math.isfinite(lease_seconds):
            raise ValueError("lease_seconds must be finite and positive")
        timestamp = time.time() if now is None else now
        lease_until = timestamp + lease_seconds
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                ids = [row["id"] for row in self._conn.execute(
                    "SELECT o.id FROM publication_outbox o WHERE o.target=? "
                    "AND o.delivered_at IS NULL AND o.terminal_at IS NULL "
                    "AND o.available_at<=? AND (o.lease_until IS NULL OR o.lease_until<=?) "
                    "AND NOT EXISTS (SELECT 1 FROM publication_outbox prior WHERE "
                    "prior.workspace_id=o.workspace_id AND prior.project=o.project "
                    "AND prior.run_id=o.run_id AND prior.attempt_id=o.attempt_id "
                    "AND prior.target=o.target AND prior.id<o.id "
                    "AND prior.delivered_at IS NULL) "
                    "ORDER BY o.id LIMIT ?",
                    (target, timestamp, timestamp, bounded_limit),
                ).fetchall()]
                if ids:
                    placeholders = ",".join("?" for _ in ids)
                    self._conn.execute(
                        f"UPDATE publication_outbox SET lease_owner=?, lease_until=? "
                        f"WHERE id IN ({placeholders})",
                        (worker_id, lease_until, *ids),
                    )
                    rows = self._conn.execute(
                        f"SELECT * FROM publication_outbox WHERE id IN ({placeholders}) "
                        "ORDER BY id", ids,
                    ).fetchall()
                else:
                    rows = []
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return [_outbox_item(row) for row in rows]

    def acknowledge(
        self,
        item_id: int,
        worker_id: str,
        *,
        now: Optional[float] = None,
    ) -> None:
        timestamp = time.time() if now is None else now
        self._finish_lease(
            item_id, worker_id,
            "UPDATE publication_outbox SET delivered_at=?, lease_owner=NULL, "
            "lease_until=NULL, last_error=NULL WHERE id=? AND lease_owner=? "
            "AND delivered_at IS NULL AND terminal_at IS NULL",
            (timestamp, item_id, worker_id),
        )

    def retry(
        self,
        item_id: int,
        worker_id: str,
        error: str,
        *,
        now: Optional[float] = None,
        base_delay: float = 1.0,
        max_delay: float = 300.0,
        max_attempts: int = 8,
    ) -> bool:
        """Release failed work with exponential backoff.

        Returns ``True`` if the item reached its terminal retry limit.
        """
        if base_delay < 0 or max_delay < base_delay:
            raise ValueError("invalid retry delay")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        timestamp = time.time() if now is None else now
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT attempt_count FROM publication_outbox WHERE id=? "
                    "AND lease_owner=? AND delivered_at IS NULL AND terminal_at IS NULL",
                    (item_id, worker_id),
                ).fetchone()
                if row is None:
                    raise LeaseConflict("outbox item is not leased by this worker")
                attempts = row["attempt_count"] + 1
                terminal = attempts >= max_attempts
                delay = min(max_delay, base_delay * (2 ** (attempts - 1)))
                self._conn.execute(
                    "UPDATE publication_outbox SET attempt_count=?, available_at=?, "
                    "lease_owner=NULL, lease_until=NULL, terminal_at=?, last_error=? "
                    "WHERE id=?",
                    (
                        attempts, timestamp + delay,
                        timestamp if terminal else None,
                        _bounded_error(error), item_id,
                    ),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return terminal

    def revive_terminal(
        self, target: str, *, cooldown_seconds: float = 3600.0,
        now: Optional[float] = None,
    ) -> int:
        """Re-open old terminal items after a circuit-breaker cooldown."""
        target = _validate_name(target, "target")
        if cooldown_seconds < 0 or not math.isfinite(cooldown_seconds):
            raise ValueError("cooldown_seconds must be finite and non-negative")
        timestamp = time.time() if now is None else now
        cutoff = timestamp - cooldown_seconds
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                attempts = self._conn.execute(
                    "SELECT DISTINCT workspace_id, project, run_id, attempt_id "
                    "FROM publication_outbox WHERE target=? AND delivered_at IS NULL "
                    "AND terminal_at IS NOT NULL AND terminal_at<=?",
                    (target, cutoff),
                ).fetchall()
                revived = self._conn.execute(
                    "UPDATE publication_outbox SET terminal_at=NULL, attempt_count=0, "
                    "available_at=?, lease_owner=NULL, lease_until=NULL "
                    "WHERE target=? AND delivered_at IS NULL AND terminal_at IS NOT NULL "
                    "AND terminal_at<=?",
                    (timestamp, target, cutoff),
                ).rowcount
                for row in attempts:
                    self._conn.execute(
                        "UPDATE publication_targets SET state='PENDING', last_error=NULL, "
                        "updated_at=? WHERE workspace_id=? AND project=? AND run_id=? "
                        "AND attempt_id=? AND target=?",
                        (timestamp, row["workspace_id"], row["project"], row["run_id"],
                         row["attempt_id"], target),
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return int(revived)

    def statuses(
        self,
        *,
        attempt: Optional[AttemptRef] = None,
        limit: int = 100,
        now: Optional[float] = None,
    ) -> list[TargetStatus]:
        """Return bounded aggregate target status without exposing payloads."""
        bounded_limit = _bounded_limit(limit)
        timestamp = time.time() if now is None else now
        params: list[Any] = [timestamp]
        where = ""
        if attempt is not None:
            where = (
                "WHERE t.workspace_id=? AND t.project=? AND t.run_id=? AND t.attempt_id=?"
            )
            params.extend(attempt.values())
        params.append(bounded_limit)
        query = f"""
            SELECT t.*, 
              COALESCE(SUM(CASE WHEN o.delivered_at IS NULL AND o.terminal_at IS NULL
                THEN 1 ELSE 0 END), 0) AS pending,
              COALESCE(SUM(CASE WHEN o.delivered_at IS NULL AND o.terminal_at IS NULL
                AND o.lease_until>? THEN 1 ELSE 0 END), 0) AS leased,
              COALESCE(SUM(CASE WHEN o.delivered_at IS NOT NULL THEN 1 ELSE 0 END), 0)
                AS delivered,
              COALESCE(SUM(CASE WHEN o.terminal_at IS NOT NULL THEN 1 ELSE 0 END), 0)
                AS terminal
            FROM publication_targets t
            LEFT JOIN publication_outbox o ON
              o.workspace_id=t.workspace_id AND o.project=t.project
              AND o.run_id=t.run_id AND o.attempt_id=t.attempt_id AND o.target=t.target
            {where}
            GROUP BY t.workspace_id, t.project, t.run_id, t.attempt_id, t.target
            ORDER BY t.updated_at DESC, t.project, t.run_id, t.attempt_id, t.target
            LIMIT ?
        """
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [_target_status(row) for row in rows]

    def _finish_lease(
        self,
        item_id: int,
        worker_id: str,
        statement: str,
        params: tuple[Any, ...],
    ) -> None:
        _validate_name(worker_id, "worker_id")
        with self._lock:
            result = self._conn.execute(statement, params)
            if result.rowcount != 1:
                self._conn.rollback()
                raise LeaseConflict("outbox item is not leased by this worker")
            self._conn.commit()

    @staticmethod
    def _prepare_record(record: OutboxRecord) -> tuple[OutboxRecord, str]:
        _validate_name(record.record_key, "record_key")
        _validate_name(record.kind, "kind")
        try:
            payload = json.dumps(
                dict(record.payload), ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("outbox payload must be finite JSON") from exc
        return record, payload


def _cursor_matches(row: Optional[sqlite3.Row], expected: Optional[SourceCursor]) -> bool:
    if row is None:
        return expected is None
    return (
        expected is not None
        and row["generation"] == expected.generation
        and row["byte_offset"] == expected.byte_offset
        and row["updated_at"] == expected.updated_at
        and row["file_identity"] == expected.file_identity
        and row["anchor_start"] == expected.anchor_start
        and row["anchor_digest"] == expected.anchor_digest
    )


def _validate_name(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512 or "\x00" in value:
        raise ValueError(f"invalid {label}")
    return value


def _bounded_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    return min(limit, _MAX_QUERY_LIMIT)


def _bounded_error(error: Optional[str]) -> Optional[str]:
    if error is None:
        return None
    # This is a final size boundary, not the primary secret sanitizer.
    return str(error).replace("\x00", "")[:_MAX_ERROR_LENGTH]


def _safe_dashboard_url(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"} or not parsed.hostname
        or parsed.username is not None or parsed.password is not None
        or parsed.query or parsed.fragment
    ):
        raise ValueError("dashboard_url must be a credential-free HTTP(S) URL")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("dashboard_url contains an invalid port") from exc
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _outbox_item(row: sqlite3.Row) -> OutboxItem:
    return OutboxItem(
        id=row["id"],
        attempt=AttemptRef(
            row["workspace_id"], row["project"], row["run_id"], row["attempt_id"],
        ),
        target=row["target"], record_key=row["record_key"], kind=row["kind"],
        payload=json.loads(row["payload_json"]), observed_at=row["observed_at"],
        created_at=row["created_at"], available_at=row["available_at"],
        attempt_count=row["attempt_count"], lease_owner=row["lease_owner"],
        lease_until=row["lease_until"],
    )


def _target_status(row: sqlite3.Row) -> TargetStatus:
    return TargetStatus(
        attempt=AttemptRef(
            row["workspace_id"], row["project"], row["run_id"], row["attempt_id"],
        ),
        target=row["target"], state=row["state"],
        dashboard_url=row["dashboard_url"], last_error=row["last_error"],
        updated_at=row["updated_at"], pending=row["pending"], leased=row["leased"],
        delivered=row["delivered"], terminal=row["terminal"],
    )
