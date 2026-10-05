"""Action state and audit events committed together using SQLite transactions.

Plans and artifacts stay in their existing directories. Legacy import/export
is an explicit offline operation implemented only in actions.migrate.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from ..storage import (
    StorageError, TransitionConflict, atomic_text,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS executions (
    action_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL CHECK (revision >= 0),
    payload TEXT NOT NULL CHECK (json_valid(payload) AND json_type(payload)='object')
);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY,
    action_id TEXT NOT NULL REFERENCES executions(action_id),
    event_key TEXT NOT NULL,
    revision INTEGER,
    payload TEXT NOT NULL CHECK (json_valid(payload) AND json_type(payload)='object'),
    UNIQUE (action_id, event_key),
    UNIQUE (action_id, revision)
);
CREATE INDEX IF NOT EXISTS events_action ON events(action_id, seq);
PRAGMA user_version=1;
"""


def _encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


class ActionDatabase:
    def __init__(self, root: Path):
        self.root = root
        self.path = root / "actions.sqlite3"
        self.marker = root / ".sqlite-authoritative"

    @contextmanager
    def transaction(self):
        # A missing database after migration must never replay stale JSON.
        if self.marker.exists() and not self.path.is_file():
            raise StorageError("authoritative Action database is missing")
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 0 and not self.marker.exists():
                connection.executescript("BEGIN IMMEDIATE;\n" + _SCHEMA + "COMMIT;")
            elif version != 1:
                raise StorageError("unsupported Action database schema")
            if not self.marker.exists():
                atomic_text(self.marker, "actions.sqlite3 is authoritative; see docs/action-storage.md\n")
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except sqlite3.DatabaseError as exc:
            raise StorageError("Action database transaction failed") from exc
        finally:
            # close rolls back an uncommitted transaction, including user errors.
            connection.close()

    def _read(self, connection, action_id):
        row = connection.execute(
            "SELECT revision,payload FROM executions WHERE action_id=?", (action_id,),
        ).fetchone()
        if row is None:
            directory = self.root / action_id
            if any((directory / name).exists() for name in ("execution.json", "journal.jsonl")):
                raise StorageError(f"legacy Action requires explicit migration: {action_id}")
            return {}
        value = json.loads(row["payload"])
        if value.get("revision", 0) != row["revision"]:
            raise StorageError(f"execution revision does not match durable state: {action_id}")
        return value

    def read(self, action_id):
        with self.transaction() as connection:
            return self._read(connection, action_id)

    def commit(self, action_id, payload, *, event, expected_status=None):
        """Check the caller's state once, inside the state+event transaction."""
        with self.transaction() as connection:
            current = self._read(connection, action_id)
            if expected_status is not None and current.get("status") != expected_status:
                raise TransitionConflict(
                    f"action state changed; expected {expected_status}, found {current.get('status')}"
                )
            revision = current.get("revision", 0)
            if payload.get("revision") != revision:
                raise TransitionConflict(
                    f"action state changed; expected revision {payload.get('revision')}, found {revision}"
                )
            value = {**payload, "revision": revision + 1}
            transition = {**event, "transition_id": uuid4().hex, "revision": revision + 1}
            connection.execute(
                "INSERT INTO executions VALUES (?,?,?) ON CONFLICT(action_id) "
                "DO UPDATE SET revision=excluded.revision,payload=excluded.payload",
                (action_id, revision + 1, _encode(value)),
            )
            connection.execute(
                "INSERT INTO events(action_id,event_key,revision,payload) VALUES (?,?,?,?)",
                (action_id, "transition:" + transition["transition_id"], revision + 1, _encode(transition)),
            )
            return value

    def append_event(self, action_id, event, *, event_id=None):
        identity = event_id or uuid4().hex
        with self.transaction() as connection:
            current = self._read(connection, action_id)
            connection.execute("INSERT OR IGNORE INTO executions VALUES (?,0,'{}')", (action_id,))
            record = {**event, "journal_event_id": identity, "state_revision": current.get("revision", 0)}
            connection.execute(
                "INSERT OR IGNORE INTO events(action_id,event_key,payload) VALUES (?,?,?)",
                (action_id, "event:" + identity, _encode(record)),
            )

    def journal(self, action_id, limit=100):
        with self.transaction() as connection:
            self._read(connection, action_id)
            rows = connection.execute(
                "SELECT payload FROM events WHERE action_id=? ORDER BY seq DESC LIMIT ?",
                (action_id, limit),
            ).fetchall()
            return [json.loads(row[0]) for row in reversed(rows)]
