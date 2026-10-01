"""Action state and audit events committed together using SQLite transactions.

Plans and artifacts stay in their existing directories. Legacy JSON is imported
once per Action, under the ActionStore lock, and is never dual-written.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from ..storage import (
    DurableJsonState, DurableSnapshot, StorageError, TransitionConflict,
    _jsonl_mappings, atomic_text,
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

    def state(self, action_id: str):
        return SqliteActionState(self, action_id)


class SqliteActionState:
    def __init__(self, database: ActionDatabase, action_id: str):
        self.database = database
        self.action_id = action_id

    def _load(self, connection):
        row = connection.execute(
            "SELECT revision,payload FROM executions WHERE action_id=?", (self.action_id,),
        ).fetchone()
        if row is None:
            snapshot, events = self._legacy()
            connection.execute(
                "INSERT INTO executions VALUES (?,?,?)",
                (self.action_id, snapshot.revision, _encode(snapshot.value)),
            )
            for index, event in enumerate(events):
                key = ("transition:" + event["transition_id"] if "transition_id" in event
                       else "event:" + event["journal_event_id"] if "journal_event_id" in event
                       else f"legacy:{index}")
                connection.execute(
                    "INSERT INTO events(action_id,event_key,revision,payload) VALUES (?,?,?,?)",
                    (self.action_id, key, event.get("revision") if "transition_id" in event else None,
                     _encode(event)),
                )
            return snapshot
        value = json.loads(row["payload"])
        if value and value.get("revision") != row["revision"]:
            raise StorageError(f"execution revision does not match durable state: {self.action_id}")
        return DurableSnapshot(value=value, revision=row["revision"])

    def _legacy(self):
        directory = self.database.root / self.action_id
        # The legacy recovery code can normalize a truncated tail. Run it only
        # on private copies; originals remain a byte-for-byte recovery point.
        with tempfile.TemporaryDirectory(prefix=".import-", dir=self.database.root) as temporary:
            copied = Path(temporary)
            for name in ("execution.json", "journal.jsonl"):
                source = directory / name
                if source.is_file():
                    shutil.copyfile(source, copied / name)
            legacy = DurableJsonState(copied / "execution.json", copied / "journal.jsonl")
            snapshot = legacy.snapshot({})
            legacy.repair_journal(snapshot)
            events = _jsonl_mappings(copied / "journal.jsonl")
        if snapshot.value and snapshot.last_transition is None and any("transition_id" in event for event in events):
            raise StorageError(f"Action journal has transitions without authoritative metadata: {self.action_id}")
        value = dict(snapshot.value)
        if value:
            if value.get("revision", snapshot.revision) != snapshot.revision:
                raise StorageError(f"execution revision does not match durable state: {self.action_id}")
            value["revision"] = snapshot.revision
        elif events:
            raise StorageError(f"Action journal exists without execution state: {self.action_id}")
        return DurableSnapshot(value=value, revision=snapshot.revision), events

    def snapshot(self, default):
        with self.database.transaction() as connection:
            return self._load(connection)

    def commit(self, value, *, event, expected_revision=None):
        with self.database.transaction() as connection:
            current = self._load(connection)
            if expected_revision is not None and current.revision != expected_revision:
                raise TransitionConflict(
                    f"durable state changed; expected revision {expected_revision}, found {current.revision}"
                )
            revision = current.revision + 1
            if value.get("revision") != revision:
                raise StorageError("execution revision does not match committed revision")
            transition = {**event, "transition_id": uuid4().hex, "revision": revision}
            connection.execute(
                "UPDATE executions SET revision=?,payload=? WHERE action_id=?",
                (revision, _encode(value), self.action_id),
            )
            connection.execute(
                "INSERT INTO events(action_id,event_key,revision,payload) VALUES (?,?,?,?)",
                (self.action_id, "transition:" + transition["transition_id"], revision, _encode(transition)),
            )
            return DurableSnapshot(value=dict(value), revision=revision, last_transition=transition)

    def append_event(self, event, *, event_id=None):
        identity = event_id or uuid4().hex
        with self.database.transaction() as connection:
            snapshot = self._load(connection)
            record = {**event, "journal_event_id": identity, "state_revision": snapshot.revision}
            connection.execute(
                "INSERT OR IGNORE INTO events(action_id,event_key,payload) VALUES (?,?,?)",
                (self.action_id, "event:" + identity, _encode(record)),
            )
            return json.loads(connection.execute(
                "SELECT payload FROM events WHERE action_id=? AND event_key=?",
                (self.action_id, "event:" + identity),
            ).fetchone()[0])

    def journal(self, limit=100):
        with self.database.transaction() as connection:
            self._load(connection)
            rows = connection.execute(
                "SELECT payload FROM events WHERE action_id=? ORDER BY seq DESC LIMIT ?",
                (self.action_id, limit),
            ).fetchall()
            return [json.loads(row[0]) for row in reversed(rows)]
