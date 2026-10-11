"""Durable experiment journal; publication uses independent cursors in the same DB.

Keep the historical database path and sequences: moving scientific evidence is
unnecessary. Accepted records remain canonical after W&B confirmation or archive
release; only publication cursor state is an outbox.
"""
from __future__ import annotations
from contextlib import contextmanager
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid
from ..storage import utc_now


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


class RecordStore:
    def __init__(self, root: Path):
        self.root = root / "tracking"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = self.root / "outbox.sqlite"
        self._display_cache = OrderedDict()
        with self.connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs (
                    project TEXT, run TEXT, route TEXT NOT NULL, config TEXT NOT NULL,
                    PRIMARY KEY(project,run));
                CREATE TABLE IF NOT EXISTS scopes (
                    id TEXT PRIMARY KEY, project TEXT, run TEXT, attempt TEXT,
                    wandb_id TEXT NOT NULL, confirmed INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'PENDING', error TEXT,
                    last_attempt_at TEXT, last_confirmed_at TEXT,
                    failures INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS events (
                    scope TEXT, seq INTEGER, event_id TEXT, payload TEXT NOT NULL, dedup TEXT,
                    PRIMARY KEY(scope,seq), UNIQUE(scope,event_id), UNIQUE(scope,dedup));
                CREATE TABLE IF NOT EXISTS event_ids (
                    scope TEXT, event_id TEXT, digest TEXT NOT NULL, PRIMARY KEY(scope,event_id));
                CREATE TABLE IF NOT EXISTS display_metrics (
                    destination TEXT, signature TEXT, public_name TEXT, context TEXT,
                    PRIMARY KEY(destination,signature), UNIQUE(destination,public_name));
            """)
            if "display_version" not in {r[1] for r in conn.execute("PRAGMA table_info(scopes)")}:
                conn.execute("ALTER TABLE scopes ADD COLUMN display_version INTEGER NOT NULL DEFAULT 0")
            if "display_confirmed" not in {r[1] for r in conn.execute("PRAGMA table_info(scopes)")}:
                conn.execute("ALTER TABLE scopes ADD COLUMN display_confirmed INTEGER NOT NULL DEFAULT 0")
            conn.execute("INSERT OR IGNORE INTO meta VALUES ('workspace',?)", (uuid.uuid4().hex,))
        self.db.chmod(0o600)


    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.db, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()


    def bind(self, project, run, route, config):
        with self.connection() as conn:
            conn.execute("INSERT OR IGNORE INTO runs VALUES (?,?,?,?)",
                         (project, run, encoded(route), encoded(config)))
            row = conn.execute("SELECT * FROM runs WHERE project=? AND run=?", (project, run)).fetchone()
            if row["route"] != encoded(route) or row["config"] != encoded(config):
                raise ValueError("tracking identity is already frozen")


    def binding(self, project, run):
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM runs WHERE project=? AND run=?", (project, run)).fetchone()
        return None if row is None else {"route": json.loads(row["route"]), "config": json.loads(row["config"])}


    def scope(self, project, run, attempt):
        binding = self.binding(project, run)
        if binding is None:
            raise ValueError("Run records are not bound")
        identity = encoded([project, run, attempt])
        with self.connection() as conn:
            workspace = conn.execute("SELECT value FROM meta WHERE key='workspace'").fetchone()[0]
            wandb_id = digest([workspace, identity, binding["route"]])[:32]
            conn.execute("INSERT OR IGNORE INTO scopes(id,project,run,attempt,wandb_id) VALUES (?,?,?,?,?)",
                         (identity, project, run, attempt, wandb_id))
            if not binding["route"].get("enabled"):
                conn.execute("UPDATE scopes SET status='LOCAL_ONLY' WHERE id=? AND status!='LOCAL_ONLY'", (identity,))
        return identity


    def append(self, scope, events):
        # The whole batch and its export queue position commit together. An HTTP
        # retry acknowledges the original IDs, never logs a second observation.
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            seq = conn.execute("SELECT COALESCE(MAX(seq),0) FROM events WHERE scope=?", (scope,)).fetchone()[0]
            for event_id, payload in events:
                body = encoded(payload)
                checksum = digest(payload)
                old = conn.execute("SELECT digest FROM event_ids WHERE scope=? AND event_id=?", (scope, event_id)).fetchone()
                if old is not None:
                    if old[0] != checksum:
                        raise ValueError("event ID is already bound to different content")
                    continue
                conn.execute("INSERT INTO event_ids VALUES (?,?,?)", (scope, event_id, checksum))
                dedup = checksum if payload["kind"] == "metrics" else None
                if dedup is not None and conn.execute("SELECT 1 FROM events WHERE scope=? AND dedup=?", (scope, dedup)).fetchone():
                    continue
                seq += 1
                conn.execute("INSERT INTO events VALUES (?,?,?,?,?)", (scope, seq, event_id, body, dedup))
            conn.execute("UPDATE scopes SET status='PENDING' WHERE confirmed<? AND error IS NULL AND status NOT IN ('LOCAL_ONLY','PENDING') AND id=?", (seq, scope))
        return seq


    def events(self, scope, *, after=0, limit=128):
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM events WHERE scope=? AND seq>? ORDER BY seq LIMIT ?", (scope, after, limit)).fetchall()
        return [{"seq": r["seq"], "event_id": r["event_id"], "payload": json.loads(r["payload"])} for r in rows]


    def metric_records(self, project, run, attempt):
        scope = encoded([project, run, attempt])
        with self.connection() as conn:
            rows = conn.execute("SELECT payload FROM events WHERE scope=? AND json_extract(payload,'$.kind')='metrics' ORDER BY seq", (scope,)).fetchall()
        payloads = [json.loads(row[0]) for row in rows]
        return [item for value in payloads for item in value["observations"]]


    def total(self, scope):
        with self.connection() as conn:
            return conn.execute("SELECT COALESCE(MAX(seq),0) FROM events WHERE scope=?", (scope,)).fetchone()[0]


    def event_id(self, scope, sequence):
        with self.connection() as conn:
            row = conn.execute("SELECT event_id FROM events WHERE scope=? AND seq=?", (scope, sequence)).fetchone()
        return row[0] if row else None


    def latest_lifecycle(self, scope):
        with self.connection() as conn:
            row = conn.execute("SELECT payload FROM events WHERE scope=? AND json_extract(payload,'$.kind')='lifecycle' ORDER BY seq DESC LIMIT 1", (scope,)).fetchone()
        return json.loads(row[0])["data"] if row else {}
