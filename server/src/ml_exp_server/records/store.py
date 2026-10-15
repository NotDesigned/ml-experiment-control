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
from ..deletion_state import RunFence
from ..application_errors import ApplicationError


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
        with self.run_guard(project, run, config=config):
            return self._bind(project, run, route, config)

    def _bind(self, project, run, route, config):
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
        with self.run_guard(project, run):
            return self._scope(project, run, attempt)

    def _scope(self, project, run, attempt):
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
        project, run, _ = json.loads(scope)
        with self.run_guard(project, run):
            return self._append(scope, events)

    def _append(self, scope, events):
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

    @staticmethod
    def run_identities(run):
        return (run, run + "--preparation", run + "--backfill")

    def is_deleted(self, project, run):
        fence = RunFence(self.root.parent)
        return any(fence.is_deleted(project, name) for name in self._fence_names(project, run))

    def _fence_names(self, project, run, config=None):
        names = [run]
        if config is None:
            binding = self.binding(project, run)
            config = binding["config"] if binding else {}
        data = config.get("ml_expd", {})
        kind, parent = data.get("record_type"), data.get("run_id")
        if kind in {"preparation", "backfill"} and isinstance(parent, str) and run == parent + "--" + kind:
            names.append(parent)
        return names

    @contextmanager
    def run_guard(self, project, run, *, config=None):
        fence = RunFence(self.root.parent)
        names = self._fence_names(project, run, config)
        with fence.guard(project, names[0]):
            for name in names[1:]:
                fence.require_live(project, name)
            yield

    def delete_run(self, project, run):
        """Remove canonical local records after the permanent fence was committed."""
        if not RunFence(self.root.parent).is_deleted(project, run):
            raise ApplicationError("Run deletion must be durably fenced first", code="RUN_DELETION_NOT_STARTED")
        from ..tracking.wandb_display import identity
        with self.connection() as conn:
            rows = conn.execute("SELECT run,config FROM runs WHERE project=? AND run IN (?,?,?)",
                                (project, *self.run_identities(run))).fetchall()
        names = [run]
        for row in rows:
            if row["run"] != run and run in self._fence_names(project, row["run"], json.loads(row["config"]))[1:]:
                names.append(row["run"])
        # Preserve the proven publication identities too: after removing their
        # binding, a late record must not recreate an orphan event stream.
        fence = RunFence(self.root.parent)
        with fence.exclusive():
            for name in names[1:]:
                fence.mark(project, name, {"publication_parent": run})
        predicate = "project=? AND run IN (" + ",".join("?" for _ in names) + ")"
        arguments = (project, *names)
        scopes = []
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            bindings = {row["run"]: json.loads(row["route"]) for row in conn.execute(
                "SELECT run,route FROM runs WHERE " + predicate, arguments)}
            scopes = [row["id"] for row in conn.execute(
                "SELECT id FROM scopes WHERE " + predicate, arguments)]
            candidates = set()
            for row in conn.execute("SELECT e.payload,s.run FROM events e JOIN scopes s ON s.id=e.scope WHERE " + predicate.replace("project=", "s.project=").replace("run IN", "s.run IN"), arguments):
                route = bindings.get(row["run"], {})
                if route.get("entity") and route.get("project"):
                    destination = encoded([route["entity"], route["project"]])
                    for item in json.loads(row["payload"]).get("observations", []):
                        candidates.add((destination, identity(item)))
            for scope in scopes:
                conn.execute("DELETE FROM events WHERE scope=?", (scope,))
                conn.execute("DELETE FROM event_ids WHERE scope=?", (scope,))
            conn.execute("DELETE FROM scopes WHERE " + predicate, arguments)
            conn.execute("DELETE FROM runs WHERE " + predicate, arguments)
            # Display names are shared per W&B destination. Preserve mappings
            # still used by another Run instead of breaking its metric names.
            surviving = set()
            for row in conn.execute("SELECT e.payload,r.route FROM events e JOIN scopes s ON s.id=e.scope JOIN runs r ON r.project=s.project AND r.run=s.run"):
                route = json.loads(row["route"])
                if route.get("entity") and route.get("project"):
                    destination = encoded([route["entity"], route["project"]])
                    for item in json.loads(row["payload"]).get("observations", []):
                        surviving.add((destination, identity(item)))
            for destination, signature in candidates - surviving:
                conn.execute("DELETE FROM display_metrics WHERE destination=? AND signature=?", (destination, signature))
        for scope in scopes:
            self._display_cache.pop(scope, None)
        return {"status": "DELETED", "scopes_removed": len(scopes)}


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
