"""Private settings and a durable, idempotent SQLite publication outbox."""
from __future__ import annotations

from contextlib import contextmanager
from collections import OrderedDict
import hashlib
import fcntl
import json
from pathlib import Path
import sqlite3
import uuid

from ..storage import atomic_json, utc_now
from .tracking_contract import resolve
from .wandb_display import Projection, identity, fingerprint, stem


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


class TrackingStore:
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

    def settings(self):
        path = self.root / "settings.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def public_settings(self):
        value = self.settings()
        return {"enabled": value.get("enabled", True), "entity": value.get("entity"),
                "resolved_entity": value.get("resolved_entity"), "project": value.get("project"),
                "credentials_configured": bool(value.get("api_key")), "error": value.get("error"),
                "project_fallback": "ml-expd-project", "cloud": "https://api.wandb.ai"}

    def configure(self, request, lookup):
        with (self.root / "settings.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._configure(request, lookup)

    def _configure(self, request, lookup):
        previous = self.settings()
        key = (None if request.clear_credentials else request.api_key.get_secret_value()
               if request.api_key is not None else previous.get("api_key"))
        entity = request.entity if "entity" in request.model_fields_set else previous.get("entity")
        project = request.project if "project" in request.model_fields_set else previous.get("project")
        enabled = request.enabled if "enabled" in request.model_fields_set else previous.get("enabled", True)
        value = {"enabled": enabled, "entity": entity, "project": project,
                 "api_key": key, "resolved_entity": entity, "error": None}
        cached = (key == previous.get("api_key") and "entity" not in request.model_fields_set
                  and previous.get("resolved_entity"))
        if key and entity is None and cached:
            value["resolved_entity"] = cached
        elif key and entity is None:
            try:
                entity = lookup(key)
                # Validate remote identity before embedding it in paths/URLs.
                from .tracking_contract import WandbOptions
                value["resolved_entity"] = WandbOptions(entity=entity).entity
                if not entity:
                    value["error"] = "DEFAULT_ENTITY_UNAVAILABLE"
            except Exception:
                value["error"] = "DEFAULT_ENTITY_UNAVAILABLE"
        atomic_json(self.root / "settings.json", value)
        (self.root / "settings.json").chmod(0o600)
        return self.public_settings()

    def route(self, options, project):
        return resolve(options, self.settings(), project)

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
        if binding is None or not binding["route"]["enabled"]:
            raise ValueError("tracking is not enabled for this Run")
        identity = encoded([project, run, attempt])
        with self.connection() as conn:
            workspace = conn.execute("SELECT value FROM meta WHERE key='workspace'").fetchone()[0]
            wandb_id = digest([workspace, identity, binding["route"]])[:32]
            conn.execute("INSERT OR IGNORE INTO scopes(id,project,run,attempt,wandb_id) VALUES (?,?,?,?,?)",
                         (identity, project, run, attempt, wandb_id))
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
            conn.execute("UPDATE scopes SET status=CASE WHEN confirmed<? AND error IS NULL THEN 'PENDING' ELSE status END WHERE id=?", (seq, scope))
        return seq

    def events(self, scope, *, after=0, limit=128):
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM events WHERE scope=? AND seq>? ORDER BY seq LIMIT ?", (scope, after, limit)).fetchall()
        return [{"seq": r["seq"], "event_id": r["event_id"], "payload": json.loads(r["payload"])} for r in rows]

    def pending(self):
        with self.connection() as conn:
            rows = conn.execute("SELECT s.* FROM scopes s WHERE confirmed < (SELECT COALESCE(MAX(seq),0) FROM events WHERE scope=s.id) OR (confirmed>0 AND (display_version<2 OR display_confirmed<confirmed)) ORDER BY COALESCE(last_attempt_at,'')").fetchall()
        return [dict(row) for row in rows]

    def outcome(self, scope, *, confirmed=None, error=None, display_version=None):
        with self.connection() as conn:
            if confirmed is not None:
                conn.execute("UPDATE scopes SET confirmed=MAX(confirmed,?),last_confirmed_at=? WHERE id=?", (confirmed, utc_now(), scope))
            if display_version is not None:
                conn.execute("UPDATE scopes SET display_version=MAX(display_version,?),display_confirmed=MAX(display_confirmed,confirmed) WHERE id=?", (display_version, scope))
            total = conn.execute("SELECT COALESCE(MAX(seq),0) FROM events WHERE scope=?", (scope,)).fetchone()[0]
            conn.execute("UPDATE scopes SET status=CASE WHEN ? IS NOT NULL THEN 'RETRY_PENDING' WHEN confirmed>=? THEN 'SYNCED' ELSE 'PENDING' END,error=?,last_attempt_at=?,failures=CASE WHEN ? IS NULL THEN 0 ELSE failures+1 END WHERE id=?", (error, total, error, utc_now(), error, scope))

    def status(self, project, run):
        binding = self.binding(project, run)
        with self.connection() as conn:
            rows = conn.execute("SELECT s.*,(SELECT COALESCE(MAX(seq),0) FROM events WHERE scope=s.id) AS total FROM scopes s WHERE project=? AND run=?", (project, run)).fetchall()
        route = binding["route"] if binding else {"enabled": False, "reason": "HISTORICAL_RUN_NOT_BOUND"}
        scopes = []
        for row in rows:
            value = dict(row)
            value.pop("id")
            value["pending"] = value["total"] - value["confirmed"]
            value["url"] = f'https://wandb.ai/{route["entity"]}/{route["project"]}/runs/{value["wandb_id"]}'
            scopes.append(value)
        return {"wandb": route, "scopes": scopes}

    def metric_records(self, project, run, attempt):
        scope = encoded([project, run, attempt])
        with self.connection() as conn:
            rows = conn.execute("SELECT payload FROM events WHERE scope=? ORDER BY seq", (scope,)).fetchall()
        payloads = [json.loads(row[0]) for row in rows]
        return [item for value in payloads if value["kind"] == "metrics" for item in value["observations"]]

    def total(self, scope):
        with self.connection() as conn:
            return conn.execute("SELECT COALESCE(MAX(seq),0) FROM events WHERE scope=?", (scope,)).fetchone()[0]

    def display(self, scope, through):
        projection = self._display_cache.pop(scope, Projection())
        if projection.cursor > through:
            projection = Projection()
        with self.connection() as conn:
            for row in conn.execute("SELECT seq,payload FROM events WHERE scope=? AND seq>? AND seq<=? ORDER BY seq",
                                    (scope, projection.cursor, through)):
                payload = json.loads(row["payload"])
                if payload["kind"] == "metrics":
                    for item in payload["observations"]:
                        projection.add(item, row["seq"])
            conn.execute("BEGIN IMMEDIATE")
            project, run, _ = json.loads(scope)
            route = json.loads(conn.execute("SELECT route FROM runs WHERE project=? AND run=?", (project, run)).fetchone()[0])
            destination = encoded([route["entity"], route["project"]])
            names = {}
            for series in projection.series.values():
                context = {k: series["context"][k] for k in ("name", "unit", "protocol_id", "dataset_id")}
                signature = identity(context)
                existing = conn.execute("SELECT public_name FROM display_metrics WHERE destination=? AND signature=?",
                                        (destination, signature)).fetchone()
                if existing:
                    names[signature] = existing[0]
                    continue
                name = stem(context["name"])
                if conn.execute("SELECT 1 FROM display_metrics WHERE destination=? AND public_name=?", (destination, name)).fetchone():
                    name += "/" + fingerprint(signature)
                conn.execute("INSERT INTO display_metrics VALUES (?,?,?,?)", (destination, signature, name, encoded(context)))
                names[signature] = name
        projection.cursor = through
        self._display_cache[scope] = projection
        if len(self._display_cache) > 16:
            self._display_cache.popitem(last=False)
        return projection.document(names)

    def event_id(self, scope, sequence):
        with self.connection() as conn:
            row = conn.execute("SELECT event_id FROM events WHERE scope=? AND seq=?", (scope, sequence)).fetchone()
        return row[0] if row else None

    def latest_lifecycle(self, scope):
        with self.connection() as conn:
            rows = conn.execute("SELECT payload FROM events WHERE scope=? ORDER BY seq DESC", (scope,)).fetchall()
        for row in rows:
            value = json.loads(row[0])
            if value["kind"] == "lifecycle":
                return value["data"]
        return {}
