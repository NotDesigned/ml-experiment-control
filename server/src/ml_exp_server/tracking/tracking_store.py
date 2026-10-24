"""Private settings and a durable, idempotent SQLite publication outbox."""
from __future__ import annotations

import fcntl
import json

from ..storage import atomic_json, utc_now
from .tracking_contract import resolve, resolve_preparation
from .wandb_display import Projection, identity, fingerprint, stem
from ..records.store import RecordStore, encoded, digest
from .wandb_dependency import sdk_status


class TrackingStore(RecordStore):


    def settings(self):
        path = self.root / "settings.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def public_settings(self):
        value = self.settings()
        return {"enabled": value.get("enabled", True), "preparation_enabled": value.get("preparation_enabled", False),
                "entity": value.get("entity"),
                "resolved_entity": value.get("resolved_entity"), "project": value.get("project"),
                "credentials_configured": bool(value.get("api_key")), "error": value.get("error"),
                "project_fallback": "ml-expd-project", "cloud": "https://api.wandb.ai",
                "publisher": sdk_status()}

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
        preparation = (request.preparation_enabled if "preparation_enabled" in request.model_fields_set
                       else previous.get("preparation_enabled", False))
        value = {"enabled": enabled, "preparation_enabled": preparation, "entity": entity, "project": project,
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


    def preparation_route(self, options, project):
        return resolve_preparation(options, self.settings(), project)


    def pending(self):
        with self.connection() as conn:
            rows = conn.execute("SELECT s.* FROM scopes s JOIN runs r ON r.project=s.project AND r.run=s.run WHERE json_extract(r.route,'$.enabled')=1 AND ( confirmed < (SELECT COALESCE(MAX(seq),0) FROM events WHERE scope=s.id) OR (confirmed>0 AND (display_version<2 OR display_confirmed<confirmed))) ORDER BY COALESCE(last_attempt_at,'')").fetchall()
        return [dict(row) for row in rows if not self.is_deleted(row["project"], row["run"])]

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
        for row in rows if route["enabled"] else []:
            value = dict(row)
            value.pop("id")
            value["pending"] = value["total"] - value["confirmed"]
            value["url"] = f'https://wandb.ai/{route["entity"]}/{route["project"]}/runs/{value["wandb_id"]}'
            scopes.append(value)
        return {"wandb": route, "scopes": scopes, "publisher": sdk_status()}


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
