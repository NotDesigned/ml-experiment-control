"""Explicit administrator deletion of sealed exact-Attempt result objects.

This authorization is independent of verified-download ACKs. Original receipts,
ACK proofs and KEEP decisions remain unchanged; the caller owns the permanent
Run tombstone and the durable deletion journal.
"""
from __future__ import annotations

from copy import deepcopy
import re

from ..application_errors import ApplicationError
from ..deletion_state import RunFence
from ..storage import utc_now
from .artifact_cache import evict_caches
from .artifact_store import ATTEMPT

_SHA = re.compile(r"^[0-9a-f]{64}$")


class AdminArtifacts:
    def __init__(self, objects):
        self.objects = objects

    @staticmethod
    def _item(project, run, attempt, receipt, *, released=False):
        sha = receipt.get("sha256")
        if (not isinstance(sha, str) or not _SHA.fullmatch(sha)
                or type(receipt.get("bytes")) is not int or receipt["bytes"] <= 0):
            raise ValueError("invalid sealed archive identity")
        canonical = "/".join([project, run, attempt, sha + ".tar"])
        if receipt.get("object_key") != canonical:
            raise ValueError("archive deletion key differs from its canonical identity")
        return {"project": project, "run_id": run, "attempt_id": attempt,
                "object_key": canonical, "sha256": sha, "bytes": receipt["bytes"],
                "already_released": released}

    def plan_objects(self, project, run):
        # The orchestration layer holds shared/exclusive RunFence as appropriate.
        # Acquiring another shared descriptor here would deadlock an EX caller.
        RunFence(self.objects.root.parent).path(project, run)
        items = {}
        for path in sorted((self.objects.root / project / run).glob("attempt-*.json")):
            attempt = path.stem
            if not ATTEMPT.fullmatch(attempt):
                raise ValueError("invalid Attempt archive record")
            with self.objects.record(project, run, attempt, bypass_fence=True) as (_, value):
                if not value or (value.get("project"), value.get("run_id"), value.get("attempt_id")) != (project, run, attempt):
                    raise ValueError("archive deletion provenance differs")
                current = value.get("receipt")
                retention = value.get("retention", {})
                if current and retention.get("status") == "RELEASED" and retention.get("sha256", current["sha256"]) != current["sha256"]:
                    raise ValueError("released archive identity differs")
                for receipt in [*value.get("receipt_history", []), *([current] if current else [])]:
                    item = self._item(project, run, attempt, receipt,
                                      released=receipt is current and retention.get("status") == "RELEASED")
                    old = items.get(item["object_key"])
                    if old and any(old[key] != item[key] for key in ("sha256", "bytes")):
                        raise ValueError("archive history identity differs")
                    if old:
                        item["already_released"] = old["already_released"] or item["already_released"]
                    items[item["object_key"]] = item
        return [items[key] for key in sorted(items)]

    def _receipt(self, item):
        project, run, attempt = (item[key] for key in ("project", "run_id", "attempt_id"))
        RunFence(self.objects.root.parent).path(project, run)
        if not ATTEMPT.fullmatch(attempt):
            raise ValueError("invalid Attempt archive identity")
        with self.objects.record(project, run, attempt, bypass_fence=True) as (_, value):
            if not value or (value.get("project"), value.get("run_id"), value.get("attempt_id")) != (project, run, attempt):
                raise ValueError("archive deletion provenance differs")
            for receipt in [value.get("receipt"), *value.get("receipt_history", [])]:
                if receipt and receipt.get("object_key") == item["object_key"]:
                    expected = self._item(project, run, attempt, receipt, released=item["already_released"])
                    if item != expected:
                        raise ValueError("archive deletion manifest differs from its sealed receipt")
                    released = receipt is value.get("receipt") and value.get("retention", {}).get("status") == "RELEASED"
                    if item["already_released"] != released:
                        raise ValueError("archive release state changed after deletion planning")
                    return deepcopy(value), deepcopy(receipt)
        raise ValueError("archive deletion receipt is missing")

    def delete_objects(self, manifest, state, persist):
        # Validate every item before allowing the first external effect.
        originals = {}
        for item in manifest:
            if not RunFence(self.objects.root.parent).is_deleted(item["project"], item["run_id"]):
                raise ApplicationError("Run deletion must be durably fenced first", code="RUN_DELETION_NOT_STARTED")
            originals[item["object_key"]] = self._receipt(item)
        progress = state.setdefault("artifact_objects", {})
        cleanup = state.setdefault("artifact_cache_cleanup", {})
        for item in manifest:
            key = item["object_key"]
            result = progress.get(key, {})
            if result.get("status") not in {"DELETED", "SKIPPED_RELEASED"}:
                if item["already_released"]:
                    progress[key] = {"status": "SKIPPED_RELEASED", "finished_at": utc_now()}
                    persist(state)
                else:
                    progress[key] = {"status": "DELETING", "started_at": utc_now()}
                    persist(state)
                    try:
                        self.objects.client().delete_object(Bucket=self.objects.config["bucket"], Key=key)
                    except Exception:
                        progress[key].update(status="RETRY_REQUIRED", diagnostic="ADMIN_OBJECT_DELETE_RETRY_REQUIRED")
                        persist(state)
                        raise ApplicationError("archive deletion needs retry", code="ADMIN_OBJECT_DELETE_RETRY_REQUIRED") from None
                    progress[key].update(status="DELETED", finished_at=utc_now())
                    persist(state)
            # A cache retry never repeats an acknowledged object-store DELETE.
            value, receipt = originals[key]
            try:
                report = evict_caches({**value, "receipt": receipt, "cache_cleanup": cleanup.get(key, {})})
            except OSError:
                cleanup[key] = {**cleanup.get(key, {}), "diagnostic": "ADMIN_CACHE_CLEANUP_RETRY_REQUIRED"}
                persist(state)
                raise ApplicationError("local archive cache deletion needs retry", code="ADMIN_CACHE_CLEANUP_RETRY_REQUIRED") from None
            if report != cleanup.get(key):
                cleanup[key] = report
                persist(state)
        return state
