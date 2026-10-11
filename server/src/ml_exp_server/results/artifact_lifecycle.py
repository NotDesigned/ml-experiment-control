"""Release downloaded archives, preserving their immutable scientific receipts."""
from __future__ import annotations

from datetime import datetime, timezone

from ..application_errors import ApplicationError
from .artifact_store import ArtifactStore
from .artifact_cache import evict_caches
from ..storage import atomic_json


class ArtifactLifecycle:
    def __init__(self, store: ArtifactStore):
        self.store = store

    @staticmethod
    def matches(receipt, proof):
        if (proof["archive_sha256"] != receipt["sha256"] or proof["archive_bytes"] != receipt["bytes"]
                or set(proof["files"]) != {"outputs/" + item["path"] for item in receipt["files"]}):
            return False
        return all(proof["files"]["outputs/" + item["path"]]["bytes"] == item["bytes"]
                   and proof["files"]["outputs/" + item["path"]]["sha256"] == item.get("sha256", proof["files"]["outputs/" + item["path"]]["sha256"])
                   for item in receipt["files"])

    def acknowledge(self, project, run, attempt, proof, *, release=True):
        with self.store.record(project, run, attempt) as (path, value):
            receipt = None if not value else value.get("receipt")
            if not receipt:
                raise ApplicationError("archive is not published", status_code=404, code="ARTIFACT_UNAVAILABLE")
            if not self.matches(receipt, proof):
                raise ApplicationError("download verification differs from the sealed receipt", code="DOWNLOAD_PROOF_CONFLICT")
            retention = value.get("retention", {})
            if retention.get("status") in {"RELEASING", "RELEASED"}:
                return retention
            now = datetime.now(timezone.utc)
            acknowledgement = value.get("download_acknowledgement", {**proof, "verified_at": now.isoformat()})
            due = min(now, datetime.fromisoformat(retention["release_after"])).isoformat() if retention.get("status") == "SCHEDULED" else now.isoformat()
            retention = {"status": "SCHEDULED" if release else "KEEP", "sha256": receipt["sha256"],
                         "release_after": due if release else None}
            value.update(download_acknowledgement=acknowledgement, retention=retention)
            atomic_json(path, value)
            return retention

    def run_due(self):
        released = 0
        for file in self.store.root.glob("*/*/attempt-*.json"):
            identity = (file.parent.parent.name, file.parent.name, file.stem)
            with self.store.record(*identity) as (path, value):
                retention = value.get("retention", {})
                if retention.get("status") not in {"SCHEDULED", "RELEASING", "RELEASED"}:
                    continue
                receipt = value["receipt"]
                canonical = "/".join([*identity, receipt["sha256"] + ".tar"])
                if (receipt["object_key"] != canonical or retention["sha256"] != receipt["sha256"]
                        or not self.matches(receipt, value["download_acknowledgement"])):
                    raise ValueError("archive release identity differs")
                if retention["status"] != "RELEASED":
                    # Durable marker before the external effect. Retry only the
                    # exact object after a crash; never restart compute.
                    retention["status"] = "RELEASING"
                    atomic_json(path, value)
                    try:
                        self.store.client().delete_object(Bucket=self.store.config["bucket"], Key=canonical)
                    except Exception:
                        retention["diagnostic"] = "ARCHIVE_RELEASE_RETRY_REQUIRED"
                        atomic_json(path, value)
                        continue
                    retention.update(status="RELEASED", released_at=datetime.now(timezone.utc).isoformat())
                    retention.pop("diagnostic", None)
                    atomic_json(path, value)
                    released += 1
                # Include historical releases and copies arriving after release.
                # Cache failures must not repeat an already completed S3 DELETE.
                try:
                    cleanup = evict_caches(value)
                except OSError:
                    cleanup = {**value.get("cache_cleanup", {}), "diagnostic": "LOCAL_CACHE_CLEANUP_RETRY_REQUIRED"}
                if cleanup != value.get("cache_cleanup"):
                    value["cache_cleanup"] = cleanup
                    atomic_json(path, value)
        return released
