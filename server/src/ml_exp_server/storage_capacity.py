"""Local disk admission shared by result collection and resumable uploads.

Claims are conservative peak byte estimates, held by OS locks. A crashed
process loses its claim automatically; upload promises come from durable
session metadata and survive restarts. These are not artifact retention rules.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import shutil
import tempfile

from .application_errors import ApplicationError


class CapacityError(ApplicationError):
    def __init__(self, required, available, reserved):
        super().__init__("insufficient local space for transfer and publication peak",
                         status_code=507, code="UPLOAD_STORAGE")
        self.required_free_bytes = required
        self.available_bytes = available
        self.reserved_bytes = reserved


class CapacityLease:
    def __init__(self, capacity, claim, ignore_upload):
        self.capacity, self.claim, self.ignore_upload = capacity, claim, ignore_upload

    def resize(self, size):
        with self.capacity.locked():
            self.capacity.check(size, ignore_upload=self.ignore_upload, ignore_claim=Path(self.claim.name))
            self.claim.seek(0)
            json.dump({"bytes": size}, self.claim)
            self.claim.truncate()
            self.claim.flush()


class StorageCapacity:
    def __init__(self, registry, config):
        self.registry = Path(registry)
        self.root = self.registry / "storage-reservations"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.headroom = config.get("local_storage_headroom_bytes", 1024 ** 3)
        if type(self.headroom) is not int or self.headroom < 0:
            raise ValueError("invalid local storage headroom")

    @contextmanager
    def locked(self):
        with (self.root / "admission.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def snapshot(self, *, ignore_upload=None, ignore_claim=None):
        # Caller holds admission.lock. Upload commits use the same lock so
        # free bytes and the remaining promises cannot straddle a commit.
        free = shutil.disk_usage(self.registry).free
        promised = 0
        for path in (self.registry / "multipart-uploads").glob("upload.*/upload.json"):
            if not re.fullmatch(r"upload\.[0-9a-f]{64}", path.parent.name):
                continue
            value = json.loads(path.read_text())
            if (path.parent.name == ignore_upload or value["status"] != "UPLOADING"
                    or datetime.fromisoformat(value["expires_at"]) <= datetime.now(timezone.utc)):
                continue
            promised += max(0, value["bytes"] - sum(part["bytes"] for part in value["parts"].values()))
        active = 0
        for path in self.root.glob("claim-*"):
            if path == ignore_claim:
                continue
            with path.open("r+") as claim:
                try:
                    fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    active += json.load(claim)["bytes"]
                else:
                    path.unlink()  # Only abandoned capacity metadata, never transfer bytes.
        return {"free_bytes": free, "upload_reserved_bytes": promised,
                "operation_reserved_bytes": active, "headroom_bytes": self.headroom,
                "admissible_bytes": max(0, free - promised - active - self.headroom)}

    def check(self, size, *, ignore_upload=None, ignore_claim=None):
        value = self.snapshot(ignore_upload=ignore_upload, ignore_claim=ignore_claim)
        reserved = value["upload_reserved_bytes"] + value["operation_reserved_bytes"]
        required = size + reserved + self.headroom
        if required > value["free_bytes"]:
            raise CapacityError(required, value["free_bytes"], reserved)
        return value

    @contextmanager
    def reserve(self, size, *, ignore_upload=None):
        with self.locked():
            self.check(size, ignore_upload=ignore_upload)
            claim = tempfile.NamedTemporaryFile(mode="w+", prefix="claim-", dir=self.root)
            try:
                fcntl.flock(claim, fcntl.LOCK_EX)
                json.dump({"bytes": size}, claim)
                claim.flush()
            except BaseException:
                claim.close()
                raise
        try:
            yield CapacityLease(self, claim, ignore_upload)
        finally:
            with self.locked():
                claim.close()
