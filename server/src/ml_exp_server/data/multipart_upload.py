"""Durable, bounded archive uploads; publish only after whole-archive validation."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import time

from ..application_errors import ApplicationError
from ..storage import atomic_json, utc_now
from ..archive_limits import exceeds

UPLOAD_ID = re.compile(r"^upload\.[0-9a-f]{64}$")
PART_BYTES = 16 * 1024 ** 2


class PartStream(io.RawIOBase):
    """Seekable view of immutable parts, without a second assembled archive."""
    def __init__(self, paths, part_bytes, size):
        super().__init__()
        self.paths, self.part_bytes, self.size, self.position = paths, part_bytes, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset if whence == 2 else -1
        if position < 0:
            raise ValueError("invalid archive seek")
        self.position = position
        return position

    def readinto(self, buffer):
        length = min(len(buffer), self.size - self.position)
        if length <= 0:
            return 0
        written = 0
        while written < length:
            index, offset = divmod(self.position, self.part_bytes)
            count = min(length - written, self.part_bytes - offset)
            with self.paths[index].open("rb") as part:
                part.seek(offset)
                data = part.read(count)
            if len(data) != count:
                raise ValueError("stored upload part is truncated")
            buffer[written:written + count] = data
            written += count
            self.position += count
        return written


class UploadStore:
    def __init__(self, root, config):
        self.root = Path(root) / "multipart-uploads"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.part_bytes = int(config.get("upload_part_bytes", PART_BYTES))
        self.ttl = int(config.get("upload_session_seconds", 86400))
        self.max_parts = int(config.get("upload_max_parts", 65536))
        if (not 1024 <= self.part_bytes <= 64 * 1024 ** 2 or not 300 <= self.ttl <= 604800
                or not 1 <= self.max_parts <= 65536):
            raise ValueError("invalid multipart upload policy")

    @contextmanager
    def record(self, upload_id):
        if not UPLOAD_ID.fullmatch(upload_id):
            raise ValueError("invalid upload ID")
        with (self.root / (upload_id + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            directory = self.root / upload_id
            path = directory / "upload.json"
            yield directory, path, json.loads(path.read_text()) if path.exists() else None

    def create(self, binding, digest, size, limit):
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or size <= 0 or exceeds(size, limit) or (size + self.part_bytes - 1) // self.part_bytes > self.max_parts:
            raise ApplicationError("archive exceeds upload limit or has an invalid identity", status_code=413, code="UPLOAD_LIMIT")
        self.prune()
        upload_id = "upload." + hashlib.sha256(json.dumps([binding, digest, size], sort_keys=True).encode()).hexdigest()
        with self.record(upload_id) as (directory, path, value):
            if value and value["status"] == "COMPLETED":
                return value
            if value and value["status"] == "UPLOADING" and not self.expired(value):
                return value
            if size + 64 * 1024 ** 2 > shutil.disk_usage(self.root).free:
                raise ApplicationError("insufficient upload staging space", status_code=507, code="UPLOAD_STORAGE")
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(mode=0o700)
            (directory / "parts").mkdir(mode=0o700)
            value = {"upload_id": upload_id, "binding": binding, "sha256": digest, "bytes": size,
                     "part_bytes": self.part_bytes, "part_count": (size + self.part_bytes - 1) // self.part_bytes,
                     "parts": {}, "status": "UPLOADING",
                     "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=self.ttl)).isoformat()}
            atomic_json(path, value)
            return value

    @staticmethod
    def expired(value):
        return datetime.fromisoformat(value["expires_at"]) <= datetime.now(timezone.utc)

    def checked(self, value, binding, *, active=False, allow_expired=False):
        if not value or value["binding"] != binding:
            raise ApplicationError("unknown upload for this target", status_code=404, code="UNKNOWN_UPLOAD")
        if not allow_expired and value["status"] != "COMPLETED" and self.expired(value):
            raise ApplicationError("upload expired; create it again", status_code=410, code="UPLOAD_EXPIRED")
        if active and value["status"] != "UPLOADING":
            raise ValueError("upload is not accepting parts")
        return value

    def read(self, upload_id, binding):
        with self.record(upload_id) as (_, _, value):
            return self.checked(value, binding)

    @staticmethod
    def part_size(value, number):
        if not 0 <= number < value["part_count"]:
            raise ValueError("invalid upload part number")
        return min(value["part_bytes"], value["bytes"] - number * value["part_bytes"])

    def part(self, upload_id, binding, number, temporary, digest, size):
        with self.record(upload_id) as (directory, path, value):
            self.checked(value, binding, active=True)
            if size != self.part_size(value, number):
                raise ValueError("upload part length differs")
            item = {"sha256": digest, "bytes": size}
            previous = value["parts"].get(str(number))
            if previous:
                if previous != item:
                    raise ValueError("upload part is already sealed with another digest")
                return previous
            destination = directory / "parts" / str(number)
            temporary.replace(destination)
            with destination.open("rb") as part:
                os.fsync(part.fileno())
            descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            value["parts"][str(number)] = item
            atomic_json(path, value)
            return item

    def complete(self, upload_id, binding, publish, reserve):
        with self.record(upload_id) as (directory, path, value):
            self.checked(value, binding)
            if value["status"] == "COMPLETED":
                shutil.rmtree(directory / "parts", ignore_errors=True)
                return value["result"]
            self.checked(value, binding, active=True)
            if len(value["parts"]) != value["part_count"]:
                raise ValueError("upload is missing parts")
            free = shutil.disk_usage(self.root).free
            if reserve > free:
                value["last_error"] = {"phase": "PUBLICATION_CAPACITY", "http_status": 507,
                                       "code": "UPLOAD_STORAGE", "required_free_bytes": reserve,
                                       "available_bytes": free, "observed_at": utc_now()}
                atomic_json(path, value)
                raise ApplicationError("insufficient space to validate and publish archive", status_code=507, code="UPLOAD_STORAGE")
            paths = [directory / "parts" / str(i) for i in range(value["part_count"])]
            with io.BufferedReader(PartStream(paths, value["part_bytes"], value["bytes"]), buffer_size=1024 ** 2) as stream:
                # Revalidate on completion, including parts recovered after a restart.
                whole = hashlib.sha256()
                for number in range(value["part_count"]):
                    digest = hashlib.sha256()
                    size = self.part_size(value, number)
                    remaining = size
                    while remaining:
                        data = stream.read(min(1024 ** 2, remaining))
                        digest.update(data)
                        whole.update(data)
                        remaining -= len(data)
                    if {"sha256": digest.hexdigest(), "bytes": size} != value["parts"][str(number)]:
                        raise ValueError("stored upload part checksum differs")
                if whole.hexdigest() != value["sha256"]:
                    raise ValueError("archive SHA256 differs from upload identity")
                stream.seek(0)
                result = publish(stream, value["bytes"], value["sha256"])
            value.update(status="COMPLETED", result=result)
            value.pop("last_error", None)
            atomic_json(path, value)
            shutil.rmtree(directory / "parts")
            return result

    def abort(self, upload_id, binding):
        with self.record(upload_id) as (directory, path, value):
            self.checked(value, binding, allow_expired=True)
            if value["status"] == "COMPLETED":
                raise ValueError("published uploads cannot be aborted")
            shutil.rmtree(directory / "parts", ignore_errors=True)
            value.update(status="ABORTED", parts={})
            atomic_json(path, value)
            return {"upload_id": upload_id, "status": "ABORTED"}

    def prune(self):
        # Only this feature's expired staging is removed. Published objects and
        # historical scientific records are never candidates for this cleanup.
        for directory in self.root.glob("upload.*"):
            if not directory.is_dir() or not UPLOAD_ID.fullmatch(directory.name):
                continue
            with (self.root / (directory.name + ".lock")).open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                path = directory / "upload.json"
                if not path.exists():
                    continue
                value = json.loads(path.read_text())
                if value["status"] == "UPLOADING" and self.expired(value):
                    shutil.rmtree(directory / "parts", ignore_errors=True)
                    value.update(status="EXPIRED", parts={})
                    atomic_json(path, value)
        for path in self.root.glob(".part-*"):
            if path.stat().st_mtime < time.time() - self.ttl:
                path.unlink(missing_ok=True)
