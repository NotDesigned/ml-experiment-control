"""Durable, bounded archive uploads; publish only after whole-archive validation."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import time
from uuid import uuid4

from ..application_errors import ApplicationError
from ..storage import atomic_json, utc_now
from ..archive_limits import exceeds
from ..storage_capacity import StorageCapacity, CapacityError

UPLOAD_ID = re.compile(r"^upload\.[0-9a-f]{64}$")
PART_BYTES = 16 * 1024 ** 2
RMTREE_DIRECTORY_FD = sys.version_info >= (3, 11)


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
    @staticmethod
    def identity(binding, digest, size):
        return "upload." + hashlib.sha256(json.dumps([binding, digest, size], sort_keys=True).encode()).hexdigest()

    def __init__(self, root, config, *, capacity=None):
        self.root = Path(root) / "multipart-uploads"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.capacity = capacity if capacity is not None else StorageCapacity(root, config)
        self.part_bytes = int(config.get("upload_part_bytes", PART_BYTES))
        self.ttl = int(config.get("upload_session_seconds", 86400))
        self.max_parts = int(config.get("upload_max_parts", 65536))
        if (not 1024 <= self.part_bytes <= 64 * 1024 ** 2 or not 300 <= self.ttl <= 604800
                or not 1 <= self.max_parts <= 65536):
            raise ValueError("invalid multipart upload policy")

    @contextmanager
    def run_binding_guard(self, binding):
        if binding.get("kind") in {"artifacts", "checkpoint"}:
            # The standalone desktop asset helper intentionally ships only
            # stdlib storage modules and never handles Run-bound capabilities.
            from ..deletion_state import RunFence
            with RunFence(self.root.parent).guard(binding["project"], binding["run_id"]):
                yield
        else:
            yield

    @contextmanager
    def record(self, upload_id):
        if not UPLOAD_ID.fullmatch(upload_id):
            raise ValueError("invalid upload ID")
        with (self.root / (upload_id + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            directory = self.root / upload_id
            path = directory / "upload.json"
            if directory.is_symlink() or path.is_symlink() or (directory / "parts").is_symlink():
                raise ValueError("multipart upload state contains a link")
            yield directory, path, json.loads(path.read_text()) if path.exists() else None

    def create(self, binding, digest, size, limit):
        with self.run_binding_guard(binding):
            return self._create(binding, digest, size, limit)

    def _create(self, binding, digest, size, limit):
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or size <= 0 or exceeds(size, limit) or (size + self.part_bytes - 1) // self.part_bytes > self.max_parts:
            raise ApplicationError("archive exceeds upload limit or has an invalid identity", status_code=413, code="UPLOAD_LIMIT")
        self.prune()
        upload_id = self.identity(binding, digest, size)
        with self.record(upload_id) as (directory, path, value):
            if value and value["status"] == "COMPLETED":
                return value
            if value and value["status"] == "UPLOADING" and not self.expired(value):
                return value
            value = {"upload_id": upload_id, "binding": binding, "sha256": digest, "bytes": size,
                     "part_bytes": self.part_bytes, "part_count": (size + self.part_bytes - 1) // self.part_bytes,
                     "parts": {}, "status": "UPLOADING",
                     "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=self.ttl)).isoformat()}
            with self.capacity.locked():
                self.capacity.check(size, ignore_upload=upload_id)
                if directory.exists():
                    shutil.rmtree(directory)
                directory.mkdir(mode=0o700)
                (directory / "parts").mkdir(mode=0o700)
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
        with self.run_binding_guard(binding):
            return self._part(upload_id, binding, number, temporary, digest, size)

    def _part(self, upload_id, binding, number, temporary, digest, size):
        with self.record(upload_id) as (directory, path, value):
            self.checked(value, binding, active=True)
            if size != self.part_size(value, number):
                raise ValueError("upload part length differs")
            item = {"sha256": digest, "bytes": size}
            previous = value["parts"].get(str(number))
            destination = directory / "parts" / str(number)
            if previous:
                if previous != item:
                    raise ValueError("upload part is already sealed with another digest")
                self.verify_part(destination, item)
                return previous
            self.verify_part(temporary, item)
            temporary.replace(destination)
            with destination.open("rb") as part:
                os.fsync(part.fileno())
            descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            with self.capacity.locked():
                value["parts"][str(number)] = item
                atomic_json(path, value)
            return item

    @staticmethod
    def verify_part(path, expected):
        digest, size = hashlib.sha256(), 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 ** 2):
                digest.update(chunk)
                size += len(chunk)
            os.fsync(stream.fileno())
        if {"sha256": digest.hexdigest(), "bytes": size} != expected:
            raise ValueError("stored upload part checksum differs")

    def complete(self, upload_id, binding, publish, reserve):
        with self.run_binding_guard(binding):
            return self._complete(upload_id, binding, publish, reserve)

    def _complete(self, upload_id, binding, publish, reserve):
        with self.record(upload_id) as (directory, path, value):
            self.checked(value, binding)
            if value["status"] == "COMPLETED":
                shutil.rmtree(directory / "parts", ignore_errors=True)
                return value["result"]
            self.checked(value, binding, active=True)
            if len(value["parts"]) != value["part_count"]:
                raise ValueError("upload is missing parts")
            try:
                with self.capacity.reserve(reserve, ignore_upload=upload_id):
                    return self.publish(directory, path, value, publish)
            except CapacityError as error:
                value["last_error"] = {"phase": "PUBLICATION_CAPACITY", "http_status": 507,
                                       "code": error.code, "required_free_bytes": error.required_free_bytes,
                                       "available_bytes": error.available_bytes, "reserved_bytes": error.reserved_bytes,
                                       "observed_at": utc_now()}
                atomic_json(path, value)
                raise

    def publish(self, directory, path, value, publish):
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
        with self.run_binding_guard(binding):
            return self._abort(upload_id, binding)

    def _abort(self, upload_id, binding):
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
        root = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            with os.scandir(root) as entries:
                names = [entry.name for entry in entries]
            for name in names:
                if not UPLOAD_ID.fullmatch(name):
                    continue
                try:
                    lock = os.open(name + ".lock", os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW, 0o600, dir_fd=root)
                    directory = None
                    with os.fdopen(lock, "a") as stream:
                        try:
                            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            continue
                        directory = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root)
                        try:
                            descriptor = os.open("upload.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
                            with os.fdopen(descriptor) as record:
                                value = json.load(record)
                            if (value.get("upload_id") != name
                                    or self.identity(value.get("binding"), value.get("sha256"), value.get("bytes")) != name):
                                continue
                            try:
                                parts = os.stat("parts", dir_fd=directory, follow_symlinks=False)
                            except FileNotFoundError:
                                parts = None
                            if parts is not None and not stat.S_ISDIR(parts.st_mode):
                                continue
                            if value["status"] == "UPLOADING" and self.expired(value):
                                if parts is not None:
                                    self.remove_expired_parts(directory)
                                value.update(status="EXPIRED", parts={})
                                self.expire_record(directory, value)
                        finally:
                            os.close(directory)
                except OSError as error:
                    if error.errno not in {errno.ENOENT, errno.ELOOP, errno.ENOTDIR}:
                        raise
            for name in names:
                if not name.startswith(".part-"):
                    continue
                try:
                    value = os.stat(name, dir_fd=root, follow_symlinks=False)
                    if stat.S_ISREG(value.st_mode) and value.st_mtime < time.time() - self.ttl:
                        os.unlink(name, dir_fd=root)
                except FileNotFoundError:
                    pass
        finally:
            os.close(root)

    @staticmethod
    def remove_expired_parts(directory):
        if not shutil.rmtree.avoids_symlink_attacks:
            raise ValueError("multipart expiry requires fd-safe directory removal")
        if RMTREE_DIRECTORY_FD:
            shutil.rmtree("parts", dir_fd=directory)
        else:
            # Python 3.10 has the same fd-safe rmtree engine but no dir_fd
            # argument. Linux procfs pins its parent to our open descriptor.
            shutil.rmtree("/proc/self/fd/" + str(directory) + "/parts")

    @staticmethod
    def expire_record(directory, value):
        """Atomically write inside the already-open, no-follow staging folder."""
        temporary = ".upload-" + uuid4().hex + ".tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, "upload.json", src_dir_fd=directory, dst_dir_fd=directory)
            os.fsync(directory)
        finally:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass
