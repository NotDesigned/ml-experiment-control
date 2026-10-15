"""One process-shared ceiling for unacknowledged upload transport bytes.

Sealed parts survive a daemon restart until the destination confirms their
digest and fsync. Incomplete request bodies are disposable transport scratch.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from ..application_errors import ApplicationError
from ..storage import atomic_json

BUFFER_BYTES = 1024 ** 3
LOCK_STRIPES = 128


def busy():
    return ApplicationError("transfer buffer is full or this part is in flight; resume pending uploads",
                            status_code=503, code="TRANSFER_BUFFER_BUSY")


def checked_receipt(state, receipt):
    expected = {key: state[key] for key in ("upload_id", "number", "sha256", "bytes")}
    if (not isinstance(receipt, dict) or receipt.get("contract") != "upload-part.v1"
            or receipt.get("durable") is not True
            or any(receipt.get(key) != value for key, value in expected.items())):
        raise ValueError("remote durable upload receipt differs")


def checked_state(value, metadata):
    if (not isinstance(value, dict) or value.get("status") not in {"RECEIVING", "SEALED", "CONFIRMED"}
            or not isinstance(value.get("upload_id"), str) or not re.fullmatch(r"upload\.[0-9a-f]{64}", value["upload_id"])
            or type(value.get("number")) is not int or not 0 <= value["number"] < 65536
            or not isinstance(value.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
            or type(value.get("bytes")) is not int or not 0 < value["bytes"] <= BUFFER_BYTES
            or set(value) - {"upload_id", "number", "sha256", "bytes", "status", "receipt"}):
        raise ValueError("invalid durable transfer buffer state")
    key = hashlib.sha256((value["upload_id"] + ":" + str(value["number"])).encode()).hexdigest()
    if metadata.name != "part-" + key + ".json":
        raise ValueError("durable transfer buffer identity differs")
    return value


class TransferLease:
    def __init__(self, buffer, path, state):
        self.buffer, self.path, self.state = buffer, path, state
        self.sealed = state.get("status") == "SEALED"

    def seal(self):
        digest, size = hashlib.sha256(), 0
        with self.path.open("rb") as stream:
            while chunk := stream.read(1024 ** 2):
                digest.update(chunk)
                size += len(chunk)
            os.fsync(stream.fileno())
        if size != self.state["bytes"] or digest.hexdigest() != self.state["sha256"]:
            raise ValueError("local buffered part length or SHA256 differs")
        self.state["status"] = "SEALED"
        atomic_json(self.path.with_suffix(".json"), self.state)
        self.sealed = True

    def acknowledge(self, receipt):
        if not self.sealed:
            raise ValueError("remote durable upload receipt differs")
        checked_receipt(self.state, receipt)
        # Persist proof before removing the only local transport copy. Recovery
        # may safely finish this cleanup if the daemon exits between operations.
        with self.buffer.locked():
            proof = {key: receipt[key] for key in ("contract", "upload_id", "number", "sha256", "bytes", "durable")}
            self.state.update(status="CONFIRMED", receipt=proof)
            atomic_json(self.path.with_suffix(".json"), self.state)
            self.buffer.discard(self.path)

    def discard_deleted_run(self, binding):
        # Only the authenticated worker part handler calls this while holding
        # this exact lease. A permanent administrator fence ends this transfer;
        # expired, unknown and dataset uploads retain the ordinary proof rule.
        from ..deletion_state import RunFence
        if binding.get("kind") not in {"artifacts", "checkpoint"}:
            return False
        fence = RunFence(self.buffer.root.parent)
        if not fence.is_deleted(binding.get("project"), binding.get("run_id")):
            return False
        with self.buffer.locked():
            self.buffer.discard(self.path)
        self.sealed = False
        return True


class TransferBuffer:
    def __init__(self, root, *, limit=BUFFER_BYTES):
        if type(limit) is not int or not 1 <= limit <= BUFFER_BYTES:
            raise ValueError("transfer buffer limit must not exceed 1 GiB")
        self.root = Path(root) / "transfer-buffer"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.limit = limit
        self.block_bytes = os.statvfs(self.root).f_frsize
        if type(self.block_bytes) is not int or self.block_bytes <= 0:
            raise ValueError("invalid transfer buffer filesystem block size")
        # Fixed striped locks bound directory/inode growth. Per-part accounting
        # covers block rounding, two bounded JSON copies and directory entries.
        self.fixed_bytes = (LOCK_STRIPES + 1) * self.block_bytes

    def charged(self, size):
        sidecar_blocks = (2048 + self.block_bytes - 1) // self.block_bytes
        return ((size + self.block_bytes - 1) // self.block_bytes + 2 * sidecar_blocks + 1) * self.block_bytes

    def part_lock(self, key):
        stripe = int(key[:8], 16) % LOCK_STRIPES
        return self.root / ("stripe-" + str(stripe).zfill(3) + ".lock")

    @contextmanager
    def locked(self):
        with (self.root / "admission.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def discard(self, path):
        path.unlink(missing_ok=True)
        path.with_suffix(".json").unlink(missing_ok=True)
        descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def snapshot(self):
        # Caller holds admission.lock. A crashed SEALED part remains charged;
        # a crashed incomplete body has never become a valid upload part.
        for temporary in self.root.glob(".part-*.tmp"):
            match = re.fullmatch(r"\.part-([0-9a-f]{64})\.json\.[0-9a-f]{32}\.tmp", temporary.name)
            if match is None:
                continue
            with self.part_lock(match.group(1)).open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                temporary.unlink(missing_ok=True)  # Uncommitted atomic JSON scratch, never part bytes.
        reserved, sealed, charged = 0, 0, self.fixed_bytes
        for metadata in self.root.glob("part-*.json"):
            value = checked_state(json.loads(metadata.read_text()), metadata)
            path = metadata.with_suffix(".part")
            with self.part_lock(metadata.stem[5:]).open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    reserved += value["bytes"]
                    charged += self.charged(value["bytes"])
                else:
                    if value["status"] == "SEALED":
                        reserved += value["bytes"]
                        sealed += value["bytes"]
                        charged += self.charged(value["bytes"])
                    else:
                        if value["status"] == "CONFIRMED":
                            checked_receipt(value, value.get("receipt"))
                        self.discard(path)
        return {"limit_bytes": self.limit, "reserved_bytes": reserved, "payload_bytes": reserved,
                "charged_bytes": charged, "fixed_overhead_bytes": self.fixed_bytes,
                "filesystem_block_bytes": self.block_bytes,
                "pending_bytes": sealed, "available_bytes": max(0, self.limit - charged)}

    @contextmanager
    def reserve(self, upload_id, number, digest, size):
        if (not re.fullmatch(r"upload\.[0-9a-f]{64}", upload_id) or type(number) is not int or not 0 <= number < 65536
                or not re.fullmatch(r"[0-9a-f]{64}", digest) or type(size) is not int or not 0 < size <= self.limit):
            raise ValueError("invalid bounded transfer part identity")
        key = hashlib.sha256((upload_id + ":" + str(number)).encode()).hexdigest()
        path = self.root / ("part-" + key + ".part")
        metadata = path.with_suffix(".json")
        expected = {"upload_id": upload_id, "number": number, "sha256": digest, "bytes": size}
        lock = self.part_lock(key).open("a")
        try:
            with self.locked():
                # Exclude this part from a second reservation; acquiring its
                # lock first also serializes retries from several API workers.
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise busy() from None
                value = checked_state(json.loads(metadata.read_text()), metadata) if metadata.exists() else None
                if value and any(value.get(key) != item for key, item in expected.items()):
                    raise ValueError("buffered part is bound to another digest")
                charged = self.charged(size) if value else 0
                snapshot = self.snapshot()
                if snapshot["charged_bytes"] - charged + self.charged(size) > self.limit:
                    raise busy()
                if shutil.disk_usage(self.root).free < (0 if value else size):
                    raise ApplicationError("insufficient transfer buffer space", status_code=507, code="UPLOAD_STORAGE")
                if value and value.get("status") == "CONFIRMED":
                    checked_receipt(value, value.get("receipt"))
                    self.discard(path)
                    value = None
                state = value or {**expected, "status": "RECEIVING"}
                atomic_json(metadata, state)
            lease = TransferLease(self, path, state)
            try:
                yield lease
            finally:
                # Valid complete bytes remain for explicit resume when a remote
                # response is lost. Invalid/truncated requests can be retried.
                if not lease.sealed:
                    with self.locked():
                        self.discard(path)
        finally:
            lock.close()

    def abort(self, upload_id):
        with self.locked():
            for metadata in self.root.glob("part-*.json"):
                if json.loads(metadata.read_text()).get("upload_id") != upload_id:
                    continue
                with self.part_lock(metadata.stem[5:]).open("a") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        raise busy() from None
                    self.discard(metadata.with_suffix(".part"))

    def pending(self, upload_id):
        """Return only this session's sealed transport claims, never their bytes."""
        with self.locked():
            self.snapshot()
            return [value for metadata in self.root.glob("part-*.json")
                    if (value := checked_state(json.loads(metadata.read_text()), metadata))["upload_id"] == upload_id
                    and value["status"] == "SEALED"]
