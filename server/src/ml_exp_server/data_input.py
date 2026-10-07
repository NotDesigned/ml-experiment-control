"""Stdlib-only verified input delivery shared by CPU copying and training."""
from __future__ import annotations

from contextlib import nullcontext
import fcntl
import hashlib
import http.client
import json
import ssl
import time
from datetime import datetime, timezone
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from urllib.parse import urlsplit


class InputDeliveryError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class DeliveryTrace:
    """Safe phase/byte evidence, with no URLs, response bodies or exception text."""
    def __init__(self, asset_id="none", expected_bytes=0):
        self.started = time.monotonic()
        self.last_emit = -10.0
        self.value = {"asset_id": asset_id, "expected_bytes": expected_bytes, "received_bytes": 0}

    def report(self, phase, *, force=True, **fields):
        self.value.update(phase=phase, **fields)
        elapsed = time.monotonic() - self.started
        if force or elapsed - self.last_emit >= 5:
            self.last_emit = elapsed
            self.value.update(at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=round(elapsed, 3),
                              bytes_per_second=round(self.value["received_bytes"] / max(elapsed, 0.001), 1))
            print("ML_EXPD_INPUT_PROGRESS=" + json.dumps(self.value, sort_keys=True), flush=True)

    def failed(self, error):
        code = ("INPUT_TIMEOUT" if isinstance(error, TimeoutError) else
                "INPUT_TLS_FAILED" if isinstance(error, ssl.SSLError) else
                "INPUT_NETWORK_OR_STORAGE_FAILED" if isinstance(error, OSError) else
                "INPUT_VALIDATION_FAILED")
        if isinstance(error, InputDeliveryError):
            code = error.code
        phase = self.value.get("phase", "INITIALIZING")
        fields = {"code": code, "error_class": type(error).__name__, "failed_phase": phase}
        if isinstance(error, OSError) and error.errno is not None:
            fields["errno"] = error.errno
        self.report("FAILED", **fields)


def report(trace, phase, **fields):
    if trace is not None:
        trace.report(phase, **fields)


def digest_stream(stream, output=None):
    digest, size = hashlib.sha256(), 0
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
        if output is not None:
            output.write(chunk)
    return digest.hexdigest(), size


def relative_path(name):
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in name or any(p.startswith(".") for p in path.parts):
        raise ValueError("invalid asset file path")
    return path


def verify_tree(tree, files):
    if tree.is_symlink() or {p.relative_to(tree).as_posix() for p in tree.rglob("*") if p.is_file() or p.is_symlink()} != {f["path"] for f in files}:
        raise InputDeliveryError("INPUT_CACHE_FILE_SET_MISMATCH", "input cache file set differs")
    for item in files:
        path = tree.joinpath(*relative_path(item["path"]).parts)
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(tree.resolve()):
            raise InputDeliveryError("INPUT_FILE_MISSING_OR_ESCAPED", "input asset file is missing or escaped")
        with path.open("rb") as stream:
            actual, size = digest_stream(stream)
        if actual != item["sha256"] or size != item["bytes"]:
            raise InputDeliveryError("INPUT_FILE_SHA256_MISMATCH", "input asset file checksum differs")


def fetch(url, token, stream, expected_sha, expected_size, trace=None):
    report(trace, "DOWNLOADING")
    target = urlsplit(url)
    if target.scheme != "https" or target.username or target.password or target.query or target.fragment:
        raise ValueError("data delivery requires a fixed HTTPS endpoint")
    connection = http.client.HTTPSConnection(target.hostname, target.port or 443, timeout=300)
    try:
        connection.request("GET", target.path, headers={"Authorization": "Bearer " + token})
        response = connection.getresponse()
        report(trace, "DOWNLOADING", http_status=response.status)
        if response.status != 200 or int(response.getheader("Content-Length", "-1")) != expected_size:
            raise InputDeliveryError("INPUT_HTTP_OR_LENGTH_REJECTED", "input archive download rejected or size differs")
        remaining, digest = expected_size, hashlib.sha256()
        while remaining:
            chunk = response.read(min(1024 * 1024, remaining))
            if not chunk:
                raise InputDeliveryError("INPUT_ARCHIVE_TRUNCATED", "input archive is truncated")
            remaining -= len(chunk)
            digest.update(chunk)
            stream.write(chunk)
            report(trace, "DOWNLOADING", force=False, received_bytes=expected_size - remaining)
        if digest.hexdigest() != expected_sha:
            raise InputDeliveryError("INPUT_ARCHIVE_SHA256_MISMATCH", "input archive checksum differs")
        stream.seek(0)
    finally:
        connection.close()


def deliver(item, token, cache, *, archive_stream=None, trace=None):
    asset_id = item["asset_id"]
    if asset_id != "asset." + item["sha256"]:
        raise ValueError("input asset identity differs")
    report(trace, "WAITING_FOR_CACHE")
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / asset_id
    with (cache / (asset_id + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.exists():
            if item.get("require_cached"):
                raise InputDeliveryError("INPUT_PREPARED_CACHE_MISSING", "prepared backend data cache is missing")
            temporary = Path(tempfile.mkdtemp(prefix=".input-", dir=cache))
            try:
                expected = {f["path"]: f for f in item["files"]}
                seen = set()
                with (tempfile.TemporaryFile(dir=cache) if archive_stream is None else nullcontext(archive_stream)) as stream:
                    if archive_stream is None:
                        fetch(item["url"], token, stream, item["sha256"], item["archive_bytes"], trace)
                    else:
                        archive_stream.seek(0)
                        digest, length = digest_stream(archive_stream)
                        if digest != item["sha256"] or length != item["archive_bytes"]:
                            raise ValueError("local checkpoint archive differs")
                        stream.seek(0)
                    report(trace, "EXTRACTING")
                    with tarfile.open(fileobj=stream, mode="r:*") as archive:
                        for member in archive:
                            if member.isdir() and member.name in {".", "./"}:
                                continue
                            path = relative_path(member.name)
                            if member.isdir():
                                continue
                            name = path.as_posix()
                            if not member.isfile() or name in seen or name not in expected or member.size != expected[name]["bytes"]:
                                raise InputDeliveryError("INPUT_ARCHIVE_MANIFEST_MISMATCH", "input archive differs from its manifest")
                            seen.add(name)
                            output = temporary.joinpath(*path.parts)
                            output.parent.mkdir(parents=True, exist_ok=True)
                            with archive.extractfile(member) as source, output.open("xb") as body:
                                digest_stream(source, body)
                if seen != set(expected):
                    raise InputDeliveryError("INPUT_ARCHIVE_FILE_SET_MISMATCH", "input archive file set differs")
                report(trace, "VERIFYING")
                verify_tree(temporary, item["files"])
                for path in temporary.rglob("*"):
                    path.chmod(0o555 if path.is_dir() else 0o444)
                report(trace, "PUBLISHING")
                temporary.rename(destination)
                destination.chmod(0o555)
            finally:
                if temporary.exists():
                    for p in temporary.rglob("*"):
                        if p.is_dir():
                            p.chmod(0o755)
                    shutil.rmtree(temporary)
        report(trace, "VERIFYING")
        verify_tree(destination, item["files"])
    return destination
