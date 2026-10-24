"""Stdlib-only checkpoint manifests; state bytes stay on backend storage."""
from __future__ import annotations

import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from urllib.parse import urlsplit


if __package__:
    from .worker_http import https_connection
else:
    from worker_http import https_connection

MANIFEST_LIMIT = 256 * 1024
CHECKPOINT_ID = re.compile(r"checkpoint\.[0-9a-f]{64}")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def manifest(value):
    if (not isinstance(value, dict) or set(value) != {"step", "files"}
            or type(value["step"]) is not int or value["step"] < 0
            or not isinstance(value["files"], list) or not 1 <= len(value["files"]) <= 20000
            or len(json.dumps(value).encode()) > MANIFEST_LIMIT):
        raise ValueError("invalid persistent checkpoint manifest")
    files, seen, generations = [], set(), set()
    for item in value["files"]:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise ValueError("invalid checkpoint file record")
        name = item["path"]
        if not isinstance(name, str):
            raise ValueError("invalid checkpoint file path")
        path = PurePosixPath(name)
        if (len(path.parts) < 3 or path.parts[0] != "checkpoints" or path.as_posix() != name
                or any(part.startswith(".") for part in path.parts) or "\\" in name or "\x00" in name
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", path.parts[1])):
            raise ValueError("checkpoint files must be in one immutable generation")
        if (name in seen or type(item["bytes"]) is not int or not 0 <= item["bytes"] < 2 ** 63
                or not isinstance(item["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
            raise ValueError("invalid checkpoint file identity")
        seen.add(name)
        generations.add(path.parts[1])
        files.append(dict(item))
    if len(generations) != 1:
        raise ValueError("checkpoint manifest mixes generations")
    return {"step": value["step"], "files": sorted(files, key=lambda item: item["path"])}


def document(context, ready):
    value = {"schema_version": 1, **context, **manifest(ready)}
    return {**value, "checkpoint_id": "checkpoint." + digest(value)}


def open_file(root, name):
    parts = PurePosixPath(name).parts
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise ValueError("checkpoint must contain regular files")
        return os.fdopen(fd, "rb")
    finally:
        os.close(descriptor)


def read_ready(root):
    with open_file(root, "checkpoint.ready.json") as body:
        raw = body.read(MANIFEST_LIMIT + 1)
    if len(raw) > MANIFEST_LIMIT:
        raise ValueError("checkpoint ready marker exceeds limit")
    return manifest(json.loads(raw))


def verify(root, ready, *, seal=False):
    ready = manifest(ready)
    generation = root.joinpath(*PurePosixPath(ready["files"][0]["path"]).parts[:2])
    expected = {item["path"] for item in ready["files"]}
    found = set()
    for path in generation.rglob("*"):
        mode = path.lstat().st_mode
        if not stat.S_ISDIR(mode):
            if not stat.S_ISREG(mode):
                raise ValueError("checkpoint generation contains a link or special file")
            found.add(path.relative_to(root).as_posix())
    if found != expected:
        raise ValueError("checkpoint generation file set differs")
    for item in ready["files"]:
        with open_file(root, item["path"]) as stream:
            before = os.fstat(stream.fileno())
            sha, size = hashlib.sha256(), 0
            while chunk := stream.read(1024 * 1024):
                sha.update(chunk)
                size += len(chunk)
            after = os.fstat(stream.fileno())
        if (sha.hexdigest() != item["sha256"] or size != item["bytes"]
                or (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
            raise ValueError("checkpoint file checksum or stability differs")
    if seal:
        for path in generation.rglob("*"):
            path.chmod(0o555 if path.is_dir() else 0o444)
        generation.chmod(0o555)
    return generation


def restore(value):
    context = {key: value[key] for key in ("project", "run_id", "attempt_id", "source_id", "image_id", "storage_scope", "state_root")}
    ready = {key: value[key] for key in ("step", "files")}
    if document(context, ready)["checkpoint_id"] != value["checkpoint_id"]:
        raise ValueError("restore checkpoint identity differs")
    return verify(Path(value["state_root"]), ready)


def register(url, token, ready, expected):
    target = urlsplit(url)
    if target.scheme != "https" or target.username or target.password or target.query or target.fragment:
        raise ValueError("checkpoint registration requires a fixed HTTPS endpoint")
    connection = https_connection(target, timeout=30)
    try:
        body = json.dumps(ready).encode()
        connection.request("PUT", target.path, body=body,
                           headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        response = connection.getresponse()
        raw = response.read(MANIFEST_LIMIT * 2 + 1)
        if response.status != 200 or len(raw) > MANIFEST_LIMIT * 2:
            raise ValueError("checkpoint registration rejected")
        receipt = json.loads(raw)
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError("checkpoint registration identity differs")
        return receipt
    finally:
        connection.close()


def publish(context, url, token):
    root = Path(context["state_root"])
    ready = read_ready(root)
    value = document(context, ready)
    verify(root, ready, seal=True)
    receipt = register(url, token, ready, value)
    receipts = root / "receipts"
    receipts.mkdir(exist_ok=True)
    temporary = receipts / (value["checkpoint_id"] + ".tmp")
    temporary.write_text(json.dumps(receipt, sort_keys=True))
    temporary.replace(receipts / (value["checkpoint_id"] + ".json"))
    return receipt
