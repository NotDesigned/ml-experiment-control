"""Stdlib-only verified input delivery shared by CPU copying and training."""
from __future__ import annotations

from contextlib import nullcontext
import fcntl
import hashlib
import http.client
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from urllib.parse import urlsplit


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
        raise ValueError("input cache file set differs")
    for item in files:
        path = tree.joinpath(*relative_path(item["path"]).parts)
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(tree.resolve()):
            raise ValueError("input asset file is missing or escaped")
        with path.open("rb") as stream:
            actual, size = digest_stream(stream)
        if actual != item["sha256"] or size != item["bytes"]:
            raise ValueError("input asset file checksum differs")


def fetch(url, token, stream, expected_sha, expected_size):
    target = urlsplit(url)
    if target.scheme != "https" or target.username or target.password or target.query or target.fragment:
        raise ValueError("data delivery requires a fixed HTTPS endpoint")
    connection = http.client.HTTPSConnection(target.hostname, target.port or 443, timeout=300)
    try:
        connection.request("GET", target.path, headers={"Authorization": "Bearer " + token})
        response = connection.getresponse()
        if response.status != 200 or int(response.getheader("Content-Length", "-1")) != expected_size:
            raise ValueError("input archive download rejected or size differs")
        remaining, digest = expected_size, hashlib.sha256()
        while remaining:
            chunk = response.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("input archive is truncated")
            remaining -= len(chunk)
            digest.update(chunk)
            stream.write(chunk)
        if digest.hexdigest() != expected_sha:
            raise ValueError("input archive checksum differs")
        stream.seek(0)
    finally:
        connection.close()


def deliver(item, token, cache, *, archive_stream=None):
    asset_id = item["asset_id"]
    if asset_id != "asset." + item["sha256"]:
        raise ValueError("input asset identity differs")
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / asset_id
    with (cache / (asset_id + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.exists():
            if item.get("require_cached"):
                raise ValueError("prepared backend data cache is missing")
            temporary = Path(tempfile.mkdtemp(prefix=".input-", dir=cache))
            try:
                expected = {f["path"]: f for f in item["files"]}
                seen = set()
                with (tempfile.TemporaryFile(dir=cache) if archive_stream is None else nullcontext(archive_stream)) as stream:
                    if archive_stream is None:
                        fetch(item["url"], token, stream, item["sha256"], item["archive_bytes"])
                    else:
                        archive_stream.seek(0)
                        digest, length = digest_stream(archive_stream)
                        if digest != item["sha256"] or length != item["archive_bytes"]:
                            raise ValueError("local checkpoint archive differs")
                        stream.seek(0)
                    with tarfile.open(fileobj=stream, mode="r:*") as archive:
                        for member in archive:
                            if member.isdir() and member.name in {".", "./"}:
                                continue
                            path = relative_path(member.name)
                            if member.isdir():
                                continue
                            name = path.as_posix()
                            if not member.isfile() or name in seen or name not in expected or member.size != expected[name]["bytes"]:
                                raise ValueError("input archive differs from its manifest")
                            seen.add(name)
                            output = temporary.joinpath(*path.parts)
                            output.parent.mkdir(parents=True, exist_ok=True)
                            with archive.extractfile(member) as source, output.open("xb") as body:
                                digest_stream(source, body)
                if seen != set(expected):
                    raise ValueError("input archive file set differs")
                verify_tree(temporary, item["files"])
                for path in temporary.rglob("*"):
                    path.chmod(0o555 if path.is_dir() else 0o444)
                temporary.rename(destination)
                destination.chmod(0o555)
            finally:
                if temporary.exists():
                    for p in temporary.rglob("*"):
                        if p.is_dir():
                            p.chmod(0o755)
                    shutil.rmtree(temporary)
        verify_tree(destination, item["files"])
    return destination
