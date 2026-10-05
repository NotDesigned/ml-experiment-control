"""Stdlib-only data delivery and explicit, immutable checkpoint publication."""
from __future__ import annotations

import fcntl
import hashlib
import http.client
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.parse import urlsplit

if __package__:
    from .container_worker import archive_outputs, upload
else:
    from legacy_worker import archive_outputs, upload


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
            temporary = Path(tempfile.mkdtemp(prefix=".input-", dir=cache))
            try:
                expected = {f["path"]: f for f in item["files"]}
                seen = set()
                with tempfile.TemporaryFile(dir=cache) as stream:
                    if archive_stream is None:
                        fetch(item["url"], token, stream, item["sha256"], item["archive_bytes"])
                    else:
                        archive_stream.seek(0)
                        digest, length = digest_stream(archive_stream, stream)
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


def link_path(target, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and path.exists() and path.samefile(target):
        return
    if path.is_symlink():
        raise ValueError("container path is already bound to another location")
    path.symlink_to(target, target_is_directory=True)


def open_output(root, name):
    parts = relative_path(name).parts
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
        if not stat.S_ISREG(os.fstat(file).st_mode):
            os.close(file)
            raise ValueError("checkpoint must be a regular file")
        return os.fdopen(file, "rb")
    finally:
        os.close(descriptor)


def read_checkpoint_ready(root):
    with open_output(root, "checkpoint.ready.json") as body:
        ready = body.read(256 * 1024 + 1)
    if len(ready) > 256 * 1024:
        raise ValueError("checkpoint manifest is invalid")
    return ready


def checkpoint_archive(root, stream, limit=4 * 1024 ** 3):
    ready = read_checkpoint_ready(root)
    document = json.loads(ready)
    files = document["files"]
    if not files or len(files) > 20000:
        raise ValueError("checkpoint manifest file count is invalid")
    seen = set()
    total = 0
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for item in files:
            name = relative_path(item["path"]).as_posix()
            if name in seen or name == "checkpoint.ready.json":
                raise ValueError("checkpoint manifest contains duplicate/reserved paths")
            seen.add(name)
            total += int(item["bytes"])
            if total > limit - 1024 * 1024:
                raise ValueError("checkpoint archive exceeds upload limit")
            with open_output(root, name) as body:
                before = os.fstat(body.fileno())
                digest, size = digest_stream(body)
                if not stat.S_ISREG(before.st_mode) or digest != item["sha256"] or size != item["bytes"]:
                    raise ValueError("checkpoint file differs from ready manifest")
                body.seek(0)
                info = tarfile.TarInfo(name)
                info.size, info.mode = size, 0o400
                archive.addfile(info, body)
                after = os.fstat(body.fileno())
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    raise ValueError("checkpoint file changed during publication")
        info = tarfile.TarInfo("checkpoint.ready.json")
        info.size, info.mode = len(ready), 0o400
        archive.addfile(info, io.BytesIO(ready))
    return hashlib.sha256(ready).hexdigest()


def cache_checkpoint(stream, cache):
    """Seed the same verified asset cache before publishing the upload receipt."""
    stream.seek(0)
    digest, length = digest_stream(stream)
    stream.seek(0)
    with tarfile.open(fileobj=stream, mode="r:") as archive:
        ready = archive.extractfile("checkpoint.ready.json").read()
    files = json.loads(ready)["files"] + [{"path": "checkpoint.ready.json", "bytes": len(ready),
                                         "sha256": hashlib.sha256(ready).hexdigest()}]
    item = {"asset_id": "asset." + digest, "sha256": digest, "archive_bytes": length, "files": files}
    return deliver(item, None, cache, archive_stream=stream)


def main(argv=None):
    url = os.environ.pop("ML_EXPD_UPLOAD_URL")
    token = os.environ.pop("ML_EXPD_UPLOAD_TOKEN")
    limit = int(os.environ.pop("ML_EXPD_UPLOAD_LIMIT"))
    patterns = json.loads(os.environ.pop("ML_EXPD_OUTPUT_PATTERNS", '["**/*"]'))
    inputs = json.loads(os.environ.pop("ML_EXPD_INPUT_ASSETS", "[]"))
    snapshot_url = os.environ.pop("ML_EXPD_SNAPSHOT_URL", "")
    interval = int(os.environ.pop("ML_EXPD_SNAPSHOT_INTERVAL", "60"))
    root = Path(os.environ["OUTPUT_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        link_path(root, Path("/outputs"))
        for item in inputs:
            cache = root.parents[4] / "data-assets"
            print("ML_EXPD_INPUT_ASSET=START " + item["asset_id"] + " archive_bytes=" + str(item["archive_bytes"]), flush=True)
            location = deliver(item, token, cache)
            link_path(location, Path(item["mount_path"]))
            print("ML_EXPD_INPUT_ASSET=READY " + item["asset_id"], flush=True)
    except Exception:
        print("ML_EXPD_INPUT_DELIVERY=FAILED", file=sys.stderr, flush=True)
        return 65
    print("ML_EXPD_INPUT_DELIVERY_SECONDS=" + str(round(time.monotonic() - started, 3)), flush=True)
    child = subprocess.Popen(argv or sys.argv[1:], start_new_session=True)
    def forward(signum, _frame):
        if child.poll() is None:
            os.killpg(child.pid, signum)
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    previous, last_poll = None, 0.0
    while True:
        code = child.poll()
        if snapshot_url and (code is not None or time.monotonic() - last_poll >= interval):
            last_poll = time.monotonic()
            if (root / "checkpoint.ready.json").exists():
                try:
                    marker = hashlib.sha256(read_checkpoint_ready(root)).hexdigest()
                    if marker != previous:
                        with tempfile.TemporaryFile(dir=root.parent) as stream:
                            if checkpoint_archive(root, stream, limit) != marker:
                                raise ValueError("checkpoint publication marker changed")
                            length = stream.tell()
                            if length > limit:
                                raise ValueError("checkpoint archive exceeds upload limit")
                            try:
                                cache_checkpoint(stream, root.parents[4] / "data-assets")
                                print("ML_EXPD_CHECKPOINT_CACHE=READY", flush=True)
                            except Exception:
                                # A cache is optional. Upload remains authoritative.
                                print("ML_EXPD_CHECKPOINT_CACHE=FAILED", file=sys.stderr, flush=True)
                            upload(snapshot_url, token, stream, length)
                            previous = marker
                            print("ML_EXPD_CHECKPOINT_UPLOAD=COMPLETE", flush=True)
                except Exception:
                    print("ML_EXPD_CHECKPOINT_UPLOAD=FAILED", file=sys.stderr, flush=True)
        if code is not None:
            break
        time.sleep(1)
    try:
        with tempfile.TemporaryFile(dir=root.parent) as stream:
            archive_outputs(root, stream, limit - 1024 * 1024, patterns)
            length = stream.tell()
            if length > limit:
                raise ValueError("artifact archive limit exceeded")
            upload(url, token, stream, length)
        print("ML_EXPD_ARTIFACT_UPLOAD=COMPLETE", flush=True)
    except Exception:
        print("ML_EXPD_ARTIFACT_UPLOAD=FAILED", file=sys.stderr, flush=True)
        return code if code > 0 else 74
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    raise SystemExit(main())
