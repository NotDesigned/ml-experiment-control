"""Fixed stdlib Slurm receiver using the existing durable UploadStore engine."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile
from types import SimpleNamespace

from ..application_errors import ApplicationError
from ..data.multipart_upload import UploadStore
from ..storage import atomic_json, utc_now
from .data_input import deliver, digest_stream, relative_path, verify_tree


def directory(path):
    """Create only normal absolute directory components; never follow links."""
    path = Path(path)
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path


def target_root(metadata):
    target, binding = metadata["target"], metadata["binding"]
    root, project = target["project_root"], binding["project"]
    path = PurePosixPath(root)
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", project)
            or not path.is_absolute() or path.as_posix() != root or ".." in path.parts
            or "\\" in root or "\x00" in root or path.name != project
            or target["backend"] != "slurm" or binding.get("kind") != "asset"
            or binding.get("target") != {key: target[key] for key in
                                        ("executor", "backend", "storage_scope", "profile_sha256")}):
        raise ValueError("remote upload target binding differs")
    return directory(root)


def inventory(stream, *, expanded_limit, file_limit):
    """Build the independent archive manifest before existing verified delivery."""
    stream.seek(0)
    seen, files, total = set(), [], 0
    with tarfile.open(fileobj=stream, mode="r|*") as archive:
        for member in archive:
            if member.isdir() and member.name in {".", "./"}:
                continue
            path = relative_path(member.name).as_posix()
            if "\x00" in member.name or path in seen:
                raise ValueError("archive contains a duplicate or invalid path")
            seen.add(path)
            if len(seen) > file_limit:
                raise ValueError("archive file count exceeds policy")
            if member.isdir():
                continue
            if not member.isfile() or member.size < 0:
                raise ValueError("archive contains a link or special file")
            total += member.size
            if expanded_limit is not None and total > expanded_limit:
                raise ValueError("expanded archive exceeds policy")
            body = archive.extractfile(member)
            if body is None:
                raise ValueError("archive file is unavailable")
            with body:
                digest, size = digest_stream(body)
            if size != member.size:
                raise ValueError("archive member is truncated")
            files.append({"path": path, "bytes": size, "sha256": digest})
    if not files:
        raise ValueError("archive contains no files")
    return sorted(files, key=lambda item: item["path"])


def sync_tree(path):
    for child in path.rglob("*"):
        with child.open("rb") if child.is_file() else directory_fd(child) as stream:
            os.fsync(stream.fileno())
    with directory_fd(path) as stream:
        os.fsync(stream.fileno())


@contextmanager
def directory_fd(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        yield SimpleNamespace(fileno=lambda: descriptor)
    finally:
        os.close(descriptor)


def publish(store, metadata, stream, size, digest):
    target, project = metadata["target"], metadata["binding"]["project"]
    root = Path(target["project_root"])
    limit = metadata.get("expanded_limit")
    if limit is None:
        limit = max(0, shutil.disk_usage(root).free - store.capacity.headroom)
    files = inventory(stream, expanded_limit=limit, file_limit=metadata["file_limit"])
    declared = metadata.get("files")
    if declared is not None and sorted(declared, key=lambda item: item["path"]) != files:
        raise ValueError("declared archive files differ from verified remote bytes")
    asset_id = "asset." + digest
    archives = directory(root / "data-archives")
    archive = archives / (asset_id + ".tar")
    cache = directory(root / "data-assets")
    records = directory(root / "data-asset-metadata")
    item = {"asset_id": asset_id, "sha256": digest, "archive_bytes": size, "files": files}
    with store.capacity.reserve(sum(item["bytes"] for item in files), ignore_upload=metadata["upload_id"]):
        if archive.exists():
            if archive.is_symlink():
                raise ValueError("remote archive is a link")
            with archive.open("rb") as body:
                if digest_stream(body) != (digest, size):
                    raise ValueError("remote archive identity differs")
        else:
            descriptor, temporary = tempfile.mkstemp(prefix=".archive-", dir=archives)
            try:
                stream.seek(0)
                with os.fdopen(descriptor, "wb") as output:
                    shutil.copyfileobj(stream, output, 1024 ** 2)
                    output.flush()
                    os.fsync(output.fileno())
                os.rename(temporary, archive)
                with directory_fd(archives) as parent:
                    os.fsync(parent.fileno())
            finally:
                Path(temporary).unlink(missing_ok=True)
        with archive.open("rb") as body:
            tree = deliver(item, None, cache, archive_stream=body)
        sync_tree(tree)
        with directory_fd(cache) as parent:
            os.fsync(parent.fileno())
        value = {"project": project, **item, "status": "READY", "bytes": sum(item["bytes"] for item in files),
                 "created_at": utc_now(), "remote_storage": "slurm", "storage_scope": target["storage_scope"],
                 "executor": target["executor"], "verification": "remote-sha256-fsync",
                 "remote_locator": {"storage_scope": target["storage_scope"], "archive_path": str(archive),
                                    "data_path": str(tree)}}
        record = records / (asset_id + ".json")
        if record.exists():
            previous = json.loads(record.read_text())
            if any(previous[key] != value[key] for key in ("project", "asset_id", "sha256", "archive_bytes", "files")):
                raise ValueError("remote READY asset identity differs")
            return {**previous, "executor": target["executor"]}
        atomic_json(record, value)
        return value


def operate(metadata, body):
    root = target_root(metadata)
    directory(root / "multipart-uploads")
    directory(root / "storage-reservations")
    config = {key: metadata[key] for key in ("upload_part_bytes", "upload_session_seconds", "upload_max_parts")}
    config["local_storage_headroom_bytes"] = metadata.get("headroom_bytes", 64 * 1024 ** 2)
    store = UploadStore(root, config)
    binding, operation = metadata["binding"], metadata["operation"]
    if operation == "create":
        return store.create(binding, metadata["sha256"], metadata["bytes"], metadata.get("archive_limit"))
    upload_id = metadata["upload_id"]
    if operation == "read":
        return store.read(upload_id, binding)
    if operation == "part":
        value = store.read(upload_id, binding)
        directory(store.root / upload_id)
        if value["status"] != "COMPLETED":
            directory(store.root / upload_id / "parts")
        size = store.part_size(value, metadata["number"])
        if metadata["bytes"] != size:
            raise ValueError("remote upload part size differs")
        descriptor, temporary = tempfile.mkstemp(prefix=".part-", dir=store.root)
        try:
            with os.fdopen(descriptor, "wb") as output:
                remaining = size
                while remaining:
                    chunk = body.read(min(remaining, 1024 ** 2))
                    if not chunk:
                        raise ValueError("remote upload part is truncated")
                    output.write(chunk)
                    remaining -= len(chunk)
                if body.read(1):
                    raise ValueError("remote upload part exceeds its declared size")
                output.flush()
                os.fsync(output.fileno())
            if value["status"] == "COMPLETED":
                item = {"sha256": metadata["sha256"], "bytes": size}
                if value["parts"].get(str(metadata["number"])) != item:
                    raise ValueError("completed upload part identity differs")
                UploadStore.verify_part(Path(temporary), item)
                archive = directory(root / "data-archives") / ("asset." + value["sha256"] + ".tar")
                if archive.is_symlink():
                    raise ValueError("completed upload archive is a link")
                digest = hashlib.sha256()
                with archive.open("rb") as stream:
                    stream.seek(metadata["number"] * value["part_bytes"])
                    remaining = size
                    while remaining:
                        chunk = stream.read(min(remaining, 1024 ** 2))
                        if not chunk:
                            raise ValueError("completed upload archive is truncated")
                        digest.update(chunk)
                        remaining -= len(chunk)
                    os.fsync(stream.fileno())
                if digest.hexdigest() != item["sha256"]:
                    raise ValueError("completed upload part checksum differs")
            else:
                item = store.part(upload_id, binding, metadata["number"], Path(temporary), metadata["sha256"], size)
            return {"contract": "upload-part.v1", "upload_id": upload_id, "number": metadata["number"],
                    **item, "durable": True}
        finally:
            Path(temporary).unlink(missing_ok=True)
    if operation == "complete":
        value = store.read(upload_id, binding)
        return store.complete(upload_id, binding, lambda stream, size, digest: publish(store, metadata, stream, size, digest), value["bytes"])
    if operation == "abort":
        return store.abort(upload_id, binding)
    raise ValueError("unknown remote upload operation")


def main():
    try:
        header = sys.stdin.buffer.read(4)
        size = int.from_bytes(header, "big")
        if len(header) != 4 or not 0 < size <= 8 * 1024 ** 2:
            raise ValueError("invalid remote upload header")
        raw = sys.stdin.buffer.read(size)
        if len(raw) != size:
            raise ValueError("remote upload metadata is truncated")
        result = operate(json.loads(raw), sys.stdin.buffer)
        print(json.dumps({"ok": True, "result": result}, separators=(",", ":")), flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"ok": False, "code": error.code if isinstance(error, ApplicationError) else "REMOTE_UPLOAD_INVALID",
                          "status_code": error.status_code if isinstance(error, ApplicationError) else 409,
                          "error_class": type(error).__name__}), flush=True)
        return 1
