"""Stdlib-only data staging on the private desktop builder's owned volume.

This module and its small store dependencies are installed in an administrator
owned container. It never executes uploaded code. No public ports are exposed.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import gzip
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile

from .application_errors import ApplicationError
from .multipart_upload import UploadStore, PartStream
from .storage import atomic_json, utc_now

IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
ASSET = re.compile(r"^asset\.[0-9a-f]{64}$")
SHA = re.compile(r"^[0-9a-f]{64}$")
RESERVE = 64 * 1024 ** 2


def unpack(stream, destination, ceiling, file_limit):
    files, seen, size = [], set(), 0
    magic = stream.read(2)
    stream.seek(0)
    expanded = gzip.GzipFile(fileobj=stream) if magic == b"\x1f\x8b" else stream
    maximum = ceiling if ceiling is not None else max(0, shutil.disk_usage(destination).free - RESERVE)
    class Reader:
        remaining = maximum + file_limit * 2048 + 1024 ** 2
        def read(self, count):
            if not 0 <= count <= min(8 * 1024 ** 2, self.remaining):
                raise ValueError("dataset archive metadata or expanded stream exceeds storage bounds")
            value = expanded.read(count)
            self.remaining -= len(value)
            return value
    with tarfile.open(fileobj=Reader(), mode="r|") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if member.isdir() and member.name in {".", "./"}:
                continue
            if (not path.parts or path.is_absolute() or ".." in path.parts or "\\" in member.name
                    or any(p.startswith(".") or p.lower() in {"credentials", "keys", "secrets"}
                           or p.lower().startswith(("credentials.", "secrets."))
                           or Path(p.lower()).suffix in {".key", ".pem", ".p12", ".pfx"} for p in path.parts)):
                raise ValueError("invalid dataset path")
            name = path.as_posix()
            if name in seen or len(seen) >= file_limit:
                raise ValueError("duplicate dataset path or file count exceeded")
            seen.add(name)
            if member.isdir():
                continue
            if not member.isfile() or member.size < 0:
                raise ValueError("dataset must contain regular files")
            size += member.size
            if (ceiling is not None and size > ceiling) or member.size + RESERVE > shutil.disk_usage(destination).free:
                raise ApplicationError("insufficient dataset storage or expanded quota", status_code=507, code="DATA_STORAGE")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            sha = hashlib.sha256()
            with archive.extractfile(member) as source, target.open("xb") as output:
                remaining = member.size
                while remaining:
                    chunk = source.read(min(1024 ** 2, remaining))
                    if not chunk:
                        raise ValueError("truncated dataset")
                    sha.update(chunk)
                    output.write(chunk)
                    remaining -= len(chunk)
                output.flush()
                os.fsync(output.fileno())
            target.chmod(0o444)
            files.append({"path": name, "bytes": member.size, "sha256": sha.hexdigest()})
    if not files:
        raise ValueError("dataset is empty")
    return sorted(files, key=lambda item: item["path"])


class DesktopUploads:
    def __init__(self, root, config):
        self.root, self.config = Path(root), config
        self.uploads = UploadStore(root, config)

    def asset_path(self, project, asset_id):
        if not IDENTITY.fullmatch(project) or not ASSET.fullmatch(asset_id):
            raise ValueError("invalid remote asset identity")
        return self.root / "assets" / project / asset_id

    def asset(self, project, asset_id):
        return json.loads((self.asset_path(project, asset_id) / "asset.json").read_text())

    def publish(self, binding, stream, size, digest, upload):
        destination = self.asset_path(binding["project"], "asset." + digest)
        if destination.exists():
            return self.asset(binding["project"], "asset." + digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".validate-", dir=destination.parent))
        try:
            tree = temporary / "tree"
            tree.mkdir()
            files = unpack(stream, tree, self.config.get("max_asset_bytes"), int(self.config.get("max_asset_files", 20000)))
            # The verified original archive is the data image's COPY payload;
            # keep its manifest, not a second expanded dataset on the desktop.
            shutil.rmtree(tree)
            result = {"project": binding["project"], "asset_id": "asset." + digest, "status": "READY",
                      "sha256": digest, "archive_bytes": size, "bytes": sum(f["bytes"] for f in files),
                      "files": files, "created_at": utc_now(), "provenance": None,
                      "remote_storage": "desktop-builder", "part_bytes": upload["part_bytes"],
                      "part_count": upload["part_count"]}
            # Hardlinks retain the already verified archive with no second byte
            # copy, and survive interruption before the upload journal commits.
            parts = temporary / "parts"
            parts.mkdir()
            source = self.uploads.root / upload["upload_id"] / "parts"
            for number in range(upload["part_count"]):
                os.link(source / str(number), parts / str(number))
            atomic_json(temporary / "asset.json", result)
            temporary.rename(destination)
            return result
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def complete(self, data):
        binding, upload_id = data["binding"], data["upload_id"]
        value = self.uploads.read(upload_id, binding)
        if value["status"] == "COMPLETED":
            return value["result"]
        # Retain the exact validated archive on the desktop; these parts also
        # supply a streaming Docker context, without assembling another archive.
        def publish(stream, size, digest):
            return self.publish(binding, stream, size, digest, value)
        # The expanded tree is validated only on the desktop. Reserve is real
        # headroom, not the configured maximum possible dataset size.
        return self.uploads.complete(upload_id, binding, publish, RESERVE)

    def stream(self, project, asset_id):
        value = self.asset(project, asset_id)
        root = self.asset_path(project, asset_id)
        paths = [root / "parts" / str(n) for n in range(value["part_count"])]
        return io.BufferedReader(PartStream(paths, value["part_bytes"], value["archive_bytes"]), buffer_size=1024 ** 2)

    def call(self, operation, data, body=None):
        if operation == "info":
            free = shutil.disk_usage(self.root)
            workers = Path(self.config.get("worker_directory", "/app/workers"))
            digest = hashlib.sha256(b"".join((workers / name).read_bytes() for name in (
                "worker.py", "legacy_worker.py", "data_preparation.py", "persistent_state.py", "data_copy_worker.py"))).hexdigest()
            return {"storage": "desktop-builder", "free_bytes": free.free, "total_bytes": free.total,
                    "data_worker_sha256": digest}
        if data.get("binding", {}).get("kind") != "asset":
            raise ValueError("desktop upload is bound to a data asset")
        if operation == "create":
            return self.uploads.create(data["binding"], data["sha256"], data["bytes"], self.config.get("max_asset_archive_bytes"))
        if operation == "read":
            return self.uploads.read(data["upload_id"], data["binding"])
        if operation == "abort":
            return self.uploads.abort(data["upload_id"], data["binding"])
        if operation == "asset":
            return self.asset(data["binding"]["project"], data["asset_id"])
        if operation == "complete":
            return self.complete(data)
        if operation != "part" or body is None:
            raise ValueError("unknown desktop upload operation")
        value = self.uploads.read(data["upload_id"], data["binding"])
        expected = self.uploads.part_size(value, data["number"])
        if expected != data["bytes"] or len(body) != expected or hashlib.sha256(body).hexdigest() != data["sha256"]:
            raise ValueError("desktop upload part checksum or length differs")
        if expected + RESERVE > shutil.disk_usage(self.root).free:
            raise ApplicationError("insufficient desktop staging space", status_code=507, code="UPLOAD_STORAGE")
        with tempfile.NamedTemporaryFile(dir=self.uploads.root, prefix=".part-", delete=False) as part:
            path = Path(part.name)
            try:
                part.write(body)
                part.flush()
                return self.uploads.part(data["upload_id"], data["binding"], data["number"], path, data["sha256"], expected)
            finally:
                path.unlink(missing_ok=True)


def store():
    return DesktopUploads("/stage", json.loads(Path("/app/upload-config.json").read_text()))


def context(service, project, asset_id, output):
    value = service.asset(project, asset_id)
    base = service.config["data_base_image"]
    recipe = (f"FROM {base}\nCOPY dataset.tar /payload/dataset.tar\n"
              "COPY asset.json /payload/asset.json\nCOPY workers/ /usr/local/lib/ml-expd/\n"
              "ENTRYPOINT []\nCMD [\"/bin/true\"]\n").encode()
    def add(archive, name, body):
        info = tarfile.TarInfo(name)
        info.size, info.mode = len(body), 0o444
        archive.addfile(info, io.BytesIO(body))
    with tarfile.open(fileobj=output, mode="w|") as archive:
        add(archive, "Dockerfile", recipe)
        add(archive, "asset.json", json.dumps(value, sort_keys=True).encode())
        for path in sorted(Path(service.config.get("worker_directory", "/app/workers")).glob("*.py")):
            add(archive, "workers/" + path.name, path.read_bytes())
        member = tarfile.TarInfo("dataset.tar")
        member.size, member.mode = value["archive_bytes"], 0o444
        with service.stream(project, asset_id) as stream:
            archive.addfile(member, stream)


class ContextHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        match = re.fullmatch(r"/contexts/([A-Za-z0-9][A-Za-z0-9_.-]{0,127})/(asset\.[0-9a-f]{64})\.tar", self.path)
        if not match:
            self.send_error(404)
            return
        service = store()
        try:
            service.asset(*match.groups())
        except (OSError, ValueError):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-tar")
        self.end_headers()
        context(service, *match.groups(), self.wfile)


def main():
    operation = sys.argv[1]
    if operation == "serve":
        with ThreadingHTTPServer(("0.0.0.0", 8080), ContextHandler) as server:
            server.serve_forever()
        return
    data = json.loads(sys.argv[2])
    service = store()
    if operation == "archive":
        with service.stream(data["project"], data["asset_id"]) as stream:
            shutil.copyfileobj(stream, sys.stdout.buffer, 1024 ** 2)
        return
    try:
        # Docker exec's attached stdin may remain open after the exact body has
        # arrived over a reverse Unix socket. The authenticated RPC already
        # enforces Content-Length; never wait for an extra byte/EOF here.
        body = sys.stdin.buffer.read(data["bytes"]) if operation == "part" else None
        result = {"ok": True, "result": service.call(operation, data, body)}
    except ApplicationError as exc:
        result = {"ok": False, "error": exc.code, "status_code": exc.status_code}
    except (OSError, ValueError, KeyError):
        result = {"ok": False, "error": "DESKTOP_UPLOAD_INVALID", "status_code": 409}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
