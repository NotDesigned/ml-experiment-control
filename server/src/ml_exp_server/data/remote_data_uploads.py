"""Private builder RPC for desktop-backed client data uploads."""
from __future__ import annotations

import hashlib
import json
import base64
from copy import deepcopy
from functools import lru_cache
import gzip
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import threading
from urllib.parse import urlencode

from ..application_errors import ApplicationError
from ..archive_limits import exceeds
from ..builds.builder_client import UnixConnection
from ..projects.source_imports import IDENTITY, source_lock
from ..results.checkpoint_registry import storage_scope
from ..storage import atomic_json
from .multipart_upload import UPLOAD_ID, UploadStore


def stage_request(socket, operation, metadata, body=b""):
    connection = UnixConnection(socket, timeout=1200)
    try:
        connection.request("POST", "/data-stage/" + operation, body=body, headers={
            "Content-Type": "application/octet-stream", "X-ML-Expd-Data": json.dumps(metadata, ensure_ascii=True)})
        response = connection.getresponse()
        raw = response.read(8 * 1024 ** 2 + 1)
        if len(raw) > 8 * 1024 ** 2:
            raise ValueError("desktop upload metadata is too large")
        value = json.loads(raw)
        if response.status != 200:
            raise ApplicationError("desktop upload is unavailable", status_code=value.get("status_code", 409),
                                   code=value.get("error", "DESKTOP_UPLOAD_UNAVAILABLE"))
        return value
    finally:
        connection.close()


class RemoteDataUploads:
    remote = True

    def __init__(self, assets, socket):
        self.assets, self.socket, self.root = assets, socket, assets.root

    def call(self, operation, binding, upload_id=None, **fields):
        value = {"binding": binding, **fields}
        if upload_id is not None:
            value["upload_id"] = upload_id
        return stage_request(self.socket, operation, value)

    def create(self, binding, digest, size, limit):
        value = self.call("create", binding, sha256=digest, bytes=size)
        if value["status"] == "COMPLETED":
            # Desktop completion may have committed before the API host lost
            # its response. Repair only the small locator, never reupload data.
            value["result"] = self.assets.publish_remote(binding["project"], value["result"])
        return value

    def read(self, upload_id, binding):
        return self.call("read", binding, upload_id)

    @staticmethod
    def part_size(value, number):
        from .multipart_upload import UploadStore
        return UploadStore.part_size(value, number)

    def part(self, upload_id, binding, number, body, digest, size):
        if len(body) != size or hashlib.sha256(body).hexdigest() != digest:
            raise ValueError("invalid data upload part")
        return stage_request(self.socket, "part", {"upload_id": upload_id, "binding": binding,
                             "number": number, "sha256": digest, "bytes": size}, body)

    def complete(self, upload_id, binding):
        value = self.call("complete", binding, upload_id)
        return self.assets.publish_remote(binding["project"], value)

    def abort(self, upload_id, binding):
        return self.call("abort", binding, upload_id)


def legacy_remote_uploads(assets):
    if assets.objects.config.get("data_upload_storage") == "desktop-builder":
        return RemoteDataUploads(assets, assets.objects.config["data_upload_socket"])
    return None


def freeze_target(runtime, project, requested):
    from ..container_execution import ContainerExecutionService
    if not IDENTITY.fullmatch(project) or not isinstance(requested, dict):
        raise ValueError("invalid data upload target")
    executor = requested.get("executor")
    if not isinstance(executor, str) or not IDENTITY.fullmatch(executor):
        raise ValueError("invalid upload executor")
    profile = ContainerExecutionService(runtime).profiles().get(executor)
    if not profile or profile.get("backend", {}).get("kind") not in {"slurm", "sensecore", "pai-dlc"}:
        raise ApplicationError("unknown data upload executor", status_code=404, code="UNKNOWN_EXECUTOR")
    profile = deepcopy(profile)
    root = profile.get("storage_root", "").rstrip("/")
    path = PurePosixPath(root)
    if not path.is_absolute() or str(path) != root or ".." in path.parts or "\\" in root or "\x00" in root:
        raise ValueError("executor storage root is not canonical")
    project_root = root + "/" + project
    scope = storage_scope(profile["backend"], project_root)
    if requested.get("storage_scope", scope) != scope:
        raise ApplicationError("executor storage scope differs", code="UPLOAD_TARGET_CHANGED")
    extra = {}
    if profile["backend"]["kind"] == "pai-dlc":
        from .oss_datasets import settings
        from .data_assets import AssetStore
        extra["oss_datasets"] = settings(AssetStore(Path(runtime.config.container_execution.artifact_store_file),runtime.config.project_registry_root_path()).objects.config)
        if (extra["oss_datasets"]["user"] != profile["backend"].get("workspace_user")
                or extra["oss_datasets"]["region"] != profile["backend"]["region"]
                or not ContainerExecutionService(runtime).profile_capabilities({executor:profile})[executor]["data"]["assets"]):
            raise ApplicationError("DLC dataset storage does not match the approved personal workspace",code="UPLOAD_TARGET_CHANGED")
    return {**extra,"executor": executor, "backend": profile["backend"]["kind"], "storage_scope": scope,
            "project_root": project_root, "profile": profile,
            "profile_sha256": hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def public_target(target):
    return {key: target[key] for key in ("executor", "backend", "storage_scope", "profile_sha256")}


@lru_cache(maxsize=1)
def receiver_source():
    root = Path(__file__).parent.parent
    names = ("application_errors", "archive_limits", "storage", "storage_capacity", "data.multipart_upload",
             "workers.worker_http", "workers.data_input", "workers.data_receiver")
    sources = {"ml_exp_server." + name: (root / (name.replace(".", "/") + ".py")).read_text() for name in names}
    payload = base64.b64encode(gzip.compress(json.dumps(sources).encode(), mtime=0)).decode()
    return ("import base64,gzip,json,sys,types\n"
            "for name in ('ml_exp_server','ml_exp_server.data','ml_exp_server.workers'):\n"
            " module=types.ModuleType(name);module.__path__=[];sys.modules[name]=module\n"
            "sources=json.loads(gzip.decompress(base64.b64decode(" + repr(payload) + ")))\n"
            "for name,source in sources.items():\n"
            " module=types.ModuleType(name);module.__package__=name.rsplit('.',1)[0];sys.modules[name]=module;exec(compile(source,name,'exec'),module.__dict__)\n"
            "raise SystemExit(sys.modules['ml_exp_server.workers.data_receiver'].main())\n")


class WydDataUploads:
    remote = True
    part_size = staticmethod(UploadStore.part_size)

    def __init__(self, assets, target, files=None, transport_sha256=None):
        self.assets, self.target, self.files = assets, deepcopy(target), files
        self.transport_sha256 = transport_sha256
        self.root = assets.root
        alias = target["profile"]["backend"].get("ssh_alias", "")
        if not IDENTITY.fullmatch(alias):
            raise ValueError("invalid upload SSH origin")
        self.alias = alias

    def call(self, operation, binding, upload_id=None, *, body=None, **fields):
        if binding.get("target") != public_target(self.target):
            raise ValueError("upload target binding differs")
        config = self.assets.objects.config
        metadata = {"operation": operation, "binding": binding, "target": {
            key: self.target[key] for key in ("executor", "backend", "storage_scope", "project_root", "profile_sha256")},
            "files": self.files, "expanded_limit": self.assets.expanded_limit, "file_limit": self.assets.file_limit,
            "archive_limit": self.assets.limit, "upload_part_bytes": config.get("upload_part_bytes", 16 * 1024 ** 2),
            "upload_session_seconds": config.get("upload_session_seconds", 86400),
            "upload_max_parts": config.get("upload_max_parts", 65536), **fields}
        if upload_id is not None:
            metadata["upload_id"] = upload_id
        header = json.dumps(metadata, separators=(",", ":")).encode()
        if len(header) > 8 * 1024 ** 2:
            raise ValueError("remote upload metadata exceeds its bound")
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", self.alias,
                   shlex.join(["python3", "-c", receiver_source()])]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        expired = threading.Event()
        def timeout():
            expired.set()
            process.kill()
        timer = threading.Timer(1800, timeout)
        timer.daemon = True
        timer.start()
        try:
            process.stdin.write(len(header).to_bytes(4, "big") + header)
            if body is not None:
                with Path(body).open("rb") as stream:
                    while chunk := stream.read(1024 ** 2):
                        process.stdin.write(chunk)
            process.stdin.close()
            output = process.stdout.read(8 * 1024 ** 2 + 1)
            code = process.wait(timeout=15)
            if expired.is_set() or len(output) > 8 * 1024 ** 2:
                raise ValueError("remote upload response exceeds its bound")
            result = json.loads(output)
            if not result.get("ok"):
                raise ApplicationError("remote data upload was not confirmed", status_code=result.get("status_code", 409),
                                       code=result.get("code", "REMOTE_UPLOAD_UNCONFIRMED"))
            if code:
                raise ValueError("remote upload process failed")
            return result["result"]
        except (OSError, ValueError, subprocess.SubprocessError):
            raise ApplicationError("remote data upload is unconfirmed; resume its saved session",
                                   status_code=503, code="REMOTE_UPLOAD_UNCONFIRMED") from None
        finally:
            timer.cancel()
            if process.poll() is None:
                process.kill()
            process.wait()
            if not process.stdin.closed:
                process.stdin.close()
            process.stdout.close()

    def create(self, binding, digest, size, limit):
        return self.call("create", binding, sha256=digest, bytes=size, archive_limit=limit)

    def read(self, upload_id, binding):
        return self.call("read", binding, upload_id)

    def part(self, upload_id, binding, number, body, digest, size):
        UploadStore.verify_part(Path(body), {"sha256": digest, "bytes": size})
        return self.call("part", binding, upload_id, body=body, number=number, sha256=digest, bytes=size)

    def complete(self, upload_id, binding):
        return self.assets.publish_remote(binding["project"], self.call("complete", binding, upload_id))

    def abort(self, upload_id, binding):
        return self.call("abort", binding, upload_id)


class ManagedDataUploads:
    """Small frozen locators around existing destination UploadStore sessions."""
    remote = True
    part_size = staticmethod(UploadStore.part_size)

    def __init__(self, assets, target, files=None, transport_sha256=None, *, saved=None):
        if files is not None:
            assets.validate_inventory(files)
            files = sorted((dict(item) for item in files), key=lambda item: item["path"])
        self.assets, self.target, self.files = assets, deepcopy(target), files
        self.transport_sha256, self.saved = transport_sha256, saved
        self.root = assets.root.parent / "remote-data-uploads"
        if target["backend"] == "slurm":
            self.driver = WydDataUploads(assets, self.target, files, transport_sha256)
        elif target["backend"] == "sensecore":
            from .ccr_uploads import CCRDataUploads
            self.driver = CCRDataUploads(assets, self.target, files, transport_sha256)
        elif target["backend"] == "pai-dlc":
            from .oss_uploads import OSSDataUploads
            self.driver = OSSDataUploads(assets,self.target,files,transport_sha256)
        else:
            raise ValueError("unsupported upload target backend")
        self.binding = saved["binding"] if saved else None

    def create(self, binding, digest, size, limit):
        config = self.assets.objects.config
        part_bytes = getattr(self.driver,"part_bytes",config.get("upload_part_bytes",16*1024**2))
        max_parts = min(config.get("upload_max_parts",65536),getattr(self.driver,"max_parts",65536))
        if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or type(size) is not int or size <= 0 or exceeds(size, limit)
                or type(part_bytes) is not int or not 1024 <= part_bytes <= 64 * 1024 ** 2
                or type(max_parts) is not int or max_parts <= 0
                or (size + part_bytes - 1) // part_bytes > max_parts):
            raise ApplicationError("archive exceeds upload limit or has an invalid identity", status_code=413, code="UPLOAD_LIMIT")
        binding = {**binding, "target": public_target(self.target)}
        if self.files is not None:
            binding["files_sha256"] = hashlib.sha256(json.dumps(self.files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        upload_id = UploadStore.identity(binding, digest, size)
        value = {"upload_id": upload_id, "binding": binding, "sha256": digest, "bytes": size,
                 "target": self.target, "files": self.files, "transport_sha256": self.transport_sha256}
        with source_lock(self.root, upload_id):
            path = self.root / (upload_id + ".json")
            if path.exists():
                if json.loads(path.read_text()) != value:
                    raise ValueError("upload is already bound to another frozen target or manifest")
            else:
                atomic_json(path, value)
        self.binding = binding
        result = self.driver.create(binding, digest, size, limit)
        if result.get("upload_id") != upload_id or result.get("binding") != binding:
            raise ValueError("destination upload identity differs")
        if result.get("status") == "COMPLETED":
            result["result"] = self.assets.publish_remote(binding["project"], result["result"])
        return result

    def checked(self, binding):
        if binding != self.binding:
            raise ApplicationError("unknown upload for this project or target", status_code=404, code="UNKNOWN_UPLOAD")

    def read(self, upload_id, binding):
        self.checked(binding)
        return self.driver.read(upload_id, binding)

    def part(self, upload_id, binding, number, body, digest, size):
        self.checked(binding)
        receipt = self.driver.part(upload_id, binding, number, body, digest, size)
        expected = {"contract": "upload-part.v1", "upload_id": upload_id, "number": number,
                    "sha256": digest, "bytes": size, "durable": True}
        if not isinstance(receipt, dict) or any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError("remote durable upload receipt differs")
        return receipt

    def complete(self, upload_id, binding):
        self.checked(binding)
        return self.driver.complete(upload_id, binding)

    def abort(self, upload_id, binding):
        self.checked(binding)
        return self.driver.abort(upload_id, binding)


def remote_uploads(assets, target=None, *, runtime=None, upload_id=None, files=None, transport_sha256=None):
    if upload_id is not None:
        if not UPLOAD_ID.fullmatch(upload_id):
            raise ValueError("invalid upload ID")
        path = assets.root.parent / "remote-data-uploads" / (upload_id + ".json")
        if path.is_file():
            saved = json.loads(path.read_text())
            if saved.get("upload_id") != upload_id:
                raise ValueError("saved remote upload identity differs")
            return ManagedDataUploads(assets, saved["target"], saved.get("files"), saved.get("transport_sha256"), saved=saved)
        local = assets.root.parent / "multipart-uploads" / upload_id / "upload.json"
        if local.is_file():
            return None
        raise ApplicationError("unknown upload; its original frozen target is unavailable", status_code=404, code="UNKNOWN_UPLOAD")
    if target is None:
        raise ApplicationError("new data uploads require an explicit executor target", status_code=422, code="UPLOAD_TARGET_REQUIRED")
    return ManagedDataUploads(assets, target, files, transport_sha256)


class RemoteArchive:
    def __init__(self, connection, response, size):
        self.connection, self.response, self.size = connection, response, size

    def iter_chunks(self, chunk_size):
        remaining = self.size
        try:
            while remaining:
                chunk = self.response.read(min(remaining, chunk_size))
                if not chunk:
                    raise ValueError("desktop archive stream is truncated")
                remaining -= len(chunk)
                yield chunk
        finally:
            self.close()

    def close(self):
        self.connection.close()


def remote_archive(socket, project, asset_id, size):
    connection = UnixConnection(socket, timeout=1200)
    try:
        connection.request("GET", "/data-stage/archive?" + urlencode({"project": project, "asset_id": asset_id}))
        response = connection.getresponse()
        if response.status != 200 or response.getheader("Content-Length") != str(size):
            raise ValueError("desktop archive stream is unavailable")
        return RemoteArchive(connection, response, size)
    except Exception:
        connection.close()
        raise
