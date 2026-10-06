"""Private OCI publisher with fixed recipes and an opt-in client Dockerfile recipe."""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import re
import shutil
import socket
import socketserver
import struct
import subprocess
import tempfile
import time
import uuid
import tarfile
import io
from contextlib import contextmanager
from urllib.parse import urlsplit, parse_qs

from dockerfile_parse import DockerfileParser

from .source_revisions import _tree_digest
from .storage import atomic_json
from .environment_build import DEPENDENCY_RECIPE, dockerfile, inspect_requirements, installer_digest, requirements_path
from .dockerfile_build import DOCKERFILE_RECIPE, INTERNAL, inspect_dockerfile, managed_dockerfile, worker_digest
from .worker_contract import CAPABILITIES, WORKER_CONTRACT, install_workers, recipe_digest
from .execution_progress import record_progress, progress_view
from .application_errors import ApplicationError
from .build_contexts import CONTEXT_ID, MAX_BYTES
from .image_build_context import BUILD_LOG, BUILD_PROGRESS, BUILD_CACHE, BUILD_REMOTE_CONTEXT, BUILD_CONTEXT_BYTES


IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}$")
ID = re.compile(r"^[0-9a-f]{64}$")
PROJECT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
RECIPE = "source-copy-docker-v2-v2"
BUILD_REVISION = "sealed-context-transfer.v1"
MANIFEST_TYPE = "application/vnd.docker.distribution.manifest.v2+json"


class BuildStorageError(ValueError):
    """Only reviewed storage diagnostics may cross the private builder socket."""

    def __init__(self, code: str, details: dict):
        super().__init__(code)
        self.code, self.details = code, details


class BuildTransportError(ValueError):
    """Bounded transport evidence; uncertain publications are never replayed."""

    def __init__(self, code, details):
        super().__init__(code)
        self.code, self.details = code, details


def bundle_id(project: str, source_id: str, base_image: str, requirements: str | None = None, *, dockerfile_path: str | None = None, legacy: bool = False) -> str:
    fields = [project, source_id, base_image, RECIPE, WORKER_CONTRACT, worker_digest(), recipe_digest()]
    if not legacy:
        fields.append(BUILD_REVISION)
    if requirements is not None:
        fields.extend([DEPENDENCY_RECIPE, requirements_path(requirements), installer_digest()])
    if dockerfile_path is not None:
        fields.extend([DOCKERFILE_RECIPE, requirements_path(dockerfile_path), worker_digest()])
    return hashlib.sha256(json.dumps(fields, separators=(",", ":")).encode()).hexdigest()


class UnixConnection(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.host)


def builder_request(path: str, payload: dict, *, timeout: int = 1200) -> dict:
    connection = UnixConnection(path, timeout=timeout)
    try:
        connection.request("POST", "/", body=json.dumps(payload), headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = json.loads(response.read(65536))
        if response.status != 200:
            if data.get("code") in {"BUILD_STORAGE_INSUFFICIENT", "BUILD_STORAGE_UNCHECKED", "BUILD_DISK_EXHAUSTED"}:
                raise BuildStorageError(data["code"], data["details"])
            if data.get("code") in {"BUILD_SESSION_TIMEOUT", "BUILD_CONTEXT_TRANSPORT", "DESKTOP_RPC_TIMEOUT", "DESKTOP_RPC_UNAVAILABLE", "DESKTOP_RPC_FAILED", "DESKTOP_RPC_RESPONSE"}:
                raise BuildTransportError(data["code"], data["details"])
            raise ValueError(data.get("error", "image packaging failed"))
        return data
    finally:
        connection.close()


class ImageBuilder:
    def __init__(self, config: dict):
        self.config = config
        proxy = config.get("buildkit_http_proxy")
        if "buildkit_http_proxy" in config and (not isinstance(proxy, str) or not re.fullmatch(r"https?://[A-Za-z0-9._-]+:[0-9]{1,5}/?", proxy)):
            raise ValueError("BuildKit HTTP proxy must be a credential-free HTTP endpoint")
        network = config.get("buildkit_network")
        if "buildkit_network" in config and (not isinstance(network, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", network)):
            raise ValueError("invalid private BuildKit network")
        host = config.get("docker_host")
        if "docker_host" in config:
            if (not isinstance(host, str) or not re.fullmatch(r"unix:///[-A-Za-z0-9_./]+", host)
                    or ".." in host.split("/") or config.get("publisher") != "buildkit"
                    or not config.get("ephemeral_buildkit") or config.get("build_storage_path") is not None
                    or not IMAGE.fullmatch(str(config.get("buildkit_image", "")))):
                raise ValueError("remote Docker requires a Unix socket, ephemeral pinned BuildKit and remote storage checks")
        self.root = Path(config["state_root"])
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.config.get("ephemeral_buildkit"):
            with (self.root / "ephemeral-buildkit.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                self._recover_probe()
                self._recover_builder()
                self._recover_source_context()

    def request(self, request: dict) -> dict:
        if request.get("operation") == "data-image":
            from .data_image_build import build
            return build(self, request)
        project, source_id, image = (request.get(k, "") for k in ("project", "source_id", "base_image"))
        if (not PROJECT.fullmatch(project) or not re.fullmatch(r"source\.[0-9a-f]{64}", source_id)
                or not IMAGE.fullmatch(image)):
            raise ValueError("invalid image packaging identity")
        requirements = request.get("requirements")
        custom_path = request.get("dockerfile")
        if custom_path is not None and requirements is not None:
            raise ValueError("Dockerfile and generated dependency recipes are mutually exclusive")
        recipe_version = DOCKERFILE_RECIPE if custom_path is not None else DEPENDENCY_RECIPE if requirements is not None else RECIPE
        if request.get("packaging_revision", recipe_version) != recipe_version:
            raise ValueError("image packaging revision mismatch")
        identity = bundle_id(project, source_id, image, requirements, dockerfile_path=custom_path)
        metadata_path = self.root / f"{identity}.json"
        if request.get("operation") in {"logs", "progress"}:
            pinned = request.get("bundle_id", identity)
            if not isinstance(pinned, str) or not ID.fullmatch(pinned):
                raise ValueError("invalid frozen build identity")
            if pinned != identity:
                receipt = self.root / f"{pinned}.json"
                definition = self.root / f"{pinned}.definition.json"
                historical = bundle_id(project, source_id, image, requirements, dockerfile_path=custom_path, legacy=True)
                evidence = receipt if receipt.is_file() else definition if definition.is_file() else None
                if evidence is not None:
                    value = json.loads(evidence.read_text())
                    if any(value.get(key) != expected for key, expected in (
                            ("bundle_id", pinned), ("project", project), ("source_id", source_id), ("base_image", image))):
                        raise ValueError("historical packaging receipt identity mismatch")
                elif pinned != historical:
                    raise ValueError("historical packaging receipt is unavailable")
            identity = pinned
            path = self.root / f"{identity}.log"
            content = ""
            if path.exists():
                with path.open("rb") as log:
                    log.seek(max(0, path.stat().st_size - 8192))
                    content = log.read().decode("utf-8", errors="replace")
            result = {"bundle_id": identity, "lines": content.splitlines(), "truncated": path.exists() and path.stat().st_size > 8192}
            if request["operation"] == "progress":
                status = "READY" if (self.root / f"{identity}.json").exists() else "EXECUTING"
                result["progress"] = progress_view(self.root / f"{identity}.progress.json", status,
                                                   active=status == "EXECUTING",
                                                   last_activity=path.stat().st_mtime if path.exists() else None)
            return result

        with (self.root / f"{identity}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                if metadata.get("bundle_id") != identity:
                    raise ValueError("image packaging metadata mismatch")
                return metadata
            if request.get("operation") == "get":
                raise ValueError("image bundle has not been published")
            if request.get("operation") != "build":
                raise ValueError("unsupported image packaging operation")
            if requirements is not None and (not self.config.get("allow_dependency_builds")
                                             or self.config.get("publisher") != "buildkit"):
                raise ValueError("dependency builds require an enabled BuildKit publisher")
            if custom_path is not None and (not self.config.get("allow_dockerfile_builds")
                                           or self.config.get("publisher") != "buildkit"):
                raise ValueError("Dockerfile builds require an enabled BuildKit publisher")
            prefixes = self.config.get("base_image_prefixes", [])
            if prefixes and not any(image.startswith(prefix) for prefix in prefixes):
                raise ValueError("base image is outside packaging policy")
            source = Path(self.config["source_root"]) / project / source_id
            metadata = json.loads((source / "source.json").read_text())
            tree = source / "tree"
            expected = "sha256:" + source_id.removeprefix("source.")
            if (source.resolve() != Path(self.config["source_root"]).resolve() / project / source_id
                    or source.is_symlink() or tree.is_symlink() or (source / "source.json").is_symlink()
                    or metadata.get("project") != project or metadata.get("tree_digest") != expected):
                raise ValueError("source packaging identity mismatch")
            if _tree_digest(tree, require_read_only=True) != expected:
                raise ValueError("source changed before packaging")
            dependencies = inspect_requirements(tree, requirements) if requirements is not None else None
            inspection = inspect_dockerfile(tree, custom_path) if custom_path is not None else None
            if inspection is not None and (image != inspection["base_images"][-1] or
                    prefixes and any(not any(base.startswith(prefix) for prefix in prefixes) for base in inspection["base_images"])):
                raise ValueError("Dockerfile base images do not match packaging policy")
            registry = self.config["repository"]
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]+", registry):
                raise ValueError("invalid image packaging repository")
            tag = registry + ":bundle-" + identity
            progress_path = self.root / f"{identity}.progress.json"
            record_progress(progress_path, "PREPARING_CONTEXT", "Validating and preparing the frozen source build context")
            with tempfile.TemporaryDirectory(prefix="build-", dir=self.root) as directory:
                context = Path(directory)
                copied = context / "source" if inspection is None else context / INTERNAL / "source"
                shutil.copytree(tree, copied, symlinks=False)
                if _tree_digest(copied) != expected:
                    raise ValueError("source changed while preparing packaging context")
                for path in copied.rglob("*"):
                    path.chmod(0o555 if path.is_dir() or path.stat().st_mode & 0o111 else 0o444)
                copied.chmod(0o555)
                if inspection is None:
                    install_workers(context)
                else:
                    shutil.copytree(tree, context, dirs_exist_ok=True, symlinks=False)
                    # copytree preserves sealed source permissions. Only these
                    # temporary generated build-control files need to be writable.
                    context.chmod(0o700)
                    for name in ("Dockerfile", ".dockerignore", "Dockerfile.dockerignore"):
                        control = context / name
                        if control.exists():
                            control.chmod(0o600)
                    # A client .dockerignore may exclude everything; the managed
                    # runtime and complete frozen source are always included.
                    ignore = context / ".dockerignore"
                    with ignore.open("a") as output:
                        output.write(f"\n!{INTERNAL}/\n!{INTERNAL}/**\n")
                    specific_ignore = context / "Dockerfile.dockerignore"
                    if specific_ignore.exists():
                        with specific_ignore.open("a") as output:
                            output.write(f"\n!{INTERNAL}/\n!{INTERNAL}/**\n")
                    install_workers(context / INTERNAL)
                if dependencies is not None:
                    shutil.copyfile(tree / requirements, context / "requirements.txt")
                    shutil.copyfile(Path(__file__).with_name("dependency_install.py"), context / "dependency_install.py")
                recipe = dockerfile(image, source_id, requirements) if inspection is None else managed_dockerfile(inspection, source_id)
                (context / "Dockerfile").write_text(recipe)
                atomic_json(self.root / f"{identity}.definition.json", {"bundle_id": identity, "project": project,
                            "source_id": source_id, "base_image": image})
                log_context = BUILD_LOG.set(self.root / f"{identity}.log")
                progress_context = BUILD_PROGRESS.set(progress_path)
                # One mutable acceleration tag per project; not part of source,
                # Docker context, execution identity or local persistent storage.
                cache_context = BUILD_CACHE.set(registry + ":buildcache-" + hashlib.sha256(project.encode()).hexdigest()[:24]
                                               if self.config.get("registry_cache", False) else None)
                try:
                    if self.config.get("publisher", "archive") == "buildkit":
                        published = self._publish_buildkit(tag, context)
                    else:
                        self._docker(["build", "--network=none", "--tag", tag, directory])
                        published = self._publish(tag, context)
                finally:
                    BUILD_LOG.reset(log_context)
                    BUILD_PROGRESS.reset(progress_context)
                    BUILD_CACHE.reset(cache_context)
            result = {"bundle_id": identity, "project": project, "source_id": source_id,
                      "base_image": image, "image": published, "recipe": recipe_version,
                      "manifest_type": MANIFEST_TYPE, "worker_contract": WORKER_CONTRACT,
                      "worker_sha256": worker_digest(), "capabilities": list(CAPABILITIES),
                      "dockerfile_sha256": hashlib.sha256(recipe.encode()).hexdigest()}
            if dependencies is not None:
                result.update(dependencies=dependencies, installer_sha256=installer_digest())
            if inspection is not None:
                result["dockerfile"] = {k: v for k, v in inspection.items() if k != "text"}
            atomic_json(metadata_path, result)
            record_progress(progress_path, "READY", "Published image manifest and worker identity verified")
            return result

    def desktop_stage(self, operation, data, body):
        container = self.config.get("data_upload_container", "")
        if (not re.fullmatch(r"ml-expd-data-stage[-A-Za-z0-9]*", container)
                or "docker_host" not in self.config or operation not in {"info", "create", "read", "part", "complete", "abort", "asset", "context-create", "context-seal", "context-remove", "part-discard"}):
            raise ValueError("desktop data staging is not configured")
        host = self.config["docker_host"]
        if operation == "part":
            return self._desktop_part(data, body)
        command = [self.config.get("docker_bin", "docker"), "--host", host, "exec", container,
                   "python", "-m", "ml_exp_server.desktop_upload", operation, json.dumps(data, ensure_ascii=True)]
        try:
            result = subprocess.run(command, input=body, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=self.config.get("data_upload_timeout_seconds", 1200), check=False)
        except subprocess.TimeoutExpired:
            raise ApplicationError("desktop RPC timed out; inspect acknowledged parts before resuming", status_code=503,
                                   code="DESKTOP_RPC_TIMEOUT") from None
        except OSError:
            raise ApplicationError("desktop RPC connection is unavailable", status_code=503, code="DESKTOP_RPC_UNAVAILABLE") from None
        return self._desktop_reply(result)

    @staticmethod
    def _desktop_reply(result):
        if result.returncode:
            raise ApplicationError("desktop RPC process failed", status_code=503, code="DESKTOP_RPC_FAILED")
        try:
            if len(result.stdout) > 8 * 1024 ** 2:
                raise ValueError("response too large")
            value = json.loads(result.stdout)
            if not isinstance(value, dict) or not isinstance(value.get("ok"), bool):
                raise ValueError("invalid response")
            return value
        except ValueError:
            raise ApplicationError("desktop RPC response was incomplete or invalid", status_code=503, code="DESKTOP_RPC_RESPONSE") from None

    def _desktop_part(self, data, body):
        if (not 0 < len(body) <= 64 * 1024 ** 2 or len(body) != data.get("bytes")
                or hashlib.sha256(body).hexdigest() != data.get("sha256")):
            raise ValueError("desktop RPC part identity differs")
        identity = "part." + uuid.uuid4().hex
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as tar:
            entry = tarfile.TarInfo(identity)
            entry.size, entry.mode = len(body), 0o600
            entry.mtime = int(time.time())
            tar.addfile(entry, io.BytesIO(body))
        container = self.config["data_upload_container"]
        prefix = [self.config.get("docker_bin", "docker"), "--host", self.config["docker_host"]]
        try:
            result = subprocess.run([*prefix, "cp", "-", container + ":/stage/rpc-parts"], input=archive.getvalue(),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=self.config.get("data_upload_timeout_seconds", 1200), check=False)
            if result.returncode:
                raise ApplicationError("desktop RPC body transport failed", status_code=503, code="DESKTOP_RPC_FAILED")
            command = [*prefix, "exec", container, "python", "-m", "ml_exp_server.desktop_upload", "part-file",
                       json.dumps({**data, "body_id": identity}, ensure_ascii=True)]
            result = subprocess.run(command, input=b"", stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=self.config.get("data_upload_timeout_seconds", 1200), check=False)
            return self._desktop_reply(result)
        except subprocess.TimeoutExpired:
            raise ApplicationError("desktop RPC part transport timed out", status_code=503, code="DESKTOP_RPC_TIMEOUT") from None
        except OSError:
            raise ApplicationError("desktop RPC part transport is unavailable", status_code=503, code="DESKTOP_RPC_UNAVAILABLE") from None
        finally:
            try:
                self.desktop_stage("part-discard", {"body_id": identity}, b"")
            except ApplicationError:
                # The helper removes only expired, owned temporary RPC bodies;
                # acknowledged session parts and published assets are separate.
                pass

    def _recover_source_context(self):
        record = self.root / "source-context.json"
        if not record.exists():
            return
        value = json.loads(record.read_text())
        if (not CONTEXT_ID.fullmatch(value.get("context_id", "")) or value.get("docker_host") != self._endpoint()
                or value.get("container") != self.config.get("data_upload_container")):
            raise ValueError("source context recovery identity changed")
        result = self.desktop_stage("context-remove", {"context_id": value["context_id"]}, b"")
        if not result["ok"]:
            raise ValueError("source context cleanup failed")
        record.unlink()

    @contextmanager
    def _source_context(self, context):
        if "docker_host" not in self.config or BUILD_REMOTE_CONTEXT.get() is not None:
            yield
            return
        container = self.config.get("data_upload_container", "")
        if not re.fullmatch(r"ml-expd-data-stage[-A-Za-z0-9]*", container):
            raise ValueError("remote source builds require the owned desktop staging helper")
        self._recover_source_context()
        identity = "context." + uuid.uuid4().hex
        record = self.root / "source-context.json"
        with tempfile.TemporaryDirectory(prefix="source-context-", dir=self.root) as temporary:
            archive = Path(temporary) / "context.tar.gz"
            self._progress("STAGING_SOURCE_CONTEXT", "Sending a sealed source archive to the desktop; no scheduler submission")
            with tarfile.open(archive, mode="w:gz", dereference=False) as tar:
                for path in sorted(context.rglob("*")):
                    if path.is_symlink() or not (path.is_file() or path.is_dir()):
                        raise ValueError("invalid source build context file")
                    tar.add(path, arcname=path.relative_to(context).as_posix(), recursive=False)
            if archive.stat().st_size > MAX_BYTES:
                raise ValueError("source context exceeds its independent limit")
            digest = hashlib.sha256()
            with archive.open("rb") as stream:
                while chunk := stream.read(1024 ** 2):
                    digest.update(chunk)
            atomic_json(record, {"context_id": identity, "docker_host": self._endpoint(), "container": container})
            token = None
            try:
                data = {"context_id": identity, "bytes": archive.stat().st_size, "sha256": digest.hexdigest()}
                result = self.desktop_stage("context-create", data, b"")
                if not result["ok"]:
                    raise ValueError("source context preparation failed")
                self._docker(["cp", str(archive), container + ":/stage/build-contexts/" + identity + "/context.tar.gz"])
                result = self.desktop_stage("context-seal", {"context_id": identity}, b"")
                if not result["ok"] or result["result"].get("sha256") != data["sha256"] or result["result"].get("bytes") != data["bytes"]:
                    raise ValueError("source context transport identity differs")
                token = BUILD_REMOTE_CONTEXT.set(f"http://{container}:8080/build-contexts/{identity}.tar.gz")
                yield
            finally:
                if token is not None:
                    BUILD_REMOTE_CONTEXT.reset(token)
                self._recover_source_context()

    def _publish_buildkit(self, tag: str, context: Path) -> str:
        if not self.config.get("ephemeral_buildkit", False):
            self._storage_preflight(context)
            return self._buildkit_image(tag, context, "default")
        image = self.config.get("buildkit_image", "")
        if not IMAGE.fullmatch(image):
            raise ValueError("ephemeral BuildKit requires a pinned builder image")
        # Serialize this application's transient CUDA layers without touching
        # the host's default builder or any other workload's cache.
        with (self.root / "ephemeral-buildkit.lock").open("a") as lock:
            self._progress("WAITING_BUILDER", "Waiting for the private image builder; no scheduler submission has occurred")
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._recover_probe()
            self._recover_builder()
            self._storage_preflight(context)
            name = "ml-expd-" + uuid.uuid4().hex
            atomic_json(self.root / "ephemeral-builder.json", {"name": name, "docker_host": self._endpoint()})
            try:
                self._docker(["buildx", "create", "--name", name, "--driver", "docker-container",
                              "--driver-opt", "image=" + image, "--driver-opt", "default-load=false",
                              *(["--driver-opt", "env.HTTP_PROXY=" + self.config["buildkit_http_proxy"]]
                                if "buildkit_http_proxy" in self.config else []),
                              *(["--driver-opt", "network=" + self.config["buildkit_network"]]
                                if "buildkit_network" in self.config else []),
                              *(["--driver-opt", "env.NO_PROXY=" + self.config["data_upload_container"]]
                                if "data_upload_container" in self.config else []),
                              *([self.config["docker_host"]] if "docker_host" in self.config else [])])
                with self._source_context(context):
                    return self._buildkit_image(tag, context, name)
            finally:
                # Removing this exact builder also removes its dedicated state
                # volume. No --keep-state, host image load or shared prune.
                self._remove_builder(name)
                (self.root / "ephemeral-builder.json").unlink()

    def _storage_preflight(self, context: Path) -> None:
        # This must be Docker's real state-volume filesystem, not /tmp or a
        # staging directory on another mount. The check runs under build lock.
        path = self.config.get("build_storage_path")
        if path is None and "docker_host" not in self.config:
            return
        self._progress("CHECKING_BUILD_STORAGE", "Checking builder bytes and inodes before downloading image layers")
        try:
            images = set()
            for instruction in DockerfileParser(path=str(context)).structure:
                if instruction["instruction"] == "FROM":
                    images.update(re.findall(r"\S+@sha256:[0-9a-f]{64}", instruction["value"]))
            compressed = 0
            for image in sorted(images):
                auth = self.config.get("registry_auth_file", "/root/.docker/config.json")
                manifest = json.loads(self._skopeo(["inspect", "--authfile", auth, "--raw", "docker://" + image], capture=True))
                if "manifests" in manifest:
                    platform = next(item for item in manifest["manifests"]
                                    if item.get("platform", {}).get("os") == "linux" and item.get("platform", {}).get("architecture") == "amd64")
                    reference = image.split("@", 1)[0] + "@" + platform["digest"]
                    manifest = json.loads(self._skopeo(["inspect", "--authfile", auth, "--raw", "docker://" + reference], capture=True))
                compressed += sum(int(layer["size"]) for layer in manifest["layers"])
            available_bytes, available_inodes = self._storage_available(path)
            context_bytes = sum(p.stat().st_size for p in context.rglob("*") if p.is_file()) + BUILD_CONTEXT_BYTES.get()
            factor = float(self.config.get("build_expansion_factor", 4))
            reserve = int(self.config.get("build_reserve_bytes", 2 * 1024 ** 3))
            required = int(compressed * factor) + 2 * context_bytes + reserve
            details = {"available_bytes": available_bytes, "required_bytes": required,
                       "available_inodes": available_inodes, "required_inodes": int(self.config.get("build_min_free_inodes", 100000)),
                       "compressed_base_bytes": compressed, "expansion_factor": factor, "reserve_bytes": reserve,
                       "estimate_kind": "conservative_compressed_layer_budget", "scheduler_submitted": False}
        except (OSError, ValueError, KeyError, TypeError, StopIteration):
            raise BuildStorageError("BUILD_STORAGE_UNCHECKED", {"scheduler_submitted": False}) from None
        # Arbitrary RUN commands may grow beyond this estimate. Runtime ENOSPC
        # still gets a distinct error and owned ephemeral-builder cleanup.
        if details["available_bytes"] < required or details["available_inodes"] < details["required_inodes"]:
            raise BuildStorageError("BUILD_STORAGE_INSUFFICIENT", details)
        self._progress("BUILD_STORAGE_READY", "Builder storage preflight passed; Dockerfile growth remains bounded by disk capacity")

    def _endpoint(self) -> str:
        return self.config.get("docker_host", "unix:///var/run/docker.sock")

    def _storage_available(self, path) -> tuple[int, int]:
        if "docker_host" not in self.config:
            stats = os.statvfs(path)
            return stats.f_bavail * stats.f_frsize, stats.f_favail
        # The temporary anonymous volume is on the remote Docker data filesystem.
        # Never use this API host's disk or pull a probe image before checking.
        name = "ml-expd-storage-" + uuid.uuid4().hex
        atomic_json(self.root / "storage-probe.json", {"name": name, "docker_host": self._endpoint()})
        try:
            raw = self._docker(["run", "--rm", "--name", name, "--pull=never", "--network=none",
                                "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                                "--mount", "type=volume,target=/probe", "--entrypoint", "/bin/sh",
                                self.config["buildkit_image"], "-c", "stat -f -c '%a %S %d' /probe"], capture=True)
            blocks, size, inodes = map(int, raw.split())
            if blocks < 0 or size <= 0 or inodes < 0:
                raise ValueError("invalid remote storage observation")
            return blocks * size, inodes
        finally:
            self._recover_probe()

    def _recover_probe(self):
        path = self.root / "storage-probe.json"
        if not path.exists():
            return
        record = json.loads(path.read_text())
        name = record["name"]
        if (not re.fullmatch(r"ml-expd-storage-[0-9a-f]{32}", name)
                or record.get("docker_host") != self._endpoint()):
            raise ValueError("remote storage probe recovery endpoint or identity changed")
        names = self._docker(["ps", "--all", "--filter", "name=^/" + name + "$",
                              "--format", "{{.Names}}"], capture=True).splitlines()
        if name in names:
            self._docker(["rm", "--force", "--volumes", name])
        path.unlink()

    def _remove_builder(self, name):
        names = self._docker(["buildx", "ls", "--format", "{{.Name}}"], capture=True).splitlines()
        if name in names:
            self._docker(["buildx", "rm", "--force", name])

    def _recover_builder(self):
        path = self.root / "ephemeral-builder.json"
        if not path.exists():
            return
        record = json.loads(path.read_text())
        name = record["name"]
        if not re.fullmatch(r"ml-expd-[0-9a-f]{32}", name):
            raise ValueError("invalid ephemeral builder recovery identity")
        if record.get("docker_host", "unix:///var/run/docker.sock") != self._endpoint():
            raise ValueError("ephemeral builder recovery endpoint changed")
        self._remove_builder(name)
        path.unlink()

    def _buildkit_image(self, tag: str, context: Path, builder: str) -> str:
        # BuildKit streams new layers to the registry and reuses existing base
        # blobs. Exporting a complete CUDA image twice exhausts small servers.
        metadata = context / "buildkit.json"
        network = "default" if (context / "requirements.txt").exists() or (context / INTERNAL).exists() else "none"
        cache = []
        if BUILD_CACHE.get() is not None:
            reference = BUILD_CACHE.get()
            cache = ["--cache-from", "type=registry,ref=" + reference,
                     "--cache-to", "type=inline", "--tag", reference]
        self._progress("BUILDING_AND_PUSHING", "Building Dockerfile and publishing image; detailed output is in build logs")
        self._docker(["buildx", "build", "--builder", builder, "--progress=plain", "--network=" + network,
                      "--provenance=false", "--sbom=false", "--tag", tag,
                      "--metadata-file", str(metadata), "--output",
                      "type=image,push=true,oci-mediatypes=false,compression=gzip",
                      *cache, BUILD_REMOTE_CONTEXT.get() or str(context)])
        value = json.loads(metadata.read_text())
        digest = value.get("containerimage.digest", "")
        config_digest = value.get("containerimage.config.digest", "")
        if not all(re.fullmatch(r"sha256:[0-9a-f]{64}", item)
                   for item in (digest, config_digest)):
            raise ValueError("BuildKit publication identity is unavailable")
        reference = self.config["repository"] + "@" + digest
        self._progress("VERIFYING_IMAGE", "Checking the remote manifest against the build's exact image and config digests")
        raw = self._skopeo(["inspect", "--authfile",
                            self.config.get("registry_auth_file", "/root/.docker/config.json"),
                            "--raw", "docker://" + reference], capture=True)
        manifest = json.loads(raw)
        if ("sha256:" + hashlib.sha256(raw.encode()).hexdigest() != digest
                or manifest.get("mediaType") != MANIFEST_TYPE
                or manifest.get("config", {}).get("digest") != config_digest):
            raise ValueError("published manifest does not match the BuildKit image")
        return reference

    def _progress(self, phase, message):
        path = BUILD_PROGRESS.get()
        if path is not None:
            record_progress(path, phase, message, timeout_seconds=self.config.get("timeout_seconds", 600))

    def _publish(self, tag: str, context: Path) -> str:
        # Legacy Docker builds on the containerd store can produce mixed OCI /
        # Docker layer media types. Use Docker's archive view and a reviewed
        # format converter; do not trust a local RepoDigests entry as a receipt.
        archive = context / "image.tar"
        digest_file = context / "published.digest"
        self._docker(["image", "save", "--output", str(archive), tag])
        source = "docker-archive:" + str(archive)
        original = json.loads(self._skopeo(["inspect", "--raw", source], capture=True))
        auth = self.config.get("registry_auth_file", "/root/.docker/config.json")
        self._skopeo(["copy", "--format", "v2s2", "--dest-precompute-digests", "--authfile", auth,
                      "--digestfile", str(digest_file), source, "docker://" + tag])
        digest = digest_file.read_text().strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ValueError("published image digest is unavailable")
        reference = self.config["repository"] + "@" + digest
        raw = self._skopeo(["inspect", "--authfile", auth, "--raw", "docker://" + reference], capture=True)
        manifest = json.loads(raw)
        if ("sha256:" + hashlib.sha256(raw.encode()).hexdigest() != digest
                or manifest.get("mediaType") != MANIFEST_TYPE
                or manifest.get("config", {}).get("digest") != original.get("config", {}).get("digest")
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(manifest.get("config", {}).get("digest")))):
            raise ValueError("published manifest does not match the fixed image")
        if self.config.get("cleanup_published_image", False):
            self._docker(["image", "rm", tag])
        return reference

    def _docker(self, arguments: list[str], *, capture: bool = False) -> str:
        host = ["--host", self.config["docker_host"]] if "docker_host" in self.config else []
        return self._command([self.config.get("docker", "/usr/bin/docker"), *host, *arguments], capture=capture)

    def _skopeo(self, arguments: list[str], *, capture: bool = False) -> str:
        return self._command([self.config.get("skopeo", "/usr/bin/skopeo"),
                              "--tmpdir", str(self.root),
                              "--command-timeout", str(self.config.get("timeout_seconds", 600)) + "s",
                              *arguments], capture=capture)

    def _command(self, command: list[str], *, capture: bool = False) -> str:
        temporary = self.root / "tmp"
        temporary.mkdir(exist_ok=True, mode=0o700)
        environment = {**os.environ, "TMPDIR": str(temporary)}
        path = BUILD_LOG.get()
        log_start = path.stat().st_size if path is not None and path.is_file() else 0
        try:
            if path is not None and not capture:
                with path.open("ab") as log:
                    subprocess.run(command, check=True, stdout=log, stderr=log,
                                   env=environment,
                                   timeout=self.config.get("timeout_seconds", 600))
                return ""
            result = subprocess.run(command,
                                    check=True, stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                                    env=environment,
                                    stderr=subprocess.DEVNULL, timeout=self.config.get("timeout_seconds", 600))
            return result.stdout.decode() if capture else ""
        except (OSError, subprocess.SubprocessError) as exc:
            tail = ""
            if not capture and path is not None and path.is_file():
                with path.open("rb") as log:
                    log.seek(max(log_start, log.seek(0, 2) - 8192))
                    tail = log.read().decode(errors="replace").lower()
            if isinstance(exc, OSError) and exc.errno == errno.ENOSPC or "no space left on device" in tail:
                raise BuildStorageError("BUILD_DISK_EXHAUSTED", {"publication_uncertain": True}) from None
            if "session healthcheck" in tail and "context deadline exceeded" in tail:
                raise BuildTransportError("BUILD_SESSION_TIMEOUT", {"publication_uncertain": True, "scheduler_submitted": False}) from None
            if len(command) > 3 and command[3] == "cp":
                raise BuildTransportError("BUILD_CONTEXT_TRANSPORT", {"publication_uncertain": True, "scheduler_submitted": False}) from None
            raise ValueError("OCI packaging failed; check registry access and base image availability") from exc


class BuilderServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        try:
            uid = struct.unpack("3i", self.connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[1]
            if uid != self.server.builder.config["client_uid"]:
                raise ValueError("desktop archive client is not authorized")
            target = urlsplit(self.path)
            query = parse_qs(target.query, strict_parsing=True)
            if target.path != "/data-stage/archive" or set(query) != {"project", "asset_id"} or any(len(v) != 1 for v in query.values()):
                raise ValueError("invalid desktop archive request")
            data = {key: value[0] for key, value in query.items()}
            result = self.server.builder.desktop_stage("asset", {"binding": {"project": data["project"], "kind": "asset"},
                                                        "asset_id": data["asset_id"]}, b"")
            if not result["ok"]:
                raise ValueError("desktop archive is unavailable")
            config = self.server.builder.config
            command = [config.get("docker_bin", "docker"), "--host", config["docker_host"], "exec", "-i",
                       config["data_upload_container"], "python", "-m", "ml_exp_server.desktop_upload", "archive", json.dumps(data)]
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except Exception:
            self.reply(409, {"error": "desktop archive is unavailable"})
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-tar")
            self.send_header("Content-Length", str(result["result"]["archive_bytes"]))
            self.end_headers()
            self.connection.settimeout(300)
            shutil.copyfileobj(process.stdout, self.wfile, 1024 ** 2)
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=15)

    def do_POST(self):
        status = 200
        try:
            uid = struct.unpack("3i", self.connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[1]
            if uid != self.server.builder.config["client_uid"]:
                raise ValueError("image packaging client is not authorized")
            size = int(self.headers.get("Content-Length", "0"))
            if self.path.startswith("/data-stage/"):
                if not 0 <= size <= 64 * 1024 ** 2:
                    raise ValueError("invalid desktop data part size")
                metadata = self.headers.get("X-ML-Expd-Data", "")
                if len(metadata) > 8192:
                    raise ValueError("invalid desktop data metadata")
                self.connection.settimeout(300)
                body = self.rfile.read(size)
                if len(body) != size:
                    raise ValueError("truncated desktop data part")
                result = self.server.builder.desktop_stage(self.path.removeprefix("/data-stage/"), json.loads(metadata), body)
                status = 200 if result["ok"] else result.get("status_code", 409)
                result = result["result"] if result["ok"] else result
                self.reply(status, result)
                return
            if not 0 < size <= 16384 or self.path != "/":
                raise ValueError("invalid image packaging request")
            self.connection.settimeout(15)
            payload = json.loads(self.rfile.read(size))
            result = self.server.builder.request(payload)
        except BuildStorageError as exc:
            status, result = 409, {"error": exc.code, "code": exc.code, "details": exc.details}
        except BuildTransportError as exc:
            status, result = 409, {"error": exc.code, "code": exc.code, "details": exc.details}
        except ApplicationError as exc:
            status, result = exc.status_code, {"error": exc.code, "code": exc.code, "status_code": exc.status_code,
                                              "details": {"publication_uncertain": True, "scheduler_submitted": False}}
        except Exception:
            status, result = 409, {"error": "image packaging request failed"}
        self.reply(status, result)

    def reply(self, status, result):
        encoded = json.dumps(result).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    socket_path = Path(config["socket"])
    socket_path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    os.chown(socket_path.parent, 0, config["client_gid"])
    socket_path.unlink(missing_ok=True)
    with BuilderServer(str(socket_path), Handler) as server:
        server.builder = ImageBuilder(config)
        os.chown(socket_path, 0, config["client_gid"])
        socket_path.chmod(0o660)
        server.serve_forever()


if __name__ == "__main__":
    main()
