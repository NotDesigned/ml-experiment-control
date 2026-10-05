"""Private OCI publisher with fixed recipes and an opt-in client Dockerfile recipe."""

from __future__ import annotations

import argparse
from contextvars import ContextVar
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
import uuid

from .source_revisions import _tree_digest
from .storage import atomic_json
from .environment_build import DEPENDENCY_RECIPE, dockerfile, inspect_requirements, installer_digest, requirements_path
from .dockerfile_build import DOCKERFILE_RECIPE, INTERNAL, inspect_dockerfile, managed_dockerfile, worker_digest
from .worker_contract import CAPABILITIES, WORKER_CONTRACT, install_workers, recipe_digest
from .execution_progress import record_progress, progress_view


IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}$")
ID = re.compile(r"^[0-9a-f]{64}$")
PROJECT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
RECIPE = "source-copy-docker-v2-v2"
MANIFEST_TYPE = "application/vnd.docker.distribution.manifest.v2+json"
BUILD_LOG = ContextVar("ml_exp_build_log", default=None)
BUILD_PROGRESS = ContextVar("ml_exp_build_progress", default=None)
BUILD_CACHE = ContextVar("ml_exp_build_cache", default=None)


def bundle_id(project: str, source_id: str, base_image: str, requirements: str | None = None, *, dockerfile_path: str | None = None) -> str:
    fields = [project, source_id, base_image, RECIPE, WORKER_CONTRACT, worker_digest(), recipe_digest()]
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


def builder_request(path: str, payload: dict, *, timeout: int = 600) -> dict:
    connection = UnixConnection(path, timeout=timeout)
    try:
        connection.request("POST", "/", body=json.dumps(payload), headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = json.loads(response.read(65536))
        if response.status != 200:
            raise ValueError(data.get("error", "image packaging failed"))
        return data
    finally:
        connection.close()


class ImageBuilder:
    def __init__(self, config: dict):
        self.config = config
        self.root = Path(config["state_root"])
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.config.get("ephemeral_buildkit"):
            with (self.root / "ephemeral-buildkit.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                self._recover_builder()

    def request(self, request: dict) -> dict:
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
                if not receipt.is_file():
                    raise ValueError("historical packaging receipt is unavailable")
                value = json.loads(receipt.read_text())
                if any(value.get(key) != expected for key, expected in (
                        ("bundle_id", pinned), ("project", project), ("source_id", source_id), ("base_image", image))):
                    raise ValueError("historical packaging receipt identity mismatch")
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

    def _publish_buildkit(self, tag: str, context: Path) -> str:
        if not self.config.get("ephemeral_buildkit", False):
            return self._buildkit_image(tag, context, "default")
        image = self.config.get("buildkit_image", "")
        if not IMAGE.fullmatch(image):
            raise ValueError("ephemeral BuildKit requires a pinned builder image")
        # Serialize this application's transient CUDA layers without touching
        # the host's default builder or any other workload's cache.
        with (self.root / "ephemeral-buildkit.lock").open("a") as lock:
            self._progress("WAITING_BUILDER", "Waiting for the private image builder; no scheduler submission has occurred")
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._recover_builder()
            name = "ml-expd-" + uuid.uuid4().hex
            atomic_json(self.root / "ephemeral-builder.json", {"name": name})
            try:
                self._docker(["buildx", "create", "--name", name, "--driver", "docker-container",
                              "--driver-opt", "image=" + image, "--driver-opt", "default-load=false"])
                return self._buildkit_image(tag, context, name)
            finally:
                # Removing this exact builder also removes its dedicated state
                # volume. No --keep-state, host image load or shared prune.
                self._remove_builder(name)
                (self.root / "ephemeral-builder.json").unlink()

    def _remove_builder(self, name):
        names = self._docker(["buildx", "ls", "--format", "{{.Name}}"], capture=True).splitlines()
        if name in names:
            self._docker(["buildx", "rm", "--force", name])

    def _recover_builder(self):
        path = self.root / "ephemeral-builder.json"
        if not path.exists():
            return
        name = json.loads(path.read_text())["name"]
        if not re.fullmatch(r"ml-expd-[0-9a-f]{32}", name):
            raise ValueError("invalid ephemeral builder recovery identity")
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
                      *cache, str(context)])
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
        return self._command([self.config.get("docker", "/usr/bin/docker"), *arguments], capture=capture)

    def _skopeo(self, arguments: list[str], *, capture: bool = False) -> str:
        return self._command([self.config.get("skopeo", "/usr/bin/skopeo"),
                              "--tmpdir", str(self.root),
                              "--command-timeout", str(self.config.get("timeout_seconds", 600)) + "s",
                              *arguments], capture=capture)

    def _command(self, command: list[str], *, capture: bool = False) -> str:
        temporary = self.root / "tmp"
        temporary.mkdir(exist_ok=True, mode=0o700)
        environment = {**os.environ, "TMPDIR": str(temporary)}
        try:
            path = BUILD_LOG.get()
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
            raise ValueError("OCI packaging failed; check registry access and base image availability") from exc


class BuilderServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        status = 200
        try:
            uid = struct.unpack("3i", self.connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[1]
            if uid != self.server.builder.config["client_uid"]:
                raise ValueError("image packaging client is not authorized")
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 16384 or self.path != "/":
                raise ValueError("invalid image packaging request")
            self.connection.settimeout(15)
            payload = json.loads(self.rfile.read(size))
            result = self.server.builder.request(payload)
        except Exception:
            status, result = 409, {"error": "image packaging request failed"}
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
