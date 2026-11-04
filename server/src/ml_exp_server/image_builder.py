"""Private OCI publisher with fixed recipes and an opt-in client Dockerfile recipe."""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
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

from dockerfile_parse import DockerfileParser

from .projects.source_revisions import _tree_digest
from .storage import atomic_json
from .dockerfile_build import DOCKERFILE_RECIPE, INTERNAL, inspect_dockerfile, managed_dockerfile, worker_digest
from .worker_contract import CAPABILITIES, WORKER_CONTRACT, install_workers
from .runs.execution_progress import record_progress, progress_view
from .application_errors import ApplicationError
from .builds.image_build_context import BUILD_LOG, BUILD_PROGRESS, BUILD_CACHE, BUILD_RETRY
from .builds.builder_client import IMAGE, ID, PROJECT, MANIFEST_TYPE, bundle_id, BuildStorageError, BuildTransportError, BuildReceiptMissingError
from .builds.build_recovery import publication_absence


class ImageBuilder:
    def __init__(self, config: dict):
        self.config = config
        proxy = config.get("buildkit_http_proxy")
        if "buildkit_http_proxy" in config and (not isinstance(proxy, str) or not re.fullmatch(r"https?://[A-Za-z0-9._-]+:[0-9]{1,5}/?", proxy)):
            raise ValueError("BuildKit HTTP proxy must be a credential-free HTTP endpoint")
        network = config.get("buildkit_network")
        if "buildkit_network" in config and (not isinstance(network, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", network)):
            raise ValueError("invalid private BuildKit network")
        if config.get("docker_host"):
            raise ValueError("remote Docker publishing was removed; use PAI DSW or local BuildKit")
        self.root = Path(config["state_root"])
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.config.get("ephemeral_buildkit"):
            with (self.root / "ephemeral-buildkit.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                self._recover_builder()

    def request(self, request: dict) -> dict:
        if str(request.get("operation", "")).startswith("native-"):
            from .builds.native_image import request as native_request
            return native_request(self, request)
        if request.get("operation") == "health":
            return {"status": "ok", "dockerfile_builds": bool(self.config.get("allow_dockerfile_builds"))
                    and self.config.get("publisher") in {"buildkit", "pai-dsw"}}
        if request.get('operation') == 'mirror':
            from .builds.image_mirror import mirror
            return mirror(self, request)
        project, source_id, image = (request.get(k, "") for k in ("project", "source_id", "base_image"))
        if (not PROJECT.fullmatch(project) or not re.fullmatch(r"source\.[0-9a-f]{64}", source_id)
                or not IMAGE.fullmatch(image)):
            raise ValueError("invalid image packaging identity")
        operation = request.get("operation")
        custom_path = request.get("dockerfile")
        if operation in {"get", "reconcile", "logs", "progress"}:
            current = bundle_id(project, source_id, image, dockerfile_path=custom_path) if isinstance(custom_path, str) else None
            pinned = request.get("bundle_id", current)
            if not isinstance(pinned, str) or not ID.fullmatch(pinned):
                raise ValueError("invalid frozen build identity")
            receipt = self.root / f"{pinned}.json"
            definition = self.root / f"{pinned}.definition.json"
            evidence = receipt if receipt.is_file() else definition if definition.is_file() else None
            value = None
            if evidence is not None:
                value = json.loads(evidence.read_text())
                if any(value.get(key) != expected for key, expected in (
                        ("bundle_id", pinned), ("project", project), ("source_id", source_id), ("base_image", image))):
                    raise ValueError("historical packaging receipt identity mismatch")
            elif pinned != current:
                raise ValueError("historical packaging receipt is unavailable")
            if operation in {"get", "reconcile"}:
                if not receipt.is_file():
                    if operation == 'reconcile' and evidence is not None and (self.root / (pinned + '.acr.json')).exists():
                        from .builds.acr_fallback import reconcile
                        recovered = reconcile(self, value)
                        if recovered is not None:
                            return recovered
                    proof = (publication_absence(self, pinned)
                             if operation == "reconcile" and pinned == current and evidence is not None else None)
                    raise BuildReceiptMissingError("image bundle has not been published", retry_proof=proof)
                return value
            path = self.root / f"{pinned}.log"
            content = ""
            if path.exists():
                with path.open("rb") as log:
                    log.seek(max(0, path.stat().st_size - 8192))
                    content = log.read().decode("utf-8", errors="replace")
            result = {"bundle_id": pinned, "lines": content.splitlines(), "truncated": path.exists() and path.stat().st_size > 8192}
            if operation == "progress":
                status = "READY" if receipt.exists() else "EXECUTING"
                result["progress"] = progress_view(self.root / f"{pinned}.progress.json", status,
                                                   active=status == "EXECUTING",
                                                   last_activity=path.stat().st_mtime if path.exists() else None)
            return result
        if (not isinstance(custom_path, str) or request.get("requirements") is not None
                or request.get("packaging_revision", DOCKERFILE_RECIPE) != DOCKERFILE_RECIPE):
            raise ValueError("new builds require the Dockerfile recipe")
        identity = bundle_id(project, source_id, image, dockerfile_path=custom_path)
        metadata_path = self.root / f"{identity}.json"
        with (self.root / f"{identity}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                if metadata.get("bundle_id") != identity:
                    raise ValueError("image packaging metadata mismatch")
                return metadata
            if operation != "build":
                raise ValueError("unsupported image packaging operation")
            retried = (self.root / (identity + '.definition.json')).is_file()
            if not self.config.get("allow_dockerfile_builds") or self.config.get("publisher") not in {"buildkit", "pai-dsw"}:
                raise ValueError("Dockerfile builds require an enabled image publisher")
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
            inspection = inspect_dockerfile(tree, custom_path)
            if image != inspection["base_images"][-1]:
                raise ValueError("Dockerfile base differs from the frozen request")
            registry = self.config["repository"]
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]+", registry):
                raise ValueError("invalid image packaging repository")
            tag = registry + ":bundle-" + identity
            progress_path = self.root / f"{identity}.progress.json"
            record_progress(progress_path, "PREPARING_CONTEXT", "Validating and preparing the frozen source build context")
            with tempfile.TemporaryDirectory(prefix="build-", dir=self.root) as directory:
                context = Path(directory)
                copied = context / INTERNAL / "source"
                shutil.copytree(tree, copied, symlinks=False)
                if _tree_digest(copied) != expected:
                    raise ValueError("source changed while preparing packaging context")
                for path in copied.rglob("*"):
                    path.chmod(0o555 if path.is_dir() or path.stat().st_mode & 0o111 else 0o444)
                copied.chmod(0o555)
                shutil.copytree(tree, context, dirs_exist_ok=True, symlinks=False)
                context.chmod(0o700)
                for name in ("Dockerfile", ".dockerignore", "Dockerfile.dockerignore"):
                    control = context / name
                    if control.exists():
                        control.chmod(0o600)
                # Preserve the complete sealed source even under a restrictive client ignore file.
                ignore = context / ".dockerignore"
                with ignore.open("a") as output:
                    output.write(f"\n!{INTERNAL}/\n!{INTERNAL}/**\n")
                specific_ignore = context / "Dockerfile.dockerignore"
                if specific_ignore.exists():
                    with specific_ignore.open("a") as output:
                        output.write(f"\n!{INTERNAL}/\n!{INTERNAL}/**\n")
                install_workers(context / INTERNAL)
                recipe = managed_dockerfile(inspection, source_id)
                (context / "Dockerfile").write_text(recipe)
                atomic_json(self.root / f"{identity}.definition.json", {"bundle_id": identity, "project": project,
                            "source_id": source_id, "base_image": image})
                log_context = BUILD_LOG.set(self.root / f"{identity}.log")
                progress_context = BUILD_PROGRESS.set(progress_path)
                retry_context = BUILD_RETRY.set(identity if retried else None)
                # One mutable acceleration tag per project; not part of source,
                # Docker context, execution identity or local persistent storage.
                cache_context = BUILD_CACHE.set(registry + ":buildcache-" + hashlib.sha256(project.encode()).hexdigest()[:24]
                                               if self.config.get("registry_cache", False) else None)
                result = {"bundle_id": identity, "project": project, "source_id": source_id,
                          "base_image": image, "recipe": DOCKERFILE_RECIPE,
                          "manifest_type": MANIFEST_TYPE, "worker_contract": WORKER_CONTRACT,
                          "worker_sha256": worker_digest(), "capabilities": list(CAPABILITIES),
                          "dockerfile_sha256": hashlib.sha256(recipe.encode()).hexdigest(),
                          "dockerfile": {k: v for k, v in inspection.items() if k != 'text'}}
                try:
                    if self.config.get("publisher") == "pai-dsw":
                        from .builds.acr_fallback import publish_runtime
                        published = publish_runtime(self, tag, context, result)
                    else:
                        published = self._publish_buildkit(tag, context)
                finally:
                    BUILD_LOG.reset(log_context)
                    BUILD_PROGRESS.reset(progress_context)
                    BUILD_CACHE.reset(cache_context)
                    BUILD_RETRY.reset(retry_context)
            result['image'] = published
            atomic_json(metadata_path, result)
            record_progress(progress_path, "READY", "Published image manifest and worker identity verified")
            return result


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
            self._recover_builder()
            self._storage_preflight(context)
            name = "ml-expd-" + uuid.uuid4().hex
            atomic_json(self.root / "ephemeral-builder.json", {"name": name})
            try:
                self._docker(["buildx", "create", "--name", name, "--driver", "docker-container",
                              "--driver-opt", "image=" + image, "--driver-opt", "default-load=false",
                              *(["--driver-opt", "env.HTTP_PROXY=" + self.config["buildkit_http_proxy"]]
                                if "buildkit_http_proxy" in self.config else []),
                              *(["--driver-opt", "network=" + self.config["buildkit_network"]]
                                if "buildkit_network" in self.config else [])])
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
        if path is None:
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
            stats = os.statvfs(path)
            available_bytes, available_inodes = stats.f_bavail * stats.f_frsize, stats.f_favail
            context_bytes = sum(p.stat().st_size for p in context.rglob("*") if p.is_file())
            factor = float(self.config.get("build_expansion_factor", 4))
            reserve = int(self.config.get("build_reserve_bytes", 2 * 1024 ** 3))
            required = int(compressed * factor) + 2 * context_bytes + reserve
            details = {"available_bytes": available_bytes, "required_bytes": required,
                       "available_inodes": available_inodes, "required_inodes": int(self.config.get("build_min_free_inodes", 100000)),
                       "compressed_base_bytes": compressed, "expansion_factor": factor, "reserve_bytes": reserve,
                       "estimate_kind": "conservative_compressed_layer_budget", "scheduler_submitted": False}
        except (OSError, ValueError, KeyError, TypeError, StopIteration, subprocess.SubprocessError) as exc:
            raise BuildStorageError("BUILD_STORAGE_UNCHECKED", {"scheduler_submitted": False,
                                    "error_class": type(exc).__name__}) from None
        # Arbitrary RUN commands may grow beyond this estimate. Runtime ENOSPC
        # still gets a distinct error and owned ephemeral-builder cleanup.
        if details["available_bytes"] < required or details["available_inodes"] < details["required_inodes"]:
            raise BuildStorageError("BUILD_STORAGE_INSUFFICIENT", details)
        self._progress("BUILD_STORAGE_READY", f"Builder storage ready: {details['available_bytes']} bytes / {details['available_inodes']} inodes available; {required} bytes / {details['required_inodes']} inodes required")


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
        self._remove_builder(name)
        path.unlink()

    def _buildkit_image(self, tag: str, context: Path, builder: str) -> str:
        # BuildKit streams new layers to the registry and reuses existing base
        # blobs. Exporting a complete CUDA image twice exhausts small servers.
        metadata = context / "buildkit.json"
        network = "default" if (context / INTERNAL).exists() else "none"
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
        log = BUILD_LOG.get()
        if log is not None:
            with log.open('a') as output:
                output.write(phase + ': ' + message + '\n')
        path = BUILD_PROGRESS.get()
        if path is not None:
            record_progress(path, phase, message, timeout_seconds=self.config.get("timeout_seconds", 600))

    def _docker(self, arguments: list[str], *, capture: bool = False) -> str:
        return self._command([self.config.get("docker", "/usr/bin/docker"),
                              "--host", "unix:///var/run/docker.sock", *arguments], capture=capture)

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


    def do_POST(self):
        status = 200
        try:
            uid = struct.unpack("3i", self.connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[1]
            if uid != self.server.builder.config["client_uid"]:
                raise ValueError("image packaging client is not authorized")
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 8 * 1024 * 1024 or self.path != "/":
                raise ValueError("invalid image packaging request")
            self.connection.settimeout(15)
            payload = json.loads(self.rfile.read(size))
            result = self.server.builder.request(payload)
        except BuildReceiptMissingError as exc:
            status, result = 409, {"error": "image bundle has not been published", "code": "BUILD_RECEIPT_MISSING"}
            if exc.retry_proof is not None:
                result["retry_proof"] = exc.retry_proof
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
