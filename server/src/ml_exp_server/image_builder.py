"""Private OCI packaging worker: fixed FROM/COPY recipe, no project build scripts."""

from __future__ import annotations

import argparse
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

from .source_revisions import _tree_digest
from .storage import atomic_json
from .environment_build import DEPENDENCY_RECIPE, dockerfile, inspect_requirements, installer_digest, requirements_path


IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}$")
ID = re.compile(r"^[0-9a-f]{64}$")
PROJECT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
RECIPE = "source-copy-docker-v2-v2"
MANIFEST_TYPE = "application/vnd.docker.distribution.manifest.v2+json"


def bundle_id(project: str, source_id: str, base_image: str, requirements: str | None = None) -> str:
    worker_sha = hashlib.sha256(Path(__file__).with_name("container_worker.py").read_bytes()).hexdigest()
    fields = [project, source_id, base_image, RECIPE, worker_sha]
    if requirements is not None:
        fields.extend([DEPENDENCY_RECIPE, requirements_path(requirements), installer_digest()])
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

    def request(self, request: dict) -> dict:
        project, source_id, image = (request.get(k, "") for k in ("project", "source_id", "base_image"))
        if (not PROJECT.fullmatch(project) or not re.fullmatch(r"source\.[0-9a-f]{64}", source_id)
                or not IMAGE.fullmatch(image)):
            raise ValueError("invalid image packaging identity")
        requirements = request.get("requirements")
        recipe_version = DEPENDENCY_RECIPE if requirements is not None else RECIPE
        if request.get("packaging_revision", recipe_version) != recipe_version:
            raise ValueError("image packaging revision mismatch")
        identity = bundle_id(project, source_id, image, requirements)
        metadata_path = self.root / f"{identity}.json"
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
            registry = self.config["repository"]
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]+", registry):
                raise ValueError("invalid image packaging repository")
            tag = registry + ":bundle-" + identity
            with tempfile.TemporaryDirectory(prefix="build-", dir=self.root) as directory:
                context = Path(directory)
                shutil.copytree(tree, context / "source", symlinks=False)
                if _tree_digest(context / "source") != expected:
                    raise ValueError("source changed while preparing packaging context")
                for path in (context / "source").rglob("*"):
                    path.chmod(0o555 if path.is_dir() or path.stat().st_mode & 0o111 else 0o444)
                (context / "source").chmod(0o555)
                shutil.copyfile(Path(__file__).with_name("container_worker.py"), context / "worker.py")
                (context / "worker.py").chmod(0o444)
                if dependencies is not None:
                    shutil.copyfile(tree / requirements, context / "requirements.txt")
                    shutil.copyfile(Path(__file__).with_name("dependency_install.py"), context / "dependency_install.py")
                recipe = dockerfile(image, source_id, requirements)
                (context / "Dockerfile").write_text(recipe)
                if self.config.get("publisher", "archive") == "buildkit":
                    published = self._publish_buildkit(tag, context)
                else:
                    self._docker(["build", "--network=none", "--tag", tag, directory])
                    published = self._publish(tag, context)
            result = {"bundle_id": identity, "project": project, "source_id": source_id,
                      "base_image": image, "image": published, "recipe": recipe_version,
                      "manifest_type": MANIFEST_TYPE}
            if dependencies is not None:
                result.update(dependencies=dependencies, dockerfile_sha256=hashlib.sha256(recipe.encode()).hexdigest(),
                              installer_sha256=installer_digest())
            atomic_json(metadata_path, result)
            return result

    def _publish_buildkit(self, tag: str, context: Path) -> str:
        # BuildKit streams new layers to the registry and reuses existing base
        # blobs. Exporting a complete CUDA image twice exhausts small servers.
        metadata = context / "buildkit.json"
        network = "default" if (context / "requirements.txt").exists() else "none"
        self._docker(["buildx", "build", "--builder", "default", "--network=" + network,
                      "--provenance=false", "--sbom=false", "--tag", tag,
                      "--metadata-file", str(metadata), "--output",
                      "type=image,push=true,oci-mediatypes=false,compression=gzip",
                      str(context)])
        value = json.loads(metadata.read_text())
        digest = value.get("containerimage.digest", "")
        config_digest = value.get("containerimage.config.digest", "")
        if not all(re.fullmatch(r"sha256:[0-9a-f]{64}", item)
                   for item in (digest, config_digest)):
            raise ValueError("BuildKit publication identity is unavailable")
        reference = self.config["repository"] + "@" + digest
        raw = self._skopeo(["inspect", "--authfile",
                            self.config.get("registry_auth_file", "/root/.docker/config.json"),
                            "--raw", "docker://" + reference], capture=True)
        manifest = json.loads(raw)
        if ("sha256:" + hashlib.sha256(raw.encode()).hexdigest() != digest
                or manifest.get("mediaType") != MANIFEST_TYPE
                or manifest.get("config", {}).get("digest") != config_digest):
            raise ValueError("published manifest does not match the BuildKit image")
        return reference

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
        return reference

    def _docker(self, arguments: list[str], *, capture: bool = False) -> str:
        return self._command([self.config.get("docker", "/usr/bin/docker"), *arguments], capture=capture)

    def _skopeo(self, arguments: list[str], *, capture: bool = False) -> str:
        return self._command([self.config.get("skopeo", "/usr/bin/skopeo"),
                              "--tmpdir", str(self.root),
                              "--command-timeout", str(self.config.get("timeout_seconds", 600)) + "s",
                              *arguments], capture=capture)

    def _command(self, command: list[str], *, capture: bool = False) -> str:
        try:
            result = subprocess.run(command,
                                    check=True, stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
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
