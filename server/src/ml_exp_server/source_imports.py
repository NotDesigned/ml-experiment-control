"""Import client source without executing repository code or copying live runs."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import gzip
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import BinaryIO
from urllib.parse import urlsplit

import yaml

from .application_errors import ApplicationError
from .project_service import ProjectApplicationService
from .source_revisions import _is_protected, _tree_digest, resolve_source_tree
from .storage import atomic_json, atomic_text, utc_now


IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")


@contextmanager
def source_lock(root: Path, project: str):
    if not IDENTITY.fullmatch(project):
        raise ApplicationError("invalid project identity", code="SOURCE_IMPORT_BLOCKED")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / f".{project}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def unpack_source(stream: BinaryIO, target: Path, policy, *, allow_empty: bool = False) -> None:
    """Extract bounded regular files; reject links, devices, secrets and escapes."""
    seen: set[str] = set()
    size = 0
    magic = stream.read(2)
    stream.seek(0)
    expanded = gzip.GzipFile(fileobj=stream) if magic == b"\x1f\x8b" else stream
    class BoundedReader:
        remaining = policy.max_source_bytes + policy.max_source_files * 2048 + 1024 * 1024
        def read(self, size):
            if size < 0 or size > self.remaining:
                raise ValueError("expanded archive stream limit exceeded")
            chunk = expanded.read(size)
            self.remaining -= len(chunk)
            return chunk
    with tarfile.open(fileobj=BoundedReader(), mode="r|") as archive:
        for member in archive:
            raw = member.name
            parts = PurePosixPath(raw).parts
            if raw in {".", "./"} and member.isdir():
                continue
            if (not parts or raw.startswith("/") or ".." in parts or "\\" in raw
                    or "\x00" in raw or _is_protected(raw)):
                raise ValueError("archive contains a forbidden source path")
            relative = PurePosixPath(*[p for p in parts if p != "."]).as_posix()
            if relative in seen:
                raise ValueError("archive contains duplicate paths")
            seen.add(relative)
            if len(seen) > policy.max_source_files:
                raise ValueError("source file limit exceeded")
            destination = target / relative
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or member.size < 0:
                raise ValueError("source archives contain only regular files and directories")
            size += member.size
            if size > policy.max_source_bytes:
                raise ValueError("expanded source size limit exceeded")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("archive file is unavailable")
            with source, destination.open("xb") as output:
                remaining = member.size
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("truncated source archive")
                    output.write(chunk)
                    remaining -= len(chunk)
                output.flush()
                os.fsync(output.fileno())
            destination.chmod(0o500 if member.mode & 0o111 else 0o400)
    if not allow_empty and not any(path.is_file() for path in target.rglob("*")):
        raise ValueError("source archive contains no files")


def seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir():
            path.chmod(0o500)
        elif path.is_file():
            path.chmod(0o500 if path.stat().st_mode & 0o111 else 0o400)
    root.chmod(0o500)


def remove_staging(root: Path) -> None:
    """Unseal only the unpublished temporary tree owned by this operation."""
    if root.exists():
        root.chmod(0o700)
        for path in root.rglob("*"):
            if path.is_dir():
                path.chmod(0o700)
        shutil.rmtree(root)


class SourceImportService:
    def __init__(self, runtime):
        self.runtime = runtime
        self.root = runtime.config.project_registry_root_path()
        self.sources = self.root / "source-revisions" / "sources"

    def require_enabled(self) -> None:
        if not self.runtime.config.action_runtime.allow_source_imports:
            raise ApplicationError("source imports are disabled", code="SOURCE_IMPORT_BLOCKED")

    def publish(self, project: str, stream: BinaryIO, observation: dict) -> dict:
        self.require_enabled()
        with source_lock(self.sources, project):
            temporary = Path(tempfile.mkdtemp(prefix=".import-", dir=self.sources))
            try:
                tree = temporary / "tree"
                tree.mkdir()
                unpack_source(stream, tree, self.runtime.config.container_execution)
                digest = _tree_digest(tree)
                source_id = "source." + digest.removeprefix("sha256:")
                metadata = {"schema_version": 1, "project": project,
                            "source_id": source_id, "tree": "tree", "tree_digest": digest,
                            "created_at": utc_now(), "observation": observation}
                destination = self.sources / project / source_id
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                if not destination.exists():
                    atomic_json(temporary / "source.json", metadata)
                    seal_tree(temporary)
                    # POSIX cross-parent directory rename needs owner write on
                    # the moved directory. Its contents remain sealed throughout.
                    temporary.chmod(0o700)
                    os.rename(temporary, destination)
                    destination.chmod(0o500)
                    fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                resolve_source_tree(self.runtime.config, project, source_id)
                registration = self._ensure_project(project)
                return {"project": project, "source_id": source_id, "tree_digest": digest,
                        "observation": observation, "registration": registration}
            except (OSError, ValueError, tarfile.TarError) as exc:
                raise ApplicationError(str(exc), code="SOURCE_IMPORT_BLOCKED") from exc
            finally:
                remove_staging(temporary)

    def _ensure_project(self, project: str) -> dict:
        records = self.runtime.project_records()
        if any(record.project == project for record in records):
            return {"existing": True}
        root = self.root / "managed" / project
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        tool = root / "tools" / "experimentctl.py"
        atomic_text(tool, "from ml_exp_server.container_controller import cli\ncli()\n")
        manifest = root / "experiments" / "research_project.yaml"
        payload = {"schema_version": 1, "project": project, "title": project,
                   "run_roots": [], "campaigns": [], "controller": {
                       "python": sys.executable, "experimentctl": "tools/experimentctl.py",
                       "workdir": ".", "capabilities": {
                           "container_execution": True, "daemon_source_revision": True,
                           "submit_outbox": True, "run_identity_v2": True,
                           "authored_campaign_revision": True, "cancel_outbox": True}}}
        atomic_text(manifest, yaml.safe_dump(payload, sort_keys=False))
        return ProjectApplicationService(self.runtime).register(manifest)

    def archive(self, project: str, data: BinaryIO, digest: str) -> dict:
        if not SHA256.fullmatch(digest):
            raise ApplicationError("sha256 must be 64 lowercase hex digits", code="SOURCE_IMPORT_BLOCKED")
        actual = hashlib.sha256()
        while chunk := data.read(1024 * 1024):
            actual.update(chunk)
        data.seek(0)
        if actual.hexdigest() != digest:
            raise ApplicationError("archive sha256 mismatch", code="SOURCE_IMPORT_BLOCKED")
        return self.publish(project, data, {"kind": "archive", "sha256": digest})

    def git(self, project: str, url: str, commit: str) -> dict:
        self.require_enabled()
        parsed = urlsplit(url)
        policy = self.runtime.config.container_execution
        if (parsed.scheme != "https" or parsed.hostname not in policy.git_hosts
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.port not in {None, 443} or not COMMIT.fullmatch(commit)):
            raise ApplicationError("Git import requires an allowed HTTPS host and exact commit", code="SOURCE_IMPORT_BLOCKED")
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".git-import-", dir=self.root) as directory:
            checkout = Path(directory)
            environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                           "HOME": "/nonexistent", "GIT_CONFIG_NOSYSTEM": "1",
                           "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_TERMINAL_PROMPT": "0"}
            command = ["git", "-c", "core.hooksPath=/dev/null", "-c", "http.followRedirects=false", "-C", directory]
            def git(*args):
                return subprocess.run([*command, *args], env=environment, check=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      timeout=60).stdout
            try:
                git("init", "-q")
                git("fetch", "--depth=1", "--no-tags", url, commit)
                if git("rev-parse", "FETCH_HEAD").decode().strip() != commit:
                    raise ValueError("fetched commit differs from requested commit")
                archive = checkout / "source.tar"
                with archive.open("wb") as output:
                    subprocess.run([*command, "archive", "--format=tar", commit], env=environment,
                                   stdout=output, stderr=subprocess.DEVNULL, check=True, timeout=60)
                if archive.stat().st_size > policy.max_archive_bytes:
                    raise ValueError("Git source archive exceeds import limit")
                with archive.open("rb") as data:
                    return self.publish(project, data, {"kind": "git", "url": url, "commit": commit})
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                raise ApplicationError("Git source import failed; check host, commit and repository access", code="SOURCE_IMPORT_BLOCKED") from exc

    def read(self, project: str, source_id: str) -> dict:
        resolve_source_tree(self.runtime.config, project, source_id)
        metadata = self.sources / project / source_id / "source.json"
        return json.loads(metadata.read_text())
