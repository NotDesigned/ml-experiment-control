"""Attempt-scoped artifact reads anchored with no-follow file descriptors."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Iterator

from .application_errors import ApplicationError


_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def open_directory(path: Path) -> int:
    """Open every component without following a mutable symlink."""
    if not path.is_absolute():
        raise ValueError("artifact root must be absolute")
    descriptor = os.open("/", _DIRECTORY)
    try:
        for part in path.parts[1:]:
            if part in {".", ".."}:
                raise ValueError("invalid artifact root")
            child = os.open(part, _DIRECTORY, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def safe_parts(relative: str) -> tuple[str, ...]:
    path = PurePosixPath(relative)
    if (not relative or path.is_absolute() or ".." in path.parts or "\\" in relative
            or "\x00" in relative or any(part.startswith(".") for part in path.parts)):
        raise ValueError("invalid artifact path")
    return path.parts


@dataclass
class OpenArtifact:
    descriptor: int
    size: int
    name: str
    etag: str

    def chunks(self, start: int = 0, length: int | None = None) -> Iterator[bytes]:
        remaining = self.size - start if length is None else length
        try:
            os.lseek(self.descriptor, start, os.SEEK_SET)
            while remaining:
                chunk = os.read(self.descriptor, min(remaining, 1024 * 1024))
                if not chunk:
                    raise OSError("artifact changed or was truncated during download")
                remaining -= len(chunk)
                yield chunk
        finally:
            self.close()

    def close(self):
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


class ArtifactService:
    def __init__(self, runtime):
        self.runtime = runtime

    def roots(self, project: str, run_id: str, attempt_id: str, *, restore=True) -> list[tuple[str, Path]]:
        row = self.runtime.index.get_run(project, run_id)
        if row is None or attempt_id not in {a.attempt_id for a in row.attempts}:
            raise ApplicationError("unknown Run/Attempt", status_code=404, code="UNKNOWN_ATTEMPT")
        config = self.runtime.config.container_execution.artifact_store_file
        if config and restore:
            from .artifact_store import ArtifactStore
            ArtifactStore(Path(config), self.runtime.config.project_registry_root_path()).restore_cache(project, run_id, attempt_id)
        attempt = Path(row.run_dir) / "attempts" / attempt_id
        # Every source is exact-Attempt evidence. Never fall back to another Attempt.
        if config:
            from .artifact_store import ArtifactStore, artifact_released
            with ArtifactStore(Path(config), self.runtime.config.project_registry_root_path()).record(project, run_id, attempt_id) as (_, value):
                if artifact_released(value):
                    return []
        return [("outputs", attempt / "uploaded_outputs"),
                ("outputs", attempt / "outputs"),
                ("outputs", attempt / "recovered_outputs"),
                ("outputs", attempt / "collected_run" / "attempts" / attempt_id / "outputs"),
                ("collected", attempt / "collected_run")]

    def list(self, project: str, run_id: str, attempt_id: str, *, checksums=False) -> dict:
        files: dict[str, dict] = {}
        truncated = False
        for namespace, root in self.roots(project, run_id, attempt_id, restore=False):
            try:
                descriptor = open_directory(root)
            except (OSError, ValueError):
                continue
            try:
                for directory, _, names, directory_fd in os.fwalk(".", dir_fd=descriptor, follow_symlinks=False):
                    for name in sorted(names):
                        relative = PurePosixPath(directory, name).as_posix()
                        try:
                            safe_parts(relative)
                            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                        except (OSError, ValueError):
                            continue
                        if not stat.S_ISREG(metadata.st_mode):
                            continue
                        path = namespace + "/" + relative
                        if path not in files:
                            files[path] = {"path": path, "bytes": metadata.st_size,
                                           "modified_ns": metadata.st_mtime_ns}
                            if checksums:
                                opened = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
                                with os.fdopen(opened, "rb") as body:
                                    before = os.fstat(body.fileno())
                                    if not stat.S_ISREG(before.st_mode):
                                        raise ValueError("artifact is not a regular file")
                                    digest = hashlib.sha256()
                                    for chunk in iter(lambda: body.read(1024 * 1024), b""):
                                        digest.update(chunk)
                                    checksum = digest.hexdigest()
                                    after = os.fstat(body.fileno())
                                if (metadata.st_ino, metadata.st_size, metadata.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                                    raise ValueError("artifact changed during verification")
                                files[path]["sha256"] = checksum
                        if len(files) >= 10000:
                            truncated = True
                            break
                    if truncated:
                        break
            finally:
                os.close(descriptor)
            if truncated:
                break
        config = self.runtime.config.container_execution.artifact_store_file
        result = None
        released = False
        if config:
            from .artifact_store import ArtifactStore
            store = ArtifactStore(Path(config), self.runtime.config.project_registry_root_path())
            with store.record(project, run_id, attempt_id) as (_, value):
                if value:
                    receipt = value.get("receipt")
                    from .artifact_store import artifact_released
                    released = artifact_released(value)
                    result = value.get("results_ready")
                    if receipt:
                        for item in receipt["files"]:
                            if "outputs/" + item["path"] not in files and len(files) >= 10000:
                                truncated = True
                                break
                            files.setdefault("outputs/" + item["path"], {**item, "path": "outputs/" + item["path"]})
        return {"project": project, "run_id": run_id, "attempt_id": attempt_id,
                "files": [files[key] for key in sorted(files)], "truncated": truncated,
                "available": bool(files) and not released, "collection_required": not bool(files) and not released,
                "storage_status": "RELEASED" if released else "AVAILABLE" if files else "MISSING",
                "training": {"status": "UNKNOWN" if result is None else "COMPLETED" if result["exit_code"] == 0 else "FAILED",
                             "exit_code": None if result is None else result["exit_code"],
                             "evidence": "worker-result-manifest" if result is not None else None}}

    def open(self, project: str, run_id: str, attempt_id: str, relative: str) -> OpenArtifact:
        try:
            parts = safe_parts(relative)
        except ValueError as exc:
            raise ApplicationError("invalid artifact path", code="ARTIFACT_PATH_BLOCKED") from exc
        for namespace, root in self.roots(project, run_id, attempt_id):
            if len(parts) < 2 or parts[0] != namespace:
                continue
            directory = None
            descriptor = None
            try:
                directory = open_directory(root)
                for part in parts[1:-1]:
                    child = os.open(part, _DIRECTORY, dir_fd=directory)
                    os.close(directory)
                    directory = child
                descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError("artifact is not a regular file")
                identity = f"{project}/{run_id}/{attempt_id}/{relative}:{metadata.st_dev}:{metadata.st_ino}:{metadata.st_size}:{metadata.st_mtime_ns}"
                etag = '"' + hashlib.sha256(identity.encode()).hexdigest() + '"'
                result = OpenArtifact(descriptor, metadata.st_size, parts[-1], etag)
                descriptor = None
                return result
            except (OSError, ValueError):
                pass
            finally:
                if directory is not None:
                    os.close(directory)
                if descriptor is not None:
                    os.close(descriptor)
        raise ApplicationError("artifact is not locally collected or path is unavailable", status_code=404, code="ARTIFACT_UNAVAILABLE")
