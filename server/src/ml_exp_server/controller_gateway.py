"""Single compatibility boundary for invoking a Project controller.

``ml-expd`` owns orchestration and durable server state. Project-specific
materialization still lives behind the project's declared ``experimentctl``
command until projects expose native ``experiment_control`` adapters. All
command construction and subprocess execution is kept here so that legacy
transport does not leak through the daemon.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

import yaml

from experiment_control.runner import CommandRunner as CoreCommandRunner
from experiment_control.runner import SubprocessRunner

from .schemas import ResearchProject


_SECRET = re.compile(
    r"(?i)(?:^|[_-])(?:secret|token|password|credential|access[_-]?key|"
    r"api[_-]?key|proxy|authorization|cookie)(?:$|[_-])"
)
_SNAPSHOT_ROOT_ARGUMENT = "{controller_snapshot}"
_TRUSTED_RUNTIME_SOURCES = {
    "yaml": Path(str(yaml.__file__)).resolve(strict=True).parent,
}


def redact(value: Any) -> Any:
    """Remove common credential shapes from controller results."""
    if isinstance(value, dict):
        return {
            str(key): ("[REDACTED]" if _SECRET.search(str(key)) else redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"(?i)(https?://)[^/@\s]+@", r"\1[REDACTED]@", value)
    return value


@dataclass(frozen=True)
class ControllerCall:
    project: str
    run_id: str
    verb: str
    argv: list[str]
    cwd: Path


class CommandRunner:
    """Adapt the core command runner to the daemon's JSON controller protocol."""

    def __init__(self, runner: CoreCommandRunner | None = None) -> None:
        self.runner = runner or SubprocessRunner()

    def __call__(self, command: list[str], *, cwd: Path, timeout: int) -> dict[str, Any]:
        try:
            result = self.runner.run(
                command, cwd=cwd, check=False, timeout_seconds=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "returncode": None, "timeout": True, "stdout": "", "stderr": str(exc),
            }
        payload: Any = None
        if result.stdout.strip():
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError:
                pass
        return {
            "returncode": result.returncode,
            "timeout": False,
            "payload": redact(payload),
            "stdout": redact(result.stdout[-2000:]),
            "stderr": redact(result.stderr[-2000:]),
        }

    def run_pinned(
        self, command: list[str], *, cwd: Path, timeout: int,
        env: dict[str, str], pass_fds: tuple[int, ...],
    ) -> dict[str, Any]:
        """Execute a reviewed interpreter/cwd through already-open descriptors."""
        try:
            completed = subprocess.run(
                command, cwd=cwd, check=False, text=True, capture_output=True,
                timeout=timeout, env=env, pass_fds=pass_fds,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "returncode": None, "timeout": True, "payload": None,
                "stdout": "", "stderr": str(exc),
            }
        except OSError as exc:
            return {
                "returncode": 127, "timeout": False, "payload": None,
                "stdout": "", "stderr": f"pinned command failed: {type(exc).__name__}",
            }
        payload: Any = None
        if completed.stdout.strip():
            try:
                payload = json.loads(completed.stdout)
            except json.JSONDecodeError:
                pass
        return {
            "returncode": completed.returncode, "timeout": False,
            "payload": redact(payload),
            "stdout": redact(completed.stdout[-2000:]),
            "stderr": redact(completed.stderr[-2000:]),
        }


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _open_regular_nofollow(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise ValueError(f"controller bundle input is not a regular file: {path}")
    return descriptor, metadata


def _file_identity(path: Path) -> dict[str, Any]:
    descriptor, metadata = _open_regular_nofollow(path)
    digest = hashlib.sha256()
    try:
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return {
        "path": str(path), "device": metadata.st_dev, "inode": metadata.st_ino,
        "size": metadata.st_size, "mtime_ns": metadata.st_mtime_ns,
        "sha256": "sha256:" + digest.hexdigest(),
    }


def _safe_source(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ValueError(f"controller bundle path must be relative: {relative}")
    current = root
    for part in candidate.parts:
        current = current / part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"controller bundle rejects symlink: {current}")
    return current


def _copy_regular(source: Path, target: Path) -> None:
    source_fd, _ = _open_regular_nofollow(source)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(target.parent, 0o700)
    target_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        while chunk := os.read(source_fd, 1024 * 1024):
            view = memoryview(chunk)
            while view:
                written = os.write(target_fd, view)
                if written <= 0:
                    raise OSError("short write while building controller snapshot")
                view = view[written:]
        os.fsync(target_fd)
    finally:
        os.close(source_fd)
        os.close(target_fd)


def _copy_source(source: Path, target: Path) -> None:
    metadata = source.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"controller bundle rejects symlink: {source}")
    if stat.S_ISREG(metadata.st_mode):
        _copy_regular(source, target)
        return
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"controller bundle rejects special file: {source}")
    target.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(target, 0o700)
    for child in sorted(source.iterdir(), key=lambda item: item.name):
        if child.name == "__pycache__" or child.suffix in {".pyc", ".pyo"}:
            continue
        _copy_source(child, target / child.name)


def _seal_snapshot_modes(root: Path) -> None:
    """Normalize every private directory/file independently of process umask."""
    os.chmod(root, 0o700)
    for path in root.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"controller snapshot contains a symlink: {path}")
        if stat.S_ISDIR(metadata.st_mode):
            os.chmod(path, 0o700)
        elif stat.S_ISREG(metadata.st_mode):
            os.chmod(path, 0o600)
        else:
            raise ValueError(f"controller snapshot contains a special file: {path}")


def _snapshot_files(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"controller snapshot contains a symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            if stat.S_IMODE(metadata.st_mode) != 0o700:
                raise ValueError(f"controller snapshot directory mode drifted: {relative}")
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"controller snapshot contains a special file: {relative}")
        if relative == Path("manifest.json"):
            continue
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise ValueError(f"controller snapshot file mode drifted: {relative}")
        identity = _file_identity(path)
        files.append({
            "path": relative.as_posix(), "size": identity["size"],
            "sha256": identity["sha256"],
        })
    return files


def _source_tree_identity(root: Path) -> dict[str, Any]:
    """Digest trusted source before and after copying it into an Action."""
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"trusted controller source contains a symlink: {path}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"trusted controller source contains a special file: {path}")
        records.append({
            "path": relative.as_posix(), "size": metadata.st_size,
            "sha256": _file_identity(path)["sha256"],
        })
    encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return {"files": records, "content_sha256": _sha256_bytes(encoded)}


class ProjectControllerGateway:
    """Resolve and invoke the controller declared by one research Project."""

    def __init__(self, runner: Callable[..., dict[str, Any]] | None = None) -> None:
        self.runner = runner or CommandRunner()

    def build(
        self,
        project: ResearchProject,
        campaign: Path,
        verb: str,
        run_id: str,
        *,
        attempt_id: str | None = None,
        dry_run: bool = False,
        extra: Iterable[str] = (),
    ) -> ControllerCall:
        controller = project.controller
        if controller is None:
            raise ValueError(f"project {project.project} has no controller config")
        project_base = (project.base_dir or Path(".")).resolve()
        workdir = Path(controller.workdir)
        if not workdir.is_absolute():
            workdir = project_base / workdir
        workdir = workdir.resolve()
        tool = Path(controller.experimentctl)
        if not tool.is_absolute():
            tool = workdir / tool
        argv = [
            controller.python, str(tool), str(campaign), verb, "--run", run_id,
        ]
        if attempt_id is not None:
            argv.extend(["--attempt-id", attempt_id])
        if dry_run:
            argv.append("--dry-run")
        argv.extend(str(item) for item in extra)
        return ControllerCall(
            project=project.project,
            run_id=run_id,
            verb=verb,
            argv=argv,
            cwd=workdir,
        )

    def execute(self, call: ControllerCall, *, timeout: int) -> dict[str, Any]:
        return self.runner(call.argv, cwd=call.cwd, timeout=timeout)

    def execute_command(
        self, command: list[str], *, cwd: Path, timeout: int,
    ) -> dict[str, Any]:
        """Execute an immutable command stored in an approved action plan."""
        return self.runner(command, cwd=cwd, timeout=timeout)

    @staticmethod
    def _python_path(project: ResearchProject) -> Path:
        controller = project.controller
        if controller is None:
            raise ValueError(f"project {project.project} has no controller config")
        value = controller.python
        if Path(value).is_absolute():
            resolved = Path(value)
        else:
            found = shutil.which(value)
            if found is None:
                raise ValueError(f"controller python is unavailable: {value}")
            resolved = Path(found)
        if not resolved.exists():
            raise ValueError(f"controller python is unavailable: {value}")
        return resolved.resolve(strict=True)

    @staticmethod
    def _git_identity(workdir: Path) -> dict[str, Any]:
        status = subprocess.run(
            ["git", "-C", str(workdir), "status", "--porcelain=v1", "--untracked-files=all"],
            check=False, text=True, capture_output=True, timeout=30,
        )
        if status.returncode != 0 or status.stdout.strip():
            raise ValueError("controller workdir must be a clean Git tree")
        values: dict[str, str] = {}
        for name, revision in (("head", "HEAD"), ("tree", "HEAD^{tree}")):
            result = subprocess.run(
                ["git", "-C", str(workdir), "rev-parse", revision], check=False,
                text=True, capture_output=True, timeout=30,
            )
            if result.returncode != 0 or not result.stdout.strip():
                raise ValueError("controller workdir has no stable Git identity")
            values[name] = result.stdout.strip()
        metadata = workdir.stat()
        return {
            **values, "realpath": str(workdir), "device": metadata.st_dev,
            "inode": metadata.st_ino,
        }

    @staticmethod
    def _package_source(python: Path, package: str) -> Path:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", package):
            raise ValueError(f"invalid controller bundle package: {package}")
        query = (
            "import importlib.util,json,sys;"
            "s=importlib.util.find_spec(sys.argv[1]);"
            "print(json.dumps({'origin':getattr(s,'origin',None),"
            "'locations':list(getattr(s,'submodule_search_locations',[]) or [])}))"
        )
        result = subprocess.run(
            [str(python), "-I", "-c", query, package], check=False,
            text=True, capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            raise ValueError(f"controller package is unavailable: {package}")
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(f"controller package lookup failed: {package}") from exc
        locations = payload.get("locations") if isinstance(payload, dict) else None
        if isinstance(locations, list) and len(locations) == 1:
            return Path(str(locations[0])).resolve(strict=True)
        origin = payload.get("origin") if isinstance(payload, dict) else None
        if isinstance(origin, str) and origin:
            return Path(origin).resolve(strict=True)
        raise ValueError(f"controller package has no stable source root: {package}")

    @staticmethod
    def _trusted_core_source() -> tuple[Path, dict[str, Any]]:
        """Resolve the exact experiment_control imported by this daemon."""
        spec = importlib.util.find_spec("experiment_control")
        locations = list(getattr(spec, "submodule_search_locations", []) or [])
        if len(locations) != 1:
            raise ValueError("daemon experiment_control source root is unavailable")
        source = Path(str(locations[0])).resolve(strict=True)
        identity = _source_tree_identity(source)
        commit: str | None = None
        result = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"], check=False,
            text=True, capture_output=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            commit = result.stdout.strip()
        return source, {
            "source_realpath": str(source), "git_commit": commit, **identity,
        }

    def snapshot_execution_bundle(
        self, project: ResearchProject, destination: Path,
        *, reviewed_inputs: dict[str, Path],
    ) -> dict[str, Any]:
        """Copy reviewed controller code and identity inputs into private storage."""
        controller = project.controller
        if controller is None or controller.execution_bundle is None:
            raise ValueError("project has no controller execution_bundle contract")
        bundle = controller.execution_bundle
        base = (project.base_dir or Path(".")).resolve()
        workdir = Path(controller.workdir)
        if not workdir.is_absolute():
            workdir = base / workdir
        workdir = workdir.resolve(strict=True)
        destination = destination.absolute()
        if destination.exists():
            manifest_path = destination / "manifest.json"
            if not manifest_path.is_file():
                raise ValueError(f"incomplete controller snapshot exists: {destination}")
            encoded = manifest_path.read_bytes()
            existing = {
                "root": str(destination),
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256_bytes(encoded),
                "manifest": json.loads(encoded),
            }
            self.verify_execution_bundle(existing)
            return existing
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = destination.parent / f".{destination.name}.{uuid4().hex}.tmp"
        temporary.mkdir(mode=0o700)
        os.chmod(temporary, 0o700)
        python = self._python_path(project)
        python_identity = _file_identity(python)
        try:
            git_identity = (
                self._git_identity(workdir) if bundle.require_clean_git else None
            )
            for relative in bundle.paths:
                source = _safe_source(workdir, relative)
                _copy_source(source, temporary / relative)
            if "experiment_control" in bundle.python_packages:
                raise ValueError(
                    "experiment_control is daemon-trusted and must not be project-declared"
                )
            trusted_core_source, trusted_core = self._trusted_core_source()
            _copy_source(
                trusted_core_source, temporary / "src" / "experiment_control",
            )
            if _source_tree_identity(trusted_core_source) != {
                "files": trusted_core["files"],
                "content_sha256": trusted_core["content_sha256"],
            }:
                raise ValueError("daemon experiment_control changed while snapshotting")
            trusted_runtime: dict[str, dict[str, Any]] = {}
            for package, source in _TRUSTED_RUNTIME_SOURCES.items():
                identity = _source_tree_identity(source)
                _copy_source(source, temporary / "src" / Path(*package.split(".")))
                if _source_tree_identity(source) != identity:
                    raise ValueError(
                        f"daemon trusted runtime package changed while snapshotting: {package}"
                    )
                trusted_runtime[package] = {
                    "source_realpath": str(source), **identity,
                }
            package_sources: dict[str, str] = {}
            for package in bundle.python_packages:
                source = self._package_source(python, package)
                target = temporary / "src" / Path(*package.split("."))
                if source.is_file():
                    target = target.with_suffix(".py")
                _copy_source(source, target)
                package_sources[package] = str(source)
            for relative, source in reviewed_inputs.items():
                relative_path = Path(relative)
                if (
                    relative_path.is_absolute() or ".." in relative_path.parts
                    or not relative_path.parts
                ):
                    raise ValueError(f"reviewed input target must be relative: {relative}")
                target = temporary / relative_path
                if target.exists():
                    raise ValueError(f"duplicate controller snapshot target: {relative}")
                _copy_source(source, target)

            if bundle.require_clean_git and self._git_identity(workdir) != git_identity:
                raise ValueError("controller Git identity changed while snapshotting")
            if _file_identity(python) != python_identity:
                raise ValueError("controller python changed while snapshotting")
            _seal_snapshot_modes(temporary)
            root_stat = temporary.stat()
            manifest = {
                "schema_version": 1,
                "entry_module": bundle.entry_module,
                "python": python_identity,
                "snapshot_root": {
                    "device": root_stat.st_dev, "inode": root_stat.st_ino,
                },
                "original_cwd": git_identity or {
                    "realpath": str(workdir), "device": workdir.stat().st_dev,
                    "inode": workdir.stat().st_ino,
                },
                "git": git_identity,
                "reviewed_inputs": sorted(reviewed_inputs),
                "python_packages": package_sources,
                "trusted_core": trusted_core,
                "trusted_runtime": trusted_runtime,
                "environment": {
                    "isolated_path": True,
                    "private_source": "{controller_snapshot}/src",
                },
                "files": _snapshot_files(temporary),
            }
            encoded = json.dumps(
                manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            ).encode("utf-8")
            manifest_path = temporary / "manifest.json"
            descriptor = os.open(
                manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
            )
            try:
                view = memoryview(encoded)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("short write while sealing controller manifest")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            directory_fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            os.replace(temporary, destination)
            parent_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        manifest_path = destination / "manifest.json"
        return {
            "root": str(destination),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_bytes(encoded),
            "manifest": manifest,
        }

    def verify_execution_bundle(self, snapshot: dict[str, Any]) -> None:
        root = Path(str(snapshot.get("root") or ""))
        manifest_path = Path(str(snapshot.get("manifest_path") or ""))
        if root.is_symlink() or not root.is_dir() or manifest_path.parent != root:
            raise ValueError("controller snapshot root identity is invalid")
        root_stat = root.lstat()
        if stat.S_IMODE(root_stat.st_mode) != 0o700:
            raise ValueError("controller snapshot root mode drifted")
        if manifest_path.is_symlink():
            raise ValueError("controller snapshot manifest is a symlink")
        manifest_identity = _file_identity(manifest_path)
        if stat.S_IMODE(manifest_path.lstat().st_mode) != 0o600:
            raise ValueError("controller snapshot manifest mode drifted")
        if manifest_identity["sha256"] != snapshot.get("manifest_sha256"):
            raise ValueError("controller snapshot manifest digest drifted")
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if loaded != snapshot.get("manifest"):
            raise ValueError("controller snapshot manifest content drifted")
        if loaded.get("snapshot_root") != {
            "device": root_stat.st_dev, "inode": root_stat.st_ino,
        }:
            raise ValueError("controller snapshot root inode drifted")
        if _snapshot_files(root) != loaded.get("files"):
            raise ValueError("controller snapshot Merkle inputs drifted")
        python = Path(str((loaded.get("python") or {}).get("path") or ""))
        if _file_identity(python) != loaded.get("python"):
            raise ValueError("reviewed controller python identity drifted")

    def execute_snapshot(
        self, snapshot: dict[str, Any], arguments: list[str], *, timeout: int,
    ) -> dict[str, Any]:
        """Run only the private source snapshot via reviewed python/cwd descriptors."""
        self.verify_execution_bundle(snapshot)
        manifest = snapshot["manifest"]
        root = Path(snapshot["root"])
        python = Path(manifest["python"]["path"])
        python_fd, python_stat = _open_regular_nofollow(python)
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        try:
            if (
                python_stat.st_dev != manifest["python"]["device"]
                or python_stat.st_ino != manifest["python"]["inode"]
            ):
                raise ValueError("reviewed controller python inode drifted")
            private_source = f"/proc/self/fd/{root_fd}/src"
            bootstrap = (
                "import runpy,sys;"
                "p=sys.argv.pop(1);m=sys.argv.pop(1);"
                "sys.path.insert(0,p);sys.argv[0]=m;"
                "runpy.run_module(m,run_name='__main__')"
            )
            command = [
                f"/proc/self/fd/{python_fd}", "-I", "-B", "-c", bootstrap,
                private_source, str(manifest["entry_module"]), *(
                    str(item).replace(
                        _SNAPSHOT_ROOT_ARGUMENT, f"/proc/self/fd/{root_fd}",
                    )
                    for item in arguments
                ),
            ]
            environment = {
                "PATH": os.environ.get("PATH", ""),
                "HOME": os.environ.get("HOME", ""),
                "LANG": os.environ.get("LANG", "C.UTF-8"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "ML_EXPD_CONTROLLER_SNAPSHOT_SHA256": str(
                    snapshot["manifest_sha256"]
                ),
            }
            cwd = Path(f"/proc/self/fd/{root_fd}")
            if hasattr(self.runner, "run_pinned"):
                result = self.runner.run_pinned(
                    command, cwd=cwd, timeout=timeout, env=environment,
                    pass_fds=(python_fd, root_fd),
                )
            else:
                result = self.runner(command, cwd=cwd, timeout=timeout)
        finally:
            os.close(root_fd)
            os.close(python_fd)
        self.verify_execution_bundle(snapshot)
        result["controller_snapshot_sha256"] = snapshot["manifest_sha256"]
        return result
