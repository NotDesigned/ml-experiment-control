"""Two-phase, daemon-host Project discovery and manifest materialization."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml

from experiment_control.manifest import atomic_create, atomic_write

from .application_errors import ApplicationError
from .project_config import load_research_project
from .project_service import ProjectApplicationService
from .runtime import ExperimentServerRuntime


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_identity(root: Path, *, ignore: Path | None = None) -> dict[str, Any]:
    """Hash non-Git repository content without following symlinks."""
    ignored = ignore.relative_to(root) if ignore is not None else None
    digest = hashlib.sha256()
    files = 0
    stack = [root]
    while stack:
        directory = stack.pop()
        directories: list[Path] = []
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            relative = path.relative_to(root)
            if relative == ignored:
                continue
            mode = path.lstat().st_mode
            encoded = relative.as_posix().encode("utf-8")
            if stat.S_ISLNK(mode):
                digest.update(b"L\0" + encoded + b"\0" + os.readlink(path).encode("utf-8"))
                files += 1
            elif stat.S_ISDIR(mode):
                directories.append(path)
            elif stat.S_ISREG(mode):
                digest.update(b"F\0" + encoded + b"\0")
                with path.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        digest.update(len(chunk).to_bytes(8, "big") + chunk)
                files += 1
            else:
                digest.update(b"S\0" + encoded + b"\0" + str(mode).encode())
                files += 1
        stack.extend(reversed(directories))
    return {
        "kind": "directory", "commit": None, "dirty": None,
        "status_digest": "sha256:" + digest.hexdigest(), "files": files,
    }


def _repository_identity(root: Path, *, ignore: Path | None = None) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        arguments = ["git", "-C", str(root), "status", "--porcelain=v1", "-z"]
        if ignore is not None:
            relative = ignore.relative_to(root).as_posix()
            arguments.extend(["--", ".", f":(exclude){relative}"])
        status = subprocess.run(
            arguments,
            check=True, capture_output=True, timeout=10,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return _directory_identity(root, ignore=ignore)
    return {
        "kind": "git", "commit": commit, "dirty": bool(status),
        "status_digest": "sha256:" + hashlib.sha256(status).hexdigest(),
    }


def _project_id(root: Path) -> str:
    candidate = re.sub(r"[^A-Za-z0-9_.-]+", "-", root.name).strip("-._")
    if not candidate:
        raise ApplicationError(
            "repository directory cannot be converted to a safe Project ID",
            code="PROJECT_IMPORT_BLOCKED",
        )
    return candidate[:128]


def _safe_project_path(
    root: Path, value: str | Path, *, label: str, base: Path | None = None,
) -> Path:
    """Resolve one manifest reference without crossing the imported repository."""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (base or root) / candidate
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ApplicationError(
            f"{label} must stay inside the imported repository",
            code="PROJECT_IMPORT_BLOCKED",
        ) from exc
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ApplicationError(
                f"{label} may not traverse a symlink",
                code="PROJECT_IMPORT_BLOCKED",
            )
    resolved = candidate.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise ApplicationError(
            f"{label} must stay inside the imported repository",
            code="PROJECT_IMPORT_BLOCKED",
        )
    return resolved


def _validate_manifest_paths(root: Path, manifest: Any) -> None:
    if not isinstance(manifest, dict):
        return
    for index, value in enumerate(manifest.get("run_roots") or []):
        if isinstance(value, str):
            _safe_project_path(root, value, label=f"run_roots[{index}]")
    questions = manifest.get("research_questions_dir")
    if isinstance(questions, str):
        _safe_project_path(root, questions, label="research_questions_dir")
    for index, campaign in enumerate(manifest.get("campaigns") or []):
        if isinstance(campaign, dict) and isinstance(campaign.get("file"), str):
            _safe_project_path(
                root, campaign["file"], label=f"campaigns[{index}].file",
            )
    controller = manifest.get("controller")
    if isinstance(controller, dict):
        workdir_value = controller.get("workdir", ".")
        if isinstance(workdir_value, str):
            workdir = _safe_project_path(
                root, workdir_value, label="controller.workdir",
            )
            tool = controller.get("experimentctl")
            if isinstance(tool, str):
                _safe_project_path(
                    root, tool, label="controller.experimentctl", base=workdir,
                )


def _resolve_repository_root(
    runtime: ExperimentServerRuntime, value: Path, *, require_canonical: bool,
) -> Path:
    """Resolve an allowed daemon path, optionally rejecting any path substitution."""
    root = value.expanduser()
    if not root.is_absolute():
        raise ApplicationError(
            "repository_root must be an absolute daemon-host path",
            code="PROJECT_IMPORT_BLOCKED",
        )
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        code = "PROJECT_IMPORT_STALE" if require_canonical else "PROJECT_IMPORT_BLOCKED"
        raise ApplicationError(str(exc), code=code) from exc
    if require_canonical and resolved != root:
        raise ApplicationError(
            "repository path changed after import preview; create a fresh preview",
            code="PROJECT_IMPORT_STALE",
        )
    if not resolved.is_dir() or resolved == Path(resolved.anchor):
        raise ApplicationError(
            "repository_root must be a non-root directory",
            code="PROJECT_IMPORT_BLOCKED",
        )
    allowed = runtime.config.project_import_root_paths()
    if not allowed or not any(
        resolved == item or item in resolved.parents for item in allowed
    ):
        code = "PROJECT_IMPORT_STALE" if require_canonical else "PROJECT_IMPORT_BLOCKED"
        raise ApplicationError(
            "repository_root is outside configured project_import_roots",
            code=code,
        )
    return resolved


def _filesystem_identity(path: Path) -> dict[str, int]:
    value = path.stat(follow_symlinks=False)
    return {"device": value.st_dev, "inode": value.st_ino}


def _descriptor_matches_path(descriptor: int, path: Path) -> bool:
    try:
        observed = os.fstat(descriptor)
        return _filesystem_identity(path) == {
            "device": observed.st_dev, "inode": observed.st_ino,
        }
    except OSError:
        return False


@contextmanager
def _opened_repository(root: Path, expected: Any) -> Iterator[int]:
    try:
        descriptor = os.open(
            root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError as exc:
        raise ApplicationError(
            "repository path changed after import preview; create a fresh preview",
            code="PROJECT_IMPORT_STALE",
        ) from exc
    try:
        observed = os.fstat(descriptor)
        identity = {"device": observed.st_dev, "inode": observed.st_ino}
        if identity != expected:
            raise ApplicationError(
                "repository directory changed after import preview; create a fresh preview",
                code="PROJECT_IMPORT_STALE",
            )
        yield descriptor
    finally:
        os.close(descriptor)


def _open_experiments(root_fd: int, *, create: bool) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        return os.open("experiments", flags, dir_fd=root_fd)
    except FileNotFoundError:
        if not create:
            raise
        os.mkdir("experiments", 0o755, dir_fd=root_fd)
        return os.open("experiments", flags, dir_fd=root_fd)


def _read_manifest_at(root_fd: int) -> tuple[bool, Any]:
    try:
        experiments_fd = _open_experiments(root_fd, create=False)
    except FileNotFoundError:
        return False, None
    try:
        try:
            descriptor = os.open(
                "research_project.yaml", os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=experiments_fd,
            )
        except FileNotFoundError:
            return False, None
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            return True, yaml.safe_load(handle)
    finally:
        os.close(experiments_fd)


def _digest_manifest_at(root_fd: int) -> str | None:
    try:
        experiments_fd = _open_experiments(root_fd, create=False)
    except FileNotFoundError:
        return None
    try:
        try:
            descriptor = os.open(
                "research_project.yaml", os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=experiments_fd,
            )
        except FileNotFoundError:
            return None
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    finally:
        os.close(experiments_fd)


def _manifest_temp_name(import_id: str) -> str:
    if not re.fullmatch(r"import-[0-9a-f]{24}", import_id):
        raise ValueError("invalid import identity for manifest transaction")
    return f".research_project.yaml.{import_id}.tmp"


def _cleanup_manifest_temp(root_fd: int, import_id: str) -> None:
    temporary = _manifest_temp_name(import_id)
    try:
        experiments_fd = _open_experiments(root_fd, create=False)
    except FileNotFoundError:
        return
    try:
        try:
            os.unlink(temporary, dir_fd=experiments_fd)
        except FileNotFoundError:
            pass
    finally:
        os.close(experiments_fd)


def _atomic_write_manifest(root_fd: int, payload: Any, import_id: str) -> None:
    experiments_fd = _open_experiments(root_fd, create=True)
    temporary = _manifest_temp_name(import_id)
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=experiments_fd,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                payload, handle, sort_keys=False, allow_unicode=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temporary, "research_project.yaml",
            src_dir_fd=experiments_fd, dst_dir_fd=experiments_fd,
        )
        os.fsync(experiments_fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=experiments_fd)
        except FileNotFoundError:
            pass
        os.close(experiments_fd)


class ProjectImportService:
    """Persist reviewable import plans before any repository write."""

    def __init__(
        self, runtime: ExperimentServerRuntime, projects: ProjectApplicationService,
    ) -> None:
        self.runtime = runtime
        self.projects = projects
        self.root = runtime.config.project_registry_root_path() / "imports"

    def preview(
        self, repository_root: Path, *, project: str | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:
        root = _resolve_repository_root(
            self.runtime, repository_root, require_canonical=False,
        )
        project_id = (project or _project_id(root)).strip()
        if not _SAFE_ID.fullmatch(project_id):
            raise ApplicationError(
                "project must use 1-128 letters, digits, '.', '_' or '-'",
                code="PROJECT_IMPORT_BLOCKED",
            )
        manifest_path = root / "experiments" / "research_project.yaml"
        _safe_project_path(root, manifest_path, label="manifest_path")
        identity = _repository_identity(root)
        identity_without_manifest = _repository_identity(root, ignore=manifest_path)
        existing = manifest_path.is_file()
        warnings: list[str] = []
        detected: dict[str, Any] = {}
        if existing:
            try:
                manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
                raise ApplicationError(
                    "existing Project manifest is unreadable",
                    code="PROJECT_IMPORT_BLOCKED",
                ) from exc
            _validate_manifest_paths(root, manifest)
            loaded = load_research_project(manifest_path)
            if project is not None and loaded.project != project_id:
                raise ApplicationError(
                    "existing manifest Project ID conflicts with requested project",
                    code="PROJECT_IMPORT_BLOCKED",
                )
            operation = "REGISTER_EXISTING"
            project_id = loaded.project
        else:
            run_roots = [
                path.relative_to(root).as_posix()
                for path in (root / "runs", root / "outputs", root / "experiments" / "runs")
                if path.is_dir() and not path.is_symlink()
            ]
            if not run_roots:
                warnings.append(
                    "no existing runs/outputs directory was detected; run_roots is empty"
                )
            manifest = {
                "schema_version": 1,
                "project": project_id,
                "title": (title or root.name.replace("-", " ").replace("_", " ").title()),
                "run_roots": run_roots,
            }
            controller = next((
                candidate for candidate in (
                    root / "tools" / "experimentctl.py",
                    root / "scripts" / "experimentctl.py",
                    root / "experimentctl.py",
                ) if candidate.is_file() and not candidate.is_symlink()
            ), None)
            if controller is not None:
                manifest["controller"] = {
                    "python": "python3",
                    "experimentctl": controller.relative_to(root).as_posix(),
                    "workdir": ".",
                    "capabilities": {},
                }
                detected["controller"] = controller.relative_to(root).as_posix()
                warnings.append(
                    "controller entrypoint was detected but its capabilities require review"
                )
            questions = root / "experiments" / "research_questions"
            if questions.is_dir() and not questions.is_symlink():
                manifest["research_questions_dir"] = "experiments/research_questions"
            campaigns: list[dict[str, str]] = []
            campaign_root = root / "experiments" / "campaigns"
            if campaign_root.is_dir() and not campaign_root.is_symlink():
                for path in sorted((*campaign_root.glob("*.yml"), *campaign_root.glob("*.yaml"))):
                    if path.is_symlink() or not path.is_file():
                        continue
                    try:
                        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
                    except (OSError, UnicodeDecodeError, yaml.YAMLError):
                        continue
                    campaign = str(payload.get("campaign") or "") if isinstance(payload, dict) else ""
                    if _SAFE_ID.fullmatch(campaign):
                        campaigns.append({
                            "name": campaign,
                            "file": path.relative_to(root).as_posix(),
                        })
            if campaigns:
                manifest["campaigns"] = campaigns
                detected["campaigns"] = len(campaigns)
            operation = "GENERATE_AND_REGISTER"
        _validate_manifest_paths(root, manifest)
        canonical = {
            "source": {"kind": "daemon_path", "repository_root": str(root)},
            "manifest_path": str(manifest_path),
            "manifest": manifest,
            "expected_manifest_digest": _digest(manifest_path),
            "repository_identity": identity,
            "repository_identity_without_manifest": identity_without_manifest,
            "repository_filesystem_identity": _filesystem_identity(root),
            "operation": operation,
        }
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        import_id = "import-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]
        plan = {
            "schema_version": 1,
            "import_id": import_id,
            **canonical,
            "detected": detected,
            "warnings": warnings,
            "confirmation": f"IMPORT {import_id}",
            "executed": False,
            "phase": "PREPARED",
        }
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            atomic_create(self.root / f"{import_id}.json", plan)
        except FileExistsError:
            with self._locked_plan(import_id) as (_path, existing_plan):
                if any(existing_plan.get(key) != value for key, value in canonical.items()):
                    raise ApplicationError(
                        "Project import plan identity collision",
                        code="PROJECT_IMPORT_BLOCKED",
                    )
                return existing_plan
        return plan

    @contextmanager
    def _locked_plan(self, import_id: str) -> Iterator[tuple[Path, dict[str, Any]]]:
        if not re.fullmatch(r"import-[0-9a-f]{24}", import_id):
            raise ApplicationError("invalid import identity", code="PROJECT_IMPORT_BLOCKED")
        path = self.root / f"{import_id}.json"
        lock_path = self.root / f".{import_id}.lock"
        self.root.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                try:
                    if not path.is_file():
                        raise ApplicationError(
                            "Project import plan not found", status_code=404,
                            code="UNKNOWN_PROJECT_IMPORT",
                        )
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("plan must be a mapping")
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    raise ApplicationError(
                        "Project import plan is unreadable", code="PROJECT_IMPORT_BLOCKED",
                    ) from exc
                yield path, payload
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def execute(self, import_id: str, confirmation: str) -> dict[str, Any]:
        with self._locked_plan(import_id) as (path, plan):
            if confirmation != plan.get("confirmation"):
                raise ApplicationError(
                    f"confirmation must equal IMPORT {import_id}",
                    code="PROJECT_IMPORT_BLOCKED",
                )
            if (
                plan.get("executed")
                and plan.get("phase") == "COMPLETED"
                and isinstance(plan.get("registration"), dict)
            ):
                return {"import": plan, "registration": plan["registration"]}
            try:
                planned_root = Path(str(plan["source"]["repository_root"]))
                root = _resolve_repository_root(
                    self.runtime, planned_root, require_canonical=True,
                )
                manifest_path = Path(str(plan["manifest_path"]))
                expected_manifest = root / "experiments" / "research_project.yaml"
                if manifest_path != expected_manifest:
                    raise ApplicationError(
                        "planned manifest path is not canonical",
                        code="PROJECT_IMPORT_BLOCKED",
                    )
                _safe_project_path(root, manifest_path, label="manifest_path")
                _validate_manifest_paths(root, plan.get("manifest"))
            except (KeyError, TypeError) as exc:
                raise ApplicationError(
                    "Project import plan has invalid paths",
                    code="PROJECT_IMPORT_BLOCKED",
                ) from exc
            identity_key = (
                "repository_identity_without_manifest"
                if plan.get("operation") == "GENERATE_AND_REGISTER"
                else "repository_identity"
            )
            with _opened_repository(
                root, plan.get("repository_filesystem_identity"),
            ) as root_fd:
                operation = plan.get("operation")
                if operation == "GENERATE_AND_REGISTER":
                    if not self.runtime.config.action_runtime.allow_project_writes:
                        raise ApplicationError(
                            "Project manifest generation is disabled by daemon policy",
                            code="PROJECT_IMPORT_BLOCKED",
                        )
                    _cleanup_manifest_temp(root_fd, import_id)
                current_identity = _repository_identity(
                    root,
                    ignore=(
                        manifest_path
                        if identity_key.endswith("without_manifest") else None
                    ),
                )
                if current_identity != plan.get(identity_key):
                    raise ApplicationError(
                        "repository changed after import preview; create a fresh preview",
                        code="PROJECT_IMPORT_STALE",
                    )
                if operation == "GENERATE_AND_REGISTER":
                    try:
                        exists, current = _read_manifest_at(root_fd)
                    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
                        raise ApplicationError(
                            "Project manifest changed after import preview; create a fresh preview",
                            code="PROJECT_IMPORT_STALE",
                        ) from exc
                    if not exists:
                        try:
                            _atomic_write_manifest(root_fd, plan["manifest"], import_id)
                        except OSError as exc:
                            raise ApplicationError(
                                f"Project manifest could not be materialized: {exc}",
                                code="PROJECT_IMPORT_BLOCKED",
                            ) from exc
                    elif current != plan.get("manifest"):
                        raise ApplicationError(
                            "Project manifest changed after import preview; create a fresh preview",
                            code="PROJECT_IMPORT_STALE",
                        )
                    plan["phase"] = "MANIFEST_APPLIED"
                    plan["applied_manifest_digest"] = _digest_manifest_at(root_fd)
                    atomic_write(path, plan)
                elif _digest_manifest_at(root_fd) != plan.get("expected_manifest_digest"):
                    raise ApplicationError(
                        "Project manifest changed after import preview; create a fresh preview",
                        code="PROJECT_IMPORT_STALE",
                    )
                if not _descriptor_matches_path(root_fd, root):
                    raise ApplicationError(
                        "repository path changed during import; create a fresh preview",
                        code="PROJECT_IMPORT_STALE",
                    )
                registration = self.projects.register(manifest_path)
            plan["phase"] = "REGISTERED"
            plan["registration"] = registration
            atomic_write(path, plan)
            plan["executed"] = True
            plan["phase"] = "COMPLETED"
            atomic_write(path, plan)
            return {"import": plan, "registration": registration}
