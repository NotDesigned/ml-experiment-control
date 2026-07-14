"""Two-phase, daemon-host Project discovery and manifest materialization."""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml

from experiment_control.manifest import atomic_write

from .application_errors import ApplicationError
from .project_config import load_research_project
from .project_service import ProjectApplicationService
from .runtime import ExperimentServerRuntime


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


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
        return {"kind": "directory", "commit": None, "dirty": None,
                "status_digest": None}
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
        root = repository_root.expanduser()
        if not root.is_absolute():
            raise ApplicationError(
                "repository_root must be an absolute daemon-host path",
                code="PROJECT_IMPORT_BLOCKED",
            )
        try:
            root = root.resolve(strict=True)
        except OSError as exc:
            raise ApplicationError(str(exc), code="PROJECT_IMPORT_BLOCKED") from exc
        if not root.is_dir() or root == Path(root.anchor):
            raise ApplicationError(
                "repository_root must be a non-root directory",
                code="PROJECT_IMPORT_BLOCKED",
            )
        allowed = self.runtime.config.project_import_root_paths()
        if not allowed or not any(root == item or item in root.parents for item in allowed):
            raise ApplicationError(
                "repository_root is outside configured project_import_roots",
                code="PROJECT_IMPORT_BLOCKED",
            )
        project_id = (project or _project_id(root)).strip()
        if not _SAFE_ID.fullmatch(project_id):
            raise ApplicationError(
                "project must use 1-128 letters, digits, '.', '_' or '-'",
                code="PROJECT_IMPORT_BLOCKED",
            )
        manifest_path = root / "experiments" / "research_project.yaml"
        identity = _repository_identity(root)
        identity_without_manifest = _repository_identity(root, ignore=manifest_path)
        existing = manifest_path.is_file()
        warnings: list[str] = []
        detected: dict[str, Any] = {}
        if existing:
            loaded = load_research_project(manifest_path)
            if project is not None and loaded.project != project_id:
                raise ApplicationError(
                    "existing manifest Project ID conflicts with requested project",
                    code="PROJECT_IMPORT_BLOCKED",
                )
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
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
            if questions.is_dir():
                manifest["research_questions_dir"] = "experiments/research_questions"
            campaigns: list[dict[str, str]] = []
            campaign_root = root / "experiments" / "campaigns"
            if campaign_root.is_dir():
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
        canonical = {
            "source": {"kind": "daemon_path", "repository_root": str(root)},
            "manifest_path": str(manifest_path),
            "manifest": manifest,
            "expected_manifest_digest": _digest(manifest_path),
            "repository_identity": identity,
            "repository_identity_without_manifest": identity_without_manifest,
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
        atomic_write(self.root / f"{import_id}.json", plan)
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
                if not path.is_file():
                    raise ApplicationError(
                        "Project import plan not found", status_code=404,
                        code="UNKNOWN_PROJECT_IMPORT",
                    )
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("plan must be a mapping")
                yield path, payload
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                if isinstance(exc, ApplicationError):
                    raise
                raise ApplicationError(
                    "Project import plan is unreadable", code="PROJECT_IMPORT_BLOCKED",
                ) from exc
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def execute(self, import_id: str, confirmation: str) -> dict[str, Any]:
        with self._locked_plan(import_id) as (path, plan):
            if confirmation != plan.get("confirmation"):
                raise ApplicationError(
                    f"confirmation must equal IMPORT {import_id}",
                    code="PROJECT_IMPORT_BLOCKED",
                )
            manifest_path = Path(str(plan["manifest_path"]))
            root = Path(str(plan["source"]["repository_root"]))
            if plan.get("executed") and isinstance(plan.get("registration"), dict):
                return {"import": plan, "registration": plan["registration"]}
            identity_key = (
                "repository_identity_without_manifest"
                if plan.get("operation") == "GENERATE_AND_REGISTER"
                else "repository_identity"
            )
            current_identity = _repository_identity(
                root, ignore=manifest_path if identity_key.endswith("without_manifest") else None,
            )
            if current_identity != plan.get(identity_key):
                raise ApplicationError(
                    "repository changed after import preview; create a fresh preview",
                    code="PROJECT_IMPORT_STALE",
                )
            if plan.get("operation") == "GENERATE_AND_REGISTER":
                if not self.runtime.config.action_runtime.allow_project_writes:
                    raise ApplicationError(
                        "Project manifest generation is disabled by daemon policy",
                        code="PROJECT_IMPORT_BLOCKED",
                    )
                current: Any = None
                if manifest_path.is_file():
                    try:
                        current = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
                    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
                        raise ApplicationError(
                            "Project manifest changed after import preview; create a fresh preview",
                            code="PROJECT_IMPORT_STALE",
                        ) from exc
                if current is None:
                    manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_write(manifest_path, plan["manifest"], yaml_format=True)
                elif current != plan.get("manifest"):
                    raise ApplicationError(
                        "Project manifest changed after import preview; create a fresh preview",
                        code="PROJECT_IMPORT_STALE",
                    )
                plan["phase"] = "MANIFEST_APPLIED"
                plan["applied_manifest_digest"] = _digest(manifest_path)
                atomic_write(path, plan)
            elif _digest(manifest_path) != plan.get("expected_manifest_digest"):
                raise ApplicationError(
                    "Project manifest changed after import preview; create a fresh preview",
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
