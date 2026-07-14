"""Two-phase import of client proposals into daemon-owned immutable source trees."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Iterator
from uuid import uuid4

from experiment_control.manifest import atomic_write

from .application_errors import ApplicationError
from .schemas import ServerConfig

if TYPE_CHECKING:
    from .runtime import ExperimentServerRuntime


_COMMIT = re.compile(r"^[0-9a-fA-F]{40,64}$")
_IMPORT_ID = re.compile(r"^source-import-[0-9a-f]{24}$")
_SOURCE_ID = re.compile(r"^source\.[0-9a-f]{64}$")
_PROJECT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_MAX_PATCH_BYTES = 2_000_000
_PROTECTED = {".git", ".ssh", ".aws", ".gnupg", "credentials", "keys", "secrets"}
_PROTECTED_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}


def resolve_source_tree(config: ServerConfig, project: str, source_id: str) -> Path:
    """Resolve one validated content-addressed tree without trusting a request path."""
    if not _PROJECT_ID.fullmatch(project) or not _SOURCE_ID.fullmatch(source_id):
        raise ValueError("invalid source revision identity")
    root = config.project_registry_root_path() / "source-revisions" / "sources"
    metadata_path = root / project / source_id / "source.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        payload.get("project") != project
        or payload.get("source_id") != source_id
        or payload.get("tree") != "tree"
    ):
        raise ValueError("source revision metadata identity mismatch")
    tree = metadata_path.parent / "tree"
    if not tree.is_dir():
        raise ValueError("source revision tree is unavailable")
    return tree


def _is_protected(value: str) -> bool:
    for part in PurePosixPath(value).parts:
        lowered = part.lower()
        if (
            lowered in _PROTECTED
            or lowered == ".env"
            or lowered.startswith(".env.")
            or lowered.startswith("credentials.")
            or lowered.startswith("secrets.")
            or Path(lowered).suffix in _PROTECTED_SUFFIXES
        ):
            return True
    return False


def _git(cwd: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "GIT_CONFIG_NOSYSTEM": "1",
        "HOME": "/nonexistent",
        "GIT_CONFIG_GLOBAL": "/dev/null",
    }
    return subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(cwd), *args],
        input=input_bytes, check=True, capture_output=True, timeout=60,
        env=environment,
    ).stdout


class SourceRevisionService:
    """Validate proposals without executing project code, then snapshot them."""

    def __init__(self, runtime: ExperimentServerRuntime) -> None:
        self.runtime = runtime
        self.root = runtime.config.project_registry_root_path() / "source-revisions"
        self.plans = self.root / "plans"
        self.sources = self.root / "sources"

    def _repository(self, project: str) -> Path:
        try:
            configured = self.runtime.project(project)
        except KeyError as exc:
            raise ApplicationError(
                f"unknown project: {project}", status_code=404, code="UNKNOWN_PROJECT",
            ) from exc
        root = Path(configured.base_dir or ".").resolve()
        try:
            repository = Path(
                _git(root, "rev-parse", "--show-toplevel").decode().strip()
            ).resolve()
        except (OSError, subprocess.SubprocessError, UnicodeDecodeError) as exc:
            raise ApplicationError(
                "source import requires a Git-backed Project root",
                code="SOURCE_IMPORT_BLOCKED",
            ) from exc
        if repository != root:
            raise ApplicationError(
                "Project manifest must be rooted at the Git repository root",
                code="SOURCE_IMPORT_BLOCKED",
            )
        return repository

    @staticmethod
    def _proposal(proposal: dict[str, Any]) -> tuple[str, bytes, str, list[str]]:
        base = str(proposal.get("base_commit") or "")
        patch = proposal.get("patch")
        digest = str(proposal.get("patch_digest") or "")
        declared = proposal.get("changed_files")
        if not _COMMIT.fullmatch(base):
            raise ApplicationError("invalid base commit", code="SOURCE_IMPORT_BLOCKED")
        if not isinstance(patch, str):
            raise ApplicationError("proposal patch must be text", code="SOURCE_IMPORT_BLOCKED")
        encoded = patch.encode("utf-8")
        if not encoded or len(encoded) > _MAX_PATCH_BYTES:
            raise ApplicationError(
                f"proposal patch must contain 1-{_MAX_PATCH_BYTES} UTF-8 bytes",
                code="SOURCE_IMPORT_BLOCKED",
            )
        expected = "sha256:" + hashlib.sha256(encoded).hexdigest()
        if digest != expected:
            raise ApplicationError("patch digest mismatch", code="SOURCE_IMPORT_BLOCKED")
        if not isinstance(declared, list) or any(not isinstance(item, str) for item in declared):
            raise ApplicationError(
                "changed_files must be a list of paths", code="SOURCE_IMPORT_BLOCKED",
            )
        return base.lower(), encoded, expected, sorted(set(declared))

    @staticmethod
    def _materialize(repository: Path, base: str, patch: bytes, target: Path) -> list[str]:
        target.parent.mkdir(parents=True, exist_ok=True)
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "C.UTF-8", "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null", "HOME": "/nonexistent",
        }
        subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "clone", "--quiet",
             "--no-local", "--no-checkout", str(repository), str(target)],
            check=True, capture_output=True, timeout=120, env=environment,
        )
        _git(target, "checkout", "--quiet", "--detach", base)
        _git(target, "apply", "--whitespace=nowarn", "-", input_bytes=patch)
        _git(target, "add", "-N", "--", ".")
        _git(target, "diff", "--check", "--")
        raw = _git(target, "diff", "--name-only", "-z", "--")
        changed = sorted(item.decode("utf-8") for item in raw.split(b"\0") if item)
        if not changed:
            raise ApplicationError("proposal produces no source change", code="SOURCE_IMPORT_BLOCKED")
        protected = [item for item in changed if _is_protected(item)]
        if protected:
            raise ApplicationError(
                "proposal targets protected paths: " + ", ".join(protected),
                code="SOURCE_IMPORT_BLOCKED",
            )
        raw_diff = _git(target, "diff", "--raw", "-z", "--")
        if b" 120000" in raw_diff or b" 160000" in raw_diff:
            raise ApplicationError(
                "proposal may not add symlinks or Git links", code="SOURCE_IMPORT_BLOCKED",
            )
        for item in changed:
            path = target / item
            if path.is_symlink():
                raise ApplicationError(
                    "proposal may not modify symlinks", code="SOURCE_IMPORT_BLOCKED",
                )
        return changed

    def preview(self, project: str, proposal: dict[str, Any]) -> dict[str, Any]:
        repository = self._repository(project)
        base, patch, patch_digest, declared = self._proposal(proposal)
        binding = proposal.get("binding")
        if isinstance(binding, dict) and binding.get("project") not in {None, project}:
            raise ApplicationError(
                "proposal is bound to a different Project",
                code="SOURCE_IMPORT_BLOCKED",
            )
        try:
            _git(repository, "cat-file", "-e", f"{base}^{{commit}}")
        except (OSError, subprocess.SubprocessError) as exc:
            raise ApplicationError(
                "base commit is not available in the daemon Project repository",
                code="SOURCE_IMPORT_BLOCKED",
            ) from exc
        temporary = self.root / f".preview-{uuid4().hex}"
        try:
            changed = self._materialize(repository, base, patch, temporary)
        except ApplicationError:
            raise
        except (OSError, subprocess.SubprocessError, UnicodeDecodeError) as exc:
            raise ApplicationError(
                "proposal patch does not apply cleanly to its exact base commit",
                code="SOURCE_IMPORT_BLOCKED",
            ) from exc
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        if changed != declared:
            raise ApplicationError(
                "declared changed_files do not match the patch",
                code="SOURCE_IMPORT_BLOCKED",
            )
        source_hash = hashlib.sha256(
            project.encode() + b"\0" + base.encode() + b"\0" + patch_digest.encode()
        ).hexdigest()
        source_id = f"source.{source_hash}"
        import_id = f"source-import-{source_hash[:24]}"
        plan = {
            "schema_version": 1,
            "import_id": import_id,
            "project": project,
            "base_commit": base,
            "patch": patch.decode("utf-8"),
            "patch_digest": patch_digest,
            "changed_files": changed,
            "proposal_provenance": {
                "proposal_id": proposal.get("proposal_id"),
                "binding": binding if isinstance(binding, dict) else {},
                "repository_identity": (
                    proposal.get("repository_identity")
                    if isinstance(proposal.get("repository_identity"), dict) else {}
                ),
                "checks": proposal.get("checks") if isinstance(proposal.get("checks"), list) else [],
            },
            "source_id": source_id,
            "confirmation": f"IMPORT SOURCE {source_id}",
            "executed": False,
        }
        self.plans.mkdir(parents=True, exist_ok=True)
        atomic_write(self.plans / f"{import_id}.json", plan)
        return plan

    @contextmanager
    def _locked_plan(self, import_id: str) -> Iterator[tuple[Path, dict[str, Any]]]:
        if not _IMPORT_ID.fullmatch(import_id):
            raise ApplicationError("invalid source import identity", code="SOURCE_IMPORT_BLOCKED")
        import fcntl
        path = self.plans / f"{import_id}.json"
        lock_path = self.plans / f".{import_id}.lock"
        self.plans.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if not path.is_file():
                    raise ApplicationError(
                        "source import plan not found", status_code=404,
                        code="UNKNOWN_SOURCE_IMPORT",
                    )
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("plan must be a mapping")
                yield path, payload
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                if isinstance(exc, ApplicationError):
                    raise
                raise ApplicationError(
                    "source import plan is unreadable", code="SOURCE_IMPORT_BLOCKED",
                ) from exc
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def execute(self, import_id: str, confirmation: str) -> dict[str, Any]:
        with self._locked_plan(import_id) as (plan_path, plan):
            source_id = str(plan.get("source_id") or "")
            if not _SOURCE_ID.fullmatch(source_id):
                raise ApplicationError("invalid planned source identity", code="SOURCE_IMPORT_BLOCKED")
            if confirmation != plan.get("confirmation"):
                raise ApplicationError(
                    f"confirmation must equal IMPORT SOURCE {source_id}",
                    code="SOURCE_IMPORT_BLOCKED",
                )
            if not self.runtime.config.action_runtime.allow_source_imports:
                raise ApplicationError(
                    "source imports are disabled by daemon policy",
                    code="SOURCE_IMPORT_BLOCKED",
                )
            repository = self._repository(str(plan.get("project") or ""))
            base, patch, patch_digest, declared = self._proposal(plan)
            if patch_digest != plan.get("patch_digest"):
                raise ApplicationError("source import plan changed", code="SOURCE_IMPORT_BLOCKED")
            final = self.sources / str(plan["project"]) / source_id
            metadata_path = final / "source.json"
            if final.is_dir() and metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("patch_digest") != patch_digest:
                    raise ApplicationError(
                        "source identity collision", code="SOURCE_IMPORT_BLOCKED",
                    )
            else:
                temporary = final.parent / f".{source_id}.{uuid4().hex}.tmp"
                try:
                    changed = self._materialize(repository, base, patch, temporary / "tree")
                    if changed != declared:
                        raise ApplicationError(
                            "source import changed after preview", code="SOURCE_IMPORT_BLOCKED",
                        )
                    shutil.rmtree(temporary / "tree" / ".git")
                    metadata = {
                        "schema_version": 1, "source_id": source_id,
                        "project": plan["project"], "base_commit": base,
                        "patch_digest": patch_digest, "changed_files": changed,
                        "proposal_provenance": plan.get("proposal_provenance", {}),
                        "tree": "tree",
                    }
                    atomic_write(temporary / "source.json", metadata)
                    for item in sorted((temporary / "tree").rglob("*"), reverse=True):
                        if item.is_symlink():
                            continue
                        mode = item.stat().st_mode & 0o777
                        item.chmod((mode & 0o555) if item.is_file() else 0o555)
                    (temporary / "tree").chmod(0o555)
                    (temporary / "source.json").chmod(0o444)
                    final.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(temporary, final)
                except ApplicationError:
                    shutil.rmtree(temporary, ignore_errors=True)
                    raise
                except (OSError, subprocess.SubprocessError, UnicodeDecodeError) as exc:
                    shutil.rmtree(temporary, ignore_errors=True)
                    raise ApplicationError(
                        "failed to materialize immutable source revision",
                        code="SOURCE_IMPORT_BLOCKED",
                    ) from exc
            plan["executed"] = True
            plan["source"] = {
                "source_id": source_id, "tree_path": str(final / "tree"),
                "metadata_path": str(metadata_path),
            }
            atomic_write(plan_path, plan)
            return {"import": plan, "source": plan["source"]}

    def get(self, project: str, source_id: str) -> dict[str, Any]:
        if not _PROJECT_ID.fullmatch(project) or not _SOURCE_ID.fullmatch(source_id):
            raise ApplicationError("invalid source identity", code="UNKNOWN_SOURCE_REVISION")
        path = self.sources / project / source_id / "source.json"
        if not path.is_file():
            raise ApplicationError(
                "source revision not found", status_code=404,
                code="UNKNOWN_SOURCE_REVISION",
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        try:
            tree = resolve_source_tree(self.runtime.config, project, source_id)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ApplicationError(
                "source revision is unreadable", code="UNKNOWN_SOURCE_REVISION",
            ) from exc
        return {**payload, "tree_path": str(tree)}
