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

from experiment_control.manifest import atomic_create, atomic_write

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


def _tree_digest(root: Path, *, require_read_only: bool = False) -> str:
    """Hash a source tree without following links or trusting filesystem modes."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("source revision tree is unavailable")
    if require_read_only and root.stat().st_mode & 0o222:
        raise ValueError("source revision tree is writable")
    digest = hashlib.sha256()

    def field(value: bytes) -> None:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)

    stack = [root]
    while stack:
        directory = stack.pop()
        children = sorted(directory.iterdir(), key=lambda item: item.name)
        directories: list[Path] = []
        for path in children:
            relative = path.relative_to(root).as_posix().encode("utf-8")
            if path.is_symlink():
                target = os.readlink(path)
                if Path(target).is_absolute():
                    raise ValueError("source revision tree contains an escaping symlink")
                try:
                    resolved = path.resolve(strict=False)
                except (OSError, RuntimeError) as exc:
                    raise ValueError("source revision tree contains an invalid symlink") from exc
                if resolved != root and root not in resolved.parents:
                    raise ValueError("source revision tree contains an escaping symlink")
                field(b"L")
                field(relative)
                field(target.encode("utf-8"))
            elif path.is_dir():
                if require_read_only and path.stat().st_mode & 0o222:
                    raise ValueError("source revision tree is writable")
                field(b"D")
                field(relative)
                directories.append(path)
            elif path.is_file():
                mode = path.stat().st_mode
                if require_read_only and mode & 0o222:
                    raise ValueError("source revision tree is writable")
                field(b"F")
                field(relative)
                field(b"X" if mode & 0o111 else b"-")
                with path.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        field(chunk)
            else:
                raise ValueError("source revision tree contains a special file")
        stack.extend(reversed(directories))
    return "sha256:" + digest.hexdigest()


def resolve_source_tree(config: ServerConfig, project: str, source_id: str) -> Path:
    """Resolve one validated content-addressed tree without trusting a request path."""
    if not _PROJECT_ID.fullmatch(project) or not _SOURCE_ID.fullmatch(source_id):
        raise ValueError("invalid source revision identity")
    root = config.project_registry_root_path() / "source-revisions" / "sources"
    metadata_path = root / project / source_id / "source.json"
    canonical_root = root.resolve(strict=True)
    canonical_source = canonical_root / project / source_id
    if metadata_path.parent.resolve(strict=True) != canonical_source:
        raise ValueError("source revision storage path is not canonical")
    if metadata_path.parent.is_symlink() or metadata_path.is_symlink():
        raise ValueError("source revision metadata path is a symlink")
    if metadata_path.stat().st_mode & 0o222:
        raise ValueError("source revision metadata is writable")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    tree_digest = str(payload.get("tree_digest") or "")
    if (
        payload.get("project") != project
        or payload.get("source_id") != source_id
        or payload.get("tree") != "tree"
        or tree_digest != "sha256:" + source_id.removeprefix("source.")
    ):
        raise ValueError("source revision metadata identity mismatch")
    tree = metadata_path.parent / "tree"
    if _tree_digest(tree, require_read_only=True) != tree_digest:
        raise ValueError("source revision tree digest mismatch")
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
            shutil.rmtree(temporary / ".git")
            tree_digest = _tree_digest(temporary)
        except ApplicationError:
            raise
        except (OSError, ValueError, subprocess.SubprocessError, UnicodeDecodeError) as exc:
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
        source_hash = tree_digest.removeprefix("sha256:")
        source_id = f"source.{source_hash}"
        provenance = {
            "proposal_id": proposal.get("proposal_id"),
            "binding": binding if isinstance(binding, dict) else {},
            "repository_identity": (
                proposal.get("repository_identity")
                if isinstance(proposal.get("repository_identity"), dict) else {}
            ),
            "checks": proposal.get("checks") if isinstance(proposal.get("checks"), list) else [],
        }
        canonical = {
            "project": project,
            "base_commit": base,
            "patch": patch.decode("utf-8"),
            "patch_digest": patch_digest,
            "tree_digest": tree_digest,
            "changed_files": changed,
            "proposal_provenance": provenance,
            "source_id": source_id,
        }
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        import_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        import_id = f"source-import-{import_hash[:24]}"
        plan = {
            "schema_version": 1,
            "import_id": import_id,
            **canonical,
            "confirmation": f"IMPORT SOURCE {source_id}",
            "executed": False,
        }
        self.plans.mkdir(parents=True, exist_ok=True)
        try:
            atomic_create(self.plans / f"{import_id}.json", plan)
        except FileExistsError:
            with self._locked_plan(import_id) as (_path, existing_plan):
                if any(existing_plan.get(key) != value for key, value in canonical.items()):
                    raise ApplicationError(
                        "source import plan identity collision",
                        code="SOURCE_IMPORT_BLOCKED",
                    )
                return existing_plan
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
                try:
                    if not path.is_file():
                        raise ApplicationError(
                            "source import plan not found", status_code=404,
                            code="UNKNOWN_SOURCE_IMPORT",
                        )
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("plan must be a mapping")
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    raise ApplicationError(
                        "source import plan is unreadable", code="SOURCE_IMPORT_BLOCKED",
                    ) from exc
                yield path, payload
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
            project = str(plan.get("project") or "")
            if not _PROJECT_ID.fullmatch(project):
                raise ApplicationError(
                    "invalid planned Project identity", code="SOURCE_IMPORT_BLOCKED",
                )
            tree_digest = str(plan.get("tree_digest") or "")
            if tree_digest != "sha256:" + source_id.removeprefix("source."):
                raise ApplicationError(
                    "source import plan tree identity changed", code="SOURCE_IMPORT_BLOCKED",
                )
            final = self.sources / project / source_id
            metadata_path = final / "source.json"
            if final.is_dir() and metadata_path.is_file():
                try:
                    resolve_source_tree(self.runtime.config, project, source_id)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    raise ApplicationError(
                        "source identity collision", code="SOURCE_IMPORT_BLOCKED",
                    ) from exc
                plan["executed"] = True
                plan["source"] = {
                    "source_id": source_id, "tree_path": str(final / "tree"),
                    "metadata_path": str(metadata_path),
                }
                atomic_write(plan_path, plan)
                return {"import": plan, "source": plan["source"]}
            if not self.runtime.config.action_runtime.allow_source_imports:
                raise ApplicationError(
                    "source imports are disabled by daemon policy",
                    code="SOURCE_IMPORT_BLOCKED",
                )
            repository = self._repository(project)
            base, patch, patch_digest, declared = self._proposal(plan)
            if patch_digest != plan.get("patch_digest"):
                raise ApplicationError("source import plan changed", code="SOURCE_IMPORT_BLOCKED")
            temporary = final.parent / f".{source_id}.{uuid4().hex}.tmp"
            try:
                changed = self._materialize(repository, base, patch, temporary / "tree")
                if changed != declared:
                    raise ApplicationError(
                        "source import changed after preview", code="SOURCE_IMPORT_BLOCKED",
                    )
                shutil.rmtree(temporary / "tree" / ".git")
                materialized_digest = _tree_digest(temporary / "tree")
                if materialized_digest != tree_digest:
                    raise ApplicationError(
                        "source import tree changed after preview",
                        code="SOURCE_IMPORT_BLOCKED",
                    )
                metadata = {
                    "schema_version": 1, "source_id": source_id,
                    "project": project, "base_commit": base,
                    "patch_digest": patch_digest, "changed_files": changed,
                    "tree_digest": tree_digest,
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
