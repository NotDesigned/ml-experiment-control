"""Workspace-local Project registration and lifecycle.

Project YAML is still the authoritative science-repository contract.  This
registry only decides whether *this daemon workspace* actively tracks that
contract.  It intentionally owns no scheduler or repository mutations.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from .storage import StorageError, atomic_json, exclusive_file_lock, read_json, utc_now
from .schemas import (
    ProjectLifecycleRecord,
    ProjectLifecycleState,
    ProjectRegistrationSource,
)


REGISTRY_SCHEMA_VERSION = 1


class ProjectRegistryError(ValueError):
    """A requested lifecycle operation is invalid for the current registry."""


class ProjectRegistry:
    """Atomic, workspace-scoped lifecycle ledger.

    daemon config ``projects`` is imported only when this ledger is first
    created. This prevents configuration from silently re-registering a
    project after an explicit removal.
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = root / "registry.json"
        self.events_path = root / "events.jsonl"
        self.lock_path = root / ".registry.lock"
        self._lock = threading.RLock()

    @contextmanager
    def _locked(self):
        """Hold both the in-process and workspace-wide registry lock."""

        with self._lock:
            with exclusive_file_lock(self.lock_path):
                yield

    def exists(self) -> bool:
        return self.path.is_file()

    @staticmethod
    def _default() -> dict:
        return {"schema_version": REGISTRY_SCHEMA_VERSION, "projects": []}

    def _load(self) -> dict:
        try:
            payload = read_json(self.path, self._default())
        except StorageError as exc:
            raise ProjectRegistryError(f"invalid project registry: {self.path}") from exc
        if not isinstance(payload, dict):
            raise ProjectRegistryError(f"invalid project registry: {self.path}")
        if payload.get("schema_version", REGISTRY_SCHEMA_VERSION) != REGISTRY_SCHEMA_VERSION:
            raise ProjectRegistryError(
                f"unsupported project registry schema: {payload.get('schema_version')}"
            )
        if not isinstance(payload.get("projects", []), list):
            raise ProjectRegistryError(f"invalid project registry projects: {self.path}")
        return payload

    @staticmethod
    def _records(payload: dict) -> list[ProjectLifecycleRecord]:
        records: list[ProjectLifecycleRecord] = []
        seen: set[str] = set()
        seen_paths: dict[Path, str] = {}
        for item in payload.get("projects", []):
            try:
                record = ProjectLifecycleRecord.model_validate(item)
            except Exception as exc:  # storage corruption is an operator-visible error
                raise ProjectRegistryError("invalid project lifecycle record") from exc
            if record.project in seen:
                raise ProjectRegistryError(f"duplicate project in registry: {record.project}")
            path = Path(record.project_file).expanduser()
            if not path.is_absolute():
                raise ProjectRegistryError(
                    f"project registry path must be absolute: {record.project_file}"
                )
            canonical = path.resolve()
            if canonical in seen_paths:
                raise ProjectRegistryError(
                    f"duplicate project manifest in registry: {canonical} is used by "
                    f"{seen_paths[canonical]!r} and {record.project!r}"
                )
            seen.add(record.project)
            seen_paths[canonical] = record.project
            records.append(record)
        return records

    def records(self) -> list[ProjectLifecycleRecord]:
        with self._locked():
            return self._records(self._load())

    @classmethod
    def read_records(cls, root: Path) -> list[ProjectLifecycleRecord]:
        """Read an existing registry without creating its directory or files."""

        path = root / "registry.json"
        if not path.is_file():
            return []
        try:
            payload = read_json(path, cls._default())
        except StorageError as exc:
            raise ProjectRegistryError(f"invalid project registry: {path}") from exc
        if not isinstance(payload, dict):
            raise ProjectRegistryError(f"invalid project registry: {path}")
        if payload.get("schema_version", REGISTRY_SCHEMA_VERSION) != REGISTRY_SCHEMA_VERSION:
            raise ProjectRegistryError(
                f"unsupported project registry schema: {payload.get('schema_version')}"
            )
        if not isinstance(payload.get("projects", []), list):
            raise ProjectRegistryError(f"invalid project registry projects: {path}")
        return cls._records(payload)

    def active_records(self) -> list[ProjectLifecycleRecord]:
        return [record for record in self.records()
                if record.state == ProjectLifecycleState.ACTIVE]

    def bootstrap(
        self,
        config_project_files: Iterable[Path],
    ) -> list[ProjectLifecycleRecord]:
        """Create the one-time ledger from configured project manifests.

        Paths are intentionally recorded before loading project manifests.  A
        malformed project YAML should remain an operator-visible startup error
        in the normal runtime load path, just as it did before migration.
        """
        with self._locked():
            if self.exists():
                return self._records(self._load())
            now = utc_now()
            pending_paths: list[Path] = []
            seen_paths: set[Path] = set()
            for value in config_project_files:
                path = Path(value).expanduser().resolve()
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                pending_paths.append(path)
            # bootstrap needs an authored Project identity.  Import lazily to
            # keep this module independent of YAML parsing during normal CRUD.
            from .project_config import load_research_project

            materialized: list[ProjectLifecycleRecord] = []
            names: set[str] = set()
            for path in pending_paths:
                project = load_research_project(path)
                if project.project in names:
                    raise ProjectRegistryError(
                        f"duplicate project name during registry bootstrap: {project.project}"
                    )
                names.add(project.project)
                materialized.append(ProjectLifecycleRecord(
                    project=project.project,
                    project_file=str(path), state=ProjectLifecycleState.ACTIVE,
                    source=ProjectRegistrationSource.CONFIG_SEED,
                    registered_at=now, updated_at=now,
                ))
            payload = self._default()
            payload["projects"] = [record.model_dump(mode="json") for record in materialized]
            atomic_json(self.path, payload)
            self._append_event("BOOTSTRAP", "", now, {
                "imported": [record.model_dump(mode="json") for record in materialized],
            })
            return materialized

    def register(
        self,
        project: str,
        project_file: Path,
        *,
        source: ProjectRegistrationSource = ProjectRegistrationSource.MANUAL,
    ) -> ProjectLifecycleRecord:
        """Add a verified project or reactivate an explicitly paused one."""
        path = str(project_file.expanduser().resolve())
        with self._locked():
            payload = self._load()
            records = self._records(payload)
            current = next((record for record in records if record.project == project), None)
            path_owner = next(
                (
                    record for record in records
                    if Path(record.project_file).expanduser().resolve() == Path(path)
                ),
                None,
            )
            if path_owner is not None and path_owner.project != project:
                raise ProjectRegistryError(
                    f"project manifest {path} is already registered as "
                    f"{path_owner.project!r}; changing Project identity in place is not allowed"
                )
            now = utc_now()
            if current is not None:
                if Path(current.project_file).resolve() != Path(path):
                    raise ProjectRegistryError(
                        f"project {project!r} is already registered from {current.project_file}"
                    )
                if current.state == ProjectLifecycleState.ARCHIVED:
                    raise ProjectRegistryError(
                        f"project {project!r} is archived; restore it before reactivating"
                    )
                if current.state == ProjectLifecycleState.ACTIVE:
                    return current
                updated = current.model_copy(update={
                    "state": ProjectLifecycleState.ACTIVE,
                    "updated_at": now,
                    "state_reason": "registered",
                })
                records = [updated if item.project == project else item for item in records]
            else:
                updated = ProjectLifecycleRecord(
                    project=project, project_file=path,
                    state=ProjectLifecycleState.ACTIVE, source=source,
                    registered_at=now, updated_at=now, state_reason="registered",
                )
                records.append(updated)
            payload["projects"] = [record.model_dump(mode="json") for record in records]
            atomic_json(self.path, payload)
            self._append_event("REGISTER", project, now, updated.model_dump(mode="json"))
            return updated

    def transition(
        self, project: str, target: ProjectLifecycleState, *, reason: str = "",
    ) -> ProjectLifecycleRecord:
        allowed = {
            ProjectLifecycleState.ACTIVE: {
                ProjectLifecycleState.PAUSED, ProjectLifecycleState.ARCHIVED,
            },
            ProjectLifecycleState.PAUSED: {
                ProjectLifecycleState.ACTIVE, ProjectLifecycleState.ARCHIVED,
            },
            ProjectLifecycleState.ARCHIVED: {ProjectLifecycleState.PAUSED},
        }
        with self._locked():
            payload = self._load()
            records = self._records(payload)
            current = next((record for record in records if record.project == project), None)
            if current is None:
                raise ProjectRegistryError(f"unknown registered project: {project}")
            if current.state == target:
                return current
            if target not in allowed[current.state]:
                raise ProjectRegistryError(
                    f"cannot transition project {project!r} from {current.state.value} "
                    f"to {target.value}"
                )
            now = utc_now()
            updated = current.model_copy(update={
                "state": target, "updated_at": now, "state_reason": reason,
            })
            payload["projects"] = [
                updated.model_dump(mode="json") if item.project == project
                else item.model_dump(mode="json") for item in records
            ]
            atomic_json(self.path, payload)
            self._append_event(target.value, project, now, {
                "from": current.state.value, "reason": reason,
            })
            return updated

    def unregister(self, project: str, *, reason: str = "") -> ProjectLifecycleRecord:
        """Forget one project from this daemon workspace, preserving its audit event."""
        with self._locked():
            payload = self._load()
            records = self._records(payload)
            current = next((record for record in records if record.project == project), None)
            if current is None:
                raise ProjectRegistryError(f"unknown registered project: {project}")
            payload["projects"] = [
                record.model_dump(mode="json") for record in records
                if record.project != project
            ]
            atomic_json(self.path, payload)
            self._append_event("UNREGISTER", project, utc_now(), {
                "from": current.state.value, "project_file": current.project_file,
                "reason": reason,
            })
            return current

    def unregister_all(self, *, reason: str = "") -> list[ProjectLifecycleRecord]:
        with self._locked():
            payload = self._load()
            records = self._records(payload)
            if not records:
                return []
            payload["projects"] = []
            atomic_json(self.path, payload)
            now = utc_now()
            for record in records:
                self._append_event("UNREGISTER", record.project, now, {
                    "from": record.state.value, "project_file": record.project_file,
                    "reason": reason or "unregister_all",
                })
            return records

    def events(self, *, limit: int = 200) -> list[dict]:
        if not self.events_path.is_file():
            return []
        items: list[dict] = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines()[-limit:]:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                items.append(item)
        return items

    def _append_event(self, event: str, project: str, at: str, detail: dict) -> None:
        payload = {"event": event, "project": project, "at": at, "detail": detail}
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
