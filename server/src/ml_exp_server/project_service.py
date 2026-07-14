"""Project registration and lifecycle application boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .application_errors import ApplicationError
from .ingest.indexer import index_project
from .project_registry import ProjectRegistryError
from .runtime import ExperimentServerRuntime
from .schemas import ProjectLifecycleState, ProjectRegistrationSource


class ProjectApplicationService:
    """Own Project use cases without depending on HTTP or CLI transports."""

    def __init__(self, runtime: ExperimentServerRuntime):
        self.runtime = runtime

    def lifecycle_list(self) -> dict[str, Any]:
        return {
            "projects": [
                record.model_dump(mode="json")
                for record in self.runtime.project_records()
            ],
            "events": self.runtime.project_registry.events(),
        }

    def register(self, project_file: Path) -> dict[str, Any]:
        if not project_file.is_absolute():
            raise ApplicationError(
                "project_file must be an absolute daemon-host path",
                code="PROJECT_REGISTRATION_BLOCKED",
            )
        try:
            project = self.runtime.register_project(
                project_file, source=ProjectRegistrationSource.MANUAL,
            )
        except (ProjectRegistryError, OSError, ValueError) as exc:
            raise ApplicationError(
                str(exc), code="PROJECT_REGISTRATION_BLOCKED",
            ) from exc

        index_error = None
        indexed_runs = None
        unavailable_roots = [
            str(root) for root in project.resolved_run_roots() if not root.is_dir()
        ]
        try:
            indexed_runs = index_project(self.runtime.index, project)
        except Exception as exc:
            # Registration is already durable; collection retries this work.
            index_error = str(exc)[:500] or type(exc).__name__
        if index_error is None and unavailable_roots:
            preview = ", ".join(unavailable_roots[:5])
            count = len(unavailable_roots) - 5
            suffix = f" (+{count} more)" if count > 0 else ""
            index_error = f"run roots are unavailable: {preview}{suffix}"
        record = next(
            item for item in self.runtime.project_records()
            if item.project == project.project
        )
        return {
            "project": record.model_dump(mode="json"),
            "initial_index": {
                "status": "DEGRADED" if index_error else "COMPLETED",
                "runs": indexed_runs,
                "error": index_error,
                "unavailable_run_roots": unavailable_roots,
            },
            "effect": (
                "project is active; initial indexing will be retried"
                if index_error else
                "project is active; initial indexing completed and collector observation may run"
            ),
        }

    def transition(
        self, project: str, target: ProjectLifecycleState, *, reason: str = "",
    ) -> dict[str, Any]:
        if target == ProjectLifecycleState.ARCHIVED and not reason.strip():
            raise ApplicationError(
                "archiving a project requires a reason",
                code="PROJECT_LIFECYCLE_BLOCKED",
            )
        try:
            record = self.runtime.transition_project(project, target, reason=reason)
        except (ProjectRegistryError, FileNotFoundError, ValueError) as exc:
            raise ApplicationError(
                str(exc), status_code=404 if "unknown" in str(exc) else 409,
                code="PROJECT_LIFECYCLE_BLOCKED",
            ) from exc
        return {
            "project": record.model_dump(mode="json"),
            "active": target == ProjectLifecycleState.ACTIVE,
            "effect": (
                "project is active; indexing and collector observation may resume"
                if target == ProjectLifecycleState.ACTIVE else
                "project is inactive; this daemon stops indexing and collecting it"
            ),
        }

    def unregister(self, project: str, *, reason: str = "") -> dict[str, Any]:
        try:
            record = self.runtime.unregister_project(project, reason=reason)
        except ProjectRegistryError as exc:
            raise ApplicationError(
                str(exc), status_code=404, code="UNKNOWN_PROJECT",
            ) from exc
        return {
            "unregistered": record.model_dump(mode="json"),
            "effect": (
                "Daemon tracking was removed; repository files, runs, artifacts, "
                "and jobs were not changed"
            ),
        }

    def unregister_all(self, *, reason: str = "") -> dict[str, Any]:
        records = self.runtime.unregister_all_projects(reason=reason)
        return {
            "unregistered": [record.model_dump(mode="json") for record in records],
            "effect": (
                "Daemon tracking was removed; repository files, runs, artifacts, "
                "and jobs were not changed"
            ),
        }
