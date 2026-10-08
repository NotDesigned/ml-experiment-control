"""Projects for ExperimentServerApplication."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from ..application_errors import ApplicationError
from ..schemas import ProjectLifecycleState

class ProjectLifecycle:
    """Projects operations; state belongs to the application facade."""

    def project_lifecycle_list(self) -> dict[str, Any]:
        return self.project_service.lifecycle_list()

    def project_register(self, project_file: Path) -> dict[str, Any]:
        return self.project_service.register(project_file)

    def project_import_preview(
        self, repository_root: Path, *, project: str | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:
        if self.project_import_service is None:
            raise ApplicationError(
                "Project import service is unavailable", code="PROJECT_IMPORT_BLOCKED",
            )
        return self.project_import_service.preview(
            repository_root, project=project, title=title,
        )

    def project_import_execute(
        self, import_id: str, confirmation: str,
    ) -> dict[str, Any]:
        if self.project_import_service is None:
            raise ApplicationError(
                "Project import service is unavailable", code="PROJECT_IMPORT_BLOCKED",
            )
        return self.project_import_service.execute(import_id, confirmation)

    def source_revision_preview(
        self, project: str, proposal: dict[str, Any],
    ) -> dict[str, Any]:
        if self.source_revision_service is None:
            raise ApplicationError(
                "source revision service is unavailable", code="SOURCE_IMPORT_BLOCKED",
            )
        return self.source_revision_service.preview(project, proposal)

    def source_revision_execute(
        self, import_id: str, confirmation: str,
    ) -> dict[str, Any]:
        if self.source_revision_service is None:
            raise ApplicationError(
                "source revision service is unavailable", code="SOURCE_IMPORT_BLOCKED",
            )
        return self.source_revision_service.execute(import_id, confirmation)

    def source_revision_get(self, project: str, source_id: str) -> dict[str, Any]:
        if self.source_revision_service is None:
            raise ApplicationError(
                "source revision service is unavailable", code="SOURCE_IMPORT_BLOCKED",
            )
        return self.source_revision_service.get(project, source_id)

    def project_lifecycle_transition(
        self, project: str, action: str, target: ProjectLifecycleState, *, reason: str = "",
    ) -> dict[str, Any]:
        return self.project_service.transition(
            project, action, target, reason=reason,
        )

    def project_unregister(self, project: str, *, reason: str = "") -> dict[str, Any]:
        return self.project_service.unregister(project, reason=reason)

    def project_unregister_all(self, *, reason: str = "") -> dict[str, Any]:
        return self.project_service.unregister_all(reason=reason)
