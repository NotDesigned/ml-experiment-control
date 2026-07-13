"""Transport-neutral construction of daemon stores and services.

Server and explicit operator entry points depend on this runtime. The runtime never
imports FastAPI, argparse, Textual, or any transport request/response type.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .actions import ActionService, ActionStore
from .ingest.indexer import RunIndex
from .identity import workspace_identity
from .project_config import load_research_project
from .project_registry import ProjectRegistry, ProjectRegistryError
from .schemas import (
    ServerConfig,
    ProjectLifecycleRecord,
    ProjectLifecycleState,
    ProjectRegistrationSource,
    ResearchProject,
)
from .telemetry import Telemetry, initialize_telemetry


def _load_registered_project(record: ProjectLifecycleRecord) -> ResearchProject:
    project = load_research_project(Path(record.project_file))
    if project.project != record.project:
        raise ProjectRegistryError(
            f"project registry identity drift for {record.project!r}: "
            f"{record.project_file} now declares {project.project!r}"
        )
    return project


@dataclass
class ExperimentServerRuntime:
    config: ServerConfig
    index: RunIndex
    projects: list[ResearchProject]
    action_store: ActionStore
    action_service: ActionService
    project_registry: ProjectRegistry
    telemetry: Telemetry
    workspace_id: str

    @classmethod
    def create(
        cls,
        config: ServerConfig,
        *,
        index: RunIndex | None = None,
        projects: list[ResearchProject] | None = None,
        on_index_update: Callable[[str, str], None] | None = None,
    ) -> "ExperimentServerRuntime":
        run_index = index or RunIndex(config.index_db_path())
        project_registry = ProjectRegistry(config.project_registry_root_path())
        if projects is None:
            records = project_registry.bootstrap(
                Path(ref.project_file) for ref in config.projects
            )
            loaded_projects = [_load_registered_project(record) for record in records
                               if record.state == ProjectLifecycleState.ACTIVE]
        else:
            # Explicit injected Projects are a test/embedding override. They
            # intentionally do not seed or mutate the durable registry.
            loaded_projects = list(projects)

        action_store = ActionStore(config.action_root_path())
        previous_callback = on_index_update

        def notify(project: str, run_id: str) -> None:
            if previous_callback is not None:
                previous_callback(project, run_id)

        run_index.on_update = notify
        telemetry = initialize_telemetry(config.telemetry)
        return cls(
            config=config,
            index=run_index,
            projects=loaded_projects,
            action_store=action_store,
            action_service=ActionService(action_store, config.action_runtime),
            project_registry=project_registry,
            telemetry=telemetry,
            workspace_id=workspace_identity(config),
        )

    def project(self, project_name: str) -> ResearchProject:
        project = next(
            (item for item in self.projects if item.project == project_name), None
        )
        if project is None:
            raise KeyError(f"unknown project: {project_name}")
        return project

    def project_records(self) -> list[ProjectLifecycleRecord]:
        return self.project_registry.records()

    def register_project(
        self, project_file: Path, *,
        source: ProjectRegistrationSource = ProjectRegistrationSource.MANUAL,
    ) -> ResearchProject:
        project = load_research_project(project_file)
        self.project_registry.register(project.project, project_file, source=source)
        if not any(item.project == project.project for item in self.projects):
            self.projects.append(project)
        return project

    def transition_project(
        self, project_name: str, target: ProjectLifecycleState, *, reason: str = "",
    ) -> ProjectLifecycleRecord:
        """Apply lifecycle change both durably and to this live runtime.

        The collector holds the same list object as ``self.projects``.  Slice
        replacement therefore makes pause/archive take effect between collector
        calls without a server restart.
        """
        activated: ResearchProject | None = None
        if target == ProjectLifecycleState.ACTIVE:
            record = next(
                (item for item in self.project_registry.records()
                 if item.project == project_name),
                None,
            )
            if record is None:
                raise ProjectRegistryError(f"unknown registered project: {project_name}")
            activated = _load_registered_project(record)
        record = self.project_registry.transition(project_name, target, reason=reason)
        if target == ProjectLifecycleState.ACTIVE:
            assert activated is not None
            if not any(item.project == project_name for item in self.projects):
                self.projects.append(activated)
        else:
            self.projects[:] = [item for item in self.projects if item.project != project_name]
        return record

    def unregister_project(self, project_name: str, *, reason: str = "") -> ProjectLifecycleRecord:
        record = self.project_registry.unregister(project_name, reason=reason)
        self.projects[:] = [item for item in self.projects if item.project != project_name]
        return record

    def unregister_all_projects(self, *, reason: str = "") -> list[ProjectLifecycleRecord]:
        records = self.project_registry.unregister_all(reason=reason)
        self.projects.clear()
        return records

    def close(self) -> None:
        self.index.close()
        self.telemetry.shutdown()

    def __enter__(self) -> "ExperimentServerRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
