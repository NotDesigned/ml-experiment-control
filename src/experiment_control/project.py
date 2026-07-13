"""Project-facing contract for the platform-neutral experiment controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from .contracts import (
    Campaign,
    CheckpointRecord,
    JsonObject,
    MetricRecord,
    ProjectRun,
    RunSummary,
)


@dataclass(frozen=True)
class AssetRequirement:
    """One immutable input required by a resolved scientific configuration."""

    kind: str
    identity: str
    reason: str


@dataclass(frozen=True)
class AssetProbe:
    """A backend-verifiable filesystem representation of an asset."""

    requirement: AssetRequirement
    path: str
    file: bool = False


@dataclass(frozen=True)
class SourceBundle:
    """Controller-local source tree and project-selected staging policy."""

    root: Path
    excludes: tuple[str, ...] = ()
    container_path: str = "/workspace"
    identity_command: tuple[str, ...] = ()
    required_paths: tuple[str, ...] = ()


class ProjectAdapter(Protocol):
    """Scientific-project behavior intentionally excluded from schedulers."""

    name: str
    safe_env_keys: frozenset[str]

    def validate_run(self, run: ProjectRun) -> None: ...
    def operational_overrides(
        self, env: Mapping[str, str], output_dir: str
    ) -> list[str]: ...
    def resolve_config(self, config_path: str, overrides: list[str]) -> JsonObject: ...
    def environment(
        self, campaign: Campaign, run: ProjectRun
    ) -> dict[str, str]: ...
    def command(self, run: ProjectRun) -> list[str]: ...
    def plan_assets(
        self, config_path: str, overrides: list[str]
    ) -> list[AssetRequirement]: ...
    def asset_probes(
        self, requirements: list[AssetRequirement], environment: Mapping[str, str]
    ) -> list[AssetProbe]: ...
    def parse_metric(self, line: str) -> MetricRecord | None: ...
    def parse_checkpoint(self, line: str) -> CheckpointRecord | None: ...
    def summarize(self, run_dir: Path) -> RunSummary: ...
    def source_bundle(self, repo_root: Path) -> SourceBundle: ...


class ProjectRegistry:
    """Explicit adapter registry; campaign data never imports arbitrary code."""

    def __init__(self, *projects: ProjectAdapter):
        self._projects = {project.name: project for project in projects}

    def get(self, name: str) -> ProjectAdapter:
        try:
            return self._projects[name]
        except KeyError as error:
            raise ValueError(f"unsupported experiment project: {name}") from error

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._projects)
