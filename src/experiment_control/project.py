"""Project-facing contract for the platform-neutral experiment controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol


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


class ProjectAdapter(Protocol):
    """Scientific-project behavior intentionally excluded from schedulers."""

    name: str
    safe_env_keys: frozenset[str]

    def validate_run(self, run: dict[str, Any]) -> None: ...
    def operational_overrides(
        self, env: Mapping[str, str], output_dir: str
    ) -> list[str]: ...
    def resolve_config(self, config_path: str, overrides: list[str]) -> dict[str, Any]: ...
    def scientific_config(self, config: dict[str, Any]) -> dict[str, Any]: ...
    def environment(
        self, campaign: dict[str, Any], run: dict[str, Any]
    ) -> dict[str, str]: ...
    def command(self, run: dict[str, Any]) -> list[str]: ...
    def plan_assets(
        self, config_path: str, overrides: list[str]
    ) -> list[AssetRequirement]: ...
    def asset_probes(
        self, requirements: list[AssetRequirement], environment: Mapping[str, str]
    ) -> list[AssetProbe]: ...
    def parse_metric(self, line: str) -> dict[str, Any] | None: ...
    def summarize(self, run_dir: Path) -> dict[str, Any]: ...
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
