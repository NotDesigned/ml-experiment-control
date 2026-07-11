"""Backend contract and registry used by the controller dispatch loop."""

from __future__ import annotations

from typing import Any, Protocol

from ..preflight import PreflightReport
from ..project import AssetProbe, SourceBundle


class Backend(Protocol):
    kind: str

    def validate(self, run: dict[str, Any]) -> None: ...
    def preflight(self, run: dict[str, Any], *, scope: str) -> PreflightReport: ...
    def environment(
        self, campaign: dict[str, Any], run: dict[str, Any], source_id: str, attempt_id: str
    ) -> dict[str, str]: ...
    def submission_request(
        self, campaign: dict[str, Any], run: dict[str, Any], attempt_id: str
    ) -> dict[str, Any]: ...
    def recover_submission(
        self, run: dict[str, Any], intent: dict[str, Any], attempt_id: str
    ) -> str | None: ...
    def verify_assets(self, run: dict[str, Any], probes: list[AssetProbe]) -> dict[str, Any]: ...
    def stage(
        self, campaign: dict[str, Any], run: dict[str, Any], source_id: str,
        source_bundle: SourceBundle,
    ) -> bool: ...
    def render(self, manifest: dict[str, Any]) -> str: ...
    def submit(self, campaign: dict[str, Any], run: dict[str, Any], manifest: dict[str, Any], *, dry_run: bool) -> str: ...
    def status(self, campaign: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]: ...
    def collect(self, campaign: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]: ...
    def logs(self, campaign: dict[str, Any], run: dict[str, Any], *, tail: int) -> dict[str, Any]: ...
    def cancel(self, campaign: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]: ...


class BackendRegistry:
    """Small explicit registry that removes backend branching from CLI code."""

    def __init__(self, *backends: Backend):
        self._backends = {backend.kind: backend for backend in backends}

    def get(self, kind: str) -> Backend:
        try:
            return self._backends[kind]
        except KeyError as error:
            raise ValueError(f"unsupported experiment backend: {kind}") from error

    @property
    def kinds(self) -> frozenset[str]:
        return frozenset(self._backends)
