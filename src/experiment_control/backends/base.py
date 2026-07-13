"""Backend contract and registry used by the controller dispatch loop."""

from __future__ import annotations

from typing import Protocol

from ..contracts import (
    AssetVerification,
    AttemptManifest,
    BackendLogs,
    BackendStatus,
    Campaign,
    CollectionResult,
    PreflightScope,
    RunSpec,
    SubmissionIntent,
    SubmissionRequest,
)
from ..preflight import PreflightReport
from ..identity import IdentityReport
from ..project import AssetProbe, SourceBundle


class Backend(Protocol):
    kind: str

    def validate(self, run: RunSpec) -> None: ...
    def preflight(self, run: RunSpec, *, scope: PreflightScope) -> PreflightReport: ...
    def environment(
        self, campaign: Campaign, run: RunSpec, source_id: str, attempt_id: str
    ) -> dict[str, str]: ...
    def submission_request(
        self, campaign: Campaign, run: RunSpec, attempt_id: str
    ) -> SubmissionRequest: ...
    def recover_submission(
        self, run: RunSpec, intent: SubmissionIntent, attempt_id: str
    ) -> str | None: ...
    def identity(
        self, campaign: Campaign, run: RunSpec, attempt_id: str
    ) -> IdentityReport: ...
    def verify_assets(self, run: RunSpec, probes: list[AssetProbe]) -> AssetVerification: ...
    def stage(
        self, campaign: Campaign, run: RunSpec, source_id: str,
        source_bundle: SourceBundle,
    ) -> bool: ...
    def render(self, manifest: AttemptManifest) -> str: ...
    def submit(
        self, campaign: Campaign, run: RunSpec, manifest: AttemptManifest, *,
        dry_run: bool, intent: SubmissionIntent | None = None,
    ) -> str: ...
    def status(self, campaign: Campaign, run: RunSpec) -> BackendStatus: ...
    def collect(self, campaign: Campaign, run: RunSpec) -> CollectionResult: ...
    def logs(self, campaign: Campaign, run: RunSpec, *, tail: int) -> BackendLogs: ...
    def cancel(self, campaign: Campaign, run: RunSpec) -> BackendStatus: ...


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
