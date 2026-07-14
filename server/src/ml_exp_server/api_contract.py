"""Versioned HTTP contract models shared by daemon route adapters.

The core package deliberately does not depend on these models.  They are the
typed serialization boundary for independently released HTTP clients.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .schemas import ProjectLifecycleRecord


API_PROTOCOL_VERSION = 1
MIN_CLIENT_PROTOCOL_VERSION = 1
VERSIONED_OPENAPI_PATH = "/api/v1/openapi.json"
CLIENT_PROTOCOL_HEADER = "X-ML-Expd-Client-Protocol"

SERVER_CAPABILITIES = (
    "actions.v1",
    "bearer-auth.v1",
    "observability.v1",
    "project-lifecycle.v1",
    "project-import.v1",
    "project-source-locator.v1",
    "source-revision-import.v1",
    "submissions.v1",
    "terminal-snapshot.v1",
    "terminal-snapshot-limits.v1",
    "tls-bind.v1",
)


class ObservabilityHealth(BaseModel):
    """Bounded service health; publishers may add non-secret counters."""

    model_config = ConfigDict(extra="allow")

    state: str = "UNKNOWN"


class PublisherLoopHealth(BaseModel):
    """Non-secret liveness state for the daemon-owned outbox loop."""

    last_success_at: float | None = None
    last_error: str | None = None
    consecutive_failures: int = Field(default=0, ge=0)


class DaemonHealth(BaseModel):
    """Stable compatibility handshake returned by ``GET /api/health``."""

    status: Literal["ok"] = "ok"
    server_version: str
    api_protocol_version: int = API_PROTOCOL_VERSION
    min_client_protocol_version: int = MIN_CLIENT_PROTOCOL_VERSION
    openapi_path: str = VERSIONED_OPENAPI_PATH
    capabilities: tuple[str, ...] = SERVER_CAPABILITIES
    authentication: Literal["none", "bearer"]
    transport_security: Literal["http", "https"]
    projects: int = Field(ge=0)
    workspace_id: str
    collector_enabled: bool
    collector_requested: bool
    collector_error: str | None = None
    project_write_recovery_errors: list[str] = Field(default_factory=list)
    project_writes: bool
    source_imports: bool
    scheduler_mutations: bool
    observability_mutations: bool
    local_evidence_rebuild: bool
    telemetry_enabled: bool
    observability: ObservabilityHealth
    publisher: PublisherLoopHealth = Field(default_factory=PublisherLoopHealth)


class InitialIndexResult(BaseModel):
    status: Literal["COMPLETED", "DEGRADED"]
    runs: int | None = Field(default=None, ge=0)
    error: str | None = None
    unavailable_run_roots: list[str] = Field(default_factory=list)


class ProjectRegistrationResponse(BaseModel):
    project: ProjectLifecycleRecord
    initial_index: InitialIndexResult
    effect: str


class ActionExecutionResponse(BaseModel):
    """Typed durable execution state while preserving operation-specific data."""

    model_config = ConfigDict(extra="allow")

    revision: int = Field(ge=0)
    status: Literal[
        "BLOCKED",
        "PREPARED",
        "AUTHORIZED",
        "EXECUTING",
        "RECONCILE_REQUIRED",
        "VERIFIED",
        "FAILED",
    ]
    authorized_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    last_reconciled_at: str | None = None
    result: Any = None
    error: str | None = None


class ActionSnapshotResponse(BaseModel):
    """Common envelope returned by prepare/authorize/execute/reconcile."""

    model_config = ConfigDict(extra="allow")

    action_id: str = Field(pattern=r"^action-[a-f0-9]{16}$")
    operation: str
    execution: ActionExecutionResponse
    journal: list[dict[str, Any]] = Field(default_factory=list)
