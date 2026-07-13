"""Pydantic models shared by ingest, API, and collector.

Canonical state lives in run-directory files (manifest.yaml, status.json,
collection.json, decision.json, events.jsonl) plus project-side YAML. Everything
here is either a read model or a transport-neutral operation contract.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1

# Vocabulary mirrored from the package-owned run-directory producer
# (experiment_control.manifest RunState et al.). Clients render unknown
# values verbatim rather than failing, so these are reference sets, not validators.
RUN_STATES = {
    "NOT_SUBMITTED", "CREATED", "SUBMITTING", "QUEUED", "STARTING", "RUNNING",
    "EVALUATING", "SUCCEEDED", "FAILED", "PREEMPTED", "CANCELLED", "UNKNOWN",
}
TERMINAL_RUN_STATES = {"SUCCEEDED", "FAILED", "CANCELLED"}


class OperationScopeType(str, Enum):
    """Object namespaces accepted by daemon read and mutation operations."""

    PROJECT = "project"
    RESEARCH_QUESTION = "research_question"
    CAMPAIGN = "campaign"
    RUN = "run"
    ATTEMPT = "attempt"


class ProjectLifecycleState(str, Enum):
    """Daemon-local lifecycle of a registered research Project.

    This is deliberately separate from the project-owned
    ``research_project.yaml`` contract.  A lifecycle change controls whether
    this daemon workspace indexes and observes the project; it never changes
    repository files, campaign provenance, Run evidence, or scheduler jobs.
    """

    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    ARCHIVED = "ARCHIVED"


class ProjectRegistrationSource(str, Enum):
    """How a project first entered this daemon workspace."""

    CONFIG_SEED = "CONFIG_SEED"
    # Read-only compatibility for registries created before explicit Project
    # lifecycle management. New registrations never emit these values.
    LEGACY_ONBOARDING = "LEGACY_ONBOARDING"
    ONBOARDING = "ONBOARDING"
    MANUAL = "MANUAL"


class CampaignRelationship(str, Enum):
    """How an immutable Run relates to the current authored campaign catalog."""

    UNRESOLVED = "UNRESOLVED"
    MATCHED = "MATCHED"
    LEGACY_INFERRED = "LEGACY_INFERRED"
    ORPHANED_CAMPAIGN = "ORPHANED_CAMPAIGN"
    UNDECLARED_RUN = "UNDECLARED_RUN"
    CAMPAIGN_REVISION_DRIFT = "CAMPAIGN_REVISION_DRIFT"
    PROJECT_MISMATCH = "PROJECT_MISMATCH"
    ROLE_MISMATCH = "ROLE_MISMATCH"
    DUPLICATE_RUN_ID = "DUPLICATE_RUN_ID"


class OperationScope(BaseModel):
    """Exact daemon object targeted by an immutable operation intent."""

    scope_type: OperationScopeType
    project: str
    object_id: str


class TelemetryConfig(BaseModel):
    enabled: bool = True
    otlp_http_endpoint: Optional[str] = "http://127.0.0.1:4318/v1/traces"
    service_name: str = "ml-expd"
    capture_content: Literal[False] = False


class ActionRuntimeConfig(BaseModel):
    """Mutation boundary; both switches default closed."""

    model_config = ConfigDict(extra="forbid")

    allow_project_writes: bool = False
    allow_scheduler_mutations: bool = False
    timeout_seconds: int = 300
    gate_ttl_seconds: int = 1800


class CampaignRunMembership(BaseModel):
    """A Run's role in one authored Campaign revision.

    Role and arm are comparison-relative. They are deliberately modeled on
    this association instead of as permanent intrinsic properties of a Run.
    """

    run_id: str
    kind: Literal["materialize", "reuse"] = "materialize"
    role: Optional[str] = None
    arm: Optional[str] = None
    replicate: Optional[int] = None
    purpose: Optional[str] = None
    included_in_analysis: bool = True


class CampaignRevision(BaseModel):
    """Resolved immutable identity of the current authored campaign file."""

    campaign: str
    project: str
    revision_id: str
    file: str
    research_contract: Optional[dict[str, Any]] = None
    memberships: list[CampaignRunMembership] = Field(default_factory=list)


class CampaignRef(BaseModel):
    """Project-owned logical Campaign catalog entry.

    ``name`` remains the v1 on-disk logical identity. ``current_revision`` is
    resolved from ``file`` at load time and is not authored separately.
    """

    name: str
    file: Optional[str] = None
    role_notes: dict[str, str] = Field(default_factory=dict)
    current_revision: Optional[CampaignRevision] = None


class CampaignBinding(BaseModel):
    """Reconciliation between frozen Run provenance and authored design."""

    relationship: CampaignRelationship = CampaignRelationship.UNRESOLVED
    issues: list[CampaignRelationship] = Field(default_factory=list)
    origin_project: Optional[str] = None
    origin_campaign: Optional[str] = None
    origin_revision: Optional[str] = None
    current_revision: Optional[str] = None
    membership: Optional[CampaignRunMembership] = None


class CampaignMembershipBinding(BaseModel):
    """One current Campaign revision that includes a materialized Run."""

    campaign: str
    revision_id: str
    membership: CampaignRunMembership
    is_origin: bool = False


class ResearchLinks(BaseModel):
    campaigns: list[str] = Field(default_factory=list)
    runs: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class ResearchQuestion(BaseModel):
    """Optional research lens linking questions to campaigns, runs, and evidence."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    id: str
    title: str
    status: str = "OPEN"
    summary: str = ""
    notes: list[str] = Field(default_factory=list)
    links: ResearchLinks = Field(default_factory=ResearchLinks)


class ControllerConfig(BaseModel):
    """How the collector invokes the project's experimentctl (observation verbs only)."""

    python: str
    experimentctl: str
    workdir: str = "."
    capabilities: dict[str, bool] = Field(default_factory=dict)


class ResearchProject(BaseModel):
    """research_project.yaml inside a science repo."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    project: str
    title: str
    run_roots: list[str]
    controller: Optional[ControllerConfig] = None
    campaigns: list[CampaignRef] = Field(default_factory=list)
    research_questions_dir: Optional[str] = None
    # Resolved at load time; not part of the on-disk schema.
    base_dir: Optional[Path] = None
    authored_file: Optional[Path] = Field(default=None, exclude=True)
    research_questions: list[ResearchQuestion] = Field(default_factory=list)

    def resolved_run_roots(self) -> list[Path]:
        base = self.base_dir or Path(".")
        return [(base / root).resolve() if not Path(root).is_absolute() else Path(root)
                for root in self.run_roots]


class ProjectRef(BaseModel):
    project_file: str


class ProjectLifecycleRecord(BaseModel):
    """Durable workspace-owned registration, not a science-repo manifest."""

    project: str
    project_file: str
    state: ProjectLifecycleState = ProjectLifecycleState.ACTIVE
    source: ProjectRegistrationSource
    registered_at: str
    updated_at: str
    state_reason: str = ""


class ServerConfig(BaseModel):
    """Top-level daemon workspace configuration."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    index_db: str = "~/.local/state/ml-expd/index.sqlite"
    action_root: str = "~/.local/state/ml-expd/actions"
    # The lifecycle registry is workspace-owned. When omitted it is derived
    # from index_db so independent daemon workspaces stay isolated.
    project_registry_root: Optional[str] = None
    action_runtime: ActionRuntimeConfig = Field(default_factory=ActionRuntimeConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    # The daemon owns live collection by default. ``--snapshot`` is the
    # explicit offline opt-out.
    collector_enabled: bool = True
    poll_interval_seconds: int = 20
    projects: list[ProjectRef] = Field(default_factory=list)

    def index_db_path(self) -> Path:
        return Path(self.index_db).expanduser()

    def action_root_path(self) -> Path:
        return Path(self.action_root).expanduser()

    def project_registry_root_path(self) -> Path:
        if self.project_registry_root:
            return Path(self.project_registry_root).expanduser()
        index = self.index_db_path()
        return index.with_name(f"{index.stem}.projects")


class EvidenceLayer(BaseModel):
    """One evidence layer's state plus its own freshness.

    `as_of` is when the underlying evidence was produced (record timestamp or
    file mtime), never "now". `stale` is set by freshness rules and must be
    rendered, not hidden.
    """

    state: Optional[str] = None
    attempt_id: Optional[str] = None
    as_of: Optional[float] = None  # unix seconds
    source: Optional[str] = None  # file the evidence came from
    detail: dict[str, Any] = Field(default_factory=dict)
    stale: bool = False
    stale_reason: Optional[str] = None


class EvidenceLayers(BaseModel):
    """The five separated evidence layers. Never collapse into one state."""

    scheduler: EvidenceLayer = Field(default_factory=EvidenceLayer)
    worker: EvidenceLayer = Field(default_factory=EvidenceLayer)
    process: EvidenceLayer = Field(default_factory=EvidenceLayer)
    model: EvidenceLayer = Field(default_factory=EvidenceLayer)
    evaluation: EvidenceLayer = Field(default_factory=EvidenceLayer)


class AttemptSummary(BaseModel):
    attempt_id: str
    state: Optional[str] = None
    backend: Optional[str] = None
    backend_job_id: Optional[str] = None
    decision: dict[str, Any] = Field(default_factory=dict)
    has_submission: bool = False


class RunIndexRow(BaseModel):
    """Flat per-run read model persisted in the SQLite index."""

    project: str
    campaign: Optional[str] = None
    campaign_source: Optional[str] = None  # manifest | argument | directory | None
    campaign_binding: CampaignBinding = Field(default_factory=CampaignBinding)
    campaign_memberships: list[CampaignMembershipBinding] = Field(default_factory=list)
    run_id: str
    role: Optional[str] = None
    role_source: Optional[str] = None  # "manifest" | "campaign_file" | "heuristic" | None
    run_dir: str
    scheduler_state: Optional[str] = None
    evidence: EvidenceLayers = Field(default_factory=EvidenceLayers)
    latest_metrics: dict[str, Any] = Field(default_factory=dict)
    eval_metrics: dict[str, Any] = Field(default_factory=dict)
    eval_variants: list[dict[str, Any]] = Field(default_factory=list)
    canonical_eval_variant_id: Optional[str] = None
    decision: dict[str, Any] = Field(default_factory=dict)
    decision_history: list[dict[str, Any]] = Field(default_factory=list)
    research_contract: Optional[dict[str, Any]] = None
    research_contract_source: Optional[str] = None
    checkpoint: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    attempts: list[AttemptSummary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    evidence_conflicts: list[str] = Field(default_factory=list)
    scanned_at: Optional[float] = None

    @property
    def is_terminal(self) -> bool:
        return (self.scheduler_state or "").upper() in TERMINAL_RUN_STATES


class CollectorRunStatus(BaseModel):
    run_id: str
    project: str
    last_poll_at: Optional[float] = None
    last_verb: Optional[str] = None
    last_error: Optional[str] = None
    verb_results: dict[str, dict[str, Any]] = Field(default_factory=dict)
