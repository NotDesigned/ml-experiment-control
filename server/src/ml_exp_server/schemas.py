"""Pydantic models shared by ingest, API, and collector.

Canonical state lives in run-directory files (manifest.yaml, status.json,
collection.json, decision.json, events.jsonl) plus project-side YAML. Everything
here is either a read model or a transport-neutral operation contract.
"""

from __future__ import annotations

from enum import Enum
import ipaddress
from pathlib import Path
import re
import sys
from typing import Any, Literal, Optional
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1
SAFE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"


def _safe_identity(label: str, value: str) -> str:
    if not re.fullmatch(SAFE_ID_PATTERN, value):
        raise ValueError(
            f"{label} must use 1-128 letters, digits, '.', '_' or '-'"
        )
    return value

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


class LocalWandbConfig(BaseModel):
    """Daemon-owned local W&B service configuration.

    This is deliberately optional and degradable: the daemon remains the
    source of truth when Docker/the W&B CLI is unavailable.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    managed: bool = True
    bind_host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    data_dir: str = "~/.local/state/ml-expd/wandb"
    image: str = "wandb/local"
    docker_executable: str = "/usr/bin/docker"
    container_uid: int = Field(default=999, ge=1, le=2**31 - 1)
    # A managed command must remain attached to the daemon. Empty selects the
    # packaged foreground Docker wrapper; replacements retain explicit
    # placeholders so ownership and storage remain reviewable.
    command: list[str] = Field(default_factory=list, max_length=128)
    environment_allowlist: list[str] = Field(default_factory=list, max_length=16)
    startup_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    external_url: Optional[str] = None
    publisher_entity: Optional[str] = None
    publisher_credential_ref: Optional[str] = None

    @field_validator("bind_host")
    @classmethod
    def _validate_bind_host(cls, value: str) -> str:
        host = value.strip()
        if not host or any(char in host for char in "/@?#") or any(char.isspace() for char in host):
            raise ValueError("bind_host must be a hostname or IP address")
        try:
            ipaddress.ip_address(host)
        except ValueError:
            if not re.fullmatch(r"[A-Za-z0-9.-]+", host):
                raise ValueError("bind_host must be a hostname or IP address")
        return host

    @field_validator("external_url")
    @classmethod
    def _validate_external_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("external_url must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("external_url must not contain user information")
        if parsed.query or parsed.fragment:
            raise ValueError("external_url must not contain a query or fragment")
        if parsed.scheme == "http":
            host = parsed.hostname or ""
            try:
                loopback = ipaddress.ip_address(host).is_loopback
            except ValueError:
                loopback = host.lower() == "localhost"
            if not loopback:
                raise ValueError("external_url must use HTTPS unless it is loopback")
        try:
            parsed.port
        except ValueError as exc:
            raise ValueError("external_url contains an invalid port") from exc
        return value.rstrip("/")

    @field_validator("environment_allowlist")
    @classmethod
    def _validate_environment_allowlist(cls, values: list[str]) -> list[str]:
        safe = {"PATH", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR", "DOCKER_HOST"}
        invalid = sorted(set(values) - safe)
        if invalid:
            raise ValueError(f"environment_allowlist contains unsupported names: {', '.join(invalid)}")
        return list(dict.fromkeys(values))

    @model_validator(mode="after")
    def _validate_lifecycle_contract(self) -> "LocalWandbConfig":
        if not self.enabled:
            return self
        if self.managed:
            if self.external_url is not None:
                raise ValueError("managed local W&B cannot use external_url")
            if self.bind_host not in {"127.0.0.1", "localhost", "::1"}:
                raise ValueError("managed local W&B must bind to a loopback host")
            if not re.fullmatch(r"[A-Za-z0-9_./:@-]+", self.image):
                raise ValueError("local W&B image contains unsupported characters")
            if not Path(self.docker_executable).is_absolute():
                raise ValueError("docker_executable must be an absolute path")
            if not self.command:
                self.command = [
                    sys.executable, "-m", "ml_exp_server.local_wandb_service",
                    "--bind-host", "{bind_host}", "--port", "{port}",
                    "--data-dir", "{data_dir}", "--image", "{image}",
                    "--docker", "{docker}",
                    "--container-uid", "{container_uid}",
                ]
            rendered = "\0".join(self.command)
            missing = [
                placeholder for placeholder in ("{bind_host}", "{port}", "{data_dir}")
                if placeholder not in rendered
            ]
            if missing:
                raise ValueError(
                    "managed local W&B command is missing placeholders: " + ", ".join(missing)
                )
        elif self.external_url is None:
            raise ValueError("external local W&B requires external_url")
        return self

    def url(self) -> str:
        if self.external_url:
            return self.external_url
        host = f"[{self.bind_host}]" if ":" in self.bind_host else self.bind_host
        return f"http://{host}:{self.port}"

    def data_path(self) -> Path:
        return Path(self.data_dir).expanduser().resolve()

    def resolved_command(self) -> list[str]:
        substitutions = {
            "{bind_host}": self.bind_host,
            "{port}": str(self.port),
            "{data_dir}": str(self.data_path()),
            "{image}": self.image,
            "{docker}": self.docker_executable,
            "{container_uid}": str(self.container_uid),
        }
        return [
            _replace_placeholders(token, substitutions)
            for token in self.command
        ]


def _replace_placeholders(value: str, substitutions: dict[str, str]) -> str:
    for key, replacement in substitutions.items():
        value = value.replace(key, replacement)
    return value


class WandbCloudConfig(BaseModel):
    """Cloud publication policy; credentials are referenced, never embedded."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    default_credential_ref: Optional[str] = None
    entity: Optional[str] = None
    api_url: str = "https://api.wandb.ai"
    dashboard_url: str = "https://wandb.ai"

    @model_validator(mode="after")
    def _validate_cloud_policy(self) -> "WandbCloudConfig":
        if self.enabled:
            if not self.default_credential_ref:
                raise ValueError(
                    "enabled W&B Cloud publication requires default_credential_ref"
                )
        for label, value in (
            ("api_url", self.api_url), ("dashboard_url", self.dashboard_url),
        ):
            parsed = urlsplit(value)
            if parsed.scheme != "https" or not parsed.hostname:
                raise ValueError(f"{label} must be an absolute HTTPS URL")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError(f"{label} must not contain credentials, query, or fragment")
        return self


class ObservabilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    log_archive_root: str = "~/.local/state/ml-expd/logs"
    credential_root: str = "~/.local/state/ml-expd/credentials"
    local_wandb: LocalWandbConfig = Field(default_factory=LocalWandbConfig)
    wandb_cloud: WandbCloudConfig = Field(default_factory=WandbCloudConfig)


class ActionRuntimeConfig(BaseModel):
    """Mutation boundary; every mutation class defaults closed."""

    model_config = ConfigDict(extra="forbid")

    allow_project_writes: bool = False
    allow_scheduler_mutations: bool = False
    allow_observability_mutations: bool = False
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

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        return _safe_identity("run_id", value)


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

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _safe_identity("campaign name", value)


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

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _safe_identity("research question id", value)


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

    @field_validator("project")
    @classmethod
    def validate_project(cls, value: str) -> str:
        return _safe_identity("project", value)

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

    @field_validator("project")
    @classmethod
    def validate_project(cls, value: str) -> str:
        return _safe_identity("project", value)


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
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
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

    def observability_db_path(self) -> Path:
        index = self.index_db_path()
        return index.with_name(f"{index.stem}.observability.sqlite")


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
    evaluation_snapshot: dict[str, Any] = Field(default_factory=dict)
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
