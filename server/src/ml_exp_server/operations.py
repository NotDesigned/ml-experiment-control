"""Renderer-neutral catalog for daemon-owned code and experiment operations.

The catalog deliberately excludes analysis, recommendations, reports, charts,
goals, and model turns.  A client may use those capabilities to author an
``OperationIntent``; the daemon only validates, prepares, authorizes, and
executes the resulting mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .schemas import OperationScope, OperationScopeType


OperationMode = Literal["intent", "direct"]
AvailabilityStatus = Literal["AVAILABLE", "BLOCKED"]
ParameterKind = Literal["text", "number", "enum"]


@dataclass(frozen=True, slots=True)
class OperationParameter:
    key: str
    label: str
    kind: ParameterKind = "text"
    required: bool = True
    default: Any = None
    placeholder: str = ""
    positive: bool = False
    choices: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class OperationDefinition:
    operation_id: str
    label: str
    description: str
    category: str
    scopes: tuple[OperationScopeType, ...]
    mode: OperationMode
    expected_effect: str
    intent_kind: str | None
    parameters: tuple[OperationParameter, ...] = ()
    priority: int = 100

    def with_parameters(self, *parameters: OperationParameter) -> "OperationDefinition":
        return replace(self, parameters=tuple(parameters))


@dataclass(frozen=True, slots=True)
class OperationAvailability:
    operation: OperationDefinition
    scope: OperationScope
    status: AvailabilityStatus
    reasons: tuple[str, ...] = ()
    expected_effect: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.status == "AVAILABLE"


@dataclass(frozen=True, slots=True)
class OperationResult:
    operation_id: str
    scope: OperationScope
    message: str = ""
    action_id: str | None = None
    affected_identity: str | None = None


REQUEST = OperationParameter(
    "request", "Operation request", placeholder="Describe the requested change",
)
REASON = OperationParameter(
    "reason", "Reason", placeholder="Record why this operation is needed",
)
GPU_BUDGET = OperationParameter(
    "max_gpu_hours", "Maximum GPU-hours", kind="number", default=1.0, positive=True,
)
NEW_ATTEMPT_ID = OperationParameter(
    "new_attempt_id", "New Attempt ID", required=False,
    placeholder="Optional attempt-NNN identity",
)
WANDB_CLOUD_SYNC = OperationParameter(
    "wandb_cloud_sync", "Sync to W&B Cloud", kind="enum", default="no",
    choices=(("No (local archive only)", "no"), ("Yes (daemon credential)", "yes")),
)


OPERATIONS: tuple[OperationDefinition, ...] = (
    OperationDefinition(
        "question.create", "Create Research Question file",
        "Prepare a project-file write from a client-authored question definition.",
        "Project code", (OperationScopeType.PROJECT,), "intent",
        "Prepare a reviewable project-file Action", "CREATE_RESEARCH_QUESTION_DRAFT",
        (REQUEST,), 20,
    ),
    OperationDefinition(
        "campaign.create", "Create Campaign",
        "Prepare a Campaign file and catalog update from a client-authored definition.",
        "Project code", (OperationScopeType.PROJECT, OperationScopeType.RESEARCH_QUESTION),
        "intent", "Prepare a reviewable Campaign write", "CREATE_CAMPAIGN_DRAFT",
        (REQUEST,), 20,
    ),
    OperationDefinition(
        "campaign.update", "Update Campaign",
        "Prepare a new Campaign revision without deriving a scientific conclusion.",
        "Project code", (OperationScopeType.CAMPAIGN,), "intent",
        "Prepare a reviewable Campaign write", "UPDATE_CAMPAIGN_DRAFT", (REQUEST,), 20,
    ),
    OperationDefinition(
        "run.derive", "Derive Run definition",
        "Prepare a client-authored Campaign/Run definition with immutable identities.",
        "Project code", (OperationScopeType.CAMPAIGN, OperationScopeType.RUN), "intent",
        "Prepare a reviewable Campaign write", "DERIVE_RUN_DRAFT", (REQUEST,), 30,
    ),
    OperationDefinition(
        "object.archive", "Archive record",
        "Append an archive record while retaining immutable evidence.", "Lifecycle",
        (OperationScopeType.CAMPAIGN, OperationScopeType.RUN, OperationScopeType.ATTEMPT),
        "direct", "Prepare an archive Action", None, (REASON,), 40,
    ),
    OperationDefinition(
        "run.submit", "Submit first Attempt",
        "Prepare the first scheduler submission for an authored Run.", "Execution",
        (OperationScopeType.RUN,), "direct", "Prepare a scheduler Action", "SUBMIT_RUN",
        (GPU_BUDGET, WANDB_CLOUD_SYNC), 30,
    ),
    OperationDefinition(
        "attempt.retry", "Retry as new Attempt",
        "Prepare a retry without reusing an immutable Attempt identity.", "Execution",
        (OperationScopeType.ATTEMPT,), "direct", "Prepare a scheduler Action",
        "RETRY_ATTEMPT", (REASON, GPU_BUDGET, NEW_ATTEMPT_ID, WANDB_CLOUD_SYNC), 30,
    ),
    OperationDefinition(
        "attempt.cancel", "Cancel active Attempt",
        "Prepare cancellation of the exact observed backend job.", "Execution",
        (OperationScopeType.ATTEMPT,), "direct", "Prepare a scheduler Action",
        "CANCEL_RUN", (REASON,), 30,
    ),
    OperationDefinition(
        "run.evaluate", "Run evaluation",
        "Prepare an immutable evaluation-as-run supplied by the client.", "Execution",
        (OperationScopeType.RUN, OperationScopeType.ATTEMPT), "intent",
        "Prepare an evaluation scheduler Action", "RUN_EVALUATION",
        (REQUEST, GPU_BUDGET), 35,
    ),
)


OPERATIONS_BY_ID = {item.operation_id: item for item in OPERATIONS}

INTENT_SCOPES: dict[str, tuple[OperationScopeType, ...]] = {
    "CREATE_RESEARCH_QUESTION_DRAFT": (OperationScopeType.PROJECT,),
    "CREATE_CAMPAIGN_DRAFT": (
        OperationScopeType.PROJECT, OperationScopeType.RESEARCH_QUESTION,
    ),
    "UPDATE_CAMPAIGN_DRAFT": (OperationScopeType.CAMPAIGN,),
    "DERIVE_RUN_DRAFT": (OperationScopeType.CAMPAIGN, OperationScopeType.RUN),
    "SUBMIT_RUN": (OperationScopeType.RUN,),
    "RETRY_ATTEMPT": (OperationScopeType.ATTEMPT,),
    "CANCEL_RUN": (OperationScopeType.ATTEMPT,),
    "RUN_EVALUATION": (OperationScopeType.RUN, OperationScopeType.ATTEMPT),
    "ARCHIVE_CAMPAIGN": (OperationScopeType.CAMPAIGN,),
    "ARCHIVE_RUN": (OperationScopeType.RUN,),
    "ARCHIVE_ATTEMPT": (OperationScopeType.ATTEMPT,),
}

def intent_scope_error(kind: str, scope_type: OperationScopeType) -> str | None:
    allowed = INTENT_SCOPES.get(kind)
    if allowed is None:
        return f"unsupported intent kind {kind!r}"
    if scope_type in allowed:
        return None
    names = ", ".join(item.value for item in allowed)
    return f"intent kind {kind} is not valid in {scope_type.value} scope; expected {names}"


def operations_for_scope(scope_type: OperationScopeType) -> tuple[OperationDefinition, ...]:
    return tuple(sorted(
        (item for item in OPERATIONS if scope_type in item.scopes),
        key=lambda item: (item.priority, item.category, item.label),
    ))
