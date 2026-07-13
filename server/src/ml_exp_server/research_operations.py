"""Renderer-neutral catalog and read model for scoped research operations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .schemas import AgentScope, AgentScopeType


OperationMode = Literal["analysis", "agent_proposal", "direct_proposal"]
AvailabilityStatus = Literal["AVAILABLE", "BLOCKED"]
ParameterKind = Literal["text", "number", "enum"]


@dataclass(frozen=True, slots=True)
class OperationParameter:
    key: str
    label: str
    kind: ParameterKind = "text"
    required: bool = True
    default: Any = None
    choices: tuple[str, ...] = ()
    placeholder: str = ""
    positive: bool = False


@dataclass(frozen=True, slots=True)
class ResearchOperation:
    operation_id: str
    label: str
    description: str
    category: str
    scopes: tuple[AgentScopeType, ...]
    mode: OperationMode
    expected_effect: str
    proposal_kind: str | None = None
    parameters: tuple[OperationParameter, ...] = ()
    priority: int = 100

    def with_parameters(self, *parameters: OperationParameter) -> "ResearchOperation":
        return replace(self, parameters=tuple(parameters))


@dataclass(frozen=True, slots=True)
class OperationAvailability:
    operation: ResearchOperation
    scope: AgentScope
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
    scope: AgentScope
    message: str = ""
    proposal_ids: tuple[str, ...] = ()
    open_inbox: bool = False
    affected_identity: str | None = None


REQUEST = OperationParameter(
    "request", "Research request", placeholder="Describe the evidence question or change",
)
REASON = OperationParameter("reason", "Reason", placeholder="Record why this operation is needed")
GPU_BUDGET = OperationParameter(
    "max_gpu_hours", "Maximum GPU-hours", kind="number", default=1.0, positive=True,
)
NEW_ATTEMPT_ID = OperationParameter(
    "new_attempt_id", "New Attempt ID", required=False,
    placeholder="Optional attempt-NNN identity",
)
OUTCOME = OperationParameter(
    "outcome", "Outcome", kind="enum",
    choices=("SUPPORTED", "REFUTED", "INCONCLUSIVE"), default="INCONCLUSIVE",
)
ASSESSMENT = OperationParameter(
    "assessment", "Evidence-bound assessment", placeholder="Summarize the reviewed evidence",
)


OPERATIONS: tuple[ResearchOperation, ...] = (
    ResearchOperation(
        "research.recommend", "Recommend next step",
        "Analyze bounded evidence and rank up to three safe next steps.", "Research",
        tuple(AgentScopeType), "analysis", "Append analysis and any executable proposals to Chat",
        priority=10,
    ),
    ResearchOperation(
        "question.create", "Create Research Question",
        "Draft a new organizational research question from the Project evidence.", "Design",
        (AgentScopeType.PROJECT,), "agent_proposal", "Create a reviewable ResearchQuestion draft",
        "CREATE_RESEARCH_QUESTION_DRAFT", (REQUEST,), 20,
    ),
    ResearchOperation(
        "campaign.create", "Create Campaign",
        "Design the smallest decisive Campaign for this scope.", "Design",
        (AgentScopeType.PROJECT, AgentScopeType.RESEARCH_QUESTION), "agent_proposal",
        "Create a reviewable Campaign draft", "CREATE_CAMPAIGN_DRAFT", (REQUEST,), 20,
    ),
    ResearchOperation(
        "campaign.update", "Revise Campaign",
        "Replace the current Campaign contract and memberships after semantic review.", "Design",
        (AgentScopeType.CAMPAIGN,), "agent_proposal", "Create a new Campaign revision",
        "UPDATE_CAMPAIGN_DRAFT", (REQUEST,), 20,
    ),
    ResearchOperation(
        "run.derive", "Derive Run in new Campaign",
        "Draft a new Campaign containing an immutable Run variant with one controlled change.", "Design",
        (AgentScopeType.CAMPAIGN, AgentScopeType.RUN), "agent_proposal",
        "Create a reviewable derived Run/Campaign draft", "DERIVE_RUN_DRAFT", (REQUEST,), 30,
    ),
    ResearchOperation(
        "campaign.complete", "Complete Campaign",
        "Freeze an evidence-bound scientific outcome for a completable Campaign.", "Lifecycle",
        (AgentScopeType.CAMPAIGN,), "direct_proposal", "Create an immutable completion record",
        "COMPLETE_CAMPAIGN", (OUTCOME, ASSESSMENT), 30,
    ),
    ResearchOperation(
        "object.archive", "Archive record",
        "Append an archive record while retaining immutable evidence.", "Lifecycle",
        (AgentScopeType.CAMPAIGN, AgentScopeType.RUN, AgentScopeType.ATTEMPT),
        "direct_proposal", "Hide the object from active workflows without deleting evidence",
        None, (REASON,), 40,
    ),
    ResearchOperation(
        "run.submit", "Submit first Attempt",
        "Create the first scheduler submission for an authored Run.", "Execution",
        (AgentScopeType.RUN,), "direct_proposal", "Create a new submitted Attempt identity",
        "SUBMIT_RUN", (GPU_BUDGET,), 30,
    ),
    ResearchOperation(
        "attempt.retry", "Retry as new Attempt",
        "Retry a failed execution without reusing its immutable Attempt identity.", "Execution",
        (AgentScopeType.ATTEMPT,), "direct_proposal", "Create the next Attempt identity",
        "RETRY_ATTEMPT", (REASON, GPU_BUDGET, NEW_ATTEMPT_ID), 30,
    ),
    ResearchOperation(
        "attempt.cancel", "Cancel active Attempt",
        "Cancel the exact observed backend job for an active Attempt.", "Execution",
        (AgentScopeType.ATTEMPT,), "direct_proposal", "Create an exact-job cancellation proposal",
        "CANCEL_RUN", (REASON,), 30,
    ),
    ResearchOperation(
        "run.evaluate", "Run evaluation",
        "Draft an immutable evaluation-as-run bound to a checkpoint and evaluation spec.",
        "Evaluation", (AgentScopeType.RUN, AgentScopeType.ATTEMPT), "agent_proposal",
        "Create a reviewable evaluation Run proposal", "RUN_EVALUATION",
        (REQUEST, GPU_BUDGET), 35,
    ),
    ResearchOperation(
        "report.generate", "Generate evidence report",
        "Generate a report bound to current exact objects, evidence and uncertainty.", "Evidence",
        (AgentScopeType.RESEARCH_QUESTION, AgentScopeType.CAMPAIGN,
         AgentScopeType.RUN, AgentScopeType.ATTEMPT),
        "agent_proposal", "Create a non-executing evidence-linked report",
        "CREATE_REPORT_DRAFT", (REQUEST,), 50,
    ),
    ResearchOperation(
        "chart.generate", "Generate chart specification",
        "Specify an evidence-linked chart without inventing missing measurements.", "Evidence",
        (AgentScopeType.RESEARCH_QUESTION, AgentScopeType.CAMPAIGN,
         AgentScopeType.RUN, AgentScopeType.ATTEMPT),
        "agent_proposal", "Create a non-executing chart specification",
        "CREATE_CHART_SPEC", (REQUEST,), 60,
    ),
)


OPERATIONS_BY_ID = {item.operation_id: item for item in OPERATIONS}

PROPOSAL_SCOPES: dict[str, tuple[AgentScopeType, ...]] = {
    operation.proposal_kind: operation.scopes
    for operation in OPERATIONS if operation.proposal_kind
}
PROPOSAL_SCOPES.update({
    "ARCHIVE_CAMPAIGN": (AgentScopeType.CAMPAIGN,),
    "ARCHIVE_RUN": (AgentScopeType.RUN,),
    "ARCHIVE_ATTEMPT": (AgentScopeType.ATTEMPT,),
    "ANALYSIS_ONLY": tuple(AgentScopeType),
})

PROPOSAL_OPERATION_IDS: dict[str, str] = {
    operation.proposal_kind: operation.operation_id
    for operation in OPERATIONS if operation.proposal_kind
}
PROPOSAL_OPERATION_IDS.update({
    "ARCHIVE_CAMPAIGN": "object.archive",
    "ARCHIVE_RUN": "object.archive",
    "ARCHIVE_ATTEMPT": "object.archive",
})


def proposal_scope_error(kind: str, scope_type: AgentScopeType) -> str | None:
    allowed = PROPOSAL_SCOPES.get(kind)
    if allowed is None:
        return None  # non-catalog legacy proposals keep their existing behavior
    if scope_type in allowed:
        return None
    names = ", ".join(item.value for item in allowed)
    return f"proposal kind {kind} is not valid in {scope_type.value} scope; expected {names}"


def operations_for_scope(scope_type: AgentScopeType) -> tuple[ResearchOperation, ...]:
    return tuple(sorted(
        (item for item in OPERATIONS if scope_type in item.scopes),
        key=lambda item: (item.priority, item.category, item.label),
    ))
