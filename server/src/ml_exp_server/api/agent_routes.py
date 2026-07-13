"""Object-scoped research-agent APIs. Phase 2 never executes proposals."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ..agent_protocol import AgentProposal
from ..application import ApplicationError, compact_evidence
from ..schemas import AgentScopeType
from .errors import application_http_error


router = APIRouter(prefix="/api/agent")


def _application(request: Request):
    return request.app.state.application


class ScopeRequest(BaseModel):
    project: str
    scope_type: AgentScopeType
    object_id: str = Field(min_length=1, max_length=512)


class GoalRequest(ScopeRequest):
    goal: str = Field(min_length=1, max_length=8000)


class BeginTurnRequest(ScopeRequest):
    message: str = Field(min_length=1, max_length=12000)
    enforce_operation_availability: bool = False


class ClaimTurnRequest(ScopeRequest):
    client_id: str = Field(min_length=1, max_length=200)
    provider: str = Field(default="openai_agents", min_length=1, max_length=100)


class ClaimNextRequest(BaseModel):
    client_id: str = Field(min_length=1, max_length=200)
    provider: str = Field(default="openai_agents", min_length=1, max_length=100)
    project: str | None = Field(default=None, max_length=200)


class TurnResultRequest(ScopeRequest):
    status: Literal["COMPLETED", "FAILED"]
    client_id: str = Field(min_length=1, max_length=200)
    session_id: str | None = Field(default=None, max_length=1000)
    message: str = Field(default="", max_length=100_000)
    proposals: list[AgentProposal] = Field(default_factory=list, max_length=3)
    evidence_digest: str = Field(min_length=1, max_length=200)
    error: str = Field(default="", max_length=1000)


class ProposalDecisionRequest(ScopeRequest):
    proposal_id: str = Field(pattern=r"^proposal-[a-f0-9]{12}$")
    decision: str
    note: str = Field(default="", max_length=4000)


def _compact_evidence(value: Any, *, depth: int = 0) -> Any:
    """Compatibility export for callers that used the former route helper."""
    return compact_evidence(value, depth=depth)


@router.get("")
def agent_snapshot(
    request: Request, project: str, scope_type: AgentScopeType, object_id: str,
):
    try:
        return _application(request).agent_snapshot(project, scope_type, object_id)
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.get("/proposal")
def proposal(
    request: Request, project: str, scope_type: AgentScopeType,
    object_id: str, proposal_id: str,
):
    try:
        return _application(request).proposal_show(
            project, scope_type, object_id, proposal_id,
        )
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.put("/goal")
def update_goal(data: GoalRequest, request: Request):
    try:
        return _application(request).update_agent_goal(
            data.project, data.scope_type, data.object_id, data.goal,
        )
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.post("/proposal-decision")
def decide_proposal(data: ProposalDecisionRequest, request: Request):
    try:
        return _application(request).decide_proposal(
            data.project, data.scope_type, data.object_id, data.proposal_id,
            data.decision, data.note,
        )
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.post("/turns")
def begin_turn(data: BeginTurnRequest, request: Request):
    try:
        return _application(request).begin_agent_turn(
            data.project, data.scope_type, data.object_id, data.message,
            enforce_operation_availability=data.enforce_operation_availability,
        )
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.post("/turns/{request_id}/claim")
def claim_turn(request_id: str, data: ClaimTurnRequest, request: Request):
    try:
        return _application(request).claim_agent_turn(
            data.project, data.scope_type, data.object_id, request_id,
            client_id=data.client_id, provider=data.provider,
        )
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.post("/turns/claim-next")
def claim_next_turn(data: ClaimNextRequest, request: Request):
    try:
        context = _application(request).claim_next_agent_turn(
            client_id=data.client_id, provider=data.provider, project=data.project,
        )
        return {"turn_context": context}
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.post("/turns/{request_id}/result")
def complete_turn(request_id: str, data: TurnResultRequest, request: Request):
    try:
        return _application(request).complete_agent_turn(
            data.project, data.scope_type, data.object_id, request_id,
            status=data.status, session_id=data.session_id, message=data.message,
            proposals=[item.model_dump(mode="json") for item in data.proposals],
            evidence_digest_value=data.evidence_digest, client_id=data.client_id,
            error=data.error,
        )
    except ApplicationError as exc:
        raise application_http_error(exc) from exc
