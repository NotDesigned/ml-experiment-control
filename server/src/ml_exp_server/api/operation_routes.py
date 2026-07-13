"""HTTP adapters for the scoped research-operation catalog."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ..application import ApplicationError
from ..schemas import AgentScopeType
from .errors import application_http_error


router = APIRouter(prefix="/api/operations")


class OperationInvokeRequest(BaseModel):
    project: str = Field(min_length=1, max_length=256)
    scope_type: AgentScopeType
    object_id: str = Field(min_length=1, max_length=512)
    operation_id: str = Field(min_length=1, max_length=128)
    parameters: dict[str, Any] = Field(default_factory=dict)


@router.get("")
def operation_availability(
    request: Request, project: str, scope_type: AgentScopeType, object_id: str,
):
    try:
        return request.app.state.application.operation_availability(
            project, scope_type, object_id,
        )
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.post("/direct")
async def invoke_direct_operation(data: OperationInvokeRequest, request: Request):
    try:
        return await run_in_threadpool(
            request.app.state.application.invoke_direct_operation,
            data.operation_id, data.project, data.scope_type, data.object_id,
            data.parameters,
        )
    except ApplicationError as exc:
        raise application_http_error(exc) from exc
