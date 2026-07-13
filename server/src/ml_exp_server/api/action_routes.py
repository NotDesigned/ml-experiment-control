"""Phase-3 controlled action APIs; approval and execution are separate calls."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ..application import ApplicationError
from ..intent_protocol import OperationIntent
from ..schemas import OperationScopeType
from .errors import application_http_error


router = APIRouter(prefix="/api/actions")


class ScopeRequest(BaseModel):
    project: str = Field(min_length=1, max_length=256)
    scope_type: OperationScopeType
    object_id: str = Field(min_length=1, max_length=512)


class PrepareActionRequest(ScopeRequest):
    intent: OperationIntent


class ActionRequest(BaseModel):
    action_id: str = Field(pattern=r"^action-[a-f0-9]{16}$")


class AuthorizeActionRequest(ActionRequest):
    note: str = Field(default="", max_length=4000)


class ExecuteActionRequest(ActionRequest):
    confirmation: str = Field(min_length=1, max_length=100)


@router.get("/policy")
def action_policy(request: Request):
    return request.app.state.config.action_runtime.model_dump()


@router.get("")
def list_actions(request: Request, project: str, scope_type: OperationScopeType,
                 object_id: str):
    try:
        return request.app.state.application.list_actions(project, scope_type, object_id)
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.post("/prepare")
async def prepare_action(data: PrepareActionRequest, request: Request):
    try:
        return await run_in_threadpool(
            request.app.state.application.prepare_action,
            data.project, data.scope_type, data.object_id,
            data.intent.model_dump(mode="json"),
        )
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.post("/authorize")
def authorize_action(data: AuthorizeActionRequest, request: Request):
    try:
        return request.app.state.application.authorize_action(data.action_id, data.note)
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.post("/execute")
async def execute_action(data: ExecuteActionRequest, request: Request):
    try:
        result = await run_in_threadpool(
            request.app.state.application.execute_action,
            data.action_id, data.confirmation,
        )
    except ApplicationError as exc:
        raise application_http_error(exc) from exc
    return result


@router.post("/reconcile")
async def reconcile_action(data: ActionRequest, request: Request):
    try:
        return await run_in_threadpool(
            request.app.state.application.reconcile_action, data.action_id,
        )
    except ApplicationError as exc:
        raise application_http_error(exc) from exc
