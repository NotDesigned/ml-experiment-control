"""HTTP adapter for first-class experiment submissions."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ..application import ApplicationError
from .errors import application_http_error


router = APIRouter(prefix="/api")


class PrepareSubmissionRequest(BaseModel):
    max_gpu_hours: float = Field(gt=0)
    reason: str = Field(default="", max_length=4000)


class AuthorizeSubmissionRequest(BaseModel):
    note: str = Field(default="", max_length=4000)


class ExecuteSubmissionRequest(BaseModel):
    confirmation: str = Field(min_length=1, max_length=100)


def _service(request: Request):
    return request.app.state.submission_service


@router.get("/experiments/{project}/{run_id}/submissions")
def list_submissions(project: str, run_id: str, request: Request):
    try:
        return _service(request).list(project, run_id)
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.post("/experiments/{project}/{run_id}/submissions/prepare")
async def prepare_submission(
    project: str, run_id: str, data: PrepareSubmissionRequest, request: Request,
):
    try:
        return await run_in_threadpool(
            _service(request).prepare_first_attempt,
            project, run_id,
            max_gpu_hours=data.max_gpu_hours,
            reason=data.reason,
        )
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.get("/submissions/{submission_id}")
def get_submission(submission_id: str, request: Request):
    try:
        return _service(request).get(submission_id)
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.post("/submissions/{submission_id}/authorize")
def authorize_submission(
    submission_id: str, data: AuthorizeSubmissionRequest, request: Request,
):
    try:
        return _service(request).authorize(submission_id, data.note)
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.post("/submissions/{submission_id}/execute")
async def execute_submission(
    submission_id: str, data: ExecuteSubmissionRequest, request: Request,
):
    try:
        return await run_in_threadpool(
            _service(request).execute, submission_id, data.confirmation,
        )
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.post("/submissions/{submission_id}/reconcile")
async def reconcile_submission(submission_id: str, request: Request):
    try:
        return await run_in_threadpool(
            _service(request).reconcile, submission_id,
        )
    except ApplicationError as exc:
        raise application_http_error(exc) from exc
