"""Match requirements and prepare experiments through a durable server workflow."""
from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from ..experiment_preparation import ExperimentPreparationRequest, PreparationRun
from ..executor_capabilities import ExecutionRequirements, ExecutorSelector
from .container_routes import invoke


router = APIRouter(prefix="/api")


class MatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    run: PreparationRun
    requirements: ExecutionRequirements = Field(default_factory=ExecutionRequirements)
    executor_selector: ExecutorSelector | None = None


@router.post("/executors/match")
async def matching(data: MatchRequest, request: Request):
    result = await invoke(request.app.state.preparation_service.select, data.project, data.run, data.requirements, data.executor_selector)
    return {key: value for key, value in result.items() if key != "profile"}


async def enqueue(request, pending):
    service = request.app.state.preparation_service
    try:
        request.app.state.submit_job(service.finish, pending)
    except RuntimeError:
        await invoke(service.submission_failed, pending)
    return await invoke(service.read, *pending)


@router.post("/projects/{project}/experiment-preparations", status_code=202)
async def prepare(project: str, data: ExperimentPreparationRequest, request: Request):
    result, pending = await invoke(request.app.state.preparation_service.accept, project, data)
    return await enqueue(request, pending) if pending is not None else result


@router.get("/projects/{project}/experiment-preparations/{preparation_id}")
async def read(project: str, preparation_id: str, request: Request):
    return await invoke(request.app.state.preparation_service.read, project, preparation_id)


@router.get("/projects/{project}/experiment-preparations/{preparation_id}/progress")
async def progress(project: str, preparation_id: str, request: Request):
    return await invoke(request.app.state.preparation_service.progress, project, preparation_id)


@router.post("/projects/{project}/experiment-preparations/{preparation_id}/continue", status_code=202)
async def continue_preparation(project: str, preparation_id: str, request: Request):
    result, pending = await invoke(request.app.state.preparation_service.continue_preparation, project, preparation_id)
    return await enqueue(request, pending) if pending is not None else result
