"""Authenticated preparation and exact data-copy callback capabilities."""
import json

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from ..data.data_delivery import DataDeliveryService
from .asset_routes import token
from .container_routes import ConfirmRequest, invoke

router = APIRouter(prefix="/api")


class PrepareData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    executor: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@router.post("/projects/{project}/assets/{asset_id}/deliveries/prepare")
async def prepare(project: str, asset_id: str, data: PrepareData, request: Request):
    return await invoke(DataDeliveryService(request.app.state.runtime).prepare, project, asset_id, data.executor)


@router.get("/projects/{project}/data-deliveries/{delivery_id}")
async def read(project: str, delivery_id: str, request: Request):
    return await invoke(DataDeliveryService(request.app.state.runtime).read, project, delivery_id)


async def start(project, delivery_id, data, request, reconcile):
    service = DataDeliveryService(request.app.state.runtime)
    pending = await invoke(service.begin, project, delivery_id, data.confirmation, reconcile=reconcile)
    if pending is not None:
        try:
            request.app.state.submit_job(service.finish, pending)
        except RuntimeError:
            await invoke(service.update, project, delivery_id, status="RECONCILE_REQUIRED", error="data preparation could not be queued")
            raise HTTPException(status_code=503, detail="data preparation executor is unavailable") from None
    return {"accepted": True, "delivery_id": delivery_id, "status": "EXECUTING" if pending is not None else "READY"}


@router.post("/projects/{project}/data-deliveries/{delivery_id}/execute", status_code=202)
async def execute(project: str, delivery_id: str, data: ConfirmRequest, request: Request):
    return await start(project, delivery_id, data, request, False)


@router.post("/projects/{project}/data-deliveries/{delivery_id}/reconcile", status_code=202)
async def reconcile(project: str, delivery_id: str, data: ConfirmRequest, request: Request):
    return await start(project, delivery_id, data, request, True)


@router.put("/data-copy-transfers/{project}/{delivery_id}", include_in_schema=False)
async def callback(project: str, delivery_id: str, request: Request):
    service = DataDeliveryService(request.app.state.runtime)
    capability = token(request)
    try:
        await run_in_threadpool(service.authorize, project, delivery_id, capability)
    except ValueError:
        raise HTTPException(status_code=401, detail="invalid data-copy capability") from None
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 16384:
            raise HTTPException(status_code=413, detail="data-copy receipt is too large")
    try:
        receipt = json.loads(body)
        if not isinstance(receipt, dict):
            raise ValueError()
        return await invoke(service.callback, project, delivery_id, capability, receipt)
    except (ValueError, UnicodeError):
        raise HTTPException(status_code=422, detail="invalid data-copy receipt") from None
