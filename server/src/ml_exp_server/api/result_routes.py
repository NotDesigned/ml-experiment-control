"""Result collection control and narrowly scoped training completion evidence."""
from __future__ import annotations

import re
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from ..result_collection import ResultCollectionService
from .container_routes import invoke
from .asset_routes import token

router = APIRouter(prefix="/api")
RESULT_PATH = re.compile(r"^/api/result-ready-transfers/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/attempt-[0-9]{3,}$")


class CollectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    retry: bool = False
    reconcile: bool = False


@router.get("/runs/{project}/{run}/attempts/{attempt}/collection")
async def status(project: str, run: str, attempt: str, request: Request):
    return await invoke(ResultCollectionService(request.app.state.runtime).read, project, run, attempt)


@router.post("/runs/{project}/{run}/attempts/{attempt}/collection")
async def collect(project: str, run: str, attempt: str, data: CollectionRequest, request: Request):
    service = ResultCollectionService(request.app.state.runtime)
    service.stopping = request.app.state._stop.is_set
    result, pending = await invoke(service.begin, project, run, attempt, retry=data.retry, reconcile=data.reconcile)
    if pending is not None:
        try:
            request.app.state.submit_job(service.finish, pending)
        except RuntimeError:
            await invoke(service.update, pending, status="RECONCILE_REQUIRED", diagnostic="EXECUTOR_UNAVAILABLE")
            raise HTTPException(status_code=503, detail="collection request was saved; reconcile before continuing") from None
    return result


@router.put("/result-ready-transfers/{project}/{run}/{attempt}")
async def ready(project: str, run: str, attempt: str, request: Request):
    service = ResultCollectionService(request.app.state.runtime)
    try:
        await invoke(service.objects.authorize, project, run, attempt, token(request))
    except HTTPException:
        raise HTTPException(status_code=401, detail="invalid Attempt transfer capability") from None
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > 4096:
            raise HTTPException(status_code=413, detail="training result exceeds size limit")
    import json
    try:
        value = json.loads(raw)
        return await invoke(service.register_training, project, run, attempt, token(request), value)
    except (ValueError, TypeError):
        raise HTTPException(status_code=422, detail="invalid training result") from None
