"""Result collection control and narrowly scoped training completion evidence."""
from __future__ import annotations

import asyncio
from functools import partial
import re
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from ..results.result_collection import ResultCollectionService
from ..application_errors import ApplicationError
from ..execution_queue import admitted_handoff
from .errors import application_http_error
from .errors import invoke
from .transfers import token

router = APIRouter(prefix="/api")
RESULT_PATH = re.compile(r"^/api/result-ready-transfers/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/attempt-[0-9]{3,}$")


class CollectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    retry: bool = False
    reconcile: bool = False


class ArchiveRepairRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@router.get("/runs/{project}/{run}/attempts/{attempt}/collection")
async def status(project: str, run: str, attempt: str, request: Request):
    return await invoke(ResultCollectionService(request.app.state.runtime).read, project, run, attempt)


@router.get("/runs/{project}/{run}/attempts/{attempt}/collection/diagnostics")
async def diagnostics(project: str, run: str, attempt: str, request: Request, refresh: bool = False):
    return await invoke(ResultCollectionService(request.app.state.runtime).diagnostics,
                        project, run, attempt, refresh=refresh)


@router.get("/runs/{project}/{run}/attempts/{attempt}/oss-archive/download")
async def oss_archive_download(project: str, run: str, attempt: str, request: Request):
    return await invoke(request.app.state.runtime.storage_service.archives.ticket, project, run, attempt)


@router.get("/runs/{project}/{run}/attempts/{attempt}/diagnostics")
async def training_diagnostics(project: str, run: str, attempt: str, request: Request, refresh: bool = False):
    return await invoke(ResultCollectionService(request.app.state.runtime).training_diagnostics,
                        project, run, attempt, refresh=refresh)


@router.post("/runs/{project}/{run}/attempts/{attempt}/collection/repair")
async def repair(project: str, run: str, attempt: str, data: ArchiveRepairRequest, request: Request):
    service = ResultCollectionService(request.app.state.runtime)
    future = request.app.state.submit_job(
        partial(service.repair, project, run, attempt), data.expected_sha256,
        executor=request.app.state.job_executors["collection"],
    )
    try:
        return await asyncio.shield(asyncio.wrap_future(future))
    except ApplicationError as exc:
        raise application_http_error(exc) from exc
    except (KeyError, ValueError, OSError) as exc:
        raise HTTPException(status_code=409, detail="operation is unavailable or its frozen identity is invalid") from exc


@router.post("/runs/{project}/{run}/attempts/{attempt}/collection")
async def collect(project: str, run: str, attempt: str, data: CollectionRequest, request: Request):
    service = ResultCollectionService(request.app.state.runtime)
    service.stopping = request.app.state._stop.is_set
    async def begin_and_enqueue():
        result, pending = await invoke(service.begin, project, run, attempt, retry=data.retry, reconcile=data.reconcile)
        if pending is not None:
            try:
                request.app.state.submit_job(service.finish, pending)
            except RuntimeError:
                await invoke(service.update, pending, status="RECONCILE_REQUIRED", diagnostic="EXECUTOR_UNAVAILABLE")
                raise HTTPException(status_code=503, detail="collection request was saved; reconcile before continuing") from None
        return result

    return await admitted_handoff(begin_and_enqueue)


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
