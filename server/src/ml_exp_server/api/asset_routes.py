"""Authenticated data assets and exact-Attempt data/checkpoint capabilities."""
from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

from fastapi import APIRouter, HTTPException, Query, Request
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from ..artifacts import ArtifactService
from ..container_execution import ContainerExecutionService
from ..data_assets import AssetStore
from .container_routes import invoke

router = APIRouter(prefix="/api")


def store(request):
    runtime = request.app.state.runtime
    config = runtime.config.container_execution.artifact_store_file
    if not config:
        raise HTTPException(status_code=404, detail="data asset storage is not configured")
    return AssetStore(Path(config), runtime.config.project_registry_root_path())


async def receive(request, service, call, *args):
    raw_size = request.headers.get("Content-Length")
    try:
        expected_size = int(raw_size) if raw_size is not None else 0
        if expected_size < 0:
            raise ValueError()
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid Content-Length") from None
    if expected_size > service.limit:
        raise HTTPException(status_code=413, detail="data archive exceeds upload limit")
    if expected_size and expected_size * 2 + 64 * 1024 ** 2 > shutil.disk_usage(service.root).free:
        raise HTTPException(status_code=507, detail="insufficient local staging space for this data archive")
    size = 0
    with tempfile.TemporaryFile(dir=service.root) as stream:
        async for chunk in request.stream():
            size += len(chunk)
            if size > service.limit:
                raise HTTPException(status_code=413, detail="data archive exceeds upload limit")
            stream.write(chunk)
        stream.seek(0)
        return await invoke(call, *args, stream, size)


def token(request):
    scheme, _, value = request.headers.get("Authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not value:
        raise HTTPException(status_code=401, detail="invalid Attempt transfer capability")
    return value


@router.get("/storage-limits")
async def limits(request: Request):
    return await invoke(store(request).limits)


@router.post("/assets/archive")
async def upload_asset(request: Request,
                       project: str = Query(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"),
                       sha256: str = Query(pattern=r"^[0-9a-f]{64}$")):
    runtime = request.app.state.runtime
    await invoke(ContainerExecutionService(runtime).require_enabled)
    await invoke(runtime.project, project)
    service = store(request)
    # The archive's content digest is the idempotency key.
    return await receive(request, service, lambda p, data, size: service.receive(p, data, sha256, size), project)


@router.get("/projects/{project}/assets")
async def assets(project: str, request: Request):
    await invoke(request.app.state.runtime.project, project)
    return await invoke(store(request).list, project)


@router.get("/projects/{project}/assets/{asset_id}")
async def asset(project: str, asset_id: str, request: Request):
    return await invoke(store(request).read, project, asset_id)


async def stream_archive(service, project, asset_id):
    value, body = await invoke(service.archive, project, asset_id)
    return StreamingResponse(body.iter_chunks(chunk_size=1024 * 1024), media_type="application/x-tar",
                             headers={"Content-Length": str(value["archive_bytes"]), "ETag": '"' + value["sha256"] + '"'},
                             background=BackgroundTask(body.close))


@router.get("/projects/{project}/assets/{asset_id}/archive")
async def asset_archive(project: str, asset_id: str, request: Request):
    return await stream_archive(store(request), project, asset_id)


@router.get("/asset-transfers/{project}/{run_id}/{attempt_id}/{asset_id}", include_in_schema=False)
async def worker_input(project: str, run_id: str, attempt_id: str, asset_id: str, request: Request):
    service = store(request)
    try:
        await run_in_threadpool(service.authorize_input, project, run_id, attempt_id, asset_id, token(request))
    except (ValueError, OSError):
        raise HTTPException(status_code=401, detail="invalid input asset capability") from None
    return await stream_archive(service, project, asset_id)


@router.put("/snapshot-transfers/{project}/{run_id}/{attempt_id}", include_in_schema=False)
async def worker_snapshot(project: str, run_id: str, attempt_id: str, request: Request):
    service = store(request)
    capability = token(request)
    try:
        value = await run_in_threadpool(service.objects.authorize, project, run_id, attempt_id, capability)
        if not value.get("checkpoint_upload"):
            raise ValueError("checkpoint publication is not enabled")
    except ValueError:
        raise HTTPException(status_code=401, detail="invalid checkpoint capability") from None
    return await receive(request, service, service.snapshot, project, run_id, attempt_id, capability)


@router.get("/runs/{project}/{run_id}/attempts/{attempt_id}/snapshots")
async def snapshots(project: str, run_id: str, attempt_id: str, request: Request):
    await invoke(ArtifactService(request.app.state.runtime).roots, project, run_id, attempt_id)
    return await invoke(store(request).snapshots, project, run_id, attempt_id)
