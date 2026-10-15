"""Authenticated data assets and exact-Attempt data/checkpoint capabilities."""
from __future__ import annotations

from pathlib import Path
import json

from fastapi import APIRouter, HTTPException, Request
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse, StreamingResponse

from ..results.artifacts import ArtifactService
from ..data.data_assets import AssetStore
from ..results.checkpoint_registry import CheckpointRegistry
from ..workers.persistent_state import MANIFEST_LIMIT
from ..application_errors import ApplicationError
from .errors import application_http_error
from .errors import invoke
from .transfers import receive, store, token

router = APIRouter(prefix="/api")


@router.get("/storage-limits")
async def limits(request: Request):
    return await invoke(store(request).limits)


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


@router.get("/projects/{project}/assets/{asset_id}/download")
async def asset_download_link(project: str, asset_id: str, request: Request):
    service = store(request)
    asset = await invoke(service.read, project, asset_id)
    if asset.get("remote_storage") in {"desktop-builder", "wyd", "sensecore-ccr"}:
        # Report the asset's frozen storage contract even without an S3 endpoint.
        await invoke(service.download, project, asset_id)
    if not service.objects.config.get("public_endpoint"):
        raise HTTPException(status_code=404, detail="direct downloads are not configured")
    return JSONResponse(await invoke(service.download, project, asset_id), headers={"Cache-Control": "no-store"})


@router.get("/asset-transfers/{project}/{run_id}/{attempt_id}/{asset_id}", include_in_schema=False)
async def worker_input(project: str, run_id: str, attempt_id: str, asset_id: str, request: Request):
    service = store(request)
    try:
        await run_in_threadpool(service.authorize_input, project, run_id, attempt_id, asset_id, token(request))
    except ApplicationError as error:
        raise application_http_error(error) from error
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
    except ApplicationError as error:
        raise application_http_error(error) from error
    except ValueError:
        raise HTTPException(status_code=401, detail="invalid checkpoint capability") from None
    return await receive(request, service, service.snapshot, project, run_id, attempt_id, capability)


@router.get("/runs/{project}/{run_id}/attempts/{attempt_id}/snapshots")
async def snapshots(project: str, run_id: str, attempt_id: str, request: Request):
    await invoke(ArtifactService(request.app.state.runtime).roots, project, run_id, attempt_id, restore=False)
    return await invoke(store(request).snapshots, project, run_id, attempt_id)


def checkpoint_store(request):
    assets = store(request)
    return CheckpointRegistry(Path(request.app.state.runtime.config.container_execution.artifact_store_file), assets.root.parent)


@router.put("/checkpoint-transfers/{project}/{run_id}/{attempt_id}", include_in_schema=False)
async def worker_checkpoint(project: str, run_id: str, attempt_id: str, request: Request):
    service = checkpoint_store(request)
    capability = token(request)
    try:
        transfer = await run_in_threadpool(service.objects.authorize, project, run_id, attempt_id, capability)
        if not transfer.get("checkpoint_state"):
            raise ValueError("checkpoint state is not enabled")
    except ApplicationError as error:
        raise application_http_error(error) from error
    except ValueError:
        raise HTTPException(status_code=401, detail="invalid persistent checkpoint capability") from None
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > MANIFEST_LIMIT:
            raise HTTPException(status_code=413, detail="checkpoint metadata exceeds limit")
    try:
        ready = json.loads(raw)
    except (ValueError, UnicodeError):
        raise HTTPException(status_code=422, detail="invalid checkpoint JSON") from None
    return await invoke(service.receive, project, run_id, attempt_id, capability, ready)


@router.get("/runs/{project}/{run_id}/attempts/{attempt_id}/checkpoints")
async def persistent_checkpoints(project: str, run_id: str, attempt_id: str, request: Request):
    await invoke(ArtifactService(request.app.state.runtime).roots, project, run_id, attempt_id, restore=False)
    return await invoke(checkpoint_store(request).list, project, run_id, attempt_id)


@router.get("/runs/{project}/{run_id}/attempts/{attempt_id}/checkpoints/{checkpoint_id}")
async def persistent_checkpoint(project: str, run_id: str, attempt_id: str, checkpoint_id: str, request: Request):
    await invoke(ArtifactService(request.app.state.runtime).roots, project, run_id, attempt_id, restore=False)
    return await invoke(checkpoint_store(request).read, project, run_id, attempt_id, checkpoint_id)
