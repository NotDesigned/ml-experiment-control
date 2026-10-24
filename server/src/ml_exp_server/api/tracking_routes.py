"""Authenticated configuration and status for optional W&B publication."""
from __future__ import annotations

from fastapi import APIRouter, Request

from ..tracking.tracking_contract import WandbOptions, WandbSettings
from ..tracking.tracking_service import store_for, backfill
from ..wandb_exporter import default_entity
from .errors import invoke

router = APIRouter(prefix="/api")


@router.get("/tracking/wandb")
async def settings(request: Request):
    return await invoke(store_for(request.app.state.runtime).public_settings)


@router.put("/tracking/wandb")
async def configure(data: WandbSettings, request: Request):
    result = await invoke(store_for(request.app.state.runtime).configure, data, default_entity)
    if not result["credentials_configured"] or not result["enabled"]:
        request.app.state.tracking_publisher.publisher.close()
    return result


@router.get("/runs/{project}/{run}/tracking")
async def status(project: str, run: str, request: Request):
    store = store_for(request.app.state.runtime)
    result = await invoke(store.status, project, run)
    result["preparation"] = (await invoke(store.status, project, run + "--preparation"))["scopes"]
    result["backfill"] = await invoke(store.status, project, run + "--backfill")
    return result


@router.post("/runs/{project}/{run}/tracking/backfill")
async def publish_history(project: str, run: str, data: WandbOptions, request: Request):
    return await invoke(backfill, request.app.state.runtime, project, run, data)


@router.post("/runs/{project}/{run}/tracking/retry")
async def retry(project: str, run: str, request: Request):
    store = store_for(request.app.state.runtime)
    for candidate in (run, run + "--preparation", run + "--backfill"):
        for scope in store.pending():
            if (scope["project"], scope["run"]) == (project, candidate):
                store.outcome(scope["id"], error=None)
    return await status(project, run, request)
