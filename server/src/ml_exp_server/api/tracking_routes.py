"""Authenticated operator configuration and write-only worker telemetry."""
from __future__ import annotations

import json
import re

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from ..results.artifact_store import ArtifactStore
from ..tracking.tracking_contract import WandbOptions, WandbSettings
from ..tracking.tracking_service import store_for, normalized_metric, backfill
from ..wandb_exporter import default_entity
from .container_routes import invoke

router = APIRouter(prefix="/api")
RECORD_PATH = re.compile(r"^/api/record-transfers/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/attempt-[0-9]{3,}$")


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


@router.post("/record-transfers/{project}/{run}/{attempt}")
async def record_transfer(project: str, run: str, attempt: str, request: Request):
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 512 * 1024:
            raise HTTPException(status_code=413, detail="record batch exceeds size limit")
    try:
        runtime = request.app.state.runtime
        store = store_for(runtime)
        config = runtime.config.container_execution.artifact_store_file
        if not config:
            raise ValueError("worker transfer is unavailable")
        from pathlib import Path
        transfer = ArtifactStore(Path(config), runtime.config.project_registry_root_path())
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            raise ValueError("worker capability is required")
        capability = await invoke(transfer.authorize, project, run, attempt, auth[7:])
        if not capability.get("record_stream"):
            raise ValueError("record stream was not authorized")
        binding = store.binding(project, run)
        data = json.loads(body)
        if not isinstance(data, dict) or set(data) != {"records"} or not isinstance(data["records"], list) or not 1 <= len(data["records"]) <= 64:
            raise ValueError("invalid record batch")
        events = []
        for item in data["records"]:
            if (not isinstance(item, dict) or set(item) != {"event_id", "kind", "data"} or
                    not isinstance(item["event_id"], str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", item["event_id"]) or
                    not isinstance(item["data"], dict)):
                raise ValueError("invalid record")
            if item["kind"] == "metrics":
                payload = normalized_metric(item["data"], binding["config"])
            elif item["kind"] in {"lifecycle", "checkpoints"}:
                allowed = {"phase", "state", "exit_code", "pid", "diagnostic", "checkpoint_id", "step", "timestamp"}
                if set(item["data"]) - allowed or any(not isinstance(value, (str, int, type(None))) for value in item["data"].values()):
                    raise ValueError("invalid lifecycle record")
                payload = {"kind": item["kind"], "data": item["data"]}
            else:
                raise ValueError("invalid record kind")
            events.append((item["event_id"], payload))
        scope = store.scope(project, run, attempt)
        sequence = await invoke(store.append, scope, events)
        return {"accepted": len(events), "last_sequence": sequence}
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=409, detail="record batch is invalid or conflicts with its frozen identity") from exc
