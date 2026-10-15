"""Exact CPU-script receipt callbacks with an independent short-lived capability."""
import json

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from ..backends.sensecore_preparation import SenseCoreScriptPreparation
from .errors import invoke
from .transfers import token

router = APIRouter(prefix="/api")


@router.put("/data-script-transfers/{project}/{operation}", include_in_schema=False)
async def callback(project: str, operation: str, request: Request):
    runtime = request.app.state.runtime
    config = runtime.config.container_execution.artifact_store_file
    if not config:
        raise HTTPException(status_code=404, detail="script preparation is not configured")
    service = SenseCoreScriptPreparation(config, runtime.config.project_registry_root_path())
    capability = token(request)
    try:
        await run_in_threadpool(service.authorize_callback, project, operation, capability)
    except ValueError:
        raise HTTPException(status_code=401, detail="invalid or expired script receipt capability") from None
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > 8 * 1024 ** 2 + 16384:
            raise HTTPException(status_code=413, detail="script receipt exceeds size limit")
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError()
    except (ValueError, UnicodeError):
        raise HTTPException(status_code=422, detail="invalid script receipt JSON") from None
    return await invoke(service.callback, project, operation, capability, payload)
