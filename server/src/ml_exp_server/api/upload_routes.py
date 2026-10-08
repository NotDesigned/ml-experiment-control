"""Resumable client assets and narrowly authorized worker archive uploads."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import shutil
import tempfile
import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from ..container_execution import ContainerExecutionService
from ..data.multipart_upload import UploadStore
from ..archive_limits import minimum_limit
from ..data.remote_data_uploads import remote_uploads
from .asset_routes import store, token
from .container_routes import invoke

router = APIRouter(prefix="/api")
WORKER_UPLOAD_PATH = re.compile(r"^/api/attempt-uploads/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/attempt-[0-9]{3,}/(?:artifacts|checkpoint)(?:/upload\.[0-9a-f]{64}(?:/complete|/parts/[0-9]+)?)?$")


class UploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(gt=0)


async def target(request, project, run=None, attempt=None, kind="asset"):
    service = store(request)
    binding = {"project": project, "kind": kind}
    if run is None:
        await invoke(ContainerExecutionService(request.app.state.runtime).require_enabled)
        await invoke(request.app.state.runtime.project, project)
    else:
        try:
            value = await run_in_threadpool(service.objects.authorize, project, run, attempt, token(request))
            if kind == "checkpoint" and not value.get("checkpoint_upload"):
                raise ValueError("checkpoint publication is not enabled")
        except ValueError:
            raise HTTPException(status_code=401, detail="invalid Attempt transfer capability") from None
        binding.update(run_id=run, attempt_id=attempt)
    uploads = remote_uploads(service) if kind == "asset" else None
    if uploads is None:
        uploads = UploadStore(service.root.parent, service.objects.config)
    return service, uploads, binding


async def create(context, data):
    service, uploads, binding = context
    limit = service.objects.limit if binding["kind"] == "artifacts" else minimum_limit(service.limit, service.objects.limit) if binding["kind"] == "checkpoint" else service.limit
    return await invoke(uploads.create, binding, data.sha256, data.bytes, limit)


async def read(context, upload_id):
    _, uploads, binding = context
    return await invoke(uploads.read, upload_id, binding)


async def part(context, upload_id, number, digest, request):
    _, uploads, binding = context
    value = await read(context, upload_id)
    expected = await invoke(uploads.part_size, value, number)
    raw = request.headers.get("Content-Length")
    if raw is not None and raw != str(expected):
        raise HTTPException(status_code=400, detail="upload part Content-Length differs")
    if getattr(uploads, "remote", False):
        slots = getattr(request.app.state, "data_upload_slots", None)
        if slots is None:
            slots = request.app.state.data_upload_slots = asyncio.Semaphore(4)
        async with slots:
            data = bytearray()
            async for chunk in request.stream():
                data.extend(chunk)
                if len(data) > expected:
                    raise HTTPException(status_code=413, detail="upload part exceeds its declared size")
            if len(data) != expected or hashlib.sha256(data).hexdigest() != digest:
                raise HTTPException(status_code=400, detail="upload part length or SHA256 differs")
            return await invoke(uploads.part, upload_id, binding, number, bytes(data), digest, expected)
    if expected + 64 * 1024 ** 2 > shutil.disk_usage(uploads.root).free:
        raise HTTPException(status_code=507, detail="insufficient upload staging space")
    with tempfile.NamedTemporaryFile(dir=uploads.root, prefix=".part-", delete=False) as temporary:
        filename = Path(temporary.name)
        try:
            sha, size = hashlib.sha256(), 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > expected:
                    raise HTTPException(status_code=413, detail="upload part exceeds its declared size")
                sha.update(chunk)
                temporary.write(chunk)
            temporary.flush()
            if size != expected or sha.hexdigest() != digest:
                raise HTTPException(status_code=400, detail="upload part length or SHA256 differs")
            # Receiving bytes never holds the session lock or blocks another
            # request on flock. Only committing the validated part takes it.
            return await invoke(uploads.part, upload_id, binding, number, filename, digest, size)
        finally:
            filename.unlink(missing_ok=True)


async def complete(context, upload_id, request):
    service, uploads, binding = context
    value = await read(context, upload_id)
    if getattr(uploads, "remote", False):
        return await invoke(uploads.complete, upload_id, binding)
    if binding["kind"] == "asset":
        publish = lambda stream, size, digest: service.receive(binding["project"], stream, digest, size)
        expanded = service.expanded_limit
    else:
        capability = token(request)
        args = binding["project"], binding["run_id"], binding["attempt_id"], capability
        if binding["kind"] == "artifacts":
            publish = lambda stream, size, digest: service.objects.receive(*args, stream, size)
            expanded = service.objects.limit
        else:
            publish = lambda stream, size, digest: service.snapshot(*args, stream, size)
            expanded = service.expanded_limit
    # The seekable part stream needs no concatenated archive. Keep room for
    # validation and the published object, which may share this filesystem.
    return await invoke(uploads.complete, upload_id, binding, publish, min(expanded, value["bytes"]) + value["bytes"] + 64 * 1024 ** 2 if expanded is not None else value["bytes"] * 2 + 64 * 1024 ** 2)


async def abort(context, upload_id):
    _, uploads, binding = context
    return await invoke(uploads.abort, upload_id, binding)


@router.post("/projects/{project}/asset-uploads")
async def asset_create(project: str, data: UploadRequest, request: Request):
    return await create(await target(request, project), data)


@router.get("/projects/{project}/asset-uploads/{upload_id}")
async def asset_read(project: str, upload_id: str, request: Request):
    return await read(await target(request, project), upload_id)


@router.put("/projects/{project}/asset-uploads/{upload_id}/parts/{number}")
async def asset_part(project: str, upload_id: str, number: int, request: Request, sha256: str = Query(pattern=r"^[0-9a-f]{64}$")):
    return await part(await target(request, project), upload_id, number, sha256, request)


@router.post("/projects/{project}/asset-uploads/{upload_id}/complete")
async def asset_complete(project: str, upload_id: str, request: Request):
    return await complete(await target(request, project), upload_id, request)


@router.delete("/projects/{project}/asset-uploads/{upload_id}")
async def asset_abort(project: str, upload_id: str, request: Request):
    return await abort(await target(request, project), upload_id)


@router.post("/attempt-uploads/{project}/{run}/{attempt}/{kind}")
async def worker_create(project: str, run: str, attempt: str, kind: Literal["artifacts", "checkpoint"], data: UploadRequest, request: Request):
    value = await create(await target(request, project, run, attempt, kind), data)
    return {k: v for k, v in value.items() if k != "result"}


@router.get("/attempt-uploads/{project}/{run}/{attempt}/{kind}/{upload_id}")
async def worker_read(project: str, run: str, attempt: str, kind: Literal["artifacts", "checkpoint"], upload_id: str, request: Request):
    value = await read(await target(request, project, run, attempt, kind), upload_id)
    return {k: v for k, v in value.items() if k != "result"}


@router.put("/attempt-uploads/{project}/{run}/{attempt}/{kind}/{upload_id}/parts/{number}")
async def worker_part(project: str, run: str, attempt: str, kind: Literal["artifacts", "checkpoint"], upload_id: str, number: int, request: Request, sha256: str = Query(pattern=r"^[0-9a-f]{64}$")):
    return await part(await target(request, project, run, attempt, kind), upload_id, number, sha256, request)


@router.post("/attempt-uploads/{project}/{run}/{attempt}/{kind}/{upload_id}/complete")
async def worker_complete(project: str, run: str, attempt: str, kind: Literal["artifacts", "checkpoint"], upload_id: str, request: Request):
    value = await complete(await target(request, project, run, attempt, kind), upload_id, request)
    return {"status": "COMPLETED", "sha256": value["sha256"]}


@router.delete("/attempt-uploads/{project}/{run}/{attempt}/{kind}/{upload_id}")
async def worker_abort(project: str, run: str, attempt: str, kind: Literal["artifacts", "checkpoint"], upload_id: str, request: Request):
    return await abort(await target(request, project, run, attempt, kind), upload_id)
