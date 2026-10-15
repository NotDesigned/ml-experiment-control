"""Resumable client assets and narrowly authorized worker archive uploads."""
from __future__ import annotations

import hashlib
import re
import shutil
import asyncio
from contextlib import nullcontext
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from ..container_execution import ContainerExecutionService
from ..storage_service import StorageService
from ..archive_limits import minimum_limit
from ..data.remote_data_uploads import remote_uploads, freeze_target
from ..data.transfer_buffer import TransferBuffer
from ..application_errors import ApplicationError
from .transfers import store, token
from .errors import invoke
from .errors import application_http_error

router = APIRouter(prefix="/api")
WORKER_UPLOAD_PATH = re.compile(r"^/api/attempt-uploads/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/attempt-[0-9]{3,}/(?:artifacts|checkpoint)(?:/upload\.[0-9a-f]{64}(?:/complete|/parts/[0-9]+)?)?$")


class UploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(gt=0)
    target: "UploadTarget | None" = None
    files: "list[UploadFile] | None" = Field(default=None, max_length=20000)
    transport_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class UploadTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    executor: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    storage_scope: str | None = Field(default=None, pattern=r"^storage\.[0-9a-f]{64}$")


class UploadFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=4096)
    bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


UploadRequest.model_rebuild()


async def target(request, project, run=None, attempt=None, kind="asset", *, upload_id=None, data=None):
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
        except ApplicationError as error:
            raise application_http_error(error) from error
        except ValueError:
            raise HTTPException(status_code=401, detail="invalid Attempt transfer capability") from None
        binding.update(run_id=run, attempt_id=attempt)
    uploads = None
    if kind == "asset":
        frozen = None
        files = None
        transport = None
        if data is not None:
            if data.target is None:
                raise application_http_error(ApplicationError("new data uploads require an executor target",
                                                              status_code=422, code="UPLOAD_TARGET_REQUIRED"))
            frozen = await invoke(freeze_target, request.app.state.runtime, project,
                                  data.target.model_dump(exclude_none=True))
            files = [item.model_dump() for item in data.files] if data.files is not None else None
            if files is not None:
                await invoke(service.validate_inventory, files)
            transport = data.transport_sha256
        uploads = await invoke(remote_uploads, service, frozen, runtime=request.app.state.runtime,
                               upload_id=upload_id, files=files, transport_sha256=transport)
        if uploads is not None and uploads.binding is not None:
            if uploads.binding.get("project") != project:
                raise HTTPException(status_code=404, detail="unknown upload for this project")
            binding = uploads.binding
    if uploads is None:
        uploads = StorageService(service.objects).uploads
    return service, uploads, binding


async def create(context, data):
    service, uploads, binding = context
    limit = service.objects.limit if binding["kind"] == "artifacts" else minimum_limit(service.limit, service.objects.limit) if binding["kind"] == "checkpoint" else service.limit
    value = await invoke(uploads.create, binding, data.sha256, data.bytes, limit)
    await confirm_buffered_parts((service, uploads, value["binding"]), value["upload_id"], value)
    return value


async def read(context, upload_id):
    _, uploads, binding = context
    return await invoke(uploads.read, upload_id, binding)


async def confirm_buffered_parts(context, upload_id, value):
    """Explicit POST resumes lost confirmations before a client skips parts."""
    service, uploads, binding = context
    if not getattr(uploads, "remote", False):
        return
    buffer = TransferBuffer(service.root.parent)
    for pending in await invoke(buffer.pending, upload_id):
        number, digest, size = pending["number"], pending["sha256"], pending["bytes"]
        if value.get("parts", {}).get(str(number)) != {"sha256": digest, "bytes": size}:
            continue
        reservation = buffer.reserve(upload_id, number, digest, size)
        try:
            lease = await invoke(reservation.__enter__)
        except HTTPException as error:
            if error.status_code == 503 and (error.headers or {}).get("X-ML-Expd-Error-Code") == "TRANSFER_BUFFER_BUSY":
                continue
            raise
        try:
            if lease.sealed:
                await invoke(lease.seal)
                receipt = await invoke(uploads.part, upload_id, binding, number, lease.path, digest, size)
                await invoke(lease.acknowledge, receipt)
        finally:
            await asyncio.shield(run_in_threadpool(reservation.__exit__, None, None, None))


async def part(context, upload_id, number, digest, request):
    service, uploads, binding = context
    value = await read(context, upload_id)
    expected = await invoke(uploads.part_size, value, number)
    raw = request.headers.get("Content-Length")
    if raw is not None and raw != str(expected):
        raise HTTPException(status_code=400, detail="upload part Content-Length differs")
    buffer = TransferBuffer(service.root.parent)
    reservation = buffer.reserve(upload_id, number, digest, expected)
    lease = await invoke(reservation.__enter__)
    capacity = service.objects.capacity if getattr(uploads, "remote", False) else uploads.capacity
    capacity_lease = capacity.reserve(expected, ignore_upload=upload_id)
    capacity_entered = False
    try:
        if not lease.sealed:
            await invoke(capacity_lease.__enter__)
            capacity_entered = True
        sha, size = hashlib.sha256(), 0
        with nullcontext(None) if lease.sealed else lease.path.open("wb") as temporary:
            async for chunk in request.stream():
                size += len(chunk)
                if size > expected:
                    raise HTTPException(status_code=413, detail="upload part exceeds its declared size")
                sha.update(chunk)
                if temporary is not None:
                    temporary.write(chunk)
            if temporary is not None:
                temporary.flush()
        if size != expected or sha.hexdigest() != digest:
            raise HTTPException(status_code=400, detail="upload part length or SHA256 differs")
        await invoke(lease.seal)
        try:
            result = await invoke(uploads.part, upload_id, binding, number, lease.path, digest, expected)
        except HTTPException as error:
            if error.status_code == 410 and (error.headers or {}).get("X-ML-Expd-Error-Code") == "RUN_DELETED":
                await invoke(lease.discard_deleted_run, binding)
            raise
        receipt = result if getattr(uploads, "remote", False) else {
            "contract": "upload-part.v1", "upload_id": upload_id, "number": number,
            "sha256": result["sha256"], "bytes": result["bytes"], "durable": True}
        await invoke(lease.acknowledge, receipt)
        return result
    finally:
        if capacity_entered:
            await asyncio.shield(run_in_threadpool(capacity_lease.__exit__, None, None, None))
        await asyncio.shield(run_in_threadpool(reservation.__exit__, None, None, None))


async def complete(context, upload_id, request):
    service, uploads, binding = context
    value = await read(context, upload_id)
    if getattr(uploads, "remote", False):
        await confirm_buffered_parts(context, upload_id, value)
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
    return await invoke(uploads.complete, upload_id, binding, publish, min(expanded, value["bytes"]) + value["bytes"] if expanded is not None else value["bytes"] * 2)


async def abort(context, upload_id):
    service, uploads, binding = context
    result = await invoke(uploads.abort, upload_id, binding)
    await invoke(TransferBuffer(service.root.parent).abort, upload_id)
    return result


@router.post("/projects/{project}/asset-uploads")
async def asset_create(project: str, data: UploadRequest, request: Request):
    return await create(await target(request, project, data=data), data)


@router.get("/projects/{project}/asset-uploads/{upload_id}")
async def asset_read(project: str, upload_id: str, request: Request):
    return await read(await target(request, project, upload_id=upload_id), upload_id)


@router.put("/projects/{project}/asset-uploads/{upload_id}/parts/{number}")
async def asset_part(project: str, upload_id: str, number: int, request: Request, sha256: str = Query(pattern=r"^[0-9a-f]{64}$")):
    return await part(await target(request, project, upload_id=upload_id), upload_id, number, sha256, request)


@router.post("/projects/{project}/asset-uploads/{upload_id}/complete")
async def asset_complete(project: str, upload_id: str, request: Request):
    return await complete(await target(request, project, upload_id=upload_id), upload_id, request)


@router.delete("/projects/{project}/asset-uploads/{upload_id}")
async def asset_abort(project: str, upload_id: str, request: Request):
    return await abort(await target(request, project, upload_id=upload_id), upload_id)


@router.post("/attempt-uploads/{project}/{run}/{attempt}/{kind}")
async def worker_create(project: str, run: str, attempt: str, kind: Literal["artifacts", "checkpoint"], data: UploadRequest, request: Request):
    if data.target is not None or data.files is not None or data.transport_sha256 is not None:
        raise HTTPException(status_code=422, detail="Attempt uploads cannot choose a data executor target")
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
