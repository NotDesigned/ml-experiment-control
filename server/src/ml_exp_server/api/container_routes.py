"""Remote source import, frozen container execution, and artifact downloads."""

from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse, StreamingResponse, Response

from ..application_errors import ApplicationError
from ..results.artifacts import ArtifactService
from ..results.artifact_store import ArtifactStore, require_artifact_available
from ..results.artifact_lifecycle import ArtifactLifecycle
from ..archive_limits import exceeds
from ..container_execution import ContainerExecutionService, RunRequest, DockerfileRuntimeSpec
from ..image_builder import builder_request
from ..projects.source_imports import SourceImportService
from ..tracking.metric_contract import MetricSchema
from .errors import application_http_error


router = APIRouter(prefix="/api")


class GitImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    url: str = Field(max_length=4096)
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")


class ConfirmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirmation: str = Field(min_length=1, max_length=256)


class DownloadedFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bytes: int = Field(strict=True, ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DownloadAcknowledgement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_bytes: int = Field(strict=True, gt=0)
    files: dict[str, DownloadedFile] = Field(max_length=20000)
    release: bool = Field(default=True, strict=True)


async def invoke(call, *args, **kwargs):
    try:
        return await run_in_threadpool(call, *args, **kwargs)
    except ApplicationError as exc:
        raise application_http_error(exc) from exc
    except (KeyError, ValueError, OSError) as exc:
        raise HTTPException(status_code=409, detail="operation is unavailable or its frozen identity is invalid") from exc


@router.post("/source-imports/git")
async def import_git(data: GitImportRequest, request: Request):
    return await invoke(SourceImportService(request.app.state.runtime).git,
                        data.project, data.url, data.commit)


@router.post("/source-imports/archive")
async def import_archive(request: Request,
                         project: str = Query(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"),
                         sha256: str = Query(pattern=r"^[0-9a-f]{64}$")):
    runtime = request.app.state.runtime
    service = SourceImportService(runtime)
    try:
        service.require_enabled()
    except ApplicationError as exc:
        raise application_http_error(exc) from exc
    limit = runtime.config.container_execution.max_archive_bytes
    size = 0
    with tempfile.TemporaryFile() as stream:
        async for chunk in request.stream():
            size += len(chunk)
            if size > limit:
                raise HTTPException(status_code=413, detail="source archive exceeds upload limit")
            stream.write(chunk)
        stream.seek(0)
        return await invoke(service.archive, project, stream, sha256)


@router.get("/projects/{project}/sources/{source_id}")
async def source(project: str, source_id: str, request: Request):
    return await invoke(SourceImportService(request.app.state.runtime).read, project, source_id)


@router.get("/executors")
async def executors(request: Request):
    return await invoke(ContainerExecutionService(request.app.state.runtime).public_profiles)


@router.get("/environments")
async def environments(request: Request):
    return await invoke(ContainerExecutionService(request.app.state.runtime).environments)


@router.get("/projects/{project}/metrics-schema")
async def project_metrics_schema(project: str, request: Request):
    return await invoke(ContainerExecutionService(request.app.state.runtime).metrics_schema, project)


@router.put("/projects/{project}/metrics-schema")
async def set_project_metrics_schema(project: str, data: MetricSchema, request: Request):
    return await invoke(ContainerExecutionService(request.app.state.runtime).set_metrics_schema, project, data)


@router.post("/projects/{project}/runtimes/prepare")
async def runtime_prepare(project: str, data: DockerfileRuntimeSpec, request: Request):
    return await invoke(ContainerExecutionService(request.app.state.runtime).prepare, project, data)


@router.get("/projects/{project}/runtimes/{runtime_id}")
async def runtime_read(project: str, runtime_id: str, request: Request):
    return await invoke(ContainerExecutionService(request.app.state.runtime).read, project, runtime_id)


@router.post("/projects/{project}/runtimes/{runtime_id}/execute", status_code=202)
async def runtime_execute(project: str, runtime_id: str, data: ConfirmRequest,
                          request: Request):
    service = ContainerExecutionService(request.app.state.runtime)
    pending = await invoke(service.begin_execute, project, runtime_id, data.confirmation)
    if pending is not None:
        await enqueue_build(request, service, pending)
    return {"runtime_id": runtime_id, "status": "EXECUTING" if pending is not None else "READY", "accepted": True}


async def enqueue_build(request, service, pending):
    try:
        return request.app.state.submit_job(service.finish_execute, pending)
    except RuntimeError as exc:
        await invoke(service.submission_failed, pending)
        raise HTTPException(status_code=503, detail="packaging executor is unavailable; inspect saved Runtime") from exc


@router.get("/projects/{project}/runtimes/{runtime_id}/logs")
async def runtime_logs(project: str, runtime_id: str, request: Request):
    return await runtime_build_observation(project, runtime_id, request, "logs")


@router.get("/projects/{project}/runtimes/{runtime_id}/progress")
async def runtime_progress(project: str, runtime_id: str, request: Request):
    return await runtime_build_observation(project, runtime_id, request, "progress")


async def runtime_build_observation(project, runtime_id, request, operation):
    service = ContainerExecutionService(request.app.state.runtime)
    value = await invoke(service.read, project, runtime_id)
    spec = value["spec"]
    payload = {"operation": operation, "project": project, "source_id": spec["source_id"],
               "base_image": value.get("base_image", spec.get("image")), "packaging_revision": spec["packaging_revision"]}
    payload.update({key: spec[key] for key in ("requirements", "dockerfile") if key in spec})
    pinned = value.get("bundle_id", value.get("build_bundle_id"))
    if pinned is not None:
        payload["bundle_id"] = pinned
    result = await invoke(builder_request, service.runtime.config.container_execution.builder_socket, payload)
    if operation == "progress":
        result.update(runtime_id=runtime_id, status=value["status"])
        result["build_error"] = value.get("build_error")
        if not result["progress"].get("events"):
            from ..runs.execution_progress import progress_view
            result["progress"] = progress_view(service.root / project / (runtime_id + ".progress.json"), value["status"], active=value["status"] == "EXECUTING")
        if value["status"] != "EXECUTING":
            result["progress"].update(status=value["status"], phase=value["status"], no_progress_warning=False, diagnostic=None)
    return result


@router.post("/projects/{project}/runtimes/{runtime_id}/reconcile")
async def runtime_reconcile(project: str, runtime_id: str, data: ConfirmRequest, request: Request):
    service = ContainerExecutionService(request.app.state.runtime)
    pending = await invoke(service.begin_execute, project, runtime_id, data.confirmation, reconcile=True)
    if pending is None:
        return await invoke(service.read, project, runtime_id)
    future = await enqueue_build(request, service, pending)
    # A client disconnect cannot cancel durable ownership of receipt recovery.
    return await asyncio.shield(asyncio.wrap_future(future))


@router.post("/projects/{project}/runs")
async def create_run(project: str, data: RunRequest, request: Request):
    return await invoke(ContainerExecutionService(request.app.state.runtime).create_run, project, data)


@router.put("/artifact-transfers/{project}/{run_id}/{attempt_id}", include_in_schema=False)
async def artifact_transfer(project: str, run_id: str, attempt_id: str, request: Request):
    runtime = request.app.state.runtime
    config = runtime.config.container_execution.artifact_store_file
    if not config:
        raise HTTPException(status_code=404, detail="artifact transfer is not configured")
    service = ArtifactStore(Path(config), runtime.config.project_registry_root_path())
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    try:
        if scheme.lower() != "bearer":
            raise ValueError()
        await run_in_threadpool(service.authorize, project, run_id, attempt_id, token)
    except ValueError:
        raise HTTPException(status_code=401, detail="invalid Attempt upload capability")
    size = 0
    with tempfile.TemporaryFile() as stream:
        async for chunk in request.stream():
            size += len(chunk)
            if exceeds(size, service.limit):
                raise HTTPException(status_code=413, detail="artifact upload exceeds limit")
            stream.write(chunk)
        stream.seek(0)
        receipt = await invoke(service.receive, project, run_id, attempt_id, token, stream, size)
        return {"sha256": receipt["sha256"], "bytes": receipt["bytes"], "files": len(receipt["files"])}


@router.get("/launch-transfers/{project}/{run_id}/{attempt_id}/{digest}", include_in_schema=False)
async def launch_transfer(project: str, run_id: str, attempt_id: str, digest: str, request: Request):
    runtime = request.app.state.runtime
    config = runtime.config.container_execution.artifact_store_file
    if not config:
        raise HTTPException(status_code=404, detail="launch transfer is not configured")
    store = ArtifactStore(Path(config), runtime.config.project_registry_root_path())
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    try:
        if scheme.lower() != "bearer":
            raise ValueError()
        body = await run_in_threadpool(store.launch, project, run_id, attempt_id, digest, token)
    except ValueError:
        raise HTTPException(status_code=401, detail="invalid Attempt launch capability or identity")
    return Response(body, media_type="application/json", headers={"Cache-Control": "no-store", "ETag": '"' + digest + '"'})


@router.get("/runs/{project}/{run_id}/attempts/{attempt_id}/files")
async def artifact_list(project: str, run_id: str, attempt_id: str, request: Request, checksums: bool = False):
    return await invoke(ArtifactService(request.app.state.runtime).list, project, run_id, attempt_id, checksums=checksums)


@router.get("/runs/{project}/{run_id}/attempts/{attempt_id}/artifacts/archive")
async def artifact_archive(project: str, run_id: str, attempt_id: str, request: Request):
    runtime = request.app.state.runtime
    await invoke(ArtifactService(runtime).roots, project, run_id, attempt_id, restore=False)
    config = runtime.config.container_execution.artifact_store_file
    if not config:
        raise HTTPException(status_code=404, detail="object storage is not configured")
    store = ArtifactStore(Path(config), runtime.config.project_registry_root_path())
    def fetch():
        with store.record(project, run_id, attempt_id) as (_, value):
            if not value or not value.get("receipt"):
                raise ApplicationError("Attempt artifacts have not been uploaded", status_code=404, code="ARTIFACT_UNAVAILABLE")
            receipt = value["receipt"]
            require_artifact_available(value)
        response = store.client().get_object(Bucket=store.config["bucket"], Key=receipt["object_key"])
        return receipt, response["Body"]
    receipt, stream = await invoke(fetch)
    name = quote(run_id + "-" + attempt_id + ".tar", safe="")
    return StreamingResponse(stream.iter_chunks(chunk_size=1024 * 1024), media_type="application/x-tar",
                             headers={"Content-Length": str(receipt["bytes"]), "ETag": '"' + receipt["sha256"] + '"',
                                      "Content-Disposition": "attachment; filename*=UTF-8''" + name},
                             background=BackgroundTask(stream.close))


@router.get("/runs/{project}/{run_id}/attempts/{attempt_id}/artifacts/download")
async def artifact_download_link(project: str, run_id: str, attempt_id: str, request: Request):
    runtime = request.app.state.runtime
    row = runtime.index.get_run(project, run_id)
    if row is None or attempt_id not in {a.attempt_id for a in row.attempts}:
        raise HTTPException(status_code=404, detail="unknown Run/Attempt")
    config = runtime.config.container_execution.artifact_store_file
    if not config:
        raise HTTPException(status_code=404, detail="object storage is not configured")
    store = ArtifactStore(Path(config), runtime.config.project_registry_root_path())
    if not store.config.get("public_endpoint"):
        raise HTTPException(status_code=404, detail="direct downloads are not configured")
    value = await invoke(store.artifact_download, project, run_id, attempt_id)
    return JSONResponse(value, headers={"Cache-Control": "no-store"})


@router.post("/runs/{project}/{run_id}/attempts/{attempt_id}/artifacts/ack")
async def acknowledge_artifact_download(project: str, run_id: str, attempt_id: str,
                                       data: DownloadAcknowledgement, request: Request):
    runtime = request.app.state.runtime
    await invoke(ContainerExecutionService(runtime).require_enabled)
    await invoke(ArtifactService(runtime).roots, project, run_id, attempt_id, restore=False)
    config = runtime.config.container_execution.artifact_store_file
    if not config:
        raise HTTPException(status_code=404, detail="object storage is not configured")
    lifecycle = ArtifactLifecycle(ArtifactStore(Path(config), runtime.config.project_registry_root_path()))
    return await invoke(lifecycle.acknowledge, project, run_id, attempt_id,
                        data.model_dump(exclude={"release"}), release=data.release)


@router.get("/runs/{project}/{run_id}/attempts/{attempt_id}/files/{path:path}")
async def artifact_download(project: str, run_id: str, attempt_id: str, path: str, request: Request):
    artifact = await invoke(ArtifactService(request.app.state.runtime).open, project, run_id, attempt_id, path)
    headers = {"Content-Disposition": "attachment; filename*=UTF-8''" + quote(artifact.name, safe=""),
               "ETag": artifact.etag, "Accept-Ranges": "bytes", "X-Content-Type-Options": "nosniff"}
    start, length, status = 0, artifact.size, 200
    selected = request.headers.get("Range")
    if selected and request.headers.get("If-Range", artifact.etag) == artifact.etag:
        match = re.fullmatch(r"bytes=([0-9]*)-([0-9]*)", selected)
        try:
            if not match or not any(match.groups()):
                raise ValueError()
            first, last = match.groups()
            start = int(first) if first else max(0, artifact.size - int(last))
            end = min(artifact.size - 1, int(last)) if first and last else artifact.size - 1
            if start > end or start >= artifact.size:
                raise ValueError()
            length, status = end - start + 1, 206
            headers["Content-Range"] = f"bytes {start}-{end}/{artifact.size}"
        except ValueError:
            artifact.close()
            raise HTTPException(status_code=416, detail="invalid artifact byte range",
                                headers={"Content-Range": f"bytes */{artifact.size}"})
    headers["Content-Length"] = str(length)
    return StreamingResponse(artifact.chunks(start, length), status_code=status, headers=headers,
                             media_type="application/octet-stream", background=BackgroundTask(artifact.close))
