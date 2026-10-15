"""Shared HTTP upload boundaries; no router or provider dependencies."""
from pathlib import Path
import tempfile

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

from ..archive_limits import exceeds
from ..data.data_assets import AssetStore
from .errors import invoke


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
    if exceeds(expected_size, service.limit):
        raise HTTPException(status_code=413, detail="data archive exceeds upload limit")
    objects = getattr(service, "objects", service)
    reservation = objects.capacity.reserve(3 * expected_size)
    lease = await invoke(reservation.__enter__)
    try:
        return await receive_body(request, service, call, args, lease, expected_size)
    finally:
        await run_in_threadpool(reservation.__exit__, None, None, None)


async def receive_body(request, service, call, args, lease, expected_size):
    size = 0
    with tempfile.TemporaryFile(dir=service.root) as stream:
        async for chunk in request.stream():
            size += len(chunk)
            if exceeds(size, service.limit):
                raise HTTPException(status_code=413, detail="data archive exceeds upload limit")
            if not expected_size or size > expected_size:
                await invoke(lease.resize, 3 * size)
            stream.write(chunk)
        stream.seek(0)
        return await invoke(call, *args, stream, size)


def token(request):
    scheme, _, value = request.headers.get("Authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not value:
        raise HTTPException(status_code=401, detail="invalid Attempt transfer capability")
    return value
