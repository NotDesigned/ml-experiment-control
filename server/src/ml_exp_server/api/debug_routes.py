"""Authenticated, bounded debug-machine lifecycle, separate from training."""
from fastapi import APIRouter, Request

from ..debug_models import DebugRequest
from ..execution_queue import admitted_handoff
from .errors import invoke

router = APIRouter(prefix="/api")


@router.get("/debug-session-profiles")
async def profiles(request: Request):
    result = await invoke(request.app.state.debug_sessions.catalog)
    return {**result, "maintenance_error": request.app.state.debug_session_error}


@router.get("/debug-sessions")
async def sessions(request: Request):
    return await invoke(request.app.state.debug_sessions.listing)


@router.post("/debug-sessions", status_code=202)
async def create(data: DebugRequest, request: Request):
    return await admitted_handoff(lambda: invoke(request.app.state.debug_sessions.create, data))


@router.get("/debug-sessions/{session_id}")
async def read(session_id: str, request: Request):
    return await invoke(request.app.state.debug_sessions.read, session_id)


@router.post("/debug-sessions/{session_id}/refresh")
async def refresh(session_id: str, request: Request):
    return await admitted_handoff(lambda: invoke(request.app.state.debug_sessions.refresh, session_id))


@router.post("/debug-sessions/{session_id}/stop")
async def stop(session_id: str, request: Request):
    return await admitted_handoff(lambda: invoke(request.app.state.debug_sessions.refresh, session_id, intent="stop"))


@router.delete("/debug-sessions/{session_id}")
async def delete(session_id: str, request: Request):
    return await admitted_handoff(lambda: invoke(request.app.state.debug_sessions.refresh, session_id, intent="delete"))
