"""Administrator-only whole-Run deletion plans and narrow worker receipts."""
import re

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from .errors import invoke
from .contract import ActionSnapshotResponse

router = APIRouter(prefix="/api")
DELETION_TRANSFER = re.compile(r"^/api/run-deletion-transfers/action-[0-9a-f]{16}/receipt$")


class DeletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(min_length=1, max_length=4000)
    idempotency_key: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


@router.post("/runs/{project}/{run}/deletion", response_model=ActionSnapshotResponse)
async def prepare(project: str, run: str, data: DeletionRequest, request: Request):
    return await invoke(request.app.state.runtime.action_service.deletion_service.prepare,
                        project, run, data.reason, data.idempotency_key)


@router.post("/run-deletion-transfers/{action_id}/receipt")
async def receipt(action_id: str, request: Request):
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 65536:
            from fastapi import HTTPException
            raise HTTPException(status_code=413, detail="deletion receipt is too large")
    import json
    authorization = request.headers.get("Authorization", "")
    token = authorization[7:] if authorization.startswith("Bearer ") else ""
    try:
        data = json.loads(body)
    except ValueError:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="invalid deletion receipt JSON") from None
    return await invoke(request.app.state.runtime.action_service.deletion_service.receive, action_id, token, data)
