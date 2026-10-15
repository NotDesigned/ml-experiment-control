"""Stable translation from application failures to transport failures."""

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

from ..application_errors import ApplicationError
from ..storage_capacity import CapacityError


ERROR_CODE_HEADER = "X-ML-Expd-Error-Code"


def application_http_error(exc: ApplicationError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail={"message": str(exc), "code": exc.code, "required_free_bytes": exc.required_free_bytes,
                "available_bytes": exc.available_bytes, "reserved_bytes": exc.reserved_bytes}
        if isinstance(exc, CapacityError) else str(exc),
        headers={ERROR_CODE_HEADER: exc.code, **({"Retry-After": "1"} if exc.status_code == 503 and exc.code == "TRANSFER_BUFFER_BUSY" else {})},
    )


async def invoke(call, *args, **kwargs):
    try:
        return await run_in_threadpool(call, *args, **kwargs)
    except ApplicationError as exc:
        raise application_http_error(exc) from exc
    except (KeyError, ValueError, OSError) as exc:
        raise HTTPException(status_code=409, detail="operation is unavailable or its frozen identity is invalid") from exc
