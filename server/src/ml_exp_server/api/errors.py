"""Stable translation from application failures to transport failures."""

from fastapi import HTTPException

from ..application import ApplicationError
from ..storage_capacity import CapacityError


ERROR_CODE_HEADER = "X-ML-Expd-Error-Code"


def application_http_error(exc: ApplicationError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail={"message": str(exc), "code": exc.code, "required_free_bytes": exc.required_free_bytes,
                "available_bytes": exc.available_bytes, "reserved_bytes": exc.reserved_bytes}
        if isinstance(exc, CapacityError) else str(exc),
        headers={ERROR_CODE_HEADER: exc.code},
    )
