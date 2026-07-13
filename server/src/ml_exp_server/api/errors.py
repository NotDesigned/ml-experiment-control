"""Stable translation from application failures to transport failures."""

from fastapi import HTTPException

from ..application import ApplicationError


ERROR_CODE_HEADER = "X-ML-Expd-Error-Code"
LEGACY_ERROR_CODE_HEADER = "X-Research-Console-Error-Code"


def application_http_error(exc: ApplicationError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail=str(exc),
        headers={ERROR_CODE_HEADER: exc.code, LEGACY_ERROR_CODE_HEADER: exc.code},
    )
