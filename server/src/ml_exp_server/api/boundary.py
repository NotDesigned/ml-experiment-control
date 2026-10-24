"""HTTP authentication, protocol negotiation and bounded admission."""
from __future__ import annotations

import hmac
import re

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .contract import API_PROTOCOL_VERSION, CLIENT_PROTOCOL_HEADER, MIN_CLIENT_PROTOCOL_VERSION
from ..results.artifact_store import TRANSFER_PATH, LAUNCH_PATH
from ..data.data_assets import WORKER_PATH
from .upload_routes import WORKER_UPLOAD_PATH
from ..data.data_delivery import COPY_TRANSFER
from ..backends.sensecore_preparation import CALLBACK_PATH
from .record_routes import RECORD_PATH
from .deletion_routes import DELETION_TRANSFER
from .result_routes import RESULT_PATH
from ..execution_queue import QueueUnavailable


_EXECUTION_ROUTES = (
    ("action", re.compile(r"/api/debug-sessions(?:/[^/]+/(?:refresh|stop))?")),
    ("action", re.compile(r"/api/(?:actions/execute|submissions/[^/]+/execute)")),
    ("preparation", re.compile(r"/api/projects/[^/]+/experiment-preparations(?:/[^/]+/continue)?")),
    ("build", re.compile(r"/api/projects/[^/]+/runtimes/[^/]+/(?:execute|reconcile)")),
    ("data", re.compile(r"/api/projects/[^/]+/data-deliveries/[^/]+/(?:execute|reconcile)")),
    ("collection", re.compile(r"/api/runs/[^/]+/[^/]+/attempts/[^/]+/collection(?:/repair)?")),
)


def _execution_lane(method, path):
    if method == "DELETE" and re.fullmatch(r"/api/debug-sessions/[^/]+", path):
        return "action"
    if method == "POST":
        for lane, pattern in _EXECUTION_ROUTES:
            if pattern.fullmatch(path):
                return lane
    return None


def install_http_boundary(app: FastAPI, bearer_token: str | None) -> None:
    """Apply admission before any route can claim a durable mutation."""
    @app.middleware("http")
    async def enforce_http_boundary(request, call_next):
        worker_transfer = request.method == "PUT" and bool(TRANSFER_PATH.fullmatch(request.url.path))
        worker_transfer = worker_transfer or (request.method == "PUT" and bool(RESULT_PATH.fullmatch(request.url.path)))
        worker_transfer = worker_transfer or (request.method == "POST" and bool(RECORD_PATH.fullmatch(request.url.path)))
        worker_transfer = worker_transfer or (request.method == "POST" and bool(DELETION_TRANSFER.fullmatch(request.url.path)))
        worker_transfer = worker_transfer or (request.method in {"GET", "PUT"} and bool(LAUNCH_PATH.fullmatch(request.url.path)))
        worker_transfer = worker_transfer or (request.method == "PUT" and bool(COPY_TRANSFER.fullmatch(request.url.path)))
        worker_transfer = worker_transfer or (request.method == "PUT" and bool(CALLBACK_PATH.fullmatch(request.url.path)))
        worker_transfer = worker_transfer or (request.method in {"GET", "PUT"} and bool(WORKER_PATH.fullmatch(request.url.path)))
        worker_transfer = worker_transfer or (request.method in {"GET", "POST", "PUT", "DELETE"} and bool(WORKER_UPLOAD_PATH.fullmatch(request.url.path)))
        if bearer_token is not None and not worker_transfer:
            scheme, separator, credential = request.headers.get(
                "Authorization", "",
            ).partition(" ")
            authenticated = (
                bool(separator)
                and scheme.lower() == "bearer"
                and hmac.compare_digest(credential, bearer_token)
            )
            if not authenticated:
                return JSONResponse(
                    {"detail": "valid bearer authentication is required"},
                    status_code=401,
                    headers={
                        "WWW-Authenticate": 'Bearer realm="ml-expd"',
                        "X-ML-Expd-Error-Code": "AUTHENTICATION_REQUIRED",
                    },
                )
        client_protocol = request.headers.get(CLIENT_PROTOCOL_HEADER)
        api_request = not worker_transfer and (request.url.path == "/api" or request.url.path.startswith("/api/"))
        protocol_bootstrap = (
            request.method in {"GET", "HEAD"}
            and request.url.path == "/api/health"
        )
        if api_request and client_protocol is None and not protocol_bootstrap:
            return JSONResponse(
                {
                    "detail": "client API protocol header is required",
                    "api_protocol_version": API_PROTOCOL_VERSION,
                    "min_client_protocol_version": MIN_CLIENT_PROTOCOL_VERSION,
                },
                status_code=426,
                headers={
                    "X-ML-Expd-Error-Code": "INCOMPATIBLE_API_PROTOCOL",
                    "X-ML-Expd-Protocol": str(API_PROTOCOL_VERSION),
                },
            )
        if api_request and client_protocol is not None:
            try:
                client_protocol_number = int(client_protocol)
            except ValueError:
                client_protocol_number = -1
            if not (
                MIN_CLIENT_PROTOCOL_VERSION
                <= client_protocol_number
                <= API_PROTOCOL_VERSION
            ):
                return JSONResponse(
                    {
                        "detail": "client and daemon API protocol versions are incompatible",
                        "api_protocol_version": API_PROTOCOL_VERSION,
                        "min_client_protocol_version": MIN_CLIENT_PROTOCOL_VERSION,
                    },
                    status_code=426,
                    headers={
                        "X-ML-Expd-Error-Code": "INCOMPATIBLE_API_PROTOCOL",
                        "X-ML-Expd-Protocol": str(API_PROTOCOL_VERSION),
                    },
                )
        if (
            request.method not in {"GET", "HEAD", "OPTIONS"}
            and not bool(getattr(app.state, "workspace_owner", False))
        ):
            return JSONResponse(
                {"detail": "workspace mutations are owned by another ml-expd process"},
                status_code=409,
                headers={"X-ML-Expd-Error-Code": "WORKSPACE_NOT_OWNER"},
            )
        lane = _execution_lane(request.method, request.url.path)
        if lane is None:
            response = await call_next(request)
        else:
            try:
                reservation = app.state.job_executors[lane].reserve()
            except QueueUnavailable:
                return JSONResponse(
                    {"detail": "execution capacity is full or stopping; retry this request later",
                     "code": "EXECUTION_CAPACITY_UNAVAILABLE", "lane": lane, "retry_after_seconds": 2},
                    status_code=503,
                    headers={"Retry-After": "2", "X-ML-Expd-Error-Code": "EXECUTION_CAPACITY_UNAVAILABLE",
                             "X-ML-Expd-Protocol": str(API_PROTOCOL_VERSION)},
                )
            with reservation:
                response = await call_next(request)
        response.headers["X-ML-Expd-Protocol"] = str(API_PROTOCOL_VERSION)
        return response
