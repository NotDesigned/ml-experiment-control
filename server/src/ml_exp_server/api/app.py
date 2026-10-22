"""FastAPI composition entry point for the independent experiment daemon."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ..ingest.indexer import RunIndex
from ..projects.project_config import load_server_config
from ..runtime import ExperimentServerRuntime
from ..schemas import ServerConfig, ResearchProject
from .auth import load_bearer_token
from .contract import VERSIONED_OPENAPI_PATH
from .lifecycle import server_lifespan
from .boundary import install_http_boundary
from .routes import router
from .action_routes import router as action_router
from .operation_routes import router as operation_router
from .submission_routes import router as submission_router
from .sse import EventBroker
from .upload_routes import router as upload_router
from .data_delivery_routes import router as data_delivery_router
from .preparation_routes import router as preparation_router
from .tracking_routes import router as tracking_router
from .deletion_routes import router as deletion_router
from .result_routes import router as result_router
from .script_preparation_routes import router as script_preparation_router


def create_app(config: ServerConfig, *, poll: Optional[bool] = None,
               index: Optional[RunIndex] = None,
               projects: Optional[list[ResearchProject]] = None) -> FastAPI:
    token_path = config.http_auth.token_path()
    bearer_token = load_bearer_token(token_path) if token_path is not None else None
    effective_poll = config.collector_enabled if poll is None else poll
    injected_projects = list(projects) if projects is not None else None

    lifespan = server_lifespan(config, effective_poll=effective_poll, index=index,
                               injected_projects=injected_projects)

    from importlib.metadata import version
    app = FastAPI(title="ml-expd", version=version("ml-experiment-server"), lifespan=lifespan)
    from fastapi.exceptions import RequestValidationError
    from fastapi.exception_handlers import request_validation_exception_handler

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        if request.url.path == "/api/tracking/wandb":
            return JSONResponse({"detail": [{key: value for key, value in error.items() if key in {"loc", "msg", "type"}}
                                            for error in exc.errors()]}, status_code=422)
        return await request_validation_exception_handler(request, exc)
    app.state.broker = EventBroker()
    app.state.config = config
    app.state.runtime = None
    from ..runs.allocation_status import AllocationStatus
    app.state.allocation_status = AllocationStatus()
    app.state.application = None
    app.state.submission_service = None
    app.state.action_executor = None
    app.state.job_executors = None
    app.state.submit_action = None
    app.state.submit_job = None
    app.state.index = None
    app.state.projects = []
    app.state.action_store = None
    app.state.action_service = None
    app.state.collector = None
    app.state.collector_owner = False
    app.state.workspace_owner = False
    app.state.collector_error = None
    app.state.collector_lease = None
    app.state.auth_mode = "bearer" if bearer_token is not None else "none"
    app.state.runtime_initializers: list[
        Callable[[ExperimentServerRuntime], None]
    ] = []

    install_http_boundary(app, bearer_token)

    app.include_router(router)
    app.include_router(action_router)
    app.include_router(operation_router)
    app.include_router(submission_router)
    from .container_routes import router as container_router
    app.include_router(container_router)
    from .asset_routes import router as asset_router
    app.include_router(asset_router)
    app.include_router(upload_router)
    app.include_router(data_delivery_router)
    app.include_router(preparation_router)
    app.include_router(tracking_router)
    app.include_router(deletion_router)
    app.include_router(result_router)
    app.include_router(script_preparation_router)

    @app.get(VERSIONED_OPENAPI_PATH, include_in_schema=False)
    async def versioned_openapi():
        return JSONResponse(app.openapi())

    @app.get("/api", include_in_schema=False)
    @app.get("/api/{path:path}", include_in_schema=False)
    async def api_not_found(path: str = ""):
        """Keep unknown API paths machine-readable."""
        return JSONResponse({"detail": "API route not found"}, status_code=404)

    return app


def create_app_from_config_file(config_path: Path, *, poll: Optional[bool] = None) -> FastAPI:
    return create_app(load_server_config(config_path), poll=poll)
