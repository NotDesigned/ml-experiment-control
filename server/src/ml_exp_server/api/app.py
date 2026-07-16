"""FastAPI app factory for the independent ML experiment daemon."""

from __future__ import annotations

import asyncio
import hmac
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ..application import ExperimentServerApplication
from ..api_contract import (
    API_PROTOCOL_VERSION,
    CLIENT_PROTOCOL_HEADER,
    MIN_CLIENT_PROTOCOL_VERSION,
    VERSIONED_OPENAPI_PATH,
)
from ..collectord import Collector, CollectorConfig, CollectorLease
from ..ingest.indexer import RunIndex, index_project
from ..http_auth import load_bearer_token
from ..project_config import load_server_config
from ..runtime import ExperimentServerRuntime
from ..schemas import ServerConfig, ResearchProject
from ..submissions import ExperimentSubmissionService
from .routes import router
from .action_routes import router as action_router
from .operation_routes import router as operation_router
from .submission_routes import router as submission_router
from .sse import EventBroker


def _publisher_loop(app: FastAPI) -> None:
    while not app.state._stop.is_set():
        try:
            app.state.runtime.observability.publish_once(limit_per_target=32)
        except Exception as exc:
            # Target-specific errors are retained in the outbox. Systemic loop
            # errors must additionally be visible through health.
            app.state.publisher_last_error = f"{type(exc).__name__}: {exc}"[:500]
            app.state.publisher_consecutive_failures += 1
        else:
            app.state.publisher_last_success_at = time.time()
            app.state.publisher_last_error = None
            app.state.publisher_consecutive_failures = 0
        if app.state._stop.wait(2.0):
            break


def _poll_loop(app: FastAPI, collector: Collector) -> None:
    while not app.state._stop.is_set():
        app.state.index.set_meta("collector_cycle_started_at", str(time.time()))
        try:
            collector.run_cycle()
            for project in app.state.projects:
                app.state.runtime.observability.collect_rows(
                    app.state.index.list_runs(project.project),
                )
            app.state.index.set_meta("collector_last_error", "")
        except Exception as exc:  # keep the loop alive; surface via meta
            app.state.index.set_meta("collector_last_error", str(exc)[:500])
        finally:
            app.state.index.set_meta("collector_cycle_started_at", "")
        if app.state._stop.wait(collector.config.poll_interval_seconds):
            break


def _start_daemon_thread(*, target, name: str, args: tuple = ()) -> threading.Thread:
    thread = threading.Thread(target=target, args=args, name=name, daemon=True)
    thread.start()
    return thread


async def _shutdown(app: FastAPI) -> None:
    stop = getattr(app.state, "_stop", None)
    if stop is not None:
        stop.set()
    thread = getattr(app.state, "_poll_thread", None)
    publisher_thread = getattr(app.state, "_publisher_thread", None)
    if thread is not None:
        # The collector owns the index while a cycle is in flight. Do not close
        # the runtime until that owner has observed the stop signal.
        await asyncio.to_thread(thread.join, 5.0)
    if publisher_thread is not None:
        await asyncio.to_thread(publisher_thread.join, 45.0)
    lease = getattr(app.state, "collector_lease", None)
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        if lease is not None:
            lease.release()
        return
    owner_threads = [
        item for item in (thread, publisher_thread) if item is not None
    ]
    if not any(item.is_alive() for item in owner_threads):
        try:
            runtime.close()
        finally:
            if lease is not None:
                lease.release()
    else:
        # Retain the lease until a long in-flight controller observation really
        # exits; otherwise another daemon could become owner while the prior
        # collector and its subprocesses are still alive.
        runtime.wandb_service.stop()

        def finish_shutdown() -> None:
            for owner in owner_threads:
                owner.join()
            try:
                runtime.close()
            finally:
                if lease is not None:
                    lease.release()

        cleanup = threading.Thread(
            target=finish_shutdown, name="collectord-shutdown", daemon=True,
        )
        cleanup.start()


def create_app(config: ServerConfig, *, poll: Optional[bool] = None,
               index: Optional[RunIndex] = None,
               projects: Optional[list[ResearchProject]] = None) -> FastAPI:
    token_path = config.http_auth.token_path()
    bearer_token = load_bearer_token(token_path) if token_path is not None else None
    effective_poll = config.collector_enabled if poll is None else poll
    injected_projects = list(projects) if projects is not None else None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        app.state.broker.bind_loop(loop)
        app.state._stop = threading.Event()
        app.state._poll_thread = None
        app.state._publisher_thread = None
        app.state.publisher_last_success_at = None
        app.state.publisher_last_error = None
        app.state.publisher_consecutive_failures = 0
        app.state.project_write_recovery_errors = []
        app.state.collector_owner = False
        app.state.workspace_owner = False
        app.state.collector_lease = None
        lease = CollectorLease(config.index_db_path())
        lease_acquired = False
        try:
            if not lease.acquire():
                raise RuntimeError(
                    "another ml-expd process owns this workspace; "
                    "a second daemon cannot start safely"
                )
            lease_acquired = True
            app.state.workspace_owner = True
            app.state.collector_lease = lease
            runtime = ExperimentServerRuntime.create(
                config,
                index=index,
                projects=injected_projects,
                on_index_update=lambda project, run_id: app.state.broker.publish_threadsafe({
                    "type": "index_updated", "project": project, "run_id": run_id,
                }),
            )
            app.state.runtime = runtime
            app.state.collector_error = None
            for initializer in tuple(app.state.runtime_initializers):
                initializer(runtime)
            app.state.application = ExperimentServerApplication(runtime)
            # Complete any previously authorized project-file transaction
            # before indexing those files into the server read model.
            recovered_writes = runtime.action_service.recover_pending_project_writes()
            app.state.project_write_recovery_errors = [
                (
                    f"{item.get('action_id')}: "
                    f"{item.get('execution', {}).get('error') or 'recovery incomplete'}"
                )[:500]
                for item in recovered_writes
                if item.get("execution", {}).get("status") != "VERIFIED"
            ][:20]
            app.state.submission_service = ExperimentSubmissionService(
                app.state.application, runtime,
            )
            # Compatibility aliases for extensions and existing integrations.
            app.state.index = runtime.index
            app.state.projects = runtime.projects
            app.state.action_store = runtime.action_store
            app.state.action_service = runtime.action_service
            app.state.observability = runtime.wandb_service.status()
            app.state.collector = (
                Collector(
                    index=runtime.index,
                    projects=runtime.projects,
                    action_store=runtime.action_store,
                    config=CollectorConfig(
                        poll_interval_seconds=config.poll_interval_seconds,
                        execution_campaign_root=(
                            config.index_db_path().parent / "collector-campaigns"
                        ),
                    ),
                )
                if effective_poll else None
            )

            collector: Optional[Collector] = app.state.collector
            def initial_index() -> None:
                for project in app.state.projects:
                    index_project(app.state.index, project)

            await loop.run_in_executor(None, initial_index)
            if collector is not None:
                app.state.collector_owner = True
                # Only the workspace lease owner may reconcile publisher
                # targets or rewind collection cursors.
                app.state.application.recover_observability_policies()
                # Projection ownership follows the same workspace lease as
                # canonical collection.  A second daemon must never spawn a
                # duplicate local service.  Startup is bounded and degradable.
                app.state.observability = await asyncio.to_thread(
                    app.state.runtime.wandb_service.start,
                )
                app.state._publisher_thread = _start_daemon_thread(
                    target=_publisher_loop, args=(app,), name="wandb-publisher",
                )
                app.state._poll_thread = _start_daemon_thread(
                    target=_poll_loop, args=(app, collector), name="collectord",
                )
            yield
        finally:
            if getattr(app.state, "runtime", None) is None:
                if lease_acquired:
                    lease.release()
            else:
                await _shutdown(app)

    app = FastAPI(title="ml-expd", version="0.1.0", lifespan=lifespan)
    app.state.broker = EventBroker()
    app.state.config = config
    app.state.runtime = None
    app.state.application = None
    app.state.submission_service = None
    app.state.index = None
    app.state.projects = []
    app.state.action_store = None
    app.state.action_service = None
    app.state.collector = None
    app.state.collector_owner = False
    app.state.workspace_owner = False
    app.state.collector_error = None
    app.state.collector_lease = None
    app.state.observability = {"state": "NOT_STARTED"}
    app.state.auth_mode = "bearer" if bearer_token is not None else "none"
    app.state.runtime_initializers: list[
        Callable[[ExperimentServerRuntime], None]
    ] = []

    @app.middleware("http")
    async def enforce_http_boundary(request, call_next):
        if bearer_token is not None:
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
        api_request = request.url.path == "/api" or request.url.path.startswith("/api/")
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
        response = await call_next(request)
        response.headers["X-ML-Expd-Protocol"] = str(API_PROTOCOL_VERSION)
        return response

    app.include_router(router)
    app.include_router(action_router)
    app.include_router(operation_router)
    app.include_router(submission_router)

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
