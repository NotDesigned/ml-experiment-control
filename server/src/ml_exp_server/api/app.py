"""FastAPI app factory for the independent ML experiment daemon."""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ..application import ExperimentServerApplication
from ..collectord import Collector, CollectorConfig, CollectorLease
from ..ingest.indexer import RunIndex, index_project
from ..project_config import load_server_config
from ..runtime import ExperimentServerRuntime
from ..schemas import ServerConfig, ResearchProject
from ..submissions import ExperimentSubmissionService
from .routes import router
from .action_routes import router as action_router
from .operation_routes import router as operation_router
from .submission_routes import router as submission_router
from .sse import EventBroker

def create_app(config: ServerConfig, *, poll: Optional[bool] = None,
               index: Optional[RunIndex] = None,
               projects: Optional[list[ResearchProject]] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        app.state.broker.bind_loop(loop)
        app.state._stop = threading.Event()

        def initial_index() -> None:
            for project in app.state.projects:
                index_project(app.state.index, project)

        await loop.run_in_executor(None, initial_index)

        collector: Optional[Collector] = app.state.collector
        if collector is not None:
            lease = CollectorLease(app.state.config.index_db_path())
            if not lease.acquire():
                app.state.collector_error = (
                    "another ml-expd process owns this workspace collector"
                )
            else:
                app.state.collector_owner = True
                app.state.collector_lease = lease
                # Projection ownership follows the same workspace lease as
                # canonical collection.  A second daemon must never spawn a
                # duplicate local service.  Startup is bounded and degradable.
                app.state.observability = await asyncio.to_thread(
                    app.state.runtime.wandb_service.start,
                )

                def poll_loop() -> None:
                    while not app.state._stop.is_set():
                        app.state.index.set_meta("collector_cycle_started_at", str(time.time()))
                        try:
                            collector.run_cycle()
                            app.state.index.set_meta("collector_last_error", "")
                        except Exception as exc:  # keep the loop alive; surface via meta
                            app.state.index.set_meta("collector_last_error", str(exc)[:500])
                        finally:
                            app.state.index.set_meta("collector_cycle_started_at", "")
                        if app.state._stop.wait(collector.config.poll_interval_seconds):
                            break

                thread = threading.Thread(target=poll_loop, name="collectord", daemon=True)
                thread.start()
                app.state._poll_thread = thread
        yield
        app.state._stop.set()
        thread = getattr(app.state, "_poll_thread", None)
        if thread is not None:
            # The collector owns the index while a cycle is in flight.  Do not
            # close the runtime until that owner has observed the stop signal.
            await asyncio.to_thread(thread.join, 5.0)
        lease = app.state.collector_lease
        if thread is None or not thread.is_alive():
            app.state.runtime.close()
            if lease is not None:
                lease.release()
        else:
            # Retain the lease until a long in-flight controller observation
            # really exits; otherwise another daemon could become owner while
            # the prior collector and its subprocesses are still alive.
            app.state.runtime.wandb_service.stop()

            def finish_shutdown() -> None:
                thread.join()
                app.state.runtime.close()
                if lease is not None:
                    lease.release()

            cleanup = threading.Thread(
                target=finish_shutdown, name="collectord-shutdown", daemon=True,
            )
            cleanup.start()

    app = FastAPI(title="ml-expd", version="0.1.0", lifespan=lifespan)
    app.state.broker = EventBroker()
    def publish_index_update(project: str, run_id: str) -> None:
        app.state.broker.publish_threadsafe(
            {"type": "index_updated", "project": project, "run_id": run_id})
    runtime = ExperimentServerRuntime.create(
        config, index=index, projects=projects, on_index_update=publish_index_update,
    )
    app.state.runtime = runtime
    app.state.application = ExperimentServerApplication(runtime)
    app.state.submission_service = ExperimentSubmissionService(
        app.state.application, runtime,
    )
    # Compatibility aliases for extensions and existing integrations. New
    # transport code should enter through app.state.application.
    app.state.config = runtime.config
    app.state.index = runtime.index
    app.state.projects = runtime.projects
    app.state.action_store = runtime.action_store
    app.state.action_service = runtime.action_service
    app.state.collector = None
    app.state.collector_owner = False
    app.state.collector_error = None
    app.state.collector_lease = None
    app.state.observability = app.state.runtime.wandb_service.status()

    effective_poll = config.collector_enabled if poll is None else poll
    if effective_poll:
        app.state.collector = Collector(
            index=app.state.index, projects=app.state.projects,
            config=CollectorConfig(poll_interval_seconds=config.poll_interval_seconds))

    app.include_router(router)
    app.include_router(action_router)
    app.include_router(operation_router)
    app.include_router(submission_router)

    @app.get("/api", include_in_schema=False)
    @app.get("/api/{path:path}", include_in_schema=False)
    async def api_not_found(path: str = ""):
        """Keep unknown API paths machine-readable."""
        return JSONResponse({"detail": "API route not found"}, status_code=404)

    return app


def create_app_from_config_file(config_path: Path, *, poll: Optional[bool] = None) -> FastAPI:
    return create_app(load_server_config(config_path), poll=poll)
