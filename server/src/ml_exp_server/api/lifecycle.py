"""Daemon ownership, startup recovery and orderly shutdown.

The workspace lease and Runtime stay owned until all admitted work has finished.
This module owns no second runtime, queue or durable state.
"""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import Future
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from ..application import ExperimentServerApplication
from ..collectord import Collector, CollectorConfig, CollectorLease
from ..ingest.indexer import RunIndex, index_project
from ..runtime import ExperimentServerRuntime
from ..schemas import ServerConfig, ResearchProject
from ..runs.submissions import ExperimentSubmissionService
from ..container_execution import ContainerExecutionService
from ..builds.runtime_jobs import recover_interrupted_builds
from ..data.data_delivery import recover_data_deliveries
from ..runs.experiment_preparation import ExperimentPreparationService
from ..tracking.tracking_service import TrackingPublisher
from ..results.result_collection import ResultCollectionService
from ..execution_queue import BoundedExecutor, job_lane


def _poll_loop(app: FastAPI, collector: Collector) -> None:
    while not app.state._stop.is_set():
        app.state.index.set_meta("collector_cycle_started_at", str(time.time()))
        try:
            try:
                config_file = app.state.runtime.config.container_execution.artifact_store_file
                if config_file and app.state.runtime.config.action_runtime.allow_project_writes:
                    app.state.runtime.storage_service.maintain(submit=app.state.submit_job)
                app.state.index.set_meta("storage_maintenance_last_error", "")
            except Exception as exc:
                # A storage maintenance fault must not hide scheduler progress.
                app.state.index.set_meta("storage_maintenance_last_error", type(exc).__name__)
            collector.run_cycle()
            if app.state.runtime.config.container_execution.artifact_store_file:
                service = ResultCollectionService(app.state.runtime)
                service.stopping = app.state._stop.is_set
                service.automatic(app.state.submit_job)
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
    publisher = getattr(app.state, "tracking_publisher", None)
    if publisher is not None:
        await asyncio.to_thread(publisher.close)
    stop = getattr(app.state, "_stop", None)
    if stop is not None:
        stop.set()
    thread = getattr(app.state, "_poll_thread", None)
    action_executor = getattr(app.state, "action_executor", None)
    preparation_executor = getattr(app.state, "preparation_executor", None)
    executors = getattr(app.state, "job_executors", None) or {
        key: item for key, item in (("action", action_executor), ("preparation", preparation_executor))
        if item is not None
    }
    executors = list(executors.values())
    for executor in executors:
        # Stop accepting new work. Claimed actions and image builds retain ownership of
        # the runtime until their durable terminal state has been recorded.
        executor.shutdown(wait=False)
    # A disconnected request may still be completing begin + enqueue in its
    # daemon-owned handoff. Finish those before snapshotting worker ownership.
    await asyncio.gather(*(executor.drain_handoffs() for executor in executors
                           if isinstance(executor, BoundedExecutor)))
    with getattr(app.state, "_action_futures_lock", threading.Lock()):
        action_futures = tuple(getattr(app.state, "_action_futures", ()))
    if thread is not None:
        # The collector owns the index while a cycle is in flight. Do not close
        # the runtime until that owner has observed the stop signal.
        await asyncio.to_thread(thread.join, 5.0)
    lease = getattr(app.state, "collector_lease", None)
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        if lease is not None:
            lease.release()
        return
    owner_threads = [
        item for item in (thread,) if item is not None
    ]
    if (not any(item.is_alive() for item in owner_threads) and not action_futures
            and not any(getattr(item, "busy", False) for item in executors)):
        try:
            runtime.close()
        finally:
            if lease is not None:
                lease.release()
    else:
        # Retain the lease until in-flight controller observations and builds really
        # exits; otherwise another daemon could become owner while the prior
        # collector and its subprocesses are still alive.

        def finish_shutdown() -> None:
            for owner in owner_threads:
                owner.join()
            for future in action_futures:
                try:
                    future.result()
                except Exception:
                    # finish_action_execution records unexpected failures in
                    # the action journal; shutdown must still release owners.
                    pass
            for executor in executors:
                executor.shutdown(wait=True)
            try:
                runtime.close()
            finally:
                if lease is not None:
                    lease.release()

        cleanup = threading.Thread(
            target=finish_shutdown, name="collectord-shutdown", daemon=True,
        )
        cleanup.start()


def server_lifespan(config: ServerConfig, *, effective_poll: bool,
                    index: Optional[RunIndex],
                    injected_projects: Optional[list[ResearchProject]]):
    """Bind one daemon lifespan to its configured workspace and dependencies."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        app.state.broker.bind_loop(loop)
        app.state._stop = threading.Event()
        app.state._poll_thread = None
        # Long orchestration waits must not consume independent image, data,
        # or result workers. Capacity includes queued and reserved requests.
        app.state.job_executors = {
            "action": BoundedExecutor("action", workers=4, capacity=16),
            "preparation": BoundedExecutor("preparation", workers=2, capacity=4),
            "build": BoundedExecutor("build", workers=1, capacity=4),
            "data": BoundedExecutor("data", workers=1, capacity=4),
            "collection": BoundedExecutor("collection", workers=1, capacity=4),
        }
        app.state.action_executor = app.state.job_executors["action"]
        app.state.preparation_executor = app.state.job_executors["preparation"]
        app.state._action_futures: set[Future[object]] = set()
        app.state._action_futures_lock = threading.Lock()
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

            def submit_job(call, pending, *, executor=None):
                owner = executor if executor is not None else app.state.job_executors[job_lane(call)]
                future = owner.submit(call, pending)
                with app.state._action_futures_lock:
                    app.state._action_futures.add(future)

                def completed(item: Future[object]) -> None:
                    with app.state._action_futures_lock:
                        app.state._action_futures.discard(item)

                future.add_done_callback(completed)
                return future

            app.state.submit_job = submit_job
            # The collector uses the same admission protocol before claiming
            # an automatic collection, without requiring an HTTP request.
            submit_job.reserve = lambda call: app.state.job_executors[job_lane(call)].reserve()
            app.state.submit_action = lambda pending: submit_job(
                app.state.application.finish_action_execution, pending,
                executor=app.state.action_executor,
            )
            app.state.recovered_runtime_builds = recover_interrupted_builds(ContainerExecutionService(runtime))
            app.state.recovered_data_deliveries = recover_data_deliveries(runtime)
            app.state.recovered_result_collections = (ResultCollectionService(runtime).recover_interrupted()
                                                      if runtime.config.container_execution.artifact_store_file else 0)
            runtime.action_service.recover_interrupted_executions()
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
            app.state.preparation_service = ExperimentPreparationService(runtime, app.state.submission_service)
            app.state.recovered_preparations = app.state.preparation_service.recover_interrupted()
            # Compatibility aliases for extensions and existing integrations.
            app.state.index = runtime.index
            app.state.projects = runtime.projects
            app.state.action_store = runtime.action_store
            app.state.action_service = runtime.action_service
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
                app.state._poll_thread = _start_daemon_thread(
                    target=_poll_loop, args=(app, collector), name="collectord",
                )
            app.state.tracking_publisher = TrackingPublisher(runtime)
            app.state.tracking_publisher.start()
            yield
        finally:
            if getattr(app.state, "runtime", None) is None:
                for executor in app.state.job_executors.values():
                    executor.shutdown(wait=False, cancel_futures=True)
                if lease_acquired:
                    lease.release()
            else:
                await _shutdown(app)

    return lifespan
