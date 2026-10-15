"""Admission rejects overload before claims and isolates long background waits."""
from __future__ import annotations

import asyncio
from concurrent.futures import CancelledError
from types import SimpleNamespace
import threading
import time

import pytest
import httpx
from fastapi.testclient import TestClient
from fastapi.routing import APIRoute
from starlette.concurrency import run_in_threadpool

from ml_exp_server.api.app import create_app
from ml_exp_server.api.boundary import _execution_lane
from ml_exp_server.collectord import CollectorLease
from ml_exp_server.api.contract import CLIENT_PROTOCOL_HEADER
from ml_exp_server.execution_queue import BoundedExecutor, QueueUnavailable, admitted_handoff, job_lane
from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.results.result_collection import ResultCollectionService
from ml_exp_server.schemas import ServerConfig


HEADERS = {CLIENT_PROTOCOL_HEADER: "2"}


def settings(tmp_path):
    return ServerConfig(index_db=str(tmp_path / "index.sqlite"),
                        action_root=str(tmp_path / "actions"),
                        project_registry_root=str(tmp_path / "registry"),
                        collector_enabled=False)


@pytest.mark.parametrize("workers,capacity", [(0, 1), (2, 1)])
def test_invalid_capacity(workers, capacity):
    with pytest.raises(ValueError):
        BoundedExecutor("test", workers=workers, capacity=capacity)


def test_capacity_counts_running_queued_and_reserved():
    executor = BoundedExecutor("test", workers=1, capacity=2)
    entered, release = threading.Event(), threading.Event()

    def waiting():
        entered.set()
        assert release.wait(5)

    try:
        reserved = executor.reserve()
        assert executor.busy
        first = reserved.submit(waiting)
        assert entered.wait(2)
        second = executor.submit(lambda: "queued")
        with pytest.raises(QueueUnavailable) as error:
            executor.reserve()
        assert error.value.lane == "test"
        with pytest.raises(RuntimeError, match="already consumed"):
            reserved.submit(waiting)
        reserved.release()
        release.set()
        first.result(timeout=2)
        assert second.result(timeout=2) == "queued"
    finally:
        release.set()
        executor.shutdown()
    assert not executor.busy
    with pytest.raises(QueueUnavailable):
        executor.submit(lambda: None)


def test_context_survives_async_thread_handoffs_and_unused_slots_release():
    executor = BoundedExecutor("test", workers=1, capacity=1)
    other = BoundedExecutor("other", workers=1, capacity=1)

    async def submit():
        return await run_in_threadpool(executor.submit, lambda: "accepted")

    try:
        with executor.reserve():
            assert asyncio.run(submit()).result(timeout=2) == "accepted"
        with executor.reserve():
            assert other.submit(lambda: "independent").result(timeout=2) == "independent"
        with pytest.raises(ValueError):
            with executor.reserve():
                raise ValueError("request validation failed")
        assert not executor.busy
        reserved = executor.reserve()
        reserved.release()
        reserved.release()
        assert not executor.busy
    finally:
        executor.shutdown()
        other.shutdown()


@pytest.mark.parametrize("submit", [True, False])
def test_shutdown_waits_for_admitted_request_and_allows_its_submission(submit):
    executor = BoundedExecutor("test", workers=1, capacity=1)
    reserved = executor.reserve()
    closed = threading.Event()

    def shutdown():
        executor.shutdown()
        closed.set()

    executor.shutdown(wait=False)
    with pytest.raises(QueueUnavailable):
        executor.reserve()
    thread = threading.Thread(target=shutdown)
    thread.start()
    assert not closed.wait(0.05)
    if submit:
        assert reserved.submit(lambda: "claimed before shutdown").result(timeout=2) == "claimed before shutdown"
    else:
        reserved.release()
    thread.join(2)
    assert closed.is_set() and not executor.busy


def test_shutdown_can_cancel_queued_work_without_leaking_capacity():
    executor = BoundedExecutor("test", workers=1, capacity=2)
    entered, release = threading.Event(), threading.Event()

    def waiting():
        entered.set()
        assert release.wait(5)

    first = executor.submit(waiting)
    assert entered.wait(2)
    second = executor.submit(lambda: "never started")
    executor.shutdown(wait=False, cancel_futures=True)
    with pytest.raises(CancelledError):
        second.result(timeout=2)
    release.set()
    first.result(timeout=2)
    executor.shutdown()
    assert not executor.busy


@pytest.mark.parametrize("class_name,lane", [
    ("ContainerExecutionService", "build"), ("DataDeliveryService", "data"),
    ("ResultCollectionService", "collection"), ("ExperimentPreparationService", "preparation"),
])
def test_bound_services_choose_independent_lanes(class_name, lane):
    service = type(class_name, (), {"finish": lambda self, pending: pending})()
    assert job_lane(service.finish) == lane
    assert job_lane(lambda pending: pending) == "preparation"


@pytest.mark.parametrize("path,lane", [
    ("/api/actions/execute", "action"), ("/api/submissions/a/execute", "action"),
    ("/api/projects/p/experiment-preparations", "preparation"),
    ("/api/projects/p/experiment-preparations/id/continue", "preparation"),
    ("/api/projects/p/runtimes/id/execute", "build"),
    ("/api/projects/p/runtimes/id/reconcile", "build"),
    ("/api/projects/p/data-deliveries/id/execute", "data"),
    ("/api/projects/p/data-deliveries/id/reconcile", "data"),
    ("/api/runs/p/r/attempts/a/collection", "collection"),
    ("/api/runs/p/r/attempts/a/collection/repair", "collection"),
    ("/api/actions/execute/unknown", None),
])
def test_exact_mutation_routes_are_admitted(path, lane):
    assert _execution_lane("POST", path) == lane
    assert _execution_lane("GET", path) is None


def test_full_http_queue_rejects_before_claim_with_retry_contract(tmp_path):
    app = create_app(settings(tmp_path))
    with TestClient(app) as client:
        claimed = []
        app.state.submission_service = SimpleNamespace(begin_execute=lambda *args: claimed.append(args))
        reservations = [app.state.action_executor.reserve() for _ in range(16)]
        try:
            response = client.post("/api/submissions/test/execute", headers=HEADERS, json={"confirmation": "go"})
            assert response.status_code == 503
            assert response.headers["Retry-After"] == "2"
            assert response.headers["X-ML-Expd-Error-Code"] == "EXECUTION_CAPACITY_UNAVAILABLE"
            assert response.json()["lane"] == "action"
            assert claimed == []
        finally:
            for reservation in reservations:
                reservation.release()
        assert not app.state.action_executor.busy


@pytest.mark.parametrize("failure", ["validation", "exception", "unused", "shutdown"])
def test_http_admission_release_and_claim_to_enqueue_shutdown_race(tmp_path, failure):
    app = create_app(settings(tmp_path))
    completed = threading.Event()
    with TestClient(app) as client:
        def claim(*_args):
            assert app.state.action_executor._reserved == 1
            if failure == "exception":
                raise RuntimeError("claim rejected")
            if failure == "shutdown":
                app.state.action_executor.shutdown(wait=False)
            return {"status": "VERIFIED" if failure == "unused" else "EXECUTING"}, None if failure == "unused" else "pending"

        app.state.submission_service = SimpleNamespace(begin_execute=claim)
        app.state.application.finish_action_execution = lambda pending: completed.set()
        if failure == "exception":
            with pytest.raises(RuntimeError, match="claim rejected"):
                client.post("/api/submissions/test/execute", headers=HEADERS, json={"confirmation": "go"})
        else:
            response = client.post("/api/submissions/test/execute", headers=HEADERS,
                                   json={"confirmation": "" if failure == "validation" else "go"})
            assert response.status_code == (422 if failure == "validation" else 200)
            if failure == "shutdown":
                assert completed.wait(2)
        assert not app.state.action_executor.busy


def test_long_lanes_are_isolated_and_automatic_admission_is_available(tmp_path):
    app = create_app(settings(tmp_path))
    entered, release = threading.Barrier(3), threading.Event()
    collection = type("ResultCollectionService", (), {"finish": lambda self, pending: pending})()
    build = type("ContainerExecutionService", (), {"finish": lambda self, pending: pending})()
    data = type("DataDeliveryService", (), {"finish": lambda self, pending: pending})()

    def waiting(_pending):
        entered.wait(2)
        assert release.wait(5)

    with TestClient(app):
        waiting_preparations = [app.state.submit_job(waiting, None) for _ in range(2)]
        entered.wait(2)
        try:
            for service in (collection, build, data):
                with app.state.submit_job.reserve(service.finish):
                    assert app.state.submit_job(service.finish, "independent").result(timeout=2) == "independent"
        finally:
            release.set()
            for future in waiting_preparations:
                future.result(timeout=2)


def test_sync_http_handler_consumes_the_middleware_reservation(tmp_path):
    app = create_app(settings(tmp_path))

    def synchronous_handler():
        executor = app.state.action_executor
        assert executor._reserved == 1
        executor.submit(lambda: None).result(timeout=2)
        assert executor._reserved == 0
        return {"accepted": True}

    app.router.routes.insert(0, APIRoute("/api/actions/execute", synchronous_handler, methods=["POST"]))
    with TestClient(app) as client:
        assert client.post("/api/actions/execute", headers=HEADERS, json={}).json() == {"accepted": True}


def test_shutdown_keeps_workspace_for_reserved_request_before_it_has_a_future(tmp_path):
    config = settings(tmp_path)
    app = create_app(config)

    async def admit_and_stop():
        async with app.router.lifespan_context(app):
            reserved = app.state.action_executor.reserve()
        return reserved

    reserved = asyncio.run(admit_and_stop())
    competing = CollectorLease(config.index_db_path())
    try:
        assert not competing.acquire()
        assert reserved.submit(lambda: "admitted owner").result(timeout=2) == "admitted owner"
        deadline = time.monotonic() + 2
        while not competing.acquire():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        competing.release()
    finally:
        reserved.release()


def test_handoff_without_admission_and_error_release():
    executor = BoundedExecutor("test", workers=1, capacity=1)

    async def complete():
        return "complete"

    async def failure():
        raise ValueError("claim rejected")

    async def exercise():
        assert await admitted_handoff(complete) == "complete"
        with executor.reserve():
            assert await admitted_handoff(complete) == "complete"
        with pytest.raises(ValueError, match="claim rejected"):
            with executor.reserve():
                await admitted_handoff(failure)
        await executor.drain_handoffs()

    try:
        asyncio.run(exercise())
        assert not executor.busy
    finally:
        executor.shutdown()


def test_cancelled_handoff_task_releases_unused_admission():
    executor = BoundedExecutor("test", workers=1, capacity=1)

    async def exercise():
        entered = asyncio.Event()

        async def operation():
            entered.set()
            await asyncio.Event().wait()

        with executor.reserve():
            waiter = asyncio.create_task(admitted_handoff(operation))
            await entered.wait()
            next(iter(executor._handoffs)).cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        assert not executor.busy

    try:
        asyncio.run(exercise())
    finally:
        executor.shutdown()


def test_http_cancel_during_claim_keeps_its_reserved_slot_until_enqueue(tmp_path):
    app = create_app(settings(tmp_path))
    entered, release, executed = threading.Event(), threading.Event(), threading.Event()
    effects = []

    def claim(*_args):
        entered.set()
        assert release.wait(5)
        effects.append("claimed exact intent")
        return {"status": "EXECUTING"}, "original pending"

    def finish(pending):
        effects.append(pending)
        executed.set()

    async def cancel_request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            request = asyncio.create_task(client.post("/api/submissions/test/execute", headers=HEADERS,
                                                      json={"confirmation": "go"}))
            assert await asyncio.to_thread(entered.wait, 2)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            await asyncio.sleep(0.05)
            executor = app.state.action_executor
            assert executor._reserved == executor._in_use == 1
            others = [executor.reserve() for _ in range(15)]
            try:
                with pytest.raises(QueueUnavailable):
                    executor.reserve()
                release.set()
                assert await asyncio.to_thread(executed.wait, 2)
                await executor.drain_handoffs()
                assert effects == ["claimed exact intent", "original pending"]
            finally:
                release.set()
                for reservation in others:
                    reservation.release()

    with TestClient(app):
        app.state.submission_service = SimpleNamespace(begin_execute=claim)
        app.state.application.finish_action_execution = finish
        asyncio.run(cancel_request())
        assert not app.state.action_executor.busy


def test_shutdown_drains_disconnected_claim_before_releasing_workspace(tmp_path):
    config = settings(tmp_path)
    app = create_app(config)
    entered, release = threading.Event(), threading.Event()
    competing = CollectorLease(config.index_db_path())

    async def exercise():
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        executor = app.state.action_executor

        async def begin_and_enqueue():
            def claim():
                entered.set()
                assert release.wait(5)
            await run_in_threadpool(claim)
            return app.state.submit_action("exact pending")

        app.state.application.finish_action_execution = lambda pending: pending
        with executor.reserve():
            request = asyncio.create_task(admitted_handoff(begin_and_enqueue))
            assert await asyncio.to_thread(entered.wait, 2)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
        shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
        try:
            await asyncio.sleep(0.05)
            assert not shutdown.done() and not competing.acquire()
            with pytest.raises(QueueUnavailable):
                executor.reserve()
        finally:
            release.set()
            await shutdown
        assert not executor.busy

    asyncio.run(exercise())
    deadline = time.monotonic() + 2
    while not competing.acquire():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    competing.release()


@pytest.mark.parametrize("outcome,status", [("ok", 200), ("application", 404), ("identity", 409)])
def test_repair_uses_collection_worker_and_preserves_errors(tmp_path, monkeypatch, outcome, status):
    app = create_app(settings(tmp_path))

    def repair(self, *_args):
        assert threading.current_thread().name.startswith("collection-executor")
        if outcome == "application":
            raise ApplicationError("unknown Attempt", status_code=404, code="UNKNOWN_ATTEMPT")
        if outcome == "identity":
            raise ValueError("wrong identity")
        return {"repaired": True}

    monkeypatch.setattr(ResultCollectionService, "repair", repair)
    monkeypatch.setattr(ResultCollectionService, "__init__", lambda self, runtime: None)
    with TestClient(app) as client:
        response = client.post("/api/runs/p/r/attempts/attempt-001/collection/repair", headers=HEADERS,
                               json={"expected_sha256": "a" * 64})
        assert response.status_code == status
        assert not app.state.job_executors["collection"].busy


def test_cancelled_repair_retains_worker_capacity(tmp_path, monkeypatch):
    app = create_app(settings(tmp_path))
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def repair(self, *_args):
        entered.set()
        assert release.wait(5)
        finished.set()
        return {"repaired": True}

    monkeypatch.setattr(ResultCollectionService, "repair", repair)
    monkeypatch.setattr(ResultCollectionService, "__init__", lambda self, runtime: None)

    async def cancel_request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            request = asyncio.create_task(client.post("/api/runs/p/r/attempts/attempt-001/collection/repair",
                headers=HEADERS, json={"expected_sha256": "a" * 64}))
            assert await asyncio.to_thread(entered.wait, 2)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            await asyncio.sleep(0.05)
            executor = app.state.job_executors["collection"]
            assert executor._in_use == 1 and executor._reserved == 0
            others = [executor.reserve() for _ in range(3)]
            try:
                response = await client.post("/api/runs/p/r/attempts/attempt-001/collection/repair",
                    headers=HEADERS, json={"expected_sha256": "a" * 64})
                assert response.status_code == 503
            finally:
                release.set()
                for reservation in others:
                    reservation.release()
            assert await asyncio.to_thread(finished.wait, 2)

    with TestClient(app):
        asyncio.run(cancel_request())
