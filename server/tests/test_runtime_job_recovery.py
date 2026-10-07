"""Image build ownership survives HTTP returns, disconnects and daemon restarts."""
import asyncio
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from ml_exp_server.api.app import create_app
from ml_exp_server.container_execution import ContainerExecutionService
from ml_exp_server.runtime_jobs import recover_interrupted_builds
from ml_exp_server.collectord import CollectorLease
from tests.test_container_api import prepare_runtime, client, import_source, runtime, wait_runtime


def prepared(api, entrypoint=None):
    source = import_source(api)
    return prepare_runtime(api, {
        "source_id": source["source_id"], "dockerfile": "Dockerfile",
        "entrypoint": entrypoint or ["python3", "train.py"]}).json()


def test_daemon_restart_marks_interrupted_build_without_touching_ready_or_rebuilding(tmp_path, monkeypatch):
    fixture = client.__wrapped__(tmp_path, monkeypatch)
    api = next(fixture)
    try:
        ready = runtime(api)
        waiting = prepared(api, ["python3", "train.py"])
        untouched = prepared(api, ["python3", "-V"])
        service = ContainerExecutionService(api.app.state.runtime)
        service.begin_execute("demo", waiting["runtime_id"], waiting["confirmation"])
        original = {p: p.read_bytes() for p in service.root.rglob("*.json*") if waiting["runtime_id"] not in p.name}
        config = api.app.state.config
        calls = []
        builder = __import__("ml_exp_server.container_execution", fromlist=["builder_request"]).builder_request
        def record(socket, payload):
            calls.append(payload["operation"])
            return builder(socket, payload)
        monkeypatch.setattr("ml_exp_server.container_execution.builder_request", record)
    finally:
        fixture.close()
    with TestClient(create_app(config, poll=False)) as restarted:
        assert restarted.app.state.recovered_runtime_builds == 1
        endpoint = "/api/projects/demo/runtimes/" + waiting["runtime_id"]
        assert restarted.get(endpoint).json()["status"] == "RECONCILE_REQUIRED"
        assert calls == [] and all(p.read_bytes() == data for p, data in original.items())
        assert restarted.post(endpoint + "/reconcile", json={"confirmation": waiting["confirmation"]}).json()["status"] == "READY"
        assert calls == ["get"]
        assert restarted.get("/api/projects/demo/runtimes/" + untouched["runtime_id"]).json()["status"] == "PREPARED"
        assert restarted.get("/api/projects/demo/runtimes/" + ready["runtime_id"]).json() == ready


def test_recovery_skips_unrelated_files_and_rejects_mismatched_runtime_identity(client):
    service = ContainerExecutionService(client.app.state.runtime)
    value = prepared(client)
    (service.root / "demo/unrelated.json").write_text("not a runtime")
    (service.root / "bad!project").mkdir()
    (service.root / "bad!project" / (value["runtime_id"] + ".json")).write_text("not a runtime")
    assert recover_interrupted_builds(service) == 0
    with service.state("demo", value["runtime_id"]) as (store, snapshot):
        store.commit({**snapshot.value, "status": "EXECUTING", "project": "other"},
                     expected_revision=snapshot.revision, event={"event": "mismatch_fixture"})
    with pytest.raises(ValueError, match="identity mismatch"):
        recover_interrupted_builds(service)


def test_execute_returns_before_packaging_and_duplicate_request_cannot_claim(client, monkeypatch):
    value = prepared(client)
    endpoint = "/api/projects/demo/runtimes/" + value["runtime_id"]
    started, release = threading.Event(), threading.Event()
    original = __import__("ml_exp_server.container_execution", fromlist=["builder_request"]).builder_request
    def slow(socket, payload):
        started.set()
        assert release.wait(5)
        return original(socket, payload)
    monkeypatch.setattr("ml_exp_server.container_execution.builder_request", slow)
    try:
        response = client.post(endpoint + "/execute", json={"confirmation": value["confirmation"]})
        assert response.status_code == 202 and response.json()["status"] == "EXECUTING"
        assert started.wait(5)
        assert client.get(endpoint).json()["status"] == "EXECUTING"
        assert client.app.state._action_futures
        assert client.post(endpoint + "/execute", json={"confirmation": value["confirmation"]}).status_code == 409
    finally:
        release.set()
    assert wait_runtime(client, endpoint)["status"] == "READY"
    assert client.post(endpoint + "/reconcile", json={"confirmation": value["confirmation"]}).json()["status"] == "READY"


def test_canceled_reconcile_keeps_build_owner_until_receipt_is_saved(client, monkeypatch):
    value = prepared(client)
    endpoint = "/api/projects/demo/runtimes/" + value["runtime_id"]
    release, started = threading.Event(), threading.Event()
    original = __import__("ml_exp_server.container_execution", fromlist=["builder_request"]).builder_request
    def slow(socket, payload):
        assert payload["operation"] == "get"
        started.set()
        assert release.wait(5)
        return original(socket, payload)
    monkeypatch.setattr("ml_exp_server.container_execution.builder_request", slow)
    async def disconnect():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=client.app), base_url="http://test",
                                     headers={"X-ML-Expd-Client-Protocol": "2"}) as api:
            request = asyncio.create_task(api.post(endpoint + "/reconcile", json={"confirmation": value["confirmation"]}))
            assert await asyncio.to_thread(started.wait, 5)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            assert client.app.state._action_futures
    try:
        asyncio.run(disconnect())
    finally:
        release.set()
    assert wait_runtime(client, endpoint)["status"] == "READY"


def test_executor_rejection_leaves_inspectable_runtime(client, monkeypatch):
    value = prepared(client)
    def unavailable(*args): raise RuntimeError("shutting down")
    monkeypatch.setattr(client.app.state, "submit_job", unavailable)
    endpoint = "/api/projects/demo/runtimes/" + value["runtime_id"]
    assert client.post(endpoint + "/execute", json={"confirmation": value["confirmation"]}).status_code == 503
    assert client.get(endpoint).json()["status"] == "RECONCILE_REQUIRED"


def test_prepare_runtimed_runtime_logs_can_fall_back_to_current_recipe(client, monkeypatch):
    value = prepared(client)
    service = ContainerExecutionService(client.app.state.runtime)
    with service.state("demo", value["runtime_id"]) as (store, snapshot):
        legacy = dict(snapshot.value)
        legacy.pop("build_bundle_id")
        store.commit(legacy, expected_revision=snapshot.revision, event={"event": "legacy_fixture"})
    def logs(socket, payload):
        assert payload["operation"] == "logs" and "bundle_id" not in payload
        return {"lines": []}
    monkeypatch.setattr("ml_exp_server.api.container_routes.builder_request", logs)
    endpoint = "/api/projects/demo/runtimes/" + value["runtime_id"] + "/logs"
    assert client.get(endpoint).json() == {"lines": []}


def test_shutdown_keeps_exclusive_workspace_until_image_build_finishes(tmp_path, monkeypatch):
    fixture = client.__wrapped__(tmp_path, monkeypatch)
    api = next(fixture)
    release, started = threading.Event(), threading.Event()
    original = __import__("ml_exp_server.container_execution", fromlist=["builder_request"]).builder_request
    def slow(socket, payload):
        started.set()
        assert release.wait(5)
        return original(socket, payload)
    monkeypatch.setattr("ml_exp_server.container_execution.builder_request", slow)
    value = prepared(api)
    config = api.app.state.config
    endpoint = "/api/projects/demo/runtimes/" + value["runtime_id"]
    try:
        assert api.post(endpoint + "/execute", json={"confirmation": value["confirmation"]}).status_code == 202
        assert started.wait(5)
        fixture.close()
        with pytest.raises(RuntimeError, match="another ml-expd process owns"):
            with TestClient(create_app(config, poll=False)):
                pass
    finally:
        release.set()
        fixture.close()
    lease = CollectorLease(config.index_db_path())
    deadline = time.monotonic() + 5
    while not lease.acquire():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    lease.release()
    with TestClient(create_app(config, poll=False)) as restarted:
        assert restarted.get(endpoint).json()["status"] == "READY"
        assert restarted.app.state.recovered_runtime_builds == 0
