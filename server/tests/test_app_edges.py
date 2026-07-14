"""Daemon lifespan, shutdown, and HTTP-boundary failure edges."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ml_exp_server.api import app as app_module
from ml_exp_server.api.app import (
    _poll_loop, _publisher_loop, _shutdown, _start_daemon_thread, create_app,
)
from ml_exp_server.api_contract import CLIENT_PROTOCOL_HEADER
from ml_exp_server.schemas import ServerConfig


def config(tmp_path: Path, *, collector=False) -> ServerConfig:
    return ServerConfig(
        index_db=str(tmp_path / "index.sqlite"),
        action_root=str(tmp_path / "actions"),
        project_registry_root=str(tmp_path / "registry"),
        projects=[], collector_enabled=collector,
    )


class Lease:
    def __init__(self):
        self.released = 0

    def release(self):
        self.released += 1


class Thread:
    def __init__(self, *, alive=False):
        self.alive = alive
        self.joined = 0

    def join(self, _timeout=None):
        self.joined += 1
        self.alive = False

    def is_alive(self):
        return self.alive


def test_shutdown_without_runtime_releases_lease_and_without_stop_is_safe():
    app = FastAPI()
    lease = Lease()
    app.state.collector_lease = lease
    app.state.runtime = None
    app.state._poll_thread = None
    app.state._publisher_thread = None
    asyncio.run(_shutdown(app))
    assert lease.released == 1

    app.state.collector_lease = None
    asyncio.run(_shutdown(app))


def test_shutdown_closes_runtime_without_lease():
    closed = []
    app = FastAPI()
    app.state.runtime = SimpleNamespace(close=lambda: closed.append(True))
    app.state.collector_lease = None
    app.state._poll_thread = None
    app.state._publisher_thread = None
    asyncio.run(_shutdown(app))
    assert closed == [True]


def test_shutdown_defers_close_until_live_owner_exits():
    closed = []
    stopped = []
    lease = Lease()
    owner = Thread(alive=True)
    owner.join = lambda timeout=None: (
        None if timeout is not None else setattr(owner, "alive", False)
    )
    app = FastAPI()
    app.state._stop = SimpleNamespace(set=lambda: None)
    app.state._poll_thread = owner
    app.state._publisher_thread = None
    app.state.collector_lease = lease
    app.state.runtime = SimpleNamespace(
        close=lambda: closed.append(True),
        wandb_service=SimpleNamespace(stop=lambda: stopped.append(True)),
    )
    asyncio.run(_shutdown(app))
    assert stopped == [True]
    deadline = time.time() + 1
    while not closed and time.time() < deadline:
        time.sleep(0.01)
    assert closed == [True] and lease.released == 1


def test_shutdown_live_owner_without_lease_closes_in_background():
    closed = []
    owner = Thread(alive=True)
    owner.join = lambda timeout=None: (
        None if timeout is not None else setattr(owner, "alive", False)
    )
    app = FastAPI()
    app.state._stop = SimpleNamespace(set=lambda: None)
    app.state._poll_thread = owner
    app.state._publisher_thread = None
    app.state.collector_lease = None
    app.state.runtime = SimpleNamespace(
        close=lambda: closed.append(True),
        wandb_service=SimpleNamespace(stop=lambda: None),
    )
    asyncio.run(_shutdown(app))
    deadline = time.time() + 1
    while not closed and time.time() < deadline:
        time.sleep(0.01)
    assert closed == [True]


def test_http_boundary_rejects_bad_protocol_and_nonowner_mutation(tmp_path):
    app = create_app(config(tmp_path), poll=False)
    with TestClient(app) as client:
        invalid = client.get("/api/projects", headers={CLIENT_PROTOCOL_HEADER: "bad"})
        assert invalid.status_code == 426
        client.app.state.workspace_owner = False
        blocked = client.post(
            "/api/terminal/refresh",
            headers={CLIENT_PROTOCOL_HEADER: "1"},
            json={},
        )
        assert blocked.status_code == 409


def test_lifespan_releases_lease_when_runtime_creation_fails(monkeypatch, tmp_path):
    lease = Lease()

    class OwnedLease:
        def __init__(self, _path):
            pass

        def acquire(self):
            return True

        def release(self):
            lease.release()

    monkeypatch.setattr(app_module, "CollectorLease", OwnedLease)
    monkeypatch.setattr(
        app_module.ExperimentServerRuntime, "create",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("startup")),
    )
    app = create_app(config(tmp_path), poll=False)

    async def run():
        async with app.router.lifespan_context(app):
            pass

    with pytest.raises(RuntimeError, match="startup"):
        asyncio.run(run())
    assert lease.released == 1


class CyclingEvent:
    def __init__(self):
        self.checks = 0
        self.stopped = False

    def reset(self):
        self.checks = 0
        self.stopped = False

    def is_set(self):
        self.checks += 1
        return self.stopped or self.checks > 1

    def wait(self, _timeout=None):
        return False

    def set(self):
        self.stopped = True


def test_background_loops_continue_once_then_finish(tmp_path):
    event = CyclingEvent()
    published = []
    metadata = []
    index = SimpleNamespace(
        set_meta=lambda *args: metadata.append(args), list_runs=lambda _project: [],
    )
    app = FastAPI()
    app.state._stop = event
    app.state.publisher_last_success_at = None
    app.state.publisher_last_error = None
    app.state.publisher_consecutive_failures = 0
    app.state.projects = [SimpleNamespace(project="demo")]
    app.state.index = index
    app.state.runtime = SimpleNamespace(
        observability=SimpleNamespace(
            publish_once=lambda **_kwargs: published.append(True),
            collect_rows=lambda _rows: None,
        ),
    )
    _publisher_loop(app)
    assert published == [True]
    event.reset()
    collector = SimpleNamespace(
        run_cycle=lambda: None,
        config=SimpleNamespace(poll_interval_seconds=1),
    )
    _poll_loop(app, collector)
    assert ("collector_last_error", "") in metadata

    stopped = SimpleNamespace(is_set=lambda: True)
    app.state._stop = stopped
    _publisher_loop(app)
    _poll_loop(app, collector)


def test_daemon_thread_start_failure_is_not_hidden(monkeypatch):
    class StartThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("thread start")

    monkeypatch.setattr(app_module.threading, "Thread", StartThread)
    with pytest.raises(RuntimeError, match="thread start"):
        _start_daemon_thread(target=lambda: None, name="test")
