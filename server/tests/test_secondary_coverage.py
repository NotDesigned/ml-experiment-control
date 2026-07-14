"""Secondary coverage for transport, runtime, process, and storage edges."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from ml_exp_server import runtime as runtime_module
from ml_exp_server.api import app as api_app, routes
from ml_exp_server.application import ApplicationError
from ml_exp_server.ingest.indexer import RunIndex
from ml_exp_server.runtime import ExperimentServerRuntime
from ml_exp_server.schemas import (
    ServerConfig,
    ProjectLifecycleRecord,
    ProjectRegistrationSource,
    ResearchProject,
)


def console_config(tmp_path: Path, **updates) -> ServerConfig:
    values = {
        "index_db": str(tmp_path / "index.sqlite"),
        "action_root": str(tmp_path / "actions"),
        "projects": [],
    }
    values.update(updates)
    return ServerConfig(**values)


def test_runtime_composition_callbacks_registration_lookup_and_close(monkeypatch, tmp_path):
    config = console_config(tmp_path)
    first = ResearchProject(project="first", title="First", run_roots=[])
    second = ResearchProject(project="second", title="Second", run_roots=[])
    monkeypatch.setattr(runtime_module.ProjectRegistry, "bootstrap", lambda self, *_: [
        ProjectLifecycleRecord(
            project="first", project_file="same", source=ProjectRegistrationSource.CONFIG_SEED,
            registered_at="2026-01-01T00:00:00Z", updated_at="2026-01-01T00:00:00Z",
        ),
        ProjectLifecycleRecord(
            project="second", project_file="new", source=ProjectRegistrationSource.MANUAL,
            registered_at="2026-01-01T00:00:00Z", updated_at="2026-01-01T00:00:00Z",
        ),
    ])
    monkeypatch.setattr(
        runtime_module, "load_research_project",
        lambda path: first if path.name == "same" else second,
    )
    observed = []
    runtime = ExperimentServerRuntime.create(
        config, on_index_update=lambda project, run: observed.append(("callback", project, run)),
    )
    assert [item.project for item in runtime.projects] == ["first", "second"]
    runtime.index.on_update("first", "run-a")
    assert observed == [("callback", "first", "run-a")]
    assert runtime.project("first") is first
    with pytest.raises(KeyError, match="unknown project"):
        runtime.project("missing")

    third = ResearchProject(project="third", title="Third", run_roots=[])
    monkeypatch.setattr(runtime_module, "load_research_project", lambda path: third)
    assert runtime.register_project(Path("third.yml")) is third
    assert runtime.register_project(Path("third.yml")) is third
    assert [item.project for item in runtime.projects].count("third") == 1
    assert runtime.__enter__() is runtime
    runtime.__exit__(None, None, None)
    with pytest.raises(Exception):
        runtime.index.list_runs("first")


def test_runtime_with_explicit_index_and_projects_skips_authored_loading(monkeypatch, tmp_path):
    config = console_config(tmp_path)
    index = RunIndex(tmp_path / "provided.sqlite")
    project = ResearchProject(project="demo", title="Demo", run_roots=[])
    runtime = ExperimentServerRuntime.create(config, index=index, projects=[project])
    assert runtime.index is index and runtime.projects == [project]
    runtime.close()


def test_runtime_constructor_failure_closes_every_acquired_resource(monkeypatch, tmp_path):
    closed = []

    class FakeIndex:
        on_update = None

        def close(self):
            closed.append("index")

    class FakeTelemetry:
        def shutdown(self):
            closed.append("telemetry")

    class FakeWandbService:
        def __init__(self, config):
            pass

        def stop(self):
            closed.append("wandb")

    class FakeObservabilityStore:
        def __init__(self, path):
            pass

        def close(self):
            closed.append("observability")

    monkeypatch.setattr(runtime_module, "RunIndex", lambda path: FakeIndex())
    monkeypatch.setattr(
        runtime_module, "initialize_telemetry", lambda config: FakeTelemetry(),
    )
    monkeypatch.setattr(runtime_module, "WandbServiceManager", FakeWandbService)
    monkeypatch.setattr(runtime_module, "ObservabilityStore", FakeObservabilityStore)
    monkeypatch.setattr(
        runtime_module, "ObservabilityCoordinator",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("coordinator failed")),
    )

    with pytest.raises(RuntimeError, match="coordinator failed"):
        ExperimentServerRuntime.create(console_config(tmp_path), projects=[])

    assert closed == ["observability", "wandb", "telemetry", "index"]


def test_runtime_close_attempts_all_resources_after_one_failure(monkeypatch, tmp_path):
    runtime = ExperimentServerRuntime.create(console_config(tmp_path), projects=[])
    closed = []
    real_observability_close = runtime.observability_store.close
    real_index_close = runtime.index.close
    real_telemetry_shutdown = runtime.telemetry.shutdown

    def fail_wandb_stop():
        closed.append("wandb")
        raise RuntimeError("stop failed")

    def close_observability():
        closed.append("observability")
        real_observability_close()

    def close_index():
        closed.append("index")
        real_index_close()

    def close_telemetry():
        closed.append("telemetry")
        real_telemetry_shutdown()

    monkeypatch.setattr(runtime.wandb_service, "stop", fail_wandb_stop)
    monkeypatch.setattr(runtime.observability_store, "close", close_observability)
    monkeypatch.setattr(runtime.index, "close", close_index)
    monkeypatch.setattr(runtime.telemetry, "shutdown", close_telemetry)

    with pytest.raises(RuntimeError, match="runtime cleanup failed.*stop failed"):
        runtime.close()

    assert closed == ["wandb", "observability", "index", "telemetry"]


class FailingApplication:
    def __getattr__(self, name):
        def fail(*args, **kwargs):
            raise ApplicationError(f"{name} failed", status_code=418)
        return fail


def api_request(application=None):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        application=application or FailingApplication(),
        index=SimpleNamespace(), projects=[], collector=None,
        broker=SimpleNamespace(stream=lambda: iter(())),
    )))


def test_route_helpers_and_application_error_mappings(tmp_path):
    with pytest.raises(HTTPException) as missing_run:
        routes._run_or_404(SimpleNamespace(get_run=lambda *a: None), "demo", "missing")
    assert missing_run.value.status_code == 404

    request = api_request()
    calls = [
        lambda: routes.campaign_lifecycle("demo", "study", request),
        lambda: routes.prepare_campaign_archive(
            "demo", "study", routes.ArchiveCampaignRequest(reason="done"), request,
        ),
        lambda: routes.run_detail("demo", "run-a", request),
        lambda: routes.run_metrics("demo", "run-a", request),
        lambda: routes.run_eval("demo", "run-a", request),
        lambda: routes.run_events("demo", "run-a", request),
    ]
    for call in calls:
        with pytest.raises(HTTPException) as caught:
            call()
        assert caught.value.status_code == 418

    with pytest.raises(HTTPException) as caught:
        routes.list_campaign_lifecycle("demo", request)
    assert caught.value.status_code == 404
    response = asyncio.run(routes.stream(request))
    assert response.media_type == "text/event-stream"


def test_app_poll_loop_records_success_and_failure(monkeypatch, tmp_path):
    cycles = []

    class FakeCollector:
        def __init__(self, **kwargs):
            self.config = SimpleNamespace(poll_interval_seconds=0.01)

        def run_cycle(self):
            cycles.append("cycle")
            if len(cycles) == 1:
                raise RuntimeError("poll failed")
            # Leave a deterministic observation window after the successful
            # cycle instead of racing the next cycle's started-at marker.
            self.config.poll_interval_seconds = 10

    monkeypatch.setattr(api_app, "Collector", FakeCollector)
    app = api_app.create_app(console_config(tmp_path), poll=True, projects=[])
    with TestClient(app) as client:
        assert client.get("/api/health").json()["collector_enabled"] is True
        for _ in range(100):
            if len(cycles) >= 2:
                break
            import time
            time.sleep(0.002)
        assert len(cycles) >= 2
        for _ in range(100):
            if app.state.index.get_meta("collector_cycle_started_at") == "":
                break
            import time
            time.sleep(0.002)
        assert app.state.index.get_meta("collector_cycle_started_at") == ""


def test_publisher_loop_exposes_systemic_failure_in_health(monkeypatch, tmp_path):
    class QuietCollector:
        def __init__(self, **kwargs):
            self.config = SimpleNamespace(poll_interval_seconds=10)

        def run_cycle(self):
            pass

    def fail_publish(*, limit_per_target):
        raise RuntimeError("publisher database unavailable")

    monkeypatch.setattr(api_app, "Collector", QuietCollector)
    app = api_app.create_app(console_config(tmp_path), poll=True, projects=[])
    app.state.runtime_initializers.append(
        lambda runtime: setattr(runtime.observability, "publish_once", fail_publish),
    )
    with TestClient(app) as client:
        import time

        for _ in range(100):
            payload = client.get("/api/health").json()
            if payload["publisher"]["last_error"]:
                break
            time.sleep(0.005)

        assert payload["publisher"]["last_error"] == (
            "RuntimeError: publisher database unavailable"
        )
        assert payload["publisher"]["consecutive_failures"] >= 1
        assert payload["publisher"]["last_success_at"] is None


def test_health_exposes_collector_loop_failure(monkeypatch, tmp_path):
    class FailingCollector:
        def __init__(self, **kwargs):
            self.config = SimpleNamespace(poll_interval_seconds=10)

        def run_cycle(self):
            raise RuntimeError("scheduler observation failed")

    monkeypatch.setattr(api_app, "Collector", FailingCollector)
    app = api_app.create_app(console_config(tmp_path), poll=True, projects=[])
    with TestClient(app) as client:
        import time

        for _ in range(100):
            payload = client.get("/api/health").json()
            if payload["collector_error"]:
                break
            time.sleep(0.005)

        assert payload["collector_error"] == "scheduler observation failed"


def test_app_config_factory(tmp_path):
    config_file = tmp_path / "console.yml"
    config_file.write_text("schema_version: 1\nprojects: []\n")
    made = api_app.create_app_from_config_file(config_file)
    assert made.title == "ml-expd"
