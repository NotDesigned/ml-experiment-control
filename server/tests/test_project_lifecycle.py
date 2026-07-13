"""Workspace-scoped Project lifecycle and legacy migration coverage."""

from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from ml_exp_server.api.app import create_app
from ml_exp_server.cli import main
from ml_exp_server.project_registry import ProjectRegistry
from ml_exp_server.runtime import ExperimentServerRuntime
from ml_exp_server.schemas import (
    ServerConfig,
    ProjectLifecycleState,
    ProjectRef,
    ProjectRegistrationSource,
)


def write_project(root: Path, name: str) -> Path:
    path = root / name / "experiments" / "research_project.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "project": name,
        "title": name.title(),
        "run_roots": [],
    }), encoding="utf-8")
    return path


def config(tmp_path: Path, projects: list[Path]) -> ServerConfig:
    return ServerConfig(
        index_db=str(tmp_path / "index.sqlite"),
        action_root=str(tmp_path / "actions"),
        projects=[ProjectRef(project_file=str(path)) for path in projects],
        collector_enabled=False,
    )


def test_registry_bootstraps_once_and_does_not_resurrect_config_sources(tmp_path):
    first = write_project(tmp_path, "first")
    later = write_project(tmp_path, "later")
    registry = ProjectRegistry(tmp_path / "workspace-projects")

    records = registry.bootstrap([first])
    assert [(item.project, item.source) for item in records] == [
        ("first", ProjectRegistrationSource.CONFIG_SEED),
    ]
    assert registry.path.is_file()

    # Configuration is a one-time seed and cannot silently reverse an
    # explicit lifecycle decision on future startups.
    assert [item.project for item in registry.bootstrap([later])] == ["first"]
    registry.unregister("first")
    assert registry.bootstrap([first]) == []
    assert any(item["event"] == "UNREGISTER" for item in registry.events())


def test_runtime_lifecycle_changes_active_collector_set_without_touching_repository(tmp_path):
    project_path = write_project(tmp_path, "demo")
    with ExperimentServerRuntime.create(config(tmp_path, [project_path])) as runtime:
        active_list = runtime.projects
        assert [item.project for item in active_list] == ["demo"]

        paused = runtime.transition_project("demo", ProjectLifecycleState.PAUSED,
                                            reason="maintenance")
        assert paused.state == ProjectLifecycleState.PAUSED
        assert runtime.projects is active_list and runtime.projects == []

        resumed = runtime.transition_project("demo", ProjectLifecycleState.ACTIVE)
        assert resumed.state == ProjectLifecycleState.ACTIVE
        assert [item.project for item in active_list] == ["demo"]

        archived = runtime.transition_project("demo", ProjectLifecycleState.ARCHIVED,
                                              reason="completed")
        assert archived.state == ProjectLifecycleState.ARCHIVED
        restored = runtime.transition_project("demo", ProjectLifecycleState.PAUSED)
        assert restored.state == ProjectLifecycleState.PAUSED
        removed = runtime.unregister_project("demo", reason="remove from console")
        assert removed.project == "demo" and runtime.projects == []
        assert project_path.is_file()  # lifecycle never mutates repository data


def test_project_lifecycle_api_pauses_resumes_archives_and_unregisters(tmp_path):
    project_path = write_project(tmp_path, "demo")
    with TestClient(create_app(config(tmp_path, [project_path]))) as client:
        listed = client.get("/api/project-lifecycle").json()
        assert listed["projects"][0]["state"] == "ACTIVE"

        paused = client.post("/api/project-lifecycle/demo/pause", json={"reason": "hold"})
        assert paused.status_code == 200 and paused.json()["project"]["state"] == "PAUSED"
        assert client.get("/api/projects").json() == []

        resumed = client.post("/api/project-lifecycle/demo/resume", json={})
        assert resumed.status_code == 200 and resumed.json()["active"] is True
        assert client.post("/api/project-lifecycle/demo/archive", json={}).status_code == 409
        archived = client.post("/api/project-lifecycle/demo/archive", json={"reason": "done"})
        assert archived.status_code == 200 and archived.json()["project"]["state"] == "ARCHIVED"
        restored = client.post("/api/project-lifecycle/demo/restore", json={})
        assert restored.status_code == 200 and restored.json()["project"]["state"] == "PAUSED"

        removed = client.post("/api/project-lifecycle/unregister-all", json={"reason": "reset"})
        assert removed.status_code == 200 and [item["project"] for item in removed.json()["unregistered"]] == ["demo"]
        assert client.get("/api/project-lifecycle").json()["projects"] == []
    assert project_path.is_file()
