"""Workspace-scoped Project lifecycle and legacy migration coverage."""

from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import yaml
import pytest
from fastapi.testclient import TestClient

from ml_exp_server.api.app import create_app
from ml_exp_server.cli import main
from ml_exp_server.project_registry import ProjectRegistry, ProjectRegistryError
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


def add_campaign(project_path: Path, campaign: str, run_id: str) -> None:
    campaign_path = project_path.parent / "campaigns" / f"{campaign}.yml"
    campaign_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "project": "demo",
        "campaign": campaign,
        "runs": [{"run_id": run_id, "research_role": "candidate"}],
    }), encoding="utf-8")
    payload = yaml.safe_load(project_path.read_text(encoding="utf-8"))
    payload["campaigns"] = [{
        "name": campaign,
        "file": f"experiments/campaigns/{campaign}.yml",
    }]
    project_path.write_text(yaml.safe_dump(payload), encoding="utf-8")


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


def test_registry_fails_closed_on_corrupt_json(tmp_path):
    registry = ProjectRegistry(tmp_path / "workspace-projects")
    registry.path.write_text("{broken", encoding="utf-8")

    with pytest.raises(ProjectRegistryError, match="invalid project registry"):
        registry.records()
    with pytest.raises(ProjectRegistryError, match="invalid project registry"):
        registry.register("demo", tmp_path / "demo.yaml")

    assert registry.path.read_text(encoding="utf-8") == "{broken"


def test_registry_serializes_writers_from_independent_daemons(tmp_path):
    root = tmp_path / "workspace-projects"
    first = ProjectRegistry(root)
    second = ProjectRegistry(root)
    first.bootstrap([])
    barrier = Barrier(2)

    def register(registry, project):
        barrier.wait(timeout=5)
        registry.register(project, tmp_path / f"{project}.yaml")

    with ThreadPoolExecutor(max_workers=2) as pool:
        calls = [
            pool.submit(register, first, "alpha"),
            pool.submit(register, second, "beta"),
        ]
        for call in calls:
            call.result()

    assert {item.project for item in ProjectRegistry(root).records()} == {"alpha", "beta"}


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


def test_reregister_and_active_transition_refresh_live_project_catalog(tmp_path):
    project_path = write_project(tmp_path, "demo")
    with ExperimentServerRuntime.create(config(tmp_path, [project_path])) as runtime:
        active_projects = runtime.projects
        original = runtime.project("demo")
        original_record = runtime.project_records()[0]
        action_service = runtime.action_service
        safety = runtime.config.action_runtime.model_dump()
        assert original.campaigns == []

        add_campaign(project_path, "study", "run-v1")
        refreshed = runtime.register_project(
            project_path, source=original_record.source,
        )

        assert runtime.projects is active_projects
        assert runtime.project("demo") is refreshed and refreshed is not original
        assert [item.name for item in refreshed.campaigns] == ["study"]
        assert [item.run_id for item in refreshed.campaigns[0].current_revision.memberships] == [
            "run-v1",
        ]
        refreshed_record = runtime.project_records()[0]
        assert refreshed_record.state == original_record.state
        assert refreshed_record.source == original_record.source
        assert refreshed_record.registered_at == original_record.registered_at
        assert runtime.action_service is action_service
        assert runtime.config.action_runtime.model_dump() == safety

        add_campaign(project_path, "study", "run-v2")
        runtime.transition_project("demo", ProjectLifecycleState.ACTIVE)
        transitioned = runtime.project("demo")
        assert runtime.projects is active_projects
        assert transitioned is not refreshed
        assert [
            item.run_id
            for item in transitioned.campaigns[0].current_revision.memberships
        ] == ["run-v2"]
