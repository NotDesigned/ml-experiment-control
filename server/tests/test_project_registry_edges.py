"""Corruption and lifecycle edge coverage for the workspace Project registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_exp_server.project_registry import ProjectRegistry, ProjectRegistryError
from ml_exp_server.schemas import ProjectLifecycleState
from ml_exp_server.storage import StorageError


NOW = "2026-01-01T00:00:00Z"


def record(project, path, state="ACTIVE"):
    return {
        "project": project, "project_file": str(path), "state": state,
        "source": "MANUAL", "registered_at": NOW, "updated_at": NOW,
    }


@pytest.mark.parametrize(("payload", "message"), [
    ([], "invalid project registry"),
    ({"schema_version": 2, "projects": []}, "unsupported"),
    ({"schema_version": 1, "projects": {}}, "invalid project registry projects"),
])
def test_read_records_rejects_structural_corruption(tmp_path, payload, message):
    root = tmp_path / "registry"
    root.mkdir()
    (root / "registry.json").write_text(json.dumps(payload))
    with pytest.raises(ProjectRegistryError, match=message):
        ProjectRegistry.read_records(root)


def test_read_records_missing_is_side_effect_free(tmp_path):
    root = tmp_path / "missing"
    assert ProjectRegistry.read_records(root) == []
    assert not root.exists()


def test_read_records_maps_invalid_json_storage_error(tmp_path):
    root = tmp_path / "registry"
    root.mkdir()
    (root / "registry.json").write_text("{broken")
    with pytest.raises(ProjectRegistryError, match="invalid project registry"):
        ProjectRegistry.read_records(root)


def test_registry_rejects_invalid_record_and_duplicate_project(tmp_path):
    registry = ProjectRegistry(tmp_path / "registry")
    registry.path.write_text(json.dumps({
        "schema_version": 1, "projects": [{"project": "missing fields"}],
    }))
    with pytest.raises(ProjectRegistryError, match="invalid project lifecycle"):
        registry.records()


def test_active_records_filters_inactive_projects(tmp_path):
    registry = ProjectRegistry(tmp_path / "registry")
    registry.bootstrap([])
    registry.register("active", tmp_path / "active.yml")
    registry.register("paused", tmp_path / "paused.yml")
    registry.transition("paused", ProjectLifecycleState.PAUSED)
    assert [item.project for item in registry.active_records()] == ["active"]


def test_bootstrap_deduplicates_paths_and_rejects_duplicate_project_names(tmp_path):
    first = tmp_path / "first.yml"
    second = tmp_path / "second.yml"
    payload = "schema_version: 1\nproject: demo\ntitle: Demo\nrun_roots: []\n"
    first.write_text(payload)
    second.write_text(payload)

    registry = ProjectRegistry(tmp_path / "deduplicated")
    assert len(registry.bootstrap([first, first])) == 1
    with pytest.raises(ProjectRegistryError, match="duplicate project name"):
        ProjectRegistry(tmp_path / "duplicate").bootstrap([first, second])

    path = (tmp_path / "project.yml").resolve()
    registry.path.write_text(json.dumps({
        "schema_version": 1,
        "projects": [record("same", path), record("same", tmp_path / "other.yml")],
    }))
    with pytest.raises(ProjectRegistryError, match="duplicate project"):
        registry.records()


def test_registry_load_rejects_schema_and_projects_shape(tmp_path):
    registry = ProjectRegistry(tmp_path / "registry")
    registry.path.write_text(json.dumps({"schema_version": 2, "projects": []}))
    with pytest.raises(ProjectRegistryError, match="unsupported"):
        registry.records()
    registry.path.write_text(json.dumps({"schema_version": 1, "projects": {}}))
    with pytest.raises(ProjectRegistryError, match="invalid project registry projects"):
        registry.records()


def test_registry_commit_wraps_storage_error(monkeypatch, tmp_path):
    registry = ProjectRegistry(tmp_path / "registry")
    monkeypatch.setattr(
        registry._state,
        "commit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(StorageError("disk")),
    )
    with pytest.raises(ProjectRegistryError, match="could not commit"):
        registry.bootstrap([])


def test_register_rejects_changed_path_and_archived_state(tmp_path):
    registry = ProjectRegistry(tmp_path / "registry")
    registry.bootstrap([])
    first = tmp_path / "first.yml"
    other = tmp_path / "other.yml"
    registry.register("demo", first)
    with pytest.raises(ProjectRegistryError, match="already registered from"):
        registry.register("demo", other)
    registry.transition("demo", ProjectLifecycleState.ARCHIVED, reason="done")
    with pytest.raises(ProjectRegistryError, match="archived"):
        registry.register("demo", first)


def test_register_reactivates_paused_project(tmp_path):
    registry = ProjectRegistry(tmp_path / "registry")
    registry.bootstrap([])
    path = tmp_path / "project.yml"
    registry.register("demo", path)
    registry.transition("demo", ProjectLifecycleState.PAUSED)
    assert registry.register("demo", path).state == ProjectLifecycleState.ACTIVE


def test_transition_and_unregister_reject_unknown_or_invalid_state(tmp_path):
    registry = ProjectRegistry(tmp_path / "registry")
    registry.bootstrap([])
    with pytest.raises(ProjectRegistryError, match="unknown registered"):
        registry.transition("missing", ProjectLifecycleState.PAUSED)
    with pytest.raises(ProjectRegistryError, match="unknown registered"):
        registry.unregister("missing")

    registry.register("demo", tmp_path / "project.yml")
    current = registry.transition("demo", ProjectLifecycleState.PAUSED)
    assert registry.transition("demo", ProjectLifecycleState.PAUSED) == current
    registry.transition("demo", ProjectLifecycleState.ARCHIVED)
    with pytest.raises(ProjectRegistryError, match="cannot transition"):
        registry.transition("demo", ProjectLifecycleState.ACTIVE)


def test_unregister_all_empty_and_events_without_journal(tmp_path):
    registry = ProjectRegistry(tmp_path / "registry")
    assert registry.unregister_all() == []
    # A brand-new registry has no committed transition journal.
    fresh = ProjectRegistry(tmp_path / "fresh")
    assert fresh.events() == []


def test_events_maps_unreadable_journal(monkeypatch, tmp_path):
    registry = ProjectRegistry(tmp_path / "registry")
    registry.bootstrap([])
    monkeypatch.setattr(
        "ml_exp_server.project_registry._jsonl_mappings",
        lambda _path: (_ for _ in ()).throw(StorageError("bad journal")),
    )
    with pytest.raises(ProjectRegistryError, match="events are unreadable"):
        registry.events()
