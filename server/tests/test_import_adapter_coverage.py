from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import yaml
from fastapi.testclient import TestClient

from ml_exp_server.api.app import create_app
from ml_exp_server.application import ExperimentServerApplication
from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.project_config import ConfigError, load_research_project
from ml_exp_server.schemas import (
    ActionRuntimeConfig,
    OperationScope,
    ProjectRef,
    ServerConfig,
)
from ml_exp_server.source_revisions import resolve_source_tree


def test_minimal_embedded_application_reports_import_services_unavailable(tmp_path):
    value = ExperimentServerApplication(SimpleNamespace())

    calls = (
        lambda: value.project_import_preview(tmp_path),
        lambda: value.project_import_execute("import-" + "a" * 24, "confirm"),
        lambda: value.source_revision_preview("demo", {}),
        lambda: value.source_revision_execute("source-import-" + "a" * 24, "confirm"),
        lambda: value.source_revision_get("demo", "source." + "a" * 64),
    )
    for call in calls:
        with pytest.raises(ApplicationError, match="service is unavailable"):
            call()


def test_submit_of_authored_imported_source_requires_source_service(tmp_path, monkeypatch):
    value = ExperimentServerApplication(SimpleNamespace())
    scope = OperationScope(project="demo", scope_type="run", object_id="run-a")
    configured = SimpleNamespace(project="demo", controller=object())
    membership = SimpleNamespace(kind="materialize")
    row = SimpleNamespace(
        scheduler_state="NOT_SUBMITTED", attempts=[],
        campaign_memberships=[SimpleNamespace(
            campaign="study", membership=membership,
        )],
        campaign="study", run_id="run-a",
        provenance={
            "source_binding": "campaign_file", "source_id": "source." + "a" * 64,
        },
    )
    monkeypatch.setattr(value, "_require_operation_available", lambda *_args: None)
    monkeypatch.setattr(value, "resolve_scope", lambda *_args: (scope, configured, row))
    monkeypatch.setattr(value, "_campaign_file", lambda *_args: tmp_path / "study.yml")

    with pytest.raises(ApplicationError, match="source revision service is unavailable"):
        value.prepare_run_submit("demo", "run-a", max_gpu_hours=1)

    def observed_get(*_args):
        raise RuntimeError("source lookup reached")

    value.source_revision_service = SimpleNamespace(get=observed_get)
    with pytest.raises(RuntimeError, match="source lookup reached"):
        value.prepare_run_submit("demo", "run-a", max_gpu_hours=1)


def test_import_route_validation_get_error_and_invalid_campaign_source(tmp_path):
    repository = tmp_path / "repo"
    campaign = repository / "experiments" / "campaigns" / "study.yml"
    campaign.parent.mkdir(parents=True)
    campaign.write_text(yaml.safe_dump({
        "schema_version": 1, "project": "demo", "campaign": "study",
        "runs": [{"run_id": "run-a", "source_id": {"not": "concrete"}}],
    }), encoding="utf-8")
    manifest = repository / "experiments" / "research_project.yaml"
    manifest.write_text(yaml.safe_dump({
        "schema_version": 1, "project": "demo", "title": "Demo", "run_roots": [],
        "campaigns": [{"name": "study", "file": "experiments/campaigns/study.yml"}],
    }), encoding="utf-8")
    with pytest.raises(ConfigError, match="source_id must be a concrete string"):
        load_research_project(manifest)

    campaign.write_text(yaml.safe_dump({
        "schema_version": 1, "project": "demo", "campaign": "study",
        "runs": [{"run_id": "run-a"}],
    }), encoding="utf-8")
    config = ServerConfig(
        index_db=str(tmp_path / "state/index.sqlite"),
        action_root=str(tmp_path / "state/actions"),
        project_registry_root=str(tmp_path / "state/projects"),
        project_import_roots=[str(tmp_path)],
        projects=[ProjectRef(project_file=str(manifest))],
        action_runtime=ActionRuntimeConfig(), collector_enabled=False,
    )
    with TestClient(create_app(config)) as client:
        invalid = client.post("/api/project-imports/preview", json={
            "source": {"kind": "daemon_path", "repository_root": 123},
        })
        missing = client.get(
            "/api/projects/demo/source-revisions/source." + "a" * 64
        )

    assert invalid.status_code == 422
    assert missing.status_code == 404


def test_source_tree_resolver_rejects_identity_metadata_and_missing_tree(tmp_path):
    config = ServerConfig(
        index_db=str(tmp_path / "state/index.sqlite"),
        action_root=str(tmp_path / "state/actions"),
        project_registry_root=str(tmp_path / "state/projects"),
        collector_enabled=False,
    )
    source_id = "source." + "a" * 64
    with pytest.raises(ValueError, match="invalid source revision identity"):
        resolve_source_tree(config, "../bad", source_id)

    source = (
        config.project_registry_root_path() / "source-revisions" / "sources"
        / "demo" / source_id
    )
    source.mkdir(parents=True)
    metadata = source / "source.json"
    metadata.write_text(json.dumps({
        "project": "other", "source_id": source_id, "tree": "tree",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata identity mismatch"):
        resolve_source_tree(config, "demo", source_id)

    metadata.write_text(json.dumps({
        "project": "demo", "source_id": source_id, "tree": "tree",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="tree is unavailable"):
        resolve_source_tree(config, "demo", source_id)

    with TestClient(create_app(config)) as client:
        response = client.get(f"/api/projects/demo/source-revisions/{source_id}")
    assert response.status_code == 409
    assert "unreadable" in response.json()["detail"]
