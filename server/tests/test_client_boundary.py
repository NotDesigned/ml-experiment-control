"""The daemon exposes evidence and Actions, never an Agent runtime."""

from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from ml_exp_server.api.app import create_app
from ml_exp_server.operations import OPERATIONS
from ml_exp_server.schemas import ProjectRef, ResearchQuestion, ServerConfig


def _client(tmp_path: Path) -> TestClient:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    project_file = experiments / "research_project.yaml"
    project_file.write_text(yaml.safe_dump({
        "schema_version": 1,
        "project": "demo",
        "title": "Demo",
        "run_roots": [],
    }))
    config = ServerConfig(
        index_db=str(tmp_path / "index.sqlite"),
        action_root=str(tmp_path / "actions"),
        projects=[ProjectRef(project_file=str(project_file))],
    )
    return TestClient(create_app(config, poll=False))


def test_agent_api_is_absent_and_runtime_has_no_agent_store(tmp_path):
    with _client(tmp_path) as client:
        response = client.get("/api/agent", params={
            "project": "demo", "scope_type": "project", "object_id": "demo",
        })
        assert response.status_code == 404
        assert not hasattr(client.app.state.runtime, "agent_store")


def test_operation_catalog_contains_no_analysis_or_report_operations():
    operation_ids = {operation.operation_id for operation in OPERATIONS}
    assert operation_ids.isdisjoint({
        "research.recommend", "report.generate", "chart.generate",
    })
    assert all(operation.mode in {"intent", "direct"} for operation in OPERATIONS)


def test_terminal_snapshot_contains_evidence_not_client_decisions(tmp_path):
    with _client(tmp_path) as client:
        payload = client.get("/api/terminal/snapshot").json()
        assert "pending_proposals" not in payload
        assert "agent" not in payload


def test_object_read_exposes_neutral_code_identity(tmp_path):
    with _client(tmp_path) as client:
        payload = client.get("/api/objects", params={
            "project": "demo", "scope_type": "project", "object_id": "demo",
        }).json()
        identity = payload["code_identity"]
        assert payload["evidence_digest"].startswith("sha256:")
        assert identity["project_file_relative"] == "experiments/research_project.yaml"
        assert identity["project_file_digest"].startswith("sha256:")
        assert identity["repository"]["kind"] in {"git", "directory"}


def test_server_config_rejects_removed_agent_state_root():
    try:
        ServerConfig(agent_root="/tmp/agents")
    except ValueError as exc:
        assert "agent_root" in str(exc)
    else:  # pragma: no cover - protects the architectural boundary
        raise AssertionError("ServerConfig unexpectedly accepted agent_root")


def test_research_question_cannot_persist_client_scientific_assessment():
    try:
        ResearchQuestion.model_validate({
            "id": "Q1", "title": "Question",
            "assessments": [{"outcome": "SUPPORTED"}],
        })
    except ValueError as exc:
        assert "assessments" in str(exc)
    else:  # pragma: no cover - protects the architectural boundary
        raise AssertionError("daemon accepted a client-owned scientific assessment")
