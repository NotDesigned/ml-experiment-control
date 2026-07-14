"""Behavioral coverage for small daemon infrastructure boundaries."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_exp_server import code_identity, runtime as runtime_module
from ml_exp_server.actions import project_writes
from ml_exp_server.actions.errors import ActionError
from ml_exp_server.actions.policy import ActionExecutionPolicy
from ml_exp_server.actions.project_writes import ProjectWriteError, ProjectWriteTransaction
from ml_exp_server.actions.store import ActionStore
from ml_exp_server.authored_runs import authored_run_placeholder
from ml_exp_server.controller_gateway import CommandRunner, ProjectControllerGateway
from ml_exp_server.operations import intent_scope_error
from ml_exp_server.runtime import ExperimentServerRuntime
from ml_exp_server.schemas import (
    ActionRuntimeConfig,
    CampaignRef,
    ProjectLifecycleRecord,
    ProjectLifecycleState,
    ProjectRegistrationSource,
    ResearchProject,
    ServerConfig,
    ControllerConfig,
)


def _config(tmp_path: Path) -> ServerConfig:
    return ServerConfig(
        index_db=str(tmp_path / "index.sqlite"),
        action_root=str(tmp_path / "actions"),
        projects=[],
    )


def _write_plan(action_id: str, target: Path, content: str = "new: true\n") -> dict:
    return {
        "action_id": action_id,
        "intent_digest": "sha256:intent",
        "operation": "WRITE_CAMPAIGN",
        "files": [{
            "path": str(target),
            "expected_sha256": None,
            "content": content,
        }],
    }


def test_project_write_closes_raw_descriptor_when_setup_fails(monkeypatch, tmp_path):
    target = tmp_path / "target.yml"
    monkeypatch.setattr(
        project_writes.os,
        "fchmod",
        lambda *_: (_ for _ in ()).throw(OSError("chmod failed")),
    )

    with pytest.raises(OSError, match="chmod failed"):
        project_writes._atomic_write_text(target, "value: 1\n")

    assert list(tmp_path.glob(".*.tmp")) == []


def test_project_write_rejects_reused_transaction_with_changed_intent(tmp_path):
    store = ActionStore(tmp_path / "actions")
    transaction = ProjectWriteTransaction(store)
    plan = _write_plan("action-a", tmp_path / "target.yml")
    writes = project_writes._planned_writes(plan)
    transaction._prepare(plan, writes)

    with pytest.raises(ProjectWriteError, match="does not match Action intent"):
        transaction._prepare({**plan, "intent_digest": "sha256:other"}, writes)


def test_project_write_rejects_explicit_empty_file_set(tmp_path):
    with pytest.raises(ProjectWriteError, match="contains no files") as caught:
        project_writes._planned_writes({"files": []})
    assert caught.value.partial is False


def test_project_write_verifies_replaced_content(monkeypatch, tmp_path):
    target = tmp_path / "target.yml"
    transaction = ProjectWriteTransaction(ActionStore(tmp_path / "actions"))
    monkeypatch.setattr(
        project_writes,
        "_atomic_write_text",
        lambda path, _content: path.write_text("wrong: content\n"),
    )

    with pytest.raises(ProjectWriteError, match="replacement verification failed"):
        transaction.apply(_write_plan("action-a", target))


def test_runtime_registered_project_identity_drift_is_rejected(monkeypatch):
    record = ProjectLifecycleRecord(
        project="expected",
        project_file="project.yml",
        source=ProjectRegistrationSource.MANUAL,
        registered_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )
    monkeypatch.setattr(
        runtime_module,
        "load_research_project",
        lambda _path: ResearchProject(project="actual", title="Actual", run_roots=[]),
    )

    with pytest.raises(Exception, match="identity drift"):
        runtime_module._load_registered_project(record)


def test_runtime_observability_executor_validates_plan_shape(tmp_path):
    runtime = ExperimentServerRuntime.create(_config(tmp_path), projects=[])
    try:
        execute = runtime.action_service.internal_executor
        assert execute is not None
        with pytest.raises(ValueError, match="no scope"):
            execute({})
        with pytest.raises(ValueError, match="no Attempts"):
            execute({"scope": {"project": "demo"}, "attempts": "invalid"})
    finally:
        runtime.close()


def test_runtime_cannot_activate_unknown_registered_project(tmp_path):
    runtime = ExperimentServerRuntime.create(_config(tmp_path), projects=[])
    try:
        with pytest.raises(Exception, match="unknown registered project"):
            runtime.transition_project("missing", ProjectLifecycleState.ACTIVE)
    finally:
        runtime.close()


def test_action_policy_blocks_disabled_internal_observability_mutation():
    expires = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    snapshot = {
        "action_id": "action-a",
        "operation": "OBSERVABILITY_BACKFILL",
        "gate_expires_at": expires,
        "intent_digest": "sha256:intent",
        "gate_bundle_digest": "sha256:gates",
        "execution": {
            "status": "AUTHORIZED",
            "authorized_intent_digest": "sha256:intent",
            "authorized_gate_bundle_digest": "sha256:gates",
        },
    }

    with pytest.raises(ActionError, match="observability mutations are disabled"):
        ActionExecutionPolicy(ActionRuntimeConfig()).validate(
            snapshot, "EXECUTE action-a",
        )


def test_code_identity_handles_non_repository_and_external_missing_project_file(
    monkeypatch, tmp_path,
):
    root = tmp_path / "root"
    root.mkdir()
    project = ResearchProject(
        project="demo",
        title="Demo",
        run_roots=[],
        base_dir=root,
        authored_file=str(tmp_path / "outside" / "missing.yml"),
    )
    monkeypatch.setattr(
        code_identity.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )

    identity = code_identity.project_code_identity(project)

    assert identity["repository"]["kind"] == "directory"
    assert identity["project_file_relative"] is None
    assert identity["project_file_digest"] is None


def test_authored_placeholder_skips_campaign_without_revision():
    project = ResearchProject(
        project="demo",
        title="Demo",
        run_roots=[],
        campaigns=[CampaignRef(name="unresolved", current_revision=None)],
    )
    assert authored_run_placeholder(project, "run-a") is None


@pytest.mark.parametrize("stdout", ["not-json", ""])
def test_controller_command_runner_tolerates_non_json_output(stdout, tmp_path):
    result = SimpleNamespace(returncode=0, stdout=stdout, stderr="")
    runner = SimpleNamespace(run=lambda *_args, **_kwargs: result)

    response = CommandRunner(runner)(["controller"], cwd=tmp_path, timeout=1)

    assert response["payload"] is None


def test_controller_gateway_requires_project_controller(tmp_path):
    project = ResearchProject(project="demo", title="Demo", run_roots=[])
    with pytest.raises(ValueError, match="has no controller config"):
        ProjectControllerGateway().build(project, tmp_path / "campaign.yml", "submit", "run-a")


def test_controller_gateway_keeps_absolute_workdir(tmp_path):
    project = ResearchProject(
        project="demo", title="Demo", run_roots=[], base_dir=tmp_path / "base",
        controller=ControllerConfig(
            python="python", experimentctl="controller.py", workdir=str(tmp_path),
        ),
    )
    call = ProjectControllerGateway().build(
        project, tmp_path / "campaign.yml", "submit", "run-a",
    )
    assert call.cwd == tmp_path.resolve()


def test_unknown_intent_kind_has_explicit_scope_error():
    assert intent_scope_error("UNKNOWN", "project") == "unsupported intent kind 'UNKNOWN'"


def test_operation_definition_can_replace_parameters():
    from ml_exp_server.operations import OPERATIONS, REQUEST

    changed = OPERATIONS[0].with_parameters(REQUEST)
    assert changed.parameters == (REQUEST,)
    assert changed.operation_id == OPERATIONS[0].operation_id


def test_http_auth_detects_file_growth_after_metadata_check(monkeypatch, tmp_path):
    from ml_exp_server import http_auth

    token = tmp_path / "token"
    token.write_text("x" * 40)
    token.chmod(0o600)
    monkeypatch.setattr(http_auth.os, "read", lambda *_: b"x" * 4097)

    with pytest.raises(http_auth.HttpAuthError, match="unexpectedly large"):
        http_auth.load_bearer_token(token)
