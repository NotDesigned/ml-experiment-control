"""Phase-3 action plans, gates, approval binding, and idempotent execution."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient
from experiment_control import runner as core_runner

from ml_exp_server.api.app import create_app
from ml_exp_server.actions import project_writes
from ml_exp_server.actions import service as action_service
from ml_exp_server.actions.service import ActionError, ActionService
from ml_exp_server.actions.store import ActionStore
from ml_exp_server.project_config import load_research_project
from ml_exp_server.schemas import (
    ActionRuntimeConfig,
    OperationScope,
    OperationScopeType,
    ControllerConfig,
    ServerConfig,
    ProjectRef,
    ResearchProject,
)


def operation_intent(kind: str, draft: str, idempotency_key: str = "intent-123456abcdef"):
    return {
        "idempotency_key": idempotency_key,
        "kind": kind,
        "title": f"Prepare {kind}",
        "target": "review-target",
        "risk": "review required",
        "draft": draft,
        "evidence_digest": "sha256:evidence",
    }


def test_action_helpers_reject_unsafe_inputs_and_classify_changes(tmp_path):
    assert action_service._file_sha(tmp_path / "missing") is None
    assert action_service._inside(tmp_path / "outside", tmp_path / "root") is False
    with pytest.raises(ActionError, match="valid YAML"):
        action_service._parse_mapping("key: [")
    with pytest.raises(ActionError, match="mapping"):
        action_service._parse_mapping("- item")

    changes = action_service._semantic_changes({}, {
        "links": ["Q1"], "budget": {"gpu": 1}, "command": ["train"], "title": "new",
    })
    assert {item["category"] for item in changes} == {
        "EXPERIMENT_DESIGN", "RESOURCE", "EXECUTION_IDENTITY", "METADATA",
    }
    assert action_service._semantic_changes({"same": 1}, {"same": 1}) == []
    assert action_service._redact({
        "api_token": "secret", "tokenizer_path": "tokenizer.model",
        "train_batch_tokens": 65536,
    }) == {
        "api_token": "[REDACTED]", "tokenizer_path": "tokenizer.model",
        "train_batch_tokens": 65536,
    }


def test_command_runner_handles_timeout_and_non_json_output(monkeypatch, tmp_path):
    runner = action_service.CommandRunner()

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(core_runner.subprocess, "run", timeout)
    assert runner(["slow"], cwd=tmp_path, timeout=1)["timeout"] is True

    monkeypatch.setattr(
        core_runner.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=2, stdout="not json", stderr="https://secret@example.test/error",
        ),
    )
    result = runner(["fail"], cwd=tmp_path, timeout=1)
    assert result["payload"] is None and result["returncode"] == 2
    assert "[REDACTED]@" in result["stderr"]


def test_reuse_only_campaign_prepares_without_scheduler_budget(tmp_path):
    project_root = tmp_path / "science"
    (project_root / "experiments" / "campaigns").mkdir(parents=True)
    project_file = project_root / "experiments" / "research_project.yaml"
    project_file.write_text(
        "schema_version: 1\nproject: demo\ntitle: Demo\nrun_roots: []\ncampaigns: []\n",
        encoding="utf-8",
    )
    project = ResearchProject(
        project="demo", title="Demo", run_roots=[], base_dir=project_root,
        authored_file=project_file,
    )
    scope = OperationScope(
        project="demo", scope_type=OperationScopeType.PROJECT, object_id="demo",
    )
    draft = yaml.safe_dump({
        "schema_version": 1,
        "project": "demo",
        "campaign": "reuse-baseline",
        "research_contract": {"required_roles": ["baseline"]},
        "run_refs": [{"run_id": "existing-run", "research_role": "baseline"}],
    })
    service = ActionService(
        ActionStore(tmp_path / "actions"),
        ActionRuntimeConfig(allow_project_writes=True),
    )
    plan = service.prepare(scope, project, operation_intent(
        "CREATE_CAMPAIGN_DRAFT", draft,
    ))
    assert plan["ready"] is True
    gates = {gate["name"]: gate for gate in plan["gates"]}
    assert gates["resource_budget"]["status"] == "PASS"
    assert len(plan["files"]) == 2
    service.authorize(plan["action_id"], "reviewed catalog registration")
    service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")
    reloaded = load_research_project(project_file)
    assert [item.name for item in reloaded.campaigns] == ["reuse-baseline"]
    assert reloaded.campaigns[0].current_revision is not None


def test_multi_file_project_write_rolls_forward_after_interrupted_replace(
    tmp_path, monkeypatch,
):
    project_root = tmp_path / "science"
    (project_root / "experiments" / "campaigns").mkdir(parents=True)
    project_file = project_root / "experiments" / "research_project.yaml"
    project_file.write_text(
        "schema_version: 1\nproject: demo\ntitle: Demo\nrun_roots: []\ncampaigns: []\n",
        encoding="utf-8",
    )
    project = ResearchProject(
        project="demo", title="Demo", run_roots=[], base_dir=project_root,
        authored_file=project_file,
    )
    scope = OperationScope(
        project="demo", scope_type=OperationScopeType.PROJECT, object_id="demo",
    )
    draft = yaml.safe_dump({
        "schema_version": 1,
        "project": "demo",
        "campaign": "recoverable-study",
        "research_contract": {"required_roles": ["baseline"]},
        "run_refs": [{"run_id": "existing-run", "research_role": "baseline"}],
    })
    store = ActionStore(tmp_path / "actions")
    service = ActionService(
        store, ActionRuntimeConfig(allow_project_writes=True),
        actor_provider=lambda: "trusted:test",
    )
    plan = service.prepare(scope, project, operation_intent(
        "CREATE_CAMPAIGN_DRAFT", draft,
    ))
    service.authorize(plan["action_id"], "reviewed")

    real_write = project_writes._atomic_write_text
    calls = 0

    def interrupt_second_replace(target, content):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected interruption")
        real_write(target, content)

    monkeypatch.setattr(project_writes, "_atomic_write_text", interrupt_second_replace)
    interrupted = service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")

    assert interrupted["execution"]["status"] == "RECONCILE_REQUIRED"
    assert Path(plan["files"][0]["path"]).is_file()
    assert yaml.safe_load(project_file.read_text())["campaigns"] == []
    transaction = json.loads((
        store.directory(plan["action_id"]) / "write_transaction.json"
    ).read_text())
    assert transaction["phase"] == "APPLYING"
    assert transaction["intent_digest"] == plan["intent_digest"]

    monkeypatch.setattr(project_writes, "_atomic_write_text", real_write)
    recovered = ActionService(
        store, ActionRuntimeConfig(allow_project_writes=True),
        actor_provider=lambda: "trusted:test",
    ).reconcile(plan["action_id"])

    assert recovered["execution"]["status"] == "VERIFIED"
    assert [item["name"] for item in yaml.safe_load(
        project_file.read_text(encoding="utf-8"),
    )["campaigns"]] == ["recoverable-study"]


def test_project_write_recovers_effect_to_state_crash_window(tmp_path, monkeypatch):
    root = tmp_path / "science"
    (root / "experiments" / "research_questions").mkdir(parents=True)
    project = ResearchProject(
        project="demo", title="Demo", run_roots=[],
        research_questions_dir="experiments/research_questions", base_dir=root,
    )
    scope = OperationScope(project="demo", scope_type="project", object_id="demo")
    store = ActionStore(tmp_path / "actions")
    service = ActionService(
        store, ActionRuntimeConfig(allow_project_writes=True),
        actor_provider=lambda: "trusted:test",
    )
    plan = service.prepare(scope, project, operation_intent(
        "CREATE_RESEARCH_QUESTION_DRAFT",
        yaml.safe_dump({"schema_version": 1, "id": "Q1", "title": "Recover me"}),
    ))
    service.authorize(plan["action_id"], "reviewed")
    real_set_execution = store.set_execution

    def interrupt_verified_state(action_id, payload, *, event, expected_status=None):
        if event == "project_write_verified":
            raise OSError("injected crash before execution state commit")
        return real_set_execution(
            action_id, payload, event=event, expected_status=expected_status,
        )

    monkeypatch.setattr(store, "set_execution", interrupt_verified_state)
    with pytest.raises(OSError, match="injected crash"):
        service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")

    target = Path(plan["target_path"])
    assert target.is_file()
    assert store.execution(plan["action_id"])["status"] == "EXECUTING"
    transaction = json.loads((
        store.directory(plan["action_id"]) / "write_transaction.json"
    ).read_text())
    assert transaction["phase"] == "APPLIED"

    monkeypatch.setattr(store, "set_execution", real_set_execution)
    blocked = ActionService(
        store, ActionRuntimeConfig(allow_project_writes=False),
    ).recover_pending_project_writes()
    assert blocked[0]["execution"]["status"] == "EXECUTING"
    assert "project writes are disabled" in blocked[0]["execution"]["error"]
    recovered = ActionService(
        store, ActionRuntimeConfig(allow_project_writes=True),
        actor_provider=lambda: "trusted:test",
    ).recover_pending_project_writes()

    assert recovered[0]["execution"]["status"] == "VERIFIED"
    assert yaml.safe_load(target.read_text(encoding="utf-8"))["title"] == "Recover me"


def test_campaign_update_targets_catalog_file_and_changes_revision(tmp_path):
    project_root = tmp_path / "science"
    experiments = project_root / "experiments"
    campaign_file = experiments / "campaigns" / "study.yaml"
    campaign_file.parent.mkdir(parents=True)
    campaign_file.write_text(yaml.safe_dump({
        "schema_version": 1, "project": "demo", "campaign": "study",
        "research_contract": {"required_roles": ["baseline"]},
        "run_refs": [{"run_id": "baseline-run", "research_role": "baseline"}],
    }, sort_keys=False), encoding="utf-8")
    project_file = experiments / "research_project.yaml"
    project_file.write_text(yaml.safe_dump({
        "schema_version": 1, "project": "demo", "title": "Demo", "run_roots": [],
        "campaigns": [{"name": "study", "file": "experiments/campaigns/study.yaml"}],
    }, sort_keys=False), encoding="utf-8")
    project = load_research_project(project_file)
    original_revision = project.campaigns[0].current_revision.revision_id
    scope = OperationScope(project="demo", scope_type=OperationScopeType.CAMPAIGN, object_id="study")
    updated = yaml.safe_dump({
        "schema_version": 1, "project": "demo", "campaign": "study",
        "research_contract": {
            "required_roles": ["baseline", "candidate"],
            "comparison": {"match_fields": ["source_id", "image_id"]},
        },
        "run_refs": [
            {"run_id": "baseline-run", "research_role": "baseline"},
            {"run_id": "candidate-run", "research_role": "candidate"},
        ],
    }, sort_keys=False)
    service = ActionService(
        ActionStore(tmp_path / "actions"),
        ActionRuntimeConfig(allow_project_writes=True),
        actor_provider=lambda: "trusted:test",
    )

    plan = service.prepare(
        scope, project, operation_intent("UPDATE_CAMPAIGN_DRAFT", updated),
    )

    assert plan["ready"] is True
    assert plan["target_path"] == str(campaign_file.resolve())
    assert "required_roles" in plan["diff"]
    service.authorize(plan["action_id"], "reviewed replacement Campaign")
    service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")
    reloaded = load_research_project(project_file)
    assert reloaded.campaigns[0].current_revision.revision_id != original_revision
    assert reloaded.campaigns[0].current_revision.research_contract["comparison"][
        "match_fields"
    ] == ["source_id", "image_id"]


def test_campaign_prepare_blocks_noncanonical_or_missing_membership_roles(tmp_path):
    project_root = tmp_path / "science"
    (project_root / "experiments" / "campaigns").mkdir(parents=True)
    project = ResearchProject(
        project="demo", title="Demo", run_roots=[], base_dir=project_root,
        authored_file=project_root / "experiments" / "research_project.yaml",
    )
    project.authored_file.write_text(
        "schema_version: 1\nproject: demo\ntitle: Demo\nrun_roots: []\ncampaigns: []\n",
        encoding="utf-8",
    )
    draft = yaml.safe_dump({
        "schema_version": 1, "project": "demo", "campaign": "bad-memberships",
        "research_contract": {"required_roles": ["baseline", "candidate"]},
        "run_refs": [
            {"run_id": "baseline-run", "role": "baseline"},
            {"run_id": "candidate-run", "arm": "candidate"},
        ],
    })
    service = ActionService(
        ActionStore(tmp_path / "actions"),
        ActionRuntimeConfig(allow_project_writes=True),
    )

    plan = service.prepare(
        OperationScope(project="demo", scope_type=OperationScopeType.PROJECT, object_id="demo"),
        project, operation_intent("CREATE_CAMPAIGN_DRAFT", draft),
    )

    gates = {gate["name"]: gate["status"] for gate in plan["gates"]}
    assert plan["ready"] is False
    assert gates["membership_schema"] == "FAIL"


@pytest.mark.parametrize(("kind", "scope_type", "object_id", "attempt_id"), [
    ("ARCHIVE_RUN", OperationScopeType.RUN, "run-a", None),
    ("ARCHIVE_ATTEMPT", OperationScopeType.ATTEMPT, "run-a::attempt-001", "attempt-001"),
])
def test_run_and_attempt_archives_append_records_without_deleting_evidence(
    tmp_path, kind, scope_type, object_id, attempt_id,
):
    project_root = tmp_path / "science"
    project = ResearchProject(project="demo", title="Demo", run_roots=[], base_dir=project_root)
    scope = OperationScope(project="demo", scope_type=scope_type, object_id=object_id)
    payload = {
        "schema_version": 1, "project": "demo", "run_id": "run-a",
        "reason": "superseded evidence", "evidence_digest": "sha256:evidence",
    }
    if attempt_id:
        payload["attempt_id"] = attempt_id
    service = ActionService(
        ActionStore(tmp_path / "actions"),
        ActionRuntimeConfig(allow_project_writes=True),
        actor_provider=lambda: "trusted:test",
    )
    plan = service.prepare(scope, project, operation_intent(kind, yaml.safe_dump(payload)))
    assert plan["ready"] is True
    service.authorize(plan["action_id"], "reviewed archive")
    result = service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")
    assert result["execution"]["status"] == "VERIFIED"
    target = Path(plan["target_path"])
    assert target.is_file()
    assert yaml.safe_load(target.read_text())["reason"] == "superseded evidence"


def test_research_question_write_requires_bound_second_approval(tmp_path):
    project_root = tmp_path / "science"
    (project_root / "experiments" / "research_questions").mkdir(parents=True)
    project = ResearchProject(
        project="demo", title="Demo", run_roots=[],
        research_questions_dir="experiments/research_questions", base_dir=project_root,
    )
    scope = OperationScope(
        project="demo", scope_type=OperationScopeType.PROJECT, object_id="demo",
    )
    draft = yaml.safe_dump({
        "schema_version": 1,
        "id": "H2",
        "title": "Capacity controls collapse",
        "status": "OPEN",
        "summary": "Separate capacity from objective effects",
        "links": {"campaigns": ["h2-control"]},
    })
    service = ActionService(
        ActionStore(tmp_path / "actions"),
        ActionRuntimeConfig(allow_project_writes=True),
        actor_provider=lambda: "trusted:test-researcher",
    )
    plan = service.prepare(scope, project, operation_intent(
        "CREATE_RESEARCH_QUESTION_DRAFT", draft,
    ))

    assert plan["ready"] is True
    assert "h2-control" in plan["diff"]
    assert any(
        change["category"] == "EXPERIMENT_DESIGN"
        for change in plan["semantic_changes"]
    )
    assert plan["execution"]["status"] == "PREPARED"
    with pytest.raises(ActionError, match="separate execution authorization"):
        service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")

    authorized = service.authorize(plan["action_id"], "reviewed")
    assert authorized["execution"]["authorization_actor"] == "trusted:test-researcher"
    assert authorized["execution"]["authorized_intent_digest"] == plan["intent_digest"]
    with pytest.raises(ActionError, match="confirmation must equal"):
        service.execute(plan["action_id"], "yes")

    result = service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")
    target = project_root / "experiments" / "research_questions" / "H2.yml"
    assert result["execution"]["status"] == "VERIFIED"
    assert target.is_file()
    assert yaml.safe_load(target.read_text())["id"] == "H2"
    assert yaml.safe_load(target.read_text())["links"]["campaigns"] == ["h2-control"]

    repeated = service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")
    assert repeated["execution"]["result"] == result["execution"]["result"]


def test_minimal_research_question_has_no_campaign_or_falsifiability_gate(tmp_path):
    root = tmp_path / "science"
    (root / "experiments" / "research_questions").mkdir(parents=True)
    project = ResearchProject(
        project="demo", title="Demo", run_roots=[],
        research_questions_dir="experiments/research_questions", base_dir=root,
    )
    scope = OperationScope(project="demo", scope_type="project", object_id="demo")
    service = ActionService(ActionStore(tmp_path / "actions"), ActionRuntimeConfig())

    plan = service.prepare(scope, project, operation_intent(
        "CREATE_RESEARCH_QUESTION_DRAFT",
        yaml.safe_dump({"schema_version": 1, "id": "Q2", "title": "Open note"}),
    ))

    assert plan["ready"] is True
    assert {gate["name"] for gate in plan["gates"]} == {"schema", "safe_target"}


def test_target_change_after_diff_fails_closed(tmp_path):
    root = tmp_path / "science"
    research_question_dir = root / "experiments" / "research_questions"
    research_question_dir.mkdir(parents=True)
    target = research_question_dir / "H2.yml"
    target.write_text("schema_version: 1\nid: H2\ntitle: old\n")
    project = ResearchProject(
        project="demo", title="Demo", run_roots=[],
        research_questions_dir="experiments/research_questions", base_dir=root,
    )
    scope = OperationScope(project="demo", scope_type="project", object_id="demo")
    draft = yaml.safe_dump({
        "schema_version": 1, "id": "H2", "title": "new",
        "status": "OPEN", "summary": "m",
        "links": {"campaigns": ["control"]},
    })
    service = ActionService(
        ActionStore(tmp_path / "actions"),
        ActionRuntimeConfig(allow_project_writes=True),
        actor_provider=lambda: "trusted:test-researcher",
    )
    plan = service.prepare(scope, project, operation_intent(
        "CREATE_RESEARCH_QUESTION_DRAFT", draft,
    ))
    service.authorize(plan["action_id"], "reviewed")
    target.write_text("schema_version: 1\nid: H2\ntitle: concurrent edit\n")

    result = service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")

    assert result["execution"]["status"] == "FAILED"
    assert "target changed" in result["execution"]["error"]
    assert "concurrent edit" in target.read_text()


def test_invalid_client_question_definition_becomes_blocked_action_plan(tmp_path):
    root = tmp_path / "science"
    (root / "experiments" / "research_questions").mkdir(parents=True)
    project = ResearchProject(
        project="demo", title="Demo", run_roots=[],
        research_questions_dir="experiments/research_questions", base_dir=root,
    )
    scope = OperationScope(project="demo", scope_type="project", object_id="demo")
    draft = yaml.safe_dump({
        "schema_version": 1, "id": "H2", "title": "Structured falsifier",
        "status": "OPEN", "summary": "m", "links": [],
    })
    service = ActionService(ActionStore(tmp_path / "actions"), ActionRuntimeConfig())

    plan = service.prepare(scope, project, operation_intent(
        "CREATE_RESEARCH_QUESTION_DRAFT", draft,
    ))

    assert plan["ready"] is False
    schema = next(item for item in plan["gates"] if item["name"] == "schema")
    assert schema["status"] == "FAIL"
    assert "valid dictionary" in schema["detail"]
    assert "links" in plan["diff"]


class FakeController:
    def __init__(
        self, *, timeout_submit: bool = False, evaluation: bool = False,
        status_visible: bool = True, status_job_id: str = "job-123",
        source_id: str = "git:abc",
    ):
        self.calls: list[list[str]] = []
        self.timeout_submit = timeout_submit
        self.evaluation = evaluation
        self.status_visible = status_visible
        self.status_job_id = status_job_id
        self.source_id = source_id

    def __call__(self, command, *, cwd, timeout):
        self.calls.append(list(command))
        verb = command[3]
        if verb == "submit" and "--dry-run" in command:
            campaign = yaml.safe_load(Path(command[2]).read_text())
            root = Path(campaign["local_root"]) / "demo-campaign" / "run-a"
            manifest_path = root / "manifest.yaml"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest = {
                "identity_version": 2,
                "run_id": "run-a", "source_id": self.source_id, "image_id": "sha256:image",
                "backend": {"kind": "slurm", "time": "04:00:00"},
                "resources": {"gpus": 2, "cpus": 8},
                "storage": {"run_dir": "/data/project/runs/run-a", "checkpoint_dir": "/data/project/runs/run-a/checkpoints"},
                "command": ["python", "train.py"],
                "checkpoint": {"expected_first_minutes": 10, "max_uncheckpointed_minutes": 15},
                "assets": [{"kind": "dataset", "identity": "dataset-v1"}],
            }
            if self.evaluation:
                manifest["evaluation"] = {
                    "checkpoint_digest": "sha256:checkpoint",
                    "spec_digest": "sha256:eval-spec",
                    "output_namespace": "eval/run-a/spec-1",
                }
            manifest_path.write_text(yaml.safe_dump(manifest))
            return {"returncode": 0, "timeout": False, "payload": [{
                "run_id": "run-a", "manifest_path": str(manifest_path),
                "scheduler_mutated": False,
            }], "stdout": "", "stderr": ""}
        if verb == "submit":
            if self.timeout_submit:
                return {"returncode": None, "timeout": True, "payload": None,
                        "stdout": "", "stderr": "timeout"}
            return {"returncode": 0, "timeout": False,
                    "payload": [{"run_id": "run-a", "backend_job_id": "job-123"}],
                    "stdout": "", "stderr": ""}
        if verb == "status":
            if not self.status_visible:
                return {"returncode": 0, "timeout": False, "payload": [],
                        "stdout": "", "stderr": ""}
            attempt_id = (
                command[command.index("--attempt-id") + 1]
                if "--attempt-id" in command else "attempt-001"
            )
            return {"returncode": 0, "timeout": False,
                    "payload": [{
                        "run_id": "run-a", "attempt_id": attempt_id,
                        "backend_job_id": self.status_job_id, "state": "QUEUED",
                    }], "stdout": "", "stderr": ""}
        return {"returncode": 0, "timeout": False, "payload": [{"ready": True}],
                "stdout": "", "stderr": ""}


def controller_project(tmp_path):
    root = tmp_path / "science"
    campaign_dir = root / "experiments" / "campaigns"
    campaign_dir.mkdir(parents=True)
    campaign = campaign_dir / "demo.yml"
    campaign.write_text(yaml.safe_dump({
        "schema_version": 1, "project": "demo", "campaign": "demo-campaign",
        "local_root": "outputs/runs", "runs": [{"run_id": "run-a"}],
    }))
    project = ResearchProject(
        project="demo", title="Demo", run_roots=[], base_dir=root,
        controller=ControllerConfig(
            python="python", experimentctl="tools/experimentctl.py", workdir=".",
            capabilities={"submit_outbox": True, "run_identity_v2": True},
        ),
    )
    return project, campaign


def submit_draft(campaign: Path):
    return yaml.safe_dump({
        "campaign_file": str(campaign), "run_id": "run-a",
        "attempt_id": "attempt-001", "max_gpu_hours": 8,
    })


@pytest.mark.parametrize(("draft_updates", "error"), [
    ({"run_id": "run-b"}, "run_id must equal the scoped run_id"),
    ({"source_attempt_id": "attempt-009", "attempt_id": "attempt-002"},
     "source_attempt_id must equal the scoped attempt_id"),
    ({"source_attempt_id": "attempt-001", "attempt_id": "attempt-001"},
     "retry must allocate a new attempt_id"),
])
def test_attempt_scoped_retry_is_strongly_bound_to_source_attempt(
    tmp_path, draft_updates, error,
):
    project, campaign = controller_project(tmp_path)
    scope = OperationScope(
        project="demo", scope_type="attempt", object_id="run-a::attempt-001",
    )
    payload = {
        "campaign_file": str(campaign), "run_id": "run-a",
        "source_attempt_id": "attempt-001", "attempt_id": "attempt-002",
        "max_gpu_hours": 8,
    }
    payload.update(draft_updates)
    service = ActionService(
        ActionStore(tmp_path / "actions"), ActionRuntimeConfig(), FakeController(),
    )

    with pytest.raises(ActionError, match=error):
        service.prepare(scope, project, operation_intent(
            "RETRY_ATTEMPT", yaml.safe_dump(payload),
        ))


def test_attempt_scoped_cancel_must_target_exact_attempt(tmp_path):
    project, campaign = controller_project(tmp_path)
    project.controller.capabilities["cancel_outbox"] = True
    scope = OperationScope(
        project="demo", scope_type="attempt", object_id="run-a::attempt-001",
    )
    draft = yaml.safe_dump({
        "campaign_file": str(campaign), "run_id": "run-a",
        "attempt_id": "attempt-002", "backend_job_id": "job-123",
    })
    service = ActionService(
        ActionStore(tmp_path / "actions"), ActionRuntimeConfig(), FakeController(),
    )

    with pytest.raises(ActionError, match="must target the scoped attempt_id"):
        service.prepare(scope, project, operation_intent("CANCEL_RUN", draft))


def test_action_prepare_rejects_intent_kind_outside_exact_scope(tmp_path):
    project, campaign = controller_project(tmp_path)
    scope = OperationScope(project="demo", scope_type="project", object_id="demo")
    service = ActionService(
        ActionStore(tmp_path / "actions"), ActionRuntimeConfig(), FakeController(),
    )

    with pytest.raises(ActionError, match="SUBMIT_RUN is not valid in project scope"):
        service.prepare(scope, project, operation_intent(
            "SUBMIT_RUN", submit_draft(campaign),
        ))


def test_exactly_bound_attempt_retry_can_prepare_and_execute(tmp_path):
    project, campaign = controller_project(tmp_path)
    legacy_root = (project.base_dir / "outputs" / "runs").resolve()
    legacy_root.mkdir(parents=True)
    project.run_roots = [str(legacy_root)]
    project.daemon_run_root = (tmp_path / "daemon-runs" / "demo").resolve()
    project.daemon_run_root.mkdir(parents=True)
    scope = OperationScope(
        project="demo", scope_type="attempt", object_id="run-a::attempt-001",
    )
    draft = yaml.safe_dump({
        "campaign_file": str(campaign), "run_id": "run-a",
        "source_attempt_id": "attempt-001", "attempt_id": "attempt-002",
        "max_gpu_hours": 8, "local_root": str(legacy_root),
    })
    runner = FakeController()
    service = ActionService(
        ActionStore(tmp_path / "actions"),
        ActionRuntimeConfig(allow_scheduler_mutations=True), runner,
        actor_provider=lambda: "trusted:pi",
    )

    plan = service.prepare(scope, project, operation_intent("RETRY_ATTEMPT", draft))
    assert plan["ready"] is True
    assert plan["run_id"] == "run-a"
    assert plan["attempt_id"] == "attempt-002"
    service.authorize(plan["action_id"], "reviewed retry lineage")
    result = service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")
    assert result["execution"]["status"] == "VERIFIED"
    live_submit = next(
        item for item in runner.calls
        if item[3] == "submit" and "--dry-run" not in item
    )
    execution_campaign = yaml.safe_load(Path(live_submit[2]).read_text())
    assert execution_campaign["local_root"] == str(legacy_root)


def test_submit_plan_gates_and_executes_once(tmp_path):
    project, campaign = controller_project(tmp_path)
    scope = OperationScope(project="demo", scope_type="run", object_id="run-a")
    runner = FakeController()
    service = ActionService(
        ActionStore(tmp_path / "actions"),
        ActionRuntimeConfig(allow_scheduler_mutations=True), runner,
        actor_provider=lambda: "trusted:pi",
    )
    plan = service.prepare(scope, project, operation_intent(
        "SUBMIT_RUN", submit_draft(campaign),
    ))

    assert plan["ready"] is True
    assert all(item["status"] == "PASS" for item in plan["gates"])
    assert plan["preflight_summary"]["run_id"] == "run-a"
    assert plan["preflight_summary"]["resources"]["gpus"] == 2
    assert plan["preflight_summary"]["requested_gpu_hours"] == 8.0
    assert next(
        item for item in plan["gates"] if item["name"] == "duplicate_run_identity"
    )["status"] == "PASS"
    assert plan["intent_digest"].startswith("sha256:")
    service.authorize(plan["action_id"], "budget approved")
    result = service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")
    assert result["execution"]["status"] == "VERIFIED"
    assert result["execution"]["result"]["submission"]["backend_job_id"] == "job-123"
    assert result["execution"]["result"]["observation"]["state"] == "QUEUED"
    submit_calls = [item for item in runner.calls if item[3] == "submit" and "--dry-run" not in item]
    assert len(submit_calls) == 1

    service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")
    submit_calls = [item for item in runner.calls if item[3] == "submit" and "--dry-run" not in item]
    assert len(submit_calls) == 1


def test_submit_plan_binds_preview_to_authored_source_revision(tmp_path):
    project, campaign = controller_project(tmp_path)
    service = ActionService(
        ActionStore(tmp_path / "actions"), ActionRuntimeConfig(), FakeController(),
    )
    scope = OperationScope(project="demo", scope_type="run", object_id="run-a")
    matching = yaml.safe_load(submit_draft(campaign))
    matching["expected_source_id"] = "git:abc"

    accepted = service.prepare(
        scope, project,
        operation_intent("SUBMIT_RUN", yaml.safe_dump(matching), "intent-source-match"),
    )
    gate = next(item for item in accepted["gates"]
                if item["name"] == "authored_source_binding")
    assert gate["status"] == "PASS"

    mismatched = {**matching, "expected_source_id": "source." + "a" * 64}
    blocked = service.prepare(
        scope, project,
        operation_intent("SUBMIT_RUN", yaml.safe_dump(mismatched), "intent-source-drift"),
    )
    gate = next(item for item in blocked["gates"]
                if item["name"] == "authored_source_binding")
    assert gate["status"] == "FAIL"
    assert blocked["ready"] is False


def test_imported_source_tree_is_passed_to_controller_preview_and_checks(tmp_path):
    project, campaign = controller_project(tmp_path)
    project.controller.capabilities["daemon_source_revision"] = True
    source_id = "source." + "a" * 64
    source_root = tmp_path / "immutable-source" / "tree"
    source_root.mkdir(parents=True)
    runner = FakeController(source_id=source_id)
    service = ActionService(
        ActionStore(tmp_path / "actions"),
        ActionRuntimeConfig(allow_scheduler_mutations=True), runner,
        actor_provider=lambda: "trusted:pi",
        source_resolver=lambda project_id, requested: (
            source_root if (project_id, requested) == ("demo", source_id) else None
        ),
    )
    draft = yaml.safe_load(submit_draft(campaign))
    draft["expected_source_id"] = source_id

    plan = service.prepare(
        OperationScope(project="demo", scope_type="run", object_id="run-a"),
        project,
        operation_intent("SUBMIT_RUN", yaml.safe_dump(draft), "intent-imported-source"),
    )

    gates = {item["name"]: item["status"] for item in plan["gates"]}
    assert gates["daemon_source_capability"] == "PASS"
    assert gates["daemon_source_available"] == "PASS"
    assert gates["authored_source_binding"] == "PASS"
    assert plan["ready"] is True
    source_calls = [call for call in runner.calls if "--source-root" in call]
    assert source_calls
    assert all(call[call.index("--source-root") + 1] == str(source_root)
               for call in source_calls)
    assert all(call[call.index("--source-id") + 1] == source_id
               for call in source_calls)
    service.authorize(plan["action_id"], "source binding reviewed")
    executed = service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")
    assert executed["execution"]["status"] == "VERIFIED"

    failing = ActionService(
        ActionStore(tmp_path / "failing-actions"), ActionRuntimeConfig(), runner,
        source_resolver=lambda *_args: (_ for _ in ()).throw(
            ValueError("metadata mismatch")
        ),
    ).prepare(
        OperationScope(project="demo", scope_type="run", object_id="run-a"),
        project,
        operation_intent("SUBMIT_RUN", yaml.safe_dump(draft), "intent-bad-source-store"),
    )
    assert next(item for item in failing["gates"]
                if item["name"] == "daemon_source_available")["status"] == "FAIL"


def test_imported_source_is_revalidated_immediately_before_submit(tmp_path):
    project, campaign = controller_project(tmp_path)
    project.controller.capabilities["daemon_source_revision"] = True
    source_id = "source." + "a" * 64
    source_root = tmp_path / "immutable-source" / "tree"
    source_root.mkdir(parents=True)
    valid = True

    def resolve(project_id, requested):
        if not valid:
            raise ValueError("tree digest mismatch")
        assert (project_id, requested) == ("demo", source_id)
        return source_root

    runner = FakeController(source_id=source_id)
    service = ActionService(
        ActionStore(tmp_path / "actions"),
        ActionRuntimeConfig(allow_scheduler_mutations=True), runner,
        actor_provider=lambda: "trusted:pi", source_resolver=resolve,
    )
    draft = yaml.safe_load(submit_draft(campaign))
    draft["expected_source_id"] = source_id
    plan = service.prepare(
        OperationScope(project="demo", scope_type="run", object_id="run-a"),
        project,
        operation_intent("SUBMIT_RUN", yaml.safe_dump(draft), "intent-source-recheck"),
    )
    assert plan["ready"] is True
    service.authorize(plan["action_id"], "source reviewed")
    valid = False
    result = service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")

    assert result["execution"]["status"] == "FAILED"
    assert "execution-time validation" in result["execution"]["error"]
    actual_submits = [
        call for call in runner.calls if call[3] == "submit" and "--dry-run" not in call
    ]
    assert actual_submits == []


def test_execution_source_revalidation_rejects_missing_resolver_and_command_drift(
    tmp_path,
):
    project, campaign = controller_project(tmp_path)
    project.controller.capabilities["daemon_source_revision"] = True
    source_id = "source." + "a" * 64
    source_root = tmp_path / "source" / "tree"
    source_root.mkdir(parents=True)
    draft = yaml.safe_load(submit_draft(campaign))
    draft["expected_source_id"] = source_id

    def prepared(name):
        runner = FakeController(source_id=source_id)
        service = ActionService(
            ActionStore(tmp_path / name), ActionRuntimeConfig(), runner,
            source_resolver=lambda *_args: source_root,
        )
        plan = service.prepare(
            OperationScope(project="demo", scope_type="run", object_id="run-a"),
            project,
            operation_intent("SUBMIT_RUN", yaml.safe_dump(draft), f"intent-{name}"),
        )
        return service, plan, service.store.snapshot(plan["action_id"])["execution"]

    service, plan, execution = prepared("missing-resolver")
    service.source_resolver = None
    result = service._execute_controller(plan, execution)
    assert result["execution"]["status"] == "FAILED"

    service, plan, execution = prepared("missing-source-id")
    plan["command_preview"].remove("--source-id")
    plan["command_preview"].remove(source_id)
    result = service._execute_controller(plan, execution)
    assert result["execution"]["status"] == "FAILED"

    service, plan, execution = prepared("changed-source-root")
    root_index = plan["command_preview"].index("--source-root") + 1
    plan["command_preview"][root_index] = str(tmp_path / "different")
    result = service._execute_controller(plan, execution)
    assert result["execution"]["status"] == "FAILED"

def test_submit_plan_blocks_existing_canonical_manifest_that_differs_from_preview(
    tmp_path,
):
    project, campaign = controller_project(tmp_path)
    actual = (
        project.base_dir / "outputs" / "runs" / "demo-campaign" / "run-a"
        / "manifest.yaml"
    )
    actual.parent.mkdir(parents=True)
    actual.write_text(yaml.safe_dump({
        "identity_version": 2,
        "run_id": "run-a",
        "source_id": "git:abc",
        "image_id": "sha256:image",
        "git_commit": "older-controller-revision",
    }))
    service = ActionService(
        ActionStore(tmp_path / "actions"), ActionRuntimeConfig(), FakeController(),
    )

    plan = service.prepare(
        OperationScope(project="demo", scope_type="run", object_id="run-a"),
        project,
        operation_intent("SUBMIT_RUN", submit_draft(campaign)),
    )

    gate = next(
        item for item in plan["gates"]
        if item["name"] == "execution_manifest_match"
    )
    assert gate["status"] == "FAIL"
    assert "conflicts with preview" in gate["detail"]
    assert plan["ready"] is False


def test_submit_execute_fails_closed_when_canonical_manifest_appears_after_prepare(
    tmp_path,
):
    project, campaign = controller_project(tmp_path)
    runner = FakeController()
    service = ActionService(
        ActionStore(tmp_path / "actions"),
        ActionRuntimeConfig(allow_scheduler_mutations=True), runner,
        actor_provider=lambda: "trusted:pi",
    )
    plan = service.prepare(
        OperationScope(project="demo", scope_type="run", object_id="run-a"),
        project,
        operation_intent("SUBMIT_RUN", submit_draft(campaign)),
    )
    service.authorize(plan["action_id"], "reviewed")
    actual = Path(plan["execution_manifest_path"])
    actual.parent.mkdir(parents=True)
    actual.write_text("run_id: run-a\ngit_commit: changed-after-review\n")

    with pytest.raises(ActionError, match="canonical execution manifest changed"):
        service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")
    assert not [
        item for item in runner.calls
        if item[3] == "submit" and "--dry-run" not in item
    ]


def test_submit_execute_fails_closed_when_campaign_changes_after_prepare(tmp_path):
    project, campaign = controller_project(tmp_path)
    runner = FakeController()
    service = ActionService(
        ActionStore(tmp_path / "actions"),
        ActionRuntimeConfig(allow_scheduler_mutations=True), runner,
        actor_provider=lambda: "trusted:pi",
    )
    plan = service.prepare(
        OperationScope(project="demo", scope_type="run", object_id="run-a"),
        project,
        operation_intent("SUBMIT_RUN", submit_draft(campaign)),
    )
    service.authorize(plan["action_id"], "reviewed")
    campaign.write_text(campaign.read_text() + "# changed after review\n")

    with pytest.raises(ActionError, match="campaign changed"):
        service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")
    assert not [
        item for item in runner.calls
        if item[3] == "submit" and "--dry-run" not in item
    ]


def test_submit_timeout_requires_reconciliation_and_never_retries(tmp_path):
    project, campaign = controller_project(tmp_path)
    scope = OperationScope(project="demo", scope_type="run", object_id="run-a")
    runner = FakeController(timeout_submit=True, status_visible=False)
    service = ActionService(
        ActionStore(tmp_path / "actions"),
        ActionRuntimeConfig(allow_scheduler_mutations=True), runner,
        actor_provider=lambda: "trusted:researcher",
    )
    plan = service.prepare(scope, project, operation_intent(
        "SUBMIT_RUN", submit_draft(campaign), "intent-fedcba654321",
    ))
    service.authorize(plan["action_id"], "approved")
    result = service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")
    assert result["execution"]["status"] == "RECONCILE_REQUIRED"
    with pytest.raises(ActionError, match="reconcile instead of retrying"):
        service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")
    runner.status_visible = True
    reconciled = service.reconcile(plan["action_id"])
    assert reconciled["execution"]["status"] == "VERIFIED"
    assert reconciled["execution"]["result"]["observation"]["backend_job_id"] == "job-123"
    submit_calls = [item for item in runner.calls if item[3] == "submit" and "--dry-run" not in item]
    assert len(submit_calls) == 1


def test_submit_job_identity_mismatch_requires_status_only_reconciliation(tmp_path):
    project, campaign = controller_project(tmp_path)
    scope = OperationScope(project="demo", scope_type="run", object_id="run-a")
    runner = FakeController(status_job_id="job-other")
    service = ActionService(
        ActionStore(tmp_path / "actions"),
        ActionRuntimeConfig(allow_scheduler_mutations=True), runner,
        actor_provider=lambda: "trusted:researcher",
    )
    plan = service.prepare(scope, project, operation_intent(
        "SUBMIT_RUN", submit_draft(campaign), "intent-job-mismatch",
    ))
    service.authorize(plan["action_id"], "approved")
    uncertain = service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")
    assert uncertain["execution"]["status"] == "RECONCILE_REQUIRED"
    assert "does not match submit result" in uncertain["execution"]["error"]

    runner.status_job_id = "job-123"
    reconciled = service.reconcile(plan["action_id"])
    assert reconciled["execution"]["status"] == "VERIFIED"
    submit_calls = [
        item for item in runner.calls if item[3] == "submit" and "--dry-run" not in item
    ]
    assert len(submit_calls) == 1


def test_incomplete_run_identity_and_undeclared_evaluation_are_blocked(tmp_path):
    project, campaign = controller_project(tmp_path)
    project.controller.capabilities["run_identity_v2"] = False
    scope = OperationScope(project="demo", scope_type="run", object_id="run-a")
    service = ActionService(ActionStore(tmp_path / "actions"), ActionRuntimeConfig(), FakeController())
    plan = service.prepare(scope, project, operation_intent(
        "SUBMIT_RUN", submit_draft(campaign), "intent-aaaaaaaaaaaa",
    ))
    assert plan["ready"] is False
    assert next(item for item in plan["gates"] if item["name"] == "submit_outbox_capability")["status"] == "FAIL"

    evaluation = service.prepare(scope, project, operation_intent(
        "RUN_EVALUATION", submit_draft(campaign), "intent-bbbbbbbbbbbb",
    ))
    assert evaluation["ready"] is False
    assert "does not declare an evaluate verb" in evaluation["gates"][-1]["detail"]


def test_declared_evaluation_is_an_immutable_submit_as_run(tmp_path):
    project, campaign = controller_project(tmp_path)
    project.controller.capabilities["evaluation_as_run"] = True
    scope = OperationScope(project="demo", scope_type="run", object_id="run-a")
    service = ActionService(
        ActionStore(tmp_path / "actions"),
        ActionRuntimeConfig(allow_scheduler_mutations=True),
        FakeController(evaluation=True), actor_provider=lambda: "trusted:pi",
    )
    plan = service.prepare(scope, project, operation_intent(
        "RUN_EVALUATION", submit_draft(campaign), "intent-cccccccccccc",
    ))

    assert plan["ready"] is True
    assert next(item for item in plan["gates"] if item["name"] == "evaluation_identity")["status"] == "PASS"
    service.authorize(plan["action_id"], "evaluation reviewed")
    result = service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")
    assert result["execution"]["status"] == "VERIFIED"


def test_action_api_prepares_client_intent_without_authorizing_execution(tmp_path):
    root = tmp_path / "science"
    research_questions = root / "experiments" / "research_questions"
    research_questions.mkdir(parents=True)
    (root / "experiments" / "research_project.yaml").write_text(yaml.safe_dump({
        "schema_version": 1, "project": "demo", "title": "Demo",
        "run_roots": [], "research_questions_dir": "experiments/research_questions",
    }))
    (research_questions / "H1.yml").write_text(yaml.safe_dump({
        "schema_version": 1, "id": "H1", "title": "Existing",
    }))
    config = ServerConfig(
        index_db=str(tmp_path / "index.sqlite"),
        action_root=str(tmp_path / "actions"),
        projects=[ProjectRef(project_file=str(root / "experiments" / "research_project.yaml"))],
    )
    with TestClient(create_app(config)) as client:
        object_payload = client.get("/api/objects", params={
            "project": "demo", "scope_type": "project", "object_id": "demo",
        }).json()
        intent = {
            "kind": "CREATE_RESEARCH_QUESTION_DRAFT", "title": "H2",
            "target": "project://demo/research_questions/H2",
            "change_summary": "add H2", "resource_estimate": "none",
            "rationale": "new question", "risk": "review",
            "draft": yaml.safe_dump({
                "schema_version": 1, "id": "H2", "title": "New",
                "status": "OPEN", "summary": "m",
                "links": {"campaigns": ["control"]},
            }),
            "evidence_digest": object_payload["evidence_digest"],
            "idempotency_key": "intent-create-h2",
        }
        prepared = client.post("/api/actions/prepare", json={
            "project": "demo", "scope_type": "project", "object_id": "demo",
            "intent": intent,
        })
        assert prepared.status_code == 200
        action_id = prepared.json()["action_id"]
        assert not (research_questions / "H2.yml").exists()
        listed = client.get("/api/actions", params={
            "project": "demo", "scope_type": "project", "object_id": "demo",
        }).json()
        assert [item["action_id"] for item in listed["actions"]] == [action_id]
        assert listed["policy"]["allow_project_writes"] is False

        authorized = client.post("/api/actions/authorize", json={
            "action_id": action_id, "note": "reviewed",
        })
        assert authorized.status_code == 200
        execution = client.post("/api/actions/execute", json={
            "action_id": action_id, "confirmation": f"EXECUTE {action_id}",
        })
        assert execution.status_code == 409
        assert "project writes are disabled" in execution.json()["detail"]
