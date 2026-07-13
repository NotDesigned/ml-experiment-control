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
from ml_exp_server.actions import service as action_service
from ml_exp_server.actions.service import ActionError, ActionService
from ml_exp_server.actions.store import ActionStore
from ml_exp_server.project_config import load_research_project
from ml_exp_server.schemas import (
    ActionRuntimeConfig,
    AgentScope,
    AgentScopeType,
    ControllerConfig,
    ServerConfig,
    ProjectRef,
    ResearchProject,
)


def approved_proposal(kind: str, draft: str, proposal_id: str = "proposal-123456abcdef"):
    return {
        "proposal_id": proposal_id,
        "kind": kind,
        "status": "APPROVED",
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
        "SCIENTIFIC", "RESOURCE", "EXECUTION_IDENTITY", "METADATA",
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
    scope = AgentScope(
        project="demo", scope_type=AgentScopeType.PROJECT, object_id="demo",
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
        ActionRuntimeConfig(allow_science_writes=True),
    )
    plan = service.prepare(scope, project, approved_proposal(
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
    scope = AgentScope(project="demo", scope_type=AgentScopeType.CAMPAIGN, object_id="study")
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
        ActionRuntimeConfig(allow_science_writes=True),
        actor_provider=lambda: "trusted:test",
    )

    plan = service.prepare(
        scope, project, approved_proposal("UPDATE_CAMPAIGN_DRAFT", updated),
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
        ActionRuntimeConfig(allow_science_writes=True),
    )

    plan = service.prepare(
        AgentScope(project="demo", scope_type=AgentScopeType.PROJECT, object_id="demo"),
        project, approved_proposal("CREATE_CAMPAIGN_DRAFT", draft),
    )

    gates = {gate["name"]: gate["status"] for gate in plan["gates"]}
    assert plan["ready"] is False
    assert gates["membership_schema"] == "FAIL"
    assert gates["required_role_coverage"] == "FAIL"


@pytest.mark.parametrize(("kind", "scope_type", "object_id", "attempt_id"), [
    ("ARCHIVE_RUN", AgentScopeType.RUN, "run-a", None),
    ("ARCHIVE_ATTEMPT", AgentScopeType.ATTEMPT, "run-a::attempt-001", "attempt-001"),
])
def test_run_and_attempt_archives_append_records_without_deleting_evidence(
    tmp_path, kind, scope_type, object_id, attempt_id,
):
    project_root = tmp_path / "science"
    project = ResearchProject(project="demo", title="Demo", run_roots=[], base_dir=project_root)
    scope = AgentScope(project="demo", scope_type=scope_type, object_id=object_id)
    payload = {
        "schema_version": 1, "project": "demo", "run_id": "run-a",
        "reason": "superseded evidence", "evidence_digest": "sha256:evidence",
    }
    if attempt_id:
        payload["attempt_id"] = attempt_id
    service = ActionService(
        ActionStore(tmp_path / "actions"),
        ActionRuntimeConfig(allow_science_writes=True),
        actor_provider=lambda: "trusted:test",
    )
    plan = service.prepare(scope, project, approved_proposal(kind, yaml.safe_dump(payload)))
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
    scope = AgentScope(
        project="demo", scope_type=AgentScopeType.PROJECT, object_id="demo",
    )
    draft = yaml.safe_dump({
        "schema_version": 1,
        "id": "H2",
        "title": "Capacity controls collapse",
        "status": "OPEN",
        "summary": "Separate capacity from objective effects",
        "links": {"campaigns": ["h2-control"]},
        "assessments": [],
    })
    service = ActionService(
        ActionStore(tmp_path / "actions"),
        ActionRuntimeConfig(allow_science_writes=True),
        actor_provider=lambda: "trusted:test-researcher",
    )
    plan = service.prepare(scope, project, approved_proposal(
        "CREATE_RESEARCH_QUESTION_DRAFT", draft,
    ))

    assert plan["ready"] is True
    assert "h2-control" in plan["diff"]
    assert any(
        change["category"] == "SCIENTIFIC"
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
    scope = AgentScope(project="demo", scope_type="project", object_id="demo")
    service = ActionService(ActionStore(tmp_path / "actions"), ActionRuntimeConfig())

    plan = service.prepare(scope, project, approved_proposal(
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
    scope = AgentScope(project="demo", scope_type="project", object_id="demo")
    draft = yaml.safe_dump({
        "schema_version": 1, "id": "H2", "title": "new",
        "status": "OPEN", "summary": "m",
        "links": {"campaigns": ["control"]}, "assessments": [],
    })
    service = ActionService(
        ActionStore(tmp_path / "actions"),
        ActionRuntimeConfig(allow_science_writes=True),
        actor_provider=lambda: "trusted:test-researcher",
    )
    plan = service.prepare(scope, project, approved_proposal(
        "CREATE_RESEARCH_QUESTION_DRAFT", draft,
    ))
    service.authorize(plan["action_id"], "reviewed")
    target.write_text("schema_version: 1\nid: H2\ntitle: concurrent edit\n")

    result = service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")

    assert result["execution"]["status"] == "FAILED"
    assert "target changed" in result["execution"]["error"]
    assert "concurrent edit" in target.read_text()


def test_invalid_agent_research_question_becomes_blocked_review_plan(tmp_path):
    root = tmp_path / "science"
    (root / "experiments" / "research_questions").mkdir(parents=True)
    project = ResearchProject(
        project="demo", title="Demo", run_roots=[],
        research_questions_dir="experiments/research_questions", base_dir=root,
    )
    scope = AgentScope(project="demo", scope_type="project", object_id="demo")
    draft = yaml.safe_dump({
        "schema_version": 1, "id": "H2", "title": "Structured falsifier",
        "status": "OPEN", "summary": "m", "links": [],
    })
    service = ActionService(ActionStore(tmp_path / "actions"), ActionRuntimeConfig())

    plan = service.prepare(scope, project, approved_proposal(
        "CREATE_RESEARCH_QUESTION_DRAFT", draft,
    ))

    assert plan["ready"] is False
    schema = next(item for item in plan["gates"] if item["name"] == "schema")
    assert schema["status"] == "FAIL"
    assert "valid dictionary" in schema["detail"]
    assert "links" in plan["diff"]


class FakeController:
    def __init__(self, *, timeout_submit: bool = False, evaluation: bool = False):
        self.calls: list[list[str]] = []
        self.timeout_submit = timeout_submit
        self.evaluation = evaluation

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
                "run_id": "run-a", "source_id": "git:abc", "image_id": "sha256:image",
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
    scope = AgentScope(
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
        service.prepare(scope, project, approved_proposal(
            "RETRY_ATTEMPT", yaml.safe_dump(payload),
        ))


def test_attempt_scoped_cancel_must_target_exact_attempt(tmp_path):
    project, campaign = controller_project(tmp_path)
    project.controller.capabilities["cancel_outbox"] = True
    scope = AgentScope(
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
        service.prepare(scope, project, approved_proposal("CANCEL_RUN", draft))


def test_action_prepare_rejects_proposal_kind_outside_exact_scope(tmp_path):
    project, campaign = controller_project(tmp_path)
    scope = AgentScope(project="demo", scope_type="project", object_id="demo")
    service = ActionService(
        ActionStore(tmp_path / "actions"), ActionRuntimeConfig(), FakeController(),
    )

    with pytest.raises(ActionError, match="SUBMIT_RUN is not valid in project scope"):
        service.prepare(scope, project, approved_proposal(
            "SUBMIT_RUN", submit_draft(campaign),
        ))


def test_exactly_bound_attempt_retry_can_prepare_and_execute(tmp_path):
    project, campaign = controller_project(tmp_path)
    scope = AgentScope(
        project="demo", scope_type="attempt", object_id="run-a::attempt-001",
    )
    draft = yaml.safe_dump({
        "campaign_file": str(campaign), "run_id": "run-a",
        "source_attempt_id": "attempt-001", "attempt_id": "attempt-002",
        "max_gpu_hours": 8,
    })
    runner = FakeController()
    service = ActionService(
        ActionStore(tmp_path / "actions"),
        ActionRuntimeConfig(allow_scheduler_mutations=True), runner,
        actor_provider=lambda: "trusted:pi",
    )

    plan = service.prepare(scope, project, approved_proposal("RETRY_ATTEMPT", draft))
    assert plan["ready"] is True
    assert plan["run_id"] == "run-a"
    assert plan["attempt_id"] == "attempt-002"
    service.authorize(plan["action_id"], "reviewed retry lineage")
    result = service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")
    assert result["execution"]["status"] == "VERIFIED"


def test_submit_plan_gates_and_executes_once(tmp_path):
    project, campaign = controller_project(tmp_path)
    scope = AgentScope(project="demo", scope_type="run", object_id="run-a")
    runner = FakeController()
    service = ActionService(
        ActionStore(tmp_path / "actions"),
        ActionRuntimeConfig(allow_scheduler_mutations=True), runner,
        actor_provider=lambda: "trusted:pi",
    )
    plan = service.prepare(scope, project, approved_proposal(
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
    assert result["execution"]["result"][0]["backend_job_id"] == "job-123"
    submit_calls = [item for item in runner.calls if item[3] == "submit" and "--dry-run" not in item]
    assert len(submit_calls) == 1

    service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")
    submit_calls = [item for item in runner.calls if item[3] == "submit" and "--dry-run" not in item]
    assert len(submit_calls) == 1


def test_submit_timeout_requires_reconciliation_and_never_retries(tmp_path):
    project, campaign = controller_project(tmp_path)
    scope = AgentScope(project="demo", scope_type="run", object_id="run-a")
    runner = FakeController(timeout_submit=True)
    service = ActionService(
        ActionStore(tmp_path / "actions"),
        ActionRuntimeConfig(allow_scheduler_mutations=True), runner,
        actor_provider=lambda: "trusted:researcher",
    )
    plan = service.prepare(scope, project, approved_proposal(
        "SUBMIT_RUN", submit_draft(campaign), "proposal-fedcba654321",
    ))
    service.authorize(plan["action_id"], "approved")
    result = service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")
    assert result["execution"]["status"] == "RECONCILE_REQUIRED"
    with pytest.raises(ActionError, match="reconcile instead of retrying"):
        service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")
    submit_calls = [item for item in runner.calls if item[3] == "submit" and "--dry-run" not in item]
    assert len(submit_calls) == 1


def test_incomplete_run_identity_and_undeclared_evaluation_are_blocked(tmp_path):
    project, campaign = controller_project(tmp_path)
    project.controller.capabilities["run_identity_v2"] = False
    scope = AgentScope(project="demo", scope_type="run", object_id="run-a")
    service = ActionService(ActionStore(tmp_path / "actions"), ActionRuntimeConfig(), FakeController())
    plan = service.prepare(scope, project, approved_proposal(
        "SUBMIT_RUN", submit_draft(campaign), "proposal-aaaaaaaaaaaa",
    ))
    assert plan["ready"] is False
    assert next(item for item in plan["gates"] if item["name"] == "submit_outbox_capability")["status"] == "FAIL"

    evaluation = service.prepare(scope, project, approved_proposal(
        "RUN_EVALUATION", submit_draft(campaign), "proposal-bbbbbbbbbbbb",
    ))
    assert evaluation["ready"] is False
    assert "does not declare an evaluate verb" in evaluation["gates"][-1]["detail"]


def test_declared_evaluation_is_an_immutable_submit_as_run(tmp_path):
    project, campaign = controller_project(tmp_path)
    project.controller.capabilities["evaluation_as_run"] = True
    scope = AgentScope(project="demo", scope_type="run", object_id="run-a")
    service = ActionService(
        ActionStore(tmp_path / "actions"),
        ActionRuntimeConfig(allow_scheduler_mutations=True),
        FakeController(evaluation=True), actor_provider=lambda: "trusted:pi",
    )
    plan = service.prepare(scope, project, approved_proposal(
        "RUN_EVALUATION", submit_draft(campaign), "proposal-cccccccccccc",
    ))

    assert plan["ready"] is True
    assert next(item for item in plan["gates"] if item["name"] == "evaluation_identity")["status"] == "PASS"
    service.authorize(plan["action_id"], "evaluation reviewed")
    result = service.execute(plan["action_id"], f"EXECUTE {plan['action_id']}")
    assert result["execution"]["status"] == "VERIFIED"


def test_action_api_keeps_proposal_approval_and_execution_authorization_separate(tmp_path):
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
        agent_root=str(tmp_path / "agents"),
        action_root=str(tmp_path / "actions"),
        projects=[ProjectRef(project_file=str(root / "experiments" / "research_project.yaml"))],
    )
    with TestClient(create_app(config)) as client:
        scope = AgentScope(project="demo", scope_type="project", object_id="demo")
        client.app.state.agent_store.ensure(scope, default_goal="manage research")
        created = client.app.state.agent_store.add_proposals(scope, [{
            "kind": "CREATE_RESEARCH_QUESTION_DRAFT", "title": "H2",
            "target": "project://demo/research_questions/H2",
            "change_summary": "add H2", "resource_estimate": "none",
            "rationale": "new question", "risk": "review",
            "draft": yaml.safe_dump({
                "schema_version": 1, "id": "H2", "title": "New",
                "status": "OPEN", "summary": "m",
                "links": {"campaigns": ["control"]}, "assessments": [],
            }),
        }], evidence_digest="sha256:evidence")
        proposal_id = created[0]["proposal_id"]

        before = client.post("/api/actions/prepare", json={
            "project": "demo", "scope_type": "project", "object_id": "demo",
            "proposal_id": proposal_id,
        })
        assert before.status_code == 409

        client.app.state.agent_store.decide_proposal(scope, proposal_id, "APPROVED")
        prepared = client.post("/api/actions/prepare", json={
            "project": "demo", "scope_type": "project", "object_id": "demo",
            "proposal_id": proposal_id,
        })
        assert prepared.status_code == 200
        action_id = prepared.json()["action_id"]
        listed = client.get("/api/actions", params={
            "project": "demo", "scope_type": "project", "object_id": "demo",
        }).json()
        assert [item["action_id"] for item in listed["actions"]] == [action_id]
        assert listed["policy"]["allow_science_writes"] is False

        authorized = client.post("/api/actions/authorize", json={
            "action_id": action_id, "note": "reviewed",
        })
        assert authorized.status_code == 200
        execution = client.post("/api/actions/execute", json={
            "action_id": action_id, "confirmation": f"EXECUTE {action_id}",
        })
        assert execution.status_code == 409
        assert "science writes are disabled" in execution.json()["detail"]
