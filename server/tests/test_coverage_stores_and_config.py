"""Focused branch coverage for durable stores and authored configuration failures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ml_exp_server.actions.store import ActionStore, read_json
from ml_exp_server.agents.store import AgentStore, _proposal_validation, _read_json
from ml_exp_server.project_config import ConfigError, load_server_config, load_projects, load_research_project, load_research_question
from ml_exp_server.schemas import AgentLifecycleState, AgentScope, AgentScopeType, ServerConfig, ProjectRef


def scope(object_id: str = "demo") -> AgentScope:
    return AgentScope(project="demo", scope_type=AgentScopeType.PROJECT, object_id=object_id)


def test_action_store_is_immutable_claimed_once_and_tolerates_corrupt_records(tmp_path):
    store = ActionStore(tmp_path / "actions")
    agent_scope = scope()
    action_id = store.action_id(agent_scope, "proposal-a")
    assert action_id == store.action_id(agent_scope, "proposal-a")
    for invalid in ("wrong", "action-bad/name", "action-"):
        with pytest.raises(ValueError, match="invalid action_id"):
            store.directory(invalid)

    plan = {
        "action_id": action_id,
        "scope": agent_scope.model_dump(mode="json"),
        "ready": False,
        "operation": "BLOCKED_TEST",
    }
    saved = store.save_plan(plan)
    assert saved["execution"]["status"] == "BLOCKED"
    assert store.save_plan({**plan, "operation": "MUTATED"})["operation"] == "BLOCKED_TEST"

    store.claim_execution(action_id)
    with pytest.raises(RuntimeError, match="already been claimed"):
        store.claim_execution(action_id)

    directory = store.directory(action_id)
    (directory / "journal.jsonl").write_text(
        "not-json\n" + json.dumps(["not", "mapping"]) + "\n" +
        json.dumps({"event": "valid"}) + "\n", encoding="utf-8",
    )
    assert store.snapshot(action_id)["journal"] == [{"event": "valid"}]
    (directory / "execution.json").write_text("{broken", encoding="utf-8")
    assert store.execution(action_id) == {}
    updated = store.set_execution(action_id, {"status": "FAILED", "error": "boom"}, event="failed")
    assert updated["execution"]["status"] == "FAILED"

    foreign = scope("other")
    foreign_id = store.action_id(foreign, "proposal-b")
    store.save_plan({
        "action_id": foreign_id, "scope": foreign.model_dump(mode="json"),
        "ready": True, "operation": "OTHER",
    })
    malformed = store.root / "action-malformed"
    malformed.mkdir()
    (malformed / "plan.json").write_text(json.dumps({
        "action_id": "not-valid", "scope": agent_scope.model_dump(mode="json"),
    }))
    assert [item["action_id"] for item in store.list_for_scope(agent_scope)] == [action_id]
    with pytest.raises(FileNotFoundError):
        store.snapshot("action-missing")

    missing = tmp_path / "missing.json"
    assert read_json(missing, {"fallback": True}) == {"fallback": True}
    missing.write_text("[]")
    assert read_json(missing, {}) == []


def test_agent_store_lifecycle_migrations_and_failure_paths(tmp_path):
    store = AgentStore(tmp_path / "agents")
    agent_scope = scope("odd / id")
    assert "odd---id" in store.agent_id(agent_scope)
    snapshot = store.ensure(agent_scope, default_goal="initial")
    assert snapshot["state"] == "IDLE"
    assert store.ensure(agent_scope, default_goal="replacement")["goal"] == "initial"

    store.set_goal(agent_scope, "new goal")
    store.set_state(
        agent_scope, AgentLifecycleState.FAILED,
        current_task="inspect", last_error="failed",
    )
    store.append_message(
        agent_scope, role="user", content="hello", thread_id="thread-1",
        evidence_digest="sha256:e", evidence_captured_at="now",
    )
    result = store.snapshot(agent_scope)
    assert result["goal"] == "new goal"
    assert result["state"] == "FAILED" and result["thread_id"] == "thread-1"

    created = store.add_proposals(agent_scope, [
        {"kind": "CREATE_REPORT_DRAFT", "title": "report", "draft": "body"},
        {"kind": "ANALYSIS_ONLY", "title": "note"},
        {"kind": "ARCHIVE_RUN", "draft": "project: demo"},
    ], evidence_digest="sha256:e")
    report, analysis, invalid = created
    assert report["artifact_id"].startswith("report-")
    assert analysis["validation"]["status"] == "NOT_REQUIRED"
    assert invalid["validation"]["status"] == "INVALID"
    assert store.pending_count("demo") == 3
    assert store.pending_count("absent") == 0

    with pytest.raises(ValueError, match="cannot be approved"):
        store.decide_proposal(agent_scope, invalid["proposal_id"], "APPROVED")
    approval = store.decide_proposal(agent_scope, analysis["proposal_id"], "REJECTED", note="no")
    assert approval["execution_enabled"] is False
    with pytest.raises(ValueError, match="already been decided"):
        store.decide_proposal(agent_scope, analysis["proposal_id"], "REJECTED")
    with pytest.raises(ValueError, match="APPROVED or REJECTED"):
        store.decide_proposal(agent_scope, report["proposal_id"], "MAYBE")
    with pytest.raises(FileNotFoundError):
        store.decide_proposal(agent_scope, "proposal-missing", "REJECTED")
    with pytest.raises(FileNotFoundError):
        store.proposal(agent_scope, "proposal-missing")

    directory = store.agent_dir(agent_scope)
    report_path = directory / "proposals" / f"{report['proposal_id']}.json"
    artifact_path = directory / "draft_artifacts" / f"{report['artifact_id']}.json"
    artifact_path.unlink()
    payload = json.loads(report_path.read_text())
    payload.pop("artifact_id")
    payload.pop("validation")
    report_path.write_text(json.dumps(payload))
    store.ensure(agent_scope, default_goal="ignored")
    migrated = store.proposal(agent_scope, report["proposal_id"])
    assert migrated["artifact_id"].startswith("report-")
    assert migrated["validation"]["status"] == "NOT_REQUIRED"

    (directory / "goal.yaml").unlink()
    (directory / "conversation.json").write_text("{bad")
    (directory / "journal.jsonl").write_text(
        "broken\n" + json.dumps([1]) + "\n" + json.dumps({"event": "kept"}) + "\n"
    )
    degraded = store.snapshot(agent_scope)
    assert degraded["goal"] == "" and degraded["messages"] == []
    assert degraded["journal"] == [{"event": "kept"}]
    assert _read_json(tmp_path / "none", 42) == 42
    (tmp_path / "none").write_text("invalid")
    assert _read_json(tmp_path / "none", 42) == 42


def valid_adapter_payload() -> dict:
    digest = "a" * 64
    return {
        "project": "demo", "title": "Demo",
        "source": {"identity": f"git:{digest}", "required_paths": ["train.py"]},
        "train": {"command": ["python", "train.py", "--out", "{run_dir}", "--x", "{x}"]},
        "container": {
            "image": f"registry/run@sha256:{digest}",
            "base_image": f"registry/base@sha256:{digest}",
            "install_command": ["pip", "install", "--require-hashes", "-r", "requirements.txt"],
        },
        "parameters": {"x": {"type": "integer", "required": True, "default": 1}},
        "outputs": {"metrics": "metrics.jsonl", "checkpoints": "checkpoints", "artifacts": "artifacts"},
        "checkpoint": {"expected_first_minutes": 5, "max_uncheckpointed_minutes": 10},
        "assets": [{"identity": f"dataset:sha256:{digest}"}],
        "backend_profile": {
            "kind": "slurm", "ssh_alias": "gpu", "partition": "p", "account": "a",
            "qos": "q", "gres": "gpu:h100:1", "time": "01:00:00", "mount_root": "/data",
            "source_dir": "/data/source", "sif_path": "/data/image.sif",
        },
        "budget": {"wall_time_minutes": 60, "gpus": 1},
        "metric_contract": {
            "primary": {"name": "loss", "direction": "minimize", "source": "metrics.jsonl", "parser": "jsonl"},
        },
    }


def test_proposal_validation_covers_valid_and_invalid_executable_shapes():
    fenced = "```yaml\n" + yaml.safe_dump(valid_adapter_payload()) + "```"
    assert _proposal_validation("CREATE_PROJECT_ADAPTER_DRAFT", fenced)["status"] == "VALID"
    empty_adapter = _proposal_validation("CREATE_PROJECT_ADAPTER_DRAFT", "{}")
    assert empty_adapter["status"] == "INVALID"
    assert len(empty_adapter["errors"]) >= 8

    campaign = {
        "schema_version": 1, "project": "demo", "campaign": "study",
        "research_contract": {"required_roles": ["a", "b"]},
        "defaults": {"resources": {"gpus": 1}},
        "runs": [
            {"run_id": "run-a", "research_role": "a"},
            {"run_id": "run-b", "template": {"research_role": "b"}},
        ],
    }
    assert _proposal_validation("CREATE_CAMPAIGN_DRAFT", yaml.safe_dump(campaign))["status"] == "VALID"
    invalid_campaign = {
        "schema_version": 2, "default_resources": {}, "budget": {"max_gpu_hours": 0},
        "defaults": {"resources": {"gpus": 0}}, "runs": [{"run_id": "bad/id", "role": "x"}],
        "run_refs": [{"run_id": "bad/id"}], "research_contract": {"required_roles": ["x"]},
    }
    errors = _proposal_validation("UPDATE_CAMPAIGN_DRAFT", yaml.safe_dump(invalid_campaign))["errors"]
    assert any("schema_version" in item for item in errors)
    assert any("unique" in item for item in errors)
    assert any("required_roles" in item for item in errors)

    completion = {
        "project": "demo", "campaign": "study", "revision_id": "r1",
        "evidence_digest": "sha256:x", "outcome": "SUPPORTED", "assessment": "ok",
        "membership_run_ids": ["r"],
    }
    assert _proposal_validation("COMPLETE_CAMPAIGN", yaml.safe_dump(completion))["status"] == "VALID"
    for kind in ("COMPLETE_CAMPAIGN", "ARCHIVE_CAMPAIGN", "CREATE_RESEARCH_QUESTION_DRAFT"):
        assert _proposal_validation(kind, "null")["status"] == "INVALID"
    assert _proposal_validation("ARCHIVE_CAMPAIGN", yaml.safe_dump({
        "project": "demo", "campaign": "c", "revision_id": "r", "reason": "done",
    }))["status"] == "VALID"
    assert _proposal_validation("CREATE_RESEARCH_QUESTION_DRAFT", yaml.safe_dump({
        "id": "Q1", "title": "Question", "status": "OPEN", "links": {}, "assessments": [],
    }))["status"] == "VALID"
    assert _proposal_validation("CREATE_RESEARCH_QUESTION_DRAFT", yaml.safe_dump({
        "id": "bad/id", "status": 1, "links": [], "assessments": {},
    }))["status"] == "INVALID"

    common = {"campaign_file": "c.yml", "run_id": "run", "attempt_id": "attempt-001"}
    for kind, extra in (
        ("SUBMIT_RUN", {"max_gpu_hours": 1}),
        ("RETRY_ATTEMPT", {"max_gpu_hours": 1, "source_attempt_id": "attempt-000"}),
        ("RUN_EVALUATION", {"max_gpu_hours": 1}),
        ("CANCEL_RUN", {"backend_job_id": "job"}),
    ):
        assert _proposal_validation(kind, yaml.safe_dump({**common, **extra}))["status"] == "VALID"
        assert _proposal_validation(kind, "{}")["status"] == "INVALID"
    assert _proposal_validation("SUBMIT_RUN", "[broken")["status"] == "INVALID"


def write_project(tmp_path: Path, project_body: str, campaigns: dict[str, str] | None = None) -> Path:
    experiments = tmp_path / "experiments"
    experiments.mkdir(parents=True, exist_ok=True)
    for name, body in (campaigns or {}).items():
        target = experiments / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    target = experiments / "research_project.yaml"
    target.write_text(project_body)
    return target


@pytest.mark.parametrize(("body", "message"), [
    ("[", "invalid YAML"),
    ("- item\n", "expected a mapping"),
    ("schema_version: 2\nprojects: []\n", "unsupported schema_version"),
    ("schema_version: 1\nprojects: wrong\n", "invalid daemon config"),
])
def test_server_config_failure_matrix(tmp_path, body, message):
    path = tmp_path / "console.yaml"
    path.write_text(body)
    with pytest.raises(ConfigError, match=message):
        load_server_config(path)


def test_project_and_question_schema_version_and_missing_file_failures(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_research_project(tmp_path / "missing.yml")
    question = tmp_path / "q.yml"
    question.write_text("schema_version: 2\nid: Q\ntitle: Question\n")
    with pytest.raises(ConfigError, match="unsupported schema_version"):
        load_research_question(question)
    question.write_text("schema_version: 1\nid: []\n")
    with pytest.raises(ConfigError, match="invalid research question"):
        load_research_question(question)
    project = write_project(tmp_path, "schema_version: 2\nproject: demo\ntitle: Demo\nrun_roots: []\n")
    with pytest.raises(ConfigError, match="unsupported schema_version"):
        load_research_project(project)
    project.write_text("schema_version: 1\nproject: []\n")
    with pytest.raises(ConfigError, match="invalid research project"):
        load_research_project(project)


@pytest.mark.parametrize(("campaign", "message"), [
    ("schema_version: 1\nproject: demo\ncampaign: study\nruns: nope\n", "runs must be a list"),
    ("schema_version: 1\nproject: demo\ncampaign: study\nrun_refs: nope\n", "run_refs must be a list"),
    ("schema_version: 1\nproject: demo\ncampaign: study\nruns: []\n", "requires runs or run_refs"),
    ("schema_version: 1\nproject: demo\ncampaign: study\nrun_refs: [bad]\n", "entries must be mappings"),
    ("schema_version: 1\nproject: demo\ncampaign: study\nrun_refs: [{run_id: 'run-{x}'}]\n", "concrete run_id"),
    ("schema_version: 1\nproject: demo\ncampaign: study\nruns: [{run_id: same}]\nrun_refs: [{run_id: same}]\n", "duplicate campaign run_id"),
])
def test_campaign_authored_shape_failure_matrix(tmp_path, campaign, message):
    project = write_project(
        tmp_path,
        "schema_version: 1\nproject: demo\ntitle: Demo\nrun_roots: []\n"
        "campaigns: [{name: study, file: experiments/study.yml}]\n",
        {"study.yml": campaign},
    )
    with pytest.raises(ConfigError, match=message):
        load_research_project(project)


def test_duplicate_campaigns_materializers_and_projects_are_rejected(tmp_path):
    duplicate_names = write_project(
        tmp_path / "names",
        "schema_version: 1\nproject: demo\ntitle: Demo\nrun_roots: []\n"
        "campaigns: [{name: study}, {name: study}]\n",
    )
    with pytest.raises(ConfigError, match="duplicate campaign name"):
        load_research_project(duplicate_names)

    common = "schema_version: 1\nproject: demo\ncampaign: {name}\nruns: [{{run_id: shared}}]\n"
    duplicate_runs = write_project(
        tmp_path / "runs",
        "schema_version: 1\nproject: demo\ntitle: Demo\nrun_roots: []\ncampaigns:\n"
        "  - {name: a, file: experiments/a.yml}\n  - {name: b, file: experiments/b.yml}\n",
        {"a.yml": common.format(name="a"), "b.yml": common.format(name="b")},
    )
    with pytest.raises(ConfigError, match="materialized by both"):
        load_research_project(duplicate_runs)

    first = write_project(tmp_path / "first", "schema_version: 1\nproject: same\ntitle: A\nrun_roots: []\n")
    second = write_project(tmp_path / "second", "schema_version: 1\nproject: same\ntitle: B\nrun_roots: []\n")
    config = ServerConfig(projects=[ProjectRef(project_file=str(first)), ProjectRef(project_file=str(second))])
    with pytest.raises(ConfigError, match="duplicate project name"):
        load_projects(config)
