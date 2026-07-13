"""Campaign lifecycle validation, planning, completion, and archival."""

from __future__ import annotations

import hashlib
import json
import math
import textwrap
from pathlib import Path

import pytest
import yaml

from ml_exp_server.application import ApplicationError, ExperimentServerApplication
from ml_exp_server.campaign_lifecycle import (
    _required_artifacts,
    _required_metrics,
    _terminal_check,
    campaign_record_path,
    campaign_snapshot,
)
from ml_exp_server.ingest.indexer import RunIndex, index_project
from ml_exp_server.project_config import load_research_project
from ml_exp_server.runtime import ExperimentServerRuntime
from ml_exp_server.schemas import (
    ActionRuntimeConfig,
    AgentScopeType,
    ServerConfig,
    ProjectRef,
)


def _science_project(tmp_path: Path, *, missing_ref: bool = False):
    experiments = tmp_path / "experiments"
    experiments.mkdir(parents=True)
    campaign_path = experiments / "study.yml"
    run_refs = (
        "run_refs:\n  - {run_id: missing-baseline, research_role: baseline}\n"
        if missing_ref else ""
    )
    campaign_text = textwrap.dedent("""\
        schema_version: 1
        project: demo
        campaign: study
        research_contract:
          required_roles: [control, treatment]
          required_metrics:
            common: [train_loss]
          required_artifacts:
            common:
              train_metrics: {min_records: 1}
          terminal_checks:
            - {metric: train_loss, op: lt, value: 2.0}
          comparison:
            match_fields: [source_id, image_id]
        runs:
          - {run_id: control-run, research_role: control}
          - {run_id: treatment-run, research_role: treatment}
    """) + run_refs
    campaign_path.write_text(campaign_text)
    revision_id = f"campaign.{hashlib.sha256(campaign_text.encode()).hexdigest()}"
    project_path = experiments / "research_project.yaml"
    project_path.write_text(textwrap.dedent("""\
        schema_version: 1
        project: demo
        title: Demo lifecycle
        run_roots: [runs]
        campaigns:
          - {name: study, file: experiments/study.yml}
    """))
    for run_id, role, loss in (
        ("control-run", "control", 1.0),
        ("treatment-run", "treatment", 0.8),
    ):
        run_dir = tmp_path / "runs" / "study" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.yaml").write_text(textwrap.dedent(f"""\
            schema_version: 2
            project: demo
            campaign: study
            campaign_id: {revision_id}
            run_id: {run_id}
            research_role: {role}
            source_id: source-v1
            image_id: sha256:image-v1
        """))
        (run_dir / "status.json").write_text(json.dumps({"state": "SUCCEEDED"}))
        (run_dir / "collection.json").write_text(json.dumps({
            "train_loss": loss,
            "artifacts": {"train_metrics": {"records": 1}},
        }))
    return project_path


def _runtime(tmp_path: Path):
    project_path = _science_project(tmp_path)
    config = ServerConfig(
        index_db=str(tmp_path / "index.sqlite"),
        agent_root=str(tmp_path / "agents"),
        action_root=str(tmp_path / "actions"),
        action_runtime=ActionRuntimeConfig(allow_science_writes=True),
        projects=[ProjectRef(project_file=str(project_path))],
    )
    runtime = ExperimentServerRuntime.create(config)
    index_project(runtime.index, runtime.project("demo"))
    return runtime


def _approve_execute(application, proposal):
    campaign = "study"
    proposal_id = proposal["proposal_id"]
    application.decide_proposal(
        "demo", AgentScopeType.CAMPAIGN, campaign, proposal_id, "APPROVED",
    )
    action = application.prepare_action(
        "demo", AgentScopeType.CAMPAIGN, campaign, proposal_id,
    )
    application.authorize_action(action["action_id"], "reviewed")
    return application.execute_action(
        action["action_id"], f"EXECUTE {action['action_id']}",
    )


def test_campaign_snapshot_is_completable_only_after_scientific_gates_pass(tmp_path):
    project = load_research_project(_science_project(tmp_path))
    index = RunIndex(tmp_path / "index.sqlite")
    index_project(index, project)

    status = campaign_snapshot(index, project, "study")

    assert status["lifecycle_state"] == "COMPLETABLE"
    assert status["validation"]["status"] == "PASS"
    assert status["completion"]["ready"] is True
    assert {item["action"] for item in status["plan"]} == {"USE_EVIDENCE"}
    assert all(gate["status"] == "PASS" for gate in status["completion"]["gates"])


def test_campaign_comparison_reads_json_harness_manifests(tmp_path):
    project_path = _science_project(tmp_path)
    for run_id in ("control-run", "treatment-run"):
        run_dir = tmp_path / "runs" / "study" / run_id
        yaml_path = run_dir / "manifest.yaml"
        payload = yaml.safe_load(yaml_path.read_text())
        payload["source"] = {"git_commit": "same-source"}
        (run_dir / "manifest.json").write_text(json.dumps(payload))
        yaml_path.unlink()
    project = load_research_project(project_path)
    project.campaigns[0].current_revision.research_contract["comparison"] = {
        "match_fields": ["source.git_commit"],
    }
    index = RunIndex(tmp_path / "index.sqlite")
    index_project(index, project)

    status = campaign_snapshot(index, project, "study")

    comparison = next(
        gate for gate in status["completion"]["gates"] if gate["name"] == "comparability"
    )
    assert comparison["status"] == "PASS"


def test_campaign_comparison_is_pending_when_a_membership_is_unresolved(tmp_path):
    project = load_research_project(_science_project(tmp_path))
    project.campaigns[0].current_revision.research_contract["comparison"] = {
        "match_fields": ["project"],
    }
    treatment = tmp_path / "runs" / "study" / "treatment-run"
    for path in treatment.iterdir():
        path.unlink()
    treatment.rmdir()
    index = RunIndex(tmp_path / "index.sqlite")
    index_project(index, project)

    status = campaign_snapshot(index, project, "study")

    comparison = next(
        gate for gate in status["completion"]["gates"] if gate["name"] == "comparability"
    )
    assert comparison["status"] == "PENDING"
    assert comparison["evidence"]["missing"]["project"] == ["treatment-run"]


def test_missing_explicit_run_ref_makes_campaign_invalid(tmp_path):
    project = load_research_project(_science_project(tmp_path, missing_ref=True))
    index = RunIndex(tmp_path / "index.sqlite")
    index_project(index, project)

    status = campaign_snapshot(index, project, "study")

    assert status["lifecycle_state"] == "INVALID"
    assert any(item["action"] == "INVALID_REF" for item in status["plan"])
    assert status["completion"]["ready"] is False


def test_campaign_without_research_contract_cannot_claim_scientific_completion(tmp_path):
    project = load_research_project(_science_project(tmp_path))
    project.campaigns[0].current_revision.research_contract = None
    index = RunIndex(tmp_path / "index.sqlite")
    index_project(index, project)

    status = campaign_snapshot(index, project, "study")

    assert status["lifecycle_state"] == "INVALID"
    assert status["validation"]["status"] == "FAIL"
    assert status["completion"]["ready"] is False
    validation = {gate["name"]: gate for gate in status["validation"]["gates"]}
    completion = {gate["name"]: gate for gate in status["completion"]["gates"]}
    assert validation["research_contract"]["status"] == "FAIL"
    assert validation["required_roles"]["status"] == "FAIL"
    assert completion["research_contract"]["status"] == "FAIL"


def test_completion_and_archive_are_immutable_controlled_records(tmp_path):
    runtime = _runtime(tmp_path)
    application = ExperimentServerApplication(runtime)

    completion = application.propose_campaign_completion(
        "demo", "study", outcome="SUPPORTED", assessment="treatment beats control",
    )
    result = _approve_execute(application, completion["proposal"])
    assert result["execution"]["status"] == "VERIFIED"
    completed = application.campaign_status("demo", "study")
    assert completed["lifecycle_state"] == "COMPLETED"
    assert completed["records"]["completion"]["outcome"] == "SUPPORTED"

    archive = application.propose_campaign_archive(
        "demo", "study", reason="decision recorded; leave active research",
    )
    result = _approve_execute(application, archive["proposal"])
    assert result["execution"]["status"] == "VERIFIED"
    archived = application.campaign_status("demo", "study")
    assert archived["lifecycle_state"] == "ARCHIVED"
    assert archived["records"]["archive"]["reason"].startswith("decision recorded")
    runtime.close()


def test_completion_execution_rejects_evidence_drift_after_authorization(tmp_path):
    runtime = _runtime(tmp_path)
    application = ExperimentServerApplication(runtime)
    proposed = application.propose_campaign_completion(
        "demo", "study", outcome="SUPPORTED", assessment="ready",
    )["proposal"]
    application.decide_proposal(
        "demo", AgentScopeType.CAMPAIGN, "study", proposed["proposal_id"], "APPROVED",
    )
    action = application.prepare_action(
        "demo", AgentScopeType.CAMPAIGN, "study", proposed["proposal_id"],
    )
    application.authorize_action(action["action_id"], "reviewed")
    collection = tmp_path / "runs" / "study" / "treatment-run" / "collection.json"
    payload = json.loads(collection.read_text())
    payload["train_loss"] = 0.7
    collection.write_text(json.dumps(payload))

    with pytest.raises(ApplicationError, match="evidence changed after authorization"):
        application.execute_action(
            action["action_id"], f"EXECUTE {action['action_id']}",
        )
    runtime.close()


@pytest.mark.parametrize(
    ("value", "operation", "expected", "result"),
    [
        (2, "gt", 1, True), (2, "gte", 2, True),
        (1, "lt", 2, True), (2, "lte", 2, True),
        (2, "eq", 2, True), (2, "unknown", 2, None),
        (None, "gt", 2, None), ("bad", "gt", 2, False),
        (None, "finite", None, None),
        (1.0, "finite", None, True), (math.inf, "finite", None, False),
        (math.inf, "nonfinite", None, True), ("bad", "nonfinite", None, False),
    ],
)
def test_terminal_check_operations(value, operation, expected, result):
    assert _terminal_check(value, operation, expected) is result


def test_contract_helpers_support_legacy_and_role_specific_shapes():
    assert _required_metrics({"required_metrics": ["loss"]}, "a0") == ["loss"]
    assert _required_metrics({}, "a0") == []
    assert _required_metrics({
        "required_metrics": {"common": ["loss"], "by_role": {"a1": ["gap", "loss"]}},
    }, "a1") == ["loss", "gap"]
    assert _required_artifacts({
        "required_artifacts": {"metrics": {"min_records": 1}},
    }, "a0") == {"metrics": {"min_records": 1}}
    assert _required_artifacts({}, "a0") == {}


def test_campaign_record_paths_reject_unsafe_identities(tmp_path):
    project = load_research_project(_science_project(tmp_path))
    revision = project.campaigns[0].current_revision.revision_id
    with pytest.raises(ValueError, match="campaign is not a safe"):
        campaign_record_path(project, "../escape", revision, "completion")
    with pytest.raises(ValueError, match="revision_id is not a safe"):
        campaign_record_path(project, "study", "../escape", "completion")
    with pytest.raises(ValueError, match="unsupported campaign record"):
        campaign_record_path(project, "study", revision, "delete")


def test_campaign_lifecycle_ready_blocked_and_waiting_evidence_states(tmp_path):
    project = load_research_project(_science_project(tmp_path))
    treatment = tmp_path / "runs" / "study" / "treatment-run"
    for child in treatment.iterdir():
        child.unlink()
    treatment.rmdir()
    index = RunIndex(tmp_path / "ready.sqlite")
    index_project(index, project)
    ready = campaign_snapshot(index, project, "study")
    assert ready["lifecycle_state"] == "READY"
    assert next(item for item in ready["plan"] if item["run_id"] == "treatment-run")[
        "action"
    ] == "MATERIALIZE"
    index.close()

    project = load_research_project(_science_project(tmp_path / "blocked"))
    status_path = tmp_path / "blocked" / "runs" / "study" / "treatment-run" / "status.json"
    status_path.write_text('{"state": "FAILED"}')
    index = RunIndex(tmp_path / "blocked.sqlite")
    index_project(index, project)
    blocked = campaign_snapshot(index, project, "study")
    assert blocked["lifecycle_state"] == "BLOCKED"
    assert next(item for item in blocked["plan"] if item["run_id"] == "treatment-run")[
        "action"
    ] == "REVIEW_FAILURE"
    index.close()

    project = load_research_project(_science_project(tmp_path / "waiting"))
    collection = tmp_path / "waiting" / "runs" / "study" / "treatment-run" / "collection.json"
    payload = json.loads(collection.read_text())
    payload.pop("train_loss")
    collection.write_text(json.dumps(payload))
    index = RunIndex(tmp_path / "waiting.sqlite")
    index_project(index, project)
    waiting = campaign_snapshot(index, project, "study")
    assert waiting["lifecycle_state"] == "WAITING_EVIDENCE"
    metrics_gate = next(gate for gate in waiting["completion"]["gates"]
                        if gate["name"] == "required_metrics")
    assert metrics_gate["status"] == "PENDING"
    index.close()


def test_campaign_comparison_mismatch_and_terminal_failure_are_explicit(tmp_path):
    project = load_research_project(_science_project(tmp_path))
    revision = project.campaigns[0].current_revision
    revision.research_contract["comparison"] = {"match_fields": ["source_id", "missing"]}
    for run_id, source in (("control-run", "source-a"), ("treatment-run", "source-b")):
        manifest = tmp_path / "runs" / "study" / run_id / "manifest.yaml"
        payload = yaml.safe_load(manifest.read_text())
        payload["source_id"] = source
        manifest.write_text(yaml.safe_dump(payload))
    collection = tmp_path / "runs" / "study" / "treatment-run" / "collection.json"
    payload = json.loads(collection.read_text())
    payload["train_loss"] = 3.0
    payload["artifacts"] = {"train_metrics": {"records": 0}}
    collection.write_text(json.dumps(payload))
    index = RunIndex(tmp_path / "index.sqlite")
    index_project(index, project)

    status = campaign_snapshot(index, project, "study")
    gates = {gate["name"]: gate for gate in status["completion"]["gates"]}
    assert gates["terminal_checks"]["status"] == "FAIL"
    assert gates["required_artifacts"]["status"] == "PENDING"
    assert gates["comparability"]["status"] == "FAIL"
    assert gates["comparability"]["evidence"]["missing"] == {
        "missing": ["control-run", "treatment-run"],
    }
