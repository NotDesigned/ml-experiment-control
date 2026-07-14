"""Campaign snapshots expose authored membership and observed data only."""

from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path

import pytest

from ml_exp_server.application import ExperimentServerApplication
from ml_exp_server.campaign_lifecycle import (
    _load_record,
    campaign_record_path,
    campaign_snapshot,
)
from ml_exp_server.ingest.indexer import RunIndex, index_project
from ml_exp_server.project_config import load_research_project
from ml_exp_server.runtime import ExperimentServerRuntime
from ml_exp_server.schemas import (
    ActionRuntimeConfig,
    CampaignRef,
    CampaignRelationship,
    OperationScopeType,
    ProjectRef,
    ResearchProject,
    ServerConfig,
)


def _project_file(tmp_path: Path, *, missing_ref: bool = False) -> Path:
    experiments = tmp_path / "experiments"
    experiments.mkdir(parents=True)
    campaign = experiments / "study.yml"
    run_refs = "run_refs:\n  - {run_id: missing, research_role: baseline}\n" if missing_ref else ""
    campaign_text = textwrap.dedent("""\
        schema_version: 1
        project: demo
        campaign: study
        research_contract:
          required_metrics: {common: [train_loss]}
          terminal_checks: [{metric: train_loss, op: lt, value: 0.1}]
        runs:
          - {run_id: control, research_role: control}
          - {run_id: treatment, research_role: treatment}
    """) + run_refs
    campaign.write_text(campaign_text)
    revision_id = f"campaign.{hashlib.sha256(campaign_text.encode()).hexdigest()}"
    project = experiments / "research_project.yaml"
    project.write_text(textwrap.dedent("""\
        schema_version: 1
        project: demo
        title: Demo
        run_roots: [runs]
        campaigns:
          - {name: study, file: experiments/study.yml}
    """))
    for run_id, loss in (("control", 1.0), ("treatment", 0.8)):
        run_dir = tmp_path / "runs" / "study" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.yaml").write_text(textwrap.dedent(f"""\
            schema_version: 2
            project: demo
            campaign: study
            campaign_id: {revision_id}
            run_id: {run_id}
            source_id: source-v1
            image_id: sha256:image-v1
        """))
        (run_dir / "status.json").write_text(json.dumps({"state": "SUCCEEDED"}))
        (run_dir / "collection.json").write_text(json.dumps({
            "train_loss": loss,
            "artifacts": {"train_metrics": {"records": 1}},
        }))
    return project


def _runtime(tmp_path: Path) -> ExperimentServerRuntime:
    project_file = _project_file(tmp_path)
    runtime = ExperimentServerRuntime.create(ServerConfig(
        index_db=str(tmp_path / "index.sqlite"),
        action_root=str(tmp_path / "actions"),
        action_runtime=ActionRuntimeConfig(allow_project_writes=True),
        projects=[ProjectRef(project_file=str(project_file))],
    ))
    index_project(runtime.index, runtime.project("demo"))
    return runtime


def _authorize_execute(application: ExperimentServerApplication, action: dict) -> dict:
    application.authorize_action(action["action_id"], "reviewed")
    return application.execute_action(action["action_id"], f"EXECUTE {action['action_id']}")


def test_campaign_snapshot_returns_raw_membership_and_observed_data(tmp_path):
    project = load_research_project(_project_file(tmp_path))
    index = RunIndex(tmp_path / "index.sqlite")
    index_project(index, project)

    snapshot = campaign_snapshot(index, project, "study")

    assert snapshot["lifecycle_state"] == "ACTIVE"
    assert snapshot["validation"]["status"] == "PASS"
    assert "completion" not in snapshot
    assert [item["run_id"] for item in snapshot["runs"]] == ["control", "treatment"]
    assert snapshot["runs"][0]["latest_metrics"]["train_loss"] == 1.0
    assert snapshot["runs"][1]["artifacts"]["train_metrics"]["records"] == 1
    assert "plan" not in snapshot
    index.close()


def test_research_contract_is_opaque_metadata_not_a_gate(tmp_path):
    project = load_research_project(_project_file(tmp_path))
    project.campaigns[0].current_revision.research_contract = None
    index = RunIndex(tmp_path / "index.sqlite")
    index_project(index, project)

    snapshot = campaign_snapshot(index, project, "study")

    assert snapshot["validation"]["status"] == "PASS"
    assert all(gate["name"] != "research_contract" for gate in snapshot["validation"]["gates"])
    index.close()


def test_missing_reused_run_is_reported_as_data_integrity_issue(tmp_path):
    project = load_research_project(_project_file(tmp_path, missing_ref=True))
    index = RunIndex(tmp_path / "index.sqlite")
    index_project(index, project)

    snapshot = campaign_snapshot(index, project, "study")

    assert snapshot["lifecycle_state"] == "INVALID"
    missing = next(item for item in snapshot["runs"] if item["run_id"] == "missing")
    assert missing["indexed"] is False
    assert "completion" not in snapshot
    index.close()


def test_campaign_archive_is_the_only_mutable_campaign_lifecycle_record(tmp_path):
    runtime = _runtime(tmp_path)
    application = ExperimentServerApplication(runtime)
    prepared = application.prepare_campaign_archive("demo", "study", reason="retired")

    result = _authorize_execute(application, prepared["action"])
    archived = application.campaign_status("demo", "study")

    assert result["execution"]["status"] == "VERIFIED"
    assert archived["lifecycle_state"] == "ARCHIVED"
    assert archived["records"]["archive"]["reason"] == "retired"
    assert "completion" not in archived["records"]
    runtime.close()


def test_campaign_archive_path_rejects_other_record_kinds(tmp_path):
    project = load_research_project(_project_file(tmp_path))
    revision = project.campaigns[0].current_revision.revision_id
    with pytest.raises(ValueError, match="unsupported campaign record"):
        campaign_record_path(project, "study", revision, "completion")
    with pytest.raises(ValueError, match="campaign is not a safe"):
        campaign_record_path(project, "../escape", revision, "archive")

    with pytest.raises(ValueError, match="revision_id is not a safe"):
        campaign_record_path(project, "study", "bad/revision", "archive")


def test_campaign_record_loader_ignores_invalid_or_non_mapping_yaml(tmp_path):
    invalid = tmp_path / "invalid.yml"
    invalid.write_text("[")
    assert _load_record(invalid) is None
    invalid.write_text("- item\n")
    assert _load_record(invalid) is None


def test_campaign_snapshot_rejects_unknown_and_reports_unresolved_revision(tmp_path):
    project = ResearchProject(
        project="demo", title="Demo", run_roots=[],
        campaigns=[CampaignRef(name="unresolved", current_revision=None)],
    )
    index = RunIndex(tmp_path / "index.sqlite")
    with pytest.raises(KeyError, match="unknown campaign"):
        campaign_snapshot(index, project, "missing")
    snapshot = campaign_snapshot(index, project, "unresolved")
    assert snapshot["lifecycle_state"] == "INVALID"
    index.close()


def test_campaign_snapshot_blocks_origin_and_identity_relationship(tmp_path):
    project = load_research_project(_project_file(tmp_path))
    index = RunIndex(tmp_path / "index.sqlite")
    index_project(index, project)
    row = index.get_run("demo", "control")
    assert row is not None
    row.campaign = "other"
    row.campaign_binding.relationship = CampaignRelationship.PROJECT_MISMATCH
    index.upsert_run(row)

    snapshot = campaign_snapshot(index, project, "study")
    gates = {item["name"]: item for item in snapshot["validation"]["gates"]}
    assert gates["origin:control"]["status"] == "FAIL"
    assert gates["binding:control"]["status"] == "FAIL"
    index.close()
