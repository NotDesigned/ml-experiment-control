"""SQLite index round-trip, change detection, and project config loading."""

import json
import hashlib
import shutil
import textwrap

from ml_exp_server.ingest.indexer import RunIndex, index_project
from ml_exp_server.ingest.runscan import (
    evaluation_variants, scan_run_dir, train_metric_records,
)
from ml_exp_server.project_config import (
    ConfigError, _declared_run_specs, _static_template_value,
    load_server_config, load_research_project,
)
from tests.conftest import A1_SCHEDULER_TS, FIXTURES

import pytest

NOW = A1_SCHEDULER_TS + 600


def test_campaign_run_extraction_handles_concrete_nested_and_template_values():
    assert _declared_run_specs(None) == []
    direct = {"run_id": "run-direct", "research_role": "a0"}
    assert _declared_run_specs(direct) == [direct]

    nested = {"matrix": {
        "placeholder": {"run_id": "run-{seed}"},
        "groups": [
            {"run_id": "run-a", "seed": 1},
            {"child": {"run_id": "run-b", "seed": 2}},
        ],
    }}
    assert [item["run_id"] for item in _declared_run_specs(nested)] == ["run-a", "run-b"]

    assert _static_template_value(None, "role") is None
    assert _static_template_value({"role": "a{seed}"}, "role") is None
    assert _static_template_value({"role": "a1"}, "role") == "a1"


def _make_project_files(tmp_path, run_roots="runs"):
    exp = tmp_path / "experiments"
    hyp = exp / "research_questions"
    hyp.mkdir(parents=True)
    (exp / "research_project.yaml").write_text(textwrap.dedent(f"""\
        schema_version: 1
        project: elf
        title: ELF plan-token fusion
        run_roots: [{run_roots}]
        campaigns:
          - name: fusion-len256-gate-h100-20260711
            role_notes: {{a1: frozen Sentence-T5}}
        research_questions_dir: experiments/research_questions
    """))
    (hyp / "h1.yml").write_text(textwrap.dedent("""\
        schema_version: 1
        id: H1
        title: sentence plan viability
        status: OPEN
        links:
          campaigns: [fusion-len256-gate-h100-20260711]
    """))
    return exp / "research_project.yaml"


def test_load_research_project_and_research_questions(tmp_path):
    path = _make_project_files(tmp_path)
    project = load_research_project(path)
    assert project.project == "elf"
    assert project.base_dir == tmp_path
    assert [h.id for h in project.research_questions] == ["H1"]
    assert project.research_questions[0].status == "OPEN"
    assert [campaign.name for campaign in project.campaigns] == [
        "fusion-len256-gate-h100-20260711"
    ]
    assert project.resolved_run_roots() == [(tmp_path / "runs").resolve()]


def test_load_project_resolves_campaign_revision_and_memberships(tmp_path):
    exp = tmp_path / "experiments"
    campaigns = exp / "campaigns"
    campaigns.mkdir(parents=True)
    campaign_path = campaigns / "study.yml"
    raw = textwrap.dedent("""\
        schema_version: 1
        project: elf
        campaign: study
        runs:
          - run_id: control-s1
            research_role: control
            arm: scratch
            replicate: 1
          - run_id: treatment-s1
            research_role: treatment
            included_in_analysis: false
    """)
    campaign_path.write_text(raw)
    project_path = exp / "research_project.yaml"
    project_path.write_text(textwrap.dedent("""\
        schema_version: 1
        project: elf
        title: study
        run_roots: [runs]
        campaigns:
          - name: study
            file: experiments/campaigns/study.yml
    """))

    project = load_research_project(project_path)
    revision = project.campaigns[0].current_revision
    assert revision is not None
    assert revision.revision_id == f"campaign.{hashlib.sha256(raw.encode()).hexdigest()}"
    assert revision.memberships[0].role == "control"
    assert revision.memberships[0].kind == "materialize"
    assert revision.memberships[0].arm == "scratch"
    assert revision.memberships[0].replicate == 1
    assert revision.memberships[1].included_in_analysis is False


def test_campaign_revision_extracts_concrete_legacy_matrix_memberships(tmp_path):
    exp = tmp_path / "experiments"
    exp.mkdir()
    campaign_path = exp / "matrix.yml"
    campaign_path.write_text(textwrap.dedent("""\
        schema_version: 1
        project: elf
        campaign: matrix-study
        runs:
          - matrix:
              variant:
                - {run_id: run-a0, research_role: a0}
                - {run_id: run-a1, research_role: a1}
            template:
              run_id: "{variant.run_id}"
              profile: gpu
    """))
    project_path = exp / "research_project.yaml"
    project_path.write_text(textwrap.dedent("""\
        schema_version: 1
        project: elf
        title: matrix
        run_roots: [runs]
        campaigns:
          - name: matrix-study
            file: experiments/matrix.yml
    """))

    project = load_research_project(project_path)
    memberships = project.campaigns[0].current_revision.memberships
    assert [(item.run_id, item.role) for item in memberships] == [
        ("run-a0", "a0"), ("run-a1", "a1"),
    ]


@pytest.mark.parametrize(
    ("project_value", "campaign_value", "message"),
    [
        ("other", "study", "campaign project mismatch"),
        ("elf", "other", "campaign catalog name mismatch"),
    ],
)
def test_campaign_catalog_identity_mismatch_is_rejected(
    tmp_path, project_value, campaign_value, message,
):
    exp = tmp_path / "experiments"
    exp.mkdir()
    campaign_path = exp / "campaign.yml"
    campaign_path.write_text(
        f"schema_version: 1\nproject: {project_value}\n"
        f"campaign: {campaign_value}\nruns: []\n"
    )
    project_path = exp / "research_project.yaml"
    project_path.write_text(textwrap.dedent("""\
        schema_version: 1
        project: elf
        title: study
        run_roots: [runs]
        campaigns:
          - name: study
            file: experiments/campaign.yml
    """))
    with pytest.raises(ConfigError, match=message):
        load_research_project(project_path)


def test_duplicate_research_question_id_rejected(tmp_path):
    path = _make_project_files(tmp_path)
    (tmp_path / "experiments" / "research_questions" / "h2.yml").write_text(
        "schema_version: 1\nid: H1\ntitle: dup\n")
    with pytest.raises(ConfigError, match="duplicate research question id"):
        load_research_project(path)


def test_legacy_hypothesis_fields_are_rejected(tmp_path):
    path = _make_project_files(tmp_path)
    project_text = path.read_text().replace(
        "research_questions_dir: experiments/research_questions",
        "hypotheses_dir: experiments/hypotheses",
    )
    path.write_text(project_text)
    with pytest.raises(ConfigError, match="hypotheses_dir"):
        load_research_project(path)

    path.write_text(project_text.replace(
        "hypotheses_dir: experiments/hypotheses",
        "research_questions_dir: experiments/research_questions",
    ))
    question = tmp_path / "experiments" / "research_questions" / "h1.yml"
    question.write_text(
        "schema_version: 1\nid: H1\ntitle: old\nverdict: PENDING\n"
    )
    with pytest.raises(ConfigError, match="verdict"):
        load_research_project(path)


def test_server_config(tmp_path):
    cfg = tmp_path / "console.yaml"
    cfg.write_text(textwrap.dedent(f"""\
        schema_version: 1
        index_db: {tmp_path}/index.sqlite
        action_root: {tmp_path}/actions
        poll_interval_seconds: 60
        projects:
          - project_file: {tmp_path}/experiments/research_project.yaml
    """))
    config = load_server_config(cfg)
    assert config.poll_interval_seconds == 60
    assert config.index_db_path().name == "index.sqlite"
    assert config.action_root_path() == tmp_path / "actions"


def test_server_config_defaults_to_twenty_second_polling(tmp_path):
    cfg = tmp_path / "console.yaml"
    cfg.write_text("schema_version: 1\nprojects: []\n")

    config = load_server_config(cfg)

    assert config.poll_interval_seconds == 20


def test_attempt_evidence_precedes_stale_run_mirror(tmp_path):
    run_dir = tmp_path / "runs" / "campaign" / "run-a"
    attempt_dir = run_dir / "attempts" / "attempt-002"
    (run_dir / "collected_run").mkdir(parents=True)
    (attempt_dir / "collected_run").mkdir(parents=True)
    (run_dir / "manifest.yaml").write_text(
        "schema_version: 2\nproject: demo\ncampaign: campaign\nrun_id: run-a\n"
    )
    (run_dir / "status.json").write_text(json.dumps({
        "state": "SUCCEEDED", "attempt_id": "attempt-002",
    }))
    (attempt_dir / "status.json").write_text(json.dumps({"state": "SUCCEEDED"}))
    (run_dir / "collection.json").write_text(json.dumps({
        "attempt_id": "attempt-002", "model_state": "OBSERVED",
    }))
    (run_dir / "collected_run" / "train_metrics.jsonl").write_text(
        json.dumps({"step": 100, "train_loss": 9.0}) + "\n"
    )
    (attempt_dir / "collected_run" / "train_metrics.jsonl").write_text(
        "\n".join(json.dumps({"step": step, "train_loss": loss})
                  for step, loss in ((100, 3.0), (200, 2.0), (300, 1.0))) + "\n"
    )
    nested = attempt_dir / "collected_run" / "train_sampling_eval" / "oracle"
    direct = attempt_dir / "collected_run" / "oracle"
    shuffled = attempt_dir / "collected_run" / "shuffled"
    for directory in (nested, direct, shuffled):
        directory.mkdir(parents=True)
    (nested / "metrics.jsonl").write_text(json.dumps({"step": 100, "ppl": 5.0}) + "\n")
    (direct / "metrics.jsonl").write_text(json.dumps({"step": 300, "ppl": 2.0}) + "\n")
    (shuffled / "metrics.jsonl").write_text(json.dumps({"step": 300, "ppl": 4.0}) + "\n")

    records, source, attempt_id = train_metric_records(run_dir)
    assert len(records) == 3
    assert records[-1]["step"] == 300
    assert attempt_id == "attempt-002"
    assert source == attempt_dir / "collected_run" / "train_metrics.jsonl"

    variants, eval_attempt_id = evaluation_variants(run_dir)
    assert eval_attempt_id == "attempt-002"
    assert [item["variant"] for item in variants] == ["oracle", "shuffled"]
    assert variants[0]["latest"] == {"step": 300, "ppl": 2.0}

    row = scan_run_dir(run_dir, "demo", now=0)
    assert row.latest_metrics["step"] == 300
    assert row.evidence.model.attempt_id == "attempt-002"
    assert row.evidence.evaluation.attempt_id == "attempt-002"
    assert row.evidence.evaluation.detail["oracle"]["step"] == 300


def test_index_upsert_and_change_detection(tmp_path, a1_run_dir):
    index = RunIndex(tmp_path / "index.sqlite")
    events = []
    index.on_update = lambda p, r: events.append((p, r))

    row = scan_run_dir(a1_run_dir, "elf", now=NOW)
    assert index.upsert_run(row) is True
    assert events == [("elf", row.run_id)]

    # Re-scan with a later clock: content identical apart from scanned_at → no change.
    row2 = scan_run_dir(a1_run_dir, "elf", now=NOW + 5)
    assert index.upsert_run(row2) is False
    assert len(events) == 1

    fetched = index.get_run("elf", row.run_id)
    assert fetched is not None
    assert fetched.evidence.worker.stale
    assert fetched.latest_metrics["step"] == 3700

    listed = index.list_runs("elf", campaign="fusion-len256-gate-h100-20260711")
    assert [r.run_id for r in listed] == [row.run_id]


def test_index_project_scans_fixture_roots(tmp_path):
    from ml_exp_server.schemas import ResearchProject
    project = ResearchProject(
        project="elf", title="t", run_roots=[str(FIXTURES / "runs")])
    index = RunIndex(tmp_path / "index.sqlite")
    count = index_project(index, project, now=NOW)
    assert count == 2
    ids = {r.run_id for r in index.list_runs("elf")}
    assert "elf-a1-frozen-t5-l256-s42-h100-v1" in ids
    assert "elf-smoke-slurm-l40s-probe-20260712T0105" in ids


def test_index_project_prunes_deleted_runs_only_after_a_complete_root_scan(tmp_path):
    from ml_exp_server.schemas import ResearchProject

    root = tmp_path / "runs"
    run_dir = root / "study" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.yaml").write_text(
        "project: demo\ncampaign: study\nrun_id: run-a\n",
        encoding="utf-8",
    )
    project = ResearchProject(project="demo", title="Demo", run_roots=[str(root)])
    index = RunIndex(tmp_path / "index.sqlite")
    updates = []
    index.on_update = lambda project_name, run_id: updates.append((project_name, run_id))

    assert index_project(index, project, now=NOW) == 1
    index.record_poll("demo", "run-a", "observe")
    shutil.rmtree(run_dir)
    assert index_project(index, project, now=NOW + 1) == 0
    assert index.get_run("demo", "run-a") is None
    assert index.collector_statuses("demo") == []
    assert updates.count(("demo", "run-a")) == 2

    run_dir.mkdir(parents=True)
    (run_dir / "manifest.yaml").write_text(
        "project: demo\ncampaign: study\nrun_id: run-a\n",
        encoding="utf-8",
    )
    assert index_project(index, project, now=NOW + 2) == 1
    shutil.rmtree(root)
    assert index_project(index, project, now=NOW + 3) == 0
    assert index.get_run("demo", "run-a") is not None


def test_index_project_preserves_observed_wandb_url_after_tail_disappears(tmp_path):
    from ml_exp_server.schemas import ResearchProject

    root = tmp_path / "runs"
    run_dir = root / "campaign-a" / "run-a"
    attempt_dir = run_dir / "attempts" / "attempt-001"
    attempt_dir.mkdir(parents=True)
    (run_dir / "manifest.yaml").write_text(textwrap.dedent("""\
        project: demo
        campaign: campaign-a
        run_id: run-a
        resolved_config:
          use_wandb: true
          wandb_project: metrics
          wandb_entity: team
          wandb_run_id: run-a
    """))
    stdout = attempt_dir / "stdout.log"
    stdout.write_text(
        "Wandb initialized: https://wandb.ai/team/metrics/runs/run-a\n"
    )
    project = ResearchProject(project="demo", title="Demo", run_roots=[str(root)])
    index = RunIndex(tmp_path / "index.sqlite")

    assert index_project(index, project, now=NOW) == 1
    assert index.get_run("demo", "run-a").provenance["wandb"]["initialized"] is True

    stdout.unlink()
    assert index_project(index, project, now=NOW + 1) == 1
    preserved = index.get_run("demo", "run-a").provenance["wandb"]
    assert preserved["initialized"] is True
    assert preserved["url"] == "https://wandb.ai/team/metrics/runs/run-a"


def test_index_reconciles_frozen_runs_with_current_campaign_revision(tmp_path):
    from ml_exp_server.schemas import CampaignRelationship

    exp = tmp_path / "experiments"
    exp.mkdir()
    campaign_path = exp / "study.yml"
    raw = textwrap.dedent("""\
        schema_version: 1
        project: elf
        campaign: study
        runs:
          - {run_id: matched, research_role: control}
          - {run_id: drifted, research_role: treatment}
          - {run_id: role-mismatch, research_role: control}
          - {run_id: foreign, research_role: control}
    """)
    campaign_path.write_text(raw)
    revision_id = f"campaign.{hashlib.sha256(raw.encode()).hexdigest()}"
    project_path = exp / "research_project.yaml"
    project_path.write_text(textwrap.dedent("""\
        schema_version: 1
        project: elf
        title: study
        run_roots: [runs]
        campaigns:
          - name: study
            file: experiments/study.yml
    """))
    for run_id, project_name, role, revision in (
        ("matched", "elf", "control", revision_id),
        ("drifted", "elf", "treatment", "campaign.old"),
        ("role-mismatch", "elf", "treatment", revision_id),
        ("foreign", "other", "control", revision_id),
        ("undeclared", "elf", "", revision_id),
    ):
        run_dir = tmp_path / "runs" / "study" / run_id
        run_dir.mkdir(parents=True)
        role_line = f"research_role: {role}\n" if role else ""
        (run_dir / "manifest.yaml").write_text(
            f"project: {project_name}\ncampaign: study\nrun_id: {run_id}\n"
            f"campaign_id: {revision}\n{role_line}"
        )

    project = load_research_project(project_path)
    index = RunIndex(tmp_path / "index.sqlite")
    assert index_project(index, project, now=NOW) == 5
    rows = {row.run_id: row for row in index.list_runs("elf")}
    assert rows["matched"].campaign_binding.relationship == CampaignRelationship.MATCHED
    assert rows["matched"].campaign_binding.membership.role == "control"
    assert rows["drifted"].campaign_binding.relationship == \
        CampaignRelationship.CAMPAIGN_REVISION_DRIFT
    assert rows["role-mismatch"].campaign_binding.relationship == \
        CampaignRelationship.ROLE_MISMATCH
    assert rows["foreign"].campaign_binding.relationship == \
        CampaignRelationship.PROJECT_MISMATCH
    assert rows["undeclared"].campaign_binding.relationship == \
        CampaignRelationship.UNDECLARED_RUN


def test_duplicate_run_identity_is_reported_without_silent_last_write(tmp_path):
    from ml_exp_server.schemas import CampaignRelationship, ResearchProject

    for root in (tmp_path / "a", tmp_path / "b"):
        run_dir = root / "study" / "same-run"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.yaml").write_text(
            "project: elf\ncampaign: study\nrun_id: same-run\n"
        )
    project = ResearchProject(
        project="elf", title="t", run_roots=[str(tmp_path / "a"), str(tmp_path / "b")],
    )
    index = RunIndex(tmp_path / "index.sqlite")
    assert index_project(index, project, now=NOW) == 2
    row = index.get_run("elf", "same-run")
    assert row.campaign_binding.relationship == CampaignRelationship.DUPLICATE_RUN_ID
    assert row.run_dir == str(tmp_path / "a" / "study" / "same-run")
    assert any("duplicate run identity" in warning for warning in row.warnings)


def test_one_run_can_participate_in_multiple_campaign_revisions(tmp_path):
    exp = tmp_path / "experiments"
    exp.mkdir()
    origin_raw = (
        "schema_version: 1\nproject: elf\ncampaign: origin\n"
        "runs:\n  - {run_id: shared-run, research_role: treatment}\n"
    )
    (exp / "origin.yml").write_text(origin_raw)
    (exp / "reuse.yml").write_text(
        "schema_version: 1\nproject: elf\ncampaign: reuse\n"
        "run_refs:\n  - {run_id: shared-run, research_role: baseline}\n"
    )
    project_path = exp / "research_project.yaml"
    project_path.write_text(textwrap.dedent("""\
        schema_version: 1
        project: elf
        title: reuse
        run_roots: [runs]
        campaigns:
          - {name: origin, file: experiments/origin.yml}
          - {name: reuse, file: experiments/reuse.yml}
    """))
    run_dir = tmp_path / "runs" / "origin" / "shared-run"
    run_dir.mkdir(parents=True)
    origin_revision = f"campaign.{hashlib.sha256(origin_raw.encode()).hexdigest()}"
    (run_dir / "manifest.yaml").write_text(
        "project: elf\ncampaign: origin\nrun_id: shared-run\n"
        f"campaign_id: {origin_revision}\nresearch_role: treatment\n"
    )

    project = load_research_project(project_path)
    index = RunIndex(tmp_path / "index.sqlite")
    index_project(index, project, now=NOW)
    row = index.get_run("elf", "shared-run")
    assert row.campaign_binding.relationship.value == "MATCHED"
    assert [binding.campaign for binding in row.campaign_memberships] == [
        "origin", "reuse",
    ]
    assert row.campaign_memberships[0].is_origin is True
    assert row.campaign_memberships[1].membership.kind == "reuse"
    assert row.campaign_memberships[1].membership.role == "baseline"
    assert [item.run_id for item in index.list_runs("elf", campaign="reuse")] == [
        "shared-run"
    ]


def test_duplicate_materialization_across_campaigns_requires_explicit_run_ref(tmp_path):
    exp = tmp_path / "experiments"
    exp.mkdir()
    for name in ("first", "second"):
        (exp / f"{name}.yml").write_text(
            f"schema_version: 1\nproject: elf\ncampaign: {name}\n"
            "runs:\n  - {run_id: shared-run}\n"
        )
    project_path = exp / "research_project.yaml"
    project_path.write_text(textwrap.dedent("""\
        schema_version: 1
        project: elf
        title: duplicate
        run_roots: [runs]
        campaigns:
          - {name: first, file: experiments/first.yml}
          - {name: second, file: experiments/second.yml}
    """))
    with pytest.raises(ConfigError, match="use run_refs for reuse"):
        load_research_project(project_path)


def test_collector_status_roundtrip(tmp_path):
    index = RunIndex(tmp_path / "index.sqlite")
    index.record_poll("elf", "run-1", "observe", None, now=100.0)
    index.record_poll("elf", "run-1", "decide", "ssh timeout", now=200.0)
    statuses = index.collector_statuses("elf")
    assert len(statuses) == 1
    assert statuses[0].last_verb == "decide"
    assert statuses[0].last_error == "ssh timeout"
    assert statuses[0].last_poll_at == 200.0
    assert statuses[0].verb_results["observe"]["outcome"] == "succeeded"
    assert statuses[0].verb_results["decide"]["outcome"] == "failed"
