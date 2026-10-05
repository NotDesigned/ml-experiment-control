"""runscan: both file-name generations, five separated evidence layers, freshness."""

import json
from pathlib import Path

import pytest

from ml_exp_server.ingest import runscan
from ml_exp_server.ingest.runscan import (
    discover_run_dirs, evidence_sources, evaluation_snapshot,
    evaluation_variants, is_run_dir, parse_iso_ts, scan_run_dir,
    train_metric_records,
)
from tests.conftest import A1_SCHEDULER_TS, A1_WORKER_TS, FIXTURES

NOW = A1_SCHEDULER_TS + 10 * 60  # ten minutes after the last scheduler poll


def test_uploaded_metric_history_wins_over_last_point_and_stays_attempt_scoped(tmp_path):
    run = tmp_path / 'run'
    current = run / 'attempts/attempt-002'
    uploaded = current / 'uploaded_outputs'
    uploaded.mkdir(parents=True)
    (current / 'observed_train_metrics.jsonl').write_text('{"step": 3}\n')
    history = [{'step': n, 'accuracy': n / 10} for n in [1, 2, 3]]
    path = uploaded / 'metrics.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in history))
    previous = run / 'attempts/attempt-001/uploaded_outputs'
    previous.mkdir(parents=True)
    (previous / 'metrics.jsonl').write_text('{"step": 999}\n')
    records, source, attempt = train_metric_records(run, attempt_id='attempt-002', exact_attempt=True)
    assert (records, source, attempt) == (history, path, 'attempt-002')
    assert train_metric_records(run, attempt_id='attempt-003', exact_attempt=True)[0] == []
    path.unlink()
    uploaded.rmdir()
    uploaded.symlink_to(previous, target_is_directory=True)
    assert train_metric_records(run, attempt_id='attempt-002', exact_attempt=True)[0] == [{'step': 3}]


def test_parse_iso_ts():
    assert parse_iso_ts("2026-07-11T14:31:48.755999Z") == A1_SCHEDULER_TS
    assert parse_iso_ts(None) is None
    assert parse_iso_ts("not a date") is None


def test_attempt_identity_cannot_escape_attempts_directory(tmp_path):
    run = tmp_path / "campaign" / "run-a"
    attempts = run / "attempts"
    attempts.mkdir(parents=True)
    secret = tmp_path / "campaign" / "secret-attempt"
    secret.mkdir()
    (secret / "train_metrics.jsonl").write_text(
        '{"step": 999, "private_marker": "LEAKED"}\n', encoding="utf-8",
    )
    (run / "manifest.yaml").write_text("run_id: run-a\n", encoding="utf-8")
    (run / "status.json").write_text(
        '{"attempt_id": "../../secret-attempt", "state": "RUNNING"}',
        encoding="utf-8",
    )

    row = scan_run_dir(run, "demo")

    assert row.evidence.model.source is None
    assert "private_marker" not in row.latest_metrics
    assert evidence_sources(
        run, attempt_id="../../secret-attempt", exact_attempt=True,
    ) == []


def test_nested_collection_latest_metric_is_exact_attempt_evidence(tmp_path):
    run = tmp_path / "campaign" / "run-a"
    attempt = run / "attempts" / "attempt-002"
    attempt.mkdir(parents=True)
    (run / "manifest.yaml").write_text(
        "project: demo\ncampaign: campaign\nrun_id: run-a\n", encoding="utf-8",
    )
    (run / "status.json").write_text(
        json.dumps({"attempt_id": "attempt-002", "state": "RUNNING"}),
        encoding="utf-8",
    )
    (attempt / "collection.json").write_text(json.dumps({
        "attempt_id": "attempt-002",
        "model_state": "OBSERVED",
        "latest_metric": {
            "step": 500.0,
            "train_loss": 0.1151,
            "bad": float("nan"),
            "nested": {"must": "not leak"},
        },
    }), encoding="utf-8")

    records, source, attempt_id = train_metric_records(
        run, attempt_id="attempt-002", exact_attempt=True,
    )
    row = scan_run_dir(run, "demo", now=NOW)

    assert records == [{"step": 500, "train_loss": 0.1151}]
    assert source == attempt / "collection.json"
    assert attempt_id == "attempt-002"
    assert row.latest_metrics["step"] == 500
    assert row.latest_metrics["train_loss"] == pytest.approx(0.1151)
    assert "bad" not in row.latest_metrics
    assert "nested" not in row.latest_metrics
    assert row.evidence.model.attempt_id == "attempt-002"


def test_collection_identity_metadata_is_not_reported_as_model_metrics(tmp_path):
    run = tmp_path / "campaign" / "run-a"
    attempt = run / "attempts" / "attempt-001"
    attempt.mkdir(parents=True)
    (run / "manifest.yaml").write_text(
        "project: demo\ncampaign: campaign\nrun_id: run-a\n", encoding="utf-8",
    )
    (run / "status.json").write_text(json.dumps({
        "attempt_id": "attempt-001", "state": "CANCELLED",
    }), encoding="utf-8")
    (run / "collection.json").write_text(json.dumps({
        "attempt_id": "attempt-001",
        "backend_job_id": "9667",
        "collected_at": "2026-07-28T00:00:00Z",
        "process_state": "CANCELLED",
        "model_state": "NOT_OBSERVED",
    }), encoding="utf-8")

    row = scan_run_dir(run, "demo", now=NOW)

    assert "backend_job_id" not in row.latest_metrics
    assert "collected_at" not in row.latest_metrics


def test_aggregate_seeds_are_exposed_as_canonical_provenance(tmp_path):
    run = tmp_path / "campaign" / "run-a"
    run.mkdir(parents=True)
    (run / "manifest.yaml").write_text(
        "project: demo\ncampaign: campaign\nrun_id: run-a\n"
        "source_id: source-1\nimage_id: image-1\nconfig_path: config.yml\n"
        "resolved_config:\n  seeds: [601, 602, 603]\n",
        encoding="utf-8",
    )

    row = scan_run_dir(run, "demo")

    assert row.provenance["seed"] == [601, 602, 603]
    assert row.provenance["resolved_config_excerpt"]["seeds"] == [
        601, 602, 603,
    ]


def test_unbound_run_mirror_is_not_attributed_to_new_retry_attempt(tmp_path):
    run = tmp_path / "campaign" / "run-a"
    first = run / "attempts" / "attempt-001"
    second = run / "attempts" / "attempt-002"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({
        "attempt_id": "attempt-002", "state": "STARTING",
    }), encoding="utf-8")
    (run / "collection.json").write_text(json.dumps({
        "latest_metric": {"step": 300, "train_loss": 0.6506},
    }), encoding="utf-8")

    records, source, attempt_id = train_metric_records(run)

    assert records == []
    assert source is None
    assert attempt_id is None


def test_symlinked_attempt_tree_is_not_evidence(tmp_path):
    run = tmp_path / "run-a"
    external = tmp_path / "external"
    external_attempt = external / "attempt-001"
    run.mkdir()
    external_attempt.mkdir(parents=True)
    (run / "attempts").symlink_to(external, target_is_directory=True)

    assert evidence_sources(
        run, attempt_id="attempt-001", exact_attempt=True,
    ) == []


def test_a1_old_generation_layout_is_recognized(a1_run_dir):
    assert is_run_dir(a1_run_dir)  # via collected_run/manifest.yaml + control_manifest.yaml


def test_smoke_new_generation_layout_is_recognized(smoke_run_dir):
    assert is_run_dir(smoke_run_dir)  # via canonical manifest.yaml


def test_a1_evidence_layers_are_separated(a1_run_dir):
    row = scan_run_dir(a1_run_dir, "elf", now=NOW)

    assert row.run_id == "elf-a1-frozen-t5-l256-s42-h100-v1"
    assert row.campaign == "fusion-len256-gate-h100-20260711"

    # Scheduler: RUNNING as of the last poll — fresh at NOW (+10 min).
    assert row.evidence.scheduler.state == "RUNNING"
    assert abs(row.evidence.scheduler.as_of - A1_SCHEDULER_TS) < 1
    assert not row.evidence.scheduler.stale

    # Worker: remote status frozen at its initial write → hours old → stale.
    assert abs(row.evidence.worker.as_of - A1_WORKER_TS) < 1
    assert row.evidence.worker.stale
    assert "worker evidence" in row.evidence.worker.stale_reason

    # Model: last train_metrics record, step 3700, with its own timestamp.
    assert row.evidence.model.detail["step"] == 3700
    assert row.evidence.model.as_of < A1_SCHEDULER_TS  # metrics older than sched poll

    # Never collapsed: scheduler fresh while worker stale must coexist.
    assert row.scheduler_state == "RUNNING"

    # Evaluation layer sees all four variants.
    assert len(row.evidence.evaluation.detail) == 4
    assert any("oracle-plan" in k for k in row.evidence.evaluation.detail)


def test_a1_metrics_and_provenance(a1_run_dir):
    row = scan_run_dir(a1_run_dir, "elf", now=NOW)
    assert row.latest_metrics["step"] == 3700
    assert 0.8 < row.latest_metrics["train_loss"] < 0.9
    assert row.eval_metrics == {}
    assert row.evaluation_snapshot["required_metrics"] == []
    assert row.canonical_eval_variant_id is None
    assert len(row.eval_variants) == 4
    by_name = {item["variant"]: item["latest"] for item in row.eval_variants}
    oracle = next(value for name, value in by_name.items() if "oracle-plan" in name)
    shuffled = next(value for name, value in by_name.items() if "shuffled-plan" in name)
    assert oracle["oracle_plan_ppl"] < shuffled["shuffled_plan_ppl"]
    assert not any("flat eval_metrics suppressed" in warning for warning in row.warnings)
    assert row.provenance["image_id"].startswith("sha256:")
    assert row.provenance["seed"] == 42
    assert row.provenance["resolved_config_excerpt"]["max_length"] == 256
    # Role inferred from run_id since old manifests lack research_role.
    assert row.role == "a1"
    assert row.role_source == "heuristic"
    assert row.attempts and row.attempts[0].attempt_id == "attempt-001"
    assert row.decision["action"] == "OBSERVE"
    assert row.decision["retries_allowed"] == 0
    assert row.decision["retries_used"] == 0






































def test_a1_model_layer_goes_stale_much_later(a1_run_dir):
    much_later = A1_SCHEDULER_TS + 6 * 3600
    row = scan_run_dir(a1_run_dir, "elf", now=much_later)
    assert row.evidence.model.stale
    assert row.evidence.scheduler.stale  # poll itself is now ancient too


def test_smoke_run_created_state_not_flagged(smoke_run_dir):
    row = scan_run_dir(smoke_run_dir, "elf", now=NOW)
    assert row.scheduler_state == "CREATED"
    assert row.campaign == "backend-smoke-slurm-probe-20260712T0105"
    # CREATED is not an active state: nothing should be marked stale.
    assert not row.evidence.scheduler.stale
    assert not row.evidence.worker.stale
    # Preparing a Slurm script does not mean the Attempt reached the scheduler.
    assert row.attempts and not row.attempts[0].has_submission
    # research_role absent and smoke id doesn't match the ablation pattern.
    assert row.provenance["git_commit"]








def test_terminal_runs_are_never_stale(tmp_path):
    run_dir = tmp_path / "camp" / "run-x"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.yaml").write_text("run_id: run-x\nproject: p\ncampaign: camp\n")
    (run_dir / "status.json").write_text(json.dumps(
        {"state": "SUCCEEDED", "updated_at": "2026-01-01T00:00:00Z"}))
    row = scan_run_dir(run_dir, "p", now=NOW)
    assert row.is_terminal
    assert not row.evidence.scheduler.stale


def test_discover_run_dirs_finds_both_generations():
    found = discover_run_dirs(FIXTURES / "runs")
    names = {p.name for p in found}
    assert "elf-a1-frozen-t5-l256-s42-h100-v1" in names
    assert "elf-smoke-slurm-l40s-probe-20260712T0105" in names


def test_discover_run_dirs_finds_instance_scoped_run_without_mirror_duplicate(tmp_path):
    run_dir = tmp_path / "runs" / "state" / "instance-a" / "study" / "run-a"
    mirror = run_dir / "attempts" / "attempt-001" / "collected_run"
    mirror.mkdir(parents=True)
    (run_dir / "manifest.yaml").write_text(
        "project: demo\ncampaign: study\nrun_id: run-a\n", encoding="utf-8",
    )
    (mirror / "manifest.yaml").write_text(
        "project: demo\ncampaign: study\nrun_id: run-a\n", encoding="utf-8",
    )

    assert discover_run_dirs(tmp_path / "runs") == [run_dir]


def test_autoresearch_harness_json_layout_is_ingested(tmp_path):
    run_dir = tmp_path / "runs" / "local-depth6"
    attempt_dir = run_dir / "attempts" / "attempt-001"
    attempt_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "project": "autoresearch",
        "run_id": "local-depth6",
        "created_at": "2026-07-13T01:00:00Z",
        "source": {
            "git_commit": "abc123",
            "origin": "/work/autoresearch-candidate",
            "source_sha256": "source-sha",
            "train_py_sha256": "train-sha",
        },
    }))
    (run_dir / "status.json").write_text(json.dumps({
        "attempt_id": "attempt-001",
        "state": "SUCCEEDED",
        "updated_at": "2026-07-13T01:06:00Z",
        "metrics": {
            "val_bpb": 1.133225,
            "depth": 6,
            "peak_vram_mb": 1168.3,
            "total_seconds": 342.2,
        },
    }))
    (attempt_dir / "attempt.json").write_text(json.dumps({
        "attempt_id": "attempt-001",
        "state": "SUCCEEDED",
        "return_code": 0,
        "started_at": "2026-07-13T01:00:10Z",
        "finished_at": "2026-07-13T01:05:52Z",
    }))
    (attempt_dir / "submission.json").write_text(json.dumps({
        "attempt_id": "attempt-001", "gpu": "0",
    }))
    (attempt_dir / "summary.json").write_text(json.dumps({
        "attempt_id": "attempt-001",
        "state": "SUCCEEDED",
        "collected_at": "2026-07-13T01:06:00Z",
        "metrics": {"val_bpb": 1.133225, "depth": 6},
        "integrity": {
            "source_sha256_matches": True,
            "train_py_sha256_matches": True,
        },
    }))
    (attempt_dir / "metrics.jsonl").write_text(
        '{"step": 2476, "train_loss": 2.1}\n', encoding="utf-8",
    )
    source_dir = run_dir / "source"
    source_dir.mkdir()
    (source_dir / "train.py").write_text(
        "DEPTH = 6\nDEVICE_BATCH_SIZE = 8  # immutable candidate\n",
        encoding="utf-8",
    )

    assert is_run_dir(run_dir)
    assert discover_run_dirs(tmp_path / "runs") == [run_dir]

    row = scan_run_dir(run_dir, "autoresearch", now=NOW)
    assert row.run_id == "local-depth6"
    assert row.scheduler_state == "SUCCEEDED"
    assert row.attempts[0].state == "SUCCEEDED"
    assert row.attempts[0].backend == "local-cuda"
    assert row.attempts[0].has_submission
    assert row.evidence.worker.state == "LOCAL_GPU"
    assert row.evidence.process.state == "SUCCEEDED"
    assert row.evidence.model.detail["step"] == 2476
    assert row.evidence.evaluation.state is None
    assert row.eval_metrics == {}
    assert row.latest_metrics["val_bpb"] == 1.133225
    assert row.latest_metrics["depth"] == 6
    assert row.latest_metrics["peak_vram_mb"] == 1168.3
    assert row.provenance["git_commit"] == "abc123"
    assert row.provenance["source_id"] == "source-sha"
    assert row.provenance["train_py_sha256"] == "train-sha"
    assert row.provenance["resolved_config_excerpt"] == {
        "depth": 6, "device_batch_size": 8,
    }
    assert row.artifacts["summary"]["records"] == 1
    assert row.artifacts["integrity"]["nonempty_records"] == 2


def test_scan_tolerates_corrupt_files(tmp_path):
    run_dir = tmp_path / "camp" / "bad-run"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.yaml").write_text("run_id: bad-run\n")
    (run_dir / "status.json").write_text("{not json")
    (run_dir / "train_metrics.jsonl").write_text('{"step": 1}\n{broken\n{"step": 2}\n')
    row = scan_run_dir(run_dir, "p", now=NOW)
    assert row.run_id == "bad-run"
    assert row.evidence.model.detail["step"] == 2  # bad line skipped


def test_scan_preserves_authored_metadata_and_observed_data(tmp_path):
    run_dir = tmp_path / "camp" / "run-a"
    attempt_dir = run_dir / "attempts" / "attempt-001"
    attempt_dir.mkdir(parents=True)
    (run_dir / "manifest.yaml").write_text(
        "run_id: run-a\nproject: p\ncampaign: camp\nresearch_role: a0\n"
        "research_contract:\n  schema_version: 1\n  question: frozen question\n"
    )
    (run_dir / "status.json").write_text('{"state": "SUCCEEDED"}')
    (run_dir / "collection.json").write_text(json.dumps({
        "latest_completed_checkpoint": "/shared/run-a/checkpoint_100",
        "latest_completed_checkpoint_step": 100,
        "artifacts": {"train_metrics": {"records": 10}},
    }))
    decision = {
        "action": "VERIFY_RESULTS",
        "reason": "scheduler succeeded",
    }
    (attempt_dir / "decision.json").write_text(json.dumps(decision))
    (run_dir / "decision.json").write_text(json.dumps(decision))

    row = scan_run_dir(run_dir, "p", now=NOW)

    assert row.research_contract["question"] == "frozen question"
    assert row.research_contract_source == "manifest"
    assert row.checkpoint["latest_completed_checkpoint_step"] == 100
    assert row.artifacts["train_metrics"]["records"] == 10
    assert len(row.decision_history) == 1
    assert row.decision_history[0]["attempt_id"] == "attempt-001"
