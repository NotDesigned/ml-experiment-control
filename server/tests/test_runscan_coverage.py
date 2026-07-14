"""Focused edge coverage for the run-directory read model."""

import json
from pathlib import Path

import pytest
import yaml

from ml_exp_server.ingest import runscan


DIMENSIONS = {
    "sampling_method": "sde",
    "num_sampling_steps": 32,
    "cfg": 1.0,
    "self_cond_cfg_scale": 3.0,
    "time_schedule": "linear",
    "time_warp_gamma": 1.0,
}


def test_evaluation_projection_and_history_edge_cases():
    assert runscan._evaluation_family_dimensions({
        "variant_dimensions": {**DIMENSIONS, "cfg": True},
    }) is None
    assert runscan._evaluation_family_dimensions({
        "sampling_config": {**DIMENSIONS, "cfg": float("inf")},
    }) is None
    assert runscan._evaluation_record({"epoch": 1.0, "step": 2.0}) == {
        "epoch": 1, "step": 2,
    }

    history = runscan._evaluation_history([
        {"epoch": 0, "mode": "ignored"},
        {"epoch": 0, "step": 1, "mode": "oracle_plan_generation",
         "sampling_config": DIMENSIONS, "oracle_plan_ppl": 2.0},
        {"epoch": 0, "step": 1, "mode": "shuffled_plan_generation",
         "sampling_config": {**DIMENSIONS, "cfg": 2.0},
         "oracle_plan_ppl": 2.0},
        {"epoch": 0, "step": 1, "mode": "oracle_plan_generation",
         "sampling_config": DIMENSIONS, "oracle_plan_ppl": 2.0},
    ])
    assert history["history_skipped_records"] == 1
    assert history["history"][0]["conflicting_metrics"] == [
        "mode", "sampling_dimensions",
    ]
    metric_conflict = runscan._evaluation_history([
        {"step": 1, "oracle_plan_ppl": 2.0},
        {"step": 1, "oracle_plan_ppl": 3.0},
        {"step": 1, "oracle_plan_ppl": 2.0},
    ])
    assert metric_conflict["history"][0]["conflicting_metrics"] == [
        "oracle_plan_ppl",
    ]


def test_evaluation_family_and_canonical_declaration_edges():
    clean_with_dimensions = [{
        "step": 1, "mode": "clean_token_reconstruction",
        "sampling_dimensions": DIMENSIONS,
    }]
    assert runscan._evaluation_variant_family(clean_with_dimensions)["status"] == (
        "CONFLICTING"
    )
    conflicting_dimensions = [
        {"step": 1, "mode": "oracle_plan_generation",
         "sampling_dimensions": DIMENSIONS},
        {"step": 2, "mode": "oracle_plan_generation",
         "sampling_dimensions": {**DIMENSIONS, "cfg": 2.0}},
    ]
    assert runscan._evaluation_variant_family(conflicting_dimensions)["status"] == (
        "CONFLICTING"
    )

    variant = {
        "variant": "only", "evaluation_family": {
            "status": "RESOLVED", "scope": "SAMPLING_FAMILY",
            "family_id": "family-a", "dimensions": DIMENSIONS,
        },
    }
    assert runscan._canonical_eval_variant_id(
        [variant], {"evaluation": {"canonical_variant_id": "only"}},
    ) == "only"
    assert runscan._canonical_eval_variant_id([variant], None) == "only"

    assert runscan._declared_evaluation_family(
        [variant], {"canonical_family": "family-a"},
    ) == ("family-a", None)
    assert "must be a string" in runscan._declared_evaluation_family(
        [variant], {"canonical_family_id": 7},
    )[1]
    assert "incomplete or invalid" in runscan._declared_evaluation_family(
        [variant], {"canonical_family_dimensions": {}},
    )[1]
    assert "must be a string" in runscan._declared_evaluation_family(
        [variant], {"canonical_variant_id": 7},
    )[1]


def test_checkpoint_projection_defensive_and_conflict_edges():
    family = {"scope": "SAMPLING_FAMILY", "family_id": "wrong",
              "dimensions": DIMENSIONS}
    records = [
        "not-a-record",
        {"epoch": 0, "step": 1, "mode": "generation_refine_decode",
         "sampling_dimensions": DIMENSIONS, "g_ppl": 1.0,
         "mean_entropy": 0.5},
        {"epoch": 0, "step": 1, "mode": "oracle_plan_generation",
         "sampling_dimensions": DIMENSIONS, "oracle_plan_ppl": 2.0,
         "conflicting_metrics": ["oracle_plan_ppl", 7]},
    ]
    snapshot = runscan._evaluation_checkpoint_snapshot((0, 1), [{
        "variant": "v", "evaluation_family": family, "history": records,
    }])
    assert set(snapshot["conflicting_metrics"]) >= {"g_ppl", "oracle_plan_ppl"}

    legacy = {"variant": "v", "evaluation_family": {}, "history": [
        {"epoch": 0, "step": 1, "mode": "generation_refine_decode",
         "g_ppl": 1.0, "mean_entropy": 0.5},
        {"epoch": 0, "step": 1, "mode": "generation_refine_decode",
         "g_ppl": 2.0, "mean_entropy": 0.5},
    ]}
    snapshot = runscan._evaluation_checkpoint_snapshot((0, 1), [legacy])
    assert "g_ppl" in snapshot["conflicting_metrics"]

    preconflicted = {"variant": "v", "evaluation_family": {}, "history": [{
        "epoch": 0, "step": 1, "mode": "generation_refine_decode",
        "g_ppl": None, "mean_entropy": 0.5,
        "conflicting_metrics": ["mode", "generation_mean_entropy"],
    }]}
    snapshot = runscan._evaluation_checkpoint_snapshot((0, 1), [preconflicted])
    assert snapshot["metrics"] == {}

    none_value = {"variant": "v", "evaluation_family": {}, "history": [{
        "epoch": 0, "step": 1, "mode": "generation_refine_decode",
        "g_ppl": None,
    }]}
    assert runscan._evaluation_checkpoint_snapshot(
        (0, 1), [none_value],
    )["metrics"] == {}

    late_conflict = {"variant": "v", "evaluation_family": {}, "history": [
        {"epoch": 0, "step": 1, "mode": "generation_refine_decode",
         "g_ppl": 1.0},
        {"epoch": 0, "step": 1, "mode": "generation_refine_decode",
         "g_ppl": 1.0, "conflicting_metrics": ["g_ppl"]},
    ]}
    snapshot = runscan._evaluation_checkpoint_snapshot((0, 1), [late_conflict])
    assert "g_ppl" in snapshot["conflicting_metrics"]


def test_loaders_and_wandb_io_edges(tmp_path, monkeypatch):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("[", encoding="utf-8")
    assert runscan._load_yaml(bad_yaml) == {}

    log = tmp_path / "stdout.log"
    log.write_text("wandb initialized: https://wandb.ai/e/p/r", encoding="utf-8")
    real_open = Path.open

    def denied_open(path, *args, **kwargs):
        if path == log:
            raise OSError("denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied_open)
    assert runscan._wandb_url_in_file(log) is None


def test_wandb_structured_and_process_tail_evidence(tmp_path):
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    (attempt / "wandb.json").write_text(json.dumps({
        "run_url": "https://wandb.ai/entity/project/runs/id",
        "initialized": False, "entity": "observed", "project": None,
        "run_id": "id", "name": None,
    }), encoding="utf-8")
    observed = runscan._wandb_provenance(
        tmp_path, attempt, {"run_id": "run", "resolved_config": {}}, {},
    )
    assert observed["initialized"] is False
    assert observed["entity"] == "observed"

    (attempt / "wandb.json").unlink()
    tailed = runscan._wandb_provenance(
        tmp_path, None,
        {"run_id": "run", "resolved_config": {"use_wandb": True}},
        {"process_evidence": {
            "stdout_tail": ["no URL here"],
            "stderr_tail": ["Wandb initialized: https://wandb.ai/e/p/runs/id"],
        }},
    )
    assert tailed["initialized"] is True
    assert tailed["evidence_source"].endswith("stderr_tail")


def test_jsonl_attempt_and_evidence_source_edges(tmp_path, monkeypatch):
    data = tmp_path / "data.jsonl"
    data.write_text("\n{}\n[]\nnot-json\n", encoding="utf-8")
    assert runscan.read_jsonl(data) == [{}]

    run_dir = tmp_path / "run"
    attempt = run_dir / "attempts" / "a1"
    attempt.mkdir(parents=True)
    real_resolve = Path.resolve

    def broken_resolve(path, *args, **kwargs):
        if path == attempt:
            raise OSError("gone")
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", broken_resolve)
    assert runscan._safe_attempt_id(run_dir, "a1") is None


def test_evaluation_variants_skips_mismatched_duplicate_and_empty_sources(
    tmp_path, monkeypatch,
):
    root = tmp_path / "root"
    variant = root / "v"
    variant.mkdir(parents=True)
    metrics = variant / "metrics.jsonl"
    metrics.write_text("", encoding="utf-8")
    sources = [
        runscan.EvidenceSource(root, "other", "mismatch"),
        runscan.EvidenceSource(root, "chosen", "one"),
        runscan.EvidenceSource(root, "chosen", "duplicate"),
    ]
    monkeypatch.setattr(runscan, "preferred_attempt_id", lambda _run: "chosen")
    monkeypatch.setattr(runscan, "evidence_sources", lambda *args, **kwargs: sources)
    assert runscan.evaluation_variants(tmp_path)[0] == []


def test_manifest_mtime_source_attempt_and_discovery_edges(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert runscan._science_manifest(run_dir) == {}
    assert runscan._autoresearch_source_parameters(run_dir) == {}
    assert runscan.discover_run_dirs(tmp_path / "missing") == []

    source = run_dir / "source" / "train.py"
    source.parent.mkdir()
    source.write_text("OTHER = 1\nDEVICE_BATCH_SIZE = 8\n", encoding="utf-8")
    assert runscan._autoresearch_source_parameters(run_dir) == {
        "device_batch_size": 8,
    }

    real_read_text = Path.read_text

    def denied_read(path, *args, **kwargs):
        if path == source:
            raise OSError("denied")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied_read)
    assert runscan._autoresearch_source_parameters(run_dir) == {}

    real_stat = Path.stat

    def denied_stat(path, *args, **kwargs):
        if path == run_dir:
            raise OSError("denied")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied_stat)
    assert runscan._mtime(run_dir) is None

    plain = run_dir / "attempts" / "README"
    plain.parent.mkdir(exist_ok=True)
    plain.write_text("not an attempt", encoding="utf-8")
    assert runscan._scan_attempts(run_dir) == []


def test_decision_history_event_and_deduplication_edges(tmp_path):
    run_dir = tmp_path / "run"
    attempt = run_dir / "attempts" / "a1"
    attempt.mkdir(parents=True)
    events = [
        {"event": "other", "payload": {}},
        {"event": "decision", "payload": "bad"},
        {"event": "decision", "payload": {"decision": "bad"}},
        {"event": "decision", "attempt_id": "a1", "timestamp": "bad",
         "payload": {"decision": {"action": "OBSERVE"}}},
        {"event": "decision", "attempt_id": "a1", "timestamp": "bad",
         "payload": {"decision": {"action": "OBSERVE"}}},
    ]
    (run_dir / "events.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in events), encoding="utf-8",
    )
    (run_dir / "decision.json").write_text(
        json.dumps({"action": "OBSERVE"}), encoding="utf-8",
    )
    history = runscan._decision_history(run_dir)
    assert len(history) == 1
    assert history[0]["attempt_id"] == "a1"


def test_eval_layer_legacy_source_and_missing_timestamp(tmp_path):
    source = tmp_path / "metrics.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    layer = runscan._eval_layer([{
        "variant": "v", "latest": {}, "source": str(source),
        "evaluation_family": {}, "history": [],
    }], None)
    assert layer.state == "present"
    assert layer.source == str(source)


def test_scan_run_dir_remaining_evidence_and_projection_edges(tmp_path, monkeypatch):
    run_dir = tmp_path / "campaign" / "run"
    attempt = run_dir / "attempts" / "a1"
    attempt.mkdir(parents=True)
    (run_dir / "manifest.yaml").write_text(
        "project: wrong\nrun_id: demo-a1-run\nsource:\n"
        "  git_commit: abc\n  source_sha256: sha256:source\n",
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text(json.dumps({
        "attempt_id": "a1", "state": "RUNNING",
    }), encoding="utf-8")
    (run_dir / "collection.json").write_text(json.dumps({
        "attempt_id": "a1", "worker_state": "ALIVE",
        "process_state": "RUNNING", "model_state": "TRAINING",
        "evidence_conflicts": [{
            "kind": "evaluation_value_disagreement", "metric": "g_ppl",
            "left": {"variant": "a"}, "right": {"variant": "b"},
        }],
    }), encoding="utf-8")
    (attempt / "summary.json").write_text(json.dumps({
        "metrics": {"val_bpb": 1.2, "tokens": 5},
        "integrity": "unknown",
    }), encoding="utf-8")
    source = run_dir / "source" / "train.py"
    source.parent.mkdir()
    source.write_text("DEPTH = 12\n", encoding="utf-8")

    row = runscan.scan_run_dir(run_dir, "demo", campaign="argument")
    assert row.campaign == "argument"
    assert row.evidence.worker.state == "ALIVE"
    assert row.evidence.process.state == "RUNNING"
    assert row.evidence.model.state == "TRAINING"
    assert row.evidence.evaluation.detail == {"val_bpb": 1.2}
    assert row.provenance["resolved_config_excerpt"] == {"depth": 12}
    assert row.campaign_binding.relationship.value == "PROJECT_MISMATCH"
    assert any("differs from configured project" in item for item in row.warnings)

    source.unlink()
    row = runscan.scan_run_dir(run_dir, "demo", campaign="argument")
    assert "resolved_config_excerpt" not in row.provenance

    monkeypatch.setattr(
        runscan, "classify_evidence_conflicts", lambda *args, **kwargs: ([], [{}]),
    )
    row = runscan.scan_run_dir(run_dir, "demo", campaign="argument")
    assert any("cross-variant or cross-family" in item for item in row.warnings)

    monkeypatch.setattr(runscan, "evaluation_snapshot", lambda *args, **kwargs: {
        "required_metrics": [], "family_state": "NOT_OBSERVED",
        "current": {}, "latest_metric_complete": {"metrics": "bad"},
    })
    row = runscan.scan_run_dir(run_dir, "demo", campaign="argument")
    assert row.eval_metrics == {}


def test_scan_without_manifest_warns(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    row = runscan.scan_run_dir(run_dir, "demo")
    assert any("no readable manifest" in item for item in row.warnings)
