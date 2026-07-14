"""runscan: both file-name generations, five separated evidence layers, freshness."""

import json
from pathlib import Path

import pytest

from ml_exp_server.ingest import runscan
from ml_exp_server.ingest.runscan import (
    discover_run_dirs, evidence_sources, evaluation_snapshot,
    evaluation_variants, is_run_dir, parse_iso_ts, scan_run_dir,
)
from tests.conftest import A1_SCHEDULER_TS, A1_WORKER_TS, FIXTURES

NOW = A1_SCHEDULER_TS + 10 * 60  # ten minutes after the last scheduler poll


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
    assert row.eval_metrics["plan_ppl_gap"] == pytest.approx(
        row.eval_metrics["shuffled_plan_ppl"] - row.eval_metrics["oracle_plan_ppl"]
    )
    assert row.evaluation_snapshot["family_state"] == "SINGLE_ELIGIBLE_FAMILY"
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


def test_declared_canonical_eval_variant_controls_flat_view(tmp_path):
    run_dir = tmp_path / "canonical-run"
    run_dir.mkdir()
    (run_dir / "manifest.yaml").write_text(
        "project: elf\nrun_id: canonical-run\n"
        "research_contract:\n  canonical_eval_variant_id: variant-b\n",
        encoding="utf-8",
    )
    for name, value in (("variant-a", 10.0), ("variant-b", 20.0)):
        variant = run_dir / name
        variant.mkdir()
        (variant / "metrics.jsonl").write_text(
            f'{{"step": 1, "g_ppl": {value}}}\n', encoding="utf-8",
        )
    row = scan_run_dir(run_dir, "elf", now=NOW)
    assert row.canonical_eval_variant_id == "variant-b"
    assert row.eval_metrics == {}
    assert row.evaluation_snapshot["family_state"] == "CANONICAL_BINDING_CONFLICT"
    assert len(row.eval_variants) == 2


def test_eval_history_preserves_matched_nonlatest_steps(tmp_path):
    histories = {}
    for run_id, steps in (("aux0", (2000, 4000, 6000, 8000, 10000)),
                          ("aux1", (2000, 4000, 6000, 8000))):
        root = tmp_path / run_id / "train_sampling_eval" / "generation"
        root.mkdir(parents=True)
        (root / "metrics.jsonl").write_text("".join(
            json.dumps({
                "epoch": 0, "step": step, "mode": "generation_refine_decode",
                "g_ppl": step / 1000, "prompt": "must not be exposed",
                "samples": ["large model output"], "api_key": "secret",
            }) + "\n"
            for step in steps
        ), encoding="utf-8")
        variants, _ = evaluation_variants(tmp_path / run_id)
        histories[run_id] = variants[0]

    assert histories["aux0"]["records"] == 5
    assert histories["aux1"]["records"] == 4
    assert histories["aux0"]["latest"]["step"] == 10000
    assert histories["aux1"]["latest"]["step"] == 8000
    assert [record["step"] for record in histories["aux0"]["history"]] == [
        2000, 4000, 6000, 8000, 10000,
    ]
    assert histories["aux1"]["history"][-1]["step"] == 8000
    for item in histories.values():
        assert item["history_limit"] == 32
        assert item["history_truncated"] is False
        assert item["history_omitted_records"] == 0
        assert not ({"prompt", "samples", "api_key"} & item["latest"].keys())
        assert all(not ({"prompt", "samples", "api_key"} & record.keys())
                   for record in item["history"])


def test_evaluation_snapshot_never_mixes_interleaved_variant_steps(tmp_path):
    run_dir = tmp_path / "interleaved"
    run_dir.mkdir()
    (run_dir / "manifest.yaml").write_text(
        "project: elf\nrun_id: interleaved\n", encoding="utf-8",
    )
    modes = {
        "clean": ("clean_token_reconstruction", "token_recon_ppl", 40.0),
        "generation": ("generation_refine_decode", "g_ppl", 1.1),
        "oracle": ("oracle_plan_generation", "oracle_plan_ppl", 2.0),
        "shuffled": ("shuffled_plan_generation", "shuffled_plan_ppl", 3.0),
    }
    for variant, (mode, metric, value) in modes.items():
        root = run_dir / "train_sampling_eval" / variant
        root.mkdir(parents=True)
        records = [{"epoch": 0, "step": 32000, "mode": mode, metric: value}]
        # Simulate independent JSONL writers: shuffled is still at 32000 while
        # the other three variants have published checkpoint 34000.
        if variant != "shuffled":
            records.append({
                "epoch": 0, "step": 34000, "mode": mode, metric: value + 0.5,
            })
        (root / "metrics.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    row = scan_run_dir(run_dir, "elf", now=NOW)
    snapshot = row.evaluation_snapshot

    assert row.evidence.evaluation.state == "step 34000 · PARTIAL"
    assert snapshot["current"]["state"] == "PARTIAL"
    assert snapshot["current"]["step"] == 34000
    assert snapshot["current"]["missing_metrics"] == ["shuffled_plan_ppl"]
    assert "shuffled_plan_ppl" not in snapshot["current"]["metrics"]
    assert "plan_ppl_gap" not in snapshot["current"]["metrics"]
    assert {
        source["step"] for source in snapshot["current"]["metric_sources"].values()
    } == {34000}

    complete = snapshot["latest_metric_complete"]
    assert complete["state"] == "COMPLETE"
    assert complete["step"] == 32000
    assert complete["metrics"]["plan_ppl_gap"] == 1.0
    assert complete["metric_sources"]["plan_ppl_gap"] == {
        "derived_from": ["oracle_plan_ppl", "shuffled_plan_ppl"],
        "epoch": 0,
        "step": 32000,
    }

    shuffled = run_dir / "train_sampling_eval" / "shuffled" / "metrics.jsonl"
    with shuffled.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "epoch": 0,
            "step": 34000,
            "mode": "shuffled_plan_generation",
            "shuffled_plan_ppl": 3.5,
        }) + "\n")

    restored_row = scan_run_dir(run_dir, "elf", now=NOW)
    restored = restored_row.evaluation_snapshot
    assert restored_row.evidence.evaluation.state == "step 34000 · COMPLETE"
    assert restored["current"]["state"] == "COMPLETE"
    assert restored["current"]["step"] == 34000
    assert restored["current"]["metrics"]["plan_ppl_gap"] == 1.0
    assert restored["latest_metric_complete"] == restored["current"]


def test_evaluation_snapshot_has_no_science_without_any_complete_checkpoint():
    variants = []
    for variant, mode, metric, step in (
        ("clean", "clean_token_reconstruction", "token_recon_ppl", 34000),
        ("generation", "generation_refine_decode", "g_ppl", 34000),
        ("oracle", "oracle_plan_generation", "oracle_plan_ppl", 34000),
        ("shuffled", "shuffled_plan_generation", "shuffled_plan_ppl", 32000),
    ):
        record = {"epoch": 0, "step": step, "mode": mode, metric: 1.0}
        variants.append({
            "variant": variant, "latest": record, "history": [record],
        })

    snapshot = evaluation_snapshot(variants)

    assert snapshot["current"]["state"] == "PARTIAL"
    assert snapshot["current"]["step"] == 34000
    assert "plan_ppl_gap" not in snapshot["current"]["metrics"]
    assert snapshot["latest_metric_complete"] is None


@pytest.mark.parametrize("rewrite", [
    {"oracle_plan_ppl": 9.0},
    {"ppl": 9.0},
    {"oracle_plan_ppl": float("nan")},
    {"oracle_plan_ppl": "9.0"},
])
def test_duplicate_oracle_rewrites_are_conflicts_not_corrections(tmp_path, rewrite):
    run_dir = tmp_path / "duplicate-conflict"
    oracle = run_dir / "oracle"
    oracle.mkdir(parents=True)
    first = {
        "epoch": 0, "step": 34000, "mode": "oracle_plan_generation",
        "oracle_plan_ppl": 2.0,
    }
    second = {
        "epoch": 0, "step": 34000, "mode": "oracle_plan_generation",
        **rewrite,
    }
    (oracle / "metrics.jsonl").write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8",
    )
    variants, _ = evaluation_variants(run_dir)

    history = variants[0]["history"][0]
    assert history["conflicting_metrics"] == ["oracle_plan_ppl"]
    assert "oracle_plan_ppl" not in history
    snapshot = evaluation_snapshot(variants)
    assert snapshot["current"]["state"] == "PARTIAL"
    assert "oracle_plan_ppl" in snapshot["current"]["conflicting_metrics"]
    assert "plan_ppl_gap" not in snapshot["current"]["metrics"]
    assert snapshot["latest_metric_complete"] is None


def test_duplicate_variant_sources_merge_and_expose_cross_source_conflicts(tmp_path):
    run_dir = tmp_path / "cross-source"
    root = run_dir / "oracle"
    nested = run_dir / "train_sampling_eval" / "oracle"
    root.mkdir(parents=True)
    nested.mkdir(parents=True)
    base = {
        "epoch": 0, "step": 10, "mode": "oracle_plan_generation",
        "oracle_plan_ppl": 2.0,
    }
    (root / "metrics.jsonl").write_text(json.dumps(base) + "\n")
    (nested / "metrics.jsonl").write_text(
        json.dumps({**base, "oracle_plan_ppl": 9.0}) + "\n"
    )

    variants, _ = evaluation_variants(run_dir)

    assert len(variants) == 1
    assert len(variants[0]["sources"]) == 2
    assert variants[0]["history"][0]["conflicting_metrics"] == [
        "oracle_plan_ppl",
    ]
    snapshot = evaluation_snapshot(variants)
    assert snapshot["current"]["state"] == "PARTIAL"
    assert "oracle_plan_ppl" in snapshot["current"]["conflicting_metrics"]
    assert "plan_ppl_gap" not in snapshot["current"]["metrics"]
    assert snapshot["latest_metric_complete"] is None

    (nested / "metrics.jsonl").write_text(json.dumps(base) + "\n")
    identical, _ = evaluation_variants(run_dir)
    assert "conflicting_metrics" not in identical[0]["history"][0]


def test_two_structured_sampling_families_require_canonical_declaration(tmp_path):
    run_dir = tmp_path / "families"
    dimensions = [
        {
            "sampling_method": "sde", "num_sampling_steps": 32, "cfg": 1.0,
            "self_cond_cfg_scale": 3.0, "time_schedule": "logit_normal",
            "time_warp_gamma": 1.5,
        },
        {
            "sampling_method": "sde", "num_sampling_steps": 64, "cfg": 1.0,
            "self_cond_cfg_scale": 3.0, "time_schedule": "logit_normal",
            "time_warp_gamma": 1.0,
        },
    ]
    records = [("clean", "clean_token_reconstruction", "token_recon_ppl", 20.0, None)]
    for family_index, family in enumerate(dimensions, start=1):
        records.extend([
            (f"f{family_index}-generation", "generation_refine_decode", "g_ppl",
             300.0 - family_index, family),
            (f"f{family_index}-oracle", "oracle_plan_generation", "oracle_plan_ppl",
             200.0 - family_index, family),
            (f"f{family_index}-shuffled", "shuffled_plan_generation",
             "shuffled_plan_ppl", 210.0 - family_index, family),
        ])
    for variant_id, mode, metric, value, family in records:
        root = run_dir / variant_id
        root.mkdir(parents=True)
        record = {"epoch": 1, "step": 38035, "mode": mode, metric: value}
        if family is not None:
            record["sampling_config"] = family
        (root / "metrics.jsonl").write_text(json.dumps(record) + "\n")

    variants, _ = evaluation_variants(run_dir)
    ambiguous = evaluation_snapshot(variants)
    assert len(variants) == 7
    assert len(ambiguous["families"]) == 2
    assert ambiguous["family_state"] == "CANONICAL_NOT_DECLARED"
    assert ambiguous["current"]["metrics"] == {}
    assert ambiguous["latest_metric_complete"] is None

    declared = evaluation_snapshot(variants, {
        "evaluation": {"canonical_family": dimensions[0]},
    })
    assert declared["family_state"] == "DECLARED"
    assert declared["current"]["state"] == "COMPLETE"
    assert declared["current"]["metrics"]["plan_ppl_gap"] == pytest.approx(10.0)

    inconsistent = evaluation_snapshot(variants, {
        "evaluation": {
            "canonical_family": dimensions[0],
            "canonical_variant_id": "f2-oracle",
        },
    })
    assert inconsistent["family_state"] == "CANONICAL_BINDING_CONFLICT"
    assert inconsistent["current"]["metrics"] == {}


def test_real_seven_variant_shape_without_structured_identity_fails_closed(tmp_path):
    run_dir = tmp_path / "real-seven"
    records = [
        ("clean", "clean_token_reconstruction", "token_recon_ppl", 20.0),
        ("f1-generation", "generation_refine_decode", "g_ppl", 297.6),
        ("f1-oracle", "oracle_plan_generation", "oracle_plan_ppl", 207.6),
        ("f1-shuffled", "shuffled_plan_generation", "shuffled_plan_ppl", 199.4),
        ("f2-generation", "generation_refine_decode", "g_ppl", 276.8),
        ("f2-oracle", "oracle_plan_generation", "oracle_plan_ppl", 191.6),
        ("f2-shuffled", "shuffled_plan_generation", "shuffled_plan_ppl", 198.7),
    ]
    for variant_id, mode, metric, value in records:
        root = run_dir / variant_id
        root.mkdir(parents=True)
        (root / "metrics.jsonl").write_text(json.dumps({
            "epoch": 1, "step": 38035, "mode": mode, metric: value,
        }) + "\n")

    variants, _ = evaluation_variants(run_dir)
    snapshot = evaluation_snapshot(variants)

    assert len(variants) == 7
    assert snapshot["family_state"] == "UNRESOLVED"
    assert len(snapshot["unresolved_variant_ids"]) == 6
    assert snapshot["required_family_dimension_fields"] == [
        "sampling_method", "num_sampling_steps", "cfg", "self_cond_cfg_scale",
        "time_schedule", "time_warp_gamma",
    ]
    assert snapshot["families"] == []
    assert snapshot["current"]["metrics"] == {}
    assert "plan_ppl_gap" not in snapshot["current"]["metrics"]


def test_identical_duplicate_records_are_idempotent_and_generic_ppl_is_fallback():
    variants = []
    for variant, mode, expected_metric, value in (
        ("clean", "clean_token_reconstruction", "token_recon_ppl", 40.0),
        ("generation", "generation_refine_decode", "g_ppl", 1.5),
        ("oracle", "oracle_plan_generation", "oracle_plan_ppl", 2.0),
        ("shuffled", "shuffled_plan_generation", "shuffled_plan_ppl", 3.0),
    ):
        raw = {"epoch": 0, "step": 1, "mode": mode, "ppl": value}
        history = runscan._evaluation_history([raw, dict(raw)])
        record = history["history"][0]
        assert record[expected_metric] == value
        assert "conflicting_metrics" not in record
        variants.append({"variant": variant, "latest": record, "history": [record]})

    snapshot = evaluation_snapshot(variants)
    assert snapshot["current"]["state"] == "COMPLETE"
    assert snapshot["current"]["metrics"]["plan_ppl_gap"] == 1.0


def test_unknown_mode_cannot_claim_a_named_primary_metric():
    variants = []
    for variant, mode, metric in (
        ("clean", "clean_token_reconstruction", "token_recon_ppl"),
        ("generation", "generation_refine_decode", "g_ppl"),
        ("oracle", "not_oracle_plan_generation", "oracle_plan_ppl"),
        ("shuffled", "shuffled_plan_generation", "shuffled_plan_ppl"),
    ):
        record = {"epoch": 0, "step": 1, "mode": mode, metric: 1.0}
        variants.append({"variant": variant, "latest": record, "history": [record]})

    snapshot = evaluation_snapshot(variants)
    assert snapshot["current"]["state"] == "PARTIAL"
    assert snapshot["current"]["missing_metrics"] == ["oracle_plan_ppl"]
    assert "plan_ppl_gap" not in snapshot["current"]["metrics"]


def test_eval_history_is_bounded_and_flags_nonidentical_duplicate_steps(tmp_path):
    root = tmp_path / "bounded" / "variant-a"
    root.mkdir(parents=True)
    records = [
        {"epoch": 0, "step": step, "ppl": float(step)}
        for step in range(35)
    ]
    records.append({"epoch": 0, "step": 34, "ppl": 0.34})
    (root / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8",
    )

    variants, _ = evaluation_variants(tmp_path / "bounded")
    item = variants[0]
    assert item["records"] == 36
    assert item["history_total"] == 35
    assert len(item["history"]) == item["history_limit"] == 32
    assert item["history_truncated"] is True
    assert item["history_omitted_records"] == 3
    assert item["history"][0]["step"] == 3
    assert item["history"][-1] == {
        "epoch": 0, "step": 34, "conflicting_metrics": ["ppl"],
    }


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


def test_wandb_identity_is_exposed_only_as_run_provenance(tmp_path):
    run_dir = tmp_path / "wandb-run"
    attempt_dir = run_dir / "attempts" / "attempt-001"
    attempt_dir.mkdir(parents=True)
    (run_dir / "manifest.yaml").write_text(
        "project: demo\n"
        "run_id: wandb-run\n"
        "resolved_config:\n"
        "  use_wandb: true\n"
        "  wandb_project: demo-metrics\n"
        "  wandb_entity: research-team\n"
        "  wandb_run_id: stable-run-id\n",
        encoding="utf-8",
    )

    row = scan_run_dir(run_dir, "demo", now=NOW)

    assert row.provenance["resolved_config_excerpt"] == {
        "use_wandb": True,
        "wandb_project": "demo-metrics",
        "wandb_entity": "research-team",
        "wandb_run_id": "stable-run-id",
    }
    assert row.provenance["wandb"] == {
        "requested": True,
        "enabled": True,
        "initialized": False,
        "entity": "research-team",
        "project": "demo-metrics",
        "run_id": "stable-run-id",
        "name": "wandb-run",
    }

    stdout = attempt_dir / "stdout.log"
    stdout.write_text(
        "startup\nWandb initialized: https://wandb.ai/research-team/"
        "demo-metrics/runs/stable-run-id (resume=allow, id=stable-run-id)\n",
        encoding="utf-8",
    )
    observed = scan_run_dir(run_dir, "demo", now=NOW)
    assert observed.provenance["wandb"]["initialized"] is True
    assert observed.provenance["wandb"]["url"] == (
        "https://wandb.ai/research-team/demo-metrics/runs/stable-run-id"
    )
    assert observed.provenance["wandb"]["evidence_source"] == str(stdout)


def test_wandb_url_prefers_attempt_collection_structured_evidence(tmp_path):
    run_dir = tmp_path / "wandb-run"
    attempt_dir = run_dir / "attempts" / "attempt-001"
    attempt_dir.mkdir(parents=True)
    (run_dir / "manifest.yaml").write_text(
        "project: demo\nrun_id: wandb-run\n"
        "resolved_config:\n  use_wandb: true\n",
        encoding="utf-8",
    )
    (attempt_dir / "collection.json").write_text(json.dumps({
        "wandb": {
            "initialized": True,
            "url": "https://wandb.ai/team/project/runs/wandb-run",
            "evidence_source": "/remote/stdout.log",
        },
    }))

    row = scan_run_dir(run_dir, "demo", now=NOW)

    assert row.provenance["wandb"]["initialized"] is True
    assert row.provenance["wandb"]["url"] == (
        "https://wandb.ai/team/project/runs/wandb-run"
    )
    assert row.provenance["wandb"]["evidence_source"] == "/remote/stdout.log"


def test_wandb_url_consumes_root_collection_and_rejects_secret_urls(tmp_path):
    run_dir = tmp_path / "wandb-run"
    run_dir.mkdir()
    (run_dir / "manifest.yaml").write_text(
        "project: demo\nrun_id: wandb-run\n"
        "resolved_config:\n  use_wandb: true\n",
        encoding="utf-8",
    )
    (run_dir / "collection.json").write_text(json.dumps({
        "wandb": {
            "initialized": True,
            "url": "https://wandb.ai/team/project/runs/wandb-run",
            "evidence_source": "/remote/stdout.log",
        },
    }))

    observed = scan_run_dir(run_dir, "demo", now=NOW)

    assert observed.provenance["wandb"]["initialized"] is True
    assert observed.provenance["wandb"]["url"].endswith("/runs/wandb-run")

    (run_dir / "collection.json").write_text(json.dumps({
        "wandb": {
            "initialized": True,
            "url": "https://user:secret@wandb.ai/runs/wandb-run",
        },
    }))
    rejected = scan_run_dir(run_dir, "demo", now=NOW)
    assert rejected.provenance["wandb"]["initialized"] is False
    assert "url" not in rejected.provenance["wandb"]


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
    assert row.evidence.evaluation.state == "OBSERVED"
    assert row.eval_metrics == {"val_bpb": 1.133225}
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
