from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import yaml

from ml_exp_server.application import ExperimentServerApplication
from ml_exp_server.evidence_conflicts import _family_id, classify_evidence_conflicts


PROJECT = "elf"
RUN_ID = "elf-aux1-mb64-ga2-h100-20260714-r1"
ATTEMPT_ID = "attempt-001"
OBSERVED_AT = 1784029337.1040568
DIMENSIONS = {
    steps: {
        "sampling_method": "sde", "num_sampling_steps": steps, "cfg": 1.0,
        "self_cond_cfg_scale": 3.0, "time_schedule": "logit_normal",
        "time_warp_gamma": gamma,
    }
    for steps, gamma in ((32, 1.5), (64, 1.0))
}


def _source(*, variant_id: str, dimensions: dict, metric: str, value: float) -> dict:
    return {
        "source": f"collected_run/train_sampling_eval/{variant_id}/metrics.jsonl",
        "value": value,
        "observed_at": OBSERVED_AT,
        "binding": {
            "project": PROJECT, "run_id": RUN_ID, "attempt_id": ATTEMPT_ID,
            "epoch": 1, "step": 38035, "variant_id": variant_id,
            "family_id": _family_id(dimensions), "metric": metric,
            "sampling_dimensions": dimensions,
        },
    }


def _cross_family_conflicts() -> list[dict]:
    values = {
        "g_ppl": (252.03480132729845, 238.58811625648886),
        "generation_mean_entropy": (4.605427395552397, 4.608551295474172),
        "oracle_plan_ppl": (188.26621848150643, 177.8099580326585),
        "shuffled_plan_ppl": (180.62738464568906, 179.51874195224673),
    }
    return [{
        "type": "legacy_flat_metric_conflict", "metric": metric,
        "sources": [
            _source(
                variant_id=f"steps32-{metric}", dimensions=DIMENSIONS[32],
                metric=metric, value=left,
            ),
            _source(
                variant_id=f"steps64-{metric}", dimensions=DIMENSIONS[64],
                metric=metric, value=right,
            ),
        ],
    } for metric, (left, right) in values.items()]


def test_cross_variant_family_values_are_not_exact_conflicts():
    blocking, reclassified = classify_evidence_conflicts(
        _cross_family_conflicts(),
        project=PROJECT, run_id=RUN_ID, attempt_id=ATTEMPT_ID,
    )

    assert blocking == []
    assert len(reclassified) == 4
    assert all(item["type"] == "cross_binding_values" for item in reclassified)


def test_same_exact_variant_rewrite_remains_blocking_and_keeps_sources():
    conflict = {
        "type": "metric_value_conflict",
        "sources": [
            _source(
                variant_id="steps32-generation", dimensions=DIMENSIONS[32],
                metric="g_ppl", value=252.03480132729845,
            ),
            _source(
                variant_id="steps32-generation", dimensions=DIMENSIONS[32],
                metric="g_ppl", value=999.0,
            ),
        ],
    }
    conflict["sources"][1]["source"] = (
        "attempts/attempt-001/train_sampling_eval/steps32-generation/metrics.jsonl"
    )
    blocking, reclassified = classify_evidence_conflicts(
        [conflict], project=PROJECT, run_id=RUN_ID, attempt_id=ATTEMPT_ID,
    )

    assert reclassified == []
    assert len(blocking) == 1
    assert blocking[0]["variant_id"] == "steps32-generation"
    assert blocking[0]["metric"] == "g_ppl"
    assert {item["value"] for item in blocking[0]["sources"]} == {
        252.03480132729845, 999.0,
    }
    assert {item["observed_at"] for item in blocking[0]["sources"]} == {
        OBSERVED_AT,
    }


def test_same_exact_variant_same_value_is_idempotent():
    first = _source(
        variant_id="steps32-generation", dimensions=DIMENSIONS[32],
        metric="g_ppl", value=252.0,
    )
    second = {
        **first,
        "source": "attempts/attempt-001/generation/metrics.jsonl",
        "binding": dict(first["binding"]),
    }

    blocking, reclassified = classify_evidence_conflicts(
        [{"type": "metric_value_conflict", "sources": [first, second]}],
        project=PROJECT, run_id=RUN_ID, attempt_id=ATTEMPT_ID,
    )

    assert blocking == []
    assert reclassified == []


@pytest.mark.parametrize("values", [(188.0, 188.0), (188.0, 199.0)])
def test_same_family_semantic_slot_cannot_have_multiple_variants(values):
    sources = [
        _source(
            variant_id=variant_id, dimensions=DIMENSIONS[32],
            metric="oracle_plan_ppl", value=value,
        )
        for variant_id, value in zip(("oracle-a", "oracle-b"), values)
    ]

    blocking, reclassified = classify_evidence_conflicts(
        [{"type": "legacy_flat_metric_conflict", "sources": sources}],
        project=PROJECT, run_id=RUN_ID, attempt_id=ATTEMPT_ID,
    )

    assert reclassified == []
    assert len(blocking) == 1
    assert blocking[0]["type"] == "metric_semantic_slot_conflict"
    assert blocking[0]["family_id"] == _family_id(DIMENSIONS[32])
    assert blocking[0]["variant_ids"] == ["oracle-a", "oracle-b"]


def test_incomplete_legacy_conflict_stays_blocking_without_label_guessing():
    raw = "g_ppl has conflicting values in steps32 and steps64"
    blocking, reclassified = classify_evidence_conflicts(
        [raw], project=PROJECT, run_id=RUN_ID, attempt_id=ATTEMPT_ID,
    )
    assert blocking == [raw]
    assert reclassified == []


@pytest.mark.parametrize("mutation", [
    "missing_family", "missing_source", "missing_dimensions",
    "fake_family_sha", "mismatched_family_hash", "blank_source",
])
def test_incomplete_or_forged_family_bindings_remain_blocking(mutation):
    left = _source(
        variant_id="steps32-generation", dimensions=DIMENSIONS[32],
        metric="g_ppl", value=252.0,
    )
    right = _source(
        variant_id="steps64-generation", dimensions=DIMENSIONS[64],
        metric="g_ppl", value=238.0,
    )
    binding = right["binding"]
    if mutation == "missing_family":
        binding.pop("family_id")
    elif mutation == "missing_source":
        right.pop("source")
    elif mutation == "missing_dimensions":
        binding.pop("sampling_dimensions")
    elif mutation == "fake_family_sha":
        binding["family_id"] = "sha256:steps64"
    elif mutation == "mismatched_family_hash":
        binding["family_id"] = _family_id(DIMENSIONS[32])
    else:
        right["source"] = " "

    raw = {"type": "legacy_flat_metric_conflict", "sources": [left, right]}
    blocking, reclassified = classify_evidence_conflicts(
        [raw], project=PROJECT, run_id=RUN_ID, attempt_id=ATTEMPT_ID,
    )

    assert blocking == [raw]
    assert reclassified == []


def test_legacy_source_records_with_direct_exact_bindings_can_be_reclassified():
    sources = []
    for variant_id, dimensions, value in (
        ("steps32-generation", DIMENSIONS[32], 252.03480132729845),
        ("steps64-generation", DIMENSIONS[64], 238.58811625648886),
    ):
        nested = _source(
            variant_id=variant_id, dimensions=dimensions,
            metric="g_ppl", value=value,
        )
        sources.append({
            **nested.pop("binding"), **nested,
        })
    blocking, reclassified = classify_evidence_conflicts(
        [{"metric": "g_ppl", "sources": sources}],
        project=PROJECT, run_id=RUN_ID, attempt_id=ATTEMPT_ID,
    )
    assert blocking == []
    assert len(reclassified) == 1


def test_attempt_validation_passes_cross_family_values_and_blocks_exact_rewrite(
    tmp_path,
):
    run_dir = tmp_path / RUN_ID
    attempt_dir = run_dir / "attempts" / ATTEMPT_ID
    attempt_dir.mkdir(parents=True)
    (run_dir / "manifest.yaml").write_text(yaml.safe_dump({
        "project": PROJECT, "run_id": RUN_ID,
    }), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "project": PROJECT, "run_id": RUN_ID, "attempt_id": ATTEMPT_ID,
        "state": "SUCCEEDED",
    }), encoding="utf-8")
    (attempt_dir / "attempt.yaml").write_text(yaml.safe_dump({
        "project": PROJECT, "run_id": RUN_ID, "attempt_id": ATTEMPT_ID,
    }), encoding="utf-8")
    row = SimpleNamespace(run_id=RUN_ID, run_dir=str(run_dir))
    attempt = SimpleNamespace(
        attempt_id=ATTEMPT_ID, backend_job_id=None, state="SUCCEEDED",
    )
    application = object.__new__(ExperimentServerApplication)

    def conflict_gate(conflicts):
        (attempt_dir / "collection.json").write_text(json.dumps({
            "attempt_id": ATTEMPT_ID, "evidence_conflicts": conflicts,
        }), encoding="utf-8")
        gates = application._attempt_validation_gates(
            PROJECT, row, attempt, attempt_dir, require_current=False,
        )
        return next(item for item in gates if item["id"] == "attempt.evidence_conflicts")

    passed = conflict_gate(_cross_family_conflicts())
    assert passed["status"] == "PASS"
    assert len(passed["evidence"]["reclassified_cross_binding"]) == 4

    rewrite = _cross_family_conflicts()[0]
    rewrite["sources"][1] = {
        **rewrite["sources"][1],
        "binding": dict(rewrite["sources"][0]["binding"]),
    }
    blocked = conflict_gate([rewrite])
    assert blocked["status"] == "BLOCKED"
    assert blocked["evidence"]["conflicts"][0]["metric"] == "g_ppl"
