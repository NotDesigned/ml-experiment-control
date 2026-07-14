from __future__ import annotations

import json
from types import SimpleNamespace

import yaml

from ml_exp_server.application import ExperimentServerApplication
from ml_exp_server.evidence_conflicts import classify_evidence_conflicts


PROJECT = "elf"
RUN_ID = "elf-aux1-mb64-ga2-h100-20260714-r1"
ATTEMPT_ID = "attempt-001"
OBSERVED_AT = 1784029337.1040568


def _source(*, variant_id: str, family_id: str, metric: str, value: float) -> dict:
    return {
        "source": f"collected_run/train_sampling_eval/{variant_id}/metrics.jsonl",
        "value": value,
        "observed_at": OBSERVED_AT,
        "binding": {
            "project": PROJECT, "run_id": RUN_ID, "attempt_id": ATTEMPT_ID,
            "epoch": 1, "step": 38035, "variant_id": variant_id,
            "family_id": family_id, "metric": metric,
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
                variant_id=f"steps32-{metric}", family_id="sha256:steps32",
                metric=metric, value=left,
            ),
            _source(
                variant_id=f"steps64-{metric}", family_id="sha256:steps64",
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
                variant_id="steps32-generation", family_id="sha256:steps32",
                metric="g_ppl", value=252.03480132729845,
            ),
            _source(
                variant_id="steps32-generation", family_id="sha256:steps32",
                metric="g_ppl", value=999.0,
            ),
        ],
    }
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


def test_incomplete_legacy_conflict_stays_blocking_without_label_guessing():
    raw = "g_ppl has conflicting values in steps32 and steps64"
    blocking, reclassified = classify_evidence_conflicts(
        [raw], project=PROJECT, run_id=RUN_ID, attempt_id=ATTEMPT_ID,
    )
    assert blocking == [raw]
    assert reclassified == []


def test_legacy_source_records_with_direct_exact_bindings_can_be_reclassified():
    sources = []
    for variant_id, family_id, value in (
        ("steps32-generation", "sha256:steps32", 252.03480132729845),
        ("steps64-generation", "sha256:steps64", 238.58811625648886),
    ):
        nested = _source(
            variant_id=variant_id, family_id=family_id,
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
