"""Completeness is independent from selecting a single evaluation context."""
import copy

import pytest

from ml_exp_server.runs.context import RunContext
from ml_exp_server.records.contract import metric_result


SCHEMA = {"definitions": {
    "squared_error": {"unit": "dimensionless", "required": True},
    "final_squared_error": {"unit": "dimensionless", "aggregation": "ratio_of_sums"},
}}


def curve(**fields):
    return {"name": "squared_error", "unit": "dimensionless", "value": 2.5,
            "step": 1, "protocol_id": "protocol-a", "dataset_id": "tiny-linear-v1", **fields}


def final(**fields):
    return {"name": "final_squared_error", "unit": "dimensionless", "value": 3.5,
            "numerator": 31.5, "denominator": 9,
            "step": 8, "protocol_id": "protocol-a", "dataset_id": "tiny-linear-v1", **fields}


def test_trial_shape_complete_history_remains_complete_with_two_variants_and_final_scalars():
    rows = [curve(step=step, value=step + offset / 10, variant_id=variant)
            for step in range(1, 9) for offset, variant in enumerate(("fast", "slow"))]
    rows.extend(final(variant_id=variant) for variant in ("fast", "slow"))
    original = copy.deepcopy(rows)
    result = RunContext._metric_payload(rows, keys=None, max_points=100, source=None,
              source_attempt_id="attempt-001", contract={"metrics_schema": SCHEMA, "protocol_id": "protocol-a"})
    summary = result["metrics"]
    assert summary["state"] == result["validation"]["state"] == "COMPLETE"
    assert summary["current"] is None and summary["missing_metrics"] == []
    assert len(summary["contexts"]) == 16 and len(summary["records"]) == 18
    assert all(context["state"] == "COMPLETE" for context in summary["contexts"])
    assert result["validation"]["incomplete_contexts"] == 0
    assert result["total_records"] == 18 and not result["downsampled"]
    assert [(row["name"], row["variant_id"], row["step"], row["value"], row.get("numerator"), row.get("denominator"))
            for row in summary["records"]] == [
            (row["name"], row["variant_id"], row["step"], row["value"], row.get("numerator"), row.get("denominator"))
            for row in original]
    assert rows == original
    sampled = RunContext._metric_payload(rows, keys=None, max_points=1, source=None,
              source_attempt_id="attempt-001", contract={"metrics_schema": SCHEMA, "protocol_id": "protocol-a"})
    assert sampled["downsampled"] and sampled["validation"]["state"] == "COMPLETE"
    assert sampled["validation"]["incomplete_contexts"] == 0


@pytest.mark.parametrize("field", ["variant_id", "protocol_id", "dataset_id", "checkpoint_id"])
def test_multiple_complete_bases_do_not_require_an_automatic_current_selection(field):
    result = metric_result([curve(**{field: "one"}), curve(**{field: "two"})], SCHEMA)
    assert result["current"] is None
    assert result["state"] == "COMPLETE" and result["missing_metrics"] == []
    assert len(result["contexts"]) == 2


@pytest.mark.parametrize("field", ["variant_id", "protocol_id", "dataset_id", "checkpoint_id"])
def test_another_base_cannot_supply_a_genuinely_missing_required_metric(field):
    rows = [curve(**{field: "complete"}), final(**{field: "incomplete"})]
    result = metric_result(rows, SCHEMA)
    assert result["current"] is None and result["state"] == "PARTIAL"
    assert result["missing_metrics"] == ["squared_error"]
    assert {context["state"] for context in result["contexts"]} == {"COMPLETE", "PARTIAL"}
    assert all(record["value"] is not None for record in result["records"])


@pytest.mark.parametrize("change", [{"value": float("nan")}, {"unit": "incompatible-unit"}, {"status": "FAILED", "value": None}])
def test_failed_base_is_not_hidden_by_a_complete_variant(change):
    result = metric_result([curve(variant_id="complete"), curve(variant_id="failed", **change)], SCHEMA)
    assert result["current"] is None and result["state"] == "FAILED"
    assert result["missing_metrics"] == ["squared_error"]
    assert next(context for context in result["contexts"] if context["identity"]["variant_id"] == "complete")["state"] == "COMPLETE"
    assert next(context for context in result["contexts"] if context["identity"]["variant_id"] == "failed")["state"] == "FAILED"


def test_optional_partial_record_retains_partial_state_without_fabricating_missing_required_names():
    result = metric_result([curve(variant_id="one"), curve(variant_id="two"),
                            final(step=1, variant_id="two", status="PARTIAL")], SCHEMA)
    assert result["current"] is None and result["state"] == "PARTIAL"
    assert result["missing_metrics"] == []


def test_existing_single_base_current_selection_is_preserved():
    result = metric_result([final(step=1), curve(step=2)], SCHEMA)
    assert result["current"]["identity"]["step"] == 2
    assert result["state"] == "COMPLETE" and result["missing_metrics"] == []
    assert result["contexts"][0]["state"] == "PARTIAL"
    assert result["contexts"][0]["missing_metrics"] == ["squared_error"]


def test_missing_names_are_actual_context_missing_union_in_schema_order():
    schema = {"definitions": {**SCHEMA["definitions"], "another_required": {"unit": "count", "required": True}}}
    result = metric_result([curve(variant_id="one"), curve(variant_id="two")], schema)
    assert result["state"] == "PARTIAL" and result["current"] is None
    assert result["missing_metrics"] == ["another_required"]
    assert all(row["value"] == 2.5 for row in result["records"])
