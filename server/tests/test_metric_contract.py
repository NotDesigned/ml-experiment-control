"""Units, scientific identity and evidence isolation are project-independent."""
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ml_exp_server.metric_contract import MetricSchema, finite, metric_result, observations, protocol_identity
from ml_exp_server.ingest.runscan import evaluation_snapshot, evaluation_variants, scan_run_dir
from ml_exp_server.container_controller import Controller
from ml_exp_server.container_execution import ContainerExecutionService, RunRequest
from ml_exp_server.application_errors import ApplicationError
from tests.test_container_api import client, runtime


SCHEMA = {"schema_version": 1, "definitions": {
    "fineweb_validation_loss": {"unit": "nats/token", "required": True,
                               "aggregation": "ratio_of_sums", "numerator_unit": "nats", "denominator_unit": "token"},
    "encoded_loss": {"unit": "bits/raw byte"},
}}


def row(**values):
    return {"name": "fineweb_validation_loss", "value": 2.5, "unit": "nats/token",
            "numerator": 25, "denominator": 10, "step": 4, **values}


def test_schema_has_no_builtin_scientific_metric_requirements():
    assert MetricSchema().definitions == {}
    result = metric_result([row()], SCHEMA)
    assert result["state"] == "COMPLETE"
    assert result["required_metrics"] == ["fineweb_validation_loss"]
    assert result["records"][0]["numerator"] == 25
    assert result["records"][0]["denominator"] == 10
    assert result["records"][0]["value"] == 2.5
    # A byte denominator has its own identity; no implicit conversion/selection.
    byte = metric_result([{"name": "encoded_loss", "unit": "bits/raw byte", "value": 3.6}], SCHEMA)
    assert byte["state"] == "PARTIAL" and byte["records"][0]["value"] == 3.6


@pytest.mark.parametrize("definition", [
    {"unit": ""}, {"unit": "  "}, {"unit": "bad\nunit"}, {"unit": "x", "unknown": True},
])
def test_bad_units_or_schema_fields_rejected(definition):
    with pytest.raises(ValidationError):MetricSchema(definitions={"loss": definition})


@pytest.mark.parametrize("name", [" ", "bad\nname", "x" * 129])
def test_invalid_schema_names_rejected(name):
    with pytest.raises(ValidationError):MetricSchema(definitions={name: {"unit": "nats/token"}})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), None, "bad", True, 10**400])
def test_nonfinite_or_missing_value_is_failure_never_zero(value):
    result = metric_result([row(value=value)], SCHEMA)
    assert result["state"] == "FAILED"
    assert result["records"][0]["value"] is None
    json.dumps(result, allow_nan=False)
    assert not finite(value)


@pytest.mark.parametrize("change,error", [
    ({"unit": "bits/raw byte"}, "UNIT_MISMATCH"),
    ({"unit": " "}, "UNIT_NOT_DECLARED"),
    ({"unit": "u" * 129}, "UNIT_NOT_DECLARED"),
    ({"unit": "bad\nunit"}, "UNIT_NOT_DECLARED"),
    ({"name": "unlisted", "unit": "nats"}, "METRIC_NOT_DECLARED"),
    ({"status": "BOGUS"}, "INVALID_STATUS"),
    ({"denominator": 0}, "INVALID_AGGREGATION_COUNTS"),
    ({"numerator": float("inf")}, "INVALID_AGGREGATION_COUNTS"),
    ({"numerator": None}, "INVALID_AGGREGATION_COUNTS"),
    ({"protocol_id": {}}, "INVALID_PROTOCOL_ID"),
    ({"protocol_id": "wrong"}, "PROTOCOL_MISMATCH"),
])
def test_invalid_observations_preserve_diagnostics(change, error):
    result = metric_result([row(**change)], SCHEMA, protocol_id="p1")
    assert error in result["records"][0]["errors"]
    assert result["state"] != "COMPLETE"


@pytest.mark.parametrize("name", [None, [], "", "bad\nname", "x" * 129])
def test_malformed_named_record_is_not_treated_as_a_metric_named_value(name):
    result = metric_result([{"name": name, "value": 1}], SCHEMA)
    assert result["state"] == "FAILED"
    assert result["records"][0]["name"] is None


def test_missing_and_partial_are_distinct_from_failures():
    assert metric_result([], SCHEMA)["state"] == "NOT_OBSERVED"
    for state in ("MISSING", "PARTIAL"):
        result = metric_result([row(status=state, value=None)], SCHEMA)
        assert result["state"] == "PARTIAL"
        assert result["records"][0]["status"] == state
    assert metric_result([row(status="FAILED", value=None, error="scorer crashed")], SCHEMA)["state"] == "FAILED"


def test_wide_legacy_records_remain_queryable_without_invented_units():
    result = metric_result([{"step": 1, "epoch": 0, "timestamp": "today", "checkpoint_id": "c1",
                             "dataset_id": "d1", "loss": 1.2, "accuracy": True, "nested": []}])
    assert result["state"] == "PARTIAL"
    assert result["records"][0]["unit"] is None
    assert result["records"][0]["value"] == 1.2
    assert len(result["records"]) == 1
    assert metric_result([{"step": 1, "loss": 1.2}], {"definitions": {"loss": {"unit": "nats/token"}}})["state"] == "COMPLETE"


def test_distinct_checkpoint_dataset_protocol_and_variant_contexts_are_not_mixed():
    for field in ("checkpoint_id", "dataset_id", "protocol_id", "variant_id"):
        result = metric_result([row(**{field: "a"}), row(step=5, **{field: "b"})], SCHEMA)
        assert result["current"] is None and result["state"] == "PARTIAL"
        assert len(result["contexts"]) == 2
    # Interleaved steps never fill the missing validation loss at another step.
    result = metric_result([row(step=1), {"name": "encoded_loss", "unit": "bits/raw byte", "value": 1, "step": 2}], SCHEMA)
    assert result["state"] == "PARTIAL" and result["current"]["identity"]["step"] == 2
    result = metric_result([row(step=1), row(step=None)], SCHEMA)
    assert result["current"] is None


def test_identical_observations_idempotent_and_conflicts_never_corrected_silently():
    assert len(metric_result([row(), row()], SCHEMA)["records"]) == 1
    result = metric_result([row(), row(value=3), row()], SCHEMA)
    assert result["state"] == "FAILED"
    assert result["records"][0]["status"] == "CONFLICTING"
    assert result["records"][0]["value"] is None


def test_protocol_id_is_canonical_and_changes_with_scoring_protocol():
    first = {"D_FT": 0, "scored_bytes": 256, "tokenizer": "sha256:a", "loss_mask": "targets"}
    assert protocol_identity(first) == protocol_identity(dict(reversed(list(first.items()))))
    assert protocol_identity(first) != protocol_identity({**first, "scored_bytes": 512})


def test_run_contract_freezes_project_default_and_scoring_protocol(client):
    bundle = runtime(client)
    endpoint = "/api/projects/demo/metrics-schema"
    assert client.get(endpoint).json()["metrics_schema"]["definitions"] == {}
    assert client.put(endpoint, json=SCHEMA).status_code == 200
    requested = {"run_id": "eval-a", "runtime_id": bundle["runtime_id"], "executor": "gpu",
                 "evaluation": {"D_FT": 0, "scored_bytes": 256, "tokenizer": "sha256:a", "loss_mask": "targets"}}
    response = client.post("/api/projects/demo/runs", json=requested)
    assert response.status_code == 200, response.text
    evaluation = response.json()["evaluation"]
    assert evaluation["metrics_schema"]["definitions"]["fineweb_validation_loss"]["unit"] == "nats/token"
    assert evaluation["protocol_id"] == protocol_identity(evaluation["protocol"])
    root = Path(client.app.state.runtime.project("demo").base_dir)
    campaign_path = root / "experiments/campaigns/run-eval-a.yaml"
    original = campaign_path.read_bytes()
    campaign = yaml.safe_load(original)
    campaign["local_root"] = str(client.app.state.runtime.config.project_run_root_path("demo"))
    controller = Controller(campaign, "eval-a", "attempt-001")
    controller.prepare()
    assert controller.store.load_manifest()["evaluation"] == evaluation
    assert client.put(endpoint, json={"definitions": {"other": {"unit": "fraction"}}}).status_code == 200
    assert campaign_path.read_bytes() == original
    requested["run_id"] = "eval-b"
    changed = client.post("/api/projects/demo/runs", json=requested).json()["evaluation"]
    assert changed["protocol_id"] != evaluation["protocol_id"]
    assert client.put(endpoint, json={"definitions": {"loss": {"unit": ""}}}).status_code == 422
    client.app.state.runtime.config.action_runtime.allow_project_writes = False
    assert client.put(endpoint, json=SCHEMA).status_code == 409


def test_schema_write_requires_authored_project_manifest(client):
    runtime(client)
    service = ContainerExecutionService(client.app.state.runtime)
    client.app.state.runtime.project("demo").authored_file = None
    with pytest.raises(ApplicationError, match="manifest is unavailable"):
        service.set_metrics_schema("demo", MetricSchema.model_validate(SCHEMA))


def test_evaluation_context_must_be_json_finite_and_bounded():
    for value in ({"bad": float("nan")}, {"data": "x" * 16384}):
        with pytest.raises(ValidationError):RunRequest(run_id="r", runtime_id="runtime."+"a"*64, executor="gpu", evaluation=value)


def test_indexed_evaluation_uses_free_names_and_exact_attempt_binding(tmp_path):
    run = tmp_path / "run"
    evidence = run / "attempts/a1/uploaded_outputs/benchmark"
    evidence.mkdir(parents=True)
    manifest = {"project": "demo", "run_id": "run", "evaluation": {"metrics_schema": SCHEMA, "protocol_id": "p1"}}
    (run / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    (run / "status.json").write_text(json.dumps({"attempt_id": "a1", "state": "SUCCEEDED"}))
    (evidence / "metrics.jsonl").write_text(json.dumps(row())+'\n')
    indexed = scan_run_dir(run, "demo")
    assert indexed.eval_metrics == {"fineweb_validation_loss": 2.5}
    assert indexed.evaluation_snapshot["current"]["state"] == "COMPLETE"
    assert indexed.evaluation_snapshot["attempt_binding_state"] == "EXACT_ATTEMPT_BOUND"
    # Evidence from a prior Attempt can never satisfy the new Attempt.
    (run / "attempts/a2").mkdir()
    (run / "status.json").write_text('{"attempt_id":"a2","state":"RUNNING"}')
    assert scan_run_dir(run, "demo").eval_metrics == {}
    assert evaluation_variants(run, attempt_id="a2", exact_attempt=True)[0] == []


def test_unbound_legacy_eval_preserved_as_history_without_scientific_promotion(tmp_path):
    run = tmp_path / "run"
    evidence = run / "collected_run/benchmark"
    evidence.mkdir(parents=True)
    (run / "manifest.yaml").write_text("run_id: run\n")
    (evidence / "metrics.jsonl").write_text(json.dumps(row())+'\n')
    indexed = scan_run_dir(run, "demo")
    assert indexed.eval_metrics == {}
    assert indexed.evaluation_snapshot["attempt_binding_state"] == "EXACT_ATTEMPT_NOT_BOUND"
    assert indexed.eval_variants[0]["latest"]["name"] == "fineweb_validation_loss"


def test_scalar_eval_history_is_bounded_and_excludes_large_task_arrays(tmp_path):
    run = tmp_path / "run"
    evidence = run / "attempts/a1/uploaded_outputs/benchmark"
    evidence.mkdir(parents=True)
    (evidence / "metrics.jsonl").write_text(''.join(json.dumps(row(step=i, tasks=[1]*100))+'\n' for i in range(40)))
    variant = evaluation_variants(run, attempt_id="a1", exact_attempt=True)[0][0]
    assert variant["history_truncated"] and len(variant["history"]) == 32
    assert all("tasks" not in item for item in variant["history"])
    assert evaluation_snapshot([variant], {"metrics_schema": SCHEMA})["current"]["step"] == 39


def test_different_units_never_silently_override_even_without_schema():
    result = metric_result([{"name": "loss", "value": 1, "unit": "nats/token"},
                            {"name": "loss", "value": 2, "unit": "bits/raw byte"}])
    assert result["state"] == "FAILED"
    assert len(result["records"]) == 2
    assert all(item["error"] == "CONFLICTING_UNITS" for item in result["records"])


def test_controller_retains_typed_units_and_failure_evidence():
    from ml_exp_server.container_controller import parse_metric
    assert parse_metric(None, json.dumps(row())) == row()
    record = parse_metric(None, '{"name":"loss","value":NaN,"unit":"nats/token","step":4,"tasks":[]}')
    assert record["status"] == "FAILED" and record["value"] is None
    assert record["error"] == "NONFINITE_VALUE" and "tasks" not in record
    result = metric_result([record])
    assert result["state"] == "FAILED"
    assert observations(row(protocol_id='x'*2001), SCHEMA)[0]["status"] == "FAILED"


def test_source_provenance_and_invalid_status_survive_normalization():
    result = metric_result([row(status=[]), row(status=[])], SCHEMA, source="/run/metrics.jsonl")
    assert result["state"] == "FAILED"
    assert result["records"][0]["sources"] == ["/run/metrics.jsonl"]
    assert result["records"][0]["source"] == "/run/metrics.jsonl"


def test_nonfinite_uploaded_history_is_json_safe_in_run_summary(tmp_path):
    run = tmp_path / "run"
    attempt = run / "attempts/a1"
    attempt.mkdir(parents=True)
    (run / "manifest.yaml").write_text("run_id: run\n")
    (run / "status.json").write_text(json.dumps({"attempt_id": "a1", "state": "RUNNING", "metrics": {"nested": [], "bad": float('inf')}}))
    (run / "collection.json").write_text(json.dumps({"bad": float('nan'), "attempt_id": "a1"}))
    (attempt / "train_metrics.jsonl").write_text(json.dumps(row(value=float('nan'), tasks=[], timestamp=100))+'\n')
    indexed = scan_run_dir(run, "demo")
    assert indexed.latest_metrics["value"] is None and indexed.latest_metrics["bad"] is None
    json.dumps(indexed.model_dump(mode="json"), allow_nan=False)
    assert evaluation_snapshot([])["current"]["state"] == "NOT_OBSERVED"


def test_unkeyed_scalar_history_accepts_empty_or_missing_steps():
    from ml_exp_server.ingest.runscan import _evaluation_history
    assert _evaluation_history([])["history"] == []
    assert _evaluation_history([{"name": "loss", "value": 1, "unit": "nats/token"}])["history"]


def test_controller_does_not_replace_malformed_unit_or_status_with_schema_defaults():
    from ml_exp_server.container_controller import parse_metric
    for invalid in ({"unit": []}, {"status": {}}, {"name": []}):
        parsed = parse_metric(None, json.dumps(row(**invalid)))
        assert metric_result([parsed], SCHEMA)["state"] == "FAILED"
