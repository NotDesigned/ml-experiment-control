"""One metric read format, explicit result intent and scoped requirements."""
import json

import pytest
from pydantic import ValidationError

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.container_controller import parse_metric
from ml_exp_server.runs.context import RunContext
from ml_exp_server.tracking.metric_contract import MetricSchema, metric_result, observations
from ml_exp_server.tracking.wandb_display import curve_record, summaries
from ml_exp_server.tracking.tracking_service import store_for
from ml_exp_server.results.artifact_store import ArtifactStore
from tests.test_container_api import client
from tests.test_sensecore_data_workflow import stored
from tests.test_tracking import create_trial, no_cloud
from tests.test_wandb_display import setup_store, append, metric, handle


def read(rows, **kwargs):
    return RunContext._metric_payload(rows, keys=None, max_points=2, source=None,
        source_attempt_id="attempt-001", format="records", **kwargs)


def test_record_view_filters_names_after_normalization_without_losing_context_or_failures():
    rows = [{"step": 1, "loss": 4, "accuracy": 0.5},
            {"name": "loss", "unit": "nats/token", "value": "nan", "step": 2},
            {"name": "loss", "unit": "nats/token", "value": 2, "step": 3, "variant_id": "a"},
            {"name": "loss", "unit": "nats/token", "value": 3, "step": 3, "variant_id": "b"}]
    contract = {"metrics_schema": {"definitions": {"loss": {"unit": "nats/token"}, "accuracy": {"unit": "fraction"}}}}
    response = read(rows, names=["loss", "absent"], contract=contract)
    assert response["total_records"] == 4 and response["downsampled"]
    assert response["missing_names"] == ["absent"]
    assert [(r["step"], r["value"]) for r in response["records"]] == [(1, 4), (3, 3)]
    assert response["records"][-1]["variant_id"] == "b"
    assert response["validation"]["coverage"] == "complete_history"
    assert response["validation"]["total_records"] == 5  # Legacy wide rows normalize before counting.
    assert response["validation"]["failed_records"] == 1 and response["validation"]["state"] == "FAILED"
    assert not {"metrics", "points", "keys", "evaluation", "source"} & response.keys()
    assert all("source" not in row and "sources" not in row for row in response["records"])
    assert read([], names=["loss"])["validation"]["state"] == "NOT_OBSERVED"
    assert not read(rows, names=["absent"], contract=contract)["records"]
    assert len(read(rows[:1], contract=contract)["records"]) == 2
    last = RunContext._metric_payload(rows, keys=None, max_points=1, source=None,
        source_attempt_id="attempt-001", format="records", names=["loss"], contract=contract)
    assert last["downsampled"] and last["records"][0]["variant_id"] == "b"
    assert read([], contract={"metrics_schema": {"definitions": {
        "loss": {"unit": "nats/token", "required": True}}}})["validation"]["missing_metrics"] == ["loss"]


@pytest.mark.parametrize("options", [
    {"format": "unknown"}, {"format": "legacy", "names": ["loss"]},
    {"format": "records", "keys": "value"},
    *({"format": "records", "names": names} for names in
      ("loss", [""], ["bad\nname"], ["x" * 129], ["loss"] * 257, [True])),
])
def test_invalid_read_options_are_explicit_errors(options):
    with pytest.raises(ApplicationError) as exc:
        RunContext._metric_payload([], **{"keys": None, "max_points": 1, "source": None,
            "source_attempt_id": None, **options})
    assert exc.value.status_code == 422


def test_summary_intent_is_validated_and_independent_from_history_identity():
    history = {"name": "loss", "value": 2, "unit": "nats/token", "step": 2}
    explicit = {**history, "summary": True, "value": 3}
    result = metric_result([history, explicit, explicit])
    assert result["state"] == "COMPLETE" and len(result["records"]) == 2
    assert "summary" not in result["records"][0]  # No legacy journal payload rewrite.
    assert parse_metric(None, json.dumps(explicit)) == explicit
    for bad in ("true", 1, None, {}, []):
        normalized = observations({**history, "summary": bad}, {})[0]
        assert normalized["status"] == "FAILED" and normalized["error"] == "INVALID_SUMMARY_FLAG"
        assert metric_result([parse_metric(None, json.dumps({**history, "summary": bad}))])["state"] == "FAILED"


def test_explicit_summary_is_selected_without_turning_it_into_a_training_point(tmp_path):
    store, scope = setup_store(tmp_path)
    append(store, scope, metric(step=1, value=5), metric(step=2, value=4),
           metric(summary=True, step=1, value=3, checkpoint_id="selected"),
           metric(step=3, value=6))
    display = store.display(scope, 4)
    h = handle()
    summaries(h, {}, display, terminal=False, coverage="full")
    assert "results/validation_loss" not in h.summary
    summaries(h, {}, display, terminal=True, coverage="full")
    assert h.summary["results/validation_loss"] == 3
    definition = json.loads(h.summary["ml_expd/display_definitions_json"])["results/validation_loss"]
    assert definition["selection"] == "explicit_summary" and definition["checkpoint_id"] == "selected"
    assert not curve_record(store.events(scope)[2], display)
    assert curve_record(store.events(scope)[3], display)["curves/validation_loss"] == 6
    # A failed selected result must not fall back to a valid training point.
    store.append(scope, [("failure", {"kind": "metrics", "observations": [
        metric(summary=True, step=2, value=None, status="FAILED")]})])
    summaries(h, dict(h.summary), store.display(scope, 5), terminal=True, coverage="full")
    assert "results/validation_loss" not in h.summary


def test_scoped_requirements_do_not_expect_validation_at_every_training_step():
    schema = {"definitions": {
        "train_loss": {"unit": "nats/token"},
        "validation_loss": {"unit": "nats/token", "required": True,
                            "required_when": {"dataset_id": "validation", "variant_id": "baseline"}},
        "valid_fraction": {"unit": "fraction"},
    }}
    train = {"name": "train_loss", "unit": "nats/token", "value": 4, "step": 1, "dataset_id": "train"}
    validation = {"name": "validation_loss", "unit": "nats/token", "value": 3, "step": 2,
                  "dataset_id": "validation", "variant_id": "baseline"}
    rows = [train, validation]
    assert metric_result(rows, schema)["state"] == "COMPLETE"
    # A separate matching evaluation context remains incomplete: no cross-step borrowing.
    rows.append({**validation, "name": "valid_fraction", "unit": "fraction", "step": 3})
    result = metric_result(rows, schema)
    assert result["state"] == "PARTIAL" and result["missing_metrics"] == ["validation_loss"]
    assert result["contexts"][0]["state"] == "COMPLETE"
    assert result["contexts"][-1]["missing_metrics"] == ["validation_loss"]


def test_schema_additions_preserve_omitted_fields_and_old_protocol_inputs():
    definition = MetricSchema(definitions={"loss": {"unit": "nats/token"}}).model_dump()["definitions"]["loss"]
    assert definition == {"unit": "nats/token", "required": False, "description": "", "aggregation": "scalar",
                          "numerator_unit": None, "denominator_unit": None}
    schema = MetricSchema(definitions={"loss": {"unit": "nats/token", "required_when": {"dataset_id": "val"}}})
    assert schema.model_dump()["definitions"]["loss"]["required_when"] == {"dataset_id": "val"}
    assert MetricSchema(definitions={"loss": {"unit": "u", "required_when": None}}).model_dump()["definitions"]["loss"]["required_when"] is None


@pytest.mark.parametrize("context", [{}, {"step": "1"}, {"dataset_id": ""}, {"dataset_id": "x" * 2001},
                                     {"variant_id": "bad\nvalue"}, {"dataset_id": 1}, []])
def test_invalid_requirement_contexts_rejected(context):
    with pytest.raises(ValidationError):
        MetricSchema(definitions={"loss": {"unit": "nats/token", "required_when": context}})


def test_http_compact_reads_are_exact_and_legacy_payloads_remain_available(client, stored):
    result, request, controller = create_trial(client, schema={"definitions": {"loss,validation": {
        "unit": "nats/token", "required": True, "required_when": {"dataset_id": "val"}}}})
    root = client.app.state.runtime.config.project_registry_root_path()
    artifacts = ArtifactStore(stored[0], root)
    _, token, _ = artifacts.issue("demo", "trial", "attempt-001", controller.root, ["**/*"], record_stream=True)
    payload = {"records": [{"event_id": "selected", "kind": "metrics", "data": {
        "name": "loss,validation", "value": 2.5, "unit": "nats/token", "dataset_id": "val", "step": 1, "summary": True}}]}
    endpoint = "/api/record-transfers/demo/trial/attempt-001"
    headers = {"Authorization": "Bearer " + token}
    assert client.post(endpoint, json=payload, headers=headers).status_code == 200
    original = store_for(client.app.state.runtime).metric_records("demo", "trial", "attempt-001")
    assert client.post(endpoint, json=payload, headers=headers).json()["last_sequence"] == 1
    endpoint = "/api/attempts/demo/trial::attempt-001/metrics"
    params = [("format", "records"), ("name", "loss,validation"), ("name", "absent"), ("max_points", "1")]
    response = client.get(endpoint, params=params)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["project"] == "demo" and body["attempt_id"] == body["source_attempt_id"] == "attempt-001"
    assert body["missing_names"] == ["absent"] and body["records"][0]["summary"] is True
    assert body["records"][0]["name"] == "loss,validation"
    assert body["validation"]["state"] == "COMPLETE" and not body["downsampled"]
    assert "points" in client.get(endpoint).json() and "metrics" not in body
    assert client.get("/api/runs/demo/trial/metrics", params=params).json() == body
    assert client.get(endpoint, params={"format": "records", "keys": "value"}).status_code == 422
    assert client.get(endpoint, params={"name": "loss"}).status_code == 422
    assert client.get(endpoint, params={"format": "invalid"}).status_code == 422
    assert client.post("/api/projects/demo/runs", json=request).json()["evaluation"] == result["evaluation"]
    assert store_for(client.app.state.runtime).metric_records("demo", "trial", "attempt-001") == original
    # A newly frozen Run without an Attempt has no fabricated metric evidence.
    empty = {**request, "run_id": "not-started"}
    assert client.post("/api/projects/demo/runs", json=empty).status_code == 200
    waiting = client.get("/api/runs/demo/not-started/metrics?format=records").json()
    assert waiting["attempt_id"] is None and waiting["records"] == []
    assert waiting["validation"]["state"] == "NOT_OBSERVED"
    assert waiting["validation"]["missing_metrics"] == [] and waiting["protocol_id"] is None
