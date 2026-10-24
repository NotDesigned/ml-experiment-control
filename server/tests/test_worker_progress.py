"""A reported worker stage is exact evidence, never a scheduler heartbeat."""
import sqlite3

import pytest
from fastapi.testclient import TestClient

from ml_exp_server.records.store import RecordStore, encoded
from ml_exp_server.runs import submissions
from ml_exp_server.schemas import RunIndexRow
from tests.test_submissions import _app, _action


IDENTITY = ["demo", "run-a", "attempt-001"]
TIMESTAMP = "2026-01-01T00:00:00Z"
PHASES = ("PREPARING_DATA", "DATA_FAILED", "TRAINING", "TRAINING_COMPLETED",
          "UPLOADING_RESULTS", "RESULTS_UPLOADED", "RESULTS_UPLOAD_FAILED")


def journal(root, *, phase="TRAINING", timestamp=TIMESTAMP):
    store = RecordStore(root)
    store.bind(*IDENTITY[:2], {"enabled": False}, {})
    scope = store.scope(*IDENTITY)
    store.append(scope, [("worker.generation.1", {"kind": "lifecycle", "data": {
        "phase": phase, "timestamp": timestamp}})])
    return store, scope


@pytest.mark.parametrize("phase", PHASES)
def test_worker_stage_is_not_overwritten_by_scheduler_or_diagnostic_events(tmp_path, phase):
    store, scope = journal(tmp_path, phase=phase)
    store.append(scope, [
        ("scheduler.new", {"kind": "lifecycle", "data": {"phase": "RESULTS_UPLOADED", "state": "RUNNING"}}),
        ("worker.generation.2", {"kind": "lifecycle", "data": {"diagnostic": "temporary transfer error"}}),
        ("worker.generation.3", {"kind": "lifecycle", "data": {"phase": "UNRECOGNIZED"}}),
        ("workerXother.4", {"kind": "lifecycle", "data": {"phase": "RESULTS_UPLOADED"}}),
        ("worker.generation.5", {"kind": "metrics", "data": {"phase": "RESULTS_UPLOADED"}}),
        ("WORKER.other.6", {"kind": "lifecycle", "data": {"phase": "RESULTS_UPLOADED"}}),
    ])
    other = store.scope("demo", "run-a", "attempt-002")
    store.append(other, [("worker.other.1", {"kind": "lifecycle", "data": {
        "phase": "RESULTS_UPLOADED", "timestamp": TIMESTAMP}})])
    event = store.latest_worker_lifecycle(scope)
    assert event == {"phase": phase, "timestamp": TIMESTAMP,
                     "event_id": "worker.generation.1", "sequence": 1}
    assert store.latest_worker_lifecycle(encoded(["demo", "run-a", "attempt-999"])) is None


def test_reading_existing_worker_journal_executes_only_selects(tmp_path, monkeypatch):
    store, scope = journal(tmp_path)
    before = store.db.read_bytes()
    connect = sqlite3.connect
    queries = []
    def traced(*args, **kwargs):
        assert str(args[0]).endswith("?mode=ro") and kwargs["uri"] is True
        connection = connect(*args, **kwargs)
        connection.set_trace_callback(queries.append)
        return connection
    monkeypatch.setattr(sqlite3, "connect", traced)
    reader = RecordStore(tmp_path, read_only=True)
    assert queries == []
    assert reader.latest_worker_lifecycle(scope)["phase"] == "TRAINING"
    assert reader.latest_worker_lifecycle(encoded(["demo", "absent", "attempt-001"])) is None
    assert reader.latest_metric_observation(scope) is None
    assert len(queries) == 3 and all(query.lstrip().startswith("SELECT") for query in queries)
    assert store.db.read_bytes() == before


def test_worker_stage_reader_keeps_missing_storage_and_read_errors_unknown(tmp_path):
    absent = tmp_path / "absent"
    assert submissions._journal_observations(absent, IDENTITY) == (None, None)
    assert not absent.exists()
    broken = tmp_path / "tracking"
    broken.mkdir()
    (broken / "outbox.sqlite").write_bytes(b"not a SQLite database")
    assert submissions._journal_observations(tmp_path, IDENTITY) == (None, None)
    (broken / "outbox.sqlite").unlink()
    RecordStore(tmp_path)
    assert submissions._journal_observations(tmp_path, IDENTITY) == (None, None)


@pytest.mark.parametrize("timestamp", [None, 1, "", "bad timestamp", "2026-01-01T00:00:00",
                                        "9999-12-31T23:59:59Z"])
def test_unusable_worker_timestamp_is_not_fresh_progress(tmp_path, timestamp):
    journal(tmp_path, timestamp=timestamp)
    event = submissions._worker_phase(submissions._journal_observations(tmp_path, IDENTITY)[0])
    assert event["phase"] == "TRAINING"
    assert event["observed_at"] is event["age_seconds"] is None


def test_stage_without_timestamp_and_latest_supported_transition(tmp_path):
    store, scope = journal(tmp_path)
    store.append(scope, [("worker.generation.2", {"kind": "lifecycle", "data": {"phase": "UPLOADING_RESULTS"}})])
    event = submissions._worker_phase(submissions._journal_observations(tmp_path, IDENTITY)[0])
    assert event["phase"] == "UPLOADING_RESULTS" and event["sequence"] == 2
    assert event["observed_at"] is event["age_seconds"] is None


@pytest.mark.parametrize("state", ["RUNNING", "STARTING", "QUEUED", "PENDING", "UNKNOWN",
                                   "SUCCEEDED", "FAILED", "PREEMPTED", "CANCELLED"])
def test_submission_progress_preserves_scheduler_and_exact_worker_evidence(tmp_path, state):
    app, runner = _app(tmp_path)
    with TestClient(app) as api:
        prepared = api.post("/api/experiments/demo/run-a/submissions/prepare", json={"max_gpu_hours": 2}).json()
        endpoint = "/api/submissions/" + prepared["submission_id"]
        api.post(endpoint + "/authorize", json={"note": "test"})
        api.post(endpoint + "/execute", json={"confirmation": prepared["confirmation"]})
        assert _action(api, prepared["submission_id"])["execution"]["status"] == "VERIFIED"
        calls = len(runner.calls)
        app.state.index.upsert_run(RunIndexRow(project="demo", run_id="run-a", run_dir=str(tmp_path),
            evidence={"scheduler": {"state": state, "attempt_id": "attempt-001", "as_of": 1000}}))
        store, scope = journal(app.state.config.project_registry_root_path())
        store.append(scope, [("scheduler.later", {"kind": "lifecycle", "data": {"state": state}})])
        response = api.get(endpoint + "/progress")
        assert response.status_code == 200, response.text
        result = response.json()
        event = result["worker_phase"]
        assert result["scheduler_state"] == state and result["attempt_id"] == "attempt-001"
        assert event["phase"] == "TRAINING" and event["observed_at"] == TIMESTAMP
        assert event["source"] == "worker-records" and event["event_id"] == "worker.generation.1"
        assert event["age_seconds"] >= 0 and result["last_progress_unix"] is None
        assert result["phase"] == ("TRAINING" if state == "RUNNING" else state)
        if state == "RUNNING":
            assert result["progress_availability"] == "WORKER_REPORTED"
            assert "last reported" in result["message"] and "not a training heartbeat" in result["message"]
        assert len(runner.calls) == calls


def test_progress_never_uses_another_attempt_or_unusable_time(tmp_path):
    app, _ = _app(tmp_path)
    with TestClient(app) as api:
        prepared = api.post("/api/experiments/demo/run-a/submissions/prepare", json={"max_gpu_hours": 2}).json()
        endpoint = "/api/submissions/" + prepared["submission_id"]
        api.post(endpoint + "/authorize", json={"note": "test"})
        api.post(endpoint + "/execute", json={"confirmation": prepared["confirmation"]})
        _action(api, prepared["submission_id"])
        app.state.index.upsert_run(RunIndexRow(project="demo", run_id="run-a", run_dir=str(tmp_path),
            evidence={"scheduler": {"state": "RUNNING", "attempt_id": "attempt-001", "as_of": 1000}}))
        root = app.state.config.project_registry_root_path()
        store = RecordStore(root)
        store.bind("demo", "run-a", {"enabled": False}, {})
        scope = store.scope("demo", "run-a", "attempt-002")
        store.append(scope, [("worker.other.1", {"kind": "lifecycle", "data": {
            "phase": "TRAINING", "timestamp": TIMESTAMP}})])
        result = api.get(endpoint + "/progress").json()
        assert result["phase"] == "RUNNING" and result["worker_phase"] is None
        assert result["last_metric"] is None
        scope = store.scope(*IDENTITY)
        store.append(scope, [("worker.exact.1", {"kind": "lifecycle", "data": {"phase": "TRAINING"}})])
        observation = {"name": "loss", "value": 3.5, "unit": "nats/token", "step": 2,
                       "protocol_id": "sha256:" + "a" * 64, "dataset_id": "train", "variant_id": "small"}
        store.append(scope, [("metric.exact.1", {"kind": "metrics", "observations": [observation]})])
        result = api.get(endpoint + "/progress").json()
        assert result["phase"] == "RUNNING" and result["worker_phase"]["age_seconds"] is None
        assert result["last_metric"] == {"observation": {**observation, "timestamp": None},
                                        "sequence": 2, "event_id": "metric.exact.1", "source": "record-journal"}


def test_recent_metric_read_retains_context_without_selecting_best_or_largest_step(tmp_path):
    store, scope = journal(tmp_path)
    older = {"name": "loss", "value": 0.1, "unit": "nats/token", "step": 100,
             "protocol_id": "protocol-a", "dataset_id": "train", "variant_id": "a", "timestamp": TIMESTAMP}
    recent = {"name": "loss", "value": 3.5, "unit": "bits/raw byte", "step": 2,
              "protocol_id": "protocol-b", "dataset_id": "benchmark", "variant_id": "b", "checkpoint_id": "frozen",
              "numerator": 7, "denominator": 2, "status": "VALID"}
    store.append(scope, [("metric.older", {"kind": "metrics", "observations": [older]}),
                         ("metric.recent", {"kind": "metrics", "observations": [older, recent]}),
                         ("metric.empty", {"kind": "metrics", "observations": []})])
    other = store.scope("demo", "run-a", "attempt-002")
    store.append(other, [("metric.other", {"kind": "metrics", "observations": [{**older, "step": 9999}]})])
    result = RecordStore(tmp_path, read_only=True).latest_metric_observation(scope)
    assert result == {"observation": {**recent, "timestamp": None}, "event_id": "metric.recent",
                      "sequence": 3, "source": "record-journal"}
    assert "timestamp" not in store.events(scope)[2]["payload"]["observations"][-1]
    failed = {**recent, "value": None, "status": "FAILED", "error": "NONFINITE_OR_MISSING_VALUE", "timestamp": TIMESTAMP}
    store.append(scope, [("metric.failed", {"kind": "metrics", "observations": [failed]})])
    assert store.latest_metric_observation(scope)["observation"] == failed
    assert store.latest_metric_observation(encoded(["demo", "run-a", "attempt-999"])) is None
