"""Optional routing, credential isolation, complete metric context and replay."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import runpy
import subprocess
import os
import ast

import pytest
import yaml
from pydantic import ValidationError

from ml_exp_server.tracking_contract import WandbOptions, WandbSettings, finite_parameters
from ml_exp_server.tracking_store import TrackingStore, encoded
from ml_exp_server.tracking_service import store_for, normalized_metric, observe, TrackingPublisher, backfill
from ml_exp_server import wandb_exporter as exporter
from ml_exp_server.tracking_service import preparation_record
from ml_exp_server.worker_records import WorkerRecords, safe_numbers
from ml_exp_server.container_controller import Controller
from ml_exp_server.schemas import RunIndexRow, AttemptSummary
from ml_exp_server.artifact_store import ArtifactStore
from tests.test_container_api import client, runtime
from tests.test_sensecore_data_workflow import stored
from tests.test_experiment_preparation import service, spec, wait


@pytest.fixture(autouse=True)
def no_cloud(monkeypatch):
    monkeypatch.setattr("ml_exp_server.tracking_service.export", lambda *args: None)


def test_default_entity_is_optional_and_credentials_are_never_public(tmp_path):
    store = TrackingStore(tmp_path)
    assert store.public_settings()["enabled"] and not store.public_settings()["credentials_configured"]
    assert store.route(None, "study")["reason"] == "CREDENTIALS_NOT_CONFIGURED"
    called = []
    settings = store.configure(WandbSettings(api_key="private-test-key"), lambda key: called.append(key) or "my-team")
    assert called == ["private-test-key"] and settings["resolved_entity"] == "my-team"
    assert "private-test-key" not in encoded(settings)
    assert store.root.stat().st_mode & 0o777 == 0o700
    assert (store.root / "settings.json").stat().st_mode & 0o777 == 0o600
    assert store.route(None, "study") == {"requested": True, "enabled": True, "reason": None, "entity": "my-team", "project": "study"}
    assert store.route(WandbOptions(project="fineweb", entity="other-team"), "study")["project"] == "fineweb"
    assert store.route(WandbOptions(enabled=False), "study")["reason"] == "DISABLED_BY_REQUEST"
    store.configure(WandbSettings(project="other", enabled=False), lambda key: pytest.fail("unchanged default entity is cached"))
    assert store.public_settings()["resolved_entity"] == "my-team"
    store.configure(WandbSettings(enabled=False, entity="team", project="default"), lambda key: pytest.fail("explicit entity must not be looked up"))
    assert store.route(None, "study")["reason"] == "DISABLED_BY_SERVER"
    assert store.settings()["api_key"] == "private-test-key"
    store.configure(WandbSettings(entity=None, enabled=True), lambda key: None)
    assert store.route(None, "study")["reason"] == "DEFAULT_ENTITY_UNAVAILABLE"
    store.configure(WandbSettings(entity=None), lambda key: "bad/identity")
    assert store.public_settings()["error"] == "DEFAULT_ENTITY_UNAVAILABLE"
    def failing(key): raise RuntimeError("private-test-key")
    store.configure(WandbSettings(entity=None), failing)
    assert "private-test-key" not in encoded(store.public_settings())
    store.configure(WandbSettings(clear_credentials=True), lambda key: pytest.fail("no key"))
    assert store.settings()["api_key"] is None
    assert finite_parameters(None) is None and finite_parameters({"lr": 1e-3}) == {"lr": 1e-3}
    with pytest.raises(ValueError): finite_parameters({"large": "x" * 17000})
    with pytest.raises(ValueError): finite_parameters({"nan": float("nan")})
    with pytest.raises(ValidationError): WandbSettings(api_key="private secret")
    preparation_record(None, {})
    preparation_record(None, {"tracking": {"enabled": False}})


def configured(store):
    store.configure(WandbSettings(api_key="test-key", entity="team"), lambda key: None)
    route = store.route(None, "study")
    store.bind("study", "trial", route, {"ml_expd": {"run_id": "trial"}})
    return store.scope("study", "trial", "attempt-001")


def test_transactional_replay_conflicts_and_remote_confirmation(tmp_path):
    store = TrackingStore(tmp_path)
    scope = configured(store)
    event = {"kind": "metrics", "observations": [{"name": "loss", "unit": "nats/token", "value": 2.5, "status": "VALID"}]}
    assert store.append(scope, [("m1", event)]) == 1
    assert store.append(scope, [("m1", event), ("archive-copy", event)]) == 1
    assert store.events(scope)[0]["event_id"] == "m1"
    assert store.metric_records("study", "trial", "attempt-001")[0]["value"] == 2.5
    with pytest.raises(ValueError): store.append(scope, [("new", {"kind": "lifecycle", "data": {}}), ("m1", {"kind": "metrics", "observations": []})])
    assert store.total(scope) == 1  # all-or-nothing, including the new ID
    assert store.latest_lifecycle(scope) == {}
    store.append(scope, [("state", {"kind": "lifecycle", "data": {"state": "RUNNING"}})])
    assert store.latest_lifecycle(scope) == {"state": "RUNNING"}
    assert store.status("study", "trial")["scopes"][0]["pending"] == 2
    store.outcome(scope, confirmed=1, error="REMOTE_ACK_PENDING")
    assert store.pending()[0]["confirmed"] == 1
    store.outcome(scope, confirmed=2, display_version=2)
    store.outcome(scope, confirmed=1)
    assert store.pending() == []
    assert store.status("study", "trial")["scopes"][0]["status"] == "SYNCED"
    assert store.status("study", "missing")["wandb"]["reason"] == "HISTORICAL_RUN_NOT_BOUND"
    assert TrackingStore(tmp_path).events(scope) == store.events(scope)
    with pytest.raises(ValueError): store.bind("study", "trial", {"enabled": False}, {})
    with pytest.raises(ValueError): store.scope("study", "missing", "attempt-001")
    store.bind("study", "off", {"enabled": False}, {})
    with pytest.raises(ValueError): store.scope("study", "off", "attempt-001")


def create_trial(client, *, name="trial", options=None):
    bundle = runtime(client)
    data = {"run_id": name, "executor": "gpu", "runtime_id": bundle["runtime_id"], "parameters": {"seed": 42, "wd": 0.1},
            "metrics_schema": {"definitions": {"loss": {"unit": "nats/token"}}}, "evaluation": {"benchmark": "fineweb"}}
    if options is not None:
        data["wandb"] = options
    response = client.post("/api/projects/demo/runs", json=data)
    assert response.status_code == 200, response.text
    result = response.json()
    project = client.app.state.runtime.project("demo")
    campaign = yaml.safe_load((Path(project.base_dir) / "experiments/campaigns" / ("run-" + name + ".yaml")).read_text())
    campaign["local_root"] = str(client.app.state.runtime.config.project_run_root_path("demo"))
    controller = Controller(campaign, name, "attempt-001")
    controller.prepare()
    client.app.state.index.upsert_run(RunIndexRow(project="demo", run_id=name, run_dir=str(controller.root),
        scheduler_state="RUNNING", attempts=[AttemptSummary(attempt_id="attempt-001", state="RUNNING", backend="slurm", has_submission=True)]))
    return result, data, controller


def test_http_routes_freeze_per_run_and_preserve_old_requests(client, stored, monkeypatch):
    monkeypatch.setattr("ml_exp_server.api.tracking_routes.default_entity", lambda key: "team")
    assert client.put("/api/tracking/wandb", json={"api_key": "private-test-key"}).json()["resolved_entity"] == "team"
    assert "private-test-key" not in client.get("/api/tracking/wandb").text
    invalid = client.put("/api/tracking/wandb", json={"api_key": "private test secret"})
    assert invalid.status_code == 422 and "private test secret" not in invalid.text
    first, data, controller = create_trial(client, options={"project": "fineweb"})
    assert first["wandb"]["project"] == "fineweb" and first["wandb"]["record_stream"]
    client.put("/api/tracking/wandb", json={"entity": "changed", "project": "changed"})
    assert client.post("/api/projects/demo/runs", json=data).json()["wandb"] == first["wandb"]
    changed = {**data, "wandb": {"project": "different"}}
    assert client.post("/api/projects/demo/runs", json=changed).status_code == 409
    assert client.post("/api/projects/demo/runs", json={**data, "parameters": {"seed": 43}}).status_code == 409
    second, _, _ = create_trial(client, name="second", options={"enabled": False})
    assert second["wandb"]["reason"] == "DISABLED_BY_REQUEST"
    assert client.get("/api/runs/demo/trial/tracking").json()["wandb"] == first["wandb"]
    # The launch contains a scoped platform record capability, never W&B auth.
    controller.dispatch_command({"command": controller.command("attempt-001"), "attempt_id": "attempt-001"})
    transfer = ArtifactStore(stored[0], client.app.state.runtime.config.project_registry_root_path())
    with transfer.record("demo", "trial", "attempt-001") as (_, value):
        assert value["record_stream"]
        assert "ML_EXPD_RECORD_URL" in value["launch_manifest"]["environment"]
        assert "private-test-key" not in encoded(value)


def test_exact_attempt_metric_stream_is_queryable_before_results(client, stored, monkeypatch):
    client.put("/api/tracking/wandb", json={"api_key": "test-key", "entity": "team"})
    result, data, controller = create_trial(client)
    store = ArtifactStore(stored[0], client.app.state.runtime.config.project_registry_root_path())
    _, token, _ = store.issue("demo", "trial", "attempt-001", controller.root, ["**/*"], record_stream=True)
    endpoint = "/api/record-transfers/demo/trial/attempt-001"
    records = [{"event_id": "metric." + variant, "kind": "metrics", "data": {"name": "loss", "unit": "nats/token", "value": value, "step": 1, "variant_id": variant}}
               for variant, value in (("baseline", 2.5), ("other", 3.0))]
    headers = {"Authorization": "Bearer " + token}
    assert client.post(endpoint, json={"records": records}, headers=headers).status_code == 200
    assert client.post(endpoint, json={"records": records}, headers=headers).json()["last_sequence"] == 2
    payload = client.get("/api/attempts/demo/trial::attempt-001/metrics").json()
    assert {item["variant_id"] for item in payload["metrics"]["records"]} == {"baseline", "other"}
    assert all(item["status"] == "VALID" for item in payload["metrics"]["records"])
    assert client.get("/api/runs/demo/trial/metrics").json()["total_records"] == 2
    records[0]["data"]["value"] = 0
    assert client.post(endpoint, json={"records": records}, headers=headers).status_code == 409
    assert client.post(endpoint, json={"records": records}).status_code == 409
    assert client.post(endpoint, json={"records": records}, headers={"Authorization": "Basic invalid"}).status_code == 409
    assert client.post(endpoint.replace("attempt-001", "attempt-002"), json={"records": records}, headers=headers).status_code == 409
    assert client.post(endpoint, content=b"x" * (512 * 1024 + 1), headers=headers).status_code == 413
    for body in ({}, {"records": []}, {"records": [{}]}, {"records": [{"event_id": "..", "kind": "other", "data": {}}]},
                 {"records": [{"event_id": "event", "kind": "lifecycle", "data": {"api_key": "secret"}}]},
                 {"records": [{"event_id": "event", "kind": "lifecycle", "data": {"state": []}}]}):
        assert client.post(endpoint, json=body, headers=headers).status_code == 409
    assert client.post(endpoint, content=b"not-json", headers=headers).status_code == 409
    _, other_token, _ = store.issue("demo", "trial", "attempt-003", controller.root, ["**/*"])
    assert client.post(endpoint.replace("attempt-001", "attempt-003"), json={"records": records}, headers={"Authorization": "Bearer " + other_token}).status_code == 409
    client.app.state.runtime.config.container_execution.artifact_store_file = None
    assert client.post(endpoint, json={"records": records}, headers=headers).status_code == 409
    client.app.state.runtime.config.container_execution.artifact_store_file = str(stored[0])
    assert client.post(endpoint, json={"records": [{"event_id": "state", "kind": "lifecycle", "data": {"phase": "TRAINING"}}]}, headers=headers).status_code == 200
    tracking = store_for(client.app.state.runtime)
    scope = tracking.pending()[0]
    tracking.outcome(scope["id"], error="REMOTE_ACK_PENDING")
    assert client.post("/api/runs/demo/trial/tracking/retry").status_code == 200
    client.put("/api/tracking/wandb", json={"enabled": False})
    assert client.put("/api/tracking/wandb", json={"clear_credentials": True}).json()["credentials_configured"] is False


def test_observer_preserves_checkpoint_identity_and_receipts(client, stored):
    client.put("/api/tracking/wandb", json={"api_key": "test-key", "entity": "team"})
    _, _, controller = create_trial(client)
    runtime = client.app.state.runtime
    store = store_for(runtime)
    root = runtime.config.project_registry_root_path()
    checkpoint = root / "persistent-checkpoints/demo/trial/attempt-001"
    checkpoint.mkdir(parents=True)
    (checkpoint / ("checkpoint." + "a" * 64 + ".json")).write_text(encoded({"checkpoint_id": "checkpoint." + "a" * 64,
        "step": 10, "files": [{"path": "final.pt", "sha256": "a" * 64, "bytes": 100}], "registered_at": "now", "status": "REGISTERED"}))
    transfer = ArtifactStore(stored[0], root)
    transfer.issue("demo", "trial", "attempt-001", controller.root, ["**/*"])
    with transfer.record("demo", "trial", "attempt-001") as (path, value):
        value["receipt"] = {"sha256": "b" * 64, "bytes": 100, "files": [], "received_at": "now", "object_key": "private/key"}
        path.write_text(encoded(value))
    outputs = controller.attempt / "uploaded_outputs"
    outputs.mkdir(parents=True)
    (outputs / "metrics.jsonl").write_text(encoded({"name": "loss", "value": 2.0, "unit": "nats/token", "step": 2}) + "\n")
    observe(runtime, store)
    scope = store.scope("demo", "trial", "attempt-001")
    events = store.events(scope)
    assert {e["payload"]["kind"] for e in events} == {"lifecycle", "metrics", "artifacts", "checkpoints"}
    assert "private/key" not in encoded(events)
    observe(runtime, store)
    assert store.events(scope) == events
    assert client.get("/api/runs/demo/trial/metrics").json()["total_records"] == 1
    assert client.get("/api/attempts/demo/trial::attempt-001/metrics").json()["total_records"] == 1
    with transfer.record("demo", "trial", "attempt-001") as (path, value):
        value["receipt"] = None
        path.write_text(encoded(value))
    (outputs / "metrics.jsonl").unlink()
    observe(runtime, store)
    # An uploaded full history remains authoritative in the public metrics API.
    assert client.get("/api/runs/demo/trial/metrics").json()["total_records"] == 1
    client.app.state.index.upsert_run(RunIndexRow(project="demo", run_id="not-opted-in", run_dir=str(controller.root)))
    observe(runtime, store)


def fake_sdk():
    rows, handles = [], []
    class Remote:
        summary = {}
        @property
        def lastHistoryStep(self): return max((r["ml_expd/sequence"] - 1 for r in rows), default=-1)
        def scan_history(self, **kw):
            assert not kw["use_cache"]
            return [row for row in rows if row["ml_expd/sequence"] > kw["min_step"]]
    class API:
        def __init__(self, **kw): assert kw["overrides"]["base_url"] == "https://api.wandb.ai"
        default_entity = "default-team"
        def run(self, path): return Remote()
    class Handle:
        def __init__(self): self.summary = {}; self.logs = []; self.finished = False
        def define_metric(self, *a, **k): pass
        def log(self, record, *, step, commit):
            assert commit is True
            assert step == record["ml_expd/sequence"] - 1
            self.logs.append(record)
        def finish(self): self.finished = True
    def initialize(**kw):
        assert kw["reinit"] == "create_new" and kw["resume"] == "allow"
        handle = Handle(); handles.append(handle); return handle
    return SimpleNamespace(Api=API, Settings=lambda **kw: kw, init=initialize, teardown=lambda: None), rows, handles


def test_official_sdk_resume_and_ack_protocol_preserves_all_context(tmp_path, monkeypatch):
    sdk, rows, handles = fake_sdk()
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    assert exporter.default_entity("private") == "default-team"
    store = TrackingStore(tmp_path); scope_id = configured(store)
    event = normalized_metric({"name": "encoded_loss", "value": 1.2, "unit": "bits/raw byte", "step": 4,
        "variant_id": "a", "checkpoint_id": "cp", "dataset_id": "programs", "numerator": 10, "denominator": 2}, {})
    store.append(scope_id, [("metric-a", event)])
    scope = store.pending()[0]
    job = {**store.binding("study", "trial"), "scope": scope, "api_key": "test-key", "session_root": str(store.root), "events": store.events(scope_id), "terminal": False}
    sessions = {}
    result = exporter.publish(job, sdk, sessions)
    assert result == {"confirmed": 0, "error": "REMOTE_ACK_PENDING"}
    assert not handles[0].finished
    assert exporter.publish(job, sdk, sessions)["error"] == "REMOTE_ACK_PENDING"
    assert len(handles[0].logs) == 1  # locally queued != remote acknowledged
    rows.extend(handles[0].logs)
    assert exporter.publish(job, sdk, sessions)["error"] == "REMOTE_DISPLAY_ACK_PENDING"
    sdk.Api(overrides={"base_url": "https://api.wandb.ai"}).run("").summary.update(handles[0].summary)
    # Hand-written fixture jobs do not carry a projection cutoff; real jobs do.
    job["display"] = {"through": 1, "series": {}}
    assert exporter.publish(job, sdk, sessions)["error"] == "REMOTE_DISPLAY_ACK_PENDING"
    sdk.Api(overrides={"base_url": "https://api.wandb.ai"}).run("").summary.update(handles[0].summary)
    assert exporter.publish(job, sdk, sessions) == {"confirmed": 1, "display_version": 2, "error": None}
    record, definitions = exporter.history_record(job["events"][0])
    context = list(definitions.values())[0]
    assert context["unit"] == "bits/raw byte" and context["variant_id"] == "a" and context["checkpoint_id"] == "cp"
    key = list(definitions)[0]
    assert record[key + "/step"] == 4 and record[key + "/numerator"] == 10
    second = normalized_metric({"name": "encoded_loss", "unit": "bits/raw byte", "value": None, "status": "FAILED", "variant_id": "b"}, {})
    other, keys = exporter.history_record({"seq": 2, "event_id": "bad", "payload": second})
    assert list(keys)[0] != key and list(keys)[0] not in other  # failures are never zero
    record, definitions = exporter.history_record({"seq": 2, "event_id": "receipt", "payload": {"kind": "artifacts", "data": {"sha256": "a" * 64}}})
    assert "sha256" in record["ml_expd/record"] and not definitions
    job["terminal"] = True
    exporter.publish(job, sdk, sessions)
    assert handles[0].finished and not sessions


@pytest.mark.parametrize("mode", ["mismatch", "hole", "pending-mismatch", "init-failed"])
def test_publisher_never_silently_replays_conflicting_or_missing_history(tmp_path, mode):
    sdk, rows, handles = fake_sdk()
    event = {"seq": 1, "event_id": "one", "payload": {"kind": "lifecycle", "data": {"state": "RUNNING"}}}
    job = {"route": {"entity": "team", "project": "study"}, "scope": {"wandb_id": "123", "run": "trial", "attempt": "attempt-001", "project": "study", "confirmed": 0},
        "config": {"ml_expd": {"run_id": "trial"}}, "api_key": "key", "session_root": str(tmp_path), "events": [event], "terminal": False}
    sessions = {}
    if mode == "mismatch": rows.append({"ml_expd/sequence": 1, "ml_expd/event_id": "different"})
    if mode == "hole": rows.append({"ml_expd/sequence": 2, "ml_expd/event_id": "two"})
    if mode == "pending-mismatch":
        exporter.publish(job, sdk, sessions)
        job["events"][0]["event_id"] = "different"
    if mode == "init-failed":
        def fail(**kw): raise RuntimeError("private-key")
        sdk.init = fail
        with pytest.raises(RuntimeError): exporter.publish(job, sdk, sessions)
        assert not list(tmp_path.iterdir())
    else:
        result = exporter.publish(job, sdk, sessions)
        assert result["error"] in {"REMOTE_IDENTITY_CONFLICT", "REMOTE_HISTORY_RECONCILE_REQUIRED"}
        if mode != "pending-mismatch": assert not handles[0].logs
        for session in sessions.values(): session["directory"].cleanup()


def test_export_keeps_backlog_on_failure_and_uses_no_key_in_argv(tmp_path):
    store = TrackingStore(tmp_path); scope_id = configured(store)
    store.append(scope_id, [("state", {"kind": "lifecycle", "data": {"state": "SUCCEEDED"}})])
    jobs = []
    def failure(job): jobs.append(job); raise RuntimeError("test-key")
    exporter.export(store, store.pending()[0], SimpleNamespace(call=failure, close=lambda: None))
    assert store.status("study", "trial")["scopes"][0]["error"] == "PUBLISHER_UNAVAILABLE"
    assert jobs[0]["terminal"]
    exporter.export(store, store.pending()[0], SimpleNamespace(call=lambda job: {"confirmed": 1, "display_version": 2, "error": None}))
    assert store.pending() == []
    store.append(scope_id, [("another", {"kind": "lifecycle", "data": {"phase": "TRAINING"}})])
    store.configure(WandbSettings(clear_credentials=True), lambda key: None)
    exporter.export(store, store.pending()[0], SimpleNamespace(call=lambda job: pytest.fail("no configured key")))
    assert store.pending()[0]["error"] == "CREDENTIALS_NOT_CONFIGURED"


def test_worker_metric_cursor_retries_without_loss_and_survives_restart(tmp_path, monkeypatch):
    root = tmp_path / "outputs"; root.mkdir()
    body = {"name": "loss", "value": 2.0, "unit": "nats/token", "step": 1, "variant_id": "a"}
    (root / "metrics.jsonl").write_text(encoded(body) + "\n" + "not-json\n" + "{\"partial\":")
    calls, status = [], [503]
    class Connection:
        def request(self, method, path, *, body, headers): calls.append(json.loads(body))
        def getresponse(self): return SimpleNamespace(status=status[0], read=lambda n: b"{}")
        def close(self): pass
    monkeypatch.setattr("ml_exp_server.worker_records.https_connection", lambda *a, **k: Connection())
    worker = WorkerRecords(root, "https://api.example/api/record-transfers/demo/trial/attempt-001", "private")
    worker.emit("lifecycle", {"phase": "TRAINING"})
    worker.flush(final=True)
    assert worker.state["offset"] == 0
    retried = WorkerRecords(root, worker.url, "private")
    status[0] = 200
    retried.flush(final=True)
    assert calls[-1] == calls[-2]
    assert retried.state["queue"] == [] and retried.state["offset"] > 0
    assert calls[-1]["records"][1]["data"]["variant_id"] == "a"
    assert calls[-1]["records"][2]["data"]["diagnostic"] == "INVALID_METRIC_JSONL"
    retried.flush(final=True)
    assert len(calls) == 2  # incomplete tail is never consumed
    (root / "metrics.jsonl").write_text(encoded(body) + "\n")
    retried.flush(final=True)  # truncated/replaced file starts a new generation
    assert len(calls) == 3
    disabled = WorkerRecords(root, "", "private")
    disabled.emit("lifecycle", {}); disabled.flush()
    assert safe_numbers([float("nan"), {"inf": float("inf")}, 2]) == ["nan", {"inf": "inf"}, 2]
    retried.url = "http://api.example/api/record-transfers/demo/trial/attempt-001"
    retried.emit("lifecycle", {}); retried.flush(final=True)
    assert retried.state["queue"]


def test_preparation_freezes_target_before_build_and_keeps_failure_evidence(client, service, monkeypatch):
    client.put("/api/tracking/wandb", json={"api_key": "test-key", "entity": "team", "project": "before"})
    from ml_exp_server.experiment_preparation import ExperimentPreparationRequest
    request = ExperimentPreparationRequest.model_validate(spec(client, run={"run_id": "trial", "executor": "gpu", "parameters": {"seed": 42}}))
    accepted, pending = service.accept("demo", request)
    client.put("/api/tracking/wandb", json={"entity": "new-team", "project": "after"})
    finished = service.finish(pending)
    assert finished["status"] == "READY", finished
    assert finished["tracking"]["project"] == "before" and finished["run"]["wandb"]["entity"] == "team"
    store = store_for(client.app.state.runtime)
    status = client.get("/api/runs/demo/trial/tracking").json()
    assert status["preparation"][0]["attempt"] == "preparation"
    assert status["scopes"] == []  # preparation is not a fabricated Attempt
    original = store.events(store.scope("demo", "trial--preparation", "preparation"))
    assert service.accept("demo", request)[0] == finished
    assert store.events(store.scope("demo", "trial--preparation", "preparation")) == original
    failed, pending = service.accept("demo", ExperimentPreparationRequest.model_validate(spec(client, run={"run_id": "failed", "executor": "gpu"})))
    monkeypatch.setattr(service.containers, "execute", lambda *a: (_ for _ in ()).throw(RuntimeError("secret")))
    # READY shared Runtime avoids execution; force the observed build to fail.
    monkeypatch.setattr(service.containers, "read", lambda *a: {"status": "FAILED"})
    finished = service.finish(pending)
    assert finished["status"] == "RECONCILE_REQUIRED"
    assert client.get("/api/runs/demo/failed/tracking").json()["preparation"]
    observe(client.app.state.runtime, store)


def test_historical_backfill_is_explicit_read_only_and_target_frozen(client, stored):
    result, _, controller = create_trial(client, options={"enabled": False})
    store = store_for(client.app.state.runtime)
    assert not store.pending()
    response = client.post("/api/runs/demo/trial/tracking/backfill", json={})
    assert response.json()["wandb"]["reason"] == "CREDENTIALS_NOT_CONFIGURED"
    manifest_bytes = controller.store.manifest_path.read_bytes()
    client.put("/api/tracking/wandb", json={"api_key": "test-key", "entity": "team"})
    output = controller.attempt / "uploaded_outputs"
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.jsonl").write_text(encoded({"name": "loss", "unit": "nats/token", "value": 2.5, "variant_id": "a"}) + "\n")
    response = client.post("/api/runs/demo/trial/tracking/backfill", json={"project": "history"})
    assert response.status_code == 200, response.text
    assert response.json()["scopes"] and response.json()["wandb"]["project"] == "history"
    scope = store.scope("demo", "trial--backfill", "attempt-001")
    assert "completeness not guaranteed" in encoded(store.events(scope))
    assert controller.store.manifest_path.read_bytes() == manifest_bytes
    client.put("/api/tracking/wandb", json={"entity": "changed", "project": "other"})
    assert client.post("/api/runs/demo/trial/tracking/backfill", json={}).json()["wandb"]["project"] == "history"
    assert client.post("/api/runs/demo/trial/tracking/backfill", json={"project": "changed"}).status_code == 409
    assert client.post("/api/runs/demo/trial/tracking/backfill", json={"entity": "changed"}).status_code == 409
    assert client.post("/api/runs/demo/trial/tracking/backfill", json={"enabled": False}).status_code == 409
    assert client.post("/api/runs/demo/missing/tracking/backfill", json={}).status_code == 404


def test_preexisting_run_without_tracking_is_not_implicitly_opted_in(client):
    result, data, controller = create_trial(client)
    project = client.app.state.runtime.project("demo")
    path = Path(project.base_dir) / "experiments/campaigns/run-trial.yaml"
    payload = yaml.safe_load(path.read_text())
    payload["runs"][0].pop("tracking")
    payload["runs"][0].pop("wandb_request")
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    client.put("/api/tracking/wandb", json={"api_key": "test-key", "entity": "team"})
    assert client.post("/api/projects/demo/runs", json=data).json()["wandb"]["reason"] == "HISTORICAL_RUN_NOT_BOUND"


def test_publisher_loop_isolated_failure_and_interruptible(tmp_path, monkeypatch):
    runtime = SimpleNamespace(config=SimpleNamespace(project_registry_root_path=lambda: tmp_path), index=SimpleNamespace(list_runs=lambda: []))
    publisher = TrackingPublisher(runtime)
    publisher.cycle()  # unconfigured is entirely idle
    scope = configured(publisher.store)
    publisher.store.append(scope, [("one", {"kind": "lifecycle", "data": {"state": "RUNNING"}})])
    calls = []
    monkeypatch.setattr("ml_exp_server.tracking_service.export", lambda *args: calls.append(args))
    publisher.cycle(); assert len(calls) == 1
    publisher.store.outcome(scope, error="REMOTE_ACK_PENDING")
    publisher.cycle(); assert len(calls) == 1  # backoff
    with publisher.store.connection() as conn:
        conn.execute("UPDATE scopes SET last_attempt_at='2020-01-01T00:00:00Z'")
    publisher.cycle(); assert len(calls) == 2
    publisher.stop.set(); publisher.cycle(); assert len(calls) == 2
    publisher.stop.clear()
    publisher.start(); publisher.close()
    assert not publisher.thread.is_alive()
    def fail(): raise RuntimeError("private")
    publisher.cycle = fail
    signals = iter((False, True))
    publisher.stop = SimpleNamespace(wait=lambda seconds: next(signals))
    publisher.loop()
    assert json.loads((publisher.store.root / "publisher-error.json").read_text())["code"] == "OBSERVATION_FAILED"


def test_worker_spool_failures_and_bounded_lines_cannot_stop_training(tmp_path, monkeypatch):
    root = tmp_path / "outputs"; root.mkdir()
    (tmp_path / "worker-records.json").write_text("not-json")
    broken = WorkerRecords(root, "https://api.example/path", "private")
    assert broken.url == ""
    (tmp_path / "worker-records.json").unlink()
    worker = WorkerRecords(root, "https://api.example/path", "private")
    def fail(): raise OSError("no space")
    monkeypatch.setattr(worker, "save", fail)
    worker.emit("lifecycle", {"phase": "TRAINING"})
    assert worker.state["queue"]

    monkeypatch.undo()
    received = []
    class Connection:
        def request(self, *a, **k): received.append(json.loads(k["body"]))
        def getresponse(self): return SimpleNamespace(status=200, read=lambda n: b"{}")
        def close(self): pass
    monkeypatch.setattr("ml_exp_server.worker_records.https_connection", lambda *a, **k: Connection())
    (root / "metrics.jsonl").write_text("x" * 65537 + "\n" + encoded([1, 2]) + "\n")
    worker.flush(final=True)
    assert any(r["data"].get("diagnostic") == "INVALID_METRIC_JSONL" for r in received[0]["records"])
    worker.last = float("inf"); worker.flush(); assert len(received) == 1
    (root / "metrics.jsonl").unlink(); (root / "metrics.jsonl").symlink_to("/no-file")
    worker.emit("lifecycle", {}); worker.flush(final=True)
    assert worker.state["queue"]


def test_sdk_process_credentials_rotation_timeout_and_owned_cleanup(tmp_path, monkeypatch):
    native_popen = subprocess.Popen
    commands = []
    # A real child tests bounded private IPC and process cleanup, without
    # touching W&B's network. It also resists TERM to test the hard-stop path.
    driver = "import json,sys,signal; signal.signal(signal.SIGTERM,signal.SIG_IGN)\nfor line in sys.stdin:\n j=json.loads(line); print(json.dumps({'confirmed':j['events'][-1]['seq'],'error':None}),flush=True)\n"
    def create(command, **kw):
        commands.append((command, kw))
        return native_popen([sys.executable, "-uc", driver], **kw)
    monkeypatch.setattr(exporter.subprocess, "Popen", create)
    monkeypatch.setenv("WANDB_API_KEY", "unrelated-private-key")
    publisher = exporter.PublisherProcess()
    job = {"api_key": "private-key", "session_root": str(tmp_path), "events": [{"seq": 1}]}
    try:
        assert publisher.call(job)["confirmed"] == 1
        first = publisher.child.pid
        assert publisher.call(job)["confirmed"] == 1 and publisher.child.pid == first
        publisher.child.kill(); publisher.child.wait()
        assert publisher.call(job)["confirmed"] == 1 and publisher.child.pid != first
        assert "private-key" not in encoded(commands[0][0])
        assert "WANDB_API_KEY" not in commands[0][1]["env"]
        publisher.call({**job, "api_key": "different-key"})
        assert publisher.child.pid != first
        monkeypatch.setattr(exporter.select, "select", lambda *args: ([], [], []))
        with pytest.raises(TimeoutError): publisher.call(job)
        assert publisher.child is None
    finally:
        publisher.close()
    assert not list(tmp_path.iterdir())


def test_publisher_main_outputs_only_fixed_diagnostics(tmp_path, monkeypatch, capsys):
    sdk, rows, handles = fake_sdk()
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    monkeypatch.setattr(sys, "stdin", io.StringIO("invalid-json\n{}\n"))
    exporter.main()
    assert [json.loads(line) for line in capsys.readouterr().out.splitlines()] == [{"error": "WANDB_REQUEST_FAILED"}] * 2
    parsed = ast.parse(Path(exporter.__file__).read_text())
    called = []
    exec(compile(ast.Module(body=[parsed.body[-1]], type_ignores=[]), exporter.__file__, "exec"),
         {"__name__": "__main__", "main": lambda: called.append(True)})
    assert called == [True]


def test_worker_records_standalone_and_batch_size(tmp_path, monkeypatch):
    from ml_exp_server import worker_records, worker_http
    monkeypatch.setitem(sys.modules, "worker_http", worker_http)
    runpy.run_path(worker_records.__file__, run_name="standalone-import")
    root = tmp_path / "outputs"; root.mkdir()
    accepted = []
    class Connection:
        def request(self, *a, **kw): accepted.append(kw["body"])
        def getresponse(self): return SimpleNamespace(status=200, read=lambda n: b"{}")
        def close(self): pass
    monkeypatch.setattr(worker_records, "https_connection", lambda *a, **kw: Connection())
    # Long Unicode records stay below the server's byte limit despite JSON's
    # ASCII escaping; a second batch resumes at the unconsumed record.
    metric = {"name": "loss", "unit": "nats/token", "value": 1, "other": "字" * 7000}
    (root / "metrics.jsonl").write_text((json.dumps(metric, ensure_ascii=False) + "\n") * 20)
    worker = WorkerRecords(root, "https://api.example/path", "private")
    worker.flush(final=True)
    assert 1 <= len(json.loads(accepted[-1])["records"]) < 20 and len(accepted[-1]) <= 512 * 1024
    worker.flush(final=True)
    assert len(accepted) == 2
    (root / "metrics.jsonl").unlink()
    worker.emit("lifecycle", {}); worker.flush(final=True)
    assert worker.state["queue"] == []
    worker.flush(final=True)
    worker.emit("checkpoints", {"checkpoint_id": "test"}); worker.flush(final=True)
    (root / "metrics.jsonl").write_text((encoded({"name": "loss", "unit": "nats/token", "value": 1}) + "\n") * 70)
    worker.flush(final=True)
    assert len(json.loads(accepted[-1])["records"]) == 64


def test_invalid_remote_cursor_does_not_lose_pending_records(tmp_path):
    store = TrackingStore(tmp_path); scope = configured(store)
    store.append(scope, [("one", {"kind": "lifecycle", "data": {"state": "RUNNING"}})])
    exporter.export(store, store.pending()[0], SimpleNamespace(call=lambda j: {"confirmed": 999}, close=lambda: None))
    assert store.pending()[0]["confirmed"] == 0
    store.configure(WandbSettings(api_key="test-key", entity="team", enabled=False), lambda key: None)
    exporter.export(store, store.pending()[0], SimpleNamespace(call=lambda j: pytest.fail("paused server")))
    assert store.pending()[0]["error"] == "DISABLED_BY_SERVER"


def test_pinned_native_sdk_accepts_concurrent_offline_handles(tmp_path, monkeypatch):
    import wandb
    sdk, rows, handles = fake_sdk()
    # Real SDK/native core and metric serialization, with cloud reads injected
    # and SDK offline mode. This never claims actual W&B cloud acceptance.
    native = SimpleNamespace(Api=sdk.Api, init=wandb.init,
        Settings=lambda **kw: wandb.Settings(mode="offline", **kw))
    monkeypatch.setenv("WANDB_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("WANDB_CACHE_DIR", str(tmp_path / "cache"))
    scope = {"wandb_id": "test123", "run": "trial", "attempt": "attempt-001", "project": "study", "confirmed": 0}
    job = {"route": {"entity": "team", "project": "study"}, "scope": scope,
        "config": {"ml_expd": {"run_id": "trial"}}, "api_key": "x" * 40, "session_root": str(tmp_path),
        "events": [{"seq": 1, "event_id": "one", "payload": normalized_metric({"name": "loss", "unit": "nats/token", "value": 2.5, "step": 1, "variant_id": "a"}, {})}], "terminal": False}
    sessions = {}
    try:
        assert exporter.publish(job, native, sessions)["error"] == "REMOTE_ACK_PENDING"
        assert exporter.publish({**job, "scope": {**scope, "wandb_id": "test456", "attempt": "attempt-002"}}, native, sessions)["error"] == "REMOTE_ACK_PENDING"
        assert len(sessions) == 2
        for session in sessions.values():
            # Explicit step must commit while the session is still alive.
            # Otherwise its last event cannot be remotely acknowledged.
            assert session["handle"].step == 1
            session["handle"].finish()
            assert list(Path(session["directory"].name).rglob("*.wandb"))
    finally:
        wandb.teardown()
        for session in sessions.values(): session["directory"].cleanup()


def test_display_repair_only_changes_metadata_and_checks_confirmed_history(tmp_path):
    store=TrackingStore(tmp_path);scope=configured(store)
    store.append(scope, [('one', normalized_metric({'name':'validation_loss','unit':'nats/token','value':3.8,'step':6112},{}))])
    store.outcome(scope,confirmed=1)
    jobs=[]
    exporter.export(store,store.pending()[0],SimpleNamespace(call=lambda job: jobs.append(job) or {'confirmed':1,'display_version':2,'error':None},close=lambda:None))
    job=jobs[0]; assert job['events']==[] and job['expected_event_id']=='one'
    sdk,rows,handles=fake_sdk();rows.append({'ml_expd/sequence':1,'ml_expd/event_id':'one'})
    remote=sdk.Api(overrides={'base_url':'https://api.wandb.ai'}).run('')
    remote.summary['metrics/old/step']=6112
    sessions={}
    assert exporter.publish(job,sdk,sessions)['error']=='REMOTE_DISPLAY_ACK_PENDING'
    assert handles[0].logs==[] # Metadata migration never replays accepted points.
    remote.summary.update(handles[0].summary)
    assert exporter.publish(job,sdk,sessions)['display_version']==2
    rows[0]['ml_expd/event_id']='changed'
    assert exporter.publish(job,sdk,sessions)['error']=='REMOTE_HISTORY_RECONCILE_REQUIRED'
    for session in sessions.values(): session['directory'].cleanup()


@pytest.mark.parametrize('outcome',[{'confirmed':1,'display_version':999}, {'confirmed':0,'display_version':2}, {'confirmed':1,'display_version':2,'error':'pending'}])
def test_invalid_display_confirmation_never_clears_repair(tmp_path,outcome):
    store=TrackingStore(tmp_path);scope=configured(store)
    store.append(scope,[('one',{'kind':'lifecycle','data':{'state':'RUNNING'}})])
    exporter.export(store,store.pending()[0],SimpleNamespace(call=lambda j:outcome,close=lambda:None))
    assert store.pending()[0]['display_version']==0
    assert store.pending()[0]['error']=='PUBLISHER_UNAVAILABLE'


def test_projection_failure_isolated_from_other_publications(tmp_path,monkeypatch):
    store=TrackingStore(tmp_path);scope=configured(store)
    store.append(scope,[('one',{'kind':'lifecycle','data':{}})])
    monkeypatch.setattr(store,'display',lambda *a:(_ for _ in ()).throw(ValueError('DISPLAY_CONTEXT_LIMIT')))
    exporter.export(store,store.pending()[0],SimpleNamespace(close=lambda:None))
    assert store.pending()[0]['error']=='PUBLISHER_UNAVAILABLE'
