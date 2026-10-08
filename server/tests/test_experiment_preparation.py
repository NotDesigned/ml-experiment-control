"""Server preparation survives disconnects without repeating uncertain effects."""
import json
import threading
import time
from pathlib import Path

import pytest
import yaml

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.container_execution import ContainerExecutionService
from ml_exp_server.runs.experiment_preparation import ExperimentPreparationRequest
from tests.test_container_api import client, import_source, runtime
from tests.test_sensecore_data_workflow import stored
from tests.test_ccr_data_delivery import delivery, receipt, secret
from tests.test_desktop_data_upload import remote_client, stage


def spec(client, **changes):
    source = import_source(client)
    value = {"runtime": {"source_id": source["source_id"], "entrypoint": ["python3", "train.py"]},
             "run": {"run_id": "trial", "executor": "gpu"}, "max_gpu_hours": 0.2}
    value.update(changes)
    return value


def submission(project, run, **kwargs):
    return {"submission_id": "action-" + "c" * 16, "project": project, "run_id": run,
            "attempt_id": "attempt-001", "ready": True, "status": "PREPARED", "next_action": "AUTHORIZE"}


@pytest.fixture
def service(client, monkeypatch):
    service = client.app.state.preparation_service
    monkeypatch.setattr(service.submissions, "prepare_first_attempt", submission)
    return service


def wait(client, value):
    path = "/api/projects/demo/experiment-preparations/" + value["preparation_id"]
    deadline = time.monotonic() + 5
    while True:
        response = client.get(path)
        assert response.status_code == 200, response.text
        value = response.json()
        if value["status"] != "EXECUTING":
            return path, value
        assert time.monotonic() < deadline, value
        time.sleep(0.01)


@pytest.mark.parametrize("backend, expected", [("slurm", "gpu"), ("sensecore", "cloud")])
def test_one_request_matches_builds_freezes_and_prepares_without_gpu(client, service, backend, expected):
    body = spec(client, run={"run_id": "trial"}, executor_selector={"backend": backend})
    preview = client.post("/api/executors/match", json={"project": "demo", "run": body["run"], "executor_selector": body["executor_selector"]})
    assert preview.status_code == 200, preview.text
    assert preview.json()["executor"] == expected and "profile" not in preview.json()
    response = client.post("/api/projects/demo/experiment-preparations", json=body)
    assert response.status_code == 202, response.text
    path, value = wait(client, response.json())
    assert value["status"] == "READY", value
    assert value["matching"]["executor"] == expected and "profile" not in value["matching"]
    assert value["run"]["executor"] == expected and value["run"]["state"] == "NOT_SUBMITTED"
    assert value["submission"]["status"] == "PREPARED"
    assert "request" not in value and "runtime_requested" not in value
    progress = client.get(path + "/progress").json()
    assert progress["phase"] == "READY" and progress["phase_seconds"] >= 0
    assert client.post("/api/projects/demo/experiment-preparations", json=body).json() == value
    assert client.post(path + "/continue").json() == value
    changed = json.loads(json.dumps(body)); changed["max_gpu_hours"] = 0.3
    assert client.post("/api/projects/demo/experiment-preparations", json=changed).headers["X-ML-Expd-Error-Code"] == "PREPARATION_INTENT_CONFLICT"


def test_duplicate_post_during_build_has_one_owner(client, service, monkeypatch):
    body = spec(client)
    started, release = threading.Event(), threading.Event()
    original = service.containers.execute
    calls = []
    def execute(*args):
        calls.append(args); started.set(); assert release.wait(5)
        return original(*args)
    monkeypatch.setattr(service.containers, "execute", execute)
    first = client.post("/api/projects/demo/experiment-preparations", json=body).json()
    try:
        assert started.wait(2)
        second = client.post("/api/projects/demo/experiment-preparations", json=body).json()
        assert first["preparation_id"] == second["preparation_id"]
        assert second["status"] == "EXECUTING" and len(calls) == 1
    finally:
        release.set()
    assert wait(client, first)[1]["status"] == "READY"


def test_budget_and_matching_rejected_before_build(client, service, monkeypatch):
    body = spec(client, max_gpu_hours=0.01)
    monkeypatch.setattr(service.containers, "execute", lambda *a: pytest.fail("must not build"))
    response = client.post("/api/projects/demo/experiment-preparations", json=body)
    assert response.status_code == 409 and response.headers["X-ML-Expd-Error-Code"] == "GPU_BUDGET_EXCEEDED"
    body["max_gpu_hours"] = 0.2; body["requirements"] = {"offline_output_export": True}
    assert client.post("/api/projects/demo/experiment-preparations", json=body).headers["X-ML-Expd-Error-Code"] == "NO_MATCHING_EXECUTOR"


def test_existing_ready_runtime_is_reused_and_worker_checked(client, service, stored, monkeypatch):
    bundle = runtime(client)
    body = {"run": {"run_id": "reuse", "executor": "gpu", "runtime_id": bundle["runtime_id"]}, "max_gpu_hours": 0.2}
    monkeypatch.setattr(service.containers, "execute", lambda *a: pytest.fail("READY reuse must not build"))
    value, pending = service.accept("demo", ExperimentPreparationRequest.model_validate(body))
    assert service.finish(pending)["status"] == "READY"
    with service.containers.state("demo", bundle["runtime_id"]) as (store, snapshot):
        store.commit({**snapshot.value, "capabilities": []}, expected_revision=snapshot.revision, event={"event": "old_worker"})
    body["run"]["run_id"] = "old-worker"; body["run"]["checkpoint_persistence"] = {}
    response = client.post("/api/projects/demo/experiment-preparations", json=body)
    assert response.headers["X-ML-Expd-Error-Code"] == "RUNTIME_CAPABILITY_MISMATCH"


def test_interrupted_build_is_observed_not_replayed_then_continue_ready(client, service, monkeypatch):
    value, pending = service.accept("demo", ExperimentPreparationRequest.model_validate(spec(client)))
    service.update(*pending, runtime_requested=True)
    assert service.recover_interrupted() == 1 and service.recover_interrupted() == 0
    resumed, pending2 = service.continue_preparation(*pending)
    assert resumed["status"] == "EXECUTING"
    monkeypatch.setattr(service.containers, "execute", lambda *a: pytest.fail("uncertain build cannot replay"))
    stopped = service.finish(pending2)
    assert stopped["status"] == "RECONCILE_REQUIRED" and stopped["error_code"] == "PREPARATION_RUNTIME_NOT_READY"
    bundle = service.containers.read("demo", value["runtime_id"])
    ContainerExecutionService(client.app.state.runtime).execute("demo", bundle["runtime_id"], bundle["confirmation"])
    _, ready = service.continue_preparation(*pending)
    assert service.finish(ready)["status"] == "READY"


def test_executor_drift_blocks_preparation_and_unknown_refs_fail(client, service):
    value, pending = service.accept("demo", ExperimentPreparationRequest.model_validate(spec(client)))
    path = Path(client.app.state.runtime.config.container_execution.profiles_file)
    profiles = yaml.safe_load(path.read_text()); profiles["executors"]["gpu"]["backend"]["partition"] = "different"
    path.write_text(yaml.safe_dump(profiles))
    assert service.finish(pending)["error_code"] == "EXECUTOR_CONFIGURATION_CHANGED"
    missing = "/api/projects/demo/experiment-preparations/preparation-" + "a" * 32
    assert client.get(missing).status_code == 404 and client.post(missing + "/continue").status_code == 404
    assert client.get("/api/projects/demo/experiment-preparations/invalid").status_code == 409


def test_enqueue_failure_is_durable_and_does_not_build(client, service, monkeypatch):
    monkeypatch.setattr(client.app.state, "submit_job", lambda *a: (_ for _ in ()).throw(RuntimeError()))
    value = client.post("/api/projects/demo/experiment-preparations", json=spec(client)).json()
    assert value["status"] == "RECONCILE_REQUIRED"
    assert service.read("demo", value["preparation_id"])["phase"] == "ACCEPTED"


def test_preparation_errors_do_not_expose_exception_secrets(client, service, monkeypatch):
    value, pending = service.accept("demo", ExperimentPreparationRequest.model_validate(spec(client)))
    monkeypatch.setattr(service.containers, "execute", lambda *a: (_ for _ in ()).throw(RuntimeError("secret signed URL")))
    value = service.finish(pending)
    assert value["error_code"] == "PREPARATION_FAILED" and "secret signed URL" not in json.dumps(value)


@pytest.mark.parametrize("runtime_spec, runtime_id", [(None, None), ({"source_id": "source." + "a" * 64, "entrypoint": ["python3"]}, "runtime." + "b" * 64)])
def test_request_needs_exactly_one_runtime(runtime_spec, runtime_id):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ExperimentPreparationRequest(runtime=runtime_spec, run={"run_id": "trial", "executor": "gpu", "runtime_id": runtime_id}, max_gpu_hours=1)


def test_server_prepares_cpu_nas_delivery_before_run_and_reuses_ready(delivery, service, monkeypatch):
    from types import SimpleNamespace
    from ml_exp_server.data.data_delivery import DataDeliveryService
    client, data, value, asset = delivery
    copy_value = value
    events = []
    def create(copy, document, **kwargs):
        events.append("cpu-copy")
        assert copy["gpus"] == 0 and copy["cpus"] == 2
        data.callback("demo", copy_value["delivery_id"], secret(data, copy_value), receipt(copy_value))
    rest = SimpleNamespace(find=lambda *a: [], specs=lambda *a: [{"name": "N6lS.Iu.I10.2c4g", "device": {"number": 0},
        "cpu": {"vcpu_allocatable": 2}, "memory": {"allocatable": 4}}], create=create)
    monkeypatch.setattr(DataDeliveryService, "rest", property(lambda self: rest))
    monkeypatch.setattr(service.submissions, "prepare_first_attempt", lambda *a, **kw: events.append("submission-check") or submission(*a, **kw))
    body = spec(client, run={"run_id": "with-data", "executor": "cloud", "inputs": [{"asset_id": asset["asset_id"], "mount_path": "/inputs/data"}]})
    value, pending = service.accept("demo", ExperimentPreparationRequest.model_validate(body))
    completed = service.finish(pending)
    assert completed["status"] == "READY", completed
    assert events == ["cpu-copy", "submission-check"]
    body["run"]["run_id"] = "reuse-data"
    _, pending = service.accept("demo", ExperimentPreparationRequest.model_validate(body))
    assert service.finish(pending)["status"] == "READY"
    assert events == ["cpu-copy", "submission-check", "submission-check"]
    original_prepare = DataDeliveryService.prepare
    used = []
    def became_ready(self, *a):
        result = original_prepare(self, *a)
        if not used:
            used.append(True)
            result = {**result, "status": "PREPARED"}
        return result
    monkeypatch.setattr(DataDeliveryService, "prepare", became_ready)
    body["run"]["run_id"] = "copy-became-ready"
    _, pending = service.accept("demo", ExperimentPreparationRequest.model_validate(body))
    assert service.finish(pending)["status"] == "READY"
    assert events.count("cpu-copy") == 1
    monkeypatch.setattr(DataDeliveryService, "prepare", original_prepare)
    original_read = DataDeliveryService.read
    statuses = iter(["QUEUED", "READY"])
    def queued_read(self, *a):
        result = original_read(self, *a)
        return {**result, "status": next(statuses, "READY")}
    monkeypatch.setattr(DataDeliveryService, "read", queued_read)
    body["run"]["run_id"] = "shared-copy"
    _, pending = service.accept("demo", ExperimentPreparationRequest.model_validate(body))
    assert service.finish(pending)["status"] == "READY"


def test_uncertain_data_delivery_keeps_exact_id_across_worker_changes(delivery, service, monkeypatch):
    from ml_exp_server.data.data_delivery import DataDeliveryService
    client, data, value, asset = delivery
    data.update("demo", value["delivery_id"], status="RECONCILE_REQUIRED")
    body = spec(client, run={"run_id": "uncertain-data", "executor": "cloud", "inputs": [{"asset_id": asset["asset_id"], "mount_path": "/inputs/data"}]})
    _, pending = service.accept("demo", ExperimentPreparationRequest.model_validate(body))
    stopped = service.finish(pending)
    assert stopped["error_code"] == "PREPARATION_DATA_NOT_READY"
    assert stopped["deliveries"][asset["asset_id"]] == value["delivery_id"]
    monkeypatch.setattr(DataDeliveryService, "prepare", lambda *a: pytest.fail("do not generate a new delivery identity"))
    _, pending = service.continue_preparation(*pending)
    assert service.finish(pending)["error_code"] == "PREPARATION_DATA_NOT_READY"


def test_object_data_is_delivered_by_worker_without_cpu_copy(client, stored, service, monkeypatch):
    from tests.test_artifact_store import archive
    import hashlib
    content = archive({"data.txt": b"training input"})
    imported = client.post("/api/assets/archive", params={"project": "demo", "sha256": hashlib.sha256(content).hexdigest()}, content=content)
    assert imported.status_code == 200, imported.text
    body = spec(client, run={"run_id": "object-data", "executor": "cloud", "inputs": [{"asset_id": imported.json()["asset_id"], "mount_path": "/inputs/data"}]})
    _, pending = service.accept("demo", ExperimentPreparationRequest.model_validate(body))
    assert service.finish(pending)["status"] == "READY"


def test_resume_selection_reads_checkpoint_scope_and_rejects_missing_registry(client, service, stored, monkeypatch):
    from ml_exp_server.results.checkpoint_registry import storage_scope, CheckpointRegistry
    from ml_exp_server.runs.executor_capabilities import ExecutionRequirements, ExecutorSelector
    from ml_exp_server.runs.experiment_preparation import PreparationRun
    profile = service.containers.profiles()["gpu"]
    scope = storage_scope(profile["backend"], profile["storage_root"] + "/demo")
    monkeypatch.setattr(CheckpointRegistry, "read", lambda *a: {"storage_scope": scope})
    run = PreparationRun(run_id="resume", resume_from={"run_id": "prior", "attempt_id": "attempt-001", "checkpoint_id": "checkpoint." + "a" * 64})
    assert service.select("demo", run, ExecutionRequirements(), ExecutorSelector(candidates=["cloud", "gpu"]))["executor"] == "gpu"
    service.runtime.config.container_execution.artifact_store_file = None
    with pytest.raises(ApplicationError, match="registry"):
        service.select("demo", run, ExecutionRequirements(), ExecutorSelector(candidates=["gpu"]))


def test_dependency_observation_waits_only_same_record_and_has_bound(monkeypatch):
    from ml_exp_server.runs import experiment_preparation as module
    calls = []
    monkeypatch.setattr(module.time, "sleep", lambda seconds: calls.append(seconds))
    values = iter([{"status": "EXECUTING"}, {"status": "READY"}])
    assert module.observe_dependency(lambda: next(values), {"EXECUTING"})["status"] == "READY"
    assert calls == [2]
    assert module.observe_dependency(lambda: {"status": "EXECUTING"}, {"EXECUTING"}, timeout=0)["status"] == "EXECUTING"


def test_recovery_ignores_non_preparation_records(client, service):
    directory = service.root / "demo"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "unrelated.json").write_text("{}")
    assert service.recover_interrupted() == 0


@pytest.mark.parametrize("state", ["EXECUTING", "PREPARED"])
def test_competing_runtime_owner_is_observed_without_second_build(client, service, monkeypatch, state):
    body = spec(client)
    value, pending = service.accept("demo", ExperimentPreparationRequest.model_validate(body))
    real = ContainerExecutionService(client.app.state.runtime)
    bundle = real.read("demo", value["runtime_id"])
    ready = real.execute("demo", bundle["runtime_id"], bundle["confirmation"])
    seen = iter(["PREPARED", state, "READY"])
    monkeypatch.setattr(service.containers, "read", lambda *a: {**ready, "status": next(seen, "READY")})
    monkeypatch.setattr(service.containers, "execute", lambda *a: (_ for _ in ()).throw(ApplicationError("claimed by another owner", code="OWNED")))
    result = service.finish(pending)
    assert result["status"] == ("READY" if state == "EXECUTING" else "RECONCILE_REQUIRED")
    if state == "PREPARED":
        assert result["error_code"] == "OWNED"
