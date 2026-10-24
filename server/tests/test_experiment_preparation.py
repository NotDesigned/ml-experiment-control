"""Server preparation survives disconnects without repeating uncertain effects."""
from copy import deepcopy
import json
import shlex
import threading
import time
from pathlib import Path
from types import SimpleNamespace

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
    from tests.test_sensecore_data_workflow import put_asset
    imported = put_asset(client, content)
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


@pytest.fixture
def cpu_download(client, stored, service, monkeypatch):
    """Real preparation/campaign/state stores with a bounded, fake CPU provider."""
    from ml_exp_server.backends.sensecore_preparation import PREFIX, SenseCoreScriptPreparation
    from tests.test_container_api import archive
    config = json.loads(stored[0].read_text())
    config["local_storage_headroom_bytes"] = 0  # Tiny fixture assets do not test host disk admission.
    config["data_delivery"] = {"aec2": "debug-cluster-01e", "worker_spec": "N6lS.Iu.I10.2c4g",
                               "gpus": 0, "cpus": 2, "memory_gb": 4,
                               "queue_timeout_seconds": 5, "copy_timeout_seconds": 5}
    stored[0].write_text(json.dumps(config))
    source = import_source(client, archive({"train.py": b"print('training')\n", "download.py": b"print('download')\n",
                                           "Dockerfile": ("FROM registry.example/python@sha256:" + "a" * 64 + "\n").encode()}))
    bundle = runtime(client, source)
    control = SimpleNamespace(state="SUCCEEDED", uncertain=False, creates=[], stops=[], jobs={}, events=[])
    def create(profile, body, **kwargs):
        assert (profile["gpus"], profile["cpus"], profile["memory_gb"], profile["quota_type"]) == (0, 2, 4, "reserved")
        assert profile["aec2"] == "debug-cluster-01e" and profile["workspace"] == "lab"
        assert body["roles"][0]["image_path"] == bundle["image"]
        definition = json.loads(shlex.split(body["roles"][0]["startup_script"])[-1])
        control.creates.append((deepcopy(profile), deepcopy(body)))
        control.events.append("cpu-script")
        prepared = service.read("demo", control.preparation_id)
        assert prepared["phase"] == "PREPARING_DATA_SCRIPT" and prepared["run"]["state"] == "NOT_SUBMITTED"
        files = [{"path": "tokens.bin", "bytes": 17, "sha256": "f" * 64}]
        from ml_exp_server.workers.data_preparation import identity
        sealed = {"schema_version": 1, "status": "READY", "definition": definition["definition"],
                  "preparation_id": definition["definition"]["preparation_id"], "files": files,
                  "content_sha256": identity(files), "dataset_id": "dataset." + identity(files), "bytes": 17}
        control.jobs[body["name"]] = {"uid": "e" * 32, "receipt": sealed}
        if definition["contract"] == "cpu-script-preparation.v2" and control.state == "SUCCEEDED":
            command = shlex.split(body["roles"][0]["startup_script"])
            environment = dict(item.split("=", 1) for item in command[5:7])
            request = json.loads(environment["ML_EXPD_SCRIPT_RECEIPT_REQUEST"])
            callback = SenseCoreScriptPreparation(control.config, service.runtime.config.project_registry_root_path())
            callback.callback("demo", request["operation_id"], environment["ML_EXPD_SCRIPT_RECEIPT_TOKEN"],
                              {key: value for key, value in request.items() if key != "url"} | {"receipt": sealed})
        if control.uncertain:
            raise RuntimeError("private provider URL and token")
        return {"name": body["name"], "uid": "e" * 32}
    rest = SimpleNamespace(
        specs=lambda _: [{"name": "N6lS.Iu.I10.2c4g", "device": {"number": 0},
                          "cpu": {"vcpu_allocatable": 2}, "memory": {"allocatable": 4}}],
        create=create,
        find=lambda profile, name: [{"name": name, "uid": control.jobs[name]["uid"]}] if name in control.jobs else [],
        describe=lambda profile, name: {"name": name, "uid": control.jobs[name]["uid"], "state": control.state},
        logs=lambda profile, name, lines: {"text": ""},
        stop=lambda *args: control.stops.append(args))
    monkeypatch.setattr(SenseCoreScriptPreparation, "rest", property(lambda self: rest))
    monkeypatch.setattr(service.submissions, "prepare_first_attempt",
                        lambda *a, **kw: control.events.append("submission-check") or submission(*a, **kw))
    body = {"run": {"run_id": "cpu-download", "executor": "cloud", "runtime_id": bundle["runtime_id"],
                    "data_preparation": {"script": "download.py", "timeout_seconds": 5}},
            "requirements": {"data_preparation_before_job": True}, "max_gpu_hours": 0.2}
    control.body, control.config, control.service, control.client = body, stored[0], service, client
    return control


def accept_cpu(control):
    value, pending = control.service.accept("demo", ExperimentPreparationRequest.model_validate(control.body))
    control.preparation_id = value["preparation_id"]
    return value, pending


def frozen_campaign(client, value):
    path = Path(client.app.state.runtime.project("demo").base_dir) / "experiments/campaigns" / (value["campaign"] + ".yaml")
    return path, yaml.safe_load(path.read_text())["runs"][0]


def test_download_script_preparation_finishes_zero_gpu_phase_before_submission(cpu_download):
    control = cpu_download
    value, pending = accept_cpu(control)
    result = control.service.finish(pending)
    assert result["status"] == "READY", result
    assert control.events == ["cpu-script", "submission-check"] and len(control.creates) == 1
    assert result["run"]["data_preparation_execution"]["phase"] == "before_job"
    assert result["run"]["data_preparation_execution"]["budget"] == "preparation"
    assert result["data_script"]["status"] == "READY" and result["data_script"]["receipt"]["bytes"] == 17
    _, frozen = frozen_campaign(control.client, result["run"])
    assert frozen["backend"]["aec2"] == "compute" and frozen["backend"]["quota_type"] == "spot"
    assert frozen["data_preparation_cpu"]["gpus"] == 0
    assert frozen["data_preparation_receipt_mode"] == "callback_v1"
    from ml_exp_server.backends.sensecore_preparation import script_receipt_worker_sha256
    assert frozen["data_preparation_receipt_worker_sha256"] == script_receipt_worker_sha256()
    assert result["data_script"]["receipt_source"] == "NAS_SEALED_CALLBACK"
    assert control.service.progress(*pending)["data_script"] == result["data_script"]
    journal = control.service.root / "demo" / (value["preparation_id"] + ".jsonl")
    assert '"PREPARING_DATA_SCRIPT"' in journal.read_text()
    same, next_pending = control.service.continue_preparation(*pending)
    assert same == result and next_pending is None and len(control.creates) == 1


@pytest.mark.parametrize("change", ["different-config", "removed-config"])
def test_new_run_freezes_cpu_profile_and_repeat_create_preserves_identity(cpu_download, change):
    control = cpu_download
    request = control.body["run"]
    first = control.client.post("/api/projects/demo/runs", json=request)
    assert first.status_code == 200, first.text
    path, original = frozen_campaign(control.client, first.json())
    before = path.read_bytes()
    config = json.loads(control.config.read_text())
    if change == "different-config":
        config["data_delivery"].update(aec2="debug-new-pool", queue_timeout_seconds=60)
    else:
        config.pop("data_delivery")
    control.config.write_text(json.dumps(config))
    again = control.client.post("/api/projects/demo/runs", json=request)
    assert again.status_code == 200, again.text
    assert again.json()["data_preparation_execution"]["phase"] == "before_job"
    assert path.read_bytes() == before
    assert frozen_campaign(control.client, again.json())[1] == original
    assert not control.creates


@pytest.mark.parametrize("seconds", [5, 7200])
def test_legacy_run_repeat_create_does_not_upgrade_data_mode_or_identity(cpu_download, seconds):
    control = cpu_download
    config = json.loads(control.config.read_text())
    configured = config.pop("data_delivery")
    control.config.write_text(json.dumps(config))
    request = deepcopy(control.body["run"])
    request["data_preparation"]["timeout_seconds"] = seconds
    first = control.client.post("/api/projects/demo/runs", json=request)
    assert first.status_code == 200, first.text
    path, frozen = frozen_campaign(control.client, first.json())
    assert "data_preparation_mode" not in frozen and "data_preparation_cpu" not in frozen
    assert "data_preparation_receipt_mode" not in frozen
    assert "data_preparation_receipt_worker_sha256" not in frozen
    before = path.read_bytes()
    config["data_delivery"] = configured
    control.config.write_text(json.dumps(config))
    again = control.client.post("/api/projects/demo/runs", json=request)
    assert again.status_code == 200, again.text
    assert again.json()["data_preparation_execution"]["phase"] == "inside_job"
    assert path.read_bytes() == before and not control.creates


def test_uncertain_cpu_create_continues_same_frozen_operation_after_config_change(cpu_download):
    control = cpu_download
    control.uncertain = True
    _, pending = accept_cpu(control)
    failed = control.service.finish(pending)
    assert failed["status"] == "RECONCILE_REQUIRED" and failed["error_code"] == "DATA_SCRIPT_PROVIDER_ERROR"
    assert "private provider" not in json.dumps(failed)
    original = failed["data_script"]["operation_id"]
    config = json.loads(control.config.read_text())
    config["data_delivery"].update(aec2="debug-different", queue_timeout_seconds=60)
    control.config.write_text(json.dumps(config))
    control.uncertain = False
    _, continued = control.service.continue_preparation(*pending)
    ready = control.service.finish(continued)
    assert ready["status"] == "READY", ready
    assert ready["data_script"]["operation_id"] == original and len(control.creates) == 1
    assert control.events == ["cpu-script", "submission-check"]
    assert control.service.continue_preparation(*pending)[1] is None


def test_failed_cpu_script_preserves_original_job_and_never_retries_on_continue(cpu_download):
    control = cpu_download
    control.state = "FAILED"
    _, pending = accept_cpu(control)
    failed = control.service.finish(pending)
    assert failed["error_code"] == "DATA_SCRIPT_FAILED" and control.events == ["cpu-script"]
    original = failed["data_script"]["operation_id"]
    for _ in range(2):
        _, continued = control.service.continue_preparation(*pending)
        result = control.service.finish(continued)
        assert result["status"] == "RECONCILE_REQUIRED" and result["data_script"]["operation_id"] == original
        assert result["error_code"] == "DATA_SCRIPT_FAILED"
    assert len(control.creates) == 1 and not control.stops
    assert control.service.progress(*pending)["data_script"]["status"] == "FAILED"


def test_changed_preparation_script_reference_is_rejected_without_new_job(cpu_download):
    control = cpu_download
    control.uncertain = True
    _, pending = accept_cpu(control)
    failed = control.service.finish(pending)
    assert failed["status"] == "RECONCILE_REQUIRED"
    control.service.update(*pending, data_script={**failed["data_script"], "operation_id": "script." + "f" * 64})
    _, continued = control.service.continue_preparation(*pending)
    result = control.service.finish(continued)
    assert result["error_code"] == "DATA_SCRIPT_IDENTITY_CHANGED" and len(control.creates) == 1
    assert control.events == ["cpu-script"]


def test_scheduler_policy_disables_cpu_script_before_provider_call(cpu_download):
    control = cpu_download
    _, pending = accept_cpu(control)
    control.service.runtime.config.action_runtime.allow_scheduler_mutations = False
    result = control.service.finish(pending)
    assert result["error_code"] == "DATA_SCRIPT_DISABLED" and not control.creates
    assert not control.events


def test_script_timeout_exceeding_cpu_limit_is_rejected_before_runtime_build(cpu_download, monkeypatch):
    control = cpu_download
    control.body["run"]["data_preparation"]["timeout_seconds"] = 3601
    monkeypatch.setattr(control.service.containers, "prepare", lambda *a: pytest.fail("matching rejects before build"))
    with pytest.raises(ApplicationError) as error:
        accept_cpu(control)
    assert error.value.code == "NO_MATCHING_EXECUTOR" and not control.creates


def test_preparation_request_hash_keeps_explicit_research_options_and_old_false_default(client, service):
    body = spec(client, run={"run_id": "hash-compatible", "executor": "cloud",
                            "wandb": {"enabled": False}, "parameters": {"seed": 42}})
    value, _ = service.accept("demo", ExperimentPreparationRequest.model_validate(body))
    with service.state("demo", value["preparation_id"]) as (_, snapshot):
        assert "data_preparation_before_job" not in snapshot.value["request"]["requirements"]
        assert snapshot.value["request"]["run"]["wandb"]["enabled"] is False
        assert snapshot.value["request"]["run"]["parameters"] == {"seed": 42}
    body["requirements"] = {"data_preparation_before_job": False}
    assert service.accept("demo", ExperimentPreparationRequest.model_validate(body)) == (value, None)


@pytest.mark.parametrize('historical',[False,True])
def test_dlc_preparation_freezes_effective_day_budget_and_preserves_old_intents(client,service,monkeypatch,historical):
    from tests.test_pai_integration import B
    from ml_exp_server.runs import experiment_preparation as module
    path=Path(client.app.state.runtime.config.container_execution.profiles_file)
    profiles=yaml.safe_load(path.read_text());profiles['executors']['pai']={'backend':B,'storage_root':'/data/lab','capacity':{'gpus':8,'cpus':216,'memory_gb':1800}};path.write_text(yaml.safe_dump(profiles))
    body=spec(client,run={'run_id':'day-budget','executor':'pai'},max_gpu_hours=0.2)
    original=module.run_walltime
    if historical:monkeypatch.setattr(module,'run_walltime',lambda resources,*args:resources.max_time)
    else:
        with pytest.raises(ApplicationError) as error:service.accept('demo',ExperimentPreparationRequest.model_validate(body))
        assert error.value.code=='GPU_BUDGET_EXCEEDED'
        body['run']['resources']={'max_time':'24:00:01'}
        with pytest.raises(ApplicationError) as error:service.accept('demo',ExperimentPreparationRequest.model_validate(body))
        assert error.value.code=='PAI_DURATION_LIMIT'
        body['run'].pop('resources');body['max_gpu_hours']=24
    value,pending=service.accept('demo',ExperimentPreparationRequest.model_validate(body))
    monkeypatch.setattr(module,'run_walltime',original)
    expected='00:10:00' if historical else '24:00:00'
    with service.state('demo',value['preparation_id']) as (_,snapshot):
        assert snapshot.value['request']['run']['resources']['max_time']==expected
    assert service.accept('demo',ExperimentPreparationRequest.model_validate(body))==(value,None)
    result=service.finish(pending)
    assert result['status']=='READY',result
    assert result['run']['resources']['max_time']==expected
    changed=deepcopy(body);changed['run']['resources']={'max_time':'00:10:00' if not historical else '24:00:00'}
    with pytest.raises(ApplicationError) as error:service.accept('demo',ExperimentPreparationRequest.model_validate(changed))
    assert error.value.code=='PREPARATION_INTENT_CONFLICT'


def test_cpu_script_controller_requires_ready_cache_and_launch_cannot_redownload(cpu_download):
    from ml_exp_server.container_controller import Controller
    from ml_exp_server.results.artifact_store import ArtifactStore
    control = cpu_download
    created = control.client.post("/api/projects/demo/runs", json=control.body["run"])
    assert created.status_code == 200, created.text
    path, _ = frozen_campaign(control.client, created.json())
    campaign = yaml.safe_load(path.read_text())
    campaign["local_root"] = str(control.service.runtime.config.project_run_root_path("demo"))
    ctl = Controller(campaign, control.body["run"]["run_id"], "attempt-001")
    with pytest.raises(ApplicationError) as error:
        ctl.prepare_gateway_data(require_cached=True)
    assert error.value.code == "DATA_SCRIPT_NOT_READY" and not control.creates
    _, pending = accept_cpu(control)
    assert control.service.finish(pending)["status"] == "READY"
    ctl.prepare_gateway_data(require_cached=False)
    ctl.prepare_gateway_data(require_cached=True)
    assert len(control.creates) == 1
    ctl.prepare()
    ctl.dispatch_command(ctl.store.load_attempt("attempt-001"))
    objects = ArtifactStore(control.config, control.service.runtime.config.project_registry_root_path())
    with objects.record("demo", ctl.run["run_id"], "attempt-001") as (_, record):
        environment = record["launch_manifest"]["environment"]
        assert environment["ML_EXPD_DATA_PREPARATION_CACHED"] == "1"
        assert json.loads(environment["ML_EXPD_DATA_PREPARATION"]) == ctl.run["data_preparation"]


@pytest.mark.parametrize("mode", ["publish", "cached", "uncertain"])
def test_slurm_desktop_data_preparation_records_receipt_and_never_replays_uncertain_copy(remote_client, service, monkeypatch, mode):
    from ml_exp_server.backends.slurm import SlurmDataStager
    from tests.test_artifact_store import archive
    from tests.test_desktop_data_upload import historical_asset
    body = archive({"tokens.bin": b"training tokens"})
    asset = historical_asset(remote_client, body)
    calls = []
    real_receipt = SlurmDataStager.receipt
    monkeypatch.setattr(SlurmDataStager, "check", lambda *a: {"status": "READY" if mode == "cached" else "NOT_FOUND"})
    def publish(self, *args):
        calls.append("copy")
        if mode == "uncertain":
            raise ApplicationError("inspect original copy", code="SLURM_DATA_PREPARATION_FAILED")
        return real_receipt(self, *args)
    monkeypatch.setattr(SlurmDataStager, "publish", publish)
    request = spec(remote_client, run={"run_id": "slurm-data", "executor": "gpu",
                                     "inputs": [{"asset_id": asset["asset_id"], "mount_path": "/inputs/data"}]})
    _, pending = service.accept("demo", ExperimentPreparationRequest.model_validate(request))
    value = service.finish(pending)
    if mode == "uncertain":
        assert value["error_code"] == "SLURM_DATA_PREPARATION_FAILED"
        _, continued = service.continue_preparation(*pending)
        again = service.finish(continued)
        assert again["error_code"] == "SLURM_DATA_NOT_READY" and calls == ["copy"]
    else:
        assert value["status"] == "READY", value
        assert value["data_receipts"][asset["asset_id"]]["transport"] == "ssh_stream_shared_storage"
        assert calls == ([] if mode == "cached" else ["copy"])
        _, frozen = frozen_campaign(remote_client, value["run"])
        assert frozen["inputs"][0]["require_cached"] is True


def object_input(control):
    import hashlib
    from tests.test_artifact_store import archive
    content = archive({"tokens.bin": b"input tokens"})
    from tests.test_sensecore_data_workflow import put_asset
    response = put_asset(control.client, content)
    return response.json()


def test_cpu_script_with_worker_delivered_inputs_is_rejected_before_build(cpu_download, monkeypatch):
    from ml_exp_server.data.data_assets import AssetStore
    control = cpu_download
    control.body["run"]["inputs"] = [{"asset_id": "asset." + "a" * 64, "mount_path": "/inputs/data"}]
    control.body["runtime"] = {"source_id": "source." + "b" * 64, "entrypoint": ["python3", "train.py"]}
    control.body["run"].pop("runtime_id")
    monkeypatch.setattr(control.service.containers, "prepare", lambda *a: pytest.fail("unsupported transport must not build"))
    reads = []
    monkeypatch.setattr(AssetStore, "read", lambda *a: reads.append(a) or {"remote_storage": None})
    with pytest.raises(ApplicationError) as error:
        accept_cpu(control)
    assert error.value.code == "DATA_SCRIPT_INPUT_TRANSPORT_UNSUPPORTED"
    assert len(reads) == 1 and not control.creates


@pytest.mark.parametrize("entry", ["preparation", "create"])
def test_cpu_script_rejects_historical_object_asset_even_under_desktop_policy(cpu_download, monkeypatch, entry):
    control = cpu_download
    asset = object_input(control)
    config = json.loads(control.config.read_text())
    config.update(data_upload_storage="desktop-builder", data_upload_socket="desktop.sock")
    control.config.write_text(json.dumps(config))
    control.body["run"]["inputs"] = [{"asset_id": asset["asset_id"], "mount_path": "/inputs/data"}]
    monkeypatch.setattr(control.service.containers, "prepare", lambda *a: pytest.fail("unsupported historical asset must not build"))
    if entry == "preparation":
        control.body["runtime"] = {"source_id": "source." + "b" * 64, "entrypoint": ["python3", "train.py"]}
        control.body["run"].pop("runtime_id")
        with pytest.raises(ApplicationError) as error:
            accept_cpu(control)
        assert error.value.code == "DATA_SCRIPT_INPUT_TRANSPORT_UNSUPPORTED"
    else:
        response = control.client.post("/api/projects/demo/runs", json=control.body["run"])
        assert response.headers["X-ML-Expd-Error-Code"] == "DATA_SCRIPT_INPUT_TRANSPORT_UNSUPPORTED"
        assert not (Path(control.service.runtime.project("demo").base_dir) / "experiments/campaigns/run-cpu-download.yaml").exists()
    assert not control.creates


def test_cpu_script_accepts_desktop_asset_and_freezes_only_cached_input(cpu_download, remote_client, stage, monkeypatch):
    from ml_exp_server.data.data_delivery import DataDeliveryService
    from tests.test_artifact_store import archive
    from tests.test_desktop_data_upload import historical_asset
    control = cpu_download
    stage.uploads.capacity.headroom = 0  # This tiny transfer verifies routing, not disk admission.
    content = archive({"tokens.bin": b"desktop input"})
    asset = historical_asset(remote_client, content)
    assert asset["remote_storage"] == "desktop-builder"
    control.body["run"]["inputs"] = [{"asset_id": asset["asset_id"], "mount_path": "/inputs/data"}]
    value, pending = accept_cpu(control)
    assert value["status"] == "EXECUTING" and pending is not None
    checked = []
    def ready(self, project, asset_id, executor):
        checked.append((project, asset_id, executor))
        return {"status": "READY", "asset_id": asset_id, "data_path": "/data/lab/demo/data-assets/" + asset_id}
    monkeypatch.setattr(DataDeliveryService, "ready_for", ready)
    response = remote_client.post("/api/projects/demo/runs", json=control.body["run"])
    assert response.status_code == 200, response.text
    _, frozen = frozen_campaign(remote_client, response.json())
    assert frozen["inputs"][0]["require_cached"] is True
    assert frozen["data_preparation"]["inputs"][0]["require_cached"] is True
    assert checked == [("demo", asset["asset_id"], "cloud")] and not control.creates


@pytest.mark.parametrize("variant", ["no-script", "non-cpu"])
def test_object_assets_remain_allowed_without_cpu_download_script(cpu_download, variant):
    control = cpu_download
    asset = object_input(control)
    control.body["run"]["inputs"] = [{"asset_id": asset["asset_id"], "mount_path": "/inputs/data"}]
    control.body["requirements"] = {}
    if variant == "no-script":
        control.body["run"].pop("data_preparation")
    else:
        control.body["run"]["executor"] = "gpu"
    assert accept_cpu(control)[0]["status"] == "EXECUTING"
    response = control.client.post("/api/projects/demo/runs", json=control.body["run"])
    assert response.status_code == 200, response.text
    assert not control.creates


def test_legacy_script_with_object_inputs_keeps_inside_job_transport_on_repeat_create(cpu_download):
    control = cpu_download
    asset = object_input(control)
    config = json.loads(control.config.read_text())
    cpu = config.pop("data_delivery")
    control.config.write_text(json.dumps(config))
    control.body["run"]["inputs"] = [{"asset_id": asset["asset_id"], "mount_path": "/inputs/data"}]
    first = control.client.post("/api/projects/demo/runs", json=control.body["run"])
    assert first.status_code == 200, first.text
    path, frozen = frozen_campaign(control.client, first.json())
    assert "data_preparation_mode" not in frozen
    original = path.read_bytes()
    config.update(data_delivery=cpu, data_upload_storage="desktop-builder", data_upload_socket="desktop.sock")
    control.config.write_text(json.dumps(config))
    again = control.client.post("/api/projects/demo/runs", json=control.body["run"])
    assert again.status_code == 200, again.text
    assert again.json()["data_preparation_execution"]["phase"] == "inside_job"
    assert path.read_bytes() == original and not control.creates


def test_historical_before_job_run_keeps_stdout_receipt_mode_on_repeat_create(cpu_download):
    control = cpu_download
    request = control.body["run"]
    first = control.client.post("/api/projects/demo/runs", json=request)
    assert first.status_code == 200, first.text
    path, frozen = frozen_campaign(control.client, first.json())
    document = yaml.safe_load(path.read_text())
    document["runs"][0].pop("data_preparation_receipt_mode")
    document["runs"][0].pop("data_preparation_receipt_worker_sha256")
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    original = path.read_bytes()
    again = control.client.post("/api/projects/demo/runs", json=request)
    assert again.status_code == 200, again.text
    assert path.read_bytes() == original
    _, preserved = frozen_campaign(control.client, again.json())
    assert preserved["data_preparation_mode"] == "before_job"
    assert "data_preparation_receipt_mode" not in preserved
    assert "data_preparation_receipt_worker_sha256" not in preserved
    assert not control.creates


def test_existing_callback_worker_pin_survives_server_wrapper_change(cpu_download, monkeypatch):
    control = cpu_download
    request = control.body["run"]
    first = control.client.post("/api/projects/demo/runs", json=request)
    assert first.status_code == 200, first.text
    path, frozen = frozen_campaign(control.client, first.json())
    original = path.read_bytes()
    monkeypatch.setattr("ml_exp_server.backends.sensecore_preparation.script_receipt_worker_sha256", lambda: "0" * 64)
    again = control.client.post("/api/projects/demo/runs", json=request)
    assert again.status_code == 200, again.text
    assert path.read_bytes() == original
    assert frozen_campaign(control.client, again.json())[1]["data_preparation_receipt_worker_sha256"] == frozen["data_preparation_receipt_worker_sha256"]
    assert not control.creates
