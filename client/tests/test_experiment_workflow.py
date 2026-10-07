"""Single config workflow preserves identities through interruption."""
import json
from pathlib import Path

import pytest

from ml_exp_client.api import ClientError
from ml_exp_client.workflow import experiment, read_config, validate_dockerfile, wait_resource


@pytest.fixture
def configuration(tmp_path):
    source = tmp_path / "source"; source.mkdir()
    (source / "Dockerfile").write_text("FROM registry.example/python@sha256:" + "a" * 64 + "\n")
    (source / "train.py").write_text("print('training')\n")
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps({"project": "demo", "run_id": "trial", "source": "source", "executor": "gpu", "max_gpu_hours": 0.2}))
    return path


class API:
    def __init__(self):
        self.calls = []; self.built = False; self.authorized = False; self.submitted = False; self.uncertain = None
        self.preparation = None; self.builds = 0
    def call(self, path, **kwargs):
        self.calls.append((path, kwargs))
        if path == "/api/executors": return {"executors": [{"id": "gpu"}]}
        if path == "/api/executors/match": return {"executor": "gpu", "live_availability_checked": False}
        if "source-imports" in path: return {"source_id": "source." + "a" * 64}
        if "experiment-preparations" in path:
            if self.preparation is None:
                self.built = True; self.builds += 1
                self.preparation = {"preparation_id": "preparation-" + "d" * 32, "status": "READY", "runtime_id": "runtime." + "b" * 64,
                    "run": {"run_id": "trial"}, "matching": {"executor": "gpu"},
                    "submission": {"submission_id": "action-" + "c" * 16, "status": "PREPARED", "ready": True,
                                   "confirmation": "EXECUTE test", "attempt_id": "attempt-001"}}
                if self.uncertain == "prepare":
                    self.uncertain = None
                    raise ClientError("lost preparation response")
            return self.preparation
        if "/runtimes" in path:
            if path.endswith("/execute"):
                if self.uncertain == "build": raise ClientError("lost build response")
                self.built = True; return {}
            return {"project": "demo", "runtime_id": "runtime." + "b" * 64, "status": "READY" if self.built else "PREPARED", "confirmation": "BUILD test"}
        if path.endswith("/runs"): return {"run_id": "trial"}
        if "submissions" in path:
            if path.endswith("/progress"): return {"phase": "RUNNING", "queue": {"reason": None}}
            if path.endswith("/authorize"): self.authorized = True
            if path.endswith("/execute"):
                if self.uncertain == "submit": raise ClientError("lost submit response")
                self.submitted = True
            return {"submission_id": "action-" + "c" * 16, "status": "VERIFIED" if self.submitted else "AUTHORIZED" if self.authorized else "PREPARED",
                    "ready": True, "confirmation": "EXECUTE test", "attempt_id": "attempt-001"}
        if path.endswith("/attempts"): return {"current_attempt_id": "attempt-001"}
        if path.startswith("/api/runs/"): return {"scheduler_state": "SUCCEEDED", "run_id": "trial"}
        raise AssertionError(path)


HEALTH = {"capabilities": ["dockerfile-only.v1", "multipart-upload.v1", "server-experiment-preparation.v1"]}


def test_script_data_preparation_is_frozen_without_asset_upload(configuration):
    source = configuration.parent / "source"
    (source / "download.py").write_text("print('download')")
    value = json.loads(configuration.read_text())
    value["data_preparation"] = {"script": "download.py", "arguments": ["--version", "1"]}
    configuration.write_text(json.dumps(value))
    api = API()
    with pytest.raises(ClientError, match="data-preparation.v1"):
        experiment(api, HEALTH, configuration, configuration.with_name("state.json"))
    assert api.calls == []
    state = experiment(api, {"capabilities": [*HEALTH["capabilities"], "data-preparation.v1"]}, configuration, configuration.with_name("state.json"))
    request = next(kwargs["data"]["run"] for path, kwargs in api.calls if path.endswith("/experiment-preparations"))
    assert request["data_preparation"] == value["data_preparation"]
    assert state["input_bindings"] == [] and not any("asset-upload" in path for path, _ in api.calls)


@pytest.mark.parametrize("spec", [None, [], {"script": "../download.py"}, {"script": "missing.py"},
    {"script": "train.py", "arguments": [""]}, {"script": "train.py", "interpreter": "bash"},
    {"script": "train.py", "interpreter": []},
    {"script": "train.py", "timeout_seconds": 0}, {"script": "train.py", "expected_content_sha256": "invalid"},
    {"script": "train.py", "unrecognized": True}])
def test_bad_script_config_fails_before_upload(configuration, spec):
    value = json.loads(configuration.read_text()); value["data_preparation"] = spec
    configuration.write_text(json.dumps(value))
    if spec is None:
        read_config(configuration)
    else:
        with pytest.raises(ClientError): read_config(configuration)


def test_prepare_then_execute_resume_reuses_runtime_run_and_exact_submission(configuration, monkeypatch):
    api = API(); state = configuration.with_name("state.json")
    prepared = experiment(api, HEALTH, configuration, state)
    assert prepared["submission"]["status"] == "PREPARED" and not api.submitted
    downloads = []
    def download(client, project, run, attempt, out):
        downloads.append((project, run, attempt)); return {"verified": True}
    monkeypatch.setattr("ml_exp_client.workflow.download", download)
    completed = experiment(api, HEALTH, configuration, state, resume=True, execute=True, out=state.with_name("results"))
    assert completed["result"]["scheduler_state"] == "SUCCEEDED" and completed["download"]["verified"]
    assert downloads == [("demo", "trial", "attempt-001")]
    assert len([p for p, _ in api.calls if p.endswith("/experiment-preparations")]) == 1
    assert not any(p.endswith(("/runtimes/prepare", "/runs")) for p, _ in api.calls)
    assert api.builds == 1
    assert len([p for p, _ in api.calls if p.endswith("/execute") and "submissions" in p]) == 1
    experiment(api, HEALTH, configuration, state, resume=True, execute=True, out=state.with_name("results"))
    assert len(downloads) == 1


@pytest.mark.parametrize("boundary", ["prepare", "submit"])
def test_lost_effectful_response_is_not_replayed(configuration, boundary):
    api = API(); api.uncertain = boundary; state = configuration.with_name("state.json")
    with pytest.raises(ClientError, match="lost"):
        experiment(api, HEALTH, configuration, state, execute=True)
    before = [p for p, _ in api.calls if p.endswith("/execute")]
    if boundary == "submit":
        with pytest.raises(ClientError, match="uncertain"):
            experiment(api, HEALTH, configuration, state, execute=True, resume=True)
        assert [p for p, _ in api.calls if p.endswith("/execute")] == before
    else:
        assert experiment(api, HEALTH, configuration, state, execute=True, resume=True)["result"]["scheduler_state"] == "SUCCEEDED"
        assert api.builds == 1


def test_resume_rejects_config_source_and_state_inside_source(configuration):
    api = API(); state = configuration.with_name("state.json")
    experiment(api, HEALTH, configuration, state)
    with pytest.raises(ClientError, match="state file exists"): experiment(api, HEALTH, configuration, state)
    with pytest.raises(ClientError, match="outside"): experiment(api, HEALTH, configuration, configuration.parent / "source/state.json")
    code = configuration.parent / "source/train.py"; code.write_text("changed source")
    with pytest.raises(ClientError, match="source changed"): experiment(api, HEALTH, configuration, state, resume=True)
    value = json.loads(configuration.read_text()); value["run_id"] = "other"; configuration.write_text(json.dumps(value))
    with pytest.raises(ClientError, match="configuration changed"): experiment(api, HEALTH, configuration, state, resume=True)


@pytest.mark.parametrize("changes", [
    {"unknown": 1}, {"source": None}, {"max_gpu_hours": float("nan")}, {"project": "../escape"},
    {"entrypoint": []}, {"arguments": "shell"}, {"env": {"API_TOKEN": "secret"}}, {"workdir": "/elsewhere"},
    {"resources": {"gpus": 0}}, {"resources": {"max_time": "forever"}}, {"resources": {"unknown": 1}},
    {"checkpoint_upload": {"interval_seconds": 1}}, {"inputs": "data"},
    {"inputs": [{"directory": "source", "mount_path": "/inputs/data"}]},
    {"inputs": [{"asset_id": "invalid", "mount_path": "/inputs/data"}]},
    {"inputs": [{"asset_id": "asset." + "a" * 64, "mount_path": "/invalid"}]},
])
def test_invalid_configuration_is_rejected_locally(configuration, changes):
    value = json.loads(configuration.read_text()); value.update(changes); configuration.write_text(json.dumps(value))
    with pytest.raises(ClientError): read_config(configuration)


def test_dockerfile_check_requires_file_and_pins_before_upload(tmp_path):
    with pytest.raises(ClientError): validate_dockerfile(tmp_path, "Dockerfile")
    file = tmp_path / "Dockerfile"
    for text in ("RUN echo no-base", "FROM python:latest"):
        file.write_text(text)
        with pytest.raises(ClientError): validate_dockerfile(tmp_path, "Dockerfile")
    file.write_text("FROM registry.example/python@sha256:" + "a" * 64 + " AS base\nFROM 0 AS second\nFROM second\n")
    validate_dockerfile(tmp_path, "Dockerfile")


def test_pending_progress_timeout_does_not_cancel_or_reexecute(monkeypatch):
    calls = []
    class Waiting:
        def call(self, path):
            calls.append(path)
            return {"status": "EXECUTING", "phase": "STAGING"}
    with pytest.raises(ClientError, match="never resubmit"):
        wait_resource(Waiting(), "/api/submissions/saved", 0)
    assert calls == ["/api/submissions/saved", "/api/submissions/saved/progress"]


@pytest.mark.parametrize("extra", [
    {"metrics_schema": []}, {"metrics_schema": {"schema_version": 2}},
    {"metrics_schema": {"unknown": True}}, {"metrics_schema": {"definitions": []}},
    {"metrics_schema": {"definitions": {"loss": {"unit": ""}}}},
    {"metrics_schema": {"definitions": {"loss": {"unit": "nats/token", "required": "yes"}}}},
    {"metrics_schema": {"definitions": {"bad\nname": {"unit": "nats/token"}}}},
    {"evaluation": []}, {"evaluation": {"value": float("nan")}},
    {"evaluation": {"value": "x" * 16384}},
])
def test_metric_and_evaluation_errors_fail_before_source_upload(configuration, extra):
    value = json.loads(configuration.read_text()); value.update(extra)
    configuration.write_text(json.dumps(value))
    api = API()
    with pytest.raises(ClientError):experiment(api, HEALTH, configuration, configuration.with_name("state.json"))
    assert api.calls == []


def test_one_config_transports_project_schema_and_scoring_protocol(configuration):
    value = json.loads(configuration.read_text())
    value.update(metrics_schema={"definitions": {"validation_loss": {"unit": "nats/token", "required": True}}},
                 evaluation={"D_FT": 0, "context_tokens": 1024})
    configuration.write_text(json.dumps(value))
    api = API(); experiment(api, HEALTH, configuration, configuration.with_name("state.json"))
    definition = next(kwargs["data"]["run"] for endpoint, kwargs in api.calls if endpoint.endswith("/experiment-preparations"))
    assert definition["metrics_schema"] == value["metrics_schema"]
    assert definition["evaluation"] == value["evaluation"]


def test_matching_request_precedes_source_upload_and_transmits_requirements(configuration):
    value = json.loads(configuration.read_text())
    value.pop("executor")
    value.update(executor_selector={"backend": "slurm", "candidates": ["gpu"]}, requirements={"exit_code": True})
    configuration.write_text(json.dumps(value))
    api = API()
    prepared = experiment(api, HEALTH, configuration, configuration.with_name("state.json"))
    assert api.calls[0][0] == "/api/executors/match"
    body = api.calls[0][1]["data"]
    assert body["executor_selector"] == value["executor_selector"] and body["requirements"] == value["requirements"]
    assert prepared["matching"]["executor"] == "gpu"


@pytest.mark.parametrize("extra", [
    {"executor_selector": {"backend": "sensecore"}},
    {"executor": None}, {"executor": []},
    {"executor": None, "executor_selector": {}},
    {"executor": None, "executor_selector": {"backend": []}},
    {"executor": None, "executor_selector": {"backend": "other"}},
    {"executor": None, "executor_selector": {"candidates": "gpu"}},
    {"executor": None, "executor_selector": {"candidates": ["../gpu"]}},
    {"requirements": {"exit_code": "yes"}}, {"requirements": {"unknown": True}},
    {"requirements": {"platform": []}},
])
def test_invalid_matching_config_fails_before_any_api(configuration, extra):
    value = json.loads(configuration.read_text()); value.update(extra)
    configuration.write_text(json.dumps(value))
    with pytest.raises(ClientError): read_config(configuration)


def test_server_preparation_requires_capability_and_explicit_continue(configuration):
    api = API()
    with pytest.raises(ClientError, match="server-experiment-preparation.v1"):
        experiment(api, {"capabilities": ["dockerfile-only.v1"]}, configuration, configuration.with_name("state.json"))
    with pytest.raises(ClientError, match="requires --resume"):
        experiment(api, HEALTH, configuration, configuration.with_name("state.json"), continue_preparation=True)
    assert api.calls == []


def test_uncertain_preparation_observes_until_explicit_continue(configuration):
    class Waiting(API):
        def call(self, path, **kwargs):
            result = super().call(path, **kwargs)
            if "experiment-preparations" in path:
                if path.endswith("/continue"):
                    self.status = "READY"
                    self.preparation["status"] = "READY"
                else:
                    self.preparation["status"] = getattr(self, "status", "RECONCILE_REQUIRED")
                return self.preparation
            return result
    api = Waiting(); state = configuration.with_name("state.json")
    with pytest.raises(ClientError, match="dependency inspection"):
        experiment(api, HEALTH, configuration, state)
    with pytest.raises(ClientError, match="dependency inspection"):
        experiment(api, HEALTH, configuration, state, resume=True)
    assert not any(path.endswith("/continue") for path, _ in api.calls)
    assert experiment(api, HEALTH, configuration, state, resume=True, continue_preparation=True)["submission"]["status"] == "PREPARED"
    assert api.builds == 1
