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
    def call(self, path, **kwargs):
        self.calls.append((path, kwargs))
        if path == "/api/executors": return {"executors": [{"id": "gpu"}]}
        if "source-imports" in path: return {"source_id": "source." + "a" * 64}
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


HEALTH = {"capabilities": ["dockerfile-only.v1", "multipart-upload.v1"]}


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
    assert len([p for p, _ in api.calls if p.endswith("/runtimes/prepare")]) == 1
    assert len([p for p, _ in api.calls if p.endswith("/runs")]) == 1
    assert len([p for p, _ in api.calls if p.endswith("/execute") and "submissions" in p]) == 1
    experiment(api, HEALTH, configuration, state, resume=True, execute=True, out=state.with_name("results"))
    assert len(downloads) == 1


@pytest.mark.parametrize("boundary", ["build", "submit"])
def test_lost_effectful_response_is_not_replayed(configuration, boundary):
    api = API(); api.uncertain = boundary; state = configuration.with_name("state.json")
    with pytest.raises(ClientError, match="lost"):
        experiment(api, HEALTH, configuration, state, execute=True)
    before = [p for p, _ in api.calls if p.endswith("/execute")]
    with pytest.raises(ClientError, match="uncertain"):
        experiment(api, HEALTH, configuration, state, execute=True, resume=True)
    assert [p for p, _ in api.calls if p.endswith("/execute")] == before


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
