"""Single config workflow preserves identities through interruption."""
import json
import hashlib
import io
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
    def download(client, project, run, attempt, out, *, keep_server_copy=False):
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
                 evaluation={"D_FT": 0, "context_tokens": 1024}, required_outputs=["weights/*.pt", "summary.json"])
    configuration.write_text(json.dumps(value))
    api = API(); experiment(api, HEALTH, configuration, configuration.with_name("state.json"))
    definition = next(kwargs["data"]["run"] for endpoint, kwargs in api.calls if endpoint.endswith("/experiment-preparations"))
    assert definition["metrics_schema"] == value["metrics_schema"]
    assert definition["evaluation"] == value["evaluation"]
    assert definition["required_outputs"] == value["required_outputs"]


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


COLLECTION_HEALTH = {"capabilities": [*HEALTH["capabilities"], "result-collection.v1"]}


def collection(status="AVAILABLE", **changes):
    return {"project": "demo", "run_id": "trial", "attempt_id": "attempt-001", "status": status,
            "verification": "SHA256_VERIFIED" if status == "AVAILABLE" else "PENDING",
            "download_available": status == "AVAILABLE", **changes}


class CollectionAPI(API):
    def __init__(self, values):
        super().__init__()
        self.collection_values = list(values)
        self.collection_reads = 0
    def call(self, path, **kwargs):
        if path.endswith("/collection"):
            assert path == "/api/runs/demo/trial/attempts/attempt-001/collection" and not kwargs
            self.calls.append((path, kwargs))
            self.collection_reads += 1
            return self.collection_values.pop(0) if len(self.collection_values) > 1 else self.collection_values[0]
        return super().call(path, **kwargs)


def test_download_waits_for_original_attempt_collection_and_verification(configuration, monkeypatch):
    values = [collection("NOT_COLLECTED"), collection("QUEUED"), collection("COLLECTING"),
              collection(verification="PENDING"), collection(download_available=False), collection()]
    api = CollectionAPI(values)
    sleeps, downloads = [], []
    monkeypatch.setattr("ml_exp_client.workflow.time.sleep", sleeps.append)
    def fetch(client, project, run, attempt, out, **kwargs):
        assert api.collection_reads == len(values)
        downloads.append((project, run, attempt))
        return {"verified": True}
    monkeypatch.setattr("ml_exp_client.workflow.download", fetch)
    state = experiment(api, COLLECTION_HEALTH, configuration, configuration.with_name("state.json"),
                       execute=True, out=configuration.with_name("results"))
    assert state["collection"] == collection() and state["download"]["verified"]
    assert downloads == [("demo", "trial", "attempt-001")] and sleeps == [5] * 5
    assert all(not kwargs for path, kwargs in api.calls if path.endswith("/collection"))


@pytest.mark.parametrize("status", ["FAILED", "RECONCILE_REQUIRED", "RELEASED", "UNRECOGNIZED", None, {"invalid": True}])
def test_collection_terminal_or_unknown_state_never_requests_recovery_or_download(configuration, monkeypatch, status):
    api = CollectionAPI([collection(status)])
    monkeypatch.setattr("ml_exp_client.workflow.download", lambda *a, **kw: pytest.fail("unexpected download"))
    state = configuration.with_name("state.json")
    with pytest.raises(ClientError, match="RELEASED|FAILED|RECONCILE_REQUIRED|unknown collection") as caught:
        experiment(api, COLLECTION_HEALTH, configuration, state, execute=True, out=configuration.with_name("results"))
    assert json.loads(state.read_text())["collection"]["status"] == status
    assert api.collection_reads == 1 and all(not kwargs for path, kwargs in api.calls if path.endswith("/collection"))
    if status == "RELEASED":
        assert caught.value.status == 410 and caught.value.error_code == "ARTIFACT_RELEASED"


def test_collection_timeout_resumes_original_job_directory_and_keep_choice(configuration, monkeypatch):
    api = CollectionAPI([collection("NOT_COLLECTED")])
    state = configuration.with_name("state.json")
    destination = configuration.with_name("results")
    with pytest.raises(ClientError, match="publication wait timed out"):
        experiment(api, COLLECTION_HEALTH, configuration, state, execute=True, seconds=0,
                   out=destination, keep_server_copy=True)
    saved = json.loads(state.read_text())
    assert saved["result"]["scheduler_state"] == "SUCCEEDED" and "download" not in saved
    assert saved["download_directory"] == str(destination.resolve()) and saved["download_keep_server_copy"]
    api.collection_values = [collection()]
    downloads = []
    def fetch(client, project, run, attempt, out, *, keep_server_copy):
        downloads.append((attempt, out, keep_server_copy))
        return {"verified": True}
    monkeypatch.setattr("ml_exp_client.workflow.download", fetch)
    experiment(api, COLLECTION_HEALTH, configuration, state, resume=True, execute=True)
    assert downloads == [("attempt-001", destination, True)]
    assert sum(path.endswith("/execute") and "submissions" in path for path, _ in api.calls) == 1
    assert sum(path.endswith("/experiment-preparations") for path, _ in api.calls) == 1
    assert all(not kwargs for path, kwargs in api.calls if path.endswith("/collection"))


@pytest.mark.parametrize("field", ["project", "run_id", "attempt_id"])
def test_collection_rejects_different_identity_before_download(configuration, monkeypatch, field):
    api = CollectionAPI([collection(**{field: "different"})])
    monkeypatch.setattr("ml_exp_client.workflow.download", lambda *a, **kw: pytest.fail("unexpected download"))
    with pytest.raises(ClientError, match="collection identity differs"):
        experiment(api, COLLECTION_HEALTH, configuration, configuration.with_name("state.json"),
                   execute=True, out=configuration.with_name("results"))


class DownloadAPI(CollectionAPI):
    def __init__(self):
        super().__init__([collection()])
        from test_direct_download import archive
        self.body = archive({"weights.pt": b"weights"})
        self.ticket = {"transport": "s3-presigned-get", "url": "https://objects.example/x?private-signature",
                       "sha256": hashlib.sha256(self.body).hexdigest(), "bytes": len(self.body),
                       "files": [{"path": "weights.pt", "bytes": 7, "sha256": hashlib.sha256(b"weights").hexdigest()}],
                       "acknowledgement": {"contract": "artifact-retention.v1"}}
        self.downloads = 0
        self.fail_ack = True
    def open_object(self, url):
        self.downloads += 1
        return io.BytesIO(self.body)
    def call(self, path, **kwargs):
        if path.endswith("/artifacts/download"):
            self.calls.append((path, kwargs))
            return self.ticket
        if path.endswith("/artifacts/ack"):
            assert path == "/api/runs/demo/trial/attempts/attempt-001/artifacts/ack"
            self.calls.append((path, kwargs))
            if self.fail_ack:
                raise ClientError("lost ACK", retryable=True)
            return {"status": "SCHEDULED" if kwargs["data"]["release"] else "KEEP"}
        return super().call(path, **kwargs)


@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_resume_revalidates_pending_ack_without_redownload_or_collection(configuration, keep, legacy):
    api = DownloadAPI()
    state, destination = configuration.with_name("state.json"), configuration.with_name("results")
    first = experiment(api, COLLECTION_HEALTH, configuration, state, execute=True, out=destination, keep_server_copy=keep)
    assert first["download"]["acknowledgement"]["status"] == "PENDING"
    if legacy:
        saved = json.loads(state.read_text())
        saved.pop("download_directory"); saved.pop("download_keep_server_copy")
        state.write_text(json.dumps(saved))
    api.fail_ack = False
    resumed = experiment(api, COLLECTION_HEALTH, configuration, state, resume=True, out=destination if legacy else None)
    assert resumed["download"]["acknowledgement"]["status"] == ("KEEP" if keep else "SCHEDULED")
    assert api.downloads == 1 and api.collection_reads == 1
    proofs = [kwargs["data"] for path, kwargs in api.calls if path.endswith("/artifacts/ack")]
    assert len(proofs) == 2 and all(proof["release"] is not keep for proof in proofs)
    assert sum(path.endswith("/execute") and "submissions" in path for path, _ in api.calls) == 1


@pytest.mark.parametrize("failure", ["file", "identity", "proof", "missing-directory"])
def test_resume_ack_rejects_corrupt_or_different_local_proof(configuration, failure):
    api = DownloadAPI()
    state, destination = configuration.with_name("state.json"), configuration.with_name("results")
    experiment(api, COLLECTION_HEALTH, configuration, state, execute=True, out=destination)
    if failure == "file":
        (destination / "outputs/weights.pt").write_bytes(b"changed")
    elif failure == "missing-directory":
        saved = json.loads(state.read_text()); saved.pop("download_directory"); state.write_text(json.dumps(saved))
    else:
        report = json.loads((destination / "verification.json").read_text())
        report["attempt_id" if failure == "identity" else "archive_sha256"] = "different"
        (destination / "verification.json").write_text(json.dumps(report))
    calls = sum(path.endswith("/artifacts/ack") for path, _ in api.calls)
    with pytest.raises(ClientError, match="differs|PENDING"):
        experiment(api, COLLECTION_HEALTH, configuration, state, resume=True)
    assert sum(path.endswith("/artifacts/ack") for path, _ in api.calls) == calls
    assert api.downloads == 1 and api.collection_reads == 1
