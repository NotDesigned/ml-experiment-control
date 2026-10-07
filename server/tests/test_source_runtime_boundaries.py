"""Import budgets and runtime transactions exercised through real stores and HTTP."""

import hashlib
import io
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import pytest
import yaml

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.container_execution import ContainerExecutionService, DockerfileRuntimeSpec
from ml_exp_server.source_imports import SourceImportService, source_lock, unpack_source, seal_tree
from tests.test_container_api import prepare_runtime, archive, client, import_source, runtime


def prepared(client):
    source = import_source(client)
    return prepare_runtime(client, {"source_id": source["source_id"],
        "dockerfile": "Dockerfile", "entrypoint": ["python", "train.py"]}).json()


def update(service, record, **changes):
    with service.state("demo", record["runtime_id"]) as (store, snapshot):
        value = dict(snapshot.value)
        value.update(changes)
        return store.commit(value, expected_revision=snapshot.revision, event={"event": "test-simulated-concurrent-transition"}).value


def test_disabled_source_import_and_invalid_direct_digest(client, tmp_path):
    client.app.state.runtime.config.action_runtime.allow_source_imports = False
    data = archive()
    response = client.post("/api/source-imports/archive", params={"project": "demo", "sha256": hashlib.sha256(data).hexdigest()}, content=data)
    assert response.status_code == 409
    service = SourceImportService(client.app.state.runtime)
    with pytest.raises(ApplicationError, match="lowercase hex"):
        service.archive("demo", io.BytesIO(data), "invalid")
    with pytest.raises(ApplicationError, match="invalid project identity"):
        with source_lock(tmp_path, "../outside"):
            pytest.fail("escaped import lock")


@pytest.mark.parametrize("case", ["empty", "duplicate", "file-count", "bytes", "pax-budget", "missing-member", "truncated-member"])
def test_archive_rejects_invalid_and_overbudget_streams(tmp_path, monkeypatch, case):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as output:
        directory = tarfile.TarInfo("./")
        directory.type = tarfile.DIRTYPE
        output.addfile(directory)
        if case != "empty":
            for name in (["train.py", "train.py"] if case == "duplicate" else ["train.py", "nested/other.py"] if case == "file-count" else ["nested/train.py"]):
                member = tarfile.TarInfo(name)
                member.size = 10
                member.mode = 0o755
                if case == "pax-budget":
                    member.pax_headers = {"comment": "x" * (2 * 1024 * 1024)}
                output.addfile(member, io.BytesIO(b"0123456789"))
    if case in {"missing-member", "truncated-member"}:
        monkeypatch.setattr(tarfile.TarFile, "extractfile", lambda *args: None if case == "missing-member" else io.BytesIO(b""))
    policy = SimpleNamespace(max_source_bytes=5 if case == "bytes" else 100, max_source_files=1 if case in {"file-count", "pax-budget"} else 100)
    stream.seek(0)
    with pytest.raises(ValueError):
        unpack_source(stream, tmp_path, policy)


def test_archive_directories_executable_modes_and_sealing(tmp_path):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as output:
        for name in [".", "nested"]:
            item = tarfile.TarInfo(name)
            item.type = tarfile.DIRTYPE
            output.addfile(item)
        item = tarfile.TarInfo("nested/run.sh")
        item.size, item.mode = 4, 0o755
        output.addfile(item, io.BytesIO(b"true"))
    stream.seek(0)
    unpack_source(stream, tmp_path, SimpleNamespace(max_source_bytes=100, max_source_files=10))
    (tmp_path / "dangling").symlink_to(tmp_path / "missing")
    seal_tree(tmp_path)
    assert (tmp_path / "nested/run.sh").stat().st_mode & 0o777 == 0o500


@pytest.mark.parametrize("failure", [None, "fetch", "commit", "budget"])
def test_git_import_verifies_exact_commit_and_never_uses_host_credentials(client, monkeypatch, failure):
    calls = []
    data = archive()
    commit = "a" * 40
    def git(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["HOME"] == "/nonexistent" and kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
        if "fetch" in command and failure == "fetch":
            raise subprocess.CalledProcessError(1, command)
        if "rev-parse" in command:
            return SimpleNamespace(stdout=(("b" * 40 if failure == "commit" else commit) + "\n").encode())
        if "archive" in command:
            kwargs["stdout"].write(data)
        return SimpleNamespace(stdout=b"")
    monkeypatch.setattr("ml_exp_server.source_imports.subprocess.run", git)
    if failure == "budget":
        client.app.state.runtime.config.container_execution.max_archive_bytes = 10
    response = client.post("/api/source-imports/git", json={"project": "demo", "url": "https://github.com/example/project.git", "commit": commit})
    assert response.status_code == (409 if failure else 200), response.text
    if not failure:
        assert response.json()["observation"]["commit"] == commit
    assert calls and not list(client.app.state.runtime.config.project_registry_root_path().glob(".git-import-*"))


@pytest.mark.parametrize("field,value", [("entrypoint", ["bad\x00argument"]), ("workdir", "/outside"), ("packaging_revision", "unreviewed")])
def test_runtime_validation_rejects_unsafe_or_unreviewed_definition(client, field, value):
    source = import_source(client)
    spec = {"source_id": source["source_id"], "dockerfile": "Dockerfile", "entrypoint": ["python", "train.py"]}
    spec[field] = value
    assert client.post("/api/projects/demo/runtimes/prepare", json=spec).status_code == 422


@pytest.mark.parametrize("field,value", [("arguments", ["bad\x00argument"]), ("env", {"TOKEN": "test"}), ("env", {"OUTPUT_DIR": "/override"}), ("env", {"GOOD": "bad\x00value"}), ("outputs", ["../outside"]), ("outputs", [".hidden"])])
def test_run_validation_rejects_credential_reserved_and_escaping_fields(client, field, value):
    bundle = runtime(client)
    request = {"run_id": "r", "runtime_id": bundle["runtime_id"], "executor": "gpu"}
    request[field] = value
    assert client.post("/api/projects/demo/runs", json=request).status_code == 422


def test_runtime_policy_legacy_project_and_missing_identity(client):
    record = prepared(client)
    service = ContainerExecutionService(client.app.state.runtime)
    assert service.prepare("demo", DockerfileRuntimeSpec.model_validate(record["spec"])) == record
    with pytest.raises(ValueError):
        with service.state("../escape", record["runtime_id"]):
            pass
    assert client.get("/api/projects/demo/runtimes/runtime." + "0" * 64).status_code == 404
    project = client.app.state.runtime.project("demo")
    project.controller.capabilities["container_execution"] = False
    with pytest.raises(ApplicationError, match="own controller"):
        service.prepare("demo", DockerfileRuntimeSpec.model_validate(record["spec"]))
    project.controller.capabilities["container_execution"] = True
    client.app.state.runtime.config.action_runtime.allow_project_writes = False
    with pytest.raises(ApplicationError, match="writes are disabled"):
        service.prepare("demo", DockerfileRuntimeSpec.model_validate(record["spec"]))


def test_runtime_execute_confirmation_reconcile_and_concurrent_fencing(client, monkeypatch):
    record = prepared(client)
    service = ContainerExecutionService(client.app.state.runtime)
    endpoint = "/api/projects/demo/runtimes/" + record["runtime_id"]
    assert client.post(endpoint + "/execute", json={"confirmation": "wrong"}).status_code == 409
    with pytest.raises(ApplicationError, match="confirmation mismatch"):
        service.execute("demo", record["runtime_id"], "wrong")
    update(service, record, status="EXECUTING")
    assert client.post(endpoint + "/execute", json={"confirmation": record["confirmation"]}).status_code == 409
    with pytest.raises(ApplicationError, match="already executing"):
        service.execute("demo", record["runtime_id"], record["confirmation"])
    update(service, record, status="RECONCILE_REQUIRED")
    assert client.post(endpoint + "/reconcile", json={"confirmation": record["confirmation"]}).json()["status"] == "READY"
    assert service.execute("demo", record["runtime_id"], record["confirmation"])["status"] == "READY"
    assert client.post(endpoint + "/execute", json={"confirmation": record["confirmation"]}).status_code == 202
    update(service, record, status="PREPARED")
    original = __import__("ml_exp_server.container_execution", fromlist=["builder_request"]).builder_request
    def changed(socket, request):
        update(service, record, status="RECONCILE_REQUIRED", error="another owner reconciled")
        return original(socket, request)
    monkeypatch.setattr("ml_exp_server.container_execution.builder_request", changed)
    assert service.execute("demo", record["runtime_id"], record["confirmation"])["error"] == "another owner reconciled"


@pytest.mark.parametrize("failure", ["no-builder", "invalid-receipt"])
def test_failed_packaging_preserves_uncertainty(client, monkeypatch, failure):
    record = prepared(client)
    service = ContainerExecutionService(client.app.state.runtime)
    if failure == "no-builder":
        client.app.state.runtime.config.container_execution.builder_socket = None
    else:
        monkeypatch.setattr("ml_exp_server.container_execution.builder_request", lambda *args: {"project": "wrong"})
    result = service.execute("demo", record["runtime_id"], record["confirmation"])
    assert result["status"] == "RECONCILE_REQUIRED" and result["image"] is None


def test_executor_profiles_are_operator_owned_and_capacity_is_recorded(client):
    config = client.app.state.runtime.config.container_execution
    path = Path(config.profiles_file)
    profiles = yaml.safe_load(path.read_text())
    profiles["executors"]["cloud"]["capacity"] = {"gpus": 1, "cpus": 8, "memory_gb": 128}
    path.write_text(yaml.safe_dump(profiles))
    bundle = runtime(client)
    request = {"run_id": "cloud", "runtime_id": bundle["runtime_id"], "executor": "cloud", "env": {"SEED": "42"}}
    assert client.post("/api/projects/demo/runs", json={**request, "resources": {"gpus": 2}}).status_code == 409
    assert client.post("/api/projects/demo/runs", json={**request, "resources": {"memory_gb": 129}}).status_code == 409
    assert client.post("/api/projects/demo/runs", json=request).json()["resources"]["memory_gb"] == 128
    assert client.post("/api/projects/demo/runs", json=request).status_code == 200
    assert client.post("/api/projects/demo/runs", json={**request, "executor": "unknown"}).status_code == 404
    profiles["executors"]["cloud"]["backend"]["kind"] = "unsupported"
    path.write_text(yaml.safe_dump(profiles))
    assert client.post("/api/projects/demo/runs", json={**request, "run_id": "unsupported"}).status_code == 409
    project = client.app.state.runtime.project("demo")
    project.controller.capabilities["container_execution"] = False
    assert client.post("/api/projects/demo/runs", json={**request, "executor": "gpu"}).status_code == 409
    path.write_text("executors: []\n")
    assert client.get("/api/executors").status_code == 409
    path.write_text("[]\n")
    assert client.get("/api/executors").json() == {"executors": []}
    config.profiles_file = None
    assert client.get("/api/executors").json() == {"executors": []}


def test_unready_runtime_cannot_create_run(client):
    record = prepared(client)
    assert client.post("/api/projects/demo/runs", json={"run_id": "r", "runtime_id": record["runtime_id"], "executor": "gpu"}).status_code == 409


def pool_profile(client, **backend):
    path = Path(client.app.state.runtime.config.container_execution.profiles_file)
    profiles = yaml.safe_load(path.read_text())
    profiles['executors']['cloud']['backend'].update(backend)
    path.write_text(yaml.safe_dump(profiles))


def test_spot_pool_selection_is_frozen_once_and_definition_changes_conflict(client, monkeypatch):
    from unittest.mock import Mock
    from experiment_control.backends.sensecore_rest import SenseCoreREST
    bundle = runtime(client)
    pool_profile(client, pool_selection='highest_spot', allowed_clusters=['compute', 'other'])
    calls = []
    def select(backend, *, gpus):
        calls.append((dict(backend), gpus))
        return {**backend, 'aec2': 'other', 'pool_selection_evidence': {
            'configured_aec2': 'compute', 'selected': 'other', 'policy': 'highest_spot',
            'observed_at': 'fixed-time', 'candidates': [{'name': 'other', 'spot_devices': '11'}]}}
    monkeypatch.setattr(SenseCoreREST, 'from_environment', lambda: Mock(select_pool=select))
    body = {'run_id': 'selected', 'runtime_id': bundle['runtime_id'], 'executor': 'cloud'}
    first = client.post('/api/projects/demo/runs', json=body)
    assert first.status_code == 200, first.text
    assert first.json()['pool_selection']['selected'] == 'other'
    repeated = client.post('/api/projects/demo/runs', json=body)
    assert repeated.status_code == 200 and repeated.json() == first.json()
    assert len(calls) == 1 and calls[0][1] == 1
    root = Path(client.app.state.runtime.project('demo').base_dir)
    campaign = yaml.safe_load((root / 'experiments/campaigns/run-selected.yaml').read_text())
    assert campaign['runs'][0]['backend']['aec2'] == 'other'
    assert client.post('/api/projects/demo/runs', json={**body, 'arguments': ['changed']}).status_code == 409
    pool_profile(client, aec2='changed-base')
    assert client.post('/api/projects/demo/runs', json=body).status_code == 409
    assert len(calls) == 1


@pytest.mark.parametrize('case', ['unsupported', 'denied', 'debug'])
def test_selection_or_debug_guard_failure_does_not_freeze_or_submit_run(client, monkeypatch, case):
    from unittest.mock import Mock
    from experiment_control.backends.sensecore_rest import SenseCoreREST, RESTError
    bundle = runtime(client)
    pool_profile(client, **({'aec2': 'debug-cluster'} if case == 'debug' else
                           {'pool_selection': 'bad' if case == 'unsupported' else 'highest_spot'}))
    monkeypatch.setattr(SenseCoreREST, 'from_environment', lambda: Mock(select_pool=Mock(side_effect=RESTError('specs', status=403))))
    response = client.post('/api/projects/demo/runs', json={'run_id': 'blocked', 'runtime_id': bundle['runtime_id'], 'executor': 'cloud'})
    assert response.status_code == 409
    assert not (Path(client.app.state.runtime.project('demo').base_dir) / 'experiments/campaigns/run-blocked.yaml').exists()


def test_new_runs_default_to_spot_but_historical_fixed_runs_are_not_reselected(client, monkeypatch):
    from unittest.mock import Mock
    from experiment_control.backends.sensecore_rest import SenseCoreREST
    bundle = runtime(client)
    body = {'run_id': 'historical', 'runtime_id': bundle['runtime_id'], 'executor': 'cloud'}
    assert client.post('/api/projects/demo/runs', json=body).status_code == 200
    path = Path(client.app.state.runtime.config.container_execution.profiles_file)
    profiles = yaml.safe_load(path.read_text())
    profiles['executors']['cloud']['backend'].pop('pool_selection')
    path.write_text(yaml.safe_dump(profiles))
    root = Path(client.app.state.runtime.project('demo').base_dir)
    frozen = root / 'experiments/campaigns/run-historical.yaml'
    campaign = yaml.safe_load(frozen.read_text())
    campaign['runs'][0]['backend'].pop('pool_selection')
    frozen.write_text(yaml.safe_dump(campaign))
    def select(backend, *, gpus):
        return {**backend, 'aec2': 'other', 'pool_selection_evidence': {
            'configured_aec2': 'compute', 'policy': 'highest_spot', 'selected': 'other'}}
    fake = Mock(select_pool=Mock(side_effect=select))
    monkeypatch.setattr(SenseCoreREST, 'from_environment', lambda: fake)
    assert client.post('/api/projects/demo/runs', json=body).status_code == 200
    fake.select_pool.assert_not_called()
    new = {**body, 'run_id': 'default-spot'}
    response = client.post('/api/projects/demo/runs', json=new)
    assert response.status_code == 200, response.text
    assert response.json()['pool_selection']['selected'] == 'other'
    assert client.post('/api/projects/demo/runs', json=new).status_code == 200
    fake.select_pool.assert_called_once()


@pytest.mark.parametrize('change', [None, {'policy': 'fixed'}, {'selected': 'unbound-pool'}, {'pool_selection_evidence': 'invalid'}])
def test_frozen_pool_evidence_cannot_be_silently_replaced_or_reselected(client, monkeypatch, change):
    from unittest.mock import Mock
    from experiment_control.backends.sensecore_rest import SenseCoreREST
    bundle = runtime(client)
    pool_profile(client, pool_selection='highest_spot')
    def select(backend, *, gpus):
        return {**backend, 'aec2': 'other', 'pool_selection_evidence': {
            'configured_aec2': 'compute', 'policy': 'highest_spot', 'selected': 'other'}}
    fake = Mock(select_pool=Mock(side_effect=select))
    monkeypatch.setattr(SenseCoreREST, 'from_environment', lambda: fake)
    body = {'run_id': 'bound-proof', 'runtime_id': bundle['runtime_id'], 'executor': 'cloud'}
    assert client.post('/api/projects/demo/runs', json=body).status_code == 200
    root = Path(client.app.state.runtime.project('demo').base_dir)
    path = root / 'experiments/campaigns/run-bound-proof.yaml'
    campaign = yaml.safe_load(path.read_text())
    backend = campaign['runs'][0]['backend']
    if change is None:
        backend.pop('pool_selection_evidence')
    elif 'pool_selection_evidence' in change:
        backend.update(change)
    else:
        backend['pool_selection_evidence'].update(change)
    path.write_text(yaml.safe_dump(campaign))
    before = path.read_bytes()
    assert client.post('/api/projects/demo/runs', json=body).status_code == 409
    assert path.read_bytes() == before
    fake.select_pool.assert_called_once()
