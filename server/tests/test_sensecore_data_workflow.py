"""HTTP data delivery, frozen Dockerfile receipts and durable checkpoint reuse."""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ml_exp_server.artifact_store import ArtifactStore
from ml_exp_server.container_controller import Controller
from ml_exp_server.container_execution import DockerfileRuntimeSpec
from ml_exp_server.data_assets import AssetStore
from ml_exp_server.dockerfile_build import DOCKERFILE_RECIPE, inspect_dockerfile, managed_dockerfile, worker_digest
from ml_exp_server.image_builder import ImageBuilder, MANIFEST_TYPE, bundle_id
from ml_exp_server.source_imports import seal_tree
from ml_exp_server.source_revisions import _tree_digest
from ml_exp_server.schemas import RunIndexRow, AttemptSummary
from tests.test_container_api import archive, client, import_source, wait_runtime, runtime

from ml_exp_server.worker_contract import WORKER_CONTRACT, CAPABILITIES

BASE = "registry.example/base@sha256:" + "a" * 64


@pytest.fixture
def stored(client, tmp_path, monkeypatch):
    config = tmp_path / "objects.json"
    config.write_text(json.dumps({"endpoint": "http://127.0.0.1:3900", "bucket": "assets", "access_key": "test", "secret_key": "test",
                                  "max_archive_bytes": 2 * 1024 ** 2, "public_transfer_base": "https://api.example/api/artifact-transfers"}))
    client.app.state.runtime.config.container_execution.artifact_store_file = str(config)
    objects = {}
    class Body(io.BytesIO):
        def iter_chunks(self, chunk_size):
            while data := self.read(chunk_size):
                yield data
    class Objects:
        def upload_fileobj(self, stream, bucket, key, **kwargs):
            objects[key] = stream.read()
        def get_object(self, Bucket, Key):
            return {"Body": Body(objects[Key])}
    monkeypatch.setattr(ArtifactStore, "client", lambda self: Objects())
    import_source(client)
    return config, objects


def custom_runtime(client, monkeypatch, *, text=None, tamper=None, legacy=False):
    source = import_source(client, archive({"train.py": b"print('training')", "Dockerfile": (text or f"FROM {BASE}\nRUN echo configured\n").encode()}))
    def build(socket, payload):
        tree = Path(client.app.state.runtime.config.project_registry_root_path()) / "source-revisions/sources/demo" / source["source_id"] / "tree"
        inspection = inspect_dockerfile(tree, "Dockerfile")
        result = {"project": "demo", "source_id": source["source_id"], "base_image": BASE,
                  "image": "registry.example/runtime@sha256:" + "b" * 64,
                  "bundle_id": bundle_id("demo", source["source_id"], BASE, dockerfile_path="Dockerfile"),
                  "dockerfile": {k: v for k, v in inspection.items() if k != "text"},
                  "dockerfile_sha256": hashlib.sha256(managed_dockerfile(inspection, source["source_id"]).encode()).hexdigest(),
                  "worker_contract": WORKER_CONTRACT, "worker_sha256": worker_digest(), "capabilities": list(CAPABILITIES)}
        if tamper:
            result[tamper] = "different"
        return result
    monkeypatch.setattr("ml_exp_server.container_execution.builder_request", build)
    response = client.post("/api/projects/demo/runtimes/prepare", json={"source_id": source["source_id"], "dockerfile": "Dockerfile", "entrypoint": ["python3", "train.py"]})
    assert response.status_code == 200, response.text
    value = response.json()
    if legacy:
        from ml_exp_server.container_execution import ContainerExecutionService
        service = ContainerExecutionService(client.app.state.runtime)
        with service.state("demo", value["runtime_id"]) as (store, snapshot):
            old = dict(snapshot.value)
            old.pop("worker_contract")
            store.commit(old, expected_revision=snapshot.revision, event={"event": "legacy_fixture"})
    endpoint = "/api/projects/demo/runtimes/" + value["runtime_id"]
    assert client.post(endpoint + "/execute", json={"confirmation": value["confirmation"]}).status_code == 202
    return wait_runtime(client, endpoint)


def controller(client, bundle, **extra):
    response = client.post("/api/projects/demo/runs", json={"run_id": "trial", "runtime_id": bundle["runtime_id"], "executor": "cloud", **extra})
    assert response.status_code == 200, response.text
    root = Path(client.app.state.runtime.project("demo").base_dir)
    campaign = yaml.safe_load((root / "experiments/campaigns/run-trial.yaml").read_text())
    campaign["local_root"] = str(client.app.state.runtime.config.project_run_root_path("demo"))
    result = Controller(campaign, "trial", "attempt-001")
    result.prepare()
    client.app.state.runtime.index.upsert_run(RunIndexRow(project="demo", run_id="trial", run_dir=str(result.root),
        attempts=[AttemptSummary(attempt_id="attempt-001", state="RUNNING")]))
    return result


def put_asset(client, data=None, **params):
    data = archive({"tokens.bin": b"tokens for a separate data asset"}) if data is None else data
    return client.post("/api/assets/archive", params={"project": "demo", "sha256": hashlib.sha256(data).hexdigest(), **params}, content=data)


def test_client_can_upload_mount_publish_and_reuse_checkpoint(client, stored, monkeypatch):
    data = archive({"tokens.bin": b"tokens for a separate data asset"})
    asset = put_asset(client, data).json()
    assert asset["status"] == "READY" and asset["files"][0]["sha256"] == hashlib.sha256(b"tokens for a separate data asset").hexdigest()
    assert put_asset(client, data).json() == asset
    assert client.get("/api/projects/demo/assets").json()["assets"] == [asset]
    assert client.get("/api/projects/demo/assets/" + asset["asset_id"]).json() == asset
    bundle = custom_runtime(client, monkeypatch)
    assert bundle["status"] == "READY" and bundle["spec"]["packaging_revision"] == DOCKERFILE_RECIPE
    ctl = controller(client, bundle, inputs=[{"asset_id": asset["asset_id"], "mount_path": "/inputs/fineweb"}], checkpoint_upload={"interval_seconds": 5})
    command = ctl.dispatch_command(ctl.store.load_attempt("attempt-001"))
    assert ctl.command("attempt-001") == ["ml-exp-worker"]
    assert "--manifest-url" in command
    transfer = ArtifactStore(stored[0], client.app.state.runtime.config.project_registry_root_path())
    with transfer.record("demo", "trial", "attempt-001") as (_, record):
        token = record["token"]
        assert record["launch_manifest"]["environment"]["INPUTS_DIR"] == "/inputs"
        assert "ML_EXPD_INPUT_ASSETS" in record["launch_manifest"]["environment"]
    endpoint = "/api/asset-transfers/demo/trial/attempt-001/" + asset["asset_id"]
    assert client.get(endpoint, headers={"Authorization": "Bearer wrong"}).status_code == 401
    response = client.get(endpoint, headers={"Authorization": "Bearer " + token})
    assert response.status_code == 200 and hashlib.sha256(response.content).hexdigest() == asset["sha256"]
    assert client.get(endpoint.replace("demo/trial", "other/trial"), headers={"Authorization": "Bearer " + token}).status_code == 401
    assert client.get(endpoint[:-64] + "c" * 64, headers={"Authorization": "Bearer " + token}).status_code == 401
    snapshot = archive({"checkpoint.pt": b"durable training state"})
    endpoint = "/api/snapshot-transfers/demo/trial/attempt-001"
    response = client.put(endpoint, content=snapshot, headers={"Authorization": "Bearer " + token})
    assert response.status_code == 200, response.text
    checkpoint = response.json()
    assert checkpoint["provenance"] == {"kind": "checkpoint", "run_id": "trial", "attempt_id": "attempt-001"}
    assert client.put(endpoint, content=snapshot, headers={"Authorization": "Bearer " + token}).json() == checkpoint
    path = "/api/runs/demo/trial/attempts/attempt-001/snapshots"
    assert client.get(path).json()["snapshots"] == [checkpoint]
    definition = {"run_id": "resumed", "runtime_id": bundle["runtime_id"], "executor": "cloud", "inputs": [{"asset_id": checkpoint["asset_id"], "mount_path": "/inputs/resume"}]}
    assert client.post("/api/projects/demo/runs", json=definition).status_code == 200
    assert client.get("/api/projects/demo/assets/" + checkpoint["asset_id"] + "/archive").content == snapshot
    assert ctl.store.load_manifest()["assets"][-1]["identity"] == asset["asset_id"]


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("tamper", [None, "dockerfile", "dockerfile_sha256", "worker_sha256", "bundle_id", "capabilities"])
def test_tampered_build_receipt_cannot_become_ready(client, monkeypatch, tamper, legacy):
    assert custom_runtime(client, monkeypatch, tamper=tamper, legacy=legacy)["status"] == ("RECONCILE_REQUIRED" if tamper else "READY")


@pytest.mark.parametrize("changes", [{"image": BASE}, {"environment_id": "torch"}, {"requirements": "requirements.txt"}, {"dockerfile": "../Dockerfile"}])
def test_ambiguous_build_definitions_fail(changes):
    with pytest.raises(ValueError):
        DockerfileRuntimeSpec.model_validate({"source_id": "source." + "a" * 64, "dockerfile": "Dockerfile", "entrypoint": ["python3"], **changes})


@pytest.mark.parametrize("text", ["FROM python:latest", "FROM ${BASE}", "FROM scratch", "RUN echo no-base", "FROM x AS", "FROM --platform=linux/arm64 " + BASE,
    "# syntax=docker/dockerfile:1\nFROM " + BASE, "FROM " + BASE + "\nRUN <<EOF\ntrue\nEOF", "FROM " + BASE + "\nADD https://remote/file /tmp", "FROM " + BASE + "\nONBUILD RUN true", "FROM " + BASE + "\nCOPY --from=remote:latest /x /x"])
def test_dynamic_or_unpinned_dockerfiles_fail_before_build(tmp_path, text):
    (tmp_path / "Dockerfile").write_text(text)
    with pytest.raises(ValueError):
        inspect_dockerfile(tmp_path, "Dockerfile")


def test_multistage_dockerfile_uses_pinned_images_and_prior_stages(tmp_path):
    (tmp_path / "Dockerfile").write_text(f"FROM --platform=linux/amd64 {BASE} AS deps\nRUN echo deps\nFROM deps\nCOPY --from=deps /x /y\nCOPY --from=0 /z /w\n")
    assert inspect_dockerfile(tmp_path, "Dockerfile")["base_images"] == [BASE]


def test_asset_upload_limits_hash_and_symlinks(client, stored):
    assert put_asset(client, sha256="a" * 64).status_code == 409
    assert put_asset(client, b"a" * (2 * 1024 ** 2 + 1)).status_code == 413
    assert client.get("/api/storage-limits").json()["asset_archive_bytes"] == 2 * 1024 ** 2
    assert put_asset(client, archive({"../escape": b"x"})).status_code == 409
    assert put_asset(client, archive({".env": b"secret"})).status_code == 409
    assert client.get("/api/projects/demo/assets/asset." + "c" * 64).status_code == 404
    assert client.get("/api/projects/demo/assets/invalid").status_code == 409


def test_inputs_require_a_ready_managed_worker_and_known_assets(client, stored):
    legacy = runtime(client)
    from ml_exp_server.container_execution import ContainerExecutionService
    service = ContainerExecutionService(client.app.state.runtime)
    with service.state("demo", legacy["runtime_id"]) as (store, snapshot):
        old = dict(snapshot.value)
        old.pop("capabilities")
        old.pop("worker_contract")
        store.commit(old, expected_revision=snapshot.revision, event={"event": "legacy_fixture"})
    definition = {"run_id": "no-worker", "runtime_id": legacy["runtime_id"], "executor": "cloud", "inputs": [{"asset_id": "asset." + "a" * 64, "mount_path": "/inputs/data"}]}
    assert client.post("/api/projects/demo/runs", json=definition).status_code == 409
    definition["inputs"][0]["mount_path"] = "/etc"
    assert client.post("/api/projects/demo/runs", json=definition).status_code == 422
