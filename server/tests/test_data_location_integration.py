"""Executor-scoped inputs use remote storage without a desktop rebuild."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.container_execution import ContainerExecutionService
from ml_exp_server.data.data_assets import AssetStore
from ml_exp_server.data.data_delivery import DataDeliveryService, digest
from ml_exp_server.data.locations import for_backend
from ml_exp_server.results.checkpoint_registry import storage_scope
from tests.test_container_api import client, runtime
from tests.test_sensecore_data_workflow import stored


def profile(kind="slurm", origin="cluster"):
    backend = {"kind": kind}
    backend.update({"ssh_alias": origin} if kind == "slurm" else {"storage_mount": origin})
    return {"storage_root": "/data/lab", "backend": backend}


def location(selected, kind):
    scope = storage_scope(selected["backend"], "/data/lab/demo")
    return scope, {"remote_storage": kind, "storage_scope": scope, "remote_locator": {"exact": kind}}


@pytest.mark.parametrize("storage_label", ["slurm", "wyd"])
def test_asset_selects_each_backend_copy_and_ignores_gpu_size(storage_label):
    slurm, cloud = profile(), profile("sensecore", "volume:/data")
    slurm_scope, slurm_copy = location(slurm, storage_label)
    cloud_scope, cloud_copy = location(cloud, "sensecore-ccr")
    original = {"asset_id": "asset." + "a" * 64, "remote_storage": storage_label,
                "remote_locations": {slurm_scope: slurm_copy, cloud_scope: cloud_copy}}
    before = deepcopy(original)
    assert for_backend(original, slurm, "demo")["remote_locator"] == {"exact": storage_label}
    assert for_backend(original, cloud, "demo")["remote_locator"] == {"exact": "sensecore-ccr"}
    larger = deepcopy(slurm); larger["capacity"] = {"gpus": 8}
    assert for_backend(original, larger, "demo") == for_backend(original, slurm, "demo")
    assert original == before
    old = {"asset_id": original["asset_id"], "remote_storage": "desktop-builder"}
    assert for_backend(old, cloud, "demo") is old


@pytest.mark.parametrize("change", ["absent", "scope", "backend", "malformed-backend", "other-project"])
def test_asset_cannot_silently_cross_storage_namespaces(change):
    selected = profile()
    scope, copy = location(selected, "wyd")
    value = {"remote_storage": "wyd", "remote_locations": {scope: copy}}
    project = "demo"
    if change == "absent": value["remote_locations"] = {}
    elif change == "scope": copy["storage_scope"] = "storage." + "0" * 64
    elif change == "backend": copy["remote_storage"] = "sensecore-ccr"
    elif change == "malformed-backend": copy["remote_storage"] = ["slurm"]
    else: project = "other"
    with pytest.raises(ApplicationError) as error:
        for_backend(value, selected, project)
    assert error.value.code == "INPUT_STORAGE_SCOPE_MISMATCH"


def publish(client, config, executor, kind):
    selected = ContainerExecutionService(client.app.state.runtime).profiles()[executor]
    files = [{"path": "tokens.bin", "bytes": 6, "sha256": hashlib.sha256(b"tokens").hexdigest()}]
    sha = "1" * 64
    value = {"project": "demo", "asset_id": "asset." + sha, "sha256": sha,
             "archive_bytes": 10240, "files": files, "status": "READY", "bytes": 6,
             "remote_storage": kind, "storage_scope": storage_scope(selected["backend"], "/data/lab/demo"),
             "executor": executor, "remote_locator": {}}
    if kind == "sensecore-ccr":
        value["remote_locator"] = {
            "project": "demo", "asset_id": value["asset_id"], "files_sha256": digest(files),
            "image": "registry.example/data@sha256:" + "2" * 64,
            "data_image_id": "data-image." + "3" * 64, "data_worker_sha256": "4" * 64,
            "recipe": "acp-data-image.stream.v1",
        }
    return AssetStore(config, client.app.state.runtime.config.project_registry_root_path()).publish_remote("demo", value)


@pytest.mark.parametrize("storage_label", ["slurm", "wyd"])
def test_slurm_remote_input_is_verified_before_run_and_never_pulls_archive(client, stored, monkeypatch, storage_label):
    bundle = runtime(client)
    asset = publish(client, stored[0], "gpu", storage_label)
    checks = []
    monkeypatch.setattr("ml_exp_server.backends.slurm.SlurmDataStager.check",
                        lambda self, *args: checks.append(args) or {"status": "READY"})
    monkeypatch.setattr(AssetStore, "archive", lambda *a: pytest.fail("do not pull a remote input onto API disk"))
    request = {"run_id": "slurm-remote", "runtime_id": bundle["runtime_id"], "executor": "gpu",
               "inputs": [{"asset_id": asset["asset_id"], "mount_path": "/inputs/data"}]}
    response = client.post("/api/projects/demo/runs", json=request)
    assert response.status_code == 200, response.text
    root = Path(client.app.state.runtime.project("demo").base_dir)
    frozen = yaml.safe_load((root / "experiments/campaigns/run-slurm-remote.yaml").read_text())["runs"][0]
    assert frozen["inputs"][0]["require_cached"] is True and len(checks) == 1
    monkeypatch.setattr("ml_exp_server.backends.slurm.SlurmDataStager.check", lambda *a: {"status": "NOT_FOUND"})
    failed = client.post("/api/projects/demo/runs", json=request | {"run_id": "slurm-not-ready"})
    assert failed.status_code == 409
    assert not (root / "experiments/campaigns/run-slurm-not-ready.yaml").exists()


def test_uploaded_ccr_image_copies_to_nas_without_any_builder_call(client, stored, monkeypatch):
    bundle = runtime(client)
    config = json.loads(stored[0].read_text())
    config["data_delivery"] = {"aec2": "debug-cluster", "worker_spec": "N6lS.Iu.I10.2c4g",
                               "gpus": 0, "cpus": 2, "memory_gb": 4,
                               "queue_timeout_seconds": 5, "copy_timeout_seconds": 5}
    stored[0].write_text(json.dumps(config))
    asset = publish(client, stored[0], "cloud", "sensecore-ccr")
    service = DataDeliveryService(client.app.state.runtime)
    value = service.prepare("demo", asset["asset_id"], "cloud")
    assert value["uploaded_image"] == asset["remote_locator"]
    assert value["worker_sha256"] == "4" * 64
    monkeypatch.setattr("ml_exp_server.data.data_delivery.builder_request", lambda *a: pytest.fail("already published CCR input never calls desktop builder"))
    monkeypatch.setattr("ml_exp_server.data.data_delivery.data_worker_digest", lambda: "9" * 64)
    creates = []

    def create(copy, document, **kw):
        creates.append(document)
        with service.state("demo", value["delivery_id"]) as (_, snapshot):
            token = snapshot.value["copy_token"]
        service.callback("demo", value["delivery_id"], token, {
            "status": "READY", "asset_id": value["asset_id"], "archive_sha256": value["archive_sha256"],
            "files_sha256": value["files_sha256"], "image": asset["remote_locator"]["image"],
            "data_path": value["copy_profile"]["data_root"] + "/" + asset["asset_id"],
        })
        return {"name": document["name"], "uid": "e" * 32}

    service._rest = SimpleNamespace(
        find=lambda *a: [], create=create,
        specs=lambda *a: [{"name": "N6lS.Iu.I10.2c4g", "device": {"number": 0},
                           "cpu": {"vcpu_allocatable": 2}, "memory": {"allocatable": 4}}],
    )
    ready = service.finish(service.begin("demo", value["delivery_id"], value["confirmation"]))
    assert ready["status"] == "READY" and ready["job_uid"] == "e" * 32
    assert creates[0]["roles"][0]["image_path"] == asset["remote_locator"]["image"]
    response = client.post("/api/projects/demo/runs", json={"run_id": "ccr-remote",
                           "runtime_id": bundle["runtime_id"], "executor": "cloud",
                           "inputs": [{"asset_id": asset["asset_id"], "mount_path": "/inputs/data"}]})
    assert response.status_code == 200, response.text
    root = Path(client.app.state.runtime.project("demo").base_dir)
    frozen = yaml.safe_load((root / "experiments/campaigns/run-ccr-remote.yaml").read_text())["runs"][0]
    assert frozen["inputs"][0]["require_cached"] is True
    assert frozen["inputs"][0]["data_delivery"]["delivery_id"] == value["delivery_id"]
