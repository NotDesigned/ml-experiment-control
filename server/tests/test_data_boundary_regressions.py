"""Input scope, lost-ACK concurrency and legacy boundary regressions."""
import asyncio
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

from fastapi import HTTPException
import pytest

from ml_exp_server.api import upload_routes
from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.backends.operations import WydOperations
from ml_exp_server.container_execution import ContainerExecutionService
from ml_exp_server.data.data_assets import AssetStore
from ml_exp_server.data.data_delivery import DataDeliveryService
from ml_exp_server.data.locations import for_backend
from ml_exp_server.data.transfer_buffer import TransferBuffer
from ml_exp_server.runs.executor_capabilities import declaration, mismatches, ExecutionRequirements
from tests.test_container_api import client, import_source
from tests.test_cpu_script_preparation import script
from tests.test_data_location_integration import profile, location, publish
from tests.test_executor_capabilities import request
from tests.test_remote_upload_api import targeted, begin
from tests.test_artifact_store import archive
from tests.test_sensecore_data_workflow import stored


def test_malformed_location_list_is_rejected_without_falling_back_to_legacy():
    original = {"remote_storage": "desktop-builder", "remote_locations": []}
    with pytest.raises(ApplicationError) as error:
        for_backend(original, profile(), "demo")
    assert error.value.code == "INPUT_STORAGE_SCOPE_MISMATCH"
    assert original == {"remote_storage": "desktop-builder", "remote_locations": []}


def test_native_wyd_cache_failure_never_launches_a_second_transfer(monkeypatch):
    selected = profile()
    scope, copy = location(selected, "wyd")
    asset = {"remote_storage": "wyd", "remote_locations": {scope: copy}}
    stager = SimpleNamespace(check=lambda *a: {"status": "NOT_FOUND"},
                             assets=SimpleNamespace(read=lambda *a: asset),
                             publish=lambda *a: pytest.fail("native data must not be retransferred"))
    monkeypatch.setattr("ml_exp_server.backends.operations.WydDataStager", lambda *a: stager)
    owner = SimpleNamespace(runtime=None, update=lambda *a, **k: pytest.fail("no new effect marker"))
    with pytest.raises(ApplicationError) as error:
        WydOperations().prepare_asset(owner, "demo", "prep", {"delivery_requested": []},
                                      {"profile": selected}, "asset." + "a" * 64, None)
    assert error.value.code == "WYD_DATA_NOT_READY"


def test_cpu_script_matching_rejects_adapter_without_before_job_input_support(client):
    selected = ContainerExecutionService(client.app.state.runtime).profiles()["cloud"]
    capabilities = declaration(selected, artifact_store=True, desktop=False, cpu_preparation=True)
    capabilities["data"]["asset_preparation"] = "inside_job"
    run = request(data_preparation={"script": "download.py"},
                  inputs=[{"asset_id": "asset." + "a" * 64, "mount_path": "/inputs/data"}])
    assert "data.download_script.inputs_require_before_job" in mismatches(
        capabilities, selected, run, ExecutionRequirements())


def test_uploaded_ccr_recipe_mismatch_is_rejected_before_cpu_intent(client, stored, monkeypatch):
    config = json.loads(stored[0].read_text())
    config["data_delivery"] = {"aec2": "debug-cluster", "worker_spec": "cpu.2c4g", "gpus": 0,
                               "cpus": 2, "memory_gb": 4, "queue_timeout_seconds": 5, "copy_timeout_seconds": 5}
    stored[0].write_text(json.dumps(config))
    asset = deepcopy(publish(client, stored[0], "cloud", "sensecore-ccr"))
    for copy in asset["remote_locations"].values():
        copy["remote_locator"]["recipe"] = "unknown-recipe"
    monkeypatch.setattr(AssetStore, "read", lambda *a: asset)
    service = DataDeliveryService(client.app.state.runtime)
    with pytest.raises(ValueError, match="uploaded data image receipt differs"):
        service.prepare("demo", asset["asset_id"], "cloud")
    assert not list(service.root.glob("**/delivery.*.json"))


def test_saved_session_is_not_visible_from_another_project(targeted):
    api, _, _ = targeted
    value = begin(api, archive({"x": b"x"})).json()
    import_source(api, project="other")
    response = api.get("/api/projects/other/asset-uploads/" + value["upload_id"])
    assert response.status_code == 404
    assert response.json()["detail"] == "unknown upload for this project"


def test_worker_upload_cannot_select_a_dataset_target_before_authorization(client, monkeypatch):
    async def reject(*a, **k):
        pytest.fail("input-only fields must fail before worker authorization")
    monkeypatch.setattr(upload_routes, "target", reject)
    response = client.post("/api/attempt-uploads/demo/trial/attempt-001/artifacts", json={
        "sha256": "a" * 64, "bytes": 1024, "target": {"executor": "gpu"}})
    assert response.status_code == 422


@pytest.mark.parametrize("case", ["not-confirmed", "busy", "other-error", "already-acknowledged"])
def test_lost_ack_confirmation_retains_unproven_bytes_and_respects_other_lease(case, tmp_path, monkeypatch):
    pending = {"number": 0, "bytes": 1024, "sha256": "a" * 64}
    monkeypatch.setattr(TransferBuffer, "pending", lambda *a: [pending])
    forwarded, exited = [], []
    uploads = SimpleNamespace(remote=True, part=lambda *a: forwarded.append(a))
    context = SimpleNamespace(root=tmp_path / "data-assets"), uploads, {"project": "demo", "kind": "asset"}
    value = {"parts": {} if case == "not-confirmed" else {"0": {"bytes": 1024, "sha256": "a" * 64}}}
    @contextmanager
    def reserve(*a, **k):
        if case in {"busy", "other-error"}:
            raise HTTPException(status_code=503, headers={"X-ML-Expd-Error-Code":
                                "TRANSFER_BUFFER_BUSY" if case == "busy" else "OTHER_STORAGE_ERROR"})
        try:
            # The other owner ACKed between pending() and reserve().
            yield SimpleNamespace(sealed=False)
        finally:
            exited.append(True)
    monkeypatch.setattr(TransferBuffer, "reserve", reserve)
    operation = upload_routes.confirm_buffered_parts(context, "upload." + "1" * 64, value)
    if case == "other-error":
        with pytest.raises(HTTPException) as error:
            asyncio.run(operation)
        assert error.value.headers["X-ML-Expd-Error-Code"] == "OTHER_STORAGE_ERROR"
    else:
        asyncio.run(operation)
    assert not forwarded
    assert exited == ([True] if case == "already-acknowledged" else [])


@pytest.mark.parametrize("operation", ["read", "publish"])
def test_remote_location_inventory_corruption_does_not_replace_frozen_asset(client, stored, operation):
    asset = publish(client, stored[0], "gpu", "wyd")
    assets = AssetStore(stored[0], client.app.state.runtime.config.project_registry_root_path())
    record = assets.root / "demo" / asset["asset_id"] / "asset.json"
    original = record.read_bytes()
    path = assets.root.parent / "remote-data-assets/demo" / (asset["asset_id"] + ".json")
    value = json.loads(path.read_text())
    value["sha256"] = "9" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="locator differs|another content identity"):
        if operation == "read":
            assets.read("demo", asset["asset_id"])
        else:
            assets.publish_remote("demo", asset)
    assert record.read_bytes() == original


def test_remote_asset_missing_scope_and_expanded_quota_fail_before_publication(client, stored):
    asset = publish(client, stored[0], "gpu", "wyd")
    assets = AssetStore(stored[0], client.app.state.runtime.config.project_registry_root_path())
    with pytest.raises(ValueError, match="storage scope differs"):
        assets.publish_remote("demo", {**asset, "storage_scope": ""})
    assets.expanded_limit = 1
    with pytest.raises(ValueError, match="expanded bytes exceed policy"):
        assets.validate_inventory(asset["files"])


@pytest.mark.parametrize("kind,executor", [("wyd", "gpu"), ("sensecore-ccr", "cloud")])
def test_remote_download_error_is_typed_without_public_object_store(client, stored, kind, executor):
    assert "public_endpoint" not in json.loads(stored[0].read_text())
    asset = publish(client, stored[0], executor, kind)
    response = client.get("/api/projects/demo/assets/" + asset["asset_id"] + "/download")
    assert response.status_code == 404
    assert response.headers["X-ML-Expd-Error-Code"] == "DIRECT_DOWNLOAD_UNAVAILABLE"
    assert not stored[1]


def test_historical_object_download_still_reports_missing_public_endpoint(client, stored):
    import io
    import hashlib
    data = archive({"tokens.bin": b"legacy object data"})
    assets = AssetStore(stored[0], client.app.state.runtime.config.project_registry_root_path())
    asset = assets.receive("demo", io.BytesIO(data), hashlib.sha256(data).hexdigest(), len(data))
    response = client.get("/api/projects/demo/assets/" + asset["asset_id"] + "/download")
    assert response.status_code == 404
    assert response.json() == {"detail": "direct downloads are not configured"}
    assert "X-ML-Expd-Error-Code" not in response.headers


def test_cpu_receipt_entrypoint_blocks_sitecustomize_and_pythonpath_before_reading_capability(
        script, tmp_path):
    import os
    import shlex
    import subprocess
    import sys
    from ml_exp_server.backends.sensecore_preparation import script_receipt_worker_sha256
    from ml_exp_server.workers.script_preparation import TOKEN_ENV
    service, run, *_ = script
    run.update(data_preparation_receipt_mode="callback_v1",
               data_preparation_receipt_worker_sha256=script_receipt_worker_sha256())
    definition = service.definition(run)
    callback = {"token": "trial-only-capability", "request": {}}
    command = shlex.split(service.command(definition, callback))
    command[command.index("python3")] = sys.executable
    poison = tmp_path / "project-pythonpath"
    poison.mkdir()
    captured = tmp_path / "captured-dummy-capability"
    code = ("import os\n"
            "with open(os.environ['TRIAL_CAPTURE'], 'w') as stream:\n"
            "    stream.write(os.environ.get('" + TOKEN_ENV + "', 'missing'))\n"
            "raise RuntimeError('unexpected project import before capability removal')\n")
    (poison / "sitecustomize.py").write_text(code)
    (poison / "json.py").write_text(code)
    environment = {**os.environ, "PYTHONPATH": str(poison), "TRIAL_CAPTURE": str(captured)}
    result = subprocess.run(command, cwd=poison, env=environment, capture_output=True, text=True, timeout=10)
    # No image helpers are installed locally, so this isolated wrapper fails
    # before data preparation and performs no callback or provider operation.
    assert result.returncode == 74
    assert "ML_EXPD_SCRIPT_RECEIPT=FAILED" in result.stdout
    assert not captured.exists()
    assert callback["token"] not in result.stdout + result.stderr
    # The same image-side poison can read the dummy capability without isolation.
    control = [item for item in command if item not in {"-I", "-S"}]
    subprocess.run(control, cwd=poison, env=environment, capture_output=True, timeout=10)
    assert captured.read_text() == callback["token"]
    legacy = deepcopy(definition)
    legacy["contract"] = "cpu-script-preparation.v1"
    assert "-I" not in shlex.split(service.command(legacy))


def test_frozen_receipt_worker_identity_includes_isolated_python_launcher(monkeypatch):
    from ml_exp_server.backends import sensecore_preparation
    isolated = sensecore_preparation.script_receipt_worker_sha256()
    monkeypatch.setattr(sensecore_preparation, "SCRIPT_RECEIPT_PYTHON", ("python3", "-u", "-c"))
    assert sensecore_preparation.script_receipt_worker_sha256() != isolated
