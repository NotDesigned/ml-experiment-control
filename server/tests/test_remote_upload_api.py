"""Targeted asset HTTP uploads never stage complete archives on the API host."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import io
from contextlib import contextmanager

import pytest
import yaml

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.data.remote_data_uploads import WydDataUploads
from ml_exp_server.data.transfer_buffer import TransferBuffer
from tests.test_sensecore_data_workflow import client, stored
from tests.test_artifact_store import archive


@pytest.fixture
def targeted(client, stored, tmp_path, monkeypatch):
    runtime = client.app.state.runtime
    profiles = Path(runtime.config.container_execution.profiles_file)
    values = yaml.safe_load(profiles.read_text())
    values["executors"]["gpu"]["storage_root"] = str(tmp_path / "remote")
    profiles.write_text(yaml.safe_dump(values))
    config = json.loads(stored[0].read_text())
    config.update(upload_part_bytes=1024, local_storage_headroom_bytes=0,
                  data_upload_storage="desktop-builder", data_upload_socket="never-contact-desktop")
    stored[0].write_text(json.dumps(config))
    original = subprocess.Popen
    def local(command, **kwargs):
        assert command[0] == "ssh"
        return original(shlex.split(command[-1]), **kwargs)
    monkeypatch.setattr("ml_exp_server.data.remote_data_uploads.subprocess.Popen", local)
    monkeypatch.setattr("ml_exp_server.data.remote_data_uploads.stage_request", lambda *a, **k: pytest.fail("desktop must not be contacted"))
    return client, runtime.config.project_registry_root_path(), profiles


def begin(client, data, **changes):
    return client.post("/api/projects/demo/asset-uploads", json={
        "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "target": {"executor": "gpu"}, **changes})


def test_new_upload_requires_target_and_exposes_bounded_buffer_metadata_without_desktop(targeted):
    client, root, _ = targeted
    data = archive({"x": b"x"})
    assert begin(client, data, target=None).status_code == 422
    assert begin(client, data, target={"executor": "missing"}).status_code == 404
    assert begin(client, data, target={"executor": "gpu", "path": "/tmp/escape"}).status_code == 422
    assert begin(client, data, target={"executor": "gpu", "storage_scope": "storage." + "c" * 64}).status_code == 409
    assert begin(client, data, bytes=4 * 1024 ** 3 + 1).status_code == 413
    limits = client.get("/api/storage-limits")
    assert limits.status_code == 200
    assert limits.json()["data_upload_storage"] == "target-executor"
    assert limits.json()["transfer_buffer"]["limit_bytes"] == 1024 ** 3
    assert not (root / "remote-data-uploads").exists()


def test_http_upload_resumes_frozen_executor_and_publishes_metadata_only(targeted, stored):
    client, root, profiles = targeted
    data = archive({"tokens.bin": b"separate dataset" * 200})
    response = begin(client, data)
    assert response.status_code == 200, response.text
    value = response.json()
    assert '"profile":' not in json.dumps(value) and "ssh_alias" not in json.dumps(value)
    endpoint = "/api/projects/demo/asset-uploads/" + value["upload_id"]
    # A subsequent profile change cannot redirect a saved upload.
    current = yaml.safe_load(profiles.read_text())
    current["executors"]["gpu"]["storage_root"] = "/different/current/root"
    profiles.write_text(yaml.safe_dump(current))
    for number in range(value["part_count"]):
        part = data[number * value["part_bytes"]:(number + 1) * value["part_bytes"]]
        result = client.put(endpoint + "/parts/" + str(number), params={"sha256": hashlib.sha256(part).hexdigest()}, content=part)
        assert result.status_code == 200, result.text
        assert result.json()["durable"] is True
    assert client.get(endpoint).json()["parts"]
    result = client.post(endpoint + "/complete")
    assert result.status_code == 200, result.text
    asset = result.json()
    assert asset["remote_storage"] == "wyd" and asset["verification"] == "remote-sha256-fsync"
    assert client.get("/api/projects/demo/assets/" + asset["asset_id"]).json() == asset
    assert client.post(endpoint + "/complete").json() == asset
    assert not list((root / "transfer-buffer").glob("*.part"))
    assert not (root / "multipart-uploads").exists()
    assert stored[1] == {}
    unavailable = client.get("/api/projects/demo/assets/" + asset["asset_id"] + "/archive")
    assert unavailable.status_code == 404
    assert unavailable.headers["X-ML-Expd-Error-Code"] == "DIRECT_DOWNLOAD_UNAVAILABLE"
    assert stored[1] == {}


def test_unconfirmed_forwarding_retains_local_part_until_exact_remote_receipt(targeted, monkeypatch):
    client, root, _ = targeted
    data = archive({"x": b"x"})
    value = begin(client, data).json()
    endpoint = "/api/projects/demo/asset-uploads/" + value["upload_id"]
    chunk = data[:value["part_bytes"]]
    digest = hashlib.sha256(chunk).hexdigest()
    original = WydDataUploads.part
    def lost(*args, **kwargs):
        original(*args, **kwargs)
        raise ApplicationError("response lost", status_code=503, code="REMOTE_UPLOAD_UNCONFIRMED")
    monkeypatch.setattr(WydDataUploads, "part", lost)
    assert client.put(endpoint + "/parts/0", params={"sha256": digest}, content=chunk).status_code == 503
    pending = list((root / "transfer-buffer").glob("*.part"))
    assert len(pending) == 1 and pending[0].read_bytes() == chunk
    # A bad retry body cannot overwrite the previously sealed original part.
    assert client.put(endpoint + "/parts/0", params={"sha256": digest}, content=b"wrong").status_code == 400
    assert pending[0].read_bytes() == chunk
    monkeypatch.setattr(WydDataUploads, "part", original)
    assert client.put(endpoint + "/parts/0", params={"sha256": digest}, content=chunk).status_code == 200
    assert not pending[0].exists()


def test_buffer_backpressure_rejects_before_receiving_another_body(targeted, monkeypatch):
    client, root, _ = targeted
    data = archive({"x": b"x"})
    value = begin(client, data).json()
    endpoint = "/api/projects/demo/asset-uploads/" + value["upload_id"]
    from ml_exp_server.data import transfer_buffer
    @contextmanager
    def full(*args, **kwargs):
        raise transfer_buffer.busy()
        yield
    monkeypatch.setattr(TransferBuffer, "reserve", full)
    chunk = data[:1024]
    response = client.put(endpoint + "/parts/0", params={"sha256": hashlib.sha256(chunk).hexdigest()}, content=chunk)
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"
    assert not list((root / "transfer-buffer").glob("*.part"))


def test_real_client_skips_committed_part_and_post_complete_confirms_its_retained_buffer(targeted, monkeypatch, tmp_path):
    from ml_exp_client.api import ClientError
    from ml_exp_client.assets import upload_asset_parts
    client, root, _ = targeted
    original = WydDataUploads.part
    calls, lost = [], False
    def forward(*args, **kwargs):
        nonlocal lost
        calls.append(args[3])
        receipt = original(*args, **kwargs)
        if not lost:
            lost = True
            raise ApplicationError("response lost", status_code=503, code="REMOTE_UPLOAD_UNCONFIRMED")
        return receipt
    monkeypatch.setattr(WydDataUploads, "part", forward)
    monkeypatch.setattr("ml_exp_client.assets.time.sleep", lambda _: None)
    requests = []
    class API:
        def call(self, path, *, data=None, raw=None, method=None, timeout=None):
            verb = method or ("POST" if data is not None else "GET")
            requests.append((verb, path))
            response = client.request(verb, path, json=data, content=raw)
            if response.status_code != 200:
                raise ClientError("transfer unavailable", status=response.status_code)
            return response.json()
    data = archive({"weights.bin": b"fixture dataset"})
    result = upload_asset_parts(API(), "demo", io.BytesIO(data), hashlib.sha256(data).hexdigest(), len(data),
                                tmp_path / "client-upload.json", target={"executor": "gpu"})
    assert result["status"] == "READY"
    assert len([path for method, path in requests if method == "PUT" and "/parts/0?" in path]) == 1
    assert calls.count(0) == 2
    assert not list((root / "transfer-buffer").glob("*.part"))


def test_get_observes_pending_part_without_replaying_remote_write_then_post_cleans_completed_part(targeted, monkeypatch):
    client, root, _ = targeted
    data = archive({"x": b"x"})
    value = begin(client, data).json()
    endpoint = "/api/projects/demo/asset-uploads/" + value["upload_id"]
    original = WydDataUploads.part
    def lost(*args, **kwargs):
        original(*args, **kwargs)
        raise ApplicationError("lost", status_code=503, code="REMOTE_UPLOAD_UNCONFIRMED")
    monkeypatch.setattr(WydDataUploads, "part", lost)
    chunk = data[:1024]
    assert client.put(endpoint + "/parts/0", params={"sha256": hashlib.sha256(chunk).hexdigest()}, content=chunk).status_code == 503
    pending = list((root / "transfer-buffer").glob("*.part"))
    assert len(pending) == 1
    monkeypatch.setattr(WydDataUploads, "part", lambda *a, **k: pytest.fail("GET cannot repeat a remote part"))
    assert client.get(endpoint).json()["parts"]["0"]["bytes"] == 1024
    assert pending[0].read_bytes() == chunk
    monkeypatch.setattr(WydDataUploads, "part", original)
    for number in range(1, value["part_count"]):
        part = data[number * 1024:(number + 1) * 1024]
        assert client.put(endpoint + "/parts/" + str(number), params={"sha256": hashlib.sha256(part).hexdigest()}, content=part).status_code == 200
    # Another worker completed the destination before the API's POST resumed.
    from ml_exp_server.data.remote_data_uploads import remote_uploads
    from ml_exp_server.data.data_assets import AssetStore
    config = client.app.state.runtime.config.container_execution.artifact_store_file
    restored = remote_uploads(AssetStore(Path(config), root), upload_id=value["upload_id"])
    restored.complete(value["upload_id"], restored.binding)
    assert client.post(endpoint + "/complete").status_code == 200
    assert not pending[0].exists()


def test_repeated_create_confirms_pending_part_with_its_saved_full_target_binding(targeted, monkeypatch):
    client, root, _ = targeted
    data = archive({"x": b"x"})
    value = begin(client, data).json()
    endpoint = "/api/projects/demo/asset-uploads/" + value["upload_id"]
    original = WydDataUploads.part
    def lost(*args, **kwargs):
        original(*args, **kwargs)
        raise ApplicationError("lost", status_code=503, code="REMOTE_UPLOAD_UNCONFIRMED")
    monkeypatch.setattr(WydDataUploads, "part", lost)
    chunk = data[:value["part_bytes"]]
    assert client.put(endpoint + "/parts/0", params={"sha256": hashlib.sha256(chunk).hexdigest()}, content=chunk).status_code == 503
    assert list((root / "transfer-buffer").glob("*.part"))
    monkeypatch.setattr(WydDataUploads, "part", original)
    resumed = begin(client, data)
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()["upload_id"] == value["upload_id"] and resumed.json()["parts"]["0"]["bytes"] == len(chunk)
    assert not list((root / "transfer-buffer").glob("*.part"))
