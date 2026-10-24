"""Execute the exact stdlib SSH receiver locally; no provider or desktop calls."""
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
from copy import deepcopy

import pytest

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.data.data_assets import AssetStore
from ml_exp_server.data.remote_data_uploads import remote_uploads, public_target
from ml_exp_server.data.transfer_buffer import TransferBuffer
from ml_exp_server.results.checkpoint_registry import storage_scope
from tests.test_artifact_store import archive


@pytest.fixture
def receiver(tmp_path, monkeypatch):
    config = tmp_path / "objects.json"
    config.write_text(json.dumps({"endpoint": "http://unused.invalid", "bucket": "unused",
        "access_key": "fixture", "secret_key": "fixture", "upload_part_bytes": 1024,
        "local_storage_headroom_bytes": 0, "max_asset_archive_bytes": 1024 ** 2, "max_asset_bytes": 1024 ** 2}))
    assets = AssetStore(config, tmp_path / "api")
    profile = {"backend": {"kind": "slurm", "ssh_alias": "isolated"}, "storage_root": str(tmp_path / "remote")}
    target = {"executor": "gpu", "backend": "slurm", "profile": profile,
              "project_root": str(tmp_path / "remote/demo"), "profile_sha256": "b" * 64,
              "storage_scope": storage_scope(profile["backend"], str(tmp_path / "remote/demo"))}
    original, commands = subprocess.Popen, []
    def local(command, **kwargs):
        assert command[0] == "ssh" and command[-2] == "isolated"
        commands.append(command)
        return original(shlex.split(command[-1]), **kwargs)
    monkeypatch.setattr("ml_exp_server.data.remote_data_uploads.subprocess.Popen", local)
    monkeypatch.setattr(assets.objects, "client", lambda: pytest.fail("no object store or desktop is involved"))
    return assets, target, commands


def upload(assets, target, data, tmp_path, *, files=None, complete=True):
    driver = remote_uploads(assets, target, files=files)
    value = driver.create({"project": "demo", "kind": "asset"}, hashlib.sha256(data).hexdigest(), len(data), assets.limit)
    buffer = TransferBuffer(assets.root.parent)
    for number in range(value["part_count"]):
        chunk = data[number * value["part_bytes"]:(number + 1) * value["part_bytes"]]
        sha = hashlib.sha256(chunk).hexdigest()
        with buffer.reserve(value["upload_id"], number, sha, len(chunk)) as part:
            part.path.write_bytes(chunk)
            part.seal()
            receipt = driver.part(value["upload_id"], value["binding"], number, part.path, sha, len(chunk))
            part.acknowledge(receipt)
    return driver, value, driver.complete(value["upload_id"], value["binding"]) if complete else None


def test_remote_upload_is_independently_verified_persisted_and_metadata_only_on_api(receiver, tmp_path):
    assets, target, commands = receiver
    data = archive({"nested/tokens.bin": b"training-data" * 400})
    driver, value, result = upload(assets, target, data, tmp_path)
    assert result["status"] == "READY" and result["verification"] == "remote-sha256-fsync"
    assert result["sha256"] == hashlib.sha256(data).hexdigest()
    assert result["files"] == [{"path": "nested/tokens.bin", "bytes": 5200,
                               "sha256": hashlib.sha256(b"training-data" * 400).hexdigest()}]
    remote = Path(target["project_root"])
    assert (remote / "data-archives" / (result["asset_id"] + ".tar")).read_bytes() == data
    assert (remote / "data-assets" / result["asset_id"] / "nested/tokens.bin").read_bytes() == b"training-data" * 400
    assert not list((assets.root.parent / "transfer-buffer").glob("*.part"))
    assert not (assets.root.parent / "multipart-uploads").exists()
    assert {p.name for p in (assets.root / "demo" / result["asset_id"]).iterdir()} == {"asset.json"}
    assert "private" not in json.dumps(commands)
    resumed = remote_uploads(assets, upload_id=value["upload_id"])
    assert resumed.target == target
    assert resumed.read(value["upload_id"], resumed.binding)["status"] == "COMPLETED"
    assert resumed.complete(value["upload_id"], resumed.binding) == result


def test_lost_part_receipt_resumes_same_remote_part_and_local_buffer(receiver, tmp_path, monkeypatch):
    assets, target, _ = receiver
    data = archive({"x": b"x"})
    driver = remote_uploads(assets, target)
    value = driver.create({"project": "demo", "kind": "asset"}, hashlib.sha256(data).hexdigest(), len(data), assets.limit)
    chunk = data[:value["part_bytes"]]
    sha = hashlib.sha256(chunk).hexdigest()
    buffer = TransferBuffer(assets.root.parent)
    with buffer.reserve(value["upload_id"], 0, sha, len(chunk)) as part:
        part.path.write_bytes(chunk)
        part.seal()
        driver.part(value["upload_id"], value["binding"], 0, part.path, sha, len(chunk))
        retained = part.path
    assert retained.read_bytes() == chunk
    restored = remote_uploads(assets, upload_id=value["upload_id"])
    assert restored.read(value["upload_id"], restored.binding)["parts"] == {"0": {"sha256": sha, "bytes": len(chunk)}}
    with buffer.reserve(value["upload_id"], 0, sha, len(chunk)) as part:
        receipt = restored.part(value["upload_id"], restored.binding, 0, part.path, sha, len(chunk))
        part.acknowledge(receipt)
    assert not retained.exists()
    assert restored.abort(value["upload_id"], restored.binding)["status"] == "ABORTED"


def test_corrupt_remote_part_cannot_be_acknowledged_or_completed(receiver, tmp_path):
    assets, target, _ = receiver
    data = archive({"x": b"x"})
    driver, value, _ = upload(assets, target, data, tmp_path, complete=False)
    part = Path(target["project_root"]) / "multipart-uploads" / value["upload_id"] / "parts/0"
    part.write_bytes(b"corruption")
    local = tmp_path / "retry"; local.write_bytes(data[:1024])
    with pytest.raises(ApplicationError):
        driver.part(value["upload_id"], value["binding"], 0, local, hashlib.sha256(data[:1024]).hexdigest(), 1024)
    with pytest.raises(ApplicationError):
        driver.complete(value["upload_id"], value["binding"])
    assert not list(assets.root.glob("demo/asset.*"))


def test_declared_inventory_is_checked_against_remote_archive_bytes(receiver, tmp_path):
    assets, target, _ = receiver
    data = archive({"x": b"actual"})
    files = [{"path": "x", "bytes": 6, "sha256": "c" * 64}]
    driver, value, _ = upload(assets, target, data, tmp_path, files=files, complete=False)
    with pytest.raises(ApplicationError):
        driver.complete(value["upload_id"], value["binding"])
    assert not list(assets.root.glob("demo/asset.*"))


@pytest.mark.parametrize("kind", ["escape", "symlink", "duplicate"])
def test_remote_archive_extraction_rejects_unsafe_members(receiver, tmp_path, kind):
    assets, target, _ = receiver
    body = io.BytesIO()
    with tarfile.open(fileobj=body, mode="w") as bundle:
        item = tarfile.TarInfo("../escaped" if kind == "escape" else "x")
        item.size = 1
        if kind == "symlink":
            item.type, item.linkname = tarfile.SYMTYPE, "/tmp/escape"
        bundle.addfile(item, io.BytesIO(b"x") if kind != "symlink" else None)
        if kind == "duplicate":
            bundle.addfile(item, io.BytesIO(b"x"))
    driver, value, _ = upload(assets, target, body.getvalue(), tmp_path, complete=False)
    with pytest.raises(ApplicationError):
        driver.complete(value["upload_id"], value["binding"])
    assert not (Path(target["project_root"]) / "escaped").exists()
    assert not list(assets.root.glob("demo/asset.*"))


def test_saved_target_and_transport_manifest_are_used_after_configuration_changes(receiver, tmp_path):
    assets, target, _ = receiver
    data = archive({"x": b"x"})
    driver = remote_uploads(assets, target, files=[{"path": "x", "bytes": 1, "sha256": hashlib.sha256(b"x").hexdigest()}],
                            transport_sha256="d" * 64)
    value = driver.create({"project": "demo", "kind": "asset"}, hashlib.sha256(data).hexdigest(), len(data), assets.limit)
    target["profile"]["storage_root"] = "/different-current-profile"
    restored = remote_uploads(assets, upload_id=value["upload_id"])
    assert restored.target["profile"]["storage_root"] != target["profile"]["storage_root"]
    assert restored.transport_sha256 == "d" * 64 and restored.files == driver.files
    with pytest.raises(ApplicationError):
        restored.read(value["upload_id"], {"project": "other", "kind": "asset"})


def test_same_content_supports_scope_aliases_and_separate_storage_without_rebinding(receiver, tmp_path):
    assets, target, _ = receiver
    assets.objects.config["upload_part_bytes"] = 16 * 1024
    data = archive({"tokens.bin": b"verified dataset"})
    _, first, original = upload(assets, target, data, tmp_path)
    path = assets.root / "demo" / original["asset_id"] / "asset.json"
    frozen = path.read_bytes()
    alias = {**deepcopy(target), "executor": "gpu-alias", "profile_sha256": "c" * 64}
    _, second, reused = upload(assets, alias, data, tmp_path)
    assert first["upload_id"] != second["upload_id"]
    assert reused["executor"] == "gpu-alias"
    assert reused["storage_scope"] == target["storage_scope"]
    assert len(reused["remote_locations"]) == 1 and path.read_bytes() == frozen
    other = deepcopy(target)
    other["executor"], other["profile_sha256"] = "other", "d" * 64
    other["profile"]["storage_root"] = str(tmp_path / "second-remote")
    other["project_root"] = str(tmp_path / "second-remote/demo")
    other["storage_scope"] = storage_scope(other["profile"]["backend"], other["project_root"])
    _, _, separated = upload(assets, other, data, tmp_path)
    assert separated["executor"] == "other" and separated["storage_scope"] == other["storage_scope"]
    assert len(separated["remote_locations"]) == 2 and path.read_bytes() == frozen
    assert separated["remote_locator"]["data_path"].startswith(other["project_root"] + "/")
    wrong = {**separated, "storage_scope": target["storage_scope"]}
    with pytest.raises(ValueError, match="already bound"):
        assets.publish_remote("demo", wrong)
