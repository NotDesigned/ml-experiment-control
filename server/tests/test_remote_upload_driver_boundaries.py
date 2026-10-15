"""Transport uncertainty and frozen destination failures without any SSH calls."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.data import remote_data_uploads as module
from ml_exp_server.data.data_assets import AssetStore
from ml_exp_server.storage import atomic_json
from tests.test_remote_wyd_uploads import receiver


def minimal(tmp_path):
    target = {"executor": "gpu", "backend": "slurm", "storage_scope": "storage." + "a" * 64,
              "profile_sha256": "b" * 64, "project_root": str(tmp_path / "remote/demo"),
              "profile": {"backend": {"kind": "slurm", "ssh_alias": "isolated"}}}
    assets = SimpleNamespace(root=tmp_path / "assets", objects=SimpleNamespace(config={}), limit=None,
                             expanded_limit=None, file_limit=100)
    return assets, target, {"project": "demo", "kind": "asset", "target": module.public_target(target)}


@pytest.mark.parametrize("mode", ["ok", "rejected", "invalid-json", "oversize", "exit", "expired", "write"])
def test_wyd_transport_returns_only_confirmed_bounded_receipts_and_always_closes(tmp_path, monkeypatch, mode):
    assets, target, binding = minimal(tmp_path)
    processes = []
    class Input(io.BytesIO):
        def write(self, value):
            if mode == "write": raise OSError("private fixture")
            return super().write(value)
    class Process:
        def __init__(self, command, **kwargs):
            assert command[0] == "ssh" and command[-2] == "isolated"
            self.stdin, self.killed = Input(), False
            raw = b"not-json" if mode == "invalid-json" else b"x" * (8 * 1024 ** 2 + 1) if mode == "oversize" else json.dumps(
                {"ok": False, "status_code": 507, "code": "UPLOAD_STORAGE"} if mode == "rejected" else {"ok": True, "result": {"status": "UPLOADING"}}).encode()
            self.stdout = io.BytesIO(raw); processes.append(self)
        def wait(self, **kwargs): return 1 if mode == "exit" else 0
        def poll(self): return None if mode == "write" and not self.killed else 0
        def kill(self): self.killed = True
    class Timer:
        def __init__(self, seconds, callback): self.callback = callback
        def start(self):
            if mode == "expired": self.callback()
        def cancel(self): pass
    monkeypatch.setattr(module.subprocess, "Popen", Process)
    monkeypatch.setattr(module.threading, "Timer", Timer)
    driver = module.WydDataUploads(assets, target)
    if mode == "ok": assert driver.read("upload." + "c" * 64, binding) == {"status": "UPLOADING"}
    else:
        with pytest.raises(ApplicationError) as failure: driver.read("upload." + "c" * 64, binding)
        assert failure.value.code == ("UPLOAD_STORAGE" if mode == "rejected" else "REMOTE_UPLOAD_UNCONFIRMED")
        assert "private fixture" not in str(failure.value)
    assert processes[0].stdin.closed and processes[0].stdout.closed


def test_wyd_binding_origin_and_metadata_bounds_fail_before_transport(tmp_path, monkeypatch):
    assets, target, binding = minimal(tmp_path)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: pytest.fail("invalid input cannot call SSH"))
    with pytest.raises(ValueError): module.WydDataUploads(assets, {**target, "profile": {"backend": {"ssh_alias": "../invalid"}}})
    with pytest.raises(ValueError): module.WydDataUploads(assets, target).call("read", {})
    with pytest.raises(ValueError): module.WydDataUploads(assets, target, files=["x" * (8 * 1024 ** 2)]).call("read", binding)


@pytest.mark.parametrize("project,requested", [("../bad", {}), ("demo", None), ("demo", {"executor": "../bad"}), ("demo", {"executor": 1})])
def test_frozen_upload_target_rejects_invalid_identity_before_profile_reads(tmp_path, monkeypatch, project, requested):
    monkeypatch.setattr("ml_exp_server.container_execution.ContainerExecutionService.profiles", lambda *a: pytest.fail("invalid identity cannot read profile"))
    with pytest.raises(ValueError): module.freeze_target(SimpleNamespace(), project, requested)


def test_frozen_profile_must_have_canonical_root(tmp_path, monkeypatch):
    runtime = SimpleNamespace(config=SimpleNamespace(project_registry_root_path=lambda: tmp_path))
    monkeypatch.setattr("ml_exp_server.container_execution.ContainerExecutionService.profiles", lambda *a: {
        "gpu": {"backend": {"kind": "slurm"}, "storage_root": "/a/../b"}})
    with pytest.raises(ValueError, match="canonical"): module.freeze_target(runtime, "demo", {"executor": "gpu"})


def test_factory_supports_frozen_ccr_and_refuses_unknown_saved_origins(tmp_path):
    assets, target, _ = minimal(tmp_path)
    ccr = module.remote_uploads(assets, {**target, "backend": "sensecore"})
    assert ccr.driver.__class__.__name__ == "CCRDataUploads"
    with pytest.raises(ValueError): module.remote_uploads(assets, {**target, "backend": "unknown"})
    with pytest.raises(ValueError): module.remote_uploads(assets, upload_id="../invalid")
    with pytest.raises(ApplicationError) as failure: module.remote_uploads(assets, upload_id="upload." + "c" * 64)
    assert failure.value.status_code == 404
    with pytest.raises(ApplicationError): module.remote_uploads(assets)
    path = assets.root.parent / "remote-data-uploads" / ("upload." + "c" * 64 + ".json")
    atomic_json(path, {"upload_id": "different"})
    with pytest.raises(ValueError, match="identity"): module.remote_uploads(assets, upload_id="upload." + "c" * 64)
    assert module.legacy_remote_uploads(assets) is None
    assets.objects.config.update(data_upload_storage="desktop-builder", data_upload_socket="fake.sock")
    assert isinstance(module.legacy_remote_uploads(assets), module.RemoteDataUploads)


def test_managed_create_rejects_changed_intent_bad_destination_and_unconfirmed_receipt(receiver, monkeypatch):
    assets, target, _ = receiver
    binding = {"kind": "asset", "project": "demo"}
    driver = module.remote_uploads(assets, target)
    def created(binding, sha, size, limit):
        return {"upload_id": module.UploadStore.identity(binding, sha, size), "binding": binding, "status": "UPLOADING"}
    monkeypatch.setattr(driver.driver, "create", created)
    value = driver.create(binding, "c" * 64, 100, 100)
    path = driver.root / (value["upload_id"] + ".json")
    saved = json.loads(path.read_text())
    atomic_json(path, {**saved, "transport_sha256": "different"})
    with pytest.raises(ValueError, match="already bound"): driver.create(binding, "c" * 64, 100, 100)
    atomic_json(path, saved)
    monkeypatch.setattr(driver.driver, "create", lambda *a: {"upload_id": "wrong", "binding": {}})
    with pytest.raises(ValueError, match="destination"): driver.create(binding, "c" * 64, 100, 100)
    monkeypatch.setattr(driver.driver, "create", lambda b, s, n, l: {**created(b, s, n, l), "status": "COMPLETED", "result": {"fixture": True}})
    monkeypatch.setattr(assets, "publish_remote", lambda project, result: {**result, "repaired": True})
    assert driver.create(binding, "c" * 64, 100, 100)["result"]["repaired"]
    monkeypatch.setattr(driver.driver, "part", lambda *a: {"durable": False})
    with pytest.raises(ValueError, match="receipt"): driver.part(value["upload_id"], driver.binding, 0, Path("unused"), "c" * 64, 100)


@pytest.mark.parametrize("changed", [{"sha": 1}, {"sha": "invalid"}, {"size": True}, {"size": 0},
    {"size": 101}, {"config": {"upload_part_bytes": 100}}, {"config": {"upload_part_bytes": "1024"}},
    {"config": {"upload_max_parts": 0}}, {"config": {"upload_max_parts": True}},
    {"size": 2049, "limit": None, "config": {"upload_part_bytes": 1024, "upload_max_parts": 2}}])
def test_invalid_managed_upload_never_writes_intent_or_contacts_destination(receiver, monkeypatch, changed):
    assets, target, _ = receiver
    assets.objects.config.update(changed.get("config", {}))
    driver = module.remote_uploads(assets, target)
    monkeypatch.setattr(driver.driver, "create", lambda *a: pytest.fail("invalid policy cannot call destination"))
    with pytest.raises(ApplicationError): driver.create({"project": "demo", "kind": "asset"}, changed.get("sha", "c" * 64), changed.get("size", 100), changed.get("limit", 100))
    assert not list(driver.root.glob("*.json"))
