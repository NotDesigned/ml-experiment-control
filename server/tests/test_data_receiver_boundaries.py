"""Exercise the fixed receiver in-process, including its exact stdin contract."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.data.multipart_upload import UploadStore
from ml_exp_server.data import data_receiver as worker
from ml_exp_server.storage import atomic_json
from tests.test_artifact_store import archive


@pytest.fixture
def receiver(tmp_path):
    target = {"executor": "gpu", "backend": "slurm", "storage_scope": "storage." + "a" * 64,
              "profile_sha256": "b" * 64, "project_root": str(tmp_path / "remote/demo")}
    return {"target": target, "binding": {"project": "demo", "kind": "asset", "target": {
                key: target[key] for key in ("executor", "backend", "storage_scope", "profile_sha256")}},
            "upload_part_bytes": 16384, "upload_session_seconds": 300, "upload_max_parts": 65536,
            "headroom_bytes": 0, "archive_limit": 1024 ** 2, "expanded_limit": 1024 ** 2, "file_limit": 100}


def sealed(receiver, data=None):
    data = archive({"nested/x": b"verified fixture"}) if data is None else data
    sha = hashlib.sha256(data).hexdigest()
    value = worker.operate({**receiver, "operation": "create", "sha256": sha, "bytes": len(data)}, io.BytesIO())
    metadata = {**receiver, "upload_id": value["upload_id"]}
    for number in range(value["part_count"]):
        body = data[number * value["part_bytes"]:(number + 1) * value["part_bytes"]]
        worker.operate({**metadata, "operation": "part", "number": number, "sha256": hashlib.sha256(body).hexdigest(),
                        "bytes": len(body)}, io.BytesIO(body))
    return metadata, data, value


def test_receiver_real_create_part_read_publish_reuse_and_abort(receiver):
    metadata, data, value = sealed(receiver)
    assert worker.operate({**metadata, "operation": "read"}, io.BytesIO())["parts"]
    result = worker.operate({**metadata, "operation": "complete"}, io.BytesIO())
    assert result["verification"] == "remote-sha256-fsync"
    assert Path(result["remote_locator"]["data_path"]).joinpath("nested/x").read_bytes() == b"verified fixture"
    assert worker.operate({**metadata, "operation": "complete"}, io.BytesIO()) == result
    store = UploadStore(Path(receiver["target"]["project_root"]), {"local_storage_headroom_bytes": 0})
    unlimited = {**metadata, "expanded_limit": None, "files": result["files"]}
    assert worker.publish(store, unlimited, io.BytesIO(data), len(data), result["sha256"]) == result
    body = data[:value["part_bytes"]]
    part = {**metadata, "operation": "part", "number": 0, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
    assert worker.operate(part, io.BytesIO(body))["durable"] is True
    second = worker.operate({**receiver, "operation": "create", "sha256": "c" * 64, "bytes": 10}, io.BytesIO())
    assert worker.operate({**receiver, "operation": "abort", "upload_id": second["upload_id"]}, io.BytesIO())["status"] == "ABORTED"
    with pytest.raises(ValueError, match="unknown remote"):
        worker.operate({**metadata, "operation": "unknown"}, io.BytesIO())


@pytest.mark.parametrize("mode", ["size", "truncated", "excess"])
def test_receiver_part_body_failures_never_commit(receiver, mode):
    data = archive({"x": b"x"})
    value = worker.operate({**receiver, "operation": "create", "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}, io.BytesIO())
    metadata = {**receiver, "operation": "part", "upload_id": value["upload_id"], "number": 0,
                "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    if mode == "size": metadata["bytes"] -= 1
    body = data[:-1] if mode == "truncated" else data + b"extra" if mode == "excess" else data
    with pytest.raises(ValueError): worker.operate(metadata, io.BytesIO(body))
    root = Path(receiver["target"]["project_root"]) / "multipart-uploads"
    assert not list(root.glob(".part-*"))
    assert not json.loads((root / value["upload_id"] / "upload.json").read_text())["parts"]


@pytest.mark.parametrize("mode", ["identity", "symlink", "truncated", "corrupt"])
def test_completed_confirmation_revalidates_actual_archive_slice(receiver, tmp_path, mode):
    metadata, data, value = sealed(receiver)
    result = worker.operate({**metadata, "operation": "complete"}, io.BytesIO())
    body = data[:value["part_bytes"]]
    part = {**metadata, "operation": "part", "number": 0, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
    archive_path = Path(result["remote_locator"]["archive_path"])
    if mode == "identity": part["sha256"] = "d" * 64
    elif mode == "symlink":
        outside = tmp_path / "outside"; outside.write_bytes(data)
        archive_path.unlink(); archive_path.symlink_to(outside)
    elif mode == "truncated": archive_path.write_bytes(b"")
    else: archive_path.write_bytes(b"x" * len(data))
    with pytest.raises(ValueError): worker.operate(part, io.BytesIO(body))


@pytest.mark.parametrize("mode", ["declared", "symlink", "archive", "record"])
def test_publish_refuses_mismatched_inventory_archive_and_ready_metadata(receiver, tmp_path, mode):
    metadata, data, value = sealed(receiver)
    result = worker.operate({**metadata, "operation": "complete"}, io.BytesIO())
    root = Path(receiver["target"]["project_root"])
    archive_path = Path(result["remote_locator"]["archive_path"])
    if mode == "declared": metadata["files"] = [{"path": "nested/x", "bytes": 16, "sha256": "d" * 64}]
    elif mode == "symlink":
        outside = tmp_path / "outside"; outside.write_bytes(data)
        archive_path.unlink(); archive_path.symlink_to(outside)
    elif mode == "archive": archive_path.write_bytes(b"corrupt")
    else:
        atomic_json(root / "data-asset-metadata" / (result["asset_id"] + ".json"), {**result, "project": "other"})
    store = UploadStore(root, {"local_storage_headroom_bytes": 0})
    with pytest.raises(ValueError): worker.publish(store, metadata, io.BytesIO(data), len(data), result["sha256"])


@pytest.mark.parametrize("mode", ["empty", "duplicate", "count", "expanded", "special"])
def test_inventory_bounded_before_extraction(mode):
    body = io.BytesIO()
    with tarfile.open(fileobj=body, mode="w") as bundle:
        for name in (".", "nested"):
            item = tarfile.TarInfo(name); item.type = tarfile.DIRTYPE; bundle.addfile(item)
        if mode != "empty":
            item = tarfile.TarInfo("nested/x"); item.size = 1
            if mode == "special": item.type = tarfile.FIFOTYPE; item.size = 0
            bundle.addfile(item, io.BytesIO(b"x"))
            if mode == "duplicate": bundle.addfile(item, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        worker.inventory(body, expanded_limit=0 if mode == "expanded" else None, file_limit=1 if mode == "count" else 100)


@pytest.mark.parametrize("mode", ["missing", "truncated", "nul"])
def test_inventory_rejects_unavailable_truncated_or_invalid_member(monkeypatch, mode):
    item = tarfile.TarInfo("x\x00bad" if mode == "nul" else "x"); item.size = 2
    class Archive:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __iter__(self): return iter([item])
        def extractfile(self, member): return None if mode == "missing" else io.BytesIO(b"x")
    monkeypatch.setattr(worker.tarfile, "open", lambda **kwargs: Archive())
    with pytest.raises(ValueError): worker.inventory(io.BytesIO(), expanded_limit=None, file_limit=100)


def test_target_binding_cannot_redirect_scope_or_follow_parent_symlink(receiver, tmp_path):
    bad = deepcopy(receiver); bad["binding"]["target"]["storage_scope"] = "storage." + "c" * 64
    with pytest.raises(ValueError): worker.target_root(bad)
    outside = tmp_path / "outside"; outside.mkdir()
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError): worker.directory(tmp_path / "link/demo")


@pytest.mark.parametrize("frame", [b"", (8 * 1024 ** 2 + 1).to_bytes(4, "big"), (10).to_bytes(4, "big") + b"{}"])
def test_stdin_header_is_exact_and_bounded(monkeypatch, capsys, frame):
    monkeypatch.setattr(worker.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(frame)))
    assert worker.main() == 1
    assert json.loads(capsys.readouterr().out)["code"] == "REMOTE_UPLOAD_INVALID"


def test_stdin_main_success_and_typed_failure_never_echo_private_details(receiver, monkeypatch, capsys):
    metadata = {**receiver, "operation": "create", "sha256": "a" * 64, "bytes": 100}
    raw = json.dumps(metadata).encode()
    frame = len(raw).to_bytes(4, "big") + raw
    monkeypatch.setattr(worker.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(frame)))
    assert worker.main() == 0 and json.loads(capsys.readouterr().out)["ok"] is True
    monkeypatch.setattr(worker.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(frame)))
    monkeypatch.setattr(worker, "operate", lambda *a: (_ for _ in ()).throw(ApplicationError("private fixture", status_code=507, code="UPLOAD_STORAGE")))
    assert worker.main() == 1
    output = capsys.readouterr().out
    assert "private fixture" not in output and json.loads(output)["status_code"] == 507
