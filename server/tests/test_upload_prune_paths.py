"""Expiry cleans only canonical owned upload directories without following links."""
import hashlib
import json
from pathlib import Path

import pytest

from ml_exp_server.data.multipart_upload import UploadStore
from ml_exp_server.storage import atomic_json


@pytest.mark.parametrize("kind", ["directory", "metadata", "parts", "identity"])
def test_expiry_never_follows_a_link_or_unknown_upload_identity(tmp_path, kind):
    store = UploadStore(tmp_path / "registry", {"upload_part_bytes": 1024, "local_storage_headroom_bytes": 0})
    binding = {"kind": "asset", "project": "demo"}
    value = store.create(binding, hashlib.sha256(b"transport").hexdigest(), 9, 100)
    value["expires_at"] = "2000-01-01T00:00:00+00:00"
    directory = store.root / value["upload_id"]
    outside = tmp_path / "outside-owned-fixture"
    outside.mkdir()
    (outside / "parts").mkdir()
    (outside / "parts" / "sentinel").write_bytes(b"outside must survive")
    atomic_json(outside / "upload.json", value)
    atomic_json(directory / "upload.json", value)
    if kind == "directory":
        (directory / "upload.json").unlink()
        (directory / "parts").rmdir()
        directory.rmdir()
        directory.symlink_to(outside, target_is_directory=True)
    elif kind == "metadata":
        (directory / "upload.json").unlink()
        (directory / "upload.json").symlink_to(outside / "upload.json")
    elif kind == "parts":
        (directory / "parts").rmdir()
        (directory / "parts").symlink_to(outside / "parts", target_is_directory=True)
    else:
        atomic_json(directory / "upload.json", {**value, "upload_id": "upload." + "c" * 64})
    before = (outside / "upload.json").read_bytes()
    store.prune()
    assert (outside / "parts" / "sentinel").read_bytes() == b"outside must survive"
    assert (outside / "upload.json").read_bytes() == before
    assert json.loads((directory / "upload.json").read_text())["status"] == "UPLOADING"


@pytest.mark.parametrize("directory_fd", [True, False])
def test_directory_swap_during_expiry_keeps_fd_bound_cleanup_inside_original_upload(tmp_path, monkeypatch, directory_fd):
    monkeypatch.setattr("ml_exp_server.data.multipart_upload.RMTREE_DIRECTORY_FD", directory_fd)
    store = UploadStore(tmp_path / "registry", {"upload_part_bytes": 1024, "local_storage_headroom_bytes": 0})
    value = store.create({"kind": "asset", "project": "demo"}, hashlib.sha256(b"transport").hexdigest(), 9, 100)
    value["expires_at"] = "2000-01-01T00:00:00+00:00"
    directory = store.root / value["upload_id"]
    atomic_json(directory / "upload.json", value)
    (directory / "parts" / "owned").write_bytes(b"expired transport")
    outside = tmp_path / "outside-owned-fixture"
    (outside / "parts").mkdir(parents=True)
    (outside / "parts" / "sentinel").write_bytes(b"outside must survive")
    atomic_json(outside / "upload.json", value)
    before = (outside / "upload.json").read_bytes()
    saved = directory.with_name("moved-owned-upload")
    original = store.expired
    def swap(record):
        directory.rename(saved)
        directory.symlink_to(outside, target_is_directory=True)
        return original(record)
    monkeypatch.setattr(store, "expired", swap)
    store.prune()
    assert (outside / "parts" / "sentinel").read_bytes() == b"outside must survive"
    assert (outside / "upload.json").read_bytes() == before
    assert not (saved / "parts").exists()
    assert json.loads((saved / "upload.json").read_text())["status"] == "EXPIRED"
