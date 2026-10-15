"""Corrupt uploads, expiry races and write failures never authorize foreign cleanup."""
import errno
import hashlib
import json
import os
from pathlib import Path

import pytest

from ml_exp_server.data import multipart_upload as module
from ml_exp_server.storage import atomic_json


def session(tmp_path):
    store = module.UploadStore(tmp_path, {"local_storage_headroom_bytes": 0})
    binding = {"kind": "asset", "project": "demo"}
    value = store.create(binding, hashlib.sha256(b"data").hexdigest(), 4, 100)
    return store, binding, value, store.root / value["upload_id"]


def test_record_rejects_links_and_duplicate_part_rechecks_remote_bytes(tmp_path):
    store, binding, value, directory = session(tmp_path)
    temporary = tmp_path / "part"; temporary.write_bytes(b"data")
    store.part(value["upload_id"], binding, 0, temporary, value["sha256"], 4)
    (directory / "parts/0").write_bytes(b"corrupt")
    temporary.write_bytes(b"data")
    with pytest.raises(ValueError, match="checksum"):
        store.part(value["upload_id"], binding, 0, temporary, value["sha256"], 4)
    original = directory / "upload.json"
    saved = directory / "saved.json"; original.rename(saved); original.symlink_to(saved)
    with pytest.raises(ValueError, match="link"): store.read(value["upload_id"], binding)


def test_expiry_recovers_missing_parts_directory_without_deleting_other_files(tmp_path):
    store, binding, value, directory = session(tmp_path)
    (directory / "parts").rmdir()
    atomic_json(directory / "upload.json", {**value, "expires_at": "2000-01-01T00:00:00+00:00"})
    store.prune()
    assert json.loads((directory / "upload.json").read_text())["status"] == "EXPIRED"


def test_expiry_reports_permission_failure_instead_of_silently_reaping(tmp_path, monkeypatch):
    store, binding, value, directory = session(tmp_path)
    original = module.os.open
    def denied(path, *args, **kwargs):
        if path == value["upload_id"]: raise PermissionError(errno.EACCES, "fixture permission")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(module.os, "open", denied)
    with pytest.raises(PermissionError): store.prune()
    assert (directory / "upload.json").exists()


def test_concurrent_expiry_can_remove_a_stale_temporary_before_stat(tmp_path, monkeypatch):
    store, _, _, _ = session(tmp_path)
    stale = store.root / ".part-vanished"; stale.write_bytes(b"scratch")
    original = module.os.stat
    def vanished(path, *args, **kwargs):
        if path == ".part-vanished": stale.unlink(missing_ok=True)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(module.os, "stat", vanished)
    store.prune()
    assert not stale.exists()


def test_expiry_requires_platform_fd_hardening_and_atomic_write_failure_keeps_old_record(tmp_path, monkeypatch):
    store, _, value, directory = session(tmp_path)
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(module.shutil.rmtree, "avoids_symlink_attacks", False)
            with pytest.raises(ValueError, match="fd-safe"): store.remove_expired_parts(descriptor)
        before = (directory / "upload.json").read_bytes()
        monkeypatch.setattr(module.os, "fsync", lambda _: (_ for _ in ()).throw(OSError(errno.EIO, "fixture fsync")))
        with pytest.raises(OSError): store.expire_record(descriptor, {**value, "status": "EXPIRED"})
        assert (directory / "upload.json").read_bytes() == before
        assert not list(directory.glob(".upload-*.tmp"))
    finally:
        os.close(descriptor)
