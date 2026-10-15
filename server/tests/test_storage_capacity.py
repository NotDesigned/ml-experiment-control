"""Capacity promises survive restarts; concurrent admission cannot oversell disk."""
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_exp_server.data.multipart_upload import UploadStore
from ml_exp_server.storage_capacity import StorageCapacity, CapacityError
from ml_exp_server.storage import atomic_json


def test_live_claims_resize_release_and_do_not_expire_on_time(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.disk_usage", lambda _: SimpleNamespace(free=1000))
    budget = StorageCapacity(tmp_path, {"local_storage_headroom_bytes": 100})
    with budget.reserve(500) as lease:
        with budget.locked():
            assert budget.snapshot()["operation_reserved_bytes"] == 500
        with pytest.raises(CapacityError) as error:
            with budget.reserve(401): pass
        assert (error.value.required_free_bytes, error.value.available_bytes, error.value.reserved_bytes) == (1001, 1000, 500)
        lease.resize(600)
        lease.resize(1)
        with budget.reserve(899): pass
    with budget.locked():
        assert budget.snapshot()["operation_reserved_bytes"] == 0
    assert not list(budget.root.glob("claim-*"))


def crash_with_claim(registry):
    with StorageCapacity(registry, {}).reserve(1):
        os._exit(0)


def test_crashed_process_claim_is_reaped_without_touching_transfer_bytes(tmp_path):
    pending = tmp_path / "multipart-uploads/upload.keep/parts/0"
    pending.parent.mkdir(parents=True)
    pending.write_bytes(b"resumable")
    context = multiprocessing.get_context("fork")
    process = context.Process(target=crash_with_claim, args=(tmp_path,))
    process.start(); process.join(5)
    assert process.exitcode == 0
    budget = StorageCapacity(tmp_path, {})
    assert list(budget.root.glob("claim-*"))
    with budget.locked(): assert budget.snapshot()["operation_reserved_bytes"] == 0
    assert pending.read_bytes() == b"resumable" and not list(budget.root.glob("claim-*"))


def admit_upload(registry, barrier, queue, digit):
    store = UploadStore(registry, {"local_storage_headroom_bytes": 0})
    barrier.wait(timeout=5)
    try:
        store.create({"project": "demo", "run_id": "run-" + digit, "attempt_id": "attempt-001", "kind": "artifacts"}, digit * 64, 600, None)
        queue.put("accepted")
    except CapacityError:
        queue.put("rejected")


def test_two_processes_cannot_admit_the_same_remaining_space(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.disk_usage", lambda _: SimpleNamespace(free=1000))
    context = multiprocessing.get_context("fork")
    barrier, queue = context.Barrier(2), context.Queue()
    processes = [context.Process(target=admit_upload, args=(tmp_path, barrier, queue, digit)) for digit in "ab"]
    for process in processes: process.start()
    for process in processes: process.join(10)
    assert [process.exitcode for process in processes] == [0, 0]
    assert sorted(queue.get(timeout=1) for _ in processes) == ["accepted", "rejected"]
    # The promise is in session metadata, not a daemon's RAM.
    restarted = StorageCapacity(tmp_path, {"local_storage_headroom_bytes": 0})
    with restarted.locked(): assert restarted.snapshot()["upload_reserved_bytes"] == 600


def test_committed_parts_are_not_double_reserved_and_expired_promises_are_ignored(tmp_path, monkeypatch):
    free = [3000]
    monkeypatch.setattr("shutil.disk_usage", lambda _: SimpleNamespace(free=free[0]))
    uploads = UploadStore(tmp_path, {"upload_part_bytes": 1024, "local_storage_headroom_bytes": 0})
    binding = {"project": "demo", "run_id": "run-a", "attempt_id": "attempt-001", "kind": "artifacts"}
    value = uploads.create(binding, "a" * 64, 2048, None)
    part = tmp_path / "part"; part.write_bytes(b"a" * 1024)
    uploads.part(value["upload_id"], binding, 0, part, hashlib.sha256(b"a" * 1024).hexdigest(), 1024)
    free[0] = 1976
    with uploads.capacity.locked():
        assert uploads.capacity.snapshot()["upload_reserved_bytes"] == 1024
        assert uploads.capacity.snapshot(ignore_upload=value["upload_id"])["upload_reserved_bytes"] == 0
    uploads.create(binding, "b" * 64, 952, None)
    with pytest.raises(CapacityError): uploads.create(binding, "c" * 64, 1, None)
    path = uploads.root / value["upload_id"] / "upload.json"
    record = json.loads(path.read_text()); record["expires_at"] = "2000-01-01T00:00:00+00:00"; atomic_json(path, record)
    invalid = uploads.root / "upload.legacy/upload.json"; invalid.parent.mkdir(); invalid.write_text("{}")
    with uploads.capacity.locked(): assert uploads.capacity.snapshot()["upload_reserved_bytes"] == 952
    # Admission only drops the expired promise. Cleanup retains its existing TTL policy.
    assert (path.parent / "parts/0").read_bytes() == b"a" * 1024


@pytest.mark.parametrize("headroom", [-1, True, "1"])
def test_invalid_headroom_is_rejected(tmp_path, headroom):
    with pytest.raises(ValueError): StorageCapacity(tmp_path, {"local_storage_headroom_bytes": headroom})


def test_claim_initialization_failure_closes_and_removes_owned_metadata(tmp_path, monkeypatch):
    budget = StorageCapacity(tmp_path, {})
    monkeypatch.setattr("ml_exp_server.storage_capacity.json.dump", lambda *a: (_ for _ in ()).throw(OSError("disk failed")))
    with pytest.raises(OSError):
        with budget.reserve(1): pass
    assert not list(budget.root.glob("claim-*"))
