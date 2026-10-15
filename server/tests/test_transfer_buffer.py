"""A global byte ceiling survives lost remote receipts and daemon restarts."""
import hashlib
import json
import os
from pathlib import Path
import threading

import pytest

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.data.transfer_buffer import TransferBuffer, BUFFER_BYTES, LOCK_STRIPES
from ml_exp_server.storage import atomic_json

UPLOAD = "upload." + "a" * 64


def receipt(number, data):
    return {"contract": "upload-part.v1", "upload_id": UPLOAD, "number": number,
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "durable": True}


def snapshot(buffer):
    with buffer.locked():
        return buffer.snapshot()


def test_sealed_unconfirmed_parts_survive_restart_and_count_globally(tmp_path):
    first, second = TransferBuffer(tmp_path, limit=1024 ** 2), TransferBuffer(tmp_path, limit=1024 ** 2)
    data = b"x" * (300 * 1024)
    sha = hashlib.sha256(data).hexdigest()
    with first.reserve(UPLOAD, 0, sha, len(data)) as part:
        part.path.write_bytes(data)
        part.seal()
    assert snapshot(second)["reserved_bytes"] == len(data)
    with pytest.raises(ApplicationError) as error:
        with second.reserve(UPLOAD, 1, sha, len(data)):
            pytest.fail("global ceiling must reject before reading a body")
    assert error.value.status_code == 503 and error.value.code == "TRANSFER_BUFFER_BUSY"
    with second.reserve(UPLOAD, 0, sha, len(data)) as part:
        assert part.sealed and part.path.read_bytes() == data
        part.acknowledge(receipt(0, data))
    assert snapshot(first)["reserved_bytes"] == 0
    assert not list(first.root.glob("*.part"))


def test_active_reservations_and_same_part_cannot_be_claimed_twice(tmp_path):
    first, second = TransferBuffer(tmp_path, limit=1024 ** 2), TransferBuffer(tmp_path, limit=1024 ** 2)
    data = b"a" * (300 * 1024)
    sha = hashlib.sha256(data).hexdigest()
    with first.reserve(UPLOAD, 0, sha, len(data)):
        assert snapshot(second)["reserved_bytes"] == len(data)
        for number in (0, 1):
            with pytest.raises(ApplicationError):
                with second.reserve(UPLOAD, number, sha, len(data)):
                    pytest.fail("part or global buffer must remain owned")
    assert snapshot(second)["reserved_bytes"] == 0


@pytest.mark.parametrize("field,value", [("sha256", "b" * 64), ("bytes", 2), ("durable", False), ("number", 1)])
def test_mismatched_remote_receipt_never_deletes_local_bytes(tmp_path, field, value):
    buffer = TransferBuffer(tmp_path, limit=1024 ** 2)
    data = b"original-part"
    good = receipt(0, data)
    with buffer.reserve(UPLOAD, 0, good["sha256"], len(data)) as part:
        part.path.write_bytes(data)
        part.seal()
        with pytest.raises(ValueError, match="receipt"):
            part.acknowledge({**good, field: value})
        path = part.path
    assert path.read_bytes() == data
    assert snapshot(buffer)["pending_bytes"] == len(data)
    with pytest.raises(ValueError, match="another digest"):
        with buffer.reserve(UPLOAD, 0, "b" * 64, len(data)):
            pass
    buffer.abort(UPLOAD)
    assert not path.exists()


def test_invalid_body_is_disposable_but_sealed_body_is_not(tmp_path):
    buffer = TransferBuffer(tmp_path, limit=1024 ** 2)
    data = b"expected"
    with buffer.reserve(UPLOAD, 0, hashlib.sha256(data).hexdigest(), len(data)) as part:
        part.path.write_bytes(b"bad")
        with pytest.raises(ValueError, match="SHA256"):
            part.seal()
    assert snapshot(buffer)["reserved_bytes"] == 0
    assert not list(buffer.root.glob("*.part"))


def test_confirmed_cleanup_recovery_requires_durable_exact_proof(tmp_path):
    buffer = TransferBuffer(tmp_path, limit=1024 ** 2)
    data = b"original"
    proof = receipt(0, data)
    with buffer.reserve(UPLOAD, 0, proof["sha256"], len(data)) as part:
        part.path.write_bytes(data)
        part.seal()
        path = part.path
    state = json.loads(path.with_suffix(".json").read_text())
    atomic_json(path.with_suffix(".json"), {**state, "status": "CONFIRMED", "receipt": {**proof, "durable": False}})
    with pytest.raises(ValueError):
        snapshot(buffer)
    assert path.read_bytes() == data
    atomic_json(path.with_suffix(".json"), {**state, "status": "CONFIRMED", "receipt": proof})
    assert snapshot(buffer)["reserved_bytes"] == 0
    assert not path.exists()


@pytest.mark.parametrize("limit", [0, BUFFER_BYTES + 1, True])
def test_buffer_limit_cannot_exceed_the_authorized_global_ceiling(tmp_path, limit):
    with pytest.raises(ValueError):
        TransferBuffer(tmp_path, limit=limit)


@pytest.mark.parametrize("field,value", [("bytes", -1), ("bytes", True), ("number", -1), ("upload_id", "invalid")])
def test_corrupt_accounting_never_understates_reserved_bytes(tmp_path, field, value):
    buffer = TransferBuffer(tmp_path, limit=1024 ** 2)
    data = b"retained"
    proof = receipt(0, data)
    with buffer.reserve(UPLOAD, 0, proof["sha256"], len(data)) as part:
        part.path.write_bytes(data)
        part.seal()
        path = part.path
    metadata = path.with_suffix(".json")
    state = json.loads(metadata.read_text())
    atomic_json(metadata, {**state, field: value})
    with pytest.raises(ValueError):
        snapshot(buffer)
    assert path.read_bytes() == data


def test_only_exact_permanently_deleted_worker_run_can_discard_unconfirmed_scratch(tmp_path):
    from ml_exp_server.deletion_state import RunFence
    buffer = TransferBuffer(tmp_path, limit=1024 ** 2)
    data = b"worker-scratch"
    proof = receipt(0, data)
    RunFence(tmp_path).mark("demo", "deleted")
    with buffer.reserve(UPLOAD, 0, proof["sha256"], len(data)) as part:
        part.path.write_bytes(data)
        part.seal()
        for binding in ({"project": "demo", "run_id": "live", "kind": "artifacts"},
                        {"project": "other", "run_id": "deleted", "kind": "checkpoint"},
                        {"project": "demo", "run_id": "deleted", "kind": "asset"}):
            assert not part.discard_deleted_run(binding)
            assert part.path.read_bytes() == data
        assert part.discard_deleted_run({"project": "demo", "run_id": "deleted", "kind": "checkpoint"})
        assert not part.path.exists()


def test_ack_cleanup_serializes_with_global_accounting_snapshot(tmp_path, monkeypatch):
    buffer = TransferBuffer(tmp_path, limit=1024 ** 2)
    data = b"concurrent confirmation"
    proof = receipt(0, data)
    ready, attempted = threading.Event(), threading.Event()
    errors = []
    with buffer.reserve(UPLOAD, 0, proof["sha256"], len(data)) as part:
        part.path.write_bytes(data)
        part.seal()
        metadata = part.path.with_suffix(".json")
        original = Path.read_text
        def read(path, *args, **kwargs):
            if path == metadata:
                ready.set()
                assert attempted.wait(2)
                assert metadata.exists()
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, "read_text", read)
        def acknowledge():
            try:
                assert ready.wait(2)
                attempted.set()
                part.acknowledge(proof)
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=acknowledge)
        thread.start()
        assert snapshot(buffer)["reserved_bytes"] == len(data)
        thread.join(2)
        assert not thread.is_alive() and not errors
        assert not part.path.exists()


def test_admission_charges_allocation_rounding_metadata_and_bounded_fixed_locks(tmp_path):
    buffer = TransferBuffer(tmp_path)
    data = b"small"
    proof = receipt(0, data)
    with buffer.reserve(UPLOAD, 0, proof["sha256"], len(data)) as part:
        part.path.write_bytes(data)
        part.seal()
        value = snapshot(buffer)
        assert value["payload_bytes"] == len(data)
        assert value["charged_bytes"] == buffer.fixed_bytes + buffer.charged(len(data))
        assert value["charged_bytes"] > value["payload_bytes"]
        assert value["available_bytes"] == value["limit_bytes"] - value["charged_bytes"]
        part.acknowledge(proof)
    assert snapshot(buffer)["charged_bytes"] == buffer.fixed_bytes


def test_fixed_stripes_bound_directory_growth_over_many_completed_uploads(tmp_path):
    buffer = TransferBuffer(tmp_path)
    data = b"small"
    sha = hashlib.sha256(data).hexdigest()
    for number in range(260):
        with buffer.reserve(UPLOAD, number, sha, len(data)) as part:
            part.path.write_bytes(data)
            part.seal()
            part.acknowledge(receipt(number, data))
    assert len(list(buffer.root.iterdir())) <= LOCK_STRIPES + 1
    assert all(path.name == "admission.lock" or path.name.startswith("stripe-") for path in buffer.root.iterdir())


def test_colliding_stripe_is_safe_backpressure_for_a_different_part(tmp_path):
    buffer = TransferBuffer(tmp_path)
    data = b"small"
    sha = hashlib.sha256(data).hexdigest()
    key = lambda number: hashlib.sha256((UPLOAD + ":" + str(number)).encode()).hexdigest()
    collision = next(number for number in range(1, 65536) if buffer.part_lock(key(number)) == buffer.part_lock(key(0)))
    with buffer.reserve(UPLOAD, 0, sha, len(data)):
        with pytest.raises(ApplicationError) as error:
            with buffer.reserve(UPLOAD, collision, sha, len(data)):
                pytest.fail("one stripe cannot have two active writers")
        assert error.value.code == "TRANSFER_BUFFER_BUSY"
    with buffer.reserve(UPLOAD, collision, sha, len(data)):
        pass


def test_crashed_atomic_json_scratch_does_not_accumulate_outside_accounting(tmp_path):
    buffer = TransferBuffer(tmp_path)
    data = b"sealed original"
    sha = hashlib.sha256(data).hexdigest()
    with buffer.reserve(UPLOAD, 0, sha, len(data)) as part:
        part.path.write_bytes(data)
        part.seal()
        path = part.path
    for number in range(3):
        temporary = path.with_name("." + path.stem + ".json." + str(number).zfill(32) + ".tmp")
        temporary.write_bytes(b"interrupted JSON replacement")
    assert len(list(buffer.root.glob("*.tmp"))) == 3
    assert snapshot(buffer)["charged_bytes"] == buffer.fixed_bytes + buffer.charged(len(data))
    assert not list(buffer.root.glob("*.tmp"))
    assert path.read_bytes() == data
