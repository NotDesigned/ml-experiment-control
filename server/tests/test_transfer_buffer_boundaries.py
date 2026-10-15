"""Recovery must distinguish valid buffered bytes from abandoned request scratch."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.data import transfer_buffer as module
from ml_exp_server.storage import atomic_json
from tests.test_transfer_buffer import UPLOAD, receipt, snapshot


def test_receiving_scratch_is_reaped_but_active_atomic_replacement_is_preserved(tmp_path):
    buffer = module.TransferBuffer(tmp_path)
    data = b"fixture"
    sha = hashlib.sha256(data).hexdigest()
    with buffer.reserve(UPLOAD, 0, sha, len(data)) as part:
        path, state = part.path, dict(part.state)
        active = path.with_name("." + path.stem + ".json." + "a" * 32 + ".tmp")
        active.write_bytes(b"uncommitted metadata")
        unknown = buffer.root / ".part-unknown.tmp"; unknown.write_bytes(b"unclassified fixture")
        assert snapshot(buffer)["reserved_bytes"] == len(data)
        assert active.exists() and unknown.exists()
        with pytest.raises(ValueError): part.acknowledge(receipt(0, data))
        assert buffer.pending(UPLOAD) == []
    path.write_bytes(b"incomplete")
    atomic_json(path.with_suffix(".json"), state)
    assert snapshot(buffer)["reserved_bytes"] == 0
    assert not active.exists() and not path.exists() and unknown.exists()


def test_confirmed_crash_recovery_can_reserve_same_identity_again(tmp_path):
    buffer = module.TransferBuffer(tmp_path)
    data = b"fixture"
    proof = receipt(0, data)
    with buffer.reserve(UPLOAD, 0, proof["sha256"], len(data)) as part:
        part.path.write_bytes(data); part.seal(); path = part.path
    state = json.loads(path.with_suffix(".json").read_text())
    atomic_json(path.with_suffix(".json"), {**state, "status": "CONFIRMED", "receipt": proof})
    with buffer.reserve(UPLOAD, 0, proof["sha256"], len(data)) as part:
        assert not part.sealed and not path.exists()
    assert snapshot(buffer)["reserved_bytes"] == 0


def test_part_metadata_path_and_filesystem_admission_fail_closed(tmp_path, monkeypatch):
    buffer = module.TransferBuffer(tmp_path)
    state = {"upload_id": UPLOAD, "number": 0, "sha256": "b" * 64, "bytes": 1, "status": "SEALED"}
    with pytest.raises(ValueError, match="identity"): module.checked_state(state, Path("part-" + "c" * 64 + ".json"))
    with pytest.raises(ValueError):
        with buffer.reserve("invalid", 0, "b" * 64, 1): pass
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(ApplicationError) as error:
        with buffer.reserve(UPLOAD, 0, "b" * 64, 1): pass
    assert error.value.status_code == 507
    monkeypatch.setattr(module.os, "statvfs", lambda _: SimpleNamespace(f_frsize=0))
    with pytest.raises(ValueError, match="block size"): module.TransferBuffer(tmp_path / "other")


def test_abort_does_not_cross_upload_identity_or_interrupt_active_writer(tmp_path):
    buffer = module.TransferBuffer(tmp_path)
    data = b"fixture"
    proof = receipt(0, data)
    with buffer.reserve(UPLOAD, 0, proof["sha256"], len(data)) as part:
        part.path.write_bytes(data); part.seal(); path = part.path
        buffer.abort("upload." + "d" * 64)
        assert path.read_bytes() == data
        with pytest.raises(ApplicationError) as error: buffer.abort(UPLOAD)
        assert error.value.code == "TRANSFER_BUFFER_BUSY"
    buffer.abort(UPLOAD)
    assert not path.exists()
