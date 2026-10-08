"""Real filesystem and hash checks; transport and timer boundaries substituted."""
import hashlib
import errno
import io
import json
import os
from pathlib import Path
import runpy
import signal
import sys
from types import SimpleNamespace

import pytest
from ml_exp_server.workers import result_worker as worker
from ml_exp_server.workers import persistent_state
from ml_exp_server.workers import worker_artifacts
from ml_exp_server.workers import worker_http


def test_training_result_survives_failed_upload_and_contains_no_capability(tmp_path, monkeypatch):
    for key, value in {"PROJECT_NAME": "demo", "RUN_ID": "run", "ATTEMPT_ID": "attempt-001"}.items():
        monkeypatch.setenv(key, value)
    calls = []
    status = 200
    class Connection:
        def request(self, method, path, body, headers):
            calls.append((method, path, json.loads(body)))
            assert headers["Authorization"] == "Bearer private-capability"
        def getresponse(self): return SimpleNamespace(status=status, read=lambda n: b"{}")
        def close(self): calls.append("closed")
    monkeypatch.setattr(worker, "https_connection", lambda *a, **kw: Connection())
    output = tmp_path / "outputs"; output.mkdir()
    worker.training_result(output, "https://api.example/api/artifact-transfers/demo/run/attempt-001", "private-capability", 0)
    assert calls[0][1] == "/api/result-ready-transfers/demo/run/attempt-001"
    saved = (tmp_path / "training-result.json").read_text()
    assert "private" not in saved and json.loads(saved)["exit_code"] == 0
    status = 401
    with pytest.raises(ValueError):
        worker.training_result(output, "https://api.example/api/artifact-transfers/demo/run/attempt-001", "private-capability", 1)
    assert json.loads((tmp_path / "training-result.json").read_text())["exit_code"] == 1
    assert calls[-1] == "closed"


@pytest.mark.parametrize("case", ["normal", "checksum", "empty", "expected-archive", "symlink", "no-checkpoint", "unexported", "absent"])
def test_recovery_uses_arbitrary_names_and_requires_original_checksums(tmp_path, monkeypatch, case):
    outputs = tmp_path / "outputs"; outputs.mkdir()
    state = tmp_path / "state"
    generation = state / "checkpoints/generation"; generation.mkdir(parents=True)
    contents = b"model"
    (generation / "chosen.bin").write_bytes(contents)
    (outputs / "chosen.bin").write_bytes(contents)
    item = {"path": "checkpoints/generation/chosen.bin", "bytes": len(contents), "sha256": hashlib.sha256(contents).hexdigest()}
    context = {"project": "demo", "run_id": "run", "attempt_id": "attempt-001", "source_id": "source",
               "image_id": "image", "storage_scope": "scope", "state_root": str(state)}
    checkpoint = persistent_state.document(context, {"step": 7, "files": [item]})
    value = {"outputs_root": str(outputs), "patterns": ["**/*"], "checkpoint": checkpoint,
             "expected_archive": None, "url": "https://api.example/api/artifact-transfers/demo/run/attempt-001",
             "token": "private-capability", "limit": 0}
    captured = []
    def upload(url, token, stream, size):
        assert token == "private-capability"; stream.seek(0)
        captured.append(stream.read())
    monkeypatch.setattr(worker, "upload_parts", upload)
    if case == "checksum": (outputs / "chosen.bin").write_bytes(b"bad")
    if case == "empty": (outputs / "chosen.bin").unlink()
    if case == "expected-archive": value["expected_archive"] = {"sha256": "a" * 64, "bytes": 1}
    if case == "symlink":
        linked = tmp_path / "alias"; linked.symlink_to(outputs); value["outputs_root"] = str(linked)
    if case == "no-checkpoint": value["checkpoint"] = None
    if case == "unexported": value["patterns"] = ["other.bin"]; (outputs / "other.bin").write_bytes(b"other")
    if case == "absent": (outputs / "chosen.bin").unlink(); (outputs / "other.bin").write_bytes(b"other")
    if case in {"checksum", "empty", "expected-archive", "symlink"}:
        with pytest.raises(ValueError): worker.recover(value)
        assert not captured
    else:
        worker.recover(value)
        assert captured
        value["expected_archive"] = {"sha256": hashlib.sha256(captured[-1]).hexdigest(), "bytes": len(captured[-1])}
        worker.recover(value)
        assert captured[0] == captured[1]


def test_relative_storage_and_timer_fail_closed_without_secret_output(tmp_path, monkeypatch, capsys):
    with pytest.raises(ValueError): worker.checked_directory(Path("relative"))
    handlers = []; alarms = []
    monkeypatch.setattr(signal, "signal", lambda sig, call: handlers.append(call))
    monkeypatch.setattr(signal, "alarm", lambda seconds: alarms.append(seconds))
    monkeypatch.setattr(worker, "recover", lambda value: None)
    assert worker.main({"seconds": 900}) == 0
    assert alarms == [900, 0]
    def expired(value): handlers[-1](None, None)
    monkeypatch.setattr(worker, "recover", expired)
    assert worker.main({"seconds": 900}) == 74
    assert alarms[-1] == 0 and "TimeoutError" in capsys.readouterr().out
    monkeypatch.setattr(worker, "recover", lambda value: (_ for _ in ()).throw(ValueError("private-secret")))
    assert worker.main({"seconds": 900}) == 74
    assert "private-secret" not in capsys.readouterr().out


def test_worker_can_import_in_existing_cpu_image_without_server_package(monkeypatch):
    for name, module in {"artifacts": worker_artifacts, "persistent_state": persistent_state, "worker_http": worker_http}.items():
        monkeypatch.setitem(sys.modules, name, module)
    value = runpy.run_path(worker.__file__, run_name="cpu-recovery")
    assert callable(value["recover"])


def recovery_input(tmp_path):
    outputs = tmp_path / "outputs"; outputs.mkdir()
    state = tmp_path / "state"
    generation = state / "checkpoints/generation"; generation.mkdir(parents=True)
    for root in (outputs, generation): (root / "model.bin").write_bytes(b"model")
    context = {"project": "demo", "run_id": "run", "attempt_id": "attempt-001", "source_id": "source",
               "image_id": "image", "storage_scope": "scope", "state_root": str(state)}
    checkpoint = persistent_state.document(context, {"step": 7, "files": [{
        "path": "checkpoints/generation/model.bin", "bytes": 5,
        "sha256": hashlib.sha256(b"model").hexdigest(),
    }]})
    return {"outputs_root": str(outputs), "patterns": ["**/*"], "checkpoint": checkpoint,
            "expected_archive": None, "url": "https://api.example/api/artifact-transfers/demo/run/attempt-001",
            "token": "private-capability", "limit": 0, "seconds": 900}


def test_recovery_uses_local_tmp_even_if_nas_writes_and_tmpdir_are_unavailable(tmp_path, monkeypatch):
    value = recovery_input(tmp_path)
    original = worker.tempfile.TemporaryFile
    locations = []
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    def temporary(*args, **kwargs):
        locations.append(kwargs.get("dir"))
        assert kwargs.get("dir") == "/tmp", "NAS must never receive disposable archives"
        return original(*args, **kwargs)
    monkeypatch.setattr(worker.tempfile, "TemporaryFile", temporary)
    captured = []
    monkeypatch.setattr(worker, "upload_parts", lambda url, token, stream, size: captured.append(size))
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    worker.recover(value)
    assert locations == ["/tmp"] and captured[0] > 5
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == before


@pytest.mark.parametrize("phase", ["outputs_directory", "checkpoint_directory", "checkpoint_verify",
                                   "output_verify", "archive_tempfile", "archive_write",
                                   "archive_verify", "archive_upload"])
def test_recovery_failure_reports_exact_phase_errno_without_secret_text(tmp_path, monkeypatch, capsys, phase):
    value = recovery_input(tmp_path)
    upload_calls = []
    monkeypatch.setattr(worker, "upload_parts", lambda *args: upload_calls.append(args))
    def fail(*args, **kwargs):
        raise OSError(errno.ENOSPC, "private-capability", "/nas/private-capability/file")
    if phase in {"outputs_directory", "checkpoint_directory"}:
        original = worker.checked_directory
        def check(root):
            if (root == Path(value["outputs_root"])) == (phase == "outputs_directory"): fail()
            return original(root)
        monkeypatch.setattr(worker, "checked_directory", check)
    elif phase == "checkpoint_verify": monkeypatch.setattr(worker, "restore", fail)
    elif phase == "output_verify": monkeypatch.setattr(worker, "open_file", fail)
    elif phase == "archive_tempfile": monkeypatch.setattr(worker.tempfile, "TemporaryFile", fail)
    elif phase == "archive_write": monkeypatch.setattr(worker, "archive_outputs", fail)
    elif phase == "archive_verify":
        value["checkpoint"] = None
        monkeypatch.setattr(worker, "sha", fail)
    else: monkeypatch.setattr(worker, "upload_parts", fail)
    assert worker.main(value) == 74
    output = capsys.readouterr().out
    row = json.loads(next(line.split("=", 1)[1] for line in output.splitlines()
                          if line.startswith("ML_EXPD_RESULT_RECOVERY_DIAGNOSTIC=")))
    assert row == {"phase": phase, "error": "OSError", "errno": 28, "os_code": "ENOSPC",
                   "file_ref": hashlib.sha256(b"/nas/private-capability/file").hexdigest()[:20]}
    assert "private-capability" not in output and "/nas/" not in output
    assert "ML_EXPD_RESULT_RECOVERY=FAILED error=OSError" in output
    assert not upload_calls


@pytest.mark.parametrize("error", [ValueError("private-capability"), OSError("private-capability"),
                                   OSError(5, "private-capability", b"/private-capability"),
                                   OSError(True, "private-capability", 123)])
def test_recovery_diagnostics_handle_missing_or_untrusted_errno_filename(tmp_path, monkeypatch, capsys, error):
    value = recovery_input(tmp_path)
    monkeypatch.setattr(worker, "checked_directory", lambda root: (_ for _ in ()).throw(error))
    assert worker.main(value) == 74
    output = capsys.readouterr().out
    row = json.loads(output.splitlines()[0].split("=", 1)[1])
    expected_errno = 5 if isinstance(error, OSError) and error.errno == 5 else None
    assert row["errno"] == expected_errno
    assert "private-capability" not in output


def test_result_upload_failure_reports_http_storage_failure_without_response_text(tmp_path, monkeypatch, capsys):
    value = recovery_input(tmp_path)
    def unavailable(*args):
        raise worker_artifacts.TransientHttpTransferError(507, 'UPLOAD_STORAGE')
    monkeypatch.setattr(worker, 'upload_parts', unavailable)
    assert worker.main(value) == 74
    output = capsys.readouterr().out
    row = json.loads(next(line.split('=', 1)[1] for line in output.splitlines()
                         if line.startswith('ML_EXPD_RESULT_RECOVERY_DIAGNOSTIC=')))
    assert row['phase'] == 'archive_upload' and row['http_status'] == 507 and row['api_code'] == 'UPLOAD_STORAGE'
    assert row['errno'] is None and row['file_ref'] is None
    assert 'private-capability' not in output
