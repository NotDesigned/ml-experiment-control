"""A failed local save or interrupted server build must remain recoverable."""
import json
from pathlib import Path

import pytest

from ml_exp_client.api import save
from ml_exp_client.cli import main


def test_abandoned_temporary_file_does_not_block_private_atomic_save(tmp_path):
    path = tmp_path / "runtime.json"
    save(path, {"status": "EXECUTING"})
    stale = path.with_suffix(".json.tmp")
    stale.write_text("incomplete old save")
    save(path, {"status": "READY"})
    assert json.loads(path.read_text()) == {"status": "READY"}
    assert path.stat().st_mode & 0o777 == 0o600
    assert stale.read_text() == "incomplete old save"
    assert list(tmp_path.glob(".runtime.json.*")) == []


@pytest.mark.parametrize("failure", ["serialize", "replace", "sync"])
def test_failed_save_preserves_previous_state_and_cleans_its_temp(tmp_path, monkeypatch, failure):
    path = tmp_path / "state.json"
    save(path, {"previous": True})
    def fail(*args, **kwargs):
        raise OSError("injected failure")
    value = {"unserializable": object()} if failure == "serialize" else {"new": True}
    if failure == "replace":
        monkeypatch.setattr(Path, "replace", fail)
    if failure == "sync":
        monkeypatch.setattr("ml_exp_client.api.os.fsync", fail)
    with pytest.raises((TypeError, OSError)):
        save(path, value)
    assert json.loads(path.read_text()) == {"previous": True}
    assert list(tmp_path.glob(".state.json.*")) == []


@pytest.mark.parametrize("status", ["EXECUTING", "RECONCILE_REQUIRED", "READY"])
def test_explicit_cli_recovery_looks_up_receipt_without_replaying_build(tmp_path, monkeypatch, capsys, status):
    calls = []
    state = {"project": "demo", "runtime_id": "runtime." + "a" * 64,
             "confirmation": "BUILD example", "status": status}
    class API:
        def __init__(self, *args): pass
        def negotiate(self): return {}
        def call(self, path, **kwargs):
            calls.append(path)
            return state
        def wait(self, path): return {**state, "status": "READY"}
    monkeypatch.setattr("ml_exp_client.cli.Client", API)
    path = tmp_path / "state.json"
    save(path, state)
    assert main(["runtime", "--state", str(path), "--reconcile"]) == 0
    assert any(item.endswith("/reconcile") for item in calls) == (status != "READY")
    assert not any(item.endswith("/execute") for item in calls)
    assert json.loads(path.read_text())["status"] == "READY"
    capsys.readouterr()
