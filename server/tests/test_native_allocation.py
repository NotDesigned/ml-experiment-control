"""Optional Python-to-Rust Slurm allocation parsing boundary."""

from __future__ import annotations

import sys

import pytest

from ml_exp_server.runs.native_allocation import parse


OUTPUT = (
    "NodeName=n1 Partitions=gpu CfgTRES=cpu=16,gres/gpu=8 "
    "AllocTRES=cpu=4,gres/gpu=3 CPUTot=16 CPUAlloc=4 RealMemory=1024 "
    "AllocMem=256 State=MIXED\nML_EXPD_QUEUE\n123\n"
)


def _validator(path, *, exit_code=0, output=True):
    path.write_text(
        "#!" + sys.executable + "\n"
        "import json,sys\n"
        + ("print(json.dumps({'resources': {}, 'nodes': [], 'pending_queue_entries': 1}))\n" if output else "")
        + f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )
    path.chmod(0o700)


def test_native_allocation_is_optional(monkeypatch, tmp_path):
    monkeypatch.delenv("ML_EXPD_ALLOCATION_BIN", raising=False)
    monkeypatch.setattr("ml_exp_server.runs.native_allocation.shutil.which", lambda _: None)
    assert parse(OUTPUT, "gpu") is None

    binary = tmp_path / "ml-exp-rs"
    _validator(binary)
    monkeypatch.setenv("ML_EXPD_ALLOCATION_BIN", str(binary))
    assert parse(OUTPUT, "gpu")["pending_queue_entries"] == 1


def test_native_allocation_failure_is_fail_closed(monkeypatch, tmp_path):
    binary = tmp_path / "ml-exp-rs"
    _validator(binary, exit_code=2, output=False)
    monkeypatch.setenv("ML_EXPD_ALLOCATION_BIN", str(binary))
    with pytest.raises(ValueError, match="native allocation evidence"):
        parse(OUTPUT, "gpu")
