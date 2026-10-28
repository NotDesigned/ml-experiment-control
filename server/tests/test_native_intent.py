"""Optional Python-to-Rust OperationIntent validation boundary."""

from __future__ import annotations

import sys

import pytest

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.runs.native_intent import validate


PAYLOAD = {"kind": "SUBMIT_RUN", "title": "native", "draft": "{}"}


def _validator(path, *, exit_code=0, output=True):
    path.write_text(
        "#!" + sys.executable + "\n"
        "import json,sys\n"
        "value=json.load(sys.stdin)\n"
        + ("print(json.dumps(value))\n" if output else "")
        + f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )
    path.chmod(0o700)


def test_native_validator_is_optional_and_checks_identity(monkeypatch, tmp_path):
    monkeypatch.delenv("ML_EXPD_CONTRACT_BIN", raising=False)
    monkeypatch.setattr("ml_exp_server.runs.native_intent.shutil.which", lambda _: None)
    assert validate(PAYLOAD) is None

    binary = tmp_path / "ml-exp-rs"
    _validator(binary)
    monkeypatch.setenv("ML_EXPD_CONTRACT_BIN", str(binary))
    assert validate(PAYLOAD)["kind"] == "SUBMIT_RUN"


def test_native_validator_failure_is_a_bounded_application_error(monkeypatch, tmp_path):
    binary = tmp_path / "ml-exp-rs"
    _validator(binary, exit_code=2, output=False)
    monkeypatch.setenv("ML_EXPD_CONTRACT_BIN", str(binary))
    with pytest.raises(ApplicationError, match="native intent validation"):
        validate(PAYLOAD)
