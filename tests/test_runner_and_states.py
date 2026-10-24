import subprocess
import sys

import pytest

from experiment_control.backends.slurm import normalize_state
from experiment_control.runner import CommandResult, SubprocessRunner
from experiment_control.states import FailureClass


def test_command_result_preserves_subprocess_failure_contract():
    result = CommandResult(("false",), 7, stderr="failed")
    with pytest.raises(subprocess.CalledProcessError) as error:
        result.check_returncode()
    assert error.value.returncode == 7
    assert error.value.stderr == "failed"


def test_subprocess_runner_preserves_success_output_and_optional_check(tmp_path):
    result = SubprocessRunner().run(
        [sys.executable, "-c", "print(input())"],
        cwd=tmp_path,
        input_text="payload",
    )
    assert result.returncode == 0
    assert result.stdout == "payload\n"

    failed = SubprocessRunner().run(
        [sys.executable, "-c", "raise SystemExit(4)"], check=False,
    )
    assert failed.returncode == 4


def test_missing_executable_is_a_structured_command_failure():
    result = SubprocessRunner().run(
        ["/definitely/missing/experimentctl-tool"], check=False
    )
    assert result.returncode == 127
    assert "FileNotFoundError" in result.stderr


def test_subprocess_runner_replaces_invalid_utf8_at_the_process_boundary():
    result = SubprocessRunner().run([
        sys.executable,
        "-c",
        "import os; os.write(1, b'valid\\ninvalid=\\x88\\n'); "
        "os.write(2, b'bad=\\xff\\n')",
    ])
    assert result.returncode == 0
    assert result.stdout == "valid\ninvalid=�\n"
    assert result.stderr == "bad=�\n"


def test_subprocess_runner_enforces_hard_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        SubprocessRunner().run(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            timeout_seconds=0.05,
        )


def test_subprocess_runner_rejects_nonpositive_timeout():
    with pytest.raises(ValueError, match="greater than zero"):
        SubprocessRunner().run([sys.executable, "-c", "pass"], timeout_seconds=0)


def test_slurm_success_requires_zero_exit():
    assert normalize_state("COMPLETED", "0:0") == "SUCCEEDED"
    assert normalize_state("COMPLETED", "1:0") == "FAILED"
    assert normalize_state("OUT_OF_MEMORY", "0:125") == "FAILED"


def test_failure_vocabulary_retains_portable_contract():
    assert FailureClass.NONE.value == "none"
    assert FailureClass.RESOURCE.value == "resource"
