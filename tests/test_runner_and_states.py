import subprocess
import sys

import pytest

from experiment_control.backends.wyd import normalize_state
from experiment_control.runner import CommandResult, SubprocessRunner
from experiment_control.states import FailureClass, classify_failure


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


def test_failure_classifier_does_not_hide_resource_or_model_failures():
    assert FailureClass.NONE.value == "none"
    assert classify_failure("unrecognized failure") is FailureClass.UNKNOWN
    assert classify_failure("CUDA out of memory") is FailureClass.RESOURCE
    assert classify_failure("loss became NaN") is FailureClass.MODEL
    assert classify_failure("TLS EOF") is FailureClass.TRANSPORT
    assert classify_failure('{"live_logs_expired": true}') is FailureClass.TRANSPORT
    assert classify_failure("worker was preempted") is FailureClass.PREEMPTION
    assert classify_failure("node failure") is FailureClass.SCHEDULER
    assert classify_failure("required metric is missing") is FailureClass.EVALUATION
    assert (
        classify_failure("ModuleNotFoundError: No module named 'pkg'")
        is FailureClass.CONFIGURATION
    )


@pytest.mark.parametrize(
    "message",
    [
        "ssh: connect to host compute.example port 22: Connection timed out",
        "Connection timed out during banner exchange",
        "TLS ClientHello handshake timeout",
    ],
)
def test_connection_timeouts_are_transport_failures(message):
    assert classify_failure(message) is FailureClass.TRANSPORT


@pytest.mark.parametrize(
    "message",
    [
        "TIMEOUT",
        "Job timed out after reaching its wall time",
        "slurmstepd: error: DUE TO TIME LIMIT",
    ],
)
def test_scheduler_wall_time_is_a_resource_failure(message):
    assert classify_failure(message) is FailureClass.RESOURCE
