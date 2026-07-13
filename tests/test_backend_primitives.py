import pytest

from experiment_control.backends.base import BackendRegistry
from experiment_control.backends.wyd import (
    checkpoint_probe_command,
    parse_accounting,
    scheduler_job_name,
)


def test_slurm_accounting_contract_normalizes_exit_code():
    result = parse_accounting(
        "42|run|accelerator|COMPLETED|00:10:00|1:0\n",
        job_id="42",
        run_id="run",
        partition="accelerator",
    )
    assert result["state"] == "FAILED"
    assert result["exit_code"] == "1:0"


def test_slurm_queue_status_preserves_pending_reason():
    result = parse_accounting(
        "42|run|accelerator|PENDING|00:00|0:0|Resources\n",
        job_id="42",
        run_id="run",
        partition="accelerator",
    )
    assert result["state"] == "QUEUED"
    assert result["reason"] == "Resources"
    assert result["detail"] == {"pending_reason": "Resources"}


def test_backend_registry_rejects_unknown_kind():
    registry = BackendRegistry()
    with pytest.raises(ValueError, match="unsupported"):
        registry.get("other")


def test_attempt_qualified_slurm_name_is_bounded_and_deterministic():
    name = scheduler_job_name("r" * 128, "attempt-123")
    assert len(name) <= 128
    assert name == scheduler_job_name("r" * 128, "attempt-123")
    assert name != scheduler_job_name("r" * 128, "attempt-124")


def test_checkpoint_probe_validates_marker_metadata_without_shell_interpolation():
    command = checkpoint_probe_command("/shared/project/run")
    assert "python3 -c" in command
    assert "metadata.get" in command
    assert command.endswith("/shared/project/run")
