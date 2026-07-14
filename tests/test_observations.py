from experiment_control.observations import merge_terminal_observation


def test_terminal_merge_retains_metric_and_tails_without_reviving_running_state():
    previous = {
        "run_id": "smoke",
        "scheduler_state": "RUNNING",
        "worker_state": "ALLOCATED",
        "process_state": "RUNNING",
        "model_state": "OBSERVED",
        "model_observed": True,
        "latest_metric": {"step": 1539, "train_loss": 3.5847},
        "metric_source": "/data/run/train_metrics.jsonl",
        "process_evidence": {
            "observed": True,
            "sources": {"stdout": "/remote/stdout.log"},
            "stdout_tail": ["Step 1539 train_loss=3.5847"],
            "stderr_tail": [],
        },
    }
    current = {
        "run_id": "smoke",
        "scheduler_state": "CANCELLED",
        "worker_state": "RELEASED",
        "process_state": "UNKNOWN",
        "model_state": "NOT_OBSERVED",
        "model_observed": False,
        "latest_metric": None,
        "process_evidence": {
            "observed": False, "stdout_tail": [], "stderr_tail": [],
        },
        "evidence_outcome": "INCONCLUSIVE",
        "evidence_unavailable_reason": "cancelled_before_observation",
    }

    merged = merge_terminal_observation(previous, current)

    assert merged["scheduler_state"] == "CANCELLED"
    assert merged["worker_state"] == "RELEASED"
    assert merged["process_state"] == "UNKNOWN"
    assert merged["latest_metric"]["step"] == 1539
    assert merged["model_state"] == "OBSERVED"
    assert merged["process_evidence"]["stdout_tail"] == [
        "Step 1539 train_loss=3.5847"
    ]
    assert merged["process_evidence"]["retained"] is True
    assert merged["evidence_outcome"] == "OBSERVED"
    assert merged["evidence_unavailable_reason"] is None
    assert merged["retained_evidence"]["source_scheduler_state"] == "RUNNING"


def test_terminal_merge_prefers_newer_current_metric():
    merged = merge_terminal_observation(
        {"run_id": "run", "latest_metric": {"step": 10, "loss": 2.0}},
        {
            "run_id": "run", "scheduler_state": "SUCCEEDED",
            "latest_metric": {"step": 11, "loss": 1.9},
        },
    )
    assert merged["latest_metric"] == {"step": 11, "loss": 1.9}
    assert "retained_evidence" not in merged


def test_terminal_merge_never_crosses_run_identity():
    current = {"run_id": "new", "scheduler_state": "FAILED", "latest_metric": None}
    assert merge_terminal_observation(
        {"run_id": "old", "latest_metric": {"step": 99}}, current,
    ) == current


def test_merge_returns_current_when_previous_is_absent_or_state_is_not_terminal():
    current = {"run_id": "run", "scheduler_state": "RUNNING"}
    assert merge_terminal_observation(None, current) == current
    assert merge_terminal_observation(
        {"run_id": "run", "latest_metric": {"step": 1}}, current,
    ) == current


def test_terminal_merge_handles_unstepped_metrics_artifacts_and_present_fields():
    merged = merge_terminal_observation(
        {
            "run_id": "run",
            "latest_metric": "legacy metric",
            "artifacts": {"old": "checkpoint-1", "shared": "old"},
            "checkpoint_path": "/old/checkpoint",
        },
        {
            "run_id": "run",
            "scheduler_state": "FAILED",
            "latest_metric": {"loss": 2.0},
            "artifacts": {"shared": "new", "current": "report"},
            "checkpoint_path": "/new/checkpoint",
        },
    )

    assert merged["latest_metric"] == {"loss": 2.0}
    assert merged["artifacts"] == {
        "old": "checkpoint-1", "shared": "new", "current": "report",
    }
    assert merged["checkpoint_path"] == "/new/checkpoint"
    assert merged["retained_evidence"]["fields"] == ["artifacts"]

    unchanged = {
        "run_id": "run",
        "scheduler_state": "SUCCEEDED",
        "artifacts": {"shared": "same"},
    }
    assert merge_terminal_observation(
        {"run_id": "run", "artifacts": {"shared": "same"}}, unchanged,
    ) == unchanged


def test_terminal_merge_process_sources_cover_retained_and_no_retention_paths():
    retained = merge_terminal_observation(
        {
            "run_id": "run",
            "scheduler_state": "RUNNING",
            "process_evidence": {
                "stdout_tail": ["old stdout"],
                "stderr_tail": [],
                "sources": {"stdout": "/old/stdout"},
            },
        },
        {
            "run_id": "run",
            "scheduler_state": "CANCELLED",
            "process_evidence": {
                "stdout_tail": [],
                "stderr_tail": ["new stderr"],
                "sources": {"stderr": "/new/stderr"},
            },
        },
    )
    assert retained["process_evidence"]["sources"] == {
        "stdout": "/old/stdout", "stderr": "/new/stderr",
    }
    assert retained["process_evidence"]["retained_streams"] == ["stdout_tail"]

    current = {
        "run_id": "run",
        "scheduler_state": "SUCCEEDED",
        "process_evidence": {"stdout_tail": ["current"], "stderr_tail": []},
    }
    assert merge_terminal_observation(
        {
            "run_id": "run",
            "process_evidence": {"stdout_tail": ["old"], "stderr_tail": []},
        },
        current,
    ) == current
