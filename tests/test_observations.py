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
