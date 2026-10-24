"""One presentation contract for real terminal, active and blocked job shapes."""
from copy import deepcopy
import shlex

import pytest

from ml_exp_client.presentation import experiment_summary, operation_summary, progress_summary, status_summary


def status(**changes):
    # Reduced from the read-only Slurm and DLC completed/released examples.
    value = {"project": "demo", "run_id": "trial", "attempt_id": "attempt-001",
             "preparation": {"status": "NOT_OBSERVED", "phase": None},
             "scheduler": {"state": "SUCCEEDED", "stale": False, "observed_at": 1792051748.25},
             "training": {"status": "COMPLETED", "exit_code": 0, "process_state": None, "stale": True},
             "publication": {"status": "RELEASED", "phase": "RELEASED", "download_available": False},
             "verification": {"status": "SHA256_VERIFIED"}}
    value.update(changes)
    return value


def test_completed_released_results_do_not_become_failure_or_training_retry():
    value = status(model={"state": "step 8", "detail": {"step": 8}, "as_of": 1792051676.24, "stale": False},
                   progress={"phase": "SUCCEEDED", "queue": {"as_of": 1792051748.25, "stale": False}})
    original = deepcopy(value)
    text = status_summary(value)
    assert "Scheduler: SUCCEEDED" in text and "Training: COMPLETED" in text and "exit 0" in text
    assert "Model: step 8; observed 2026-" in text
    assert "Results: RELEASED; download unavailable" in text and "Verification: SHA256_VERIFIED" in text
    assert "FAILED" not in text and "retry" not in text.lower() and "Waiting:" not in text
    assert value == original
    for backend in ("slurm", "sensecore", "pai-dlc"):
        assert status_summary({**value, "backend": backend}) == text


@pytest.mark.parametrize("scheduler_state", ["RUNNING", "CANCELLED", "SUCCEEDED"])
def test_scheduler_state_never_invents_training_or_model_progress(scheduler_state):
    # RUNNING is the active DLC example; CANCELLED is the unbound CCI/ACP example.
    text = status_summary(status(scheduler={"state": scheduler_state}, training={"status": "UNKNOWN"},
        publication={"status": "NOT_COLLECTED", "download_available": False}, verification={"status": "PENDING"}))
    assert "Scheduler: " + scheduler_state in text
    assert "Training: UNKNOWN; process not observed" in text and "Model: NOT_OBSERVED" in text
    assert "recent progress not observed" in text and "Results: NOT_COLLECTED" in text
    assert "Training: RUNNING" not in text and "Training: FAILED" not in text


def test_real_queue_blocker_is_visible_and_unknown_eta_is_not_invented():
    reason = "Current Limit is CPU: 200 limited by Role PAI.AlgoDeveloper.GPU: 8 limited by Role PAI.AlgoDeveloper."
    queue = {"reason": reason, "provider_code": "CheckSelfQuotaFailed", "position": 1,
             "blocker": {"CodeType": "ConfigRule", "Code": "roleMaximumResource", "Reason": reason},
             "as_of": "2026-10-24T00:00:00Z", "stale": True, "estimated_start_at": None}
    text = status_summary(status(scheduler={"state": "QUEUED", "queue": queue}, training={"status": "UNKNOWN"}))
    assert reason in text and "roleMaximumResource" in text and "CheckSelfQuotaFailed" in text
    assert "position 1" in text and "stale evidence" in text and "ETA" not in text
    assert "Waiting: reason not observed" in status_summary(status(scheduler={"state": "QUEUED"}))


def test_preparation_no_progress_and_failed_collection_remain_separate():
    value = status(preparation={"status": "EXECUTING", "phase": "BUILDING_IMAGE"},
        scheduler={"state": "NOT_SUBMITTED"}, progress={"phase": "WAITING_BUILDER", "seconds_since_progress": 121,
            "no_progress_warning": True, "diagnostic": "Inspect the original build before any retry"},
        publication={"status": "FAILED", "download_available": False,
            "diagnostic": "CPU_JOB_EXITED_WITHOUT_RECEIPT", "error": {"status": 409, "error_code": "FROZEN_IDENTITY_INVALID"}})
    text = status_summary(value)
    assert "Stage: WAITING_BUILDER" in text and "Preparation: EXECUTING" in text
    assert "last recorded 2m 1s ago" in text and "this does not establish failure" in text
    assert "Training: COMPLETED" in text and "Results: FAILED" in text
    assert "CPU_JOB_EXITED_WITHOUT_RECEIPT" in text and "FROZEN_IDENTITY_INVALID" in text


def test_data_transfer_counts_rate_recency_and_errors_survive_formatting():
    value = {"phase": "RUNNING", "message": "Scheduler reports RUNNING", "input_delivery": {
        "phase": "DOWNLOADING", "received_bytes": 1048576, "expected_bytes": 2097152,
        "bytes_per_second": 524288, "seconds_since_progress": 5,
        "at": "2026-10-24T00:00:00Z", "retries": 2, "http_status": 503, "errno": 28}}
    original = deepcopy(value)
    text = progress_summary({"progress": value})
    assert "Input evidence: DOWNLOADING; 1.0 MiB / 2.0 MiB; 512.0 KiB/s" in text
    assert "last progress 5s ago" in text and "retries=2" in text and "http_status=503" in text and "errno=28" in text
    assert value == original
    assert "Input evidence: DOWNLOADING" in status_summary(status(input_delivery=value["input_delivery"]))


def test_empty_and_invalid_numbers_preserve_unknowns_and_zero():
    assert "Attempt not observed" in status_summary({})
    assert "Stage: UNKNOWN" in progress_summary({})
    for bad in (-1, True, float("nan"), float("inf"), "1"):
        text = progress_summary({"seconds_since_progress": bad, "input_delivery": {
            "phase": "VERIFYING", "received_bytes": 0, "expected_bytes": bad, "bytes_per_second": bad}})
        assert "recent progress not observed" in text and "0.0 B / unknown" in text and "/s" not in text
    assert "last recorded 0s ago" in progress_summary({"seconds_since_progress": 0})
    assert "last recorded 1h 1m ago" in progress_summary({"seconds_since_progress": 3660})
    assert "observed 1970-01-01T00:00:00+00:00" in status_summary(status(model={"step": 0, "observed_at": 0}))
    assert "observed" not in status_summary(status(model={"step": 1, "observed_at": 10**19})).split("Model: ", 1)[1].splitlines()[0]


def test_next_actions_are_explicit_shell_quoted_and_never_executed():
    argv = ["ml-exp", "run", "retry", "--state", "state with spaces.json", "--execute"]
    text = status_summary(status(next_actions=[{"kind": "EXECUTE_SAVED_PLAN", "reason": "Review the exact Attempt first",
        "argv": argv, "requires_execution": True}, {"kind": "OBSERVE", "argv": ["ml-exp", "run", "status"]}]))
    assert shlex.join(argv) in text and "explicit execution required" in text and "Next: OBSERVE" in text
    assert shlex.split(text.splitlines()[-3].strip()) == argv
    assert "Next: OBSERVE" in status_summary(status(next_action="OBSERVE", next_actions=[None]))


@pytest.mark.parametrize("argv", [[], "ml-exp retry", ["ml-exp", None], ["ml-exp", ""],
                                  ["ml-exp", "x\n--execute"], ["ml-exp", "x" * 2050], ["x"] * 65])
def test_invalid_or_long_commands_are_not_truncated_into_copyable_commands(argv):
    text = status_summary(status(next_actions=[{"kind": "INSPECT", "argv": argv}]))
    assert "Command unavailable; inspect JSON details." in text


def test_text_is_bounded_and_terminal_control_characters_are_removed():
    value = status(next_actions=[{"kind": "OBSERVE", "reason": "\x1b[2J\r\n" + "x" * 10000}] * 100,
                   scheduler={"state": "RUNNING", "reason": "one\nFAKE-LINE\x1b[31m"})
    text = status_summary(value)
    assert "\x1b" not in text and "\r" not in text and "\nFAKE-LINE" not in text
    assert len(text) < 4000 and text.count("Next:") == 4 and "…" in text


def operation(**changes):
    value = {"operation": "retry", "project": "demo", "run_id": "trial", "source_attempt_id": "attempt-001",
        "request": {"max_gpu_hours": 2}, "action": {"action_id": "action-" + "a" * 16, "attempt_id": "attempt-002",
            "ready": True, "execution": {"status": "PREPARED"}, "gates": [{"name": "budget", "status": "PASS"}],
            "preflight_summary": {"requested_gpu_hours": 1}},
        "retry_semantics": "New Attempt in the same frozen Run; no checkpoint is selected or changed.",
        "next_action": "EXECUTE_SAVED_PLAN", "execute_argv": ["ml-exp", "run", "retry", "--state", "retry.json", "--execute"]}
    value.update(changes)
    return value


def test_retry_review_shows_identity_budget_semantics_and_explicit_execution():
    value = operation()
    original = deepcopy(value)
    text = operation_summary(value)
    assert "attempt-001 → attempt-002" in text and "GPU hours: requested 1; cap 2" in text
    assert "Gates: PASS" in text and "Ready: yes" in text and value["retry_semantics"] in text
    assert "Next: EXECUTE_SAVED_PLAN" in text and "Execute after review: ml-exp run retry" in text
    assert value == original


def test_blocked_cancel_and_unknown_review_have_no_invented_execution():
    value = operation(operation="cancel", request={}, retry_semantics=None, execute_argv=None, next_action="INSPECT_ACTION")
    value["action"].update(attempt_id="attempt-001", ready=False,
        gates=[{"name": "snapshot", "status": "FAIL", "detail": "Exact job identity unavailable"},
               {"name": "old_logs", "status": "WARN", "detail": "Logs delayed"}])
    text = operation_summary(value)
    assert "Ready: no" in text and "snapshot: FAIL" in text and "old_logs: WARN" in text
    assert "Exact job identity unavailable" in text and "GPU hours:" in text  # known preflight estimate remains visible
    assert "Execute after review:" not in text and "Retry:" not in text
    assert "Ready: unknown" in operation_summary({}) and "Gates: NOT_OBSERVED" in operation_summary({})


def test_progress_warnings_and_timestamp_are_observational_only():
    text = progress_summary({"phase": "RUNNING", "last_progress_unix": 1792051748.25,
        "stale": True, "no_progress_warning": True, "diagnostic": {"code": "DATA_PENDING", "message": "Read logs"}})
    assert "stale evidence" in text and "Warning:" in text and "DATA_PENDING; Read logs" in text
    assert "FAILED" not in text


def test_input_ready_without_counts_and_large_transfer_units_remain_readable():
    assert "Input evidence: READY" in progress_summary({"input_delivery": {"phase": "READY"}})
    text = progress_summary({"input_delivery": {"phase": "DOWNLOADING", "received_bytes": 2**30,
                                               "expected_bytes": 2**50}})
    assert "1.0 GiB / 1024.0 TiB" in text


def test_coordinated_queue_is_read_from_top_level_for_all_backends():
    queue = {"code": "roleMaximumResource", "reason": "Role GPU limit is 8", "stale": True}
    text = status_summary(status(scheduler={"state": "QUEUED"}, queue=queue))
    assert "Waiting: Role GPU limit is 8" in text and "roleMaximumResource" in text and "stale evidence" in text
    polling = progress_summary({"progress": {"phase": "QUEUED"}, "queue": queue})
    assert "Waiting: Role GPU limit is 8" in polling


@pytest.mark.parametrize("render", [status_summary, progress_summary])
def test_worker_phase_is_a_dated_report_not_a_fresh_training_heartbeat(render):
    worker = {"phase": "TRAINING", "observed_at": "2026-10-24T00:00:00Z", "age_seconds": 7200,
              "source": "record-journal", "event_id": "worker.1", "sequence": 12}
    value = status(progress={"phase": "TRAINING"}, worker_phase=worker, training={"status": "UNKNOWN"})
    text = render(value)
    assert "Stage: TRAINING (last reported worker phase; not a heartbeat)" in text
    assert "observed 2026-10-24T00:00:00Z" in text and "report 2h 0m ago" in text
    assert "Training: COMPLETED" not in text and "Training: RUNNING" not in text
    nested = progress_summary({"progress": {"phase": "RUNNING", "worker_phase": worker}})
    assert "Stage: RUNNING" in nested and "Worker phase: TRAINING; last reported; not a heartbeat" in nested


@pytest.mark.parametrize("nested", [False, True])
def test_collection_polling_shows_training_and_publication_independently(nested):
    collection = {"phase": "UPLOADING", "status": "COLLECTING", "training": {"status": "COMPLETED", "exit_code": 0},
                  "verification": "PENDING", "download_available": False, "diagnostic": "UPLOAD_RETRY_PENDING"}
    value = {"progress": {"phase": "SUCCEEDED", "result_collection": collection}} if nested else collection
    text = progress_summary(value)
    assert "Training: COMPLETED; exit 0" in text and "Results: COLLECTING; download unavailable" in text
    assert "Verification: PENDING" in text and "UPLOAD_RETRY_PENDING" in text
    collection.update(status="AVAILABLE", verification="SHA256_VERIFIED", download_available=True)
    assert "Results: AVAILABLE; download available" in progress_summary(value)
    collection['verification'] = "PENDING"
    assert "download available" not in progress_summary(value)


def test_top_level_collection_and_worker_report_without_date_do_not_guess_age():
    text = progress_summary({"progress": {"phase": "RUNNING"}, "worker_phase": {"phase": "TRAINING"},
                             "result_collection": {"status": "NOT_COLLECTED", "verification": None}})
    assert "Worker phase: TRAINING; last reported; not a heartbeat" in text
    assert "Training: UNKNOWN" in text and "Verification: NOT_OBSERVED" in text and "ago" not in text


def test_experiment_review_preserves_resources_budget_attempt_and_warning_status(tmp_path):
    value = {"project": "demo", "run_id": "trial", "submission": {
        "status": "PREPARED", "attempt_id": "attempt-001", "preflight_summary": {
            "resources": {"gpus": 1, "cpus": 4, "memory_gb": 8, "max_time": "00:10:00", "other": "hidden"},
            "requested_gpu_hours": 0}, "gates": [
            {"name": "okay", "status": "PASS"},
            {"name": "provider_logs", "status": "WARNING", "detail": "Logs delayed"},
            {"name": "optional_metadata", "status": "WARN"},
            {"name": "budget", "status": "FAIL", "detail": "Budget too small"},
            {"name": "storage", "status": "BLOCKED"}, None]}}
    original = deepcopy(value)
    state = tmp_path / "state with spaces.json"
    text = experiment_summary(value, state_file=state, executed=False)
    assert "Experiment: demo/trial" in text and "Submission: PREPARED" in text
    assert "Resources: gpus=1, cpus=4, memory_gb=8, max_time=00:10:00" in text and "hidden" not in text
    assert "GPU hours requested: 0" in text and "Attempt: attempt-001" in text
    assert "Gate: provider_logs: WARNING — Logs delayed" in text and "Gate: optional_metadata: WARN" in text
    assert "Blocked: provider_logs" not in text and "Blocked: optional_metadata" not in text
    assert "Blocked: budget: FAIL" in text and "Blocked: storage: BLOCKED" in text and "okay" not in text
    assert "not been authorized by this command" in text
    assert shlex.split(text.split("Inspect: ", 1)[1].splitlines()[0]) == ["ml-exp", "run", "status", "--state", str(state)]
    assert value == original


@pytest.mark.parametrize("proof,expected", [
    ({"verification": "SHA256_VERIFIED"}, "SHA256_VERIFIED"),
    ({"status": "VERIFIED"}, "VERIFIED"),
    ({"files": {}}, "downloaded; see verification.json")])
def test_experiment_terminal_summary_does_not_infer_additional_verification(proof, expected):
    text = experiment_summary({"result": {"scheduler_state": "SUCCEEDED"}, "download": proof},
                              state_file="trial.state", executed=True)
    assert "Scheduler: SUCCEEDED" in text and "Results: " + expected in text
    assert "not been authorized" not in text and "Submission: NOT_OBSERVED" in text
    assert "Attempt:" not in text and "Resources:" not in text and "GPU hours requested:" not in text


def test_experiment_summary_sanitizes_untrusted_fields_and_limits_notable_gates():
    injection = "\x1b[2J\r\n" + "x" * 5000
    value = {"project": injection, "run_id": injection, "submission": {
        "gates": [{"name": injection, "status": "WARNING", "detail": injection}] * 100,
        "preflight_summary": {"resources": {"cpus": injection}}}}
    text = experiment_summary(value, state_file=injection, executed=False)
    assert "\x1b" not in text and "\r" not in text and "\nxxx" not in text
    assert text.count("Gate:") == 6 and "Blocked:" not in text and len(text) < 7500
    assert "Inspect: Command unavailable; inspect JSON details." in text and "…" in text
    assert "Gate:" not in experiment_summary({"submission": {"gates": None}}, state_file="trial.state", executed=True)


def test_native_identity_and_last_metric_preserve_missing_clock():
    value = {'project': 'demo', 'run_id': 'trial', 'attempt_id': 'attempt-001',
             'scheduler': {'state': 'RUNNING', 'backend': 'pai-dlc', 'job_id': 'owned-job', 'native_state': 'Running'},
             'model': {'state': 'step 2927', 'step': 2927, 'last_metric': {
                 'name': 'train_loss', 'value': 3.4, 'unit': 'nats/token', 'status': 'VALID'}}}
    shown = status_summary(value)
    assert 'Backend: pai-dlc; job owned-job; native Running' in shown
    assert 'Last metric: train_loss=3.4 nats/token; VALID' in shown
    assert 'freshness is unknown' in shown
    value['model']['observed_at'] = '2026-10-24T15:00:00Z'
    assert 'Metric time not recorded' not in status_summary(value)
