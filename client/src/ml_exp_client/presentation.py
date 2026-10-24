"""Bounded human-readable views of already validated experiment evidence.

These functions never query a provider, infer success, or execute an action.
All backends use the same fields. Machine-readable responses stay unchanged.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
import shlex


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _text(value, default="UNKNOWN", *, limit=400):
    if not isinstance(value, (str, int, float, bool)):
        return default
    value = " ".join("".join(char if char.isprintable() else " " for char in str(value)).split())
    return (value[:limit] + "…" if len(value) > limit else value) or default


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _bytes(value):
    if not _number(value):
        return "unknown"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    index = 0
    while value >= 1024 and index < len(units) - 1:
        value /= 1024
        index += 1
    return f"{value:.1f} {units[index]}"


def _age(value):
    if not _number(value):
        return None
    seconds = int(value)
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s ago"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m ago"


def _observed(value):
    timestamp = next((value.get(key) for key in ("observed_at", "as_of", "at") if value.get(key) is not None), None)
    if _number(timestamp):
        try:
            timestamp = datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds")
        except (OverflowError, OSError, ValueError):
            timestamp = None
    parts = ["observed " + _text(timestamp)] if timestamp is not None else []
    if value.get("stale") is True:
        parts.append("stale evidence")
    return "; ".join(parts)


def _diagnostic(value):
    if isinstance(value, dict):
        parts = [_text(value[key]) for key in ("code", "reason", "message", "error_code") if value.get(key)]
        return "; ".join(parts)
    return _text(value, "")


def _command(argv):
    if (not isinstance(argv, list) or not argv or len(argv) > 64
            or any(not isinstance(arg, str) or not arg or not arg.isprintable() for arg in argv)):
        return None
    # Never truncate a command: that could change the exact target or operation.
    command = shlex.join(argv)
    return command if len(command) <= 2048 else None


def _next_lines(value):
    actions = value.get("next_actions")
    lines = []
    if isinstance(actions, list):
        for item in actions[:4]:
            if not isinstance(item, dict):
                continue
            label = "Next: " + _text(item.get("kind"))
            if item.get("requires_execution") is True:
                label += " (explicit execution required)"
            if item.get("reason"):
                label += " — " + _text(item["reason"])
            lines.append(label)
            if item.get("argv") is not None:
                lines.append("  " + (_command(item["argv"]) or "Command unavailable; inspect JSON details."))
    if not lines:
        lines.append("Next: " + _text(value.get("next_action"), "inspect existing evidence; no action specified"))
    return lines


def _queue_lines(queue, reason=None, *, waiting=False):
    blocker = _mapping(queue.get("blocker"))
    reason = queue.get("reason") or blocker.get("Reason") or reason
    codes = [queue.get("code") or blocker.get("Code"), queue.get("provider_code")]
    details = [_text(code) for code in codes if code]
    if _number(queue.get("position")):
        details.append("position " + _text(queue["position"]))
    if not reason and not details and not waiting:
        return []
    observed = _observed(queue)
    if observed:
        details.append(observed)
    return [("Waiting: " if waiting else "Scheduler reason: ") + _text(reason, "reason not observed")
            + (" (" + "; ".join(details) + ")" if details else "")]


def _progress_lines(progress):
    age = _age(progress.get("seconds_since_progress"))
    observed = _observed({"observed_at": progress.get("last_progress_unix"), "stale": progress.get("stale")})
    parts = (["last recorded " + age] if age else []) + ([observed] if observed else [])
    lines = ["Progress: " + ("; ".join(parts) or "recent progress not observed")]
    if progress.get("no_progress_warning") is True:
        lines.append("Warning: no recorded progress; this does not establish failure.")
    return lines


def _stage_line(phase, worker):
    line = "Stage: " + _text(phase)
    if worker.get("phase") and phase == worker["phase"]:
        line += " (last reported worker phase; not a heartbeat)"
    return line


def _worker_lines(worker):
    if not worker:
        return []
    parts = [_text(worker.get("phase")), "last reported; not a heartbeat"]
    observed = _observed(worker)
    if observed:
        parts.append(observed)
    age = _age(worker.get("age_seconds"))
    if age:
        parts.append("report " + age)
    return ["Worker phase: " + "; ".join(parts)]


def _training_line(training, *, process=False):
    parts = [_text(training.get("status"))]
    if process:
        parts.append("process " + _text(training.get("process_state"), "not observed"))
    if training.get("exit_code") is not None:
        parts.append("exit " + _text(training["exit_code"]))
    if process and _observed(training):
        parts.append("process evidence: " + _observed(training))
    return "Training: " + "; ".join(parts)


def _result_lines(publication, verification):
    verified = verification.get("status") if isinstance(verification, dict) else verification
    available = (publication.get("status") == "AVAILABLE" and verified == "SHA256_VERIFIED"
                 and publication.get("download_available") is True)
    return ["Results: " + _text(publication.get("status"), "NOT_OBSERVED")
            + "; download " + ("available" if available else "unavailable"),
            "Verification: " + _text(verified, "NOT_OBSERVED")]


def _input_line(value):
    if not value:
        return None
    parts = [_text(value.get("phase"))]
    if _number(value.get("received_bytes")):
        parts.append(_bytes(value["received_bytes"]) + " / " + _bytes(value.get("expected_bytes")))
    if _number(value.get("bytes_per_second")):
        parts.append(_bytes(value["bytes_per_second"]) + "/s")
    age = _age(value.get("seconds_since_progress"))
    if age:
        parts.append("last progress " + age)
    observed = _observed(value)
    if observed:
        parts.append(observed)
    for field in ("code", "failed_phase", "http_status", "errno", "retries", "retry_count"):
        if value.get(field) is not None:
            parts.append(field + "=" + _text(value[field]))
    return "Input evidence: " + "; ".join(parts)


def status_summary(value: dict) -> str:
    """Render independently reported scheduler, training and result states."""
    preparation = _mapping(value.get("preparation"))
    scheduler = _mapping(value.get("scheduler"))
    training = _mapping(value.get("training"))
    publication = _mapping(value.get("publication"))
    verification = _mapping(value.get("verification"))
    progress = _mapping(value.get("progress"))
    model = _mapping(value.get("model"))
    worker = _mapping(value.get("worker_phase")) or _mapping(progress.get("worker_phase"))
    lines = ["Run: " + _text(value.get("project")) + "/" + _text(value.get("run_id"))
             + " · " + _text(value.get("attempt_id"), "Attempt not observed")]
    lines.append(_stage_line(progress.get("phase") or preparation.get("phase") or scheduler.get("state"), worker))
    if preparation.get("status") not in (None, "NOT_OBSERVED"):
        lines.append("Preparation: " + _text(preparation["status"]))
    lines.append("Scheduler: " + _text(scheduler.get("state"))
                 + ("; " + _observed(scheduler) if _observed(scheduler) else ""))
    lines.append("Backend: " + _text(scheduler.get("backend"), "not observed")
                 + "; job " + _text(scheduler.get("job_id"), "not observed")
                 + "; native " + _text(scheduler.get("native_state"), "not observed"))
    lines.extend(_queue_lines(_mapping(value.get("queue")) or _mapping(scheduler.get("queue")) or _mapping(progress.get("queue")),
                              scheduler.get("reason"), waiting=scheduler.get("state") in {"QUEUED", "PENDING"}))
    lines.append(_training_line(training, process=True))
    lines.extend(_worker_lines(worker))
    step = model.get("step", _mapping(model.get("detail")).get("step"))
    measurement = "step " + _text(step) if _number(step) else _text(model.get("state"), "NOT_OBSERVED")
    lines.append("Model: " + measurement + ("; " + _observed(model) if _observed(model) else ""))
    metric = _mapping(model.get("last_metric"))
    if metric:
        lines.append("Last metric: " + _text(metric.get("name")) + "=" + _text(metric.get("value"), "missing")
                     + " " + _text(metric.get("unit")) + "; " + _text(metric.get("status"), "status not recorded"))
        if model.get("observed_at") is None:
            lines.append("Metric time not recorded; freshness is unknown.")
    lines.extend(_progress_lines(progress))
    input_line = _input_line(_mapping(value.get("input_delivery")) or _mapping(progress.get("input_delivery")))
    if input_line:
        lines.append(input_line)
    lines.extend(_result_lines(publication, verification))
    for item in (progress.get("diagnostic"), publication.get("diagnostic"), publication.get("error")):
        diagnostic = _diagnostic(item)
        if diagnostic:
            lines.append("Diagnostic: " + diagnostic)
    lines.extend(_next_lines(value))
    return "\n".join(lines)


def progress_summary(value: dict) -> str:
    """Render a polling update without treating scheduler state as a heartbeat."""
    progress = _mapping(value.get("progress", value))
    worker = _mapping(value.get("worker_phase")) or _mapping(progress.get("worker_phase"))
    lines = [_stage_line(progress.get("phase", progress.get("status")), worker)]
    if progress.get("message"):
        lines.append(_text(progress["message"]))
    lines.extend(_queue_lines(_mapping(value.get("queue")) or _mapping(progress.get("queue")),
                              waiting=progress.get("phase") in {"QUEUED", "PENDING"}))
    lines.extend(_progress_lines(progress))
    lines.extend(_worker_lines(worker))
    input_line = _input_line(_mapping(progress.get("input_delivery")))
    if input_line:
        lines.append(input_line)
    diagnostic = _diagnostic(progress.get("diagnostic"))
    if diagnostic:
        lines.append("Diagnostic: " + diagnostic)
    collection = _mapping(progress.get("result_collection")) or _mapping(value.get("result_collection"))
    if not collection and any(key in progress for key in ("training", "verification", "download_available")):
        collection = progress
    if collection:
        lines.append(_training_line(_mapping(collection.get("training"))))
        lines.extend(_result_lines(collection, collection.get("verification")))
        diagnostic = _diagnostic(collection.get("diagnostic"))
        if diagnostic:
            lines.append("Diagnostic: " + diagnostic)
    return "\n".join(lines)


def experiment_summary(value: dict, *, state_file, executed: bool, client_config=None) -> str:
    """Summarize a workflow without exposing raw terminal text or hiding warnings."""
    submission = _mapping(value.get("submission"))
    preflight = _mapping(submission.get("preflight_summary"))
    resources = _mapping(preflight.get("resources"))
    state = str(state_file)
    lines = ["Experiment: " + _text(value.get("project")) + "/" + _text(value.get("run_id")),
             "State: " + _text(state),
             "Submission: " + _text(submission.get("status"), "NOT_OBSERVED")]
    if resources:
        lines.append("Resources: " + ", ".join(key + "=" + _text(resources[key])
                     for key in ("gpus", "cpus", "memory_gb", "max_time") if key in resources))
    if preflight.get("requested_gpu_hours") is not None:
        lines.append("GPU hours requested: " + _text(preflight["requested_gpu_hours"]))
    gates = submission.get("gates")
    gates = [item for item in gates if isinstance(item, dict)] if isinstance(gates, list) else []
    notable = [gate for gate in gates if gate.get("status") != "PASS"]
    for gate in notable[:6]:
        label = "Blocked: " if gate.get("status") in {"FAIL", "BLOCKED"} else "Gate: "
        lines.append(label + _text(gate.get("name")) + ": " + _text(gate.get("status"))
                     + (" — " + _text(gate["detail"]) if gate.get("detail") else ""))
    if submission.get("attempt_id"):
        lines.append("Attempt: " + _text(submission["attempt_id"]))
    result = _mapping(value.get("result"))
    if result:
        lines.append("Scheduler: " + _text(result.get("scheduler_state")))
    proof = _mapping(value.get("download"))
    if proof:
        lines.append("Results: " + _text(proof.get("verification", proof.get("status")),
                                        "downloaded; see verification.json"))
    prefix = ["ml-exp"] + (["--config", str(client_config)] if client_config is not None else [])
    command = _command([*prefix, "run", "status", "--state", state])
    lines.append("Inspect: " + (command or "Command unavailable; inspect JSON details."))
    if not executed:
        lines.append("Training has not been authorized by this command. Review the saved preparation before --execute.")
    return "\n".join(lines)


def operation_summary(value: dict) -> str:
    """Display a saved retry/cancel review without changing or executing it."""
    action = _mapping(value.get("action"))
    execution = _mapping(action.get("execution"))
    request = _mapping(value.get("request"))
    preflight = _mapping(action.get("preflight_summary"))
    lines = ["Operation: " + _text(value.get("operation")) + " — "
             + _text(value.get("project")) + "/" + _text(value.get("run_id")),
             "Attempts: " + _text(value.get("source_attempt_id")) + " → " + _text(action.get("attempt_id")),
             "Action: " + _text(action.get("action_id")) + "; " + _text(execution.get("status")),
             "Ready: " + ("yes" if action.get("ready") is True else "no" if action.get("ready") is False else "unknown")]
    cap, requested = request.get("max_gpu_hours"), preflight.get("requested_gpu_hours")
    if _number(cap) or _number(requested):
        lines.append("GPU hours: requested " + (_text(requested) if _number(requested) else "unknown")
                     + "; cap " + (_text(cap) if _number(cap) else "unknown"))
    gates = action.get("gates")
    gates = [item for item in gates if isinstance(item, dict)] if isinstance(gates, list) else []
    notable = [gate for gate in gates if gate.get("status") != "PASS"]
    lines.append("Gates: " + ("PASS" if gates and not notable else "inspect" if gates else "NOT_OBSERVED"))
    for gate in notable[:6]:
        lines.append("  " + _text(gate.get("name")) + ": " + _text(gate.get("status"))
                     + (" — " + _text(gate["detail"]) if gate.get("detail") else ""))
    if value.get("retry_semantics"):
        lines.append("Retry: " + _text(value["retry_semantics"]))
    lines.extend(_next_lines(value))
    if value.get("execute_argv") is not None:
        lines.append("Execute after review: " + (_command(value["execute_argv"]) or "Command unavailable; inspect JSON details."))
    return "\n".join(lines)
