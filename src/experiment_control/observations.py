"""Field-level merging for durable observations of one exact Attempt."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


TERMINAL_SCHEDULER_STATES = {
    "SUCCEEDED", "FAILED", "PREEMPTED", "CANCELLED",
}

_MODEL_EVIDENCE_FIELDS = {
    "latest_metric",
    "metric_log_lines",
    "model_observed",
    "step",
    "optimizer_step",
    "latest_completed_checkpoint",
    "latest_completed_checkpoint_step",
    "artifacts",
    "evaluation_variants",
}


def _present(value: Any) -> bool:
    return value is not None and value is not False and value not in ("", [], {})


def _model_field(name: str) -> bool:
    if name in _MODEL_EVIDENCE_FIELDS:
        return True
    if name == "model_state":
        return False
    return name.startswith((
        "latest_metric_", "metric_", "checkpoint_", "artifact_", "evaluation_",
        "model_evidence_",
    ))


def _metric_step(value: Any) -> float | None:
    if not isinstance(value, dict):
        return None
    step = value.get("step", value.get("optimizer_step"))
    return float(step) if isinstance(step, (int, float)) else None


def _new_metric_is_stronger(previous: Any, current: Any) -> bool:
    if not _present(current):
        return False
    previous_step = _metric_step(previous)
    current_step = _metric_step(current)
    if previous_step is not None and current_step is not None:
        return current_step >= previous_step
    return True


def merge_terminal_observation(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> dict[str, Any]:
    """Keep prior strong evidence without reviving stale lifecycle state.

    The current terminal scheduler/worker/process fields are always authoritative.
    Only evidence fields omitted by a terminal backend observation may be retained,
    and every retention is explicitly marked with its source state.
    """
    result = deepcopy(current)
    if not previous:
        return result
    if str(current.get("scheduler_state") or "").upper() not in TERMINAL_SCHEDULER_STATES:
        return result
    previous_run = previous.get("run_id")
    current_run = current.get("run_id")
    if previous_run and current_run and previous_run != current_run:
        return result

    retained: list[str] = []
    for name, value in previous.items():
        if not _model_field(name) or not _present(value):
            continue
        current_value = result.get(name)
        if name == "latest_metric":
            if _new_metric_is_stronger(value, current_value):
                continue
            result[name] = deepcopy(value)
            retained.append(name)
            continue
        if name == "artifacts" and isinstance(value, dict) and isinstance(current_value, dict):
            merged_artifacts = deepcopy(value)
            merged_artifacts.update(deepcopy(current_value))
            if merged_artifacts != current_value:
                result[name] = merged_artifacts
                retained.append(name)
            continue
        if _present(current_value):
            continue
        result[name] = deepcopy(value)
        retained.append(name)

    previous_process = previous.get("process_evidence")
    current_process = result.get("process_evidence")
    if isinstance(previous_process, dict):
        merged_process = deepcopy(current_process) if isinstance(current_process, dict) else {}
        retained_streams: list[str] = []
        for stream in ("stdout_tail", "stderr_tail"):
            if _present(merged_process.get(stream)) or not _present(previous_process.get(stream)):
                continue
            merged_process[stream] = deepcopy(previous_process[stream])
            retained_streams.append(stream)
        if retained_streams:
            previous_sources = previous_process.get("sources")
            current_sources = merged_process.get("sources")
            sources = dict(previous_sources) if isinstance(previous_sources, dict) else {}
            if isinstance(current_sources, dict):
                sources.update(current_sources)
            merged_process.update({
                "observed": True,
                "sources": sources,
                "retained": True,
                "retained_streams": retained_streams,
            })
            result["process_evidence"] = merged_process
            retained.extend(f"process_evidence.{name}" for name in retained_streams)

    model_retained = any(not name.startswith("process_evidence.") for name in retained)
    if model_retained:
        result["model_observed"] = True
        result["model_state"] = "OBSERVED"
    if retained:
        result["evidence_unavailable_reason"] = None
        result["evidence_outcome"] = "OBSERVED"
        result["retained_evidence"] = {
            "retained": True,
            "fields": sorted(retained),
            "reason": "terminal observation omitted previously observed evidence",
            "source_scheduler_state": previous.get("scheduler_state"),
        }
    return result
