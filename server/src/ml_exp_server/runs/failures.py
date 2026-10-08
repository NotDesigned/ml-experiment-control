"""Shared ExperimentServerApplication evidence and planning helpers."""
from __future__ import annotations
import hashlib
import json
import re
from typing import Any
from ..ingest.runscan import parse_iso_ts

_OMITTED_EVIDENCE_KEYS = {
    "stdout_tail", "stderr_tail", "raw_stdout", "raw_stderr", "startup_script",
}


def _reviewable_unknown_retry(
    decision: dict[str, Any], resource_approval: str,
    *, operator_reason: str | None = None,
) -> bool:
    """Allow an unknown failure to reach review, never automatic execution."""
    action = str(decision.get("action") or "").upper()
    failure = str(decision.get("failure_class") or "").lower()
    allowed = decision.get("retries_allowed")
    used = decision.get("retries_used", 0)
    return (
        action in {"REVIEW_RETRY", "DO_NOT_RETRY"}
        and failure == "unknown"
        and isinstance(allowed, int)
        and not isinstance(allowed, bool)
        and isinstance(used, int)
        and not isinstance(used, bool)
        and used < allowed
        and resource_approval == "review_exact"
        and (operator_reason is None or bool(operator_reason.strip()))
    )


def compact_evidence(value: Any, *, depth: int = 0) -> Any:
    if depth >= 8:
        return "[nested evidence omitted]"
    if isinstance(value, dict):
        return {
            str(key): compact_evidence(item, depth=depth + 1)
            for key, item in value.items()
            if str(key) not in _OMITTED_EVIDENCE_KEYS
        }
    if isinstance(value, list):
        compact = [compact_evidence(item, depth=depth + 1) for item in value[:40]]
        if len(value) > 40:
            compact.append(f"[{len(value) - 40} additional records omitted]")
        return compact
    if isinstance(value, str) and len(value) > 2000:
        return value[:2000] + "…[truncated]"
    return value


_OOM_RE = re.compile(
    r"Tried to allocate (?P<requested>[\d.]+) (?P<requested_unit>[GM]iB).*?"
    r"total capacity of (?P<total>[\d.]+) (?P<total_unit>[GM]iB) of which "
    r"(?P<free>[\d.]+) (?P<free_unit>[GM]iB) is free.*?"
    r"allocated memory (?P<allocated>[\d.]+) (?P<allocated_unit>[GM]iB) is allocated",
    re.IGNORECASE,
)


def _memory_bytes(value: str, unit: str) -> int:
    factor = 1024 ** 3 if unit.lower() == "gib" else 1024 ** 2
    return round(float(value) * factor)


def _normalized_failure_class(value: Any) -> str | None:
    normalized = str(value or "").strip()
    return None if normalized.lower() in {"", "none", "null"} else normalized


def _evidence_timestamp(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return parse_iso_ts(value)


def _raw_process_failure_summary(
    collection: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract a process signal without deciding whether it is applicable."""
    process = collection.get("process_evidence")
    process = process if isinstance(process, dict) else {}
    stderr_tail = process.get("stderr_tail")
    stdout_tail = process.get("stdout_tail")
    stderr = "\n".join(str(item) for item in stderr_tail or [])
    stdout = "\n".join(str(item) for item in stdout_tail or [])
    combined = f"{stdout}\n{stderr}"
    failure_class = _normalized_failure_class(collection.get("failure_class"))

    oom = _OOM_RE.search(combined)
    if oom:
        ranks = {
            int(value)
            for value in re.findall(r"\[rank(\d+)\].*?OutOfMemoryError", combined)
        }
        world_match = re.search(r"(?:world_size|nproc_per_node)=(\d+)", stdout)
        world_size = int(world_match.group(1)) if world_match else None
        phase = "first_backward" if (
            "Performing initial training step" in combined and ".backward()" in combined
        ) else ("backward" if ".backward()" in combined else "unknown")
        return {
            "failure_signature": "CUDA_OOM",
            # Concrete process OOM evidence must override a coarse/stale
            # scheduler or log-transport classification.
            "failure_class": "resource",
            "phase": phase,
            "requested_bytes": _memory_bytes(oom["requested"], oom["requested_unit"]),
            "free_bytes": _memory_bytes(oom["free"], oom["free_unit"]),
            "total_bytes": _memory_bytes(oom["total"], oom["total_unit"]),
            "allocated_bytes": _memory_bytes(oom["allocated"], oom["allocated_unit"]),
            "rank_count": world_size or len(ranks) or None,
            "observed_oom_rank_count": len(ranks) or None,
            "source": "collection.process_evidence",
        }
    signatures = (
        ("OutOfMemoryError", "CUDA_OOM", "resource"),
        ("CUDA out of memory", "CUDA_OOM", "resource"),
        ("ModuleNotFoundError", "MISSING_PYTHON_MODULE", "configuration"),
        ("no kernel image is available", "UNSUPPORTED_CUDA_KERNEL", "configuration"),
        ("TIMEOUT", "TIMEOUT", "timeout"),
    )
    for needle, signature, default_class in signatures:
        if needle.lower() in combined.lower():
            return {
                "failure_signature": signature,
                "failure_class": (
                    default_class if signature == "CUDA_OOM"
                    else failure_class or default_class
                ),
                "phase": "unknown",
                "source": "collection.process_evidence",
            }
    if failure_class or str(collection.get("process_state") or "").upper() == "FAILED":
        return {
            "failure_signature": "UNCLASSIFIED_PROCESS_FAILURE",
            "failure_class": failure_class or "unknown",
            "phase": "unknown",
            "source": "collection.process_evidence",
        }
    return None


_FAILED_ATTEMPT_STATES = frozenset({"FAILED", "PREEMPTED"})


_FAILED_PROCESS_STATES = frozenset({"FAILED", "PREEMPTED"})


_FAILED_SCHEDULER_STATES = frozenset({
    "FAILED", "PREEMPTED", "NODE_FAIL", "OUT_OF_MEMORY", "TIMEOUT",
})


_FAILED_WORKER_STATES = frozenset({"FAILED", "LOST", "ERROR", "NODE_FAIL"})


def attempt_failure_evidence_assessment(
    collection: dict[str, Any],
    decision: dict[str, Any] | None = None,
    *,
    attempt_id: str,
    attempt_state: str | None,
    attempt_started_at: Any,
    observed_at: Any = None,
    domain_observed_at: dict[str, Any] | None = None,
    domain_attempt_ids: dict[str, Any] | None = None,
    domain_evidence_sources: dict[str, str | None] | None = None,
    decision_observed_at: Any = None,
    decision_source_binding: str = "EXACT_ATTEMPT",
    decision_evidence_source: str | None = None,
) -> dict[str, Any]:
    """Separate applicable terminal failure evidence from diagnostic signals.

    Failure classification is fail-closed: collected evidence must bind to the
    exact Attempt, be no older than that Attempt, and the exact Attempt must be
    in a failed terminal state.  Signals that do not meet all three conditions
    remain visible as explicitly non-applicable diagnostics.
    """
    expected_attempt_id = str(attempt_id)
    state = str(attempt_state or "UNKNOWN").upper()
    process_state = str(collection.get("process_state") or "UNKNOWN").upper()
    scheduler_state = str(collection.get("scheduler_state") or "UNKNOWN").upper()
    worker_state = str(collection.get("worker_state") or "UNKNOWN").upper()
    started_ts = _evidence_timestamp(attempt_started_at)
    terminal_failure = state in _FAILED_ATTEMPT_STATES
    observed_by_domain = dict(domain_observed_at or {})
    identities_by_domain = dict(domain_attempt_ids or {})
    sources_by_domain = dict(domain_evidence_sources or {})
    for domain in ("process", "scheduler", "worker"):
        observed_by_domain.setdefault(domain, observed_at)
        identities_by_domain.setdefault(domain, collection.get("attempt_id"))

    def applicability_for(domain: str, *, domain_terminal: bool) \
            -> tuple[str, str, str | None, float | None]:
        observed_id = str(identities_by_domain.get(domain) or "") or None
        observed_ts = _evidence_timestamp(observed_by_domain.get(domain))
        if observed_id != expected_attempt_id:
            return (
                "ATTEMPT_MISMATCH",
                f"{domain} evidence names {observed_id or 'no Attempt'} instead of "
                f"{expected_attempt_id}", observed_id, observed_ts,
            )
        if started_ts is None or observed_ts is None:
            return (
                "UNKNOWN_APPLICABILITY",
                f"actual Attempt start or {domain} observation time is unavailable",
                observed_id, observed_ts,
            )
        if observed_ts < started_ts:
            return (
                "STALE", f"{domain} evidence predates the exact Attempt start",
                observed_id, observed_ts,
            )
        if not terminal_failure:
            return (
                "NON_APPLICABLE",
                f"exact Attempt state {state} is not a failed terminal state",
                observed_id, observed_ts,
            )
        if not domain_terminal:
            return (
                "NON_APPLICABLE", f"{domain} layer state is not terminal failed",
                observed_id, observed_ts,
            )
        return (
            "APPLICABLE", f"exact, fresh terminal {domain} evidence",
            observed_id, observed_ts,
        )

    failure_summary: dict[str, Any] | None = None
    diagnostics: list[dict[str, Any]] = []
    raw_process = _raw_process_failure_summary(collection)

    candidates: list[tuple[str, dict[str, Any], bool]] = []
    process_terminal = process_state in _FAILED_PROCESS_STATES
    if process_terminal or raw_process is not None:
        process_candidate = raw_process or {
            "failure_signature": "UNCLASSIFIED_PROCESS_FAILURE",
            "failure_class": _normalized_failure_class(collection.get("failure_class"))
            or "unknown",
            "phase": "unknown",
            "source": "collection.process_evidence",
        }
        candidates.append((
            "process", {**process_candidate, "failure_domain": "process"},
            process_terminal,
        ))
    scheduler_terminal = scheduler_state in _FAILED_SCHEDULER_STATES
    if scheduler_terminal:
        candidates.append(("scheduler", {
            "failure_signature": "SCHEDULER_TERMINAL_FAILURE",
            "failure_class": _normalized_failure_class(collection.get("failure_class"))
            or "scheduler",
            "failure_domain": "scheduler",
            "phase": "unknown",
            "source": "collection.scheduler_state",
            "scheduler_state": scheduler_state,
        }, True))
    worker_terminal = worker_state in _FAILED_WORKER_STATES
    if worker_terminal:
        candidates.append(("worker", {
            "failure_signature": "WORKER_TERMINAL_FAILURE",
            "failure_class": _normalized_failure_class(collection.get("failure_class"))
            or "worker",
            "failure_domain": "worker",
            "phase": "unknown",
            "source": "collection.worker_state",
            "worker_state": worker_state,
        }, True))

    assessed: list[tuple[str, dict[str, Any], str, str, str | None, float | None]] = []
    for domain, candidate, domain_terminal in candidates:
        applicability, reason, observed_id, observed_ts = applicability_for(
            domain, domain_terminal=domain_terminal,
        )
        assessed.append((
            domain, candidate, applicability, reason, observed_id, observed_ts,
        ))

    selected = next((item for item in assessed if item[2] == "APPLICABLE"), None)
    if selected is not None:
        _, candidate, _, _, _, observed_ts = selected
        failure_summary = {
            **candidate,
            "attempt_id": expected_attempt_id,
            "observed_at": observed_ts,
            "applicability": "APPLICABLE",
            "evidence_source": sources_by_domain.get(selected[0]),
        }
    for domain, candidate, applicability, reason, observed_id, observed_ts in assessed:
        if selected is not None and candidate is selected[1]:
            continue
        if applicability != "APPLICABLE":
            diagnostics.append({
                "kind": "failure_signal",
                **candidate,
                "attempt_id": observed_id,
                "expected_attempt_id": expected_attempt_id,
                "attempt_state": state,
                "observed_at": observed_ts,
                "applicability": applicability,
                "evidence_source": sources_by_domain.get(domain),
                "reason": reason,
            })

    decision_class = (
        _normalized_failure_class(decision.get("failure_class"))
        if isinstance(decision, dict) else None
    )
    if decision_class is not None and failure_summary is None:
        diagnostics.append({
            "kind": "preliminary_failure_classification",
            "failure_class": decision_class,
            "source": "decision.failure_class",
            "attempt_id": expected_attempt_id,
            "attempt_state": state,
            "observed_at": _evidence_timestamp(decision_observed_at),
            "applicability": "NON_APPLICABLE",
            "source_binding": decision_source_binding,
            "evidence_source": decision_evidence_source,
            "reason": (
                "decision metadata is contextual only and is not exact, fresh, "
                "terminal failure evidence"
            ),
        })

    return {
        "failure_summary": failure_summary,
        "diagnostic_evidence": compact_evidence(diagnostics),
    }


def structured_failure_summary(
    collection: dict[str, Any], decision: dict[str, Any] | None = None, *,
    attempt_id: str | None = None, attempt_state: str | None = None,
    attempt_started_at: Any = None, observed_at: Any = None,
) -> dict[str, Any] | None:
    """Return only exact, fresh, terminal failure evidence.

    Callers without the Attempt applicability context receive no failure rather
    than silently promoting historical process text into a current failure.
    """
    if attempt_id is None:
        return None
    return attempt_failure_evidence_assessment(
        collection, decision,
        attempt_id=attempt_id,
        attempt_state=attempt_state,
        attempt_started_at=attempt_started_at,
        observed_at=observed_at,
    )["failure_summary"]


def evidence_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
