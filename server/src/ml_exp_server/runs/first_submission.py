"""One first-submission intent for authored and API-managed Runs."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Callable

import yaml

from ..application_errors import ApplicationError
from .failures import evidence_digest


def validate_gpu_budget(max_gpu_hours: float | None, resource_approval: str) -> None:
    if resource_approval not in {"budget_cap", "review_exact"} or (
        resource_approval == "budget_cap" and (
            max_gpu_hours is None or not math.isfinite(max_gpu_hours) or max_gpu_hours <= 0
        )
    ):
        raise ApplicationError("invalid GPU-hour budget or resource approval mode",
                               status_code=422, code="INVALID_GPU_BUDGET")


def first_submission_intent(
    configured, row, *, run_id: str, campaign_file: Callable[[], Path],
    max_gpu_hours: float | None, resource_approval: str, reason: str,
    evidence: Callable[[], dict[str, Any]], source_service, submission_number: int | None = None,
) -> dict[str, Any]:
    """Validate and freeze the exact first Attempt; never authorize or execute."""
    validate_gpu_budget(max_gpu_hours, resource_approval)
    if configured.controller is None:
        raise ApplicationError("project has no controller configuration", code="CONTROLLER_UNAVAILABLE")
    attempts = row.attempts if row is not None else []
    state = str(row.scheduler_state or "NOT_SUBMITTED").upper() if row is not None else "NOT_SUBMITTED"
    if state != "NOT_SUBMITTED":
        raise ApplicationError(f"Run state {state} is not eligible for first submission", code="RUN_NOT_SUBMITTABLE")
    if any(attempt.has_submission for attempt in attempts):
        raise ApplicationError("Run already has submitted Attempt evidence; use exact Attempt retry", code="RUN_ALREADY_SUBMITTED")
    attempt_id = attempts[-1].attempt_id if attempts else "attempt-001"
    provenance = getattr(row, "provenance", {}) or {}
    expected_source = str(provenance.get("source_id") or "") if provenance.get("source_binding") == "campaign_file" else ""
    if expected_source.startswith("source."):
        if source_service is None:
            raise ApplicationError("source revision service is unavailable", code="SOURCE_IMPORT_BLOCKED")
        source_service.get(configured.project, expected_source)
    draft: dict[str, Any] = {
        "campaign_file": str(campaign_file()), "run_id": run_id,
        "attempt_id": attempt_id, "resource_approval": resource_approval,
    }
    if expected_source:
        draft["expected_source_id"] = expected_source
    if max_gpu_hours is not None:
        draft["max_gpu_hours"] = max_gpu_hours
    intent = {
        "kind": "SUBMIT_RUN", "title": f"Launch {run_id} as {attempt_id}",
        "target": f"run://{configured.project}/{run_id}",
        "change_summary": reason or "submit the authored Run's first Attempt",
        "resource_estimate": (
            "exact dry-run resources require human Action confirmation"
            if resource_approval == "review_exact" else f"up to {max_gpu_hours:g} GPU-hours"
        ),
        "rationale": "materialize one authored Campaign membership",
        "risk": "scheduler mutation requires separate authorization and execution",
        "draft": yaml.safe_dump(draft, sort_keys=False),
        "evidence_digest": evidence_digest(evidence()),
    }
    if submission_number is not None:
        intent["idempotency_key"] = "submission-" + hashlib.sha256(
            f"{configured.project}:{run_id}:{attempt_id}:{submission_number}".encode()
        ).hexdigest()[:20]
    return intent
