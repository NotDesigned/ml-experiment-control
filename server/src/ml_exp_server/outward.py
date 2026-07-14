"""Sanitized transport projections for internal experiment read models.

Internal index rows retain producer-authored failure classifications for
diagnostic reconstruction.  Transport DTOs must expose those classifications
only through a daemon-built ``failure_assessment``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_OPERATIONAL_DECISION_KEYS = (
    "action", "reason", "resume_checkpoint", "retries_allowed", "retries_used",
)


def operational_decision(value: Any) -> dict[str, Any]:
    value = value if isinstance(value, Mapping) else {}
    return {
        key: value[key]
        for key in _OPERATIONAL_DECISION_KEYS
        if value.get(key) is not None
    }


def sanitized_outward(value: Any) -> Any:
    """Recursively remove raw classifications from one transport object.

    Call this before attaching ``failure_assessment``; applicable summaries and
    non-applicable diagnostics inside that assessment intentionally retain their
    bounded ``failure_class`` field.
    """
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key == "failure_class":
                continue
            if key == "decision":
                result[key] = operational_decision(item)
            else:
                result[key] = sanitized_outward(item)
        return result
    if isinstance(value, (list, tuple)):
        return [sanitized_outward(item) for item in value]
    return value


def attempt_dto(attempt: Any) -> dict[str, Any]:
    value = sanitized_outward(attempt)
    return value if isinstance(value, dict) else {}


def run_dto(row: Any) -> dict[str, Any]:
    value = sanitized_outward(row)
    return value if isinstance(value, dict) else {}
