"""Durable submission identities shared by state storage and backends."""

from __future__ import annotations

import re
import secrets
from collections.abc import Mapping
from typing import Any


SUBMISSION_TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")


def new_submission_token() -> str:
    """Return a non-reusable 128-bit identity for one scheduler mutation."""
    return secrets.token_hex(16)


def validate_submission_token(value: Any) -> str:
    token = str(value or "")
    if not SUBMISSION_TOKEN_RE.fullmatch(token):
        raise RuntimeError(
            "durable submission intent requires a 128-bit hexadecimal submission_token"
        )
    return token


def require_submission_intent(
    intent: Mapping[str, Any] | None,
) -> tuple[str, Mapping[str, Any]]:
    """Return the durable token and frozen backend request, or fail closed."""
    if not isinstance(intent, Mapping):
        raise RuntimeError("non-dry-run submission requires a durable submission intent")
    token = validate_submission_token(intent.get("submission_token"))
    request = intent.get("request")
    if request is None:
        # Accept the original flat structural contract for callers that already
        # persist it atomically. ExperimentStateStore returns the nested form.
        request = intent
    if not isinstance(request, Mapping):
        raise RuntimeError("submission intent request must be a mapping")
    return token, request


def submission_marker(token: str) -> str:
    """Return the scheduler-visible correlation marker for one intent."""
    return f"ml-exp-{validate_submission_token(token)}"
