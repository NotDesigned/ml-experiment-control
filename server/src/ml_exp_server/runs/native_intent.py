"""Optional bridge to the native Rust OperationIntent validator."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any

from ..application_errors import ApplicationError


def validate(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Validate an intent through ``ml-exp-rs`` when the binary is available.

    The returned JSON is only a validation receipt. It never authorizes an
    Action or contacts the daemon API. If no native binary is installed, the
    existing Pydantic validation remains the compatibility path.
    """
    configured = os.environ.get("ML_EXPD_CONTRACT_BIN")
    executable = configured or shutil.which("ml-exp-rs")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "contract", "intent-validate"],
            input=json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ApplicationError(
            "native intent validator is unavailable",
            status_code=503,
            code="NATIVE_INTENT_UNAVAILABLE",
        ) from exc
    if result.returncode != 0:
        raise ApplicationError(
            "native intent validation rejected the request",
            status_code=422,
            code="NATIVE_INTENT_INVALID",
        )
    try:
        receipt = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ApplicationError(
            "native intent validator returned malformed output",
            status_code=503,
            code="NATIVE_INTENT_UNAVAILABLE",
        ) from exc
    if not isinstance(receipt, dict) or any(
        receipt.get(key) != payload.get(key)
        for key in ("kind", "title", "draft")
    ):
        raise ApplicationError(
            "native intent validator identity differs",
            status_code=503,
            code="NATIVE_INTENT_UNAVAILABLE",
        )
    return receipt
