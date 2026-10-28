"""Optional bridge to the native Rust Slurm allocation parser."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any


def parse(output: str, partition: str) -> dict[str, Any] | None:
    """Parse allocation evidence with Rust when the preview binary is installed.

    Returning ``None`` means the native binary is not installed and lets the
    existing Python parser remain the compatibility path.  A configured or
    discovered binary that rejects evidence raises ``ValueError`` so callers
    retain the existing fail-closed observation behavior.
    """
    executable = os.environ.get("ML_EXPD_ALLOCATION_BIN") or shutil.which("ml-exp-rs")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "contract", "slurm-allocation-validate"],
            input=json.dumps(
                {"output": output, "partition": partition},
                ensure_ascii=True,
                separators=(",", ":"),
            ),
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("native allocation validator is unavailable") from exc
    if result.returncode != 0:
        raise ValueError("native allocation evidence rejected")
    try:
        value = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("native allocation validator returned malformed output") from exc
    if not isinstance(value, dict):
        raise ValueError("native allocation validator returned invalid output")
    return value
