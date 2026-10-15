"""Parse controller output and keep bounded, credential-safe diagnostics."""
from __future__ import annotations

import json
import re
import subprocess
from typing import Any

from experiment_control.redaction import redact_line

_SECRET = re.compile(
    r"(?i)(?:^|[_-])(?:secret|token|password|credential|access[_-]?key|"
    r"api[_-]?key|proxy|authorization|cookie)(?:$|[_-])"
)
STDOUT_LOG_BYTES = 32 * 1024
STDERR_LOG_BYTES = 64 * 1024
_ENCODED_CONTENT = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{128,}={0,2}")
_COMMAND_EXCEPTION = re.compile(
    r"(?m)^.*(?:subprocess\.(?:CalledProcessError|TimeoutExpired):|"
    r"Command ['\"].*(?:returned non-zero exit status|timed out after)).*$"
)


def redact(value: Any) -> Any:
    """Remove common credential shapes from controller results."""
    if isinstance(value, dict):
        return {
            str(key): ("[REDACTED]" if _SECRET.search(str(key)) else redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return redact_line(value).replace("<redacted>", "[REDACTED]")
    return value


def safe_log_tail(value: str | bytes | None, *, limit: int, command: list[str]) -> str:
    """Keep diagnostic tails without serializing subprocess command capsules."""
    decoded = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
    safe = _COMMAND_EXCEPTION.sub("subprocess command failed; inspect preceding diagnostics", decoded)
    # Exception messages can quote either the list or tuple form of argv.
    for representation in (repr(command), repr(tuple(command))):
        safe = safe.replace(representation, "[COMMAND_REDACTED]")
    safe = _ENCODED_CONTENT.sub("[ENCODED_CONTENT_REDACTED]", str(redact(safe)))
    return safe.encode("utf-8")[-limit:].decode("utf-8", errors="ignore")


def timeout_result(exc: subprocess.TimeoutExpired, command: list[str]) -> dict[str, Any]:
    return {
        "returncode": None, "timeout": True, "payload": None,
        "stdout": safe_log_tail(exc.stdout, limit=STDOUT_LOG_BYTES, command=command),
        "stderr": safe_log_tail(exc.stderr, limit=STDERR_LOG_BYTES, command=command),
        "error": "controller command exceeded its configured timeout",
    }


def completed_result(returncode: int, stdout: str, stderr: str, command: list[str]) -> dict[str, Any]:
    """Use the same JSON/redaction contract for ordinary and pinned execution."""
    payload: Any = None
    if stdout.strip():
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            pass
    return {
        "returncode": returncode, "timeout": False, "payload": redact(payload),
        "stdout": safe_log_tail(stdout, limit=STDOUT_LOG_BYTES, command=command),
        "stderr": safe_log_tail(stderr, limit=STDERR_LOG_BYTES, command=command),
    }
