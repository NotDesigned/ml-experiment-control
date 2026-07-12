"""Credential redaction shared by Python-side backend diagnostics."""

from __future__ import annotations

import re


SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:secret|token|password|passwd|credential|access[_-]?key(?:[_-]?(?:id|secret))?"
    r"|api[_-]?key|proxy|authorization|cookie)[\w.-]*\b[\s\"']*[=:][\s\"']*)"
    r"([^\s,;\"']+)"
)
BEARER_RE = re.compile(r"(?i)(\b(?:authorization\s*:\s*)?bearer\s+)[^\s,;]+")
URL_USERINFO_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://)[^/@\s]+@")
SENSITIVE_QUERY_RE = re.compile(
    r"(?i)([?&](?:access[_-]?key(?:[_-]?(?:id|secret))?|api[_-]?key|secret|token|signature)=)"
    r"[^&#\s]+"
)


def redact_line(line: str) -> str:
    """Redact credential forms from backend diagnostic lines."""
    line = URL_USERINFO_RE.sub(r"\1<redacted>@", line)
    line = BEARER_RE.sub(r"\1<redacted>", line)
    line = SENSITIVE_QUERY_RE.sub(r"\1<redacted>", line)
    return SECRET_ASSIGNMENT_RE.sub(r"\1<redacted>", line)
