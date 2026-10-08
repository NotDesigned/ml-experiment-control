"""Credential redaction shared by Python-side backend diagnostics."""

from __future__ import annotations

import re


SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:[\w.-]*[_-])?(?:secret|token|password|passwd|credential|access[_-]?key(?:[_-]?(?:id|secret))?"
    r"|api[_-]?key|proxy|authorization|cookie)[\w.-]*\b[\s\"']*[=:][\s\"']*)"
    r"([^\s,;&\"']+)"
)
BEARER_RE = re.compile(r"(?i)(\b(?:authorization\s*:\s*)?bearer\s+)[^\s,;]+")
URL_USERINFO_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://)[^/@\s]+@")
SENSITIVE_QUERY_RE = re.compile(
    r"(?i)([?&](?:access[_-]?key(?:[_-]?(?:id|secret))?|api[_-]?key|secret|token|signature"
    r"|x-amz-[\w-]+|x-goog-[\w-]+|sig)=)"
    r"[^&#\s]+"
)
URL_QUERY_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://[^?\s]+)\?([^#\s\"']*)")
SIGNED_QUERY_RE = re.compile(
    r"(?i)(?:^|&)(?:x-amz-signature|x-goog-signature|signature|sig)=",
)


def _redact_signed_query(match: re.Match[str]) -> str:
    if SIGNED_QUERY_RE.search(match[2]):
        return match[1] + "?<redacted>"
    return match[0]


def redact_line(line: str) -> str:
    """Redact credential forms from backend diagnostic lines."""
    line = URL_USERINFO_RE.sub(r"\1<redacted>@", line)
    line = BEARER_RE.sub(r"\1<redacted>", line)
    # A signed URL is itself a credential. Keep its location for diagnostics,
    # but remove the entire query, including provider credential identifiers.
    line = URL_QUERY_RE.sub(_redact_signed_query, line)
    line = SENSITIVE_QUERY_RE.sub(r"\1<redacted>", line)
    return SECRET_ASSIGNMENT_RE.sub(r"\1<redacted>", line)
