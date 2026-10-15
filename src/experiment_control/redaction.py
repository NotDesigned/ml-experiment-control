"""Credential redaction shared by Python-side backend diagnostics."""

from __future__ import annotations

import re


SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:[\w.-]*[_-])?(?:secret|token|password|passwd|credential|access[_-]?key(?:[_-]?(?:id|secret))?"
    r"|api[_-]?key|private[_-]?key|proxy|authorization|cookie)[\w.-]*\b[\s\"']*[=:][\s\"']*)"
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


def sensitive_json_key(key: str) -> bool:
    """Recognize credential keys without masking scientific token/loss fields.

    Keep this vocabulary aligned with the packaged native log sanitizer.
    Credentials use exact names or explicit suffixes, e.g. API_TOKEN;
    token_recon_ppl, tokenizer_path and proxy_loss remain ordinary evidence.
    """
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key.replace("-", "_").replace(".", "_")).lower()
    suffixes = ("token", "secret", "password", "passwd", "credential", "authorization", "cookie", "proxy",
                "signature", "api_key", "access_key", "access_key_id", "access_key_secret", "private_key",
                "apikey", "accesskey", "accesskeyid", "accesskeysecret", "privatekey")
    return normalized.startswith("authorization_") or any(
        normalized == suffix or normalized.endswith("_" + suffix) for suffix in suffixes)
