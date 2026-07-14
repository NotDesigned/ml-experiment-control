"""Normalized lifecycle failures and conservative text classification."""

from __future__ import annotations

import re
from enum import Enum


class FailureClass(str, Enum):
    NONE = "none"
    TRANSPORT = "transport"
    SCHEDULER = "scheduler"
    PREEMPTION = "preemption"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"
    MODEL = "model"
    EVALUATION = "evaluation"
    UNKNOWN = "unknown"


def classify_failure(text: str = "") -> FailureClass:
    """Classify known failures without automatically authorizing a retry."""
    haystack = text.lower()
    if "eviction" in haystack or "preempted" in haystack:
        return FailureClass.PREEMPTION
    if "node failure" in haystack or "boot failure" in haystack:
        return FailureClass.SCHEDULER
    # A process OOM is stronger evidence than incidental transport diagnostics
    # collected alongside it (for example a log stream closing after failure).
    if any(token in haystack for token in (
        "out of memory", "cuda oom", "outofmemoryerror",
    )):
        return FailureClass.RESOURCE
    timeout = "timed out" in haystack or bool(re.search(r"\btimeout\b", haystack))
    transport_timeout_context = any(token in haystack for token in (
        "connection", "connect to host", "banner exchange", "ssh", "tls",
        "clienthello", "handshake", "proxy", "dial tcp", "i/o", "read timeout",
        "request timeout", "client.timeout", "context deadline exceeded",
    ))
    live_logs_expired = bool(re.search(
        r'["\']?live_logs_expired["\']?\s*[:=]\s*(?:true|1)\b', haystack,
    ))
    if any(token in haystack for token in (
        "tls", "eof", "502", "connection reset", "expired log",
    )) or live_logs_expired or (timeout and transport_timeout_context):
        return FailureClass.TRANSPORT
    if (
        timeout
        or "time limit" in haystack
        or "wall-time" in haystack
        or "wall time" in haystack
    ):
        return FailureClass.RESOURCE
    if any(token in haystack for token in (
        "missing cache", "no such file", "invalid config", "not mounted",
        "modulenotfounderror", "no module named", "importerror",
    )):
        return FailureClass.CONFIGURATION
    if any(token in haystack for token in ("nan", "diverg", "collapse")):
        return FailureClass.MODEL
    if "required metric" in haystack or "evaluation" in haystack and "missing" in haystack:
        return FailureClass.EVALUATION
    return FailureClass.UNKNOWN
