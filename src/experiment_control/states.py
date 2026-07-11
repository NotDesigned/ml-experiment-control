"""Normalized lifecycle failures and conservative text classification."""

from __future__ import annotations

from enum import Enum


class FailureClass(str, Enum):
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
    if "out of memory" in haystack or "cuda oom" in haystack or "timed out" in haystack:
        return FailureClass.RESOURCE
    if any(token in haystack for token in ("tls", "eof", "502", "connection reset", "expired log")):
        return FailureClass.TRANSPORT
    if any(token in haystack for token in ("missing cache", "no such file", "invalid config", "not mounted")):
        return FailureClass.CONFIGURATION
    if any(token in haystack for token in ("nan", "diverg", "collapse")):
        return FailureClass.MODEL
    if "required metric" in haystack or "evaluation" in haystack and "missing" in haystack:
        return FailureClass.EVALUATION
    return FailureClass.UNKNOWN
