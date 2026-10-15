"""Normalized failure vocabulary shared by backend contracts."""

from __future__ import annotations

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
