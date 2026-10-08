"""Observational progress; never grants authorization or retries an operation."""
from __future__ import annotations

import time
from pathlib import Path

from ..storage import atomic_json, read_json, utc_now


def record_progress(path: Path, phase: str, message: str, *, timeout_seconds=None):
    previous = read_json(path, {})
    event = {"phase": phase, "message": message, "at": utc_now()}
    value = {**event, "updated_unix": time.time(),
             "timeout_seconds": timeout_seconds,
             "events": [*(previous.get("events") or []), event][-64:]}
    atomic_json(path, value)


def progress_view(path: Path, status: str, *, active=False, last_activity=None):
    value = read_json(path, {})
    updated = value.get("updated_unix")
    if last_activity is not None:
        updated = max(updated or 0, last_activity)
    age = max(0, time.time() - updated) if updated is not None else None
    stalled = active and age is not None and age >= 120
    return {"status": status, "phase": value.get("phase", status) if active else status,
            "message": value.get("message"), "phase_started_at": value.get("at"),
            "last_progress_unix": updated, "seconds_since_progress": age,
            "no_progress_warning": bool(stalled),
            "diagnostic": "No recorded progress for at least 120 seconds; inspect logs. This does not prove failure; do not resubmit." if stalled else None,
            "phase_timeout_seconds": value.get("timeout_seconds"),
            "events": value.get("events", [])}
