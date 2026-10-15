"""Staleness rules for the separated evidence layers.

The trap this module exists for: a root status.json whose mtime is recent but
whose content is just another RUNNING poll, while the worker-side status has
not moved for hours. Staleness is judged per layer, against that layer's own
evidence timestamp, and only flags combinations that are actually suspicious.
"""

from __future__ import annotations

from typing import Optional

from ..schemas import RunIndexRow

# Without an explicit heartbeat contract, use the same conservative age bound
# for every workload. Training throughput does not imply a logging interval.
DEFAULT_STALE_AFTER_SECONDS = 30 * 60


def _age(now: float, as_of: Optional[float]) -> Optional[float]:
    if as_of is None:
        return None
    return max(0.0, now - as_of)


def _fmt_age(seconds: float) -> str:
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def apply_freshness(row: RunIndexRow, *, now: float) -> None:
    """Set stale/stale_reason on each evidence layer in place."""
    scheduler_active = (row.scheduler_state or "").upper() in {
        "SUBMITTING", "QUEUED", "RUNNING", "EVALUATING", "STARTING",
    }
    if not scheduler_active:
        return  # terminal or unsubmitted runs are not expected to move

    checks = (
        ("worker", row.evidence.worker, DEFAULT_STALE_AFTER_SECONDS),
        ("process", row.evidence.process, DEFAULT_STALE_AFTER_SECONDS),
        ("model", row.evidence.model, DEFAULT_STALE_AFTER_SECONDS),
    )
    for name, layer, bound in checks:
        age = _age(now, layer.as_of)
        if age is not None and age > bound:
            layer.stale = True
            layer.stale_reason = (
                f"scheduler is {row.scheduler_state} but {name} evidence is "
                f"{_fmt_age(age)} old (threshold {_fmt_age(bound)})"
            )

    # The scheduler snapshot itself can also rot (collector died).
    scheduler_age = _age(now, row.evidence.scheduler.as_of)
    if scheduler_age is not None and scheduler_age > 2 * DEFAULT_STALE_AFTER_SECONDS:
        row.evidence.scheduler.stale = True
        row.evidence.scheduler.stale_reason = (
            f"last scheduler observation is {_fmt_age(scheduler_age)} old"
        )
