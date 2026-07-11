"""Typed read-only scheduler identity evidence returned by every backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class IdentityReport:
    """Availability and ambiguity evidence for one run/attempt scheduler key."""

    available: bool
    ambiguous: bool
    scheduler_job_ids: tuple[str, ...] = ()
    remote_manifest_exists: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scheduler_job_ids"] = list(self.scheduler_job_ids)
        return payload
