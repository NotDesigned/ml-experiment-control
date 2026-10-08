"""Durable ownership of image builds, independent of HTTP request lifetimes."""
from __future__ import annotations

from dataclasses import dataclass
import re

from ..projects.source_imports import IDENTITY
from ..storage import utc_now


@dataclass(frozen=True)
class PendingRuntimeBuild:
    project: str
    runtime_id: str
    value: dict
    revision: int
    reconcile: bool = False


def recover_interrupted_builds(service) -> int:
    # Called only by the daemon holding the exclusive workspace lease. Recovery
    # changes uncertainty into an inspectable state; it never starts a build.
    recovered = 0
    for path in sorted(service.root.glob("*/*.json")):
        project, runtime_id = path.parent.name, path.stem
        if not IDENTITY.fullmatch(project) or not re.fullmatch(r"runtime\.[0-9a-f]{64}", runtime_id):
            continue
        with service.state(project, runtime_id) as (store, snapshot):
            value = dict(snapshot.value)
            if value.get("status") != "EXECUTING":
                continue
            if value.get("project") != project or value.get("runtime_id") != runtime_id:
                raise ValueError("interrupted runtime identity mismatch")
            value.update(status="RECONCILE_REQUIRED", error="daemon stopped during packaging; reconcile the published receipt")
            store.commit(value, expected_revision=snapshot.revision,
                         event={"event": "runtime_execution_interrupted", "timestamp": utc_now()})
            recovered += 1
    return recovered
