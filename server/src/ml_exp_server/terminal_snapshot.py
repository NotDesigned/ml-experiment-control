"""Server-owned terminal read model shared with HTTP renderers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .campaign_lifecycle import campaign_snapshot
from .ingest.indexer import RunIndex, index_project
from .schemas import ResearchProject, RunIndexRow


ACTIVE_STATES = {"SUBMITTING", "QUEUED", "STARTING", "RUNNING", "EVALUATING"}
_LAYERS = ("scheduler", "worker", "process", "model", "evaluation")


@dataclass
class Snapshot:
    """Immutable terminal display data, independent of its transport."""

    projects: list[ResearchProject]
    runs: dict[str, list[RunIndexRow]]
    attention: dict[str, list[tuple[str, str, str]]]
    historical_failures: dict[str, int] = field(default_factory=dict)
    campaign_statuses: dict[tuple[str, str], dict] = field(default_factory=dict)
    collector_errors: dict[tuple[str, str], str] = field(default_factory=dict)
    loaded_at: float = field(default_factory=time.time)


def _stale_layers(row: RunIndexRow) -> list[str]:
    return [name for name in _LAYERS if getattr(row.evidence, name).stale]


def is_current_collector_error(row: RunIndexRow, status: Any) -> bool:
    """Return whether a poll error is newer than the scheduler evidence.

    Operators may recover a run with a direct controller observation after an
    earlier collector failure. Keeping that failure visible after a newer
    canonical status has landed turns successful terminal runs into permanent
    false alarms.
    """
    if not getattr(status, "last_error", None):
        return False
    scheduler_as_of = row.evidence.scheduler.as_of
    last_poll_at = getattr(status, "last_poll_at", None)
    return scheduler_as_of is None or last_poll_at is None or last_poll_at >= scheduler_as_of


def build_snapshot(index: RunIndex, projects: list[ResearchProject],
                   *, reindex: bool = False) -> Snapshot:
    """Build the exact terminal view from an already-owned runtime read model."""
    runs: dict[str, list[RunIndexRow]] = {}
    attention: dict[str, list[tuple[str, str, str]]] = {}
    historical_failures: dict[str, int] = {}
    campaign_statuses: dict[tuple[str, str], dict] = {}
    collector_errors: dict[tuple[str, str], str] = {}
    for project in projects:
        if reindex:
            index_project(index, project)
        rows = index.list_runs(project.project)
        runs[project.project] = rows
        for campaign in project.campaigns:
            campaign_statuses[(project.project, campaign.name)] = campaign_snapshot(
                index, project, campaign.name,
            )
        active_campaigns = {
            campaign.name for campaign in project.campaigns
            if campaign_statuses[(project.project, campaign.name)].get("lifecycle_state")
            not in {"COMPLETED", "ARCHIVED"}
        }

        def in_active_research(row: RunIndexRow) -> bool:
            return row.campaign in active_campaigns or any(
                binding.campaign in active_campaigns
                and binding.membership.included_in_analysis
                for binding in row.campaign_memberships
            )

        items: list[tuple[str, str, str]] = []
        historical_failure_count = 0
        for row in rows:
            state = (row.scheduler_state or "").upper()
            if state in {"FAILED", "PREEMPTED"}:
                if in_active_research(row):
                    failure = row.decision.get("failure_class")
                    items.append(("FAILED", row.run_id,
                                  state + (f" ({failure})" if failure else "")))
                else:
                    historical_failure_count += 1
            stale = _stale_layers(row)
            if stale and (in_active_research(row) or state in ACTIVE_STATES):
                reasons = "; ".join(filter(None, (
                    getattr(row.evidence, layer).stale_reason for layer in stale)))
                items.append(("STALE", row.run_id,
                              reasons or "stale: " + ",".join(stale)))
        historical_failures[project.project] = historical_failure_count
        row_by_id = {row.run_id: row for row in rows}
        for status in index.collector_statuses(project.project):
            row = row_by_id.get(status.run_id)
            state = (row.scheduler_state or "").upper() if row else ""
            if row is not None and is_current_collector_error(row, status) and (
                in_active_research(row) or state in ACTIVE_STATES
            ):
                items.append(("COLLECTOR", status.run_id, status.last_error))
                collector_errors[(project.project, status.run_id)] = status.last_error
        state_by_run = {
            row.run_id: (row.scheduler_state or "UNKNOWN").upper() for row in rows
        }
        items.sort(key=lambda item: (
            0 if item[0] == "STALE" and state_by_run.get(item[1]) in ACTIVE_STATES else
            1 if item[0] == "COLLECTOR" and state_by_run.get(item[1]) in ACTIVE_STATES else
            2 if item[0] == "FAILED" else 3,
            item[1], item[0],
        ))
        attention[project.project] = items
    return Snapshot(
        projects=projects, runs=runs, attention=attention,
        historical_failures=historical_failures,
        campaign_statuses=campaign_statuses, collector_errors=collector_errors,
    )


def snapshot_payload(snapshot: Snapshot) -> dict[str, Any]:
    """Return JSON-compatible transport without tuple keys or Python paths."""
    return {
        "projects": [project.model_dump(mode="json") for project in snapshot.projects],
        "runs": {
            project: [row.model_dump(mode="json") for row in rows]
            for project, rows in snapshot.runs.items()
        },
        "attention": snapshot.attention,
        "historical_failures": snapshot.historical_failures,
        "campaign_statuses": [
            {"project": project, "campaign": campaign, "status": status}
            for (project, campaign), status in snapshot.campaign_statuses.items()
        ],
        "collector_errors": [
            {"project": project, "run_id": run_id, "error": error}
            for (project, run_id), error in snapshot.collector_errors.items()
        ],
        "loaded_at": snapshot.loaded_at,
    }


def snapshot_from_payload(payload: dict[str, Any]) -> Snapshot:
    """Validate a server snapshot before a terminal renderer trusts it."""
    projects = [ResearchProject.model_validate(item) for item in payload.get("projects", [])]
    runs = {
        str(project): [RunIndexRow.model_validate(row) for row in rows]
        for project, rows in (payload.get("runs") or {}).items()
        if isinstance(rows, list)
    }
    attention = {
        str(project): [tuple(map(str, item)) for item in rows if isinstance(item, list)]
        for project, rows in (payload.get("attention") or {}).items()
        if isinstance(rows, list)
    }
    campaign_statuses: dict[tuple[str, str], dict] = {}
    for item in payload.get("campaign_statuses") or []:
        if isinstance(item, dict) and isinstance(item.get("status"), dict):
            campaign_statuses[(str(item.get("project") or ""),
                               str(item.get("campaign") or ""))] = item["status"]
    collector_errors: dict[tuple[str, str], str] = {}
    for item in payload.get("collector_errors") or []:
        if isinstance(item, dict) and item.get("error"):
            collector_errors[(str(item.get("project") or ""),
                              str(item.get("run_id") or ""))] = str(item["error"])
    return Snapshot(
        projects=projects,
        runs=runs,
        attention=attention,
        historical_failures={str(key): int(value) for key, value in
                             (payload.get("historical_failures") or {}).items()},
        campaign_statuses=campaign_statuses,
        collector_errors=collector_errors,
        loaded_at=float(payload.get("loaded_at") or time.time()),
    )
