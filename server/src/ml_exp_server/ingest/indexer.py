"""SQLite query index over run directories. Files stay canonical; the index
can be dropped and rebuilt at any time."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from ..schemas import (
    CampaignMembershipBinding,
    CampaignRelationship,
    CollectorRunStatus,
    ResearchProject,
    RunIndexRow,
)
from .runscan import discover_run_dirs, scan_run_dir

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    project TEXT NOT NULL,
    run_id TEXT NOT NULL,
    campaign TEXT,
    role TEXT,
    scheduler_state TEXT,
    run_dir TEXT NOT NULL,
    row_json TEXT NOT NULL,
    scanned_at REAL,
    PRIMARY KEY (project, run_id)
);
CREATE TABLE IF NOT EXISTS collector_status (
    project TEXT NOT NULL,
    run_id TEXT NOT NULL,
    last_poll_at REAL,
    last_verb TEXT,
    last_error TEXT,
    verb_results TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (project, run_id)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class RunIndex:
    """Thread-safe wrapper around the SQLite index."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        columns = {
            row[1] for row in self._conn.execute(
                "PRAGMA table_info(collector_status)"
            ).fetchall()
        }
        if "verb_results" not in columns:
            self._conn.execute(
                "ALTER TABLE collector_status ADD COLUMN verb_results "
                "TEXT NOT NULL DEFAULT '{}'"
            )
            self._conn.commit()
        # Notified with (project, run_id) after each upsert that changed data.
        self.on_update: Optional[Callable[[str, str], None]] = None

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def upsert_run(self, row: RunIndexRow) -> bool:
        """Insert or update one run. Returns True when content changed."""
        payload = row.model_dump_json()
        with self._lock:
            cursor = self._conn.execute(
                "SELECT row_json FROM runs WHERE project=? AND run_id=?",
                (row.project, row.run_id),
            )
            existing = cursor.fetchone()
            if existing is not None:
                # scanned_at always moves; compare content without it.
                old = json.loads(existing[0]); old.pop("scanned_at", None)
                new = json.loads(payload); new.pop("scanned_at", None)
                changed = old != new
            else:
                changed = True
            self._conn.execute(
                "INSERT INTO runs (project, run_id, campaign, role, scheduler_state,"
                " run_dir, row_json, scanned_at) VALUES (?,?,?,?,?,?,?,?)"
                " ON CONFLICT(project, run_id) DO UPDATE SET campaign=excluded.campaign,"
                " role=excluded.role, scheduler_state=excluded.scheduler_state,"
                " run_dir=excluded.run_dir, row_json=excluded.row_json,"
                " scanned_at=excluded.scanned_at",
                (row.project, row.run_id, row.campaign, row.role,
                 row.scheduler_state, row.run_dir, payload, row.scanned_at),
            )
            self._conn.commit()
        if changed and self.on_update is not None:
            self.on_update(row.project, row.run_id)
        return changed

    def get_run(self, project: str, run_id: str) -> Optional[RunIndexRow]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT row_json FROM runs WHERE project=? AND run_id=?",
                (project, run_id),
            )
            record = cursor.fetchone()
        return RunIndexRow.model_validate_json(record[0]) if record else None

    def list_runs(self, project: Optional[str] = None,
                  campaign: Optional[str] = None) -> list[RunIndexRow]:
        query = "SELECT row_json FROM runs"
        clauses, params = [], []
        if project is not None:
            clauses.append("project=?"); params.append(project)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY run_id"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        parsed = [RunIndexRow.model_validate_json(r[0]) for r in rows]
        if campaign is not None:
            parsed = [
                row for row in parsed
                if row.campaign == campaign or any(
                    binding.campaign == campaign
                    for binding in row.campaign_memberships
                )
            ]
        return parsed

    def record_poll(self, project: str, run_id: str, verb: str,
                    error: Optional[str] = None, *, now: Optional[float] = None,
                    outcome: Optional[str] = None) -> None:
        observed_at = now if now is not None else time.time()
        resolved_outcome = outcome or ("failed" if error else "succeeded")
        with self._lock:
            record = self._conn.execute(
                "SELECT verb_results FROM collector_status WHERE project=? AND run_id=?",
                (project, run_id),
            ).fetchone()
            try:
                verb_results = json.loads(record[0]) if record and record[0] else {}
            except json.JSONDecodeError:
                verb_results = {}
            verb_results[verb] = {
                "outcome": resolved_outcome,
                "at": observed_at,
                "error": error,
            }
            active_errors = [
                str(item["error"])
                for item in verb_results.values()
                if isinstance(item, dict) and item.get("error")
            ]
            aggregate_error = "; ".join(active_errors) or None
            self._conn.execute(
                "INSERT INTO collector_status (project, run_id, last_poll_at, last_verb,"
                " last_error, verb_results) VALUES (?,?,?,?,?,?)"
                " ON CONFLICT(project, run_id) DO UPDATE SET"
                " last_poll_at=excluded.last_poll_at, last_verb=excluded.last_verb,"
                " last_error=excluded.last_error, verb_results=excluded.verb_results",
                (project, run_id, observed_at, verb, aggregate_error,
                 json.dumps(verb_results, sort_keys=True)),
            )
            self._conn.commit()

    def collector_statuses(self, project: Optional[str] = None) -> list[CollectorRunStatus]:
        query = (
            "SELECT project, run_id, last_poll_at, last_verb, last_error, verb_results "
            "FROM collector_status"
        )
        params: list = []
        if project is not None:
            query += " WHERE project=?"; params.append(project)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [CollectorRunStatus(
            project=r[0], run_id=r[1], last_poll_at=r[2], last_verb=r[3],
            last_error=r[4], verb_results=json.loads(r[5] or "{}"),
        ) for r in rows]

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?,?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
            self._conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        with self._lock:
            record = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return record[0] if record else None


def index_project(index: RunIndex, project: ResearchProject,
                  *, now: Optional[float] = None) -> int:
    """Scan all run roots of a project into the index. Returns run count."""
    rows: list[RunIndexRow] = []
    for root in project.resolved_run_roots():
        for run_dir in discover_run_dirs(root):
            row = scan_run_dir(run_dir, project.project, now=now)
            _reconcile_campaign_binding(row, project)
            rows.append(row)

    by_id: dict[str, list[RunIndexRow]] = {}
    for row in rows:
        by_id.setdefault(row.run_id, []).append(row)
    for run_id, candidates in sorted(by_id.items()):
        candidates.sort(key=lambda item: item.run_dir)
        row = candidates[0]
        if len(candidates) > 1:
            _add_issue(row, CampaignRelationship.DUPLICATE_RUN_ID)
            locations = [candidate.run_dir for candidate in candidates]
            row.warnings.append(
                f"duplicate run identity {project.project}/{run_id} found at {locations}"
            )
        previous = index.get_run(row.project, row.run_id)
        prior_wandb = previous.provenance.get("wandb") if previous else None
        current_wandb = row.provenance.get("wandb")
        if (
            isinstance(prior_wandb, dict)
            and prior_wandb.get("initialized")
            and not (
                isinstance(current_wandb, dict)
                and current_wandb.get("initialized")
            )
        ):
            # Runtime URLs are append-only observations. Preserve one captured
            # URL when a later bounded log tail no longer contains its init line.
            row.provenance["wandb"] = prior_wandb
        index.upsert_run(row)
    return len(rows)


_RELATIONSHIP_PRIORITY = (
    CampaignRelationship.DUPLICATE_RUN_ID,
    CampaignRelationship.PROJECT_MISMATCH,
    CampaignRelationship.ORPHANED_CAMPAIGN,
    CampaignRelationship.UNDECLARED_RUN,
    CampaignRelationship.ROLE_MISMATCH,
    CampaignRelationship.CAMPAIGN_REVISION_DRIFT,
    CampaignRelationship.LEGACY_INFERRED,
)


def _refresh_relationship(row: RunIndexRow) -> None:
    row.campaign_binding.relationship = next(
        (item for item in _RELATIONSHIP_PRIORITY
         if item in row.campaign_binding.issues),
        CampaignRelationship.MATCHED,
    )


def _add_issue(row: RunIndexRow, issue: CampaignRelationship) -> None:
    if issue not in row.campaign_binding.issues:
        row.campaign_binding.issues.append(issue)
    _refresh_relationship(row)


def _reconcile_campaign_binding(row: RunIndexRow, project: ResearchProject) -> None:
    """Compare frozen Run provenance with the current authored design.

    Reconciliation annotates the read model only. It never rewrites immutable
    Run identity and never hides historical or orphaned evidence.
    """
    catalog = {campaign.name: campaign for campaign in project.campaigns}
    row.campaign_memberships = []
    for campaign in project.campaigns:
        revision = campaign.current_revision
        if revision is None:
            continue
        membership = next(
            (item for item in revision.memberships if item.run_id == row.run_id), None
        )
        is_origin = campaign.name == row.campaign
        if membership is not None and (is_origin or membership.kind == "reuse"):
            row.campaign_memberships.append(CampaignMembershipBinding(
                campaign=campaign.name,
                revision_id=revision.revision_id,
                membership=membership,
                is_origin=is_origin,
            ))
    ref = catalog.get(row.campaign or "")
    if ref is None:
        _add_issue(row, CampaignRelationship.ORPHANED_CAMPAIGN)
        return
    revision = ref.current_revision
    if revision is None:
        _add_issue(row, CampaignRelationship.LEGACY_INFERRED)
        return

    row.campaign_binding.current_revision = revision.revision_id
    membership = next(
        (item for item in revision.memberships if item.run_id == row.run_id), None
    )
    row.campaign_binding.membership = membership
    if membership is None:
        _add_issue(row, CampaignRelationship.UNDECLARED_RUN)
    elif membership.role:
        if row.role_source == "manifest" and row.role != membership.role:
            _add_issue(row, CampaignRelationship.ROLE_MISMATCH)
        elif row.role_source != "manifest":
            row.role = membership.role
            row.role_source = "campaign_file"

    origin_revision = row.campaign_binding.origin_revision
    if origin_revision is None:
        _add_issue(row, CampaignRelationship.LEGACY_INFERRED)
    elif origin_revision != revision.revision_id:
        _add_issue(row, CampaignRelationship.CAMPAIGN_REVISION_DRIFT)
    _refresh_relationship(row)
