"""Campaign catalog read model and immutable archive record locations.

The daemon reports authored memberships and observed Run data.  It does not
interpret metric thresholds as a scientific conclusion or publish Campaign
completion records.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .ingest.indexer import RunIndex
from .schemas import CampaignRelationship, ResearchProject


_BLOCKING_RELATIONSHIPS = {
    CampaignRelationship.DUPLICATE_RUN_ID,
    CampaignRelationship.PROJECT_MISMATCH,
    CampaignRelationship.ROLE_MISMATCH,
    CampaignRelationship.UNDECLARED_RUN,
}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _gate(name: str, status: str, detail: str, evidence: Any = None) -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail, "evidence": evidence}


def _campaign_ref(project: ResearchProject, campaign: str):
    return next((item for item in project.campaigns if item.name == campaign), None)


def campaign_record_root(project: ResearchProject, campaign: str) -> Path:
    if not _SAFE_ID.fullmatch(campaign):
        raise ValueError("campaign is not a safe record identity")
    base = project.base_dir or Path(".")
    return (base / "experiments" / "campaign_records" / campaign).resolve()


def campaign_record_path(
    project: ResearchProject, campaign: str, revision_id: str, kind: str,
) -> Path:
    if kind != "archive":
        raise ValueError(f"unsupported campaign record kind: {kind}")
    if not _SAFE_ID.fullmatch(revision_id):
        raise ValueError("revision_id is not a safe record identity")
    return campaign_record_root(project, campaign) / f"{revision_id}.archive.yml"


def _load_record(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return payload if isinstance(payload, dict) else None


def campaign_snapshot(
    index: RunIndex, project: ResearchProject, campaign: str,
) -> dict[str, Any]:
    """Return authored Campaign membership and observed Run data without verdicts."""
    ref = _campaign_ref(project, campaign)
    if ref is None:
        raise KeyError(f"unknown campaign: {campaign}")
    revision = ref.current_revision
    if revision is None:
        return {
            "project": project.project,
            "campaign": campaign,
            "revision_id": None,
            "lifecycle_state": "INVALID",
            "validation": {"status": "FAIL", "gates": [
                _gate("current_revision", "FAIL", "campaign has no authored revision"),
            ]},
            "runs": [],
            "records": {"archive": None, "archive_path": None},
        }

    all_rows = {row.run_id: row for row in index.list_runs(project.project)}
    validation_gates = [
        _gate("current_revision", "PASS", "authored Campaign revision is resolved",
              {"revision_id": revision.revision_id}),
    ]
    observed_runs = []
    for membership in revision.memberships:
        row = all_rows.get(membership.run_id)
        if row is None:
            observed_runs.append({
                **membership.model_dump(mode="json"),
                "indexed": False,
                "scheduler_state": None,
                "relationship": None,
            })
            if membership.kind == "reuse":
                validation_gates.append(_gate(
                    f"run_ref:{membership.run_id}", "FAIL",
                    "explicit run_ref does not resolve in the Project Run namespace",
                ))
            continue
        if membership.kind == "materialize" and row.campaign != campaign:
            validation_gates.append(_gate(
                f"origin:{membership.run_id}", "FAIL",
                "materialized Run belongs to a different origin Campaign",
                {"origin_campaign": row.campaign},
            ))
        if membership.kind == "materialize" and (
            row.campaign_binding.relationship in _BLOCKING_RELATIONSHIPS
        ):
            validation_gates.append(_gate(
                f"binding:{membership.run_id}", "FAIL",
                "Run has a blocking Campaign identity relationship",
                row.campaign_binding.model_dump(mode="json"),
            ))
        observed_runs.append({
            **membership.model_dump(mode="json"),
            "indexed": True,
            "scheduler_state": row.scheduler_state,
            "relationship": row.campaign_binding.relationship.value,
            "evidence": row.evidence.model_dump(mode="json"),
            "latest_metrics": row.latest_metrics,
            "eval_metrics": row.eval_metrics,
            "eval_variants": row.eval_variants,
            "evaluation_snapshot": row.evaluation_snapshot,
            "checkpoint": row.checkpoint,
            "artifacts": row.artifacts,
            "provenance": row.provenance,
            "warnings": row.warnings,
            "evidence_conflicts": row.evidence_conflicts,
        })

    validation_status = "FAIL" if any(
        gate["status"] == "FAIL" for gate in validation_gates
    ) else "PASS"
    archive_path = campaign_record_path(project, campaign, revision.revision_id, "archive")
    archive_record = _load_record(archive_path)
    lifecycle_state = "ARCHIVED" if archive_record else (
        "INVALID" if validation_status == "FAIL" else "ACTIVE"
    )
    payload = {
        "project": project.project,
        "campaign": campaign,
        "revision_id": revision.revision_id,
        "lifecycle_state": lifecycle_state,
        "validation": {"status": validation_status, "gates": validation_gates},
        "runs": observed_runs,
        "records": {
            "archive": archive_record,
            "archive_path": str(archive_path),
        },
    }
    return payload
