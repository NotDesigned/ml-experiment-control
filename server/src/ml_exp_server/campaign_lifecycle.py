"""Campaign lifecycle read model and immutable scientific record locations."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

from .ingest.indexer import RunIndex
from .schemas import CampaignRelationship, ResearchProject, RunIndexRow


_BLOCKING_RELATIONSHIPS = {
    CampaignRelationship.DUPLICATE_RUN_ID,
    CampaignRelationship.PROJECT_MISMATCH,
    CampaignRelationship.ROLE_MISMATCH,
    CampaignRelationship.UNDECLARED_RUN,
}
_ACTIVE_STATES = {"SUBMITTING", "QUEUED", "STARTING", "RUNNING", "EVALUATING"}
_FAILED_STATES = {"FAILED", "CANCELLED", "PREEMPTED"}
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
    if kind not in {"completion", "archive"}:
        raise ValueError(f"unsupported campaign record kind: {kind}")
    if not _SAFE_ID.fullmatch(revision_id):
        raise ValueError("revision_id is not a safe record identity")
    return campaign_record_root(project, campaign) / f"{revision_id}.{kind}.yml"


def _load_record(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return payload if isinstance(payload, dict) else None


def _membership_role(row: RunIndexRow, campaign: str) -> str | None:
    binding = next(
        (item for item in row.campaign_memberships if item.campaign == campaign), None
    )
    return binding.membership.role if binding and binding.membership.role else row.role


def _stable_evidence(row: RunIndexRow) -> dict[str, Any]:
    layers = {}
    for name in ("scheduler", "worker", "process", "model", "evaluation"):
        layer = getattr(row.evidence, name)
        layers[name] = {
            "state": layer.state,
            "attempt_id": layer.attempt_id,
            "as_of": layer.as_of,
            "source": layer.source,
            "detail": layer.detail,
        }
    return {
        "run_id": row.run_id,
        "scheduler_state": row.scheduler_state,
        "campaign_binding": row.campaign_binding.model_dump(mode="json"),
        "campaign_memberships": [
            item.model_dump(mode="json") for item in row.campaign_memberships
        ],
        "evidence": layers,
        "latest_metrics": row.latest_metrics,
        "eval_metrics": row.eval_metrics,
        "checkpoint": row.checkpoint,
        "artifacts": row.artifacts,
        "decision": row.decision,
        "provenance": row.provenance,
        "evidence_conflicts": row.evidence_conflicts,
    }


def _evidence_digest(revision, rows: list[RunIndexRow]) -> str:
    payload = {
        "revision_id": revision.revision_id,
        "memberships": [item.model_dump(mode="json") for item in revision.memberships],
        "runs": [_stable_evidence(row) for row in sorted(rows, key=lambda item: item.run_id)],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _manifest(row: RunIndexRow) -> dict[str, Any]:
    root = Path(row.run_dir)
    for relative in ("manifest.yaml", "manifest.json", "collected_run/manifest.yaml",
                     "control_manifest.yaml"):
        path = root / relative
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
            payload = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
        except (OSError, json.JSONDecodeError, yaml.YAMLError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _nested(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _required_metrics(contract: dict[str, Any], role: str | None) -> list[str]:
    declared = contract.get("required_metrics")
    if isinstance(declared, list):
        return [str(item) for item in declared]
    if not isinstance(declared, dict):
        return []
    common = declared.get("common") if isinstance(declared.get("common"), list) else []
    by_role = declared.get("by_role") if isinstance(declared.get("by_role"), dict) else {}
    specific = by_role.get(role) if isinstance(by_role.get(role), list) else []
    return list(dict.fromkeys(str(item) for item in [*common, *specific]))


def _required_artifacts(contract: dict[str, Any], role: str | None) -> dict[str, Any]:
    declared = contract.get("required_artifacts")
    if not isinstance(declared, dict):
        return {}
    if "common" not in declared and "by_role" not in declared:
        return declared
    common = declared.get("common") if isinstance(declared.get("common"), dict) else {}
    by_role = declared.get("by_role") if isinstance(declared.get("by_role"), dict) else {}
    specific = by_role.get(role) if isinstance(by_role.get(role), dict) else {}
    return {**common, **specific}


def _check_artifacts(row: RunIndexRow, required: dict[str, Any]) -> list[dict[str, Any]]:
    missing = []
    for name, rule in required.items():
        observed = row.artifacts.get(name)
        observed = observed if isinstance(observed, dict) else {}
        rule = rule if isinstance(rule, dict) else {}
        for field, threshold in (("records", "min_records"),
                                 ("nonempty_records", "min_nonempty_records")):
            minimum = rule.get(threshold)
            if isinstance(minimum, (int, float)) and (observed.get(field) or 0) < minimum:
                missing.append({
                    "artifact": name, "field": field,
                    "required": minimum, "observed": observed.get(field) or 0,
                })
    return missing


def _terminal_check(value: Any, operation: str, expected: Any) -> bool | None:
    if value is None:
        return None
    if operation == "finite":
        return isinstance(value, (int, float)) and math.isfinite(float(value))
    if operation == "nonfinite":
        return isinstance(value, (int, float)) and not math.isfinite(float(value))
    operators = {
        "gt": lambda: value > expected,
        "gte": lambda: value >= expected,
        "lt": lambda: value < expected,
        "lte": lambda: value <= expected,
        "eq": lambda: value == expected,
    }
    try:
        return operators[operation]() if operation in operators else None
    except TypeError:
        return False


def campaign_snapshot(
    index: RunIndex, project: ResearchProject, campaign: str,
) -> dict[str, Any]:
    """Return validation, execution plan, completion readiness, and records."""
    ref = _campaign_ref(project, campaign)
    if ref is None:
        raise KeyError(f"unknown campaign: {campaign}")
    revision = ref.current_revision
    if revision is None:
        return {
            "project": project.project, "campaign": campaign, "revision_id": None,
            "lifecycle_state": "INVALID",
            "validation": {"status": "FAIL", "gates": [
                _gate("current_revision", "FAIL", "campaign has no authored revision"),
            ]},
            "plan": [], "completion": {"ready": False, "gates": []},
            "records": {"completion": None, "archive": None},
        }

    all_rows = {row.run_id: row for row in index.list_runs(project.project)}
    rows = [all_rows[item.run_id] for item in revision.memberships if item.run_id in all_rows]
    validation_gates = [
        _gate("current_revision", "PASS", "authored Campaign revision is resolved",
              {"revision_id": revision.revision_id}),
    ]
    required_roles = []
    has_contract = isinstance(revision.research_contract, dict) and bool(
        revision.research_contract
    )
    contract = revision.research_contract or {}
    validation_gates.append(_gate(
        "research_contract", "PASS" if has_contract else "FAIL",
        "Campaign declares a scientific research contract" if has_contract
        else "Campaign has no research contract; scheduler success cannot establish scientific completion",
    ))
    roles_declared = isinstance(contract.get("required_roles"), list) and bool(
        contract.get("required_roles")
    )
    if roles_declared:
        required_roles = [str(item) for item in contract["required_roles"]]
    analysis_memberships = [
        item for item in revision.memberships if item.included_in_analysis
    ]
    validation_gates.append(_gate(
        "included_memberships", "PASS" if analysis_memberships else "FAIL",
        "Campaign has memberships included in analysis" if analysis_memberships
        else "Campaign has no memberships included in analysis",
    ))
    declared_roles = {item.role for item in analysis_memberships if item.role}
    missing_roles = [role for role in required_roles if role not in declared_roles]
    validation_gates.append(_gate(
        "required_roles", "FAIL" if not has_contract or not roles_declared or missing_roles else "PASS",
        "required research roles cannot be validated without a research contract"
        if not has_contract else "research_contract.required_roles must be a non-empty list"
        if not roles_declared else "required research roles are declared" if not missing_roles
        else "required research roles are missing",
        {"required": required_roles, "missing": missing_roles},
    ))

    plan = []
    for membership in revision.memberships:
        row = all_rows.get(membership.run_id)
        if row is None:
            action = "MATERIALIZE" if membership.kind == "materialize" else "INVALID_REF"
            plan.append({
                **membership.model_dump(mode="json"), "action": action,
                "scheduler_state": None, "detail": "Run is not indexed",
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
        state = (row.scheduler_state or "NOT_SUBMITTED").upper()
        if state in _ACTIVE_STATES:
            action = "WAIT"
        elif state == "SUCCEEDED":
            action = "USE_EVIDENCE"
        elif state in _FAILED_STATES:
            action = "REVIEW_FAILURE"
        elif membership.kind == "materialize":
            action = "SUBMIT"
        else:
            action = "WAIT_FOR_EVIDENCE"
        plan.append({
            **membership.model_dump(mode="json"), "action": action,
            "scheduler_state": row.scheduler_state,
            "relationship": row.campaign_binding.relationship.value,
        })

    validation_status = "FAIL" if any(
        gate["status"] == "FAIL" for gate in validation_gates
    ) else "PASS"

    completion_gates = [_gate(
        "research_contract", "PASS" if has_contract else "FAIL",
        "scientific completion is governed by the Campaign research contract"
        if has_contract else "scientific completion is unavailable without a research contract",
    )]
    included = analysis_memberships
    included_rows = [
        all_rows[item.run_id] for item in included if item.run_id in all_rows
    ]
    missing_runs = [item.run_id for item in included if item.run_id not in all_rows]
    completion_gates.append(_gate(
        "included_runs", "PENDING" if missing_runs else "PASS",
        "all included memberships resolve" if not missing_runs
        else "included memberships are not yet materialized or resolvable",
        {"missing": missing_runs},
    ))
    non_success = {}
    for membership in included:
        row = all_rows.get(membership.run_id)
        if row is not None and (row.scheduler_state or "UNKNOWN").upper() != "SUCCEEDED":
            non_success[row.run_id] = row.scheduler_state or "UNKNOWN"
    completion_gates.append(_gate(
        "terminal_success", "PENDING" if non_success else "PASS",
        "all included Runs succeeded" if not non_success else "included Runs are not successful",
        non_success,
    ))

    missing_metrics: dict[str, list[str]] = {}
    missing_artifacts: dict[str, list[dict[str, Any]]] = {}
    terminal_failures: list[dict[str, Any]] = []
    terminal_pending: list[dict[str, Any]] = []
    for membership in included:
        row = all_rows.get(membership.run_id)
        if row is None:
            continue
        role = membership.role or _membership_role(row, campaign)
        metrics = {**row.latest_metrics, **row.eval_metrics}
        missing = [name for name in _required_metrics(contract, role) if metrics.get(name) is None]
        if missing:
            missing_metrics[row.run_id] = missing
        artifact_missing = _check_artifacts(row, _required_artifacts(contract, role))
        if artifact_missing:
            missing_artifacts[row.run_id] = artifact_missing
        for check in contract.get("terminal_checks") or []:
            if not isinstance(check, dict):
                continue
            metric = str(check.get("metric") or "")
            result = _terminal_check(metrics.get(metric), str(check.get("op") or ""),
                                     check.get("value"))
            item = {"run_id": row.run_id, "metric": metric, "check": check,
                    "observed": metrics.get(metric)}
            if result is False:
                terminal_failures.append(item)
            elif result is None:
                terminal_pending.append(item)
    completion_gates.append(_gate(
        "required_metrics", "PENDING" if missing_metrics else "PASS",
        "required metrics are present" if not missing_metrics else "required metrics are missing",
        missing_metrics,
    ))
    completion_gates.append(_gate(
        "required_artifacts", "PENDING" if missing_artifacts else "PASS",
        "required artifacts are present" if not missing_artifacts else "required artifacts are missing",
        missing_artifacts,
    ))
    terminal_status = "FAIL" if terminal_failures else "PENDING" if terminal_pending else "PASS"
    completion_gates.append(_gate(
        "terminal_checks", terminal_status,
        "terminal research checks pass" if terminal_status == "PASS"
        else "terminal research checks are failed or unresolved",
        {"failures": terminal_failures, "pending": terminal_pending},
    ))

    match_fields = []
    comparison = contract.get("comparison")
    if isinstance(comparison, dict) and isinstance(comparison.get("match_fields"), list):
        match_fields = [str(item) for item in comparison["match_fields"]]
    comparison_missing: dict[str, list[str]] = {}
    comparison_mismatches: dict[str, dict[str, Any]] = {}
    rows_by_id = {row.run_id: row for row in included_rows}
    manifests = {
        membership.run_id: (
            _manifest(rows_by_id[membership.run_id])
            if membership.run_id in rows_by_id else {}
        )
        for membership in included
    }
    for field in match_fields:
        values = {run_id: _nested(manifest, field) for run_id, manifest in manifests.items()}
        missing = [run_id for run_id, value in values.items() if value is None]
        if missing:
            comparison_missing[field] = missing
        present = [value for value in values.values() if value is not None]
        if present and any(value != present[0] for value in present[1:]):
            comparison_mismatches[field] = values
    comparison_status = (
        "PENDING" if len(included) > 1 and not match_fields else
        "FAIL" if comparison_mismatches else "PENDING" if comparison_missing else "PASS"
    )
    completion_gates.append(_gate(
        "comparability", comparison_status,
        "comparison match fields agree" if comparison_status == "PASS"
        else "comparison.match_fields is required for multiple included Runs"
        if len(included) > 1 and not match_fields
        else "comparison identity is mismatched or incomplete",
        {"missing": comparison_missing, "mismatches": comparison_mismatches},
    ))

    evidence_digest = _evidence_digest(revision, included_rows)
    completion_ready = validation_status == "PASS" and all(
        gate["status"] == "PASS" for gate in completion_gates
    )
    completion_path = campaign_record_path(project, campaign, revision.revision_id, "completion")
    archive_path = campaign_record_path(project, campaign, revision.revision_id, "archive")
    completion_record = _load_record(completion_path)
    archive_record = _load_record(archive_path)
    if archive_record:
        lifecycle_state = "ARCHIVED"
    elif completion_record:
        lifecycle_state = "COMPLETED"
    elif validation_status == "FAIL":
        lifecycle_state = "INVALID"
    elif any((row.scheduler_state or "").upper() in _FAILED_STATES for row in included_rows):
        lifecycle_state = "BLOCKED"
    elif any((row.scheduler_state or "").upper() in _ACTIVE_STATES for row in included_rows):
        lifecycle_state = "ACTIVE"
    elif completion_ready:
        lifecycle_state = "COMPLETABLE"
    elif any(item["action"] in {"MATERIALIZE", "SUBMIT"} for item in plan):
        lifecycle_state = "READY"
    else:
        lifecycle_state = "WAITING_EVIDENCE"

    return {
        "project": project.project,
        "campaign": campaign,
        "revision_id": revision.revision_id,
        "lifecycle_state": lifecycle_state,
        "validation": {"status": validation_status, "gates": validation_gates},
        "plan": plan,
        "completion": {
            "ready": completion_ready,
            "evidence_digest": evidence_digest,
            "membership_run_ids": [item.run_id for item in included],
            "gates": completion_gates,
        },
        "records": {
            "completion": completion_record,
            "completion_path": str(completion_path),
            "archive": archive_record,
            "archive_path": str(archive_path),
        },
    }
