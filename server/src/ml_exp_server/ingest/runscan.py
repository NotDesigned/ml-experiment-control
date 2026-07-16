"""Parse one run directory into a RunIndexRow. Pure functions, no I/O beyond reads.

Layout compatibility: two generations of run directories exist.
  new (canonical): manifest.yaml, attempts/*/attempt.yaml
  old (pre-refactor): control_manifest.yaml, attempts/*/control_attempt.yaml,
      science manifest mirrored under collected_run/manifest.yaml
Root-level status/backend/collection/decision json are mirrors of the current
attempt in both generations.

Evidence layers are kept separate on purpose: a root status.json with a fresh
mtime can still carry a stale RUNNING scheduler snapshot while collection.json
holds newer science metrics. Never collapse them into one state.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

import yaml

from ..schemas import (
    AttemptSummary,
    CampaignBinding,
    CampaignRelationship,
    EvidenceLayer,
    EvidenceLayers,
    RunIndexRow,
)
from ..evidence_conflicts import classify_evidence_conflicts

# collection.json keys that are operational rather than scientific metrics.
_COLLECTION_NON_METRIC_KEYS = {
    "attempt_id", "backend", "collected_from", "project", "run_dir", "run_id",
    "state", "runtime_state", "scheduler_state", "worker_state", "process_state",
    "model_state", "evidence_outcome", "evidence_unavailable_reason",
    "metric_evidence", "evidence_conflicts", "artifacts", "warnings",
    "latest_completed_checkpoint",
}

_EVAL_METRIC_KEYS = (
    "g_ppl", "oracle_plan_ppl", "shuffled_plan_ppl", "plan_ppl_gap",
    "token_recon_ppl", "generation_mean_entropy", "generation_nonempty_fraction",
    "val_bpb",
)

# Evaluation JSONL may contain arbitrary model outputs in addition to summary
# metrics.  The read model intentionally projects only this small, stable set
# and retains at most this many keyed checkpoints per variant.
_EVAL_HISTORY_LIMIT = 32
_EVAL_HISTORY_METRIC_KEYS = (
    "g_ppl", "oracle_plan_ppl", "shuffled_plan_ppl", "plan_ppl_gap",
    "token_recon_ppl", "ppl", "mean_entropy", "generation_mean_entropy",
    "generation_nonempty_fraction", "val_bpb", "bleu", "rouge1", "rouge2",
    "rougeL",
)

_EVAL_PRIMARY_METRICS_BY_MODE = {
    "clean_token_reconstruction": "token_recon_ppl",
    "generation_refine_decode": "g_ppl",
    "oracle_plan_generation": "oracle_plan_ppl",
    "shuffled_plan_generation": "shuffled_plan_ppl",
}
_EVAL_REQUIRED_PRIMARY_METRICS = tuple(_EVAL_PRIMARY_METRICS_BY_MODE.values())
_EVAL_FAMILY_DIMENSION_KEYS = (
    "sampling_method", "num_sampling_steps", "cfg", "self_cond_cfg_scale",
    "time_schedule", "time_warp_gamma",
)


def _evaluation_family_dimensions(record: dict[str, Any]) -> dict[str, Any] | None:
    """Read explicit producer-authored sampling identity without parsing labels."""
    raw = record.get("sampling_config")
    if not isinstance(raw, dict):
        raw = record.get("variant_dimensions")
    if not isinstance(raw, dict):
        return None
    dimensions: dict[str, Any] = {}
    for key in _EVAL_FAMILY_DIMENSION_KEYS:
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return None
        if isinstance(value, float) and not math.isfinite(value):
            return None
        dimensions[key] = value
    return dimensions


def _evaluation_family_id(dimensions: dict[str, Any]) -> str:
    encoded = json.dumps(
        dimensions, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _evaluation_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return the bounded, non-text projection exposed through read APIs."""
    projected: dict[str, Any] = {}
    for key in ("epoch", "step"):
        value = record.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            projected[key] = value
        elif (
            isinstance(value, float) and math.isfinite(value)
            and value.is_integer()
        ):
            projected[key] = int(value)
    mode = record.get("mode")
    if isinstance(mode, str) and len(mode) <= 128:
        projected["mode"] = mode
    dimensions = _evaluation_family_dimensions(record)
    if dimensions is not None:
        projected["sampling_dimensions"] = dimensions
    for key in _EVAL_HISTORY_METRIC_KEYS:
        value = record.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) \
                and math.isfinite(float(value)):
            projected[key] = value
    return projected


def _evaluation_observation(value: Any) -> tuple[tuple[Any, ...], Any | None]:
    """Return a comparison token and safe projected value for one raw metric."""
    if (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        return ("number", float(value)), value
    if isinstance(value, float) and not math.isfinite(value):
        return ("invalid", "nonfinite", repr(value)), None
    return ("invalid", type(value).__name__, repr(value)), None


def _evaluation_metric_observations(
    record: dict[str, Any],
) -> list[tuple[str, Any]]:
    """Canonicalize literal and semantic aliases without hiding disagreement."""
    observations = [
        (key, record[key]) for key in _EVAL_HISTORY_METRIC_KEYS if key in record
    ]
    mode = record.get("mode")
    primary = _EVAL_PRIMARY_METRICS_BY_MODE.get(mode)
    # ``ppl`` is a semantic alias, not a lower-precedence escape hatch.  Keep
    # both observations when a producer writes both spellings so a disagreeing
    # pair becomes explicit conflict evidence; equal pairs remain idempotent.
    if primary is not None and "ppl" in record:
        observations.append((primary, record["ppl"]))
    if mode == "generation_refine_decode" and "mean_entropy" in record:
        observations.append(("generation_mean_entropy", record["mean_entropy"]))
    return observations


def _evaluation_history(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build bounded history without treating conflicting rewrites as corrections."""
    by_identity: dict[tuple[Any, Any], dict[str, Any]] = {}
    observations: dict[tuple[Any, Any], dict[str, tuple[Any, ...]]] = {}
    conflicts: dict[tuple[Any, Any], set[str]] = {}
    skipped = 0
    for record in records:
        projected = _evaluation_record(record)
        step = projected.get("step")
        if step is None:
            skipped += 1
            continue
        identity = (projected.get("epoch"), step)
        merged = by_identity.setdefault(identity, {
            **({"epoch": projected["epoch"]} if "epoch" in projected else {}),
            "step": step,
        })
        seen = observations.setdefault(identity, {})
        conflict_set = conflicts.setdefault(identity, set())

        if "mode" in record:
            mode_token = ("mode", type(record["mode"]).__name__, repr(record["mode"]))
            previous_mode = seen.setdefault("mode", mode_token)
            if previous_mode != mode_token:
                conflict_set.add("mode")
                merged.pop("mode", None)
            elif "mode" in projected and "mode" not in conflict_set:
                merged["mode"] = projected["mode"]

        if "sampling_dimensions" in projected:
            dimensions_token = (
                "sampling_dimensions",
                json.dumps(projected["sampling_dimensions"], sort_keys=True),
            )
            previous_dimensions = seen.setdefault(
                "sampling_dimensions", dimensions_token,
            )
            if previous_dimensions != dimensions_token:
                conflict_set.add("sampling_dimensions")
                merged.pop("sampling_dimensions", None)
            elif "sampling_dimensions" not in conflict_set:
                merged["sampling_dimensions"] = projected["sampling_dimensions"]

        for metric, raw_value in _evaluation_metric_observations(record):
            token, safe_value = _evaluation_observation(raw_value)
            previous = seen.setdefault(metric, token)
            if previous != token:
                conflict_set.add(metric)
                merged.pop(metric, None)
            elif metric not in conflict_set and safe_value is not None:
                merged[metric] = safe_value

    for identity, conflict_set in conflicts.items():
        if conflict_set:
            by_identity[identity]["conflicting_metrics"] = sorted(conflict_set)
    ordered = sorted(
        by_identity.values(),
        key=lambda item: (
            float(item.get("epoch", float("-inf"))),
            float(item["step"]),
        ),
    )
    total = len(ordered)
    omitted = max(0, total - _EVAL_HISTORY_LIMIT)
    return {
        "history": ordered[-_EVAL_HISTORY_LIMIT:],
        "history_total": total,
        "history_limit": _EVAL_HISTORY_LIMIT,
        "history_truncated": omitted > 0,
        "history_omitted_records": omitted,
        "history_skipped_records": skipped,
    }


def _evaluation_variant_family(history: list[dict[str, Any]]) -> dict[str, Any]:
    modes = [record.get("mode") for record in history]
    if modes and all(mode == "clean_token_reconstruction" for mode in modes):
        if any(record.get("sampling_dimensions") is not None for record in history):
            return {
                "status": "CONFLICTING",
                "reason": "family-independent reconstruction carries sampling dimensions",
                "required_producer_fields": list(_EVAL_FAMILY_DIMENSION_KEYS),
            }
        return {
            "status": "RESOLVED",
            "scope": "FAMILY_INDEPENDENT_RECONSTRUCTION",
        }
    sampling_modes = {
        "generation_refine_decode", "oracle_plan_generation",
        "shuffled_plan_generation",
    }
    if not modes or any(mode not in sampling_modes for mode in modes) \
            or len(set(modes)) != 1:
        return {
            "status": "UNRESOLVED",
            "reason": "one variant must contain exactly one recognized evaluation mode",
            "required_producer_fields": list(_EVAL_FAMILY_DIMENSION_KEYS),
        }
    dimensions = [
        record.get("sampling_dimensions") for record in history
    ]
    required = list(_EVAL_FAMILY_DIMENSION_KEYS)
    if not dimensions or any(not isinstance(item, dict) for item in dimensions):
        return {
            "status": "UNRESOLVED",
            "reason": "sampling family dimensions are not present in evaluation records",
            "required_producer_fields": required,
        }
    identities = {
        json.dumps(item, sort_keys=True, separators=(",", ":"))
        for item in dimensions if isinstance(item, dict)
    }
    if len(identities) != 1:
        return {
            "status": "CONFLICTING",
            "reason": "one variant contains multiple structured sampling identities",
            "required_producer_fields": required,
        }
    normalized = dict(dimensions[0])
    return {
        "status": "RESOLVED",
        "scope": "SAMPLING_FAMILY",
        "family_id": _evaluation_family_id(normalized),
        "dimensions": normalized,
    }


def _canonical_eval_variant_id(
    variants: list[dict[str, Any]], contract: Optional[dict[str, Any]],
) -> Optional[str]:
    """Resolve a flat eval view only when its variant identity is unambiguous."""
    declared = None
    if isinstance(contract, dict):
        declared = contract.get("canonical_eval_variant_id")
        evaluation = contract.get("evaluation")
        if declared is None and isinstance(evaluation, dict):
            declared = evaluation.get("canonical_variant_id")
    names = {str(item.get("variant")) for item in variants}
    if isinstance(declared, str) and declared in names:
        return declared
    if len(variants) == 1:
        return str(variants[0].get("variant"))
    return None


def _evaluation_checkpoint_snapshot(
    identity: tuple[Any, Any],
    variants: list[dict[str, Any]],
) -> dict[str, Any]:
    """Project scientific metrics that really belong to one checkpoint.

    Variant JSONL files are written independently.  A checkpoint therefore
    cannot be represented by taking the latest record from every file: during
    an interleaved write those records commonly refer to different steps.
    """
    epoch, step = identity
    candidates: dict[
        str, dict[str, list[tuple[Any, dict[str, Any], dict[str, Any]]]]
    ] = {}
    conflicting_metrics: set[str] = set()
    for variant in variants:
        for record in variant.get("history") or []:
            if not isinstance(record, dict):
                continue
            if (record.get("epoch"), record.get("step")) != identity:
                continue
            record_conflicts = record.get("conflicting_metrics")
            if isinstance(record_conflicts, list):
                conflicting_metrics.update(
                    str(metric) for metric in record_conflicts
                    if isinstance(metric, str)
                )
            mode = record.get("mode")
            primary = _EVAL_PRIMARY_METRICS_BY_MODE.get(mode)
            family = variant.get("evaluation_family")
            family = family if isinstance(family, dict) else {}
            dimensions = record.get("sampling_dimensions")
            if family.get("scope") == "FAMILY_INDEPENDENT_RECONSTRUCTION":
                family_matches = (
                    mode == "clean_token_reconstruction" and dimensions is None
                )
            elif family.get("scope") == "SAMPLING_FAMILY":
                family_matches = (
                    isinstance(dimensions, dict)
                    and dimensions == family.get("dimensions")
                    and family.get("family_id") == _evaluation_family_id(dimensions)
                )
            else:
                # Legacy single-family records have no structured dimensions.
                family_matches = dimensions is None
            if primary is None or not family_matches:
                conflicting_metrics.add("mode" if primary is None else primary)
                continue
            metric_keys = [primary] if primary is not None else []
            if mode == "generation_refine_decode":
                if record.get("generation_mean_entropy") is not None:
                    metric_keys.append("generation_mean_entropy")
                elif record.get("mean_entropy") is not None:
                    metric_keys.append("generation_mean_entropy")
            for metric in metric_keys:
                if "mode" in conflicting_metrics or metric in conflicting_metrics:
                    continue
                value = (
                    record.get("mean_entropy")
                    if metric == "generation_mean_entropy"
                    and record.get("generation_mean_entropy") is None
                    else record.get(metric)
                )
                if value is not None:
                    variant_id = str(variant.get("variant") or "")
                    candidates.setdefault(metric, {}).setdefault(
                        variant_id, [],
                    ).append((value, variant, record))

    metrics: dict[str, Any] = {}
    metric_sources: dict[str, Any] = {}
    for metric, by_variant in sorted(candidates.items()):
        if metric in conflicting_metrics:
            continue
        if len(by_variant) != 1:
            conflicting_metrics.add(metric)
            continue
        observations = next(iter(by_variant.values()))
        values = {float(value) for value, _, _ in observations}
        modes = {record.get("mode") for _, _, record in observations}
        if len(values) != 1 or len(modes) != 1:
            conflicting_metrics.add(metric)
            continue
        value, variant, record = observations[0]
        metrics[metric] = value
        source = {
            "variant_id": variant.get("variant"),
            "mode": record.get("mode"),
            "epoch": epoch,
            "step": step,
        }
        metric_sources[metric] = {
            key: value for key, value in source.items() if value is not None
        }

    if not conflicting_metrics and (
        isinstance(metrics.get("oracle_plan_ppl"), (int, float))
        and isinstance(metrics.get("shuffled_plan_ppl"), (int, float))
    ):
        metrics["plan_ppl_gap"] = (
            metrics["shuffled_plan_ppl"] - metrics["oracle_plan_ppl"]
        )
        metric_sources["plan_ppl_gap"] = {
            "derived_from": ["oracle_plan_ppl", "shuffled_plan_ppl"],
            **({"epoch": epoch} if epoch is not None else {}),
            "step": step,
        }

    missing = [
        metric for metric in _EVAL_REQUIRED_PRIMARY_METRICS
        if metric not in metrics
    ]
    return {
        "state": "COMPLETE" if not missing and not conflicting_metrics else "PARTIAL",
        **({"epoch": epoch} if epoch is not None else {}),
        "step": step,
        "metrics": metrics,
        "metric_sources": metric_sources,
        "missing_metrics": missing,
        "conflicting_metrics": sorted(conflicting_metrics),
    }


def _evaluation_checkpoint_series(variants: list[dict[str, Any]]) -> dict[str, Any]:
    """Build checkpoint projections for one already-resolved family."""
    identities = {
        (record.get("epoch"), record.get("step"))
        for variant in variants
        for record in (variant.get("history") or [])
        if isinstance(record, dict)
        and record.get("epoch") is not None
        and record.get("step") is not None
    }
    ordered = sorted(
        identities,
        key=lambda identity: (
            float(identity[0]) if identity[0] is not None else float("-inf"),
            float(identity[1]),
        ),
    )
    checkpoints = [
        _evaluation_checkpoint_snapshot(identity, variants) for identity in ordered
    ]
    current = checkpoints[-1] if checkpoints else {
        "state": "NOT_OBSERVED",
        "metrics": {},
        "metric_sources": {},
        "missing_metrics": list(_EVAL_REQUIRED_PRIMARY_METRICS),
        "conflicting_metrics": [],
    }
    latest_complete = next(
        (item for item in reversed(checkpoints) if item["state"] == "COMPLETE"),
        None,
    )
    return {
        "required_metrics": list(_EVAL_REQUIRED_PRIMARY_METRICS),
        "current": current,
        "latest_metric_complete": latest_complete,
    }


def _contract_evaluation(contract: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(contract, dict):
        return {}
    evaluation = contract.get("evaluation")
    return evaluation if isinstance(evaluation, dict) else contract


def _declared_evaluation_family(
    variants: list[dict[str, Any]], contract: Optional[dict[str, Any]],
) -> tuple[str | None, str | None]:
    """Return a canonical family declaration and any binding error."""
    configured = _contract_evaluation(contract)
    direct = configured.get("canonical_family_id")
    canonical_family = configured.get("canonical_family")
    if direct is None and isinstance(canonical_family, str):
        direct = canonical_family
    if direct is not None and not isinstance(direct, str):
        return None, "canonical_family_id must be a string"
    dimensions = configured.get("canonical_family_dimensions")
    if dimensions is None and isinstance(canonical_family, dict):
        dimensions = canonical_family
    dimension_id: str | None = None
    if dimensions is not None:
        normalized = _evaluation_family_dimensions({"sampling_config": dimensions})
        if normalized is None:
            return None, "canonical_family_dimensions is incomplete or invalid"
        dimension_id = _evaluation_family_id(normalized)
    variant_id = (
        configured.get("canonical_variant_id")
        or (contract or {}).get("canonical_eval_variant_id")
    )
    variant_family: str | None = None
    if variant_id is not None:
        if not isinstance(variant_id, str):
            return None, "canonical variant identity must be a string"
        matched = next(
            (item for item in variants if item.get("variant") == variant_id), None,
        )
        family = matched.get("evaluation_family") if isinstance(matched, dict) else None
        if not isinstance(family, dict) or family.get("scope") != "SAMPLING_FAMILY":
            return None, "canonical variant does not bind to a resolved sampling family"
        variant_family = str(family.get("family_id"))
    declared = [item for item in (direct, dimension_id, variant_family) if item]
    if len(set(declared)) > 1:
        return None, "canonical family and variant declarations do not bind consistently"
    return (declared[0] if declared else None), None


def evaluation_snapshot(
    variants: list[dict[str, Any]], contract: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build family-aware evaluation science without inferring family from labels."""
    reconstruction = [
        item for item in variants
        if (item.get("evaluation_family") or {}).get("scope")
        == "FAMILY_INDEPENDENT_RECONSTRUCTION"
    ]
    sampling = [item for item in variants if item not in reconstruction]
    resolved: dict[str, list[dict[str, Any]]] = {}
    unresolved: list[str] = []
    for item in sampling:
        family = item.get("evaluation_family")
        family = family if isinstance(family, dict) else {}
        family_id = family.get("family_id")
        if family.get("status") == "RESOLVED" and isinstance(family_id, str):
            resolved.setdefault(family_id, []).append(item)
        else:
            unresolved.append(str(item.get("variant") or "unknown"))

    # A legacy four-slot set is unambiguous without pretending it has a stable
    # family identity. Multiple observations of any conditioning mode require
    # explicit producer-authored dimensions and therefore fail closed.
    legacy_single = False
    if sampling and not resolved:
        mode_counts: dict[str, int] = {}
        for item in sampling:
            mode = (item.get("latest") or {}).get("mode")
            if isinstance(mode, str):
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
        legacy_single = (
            len(reconstruction) == 1
            and len(sampling) == 3
            and mode_counts == {
                "generation_refine_decode": 1,
                "oracle_plan_generation": 1,
                "shuffled_plan_generation": 1,
            }
        )

    families: list[dict[str, Any]] = []
    for family_id, members in sorted(resolved.items()):
        dimensions = (members[0].get("evaluation_family") or {}).get("dimensions")
        family_snapshot = _evaluation_checkpoint_series([*reconstruction, *members])
        families.append({
            "family_id": family_id,
            "status": "RESOLVED",
            "dimensions": dimensions,
            "variant_ids": [str(item.get("variant")) for item in members],
            "reconstruction_variant_ids": [
                str(item.get("variant")) for item in reconstruction
            ],
            **family_snapshot,
        })
    if legacy_single:
        families.append({
            "family_id": None,
            "status": "UNLABELED_SINGLE_FAMILY",
            "dimensions": None,
            "variant_ids": [str(item.get("variant")) for item in sampling],
            "reconstruction_variant_ids": [
                str(item.get("variant")) for item in reconstruction
            ],
            **_evaluation_checkpoint_series(variants),
        })
        unresolved = []

    declared, declaration_error = _declared_evaluation_family(variants, contract)
    selected: dict[str, Any] | None = None
    if declaration_error:
        family_state = "CANONICAL_BINDING_CONFLICT"
    elif unresolved:
        family_state = "UNRESOLVED"
    elif declared is not None:
        selected = next(
            (item for item in families if item.get("family_id") == declared), None,
        )
        family_state = "DECLARED" if selected is not None else "CANONICAL_NOT_FOUND"
    elif len(families) == 1:
        selected = families[0]
        family_state = "SINGLE_ELIGIBLE_FAMILY"
    elif len(families) > 1:
        family_state = "CANONICAL_NOT_DECLARED"
    else:
        family_state = "NOT_OBSERVED"

    if selected is None:
        current = {
            "state": family_state,
            "metrics": {}, "metric_sources": {},
            "missing_metrics": list(_EVAL_REQUIRED_PRIMARY_METRICS),
            "conflicting_metrics": [],
        }
        latest_complete = None
    else:
        current = selected["current"]
        latest_complete = selected["latest_metric_complete"]

    return {
        "schema_version": 1,
        "required_metrics": list(_EVAL_REQUIRED_PRIMARY_METRICS),
        "family_state": family_state,
        "required_family_dimension_fields": list(_EVAL_FAMILY_DIMENSION_KEYS),
        "unresolved_variant_ids": unresolved,
        "canonical_family_id": selected.get("family_id") if selected else None,
        "canonical_declaration_error": declaration_error,
        "families": families,
        "current": current,
        "latest_metric_complete": latest_complete,
    }


def _bind_evaluation_snapshot_to_attempt(
    snapshot: dict[str, Any], *, attempt_id: Optional[str], exact: bool,
) -> dict[str, Any]:
    """Suppress top-level science unless raw evidence binds to one Attempt."""
    if exact:
        return {
            **snapshot,
            "attempt_binding_state": "EXACT_ATTEMPT_BOUND",
            "source_attempt_id": attempt_id,
            "unresolved_evidence": [],
        }
    current = snapshot.get("current")
    current = current if isinstance(current, dict) else {}
    return {
        **snapshot,
        "attempt_binding_state": "EXACT_ATTEMPT_NOT_BOUND",
        "source_attempt_id": attempt_id,
        "unresolved_evidence": ["exact_attempt_binding"],
        "current": {
            **current,
            "state": "EXACT_ATTEMPT_NOT_BOUND",
            "metrics": {},
            "metric_sources": {},
            "missing_metrics": list(snapshot.get("required_metrics") or []),
        },
        "latest_metric_complete": None,
    }

_ROLE_PATTERN = re.compile(r"^[a-z]+-([a-z]\d+)-")

_PROVENANCE_KEYS = ("git_commit", "source_id", "image_id", "config_path", "seed",
                    "campaign_id", "runtime_tree_id", "created_at")
_CONFIG_EXCERPT_KEYS = (
    "seed", "max_length", "global_batch_size", "epochs", "grad_accum_steps",
    "use_sentence_plan", "sentence_encoder_type", "sentence_encoder_grad",
    "plan_aux_passes", "plan_aux_token_context",
    "use_wandb", "wandb_base_url", "wandb_project", "wandb_entity",
    "wandb_run_id", "wandb_run_name", "wandb_url",
    "depth", "device_batch_size",
)

_CHECKPOINT_KEYS = (
    "latest_completed_checkpoint",
    "latest_completed_checkpoint_step",
    "checkpoint_exposure",
    "checkpoint_exposure_minutes",
)
_WANDB_INIT_PATTERN = re.compile(
    r"wandb initialized:\s*(https?://[^\s)>\]\"'?#]+)", re.IGNORECASE
)
_WANDB_LOG_SCAN_LIMIT = 8 * 1024 * 1024


def parse_iso_ts(value: Any) -> Optional[float]:
    """Parse an ISO-8601 timestamp (as written in status.json/events.jsonl) to unix seconds."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text())
    except (yaml.YAMLError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _safe_wandb_url(value: Any) -> Optional[str]:
    if not isinstance(value, str) or len(value) > 2048:
        return None
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return None
    return value


def _wandb_url_in_text(value: str) -> Optional[str]:
    match = _WANDB_INIT_PATTERN.search(value)
    return _safe_wandb_url(match.group(1)) if match else None


def _wandb_url_in_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return _wandb_url_in_text(handle.read(_WANDB_LOG_SCAN_LIMIT))
    except OSError:
        return None


def _wandb_provenance(
    run_dir: Path,
    attempt_dir: Optional[Path],
    manifest: dict[str, Any],
    collection: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Resolve requested W&B identity and observed runtime URL from durable evidence."""
    resolved = manifest.get("resolved_config")
    resolved = resolved if isinstance(resolved, dict) else {}
    requested = _truthy(resolved.get("use_wandb"))
    result: dict[str, Any] = {
        "requested": requested,
        "enabled": requested,
        "initialized": False,
        "entity": resolved.get("wandb_entity"),
        "project": resolved.get("wandb_project"),
        "run_id": resolved.get("wandb_run_id") or manifest.get("run_id"),
        "name": resolved.get("wandb_run_name") or manifest.get("run_id"),
    }

    attempt_collection = (
        _load_json(attempt_dir / "collection.json") if attempt_dir is not None else {}
    )
    for source, payload in (
        ("attempt.collection.wandb", attempt_collection.get("wandb")),
        ("run.collection.wandb", collection.get("wandb")),
    ):
        observed = payload if isinstance(payload, dict) else {}
        url = _safe_wandb_url(observed.get("url"))
        if url is not None:
            result.update({
                "initialized": bool(observed.get("initialized", True)),
                "url": url,
                "evidence_source": observed.get("evidence_source") or source,
            })
            return result

    roots = [path for path in (attempt_dir, run_dir) if path is not None]
    structured_candidates = []
    log_candidates = []
    for root in roots:
        structured_candidates.extend((
            root / "wandb.json",
            root / "collected_run" / "wandb.json",
        ))
        log_candidates.extend((
            root / "stdout.log", root / "stderr.log",
            root / "collected_run" / "stdout.log",
            root / "collected_run" / "stderr.log",
        ))

    for path in structured_candidates:
        observed = _load_json(path)
        url = _safe_wandb_url(observed.get("url") or observed.get("run_url"))
        if url is not None:
            result.update({
                "initialized": bool(observed.get("initialized", True)),
                "url": url,
                "evidence_source": str(path),
            })
            for key in ("entity", "project", "run_id", "name"):
                if observed.get(key) is not None:
                    result[key] = observed[key]
            return result

    for path in log_candidates:
        url = _wandb_url_in_file(path)
        if url:
            result.update({
                "initialized": True,
                "url": url,
                "evidence_source": str(path),
            })
            return result

    process = collection.get("process_evidence")
    process = process if isinstance(process, dict) else {}
    for stream in ("stdout_tail", "stderr_tail"):
        lines = process.get(stream)
        if not isinstance(lines, list):
            continue
        url = _wandb_url_in_text("\n".join(str(item) for item in lines))
        if url:
            result.update({
                "initialized": True,
                "url": url,
                "evidence_source": f"collection.process_evidence.{stream}",
            })
            return result

    return result if requested else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping unparsable lines (partial writes on live runs)."""
    if not path.is_file():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


@dataclass(frozen=True)
class EvidenceSource:
    """One ordered filesystem source for execution evidence."""

    root: Path
    attempt_id: Optional[str]
    kind: str


def _safe_attempt_id(run_dir: Path, value: object) -> Optional[str]:
    attempts_dir = run_dir / "attempts"
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value,
    ):
        return None
    candidate = attempts_dir / value
    try:
        resolved_run = run_dir.resolve()
        resolved_root = attempts_dir.resolve()
        resolved = candidate.resolve()
    except OSError:
        return None
    if (
        attempts_dir.is_symlink()
        or resolved_root.parent != resolved_run
        or candidate.is_symlink()
        or resolved.parent != resolved_root
        or not resolved.is_dir()
    ):
        return None
    return value


def preferred_attempt_id(run_dir: Path) -> Optional[str]:
    """Resolve the current Attempt without inferring it from scientific metrics.

    Root status/collection/decision files are controller-maintained mirrors of
    the current Attempt. If none names an existing Attempt, use the last
    append-only attempt directory as a deterministic fallback.
    """
    attempts_dir = run_dir / "attempts"
    for mirror in (run_dir / "status.json", run_dir / "collection.json",
                   run_dir / "decision.json"):
        attempt_id = _safe_attempt_id(run_dir, _load_json(mirror).get("attempt_id"))
        if attempt_id is not None:
            return attempt_id
    if not attempts_dir.is_dir():
        return None
    attempts = sorted(
        value for path in attempts_dir.iterdir()
        if (value := _safe_attempt_id(run_dir, path.name)) is not None
    )
    return attempts[-1] if attempts else None


def evidence_sources(run_dir: Path, *, attempt_id: Optional[str] = None,
                     exact_attempt: bool = False) -> list[EvidenceSource]:
    """Return evidence roots in precedence order.

    Exact Attempt queries never fall back to another Attempt or a Run mirror.
    Run aggregation first reads the selected Attempt and uses root mirrors only
    for legacy/incomplete layouts.
    """
    selected = (
        _safe_attempt_id(run_dir, attempt_id)
        if attempt_id is not None else preferred_attempt_id(run_dir)
    )
    sources: list[EvidenceSource] = []
    if selected:
        attempt_root = run_dir / "attempts" / selected
        # ``selected`` is returned only by ``_safe_attempt_id``, which already
        # verifies that the resolved Attempt directory exists and is contained.
        sources.extend([
            EvidenceSource(attempt_root / "collected_run", selected,
                           "attempt_collected"),
            EvidenceSource(attempt_root, selected, "attempt_local"),
        ])
    if exact_attempt:
        return sources

    mirror_attempt = _load_json(run_dir / "collection.json").get("attempt_id")
    include_mirrors = True
    if not isinstance(mirror_attempt, str):
        # A Run-level mirror without an Attempt binding is safe only for a
        # legacy/single-Attempt layout.  During retry submission the controller
        # updates status.json before collection.json; treating the still-old
        # collection as the newly selected Attempt can otherwise expose the
        # previous Attempt's metrics under the retry identity.
        attempts_dir = run_dir / "attempts"
        known_attempts = (
            [
                value for path in attempts_dir.iterdir()
                if (value := _safe_attempt_id(run_dir, path.name)) is not None
            ]
            if attempts_dir.is_dir() else []
        )
        if len(known_attempts) <= 1:
            mirror_attempt = selected
        else:
            include_mirrors = False
    if include_mirrors:
        sources.extend([
            EvidenceSource(run_dir / "collected_run", mirror_attempt, "run_mirror"),
            EvidenceSource(run_dir, mirror_attempt, "run_root"),
        ])
    # Attempt roots and Run mirrors are disjoint by construction.
    return sources


def collection_latest_metric(collection: dict[str, Any]) -> dict[str, Any]:
    """Return the finite scalar metric record published by a controller.

    Project controllers commonly cannot mirror a remote ``train_metrics.jsonl``
    file, but can still return its latest parsed record in
    ``collection.json.latest_metric``.  Keep this projection deliberately
    bounded and scalar-only: collection payloads can also contain logs,
    artifacts, and other unbounded backend data that must never enter the
    metric API.
    """
    raw = collection.get("latest_metric")
    if not isinstance(raw, dict):
        return {}
    record: dict[str, Any] = {}
    for key, value in list(raw.items())[:256]:
        if not isinstance(key, str) or not key or len(key) > 128:
            continue
        if isinstance(value, bool):
            record[key] = value
        elif isinstance(value, int):
            record[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            record[key] = value
        elif isinstance(value, str) and len(value) <= 1024:
            record[key] = value
    step = record.get("step")
    if isinstance(step, float) and step.is_integer():
        record["step"] = int(step)
    return record


def train_metric_records(
    run_dir: Path, *, attempt_id: Optional[str] = None, exact_attempt: bool = False,
) -> tuple[list[dict[str, Any]], Optional[Path], Optional[str]]:
    """Read training metrics from one coherent Attempt-scoped source."""
    sources = evidence_sources(
        run_dir, attempt_id=attempt_id, exact_attempt=exact_attempt,
    )
    # Prefer producer-authored histories across every applicable evidence root.
    # A daemon-derived latest-point history must not shadow a real metric file
    # that becomes available later in a lower-precedence mirror.
    for source in sources:
        for filename in ("train_metrics.jsonl", "metrics.jsonl"):
            path = source.root / filename
            records = read_jsonl(path)
            if records:
                return records, path, source.attempt_id
    for source in sources:
        path = source.root / "observed_train_metrics.jsonl"
        records = read_jsonl(path)
        if records:
            return records, path, source.attempt_id
    # A newly observed Run is readable immediately, even before the collector
    # has accumulated its first daemon-owned history point.
    for source in sources:
        path = source.root / "collection.json"
        collection = _load_json(path)
        bound_attempt = collection.get("attempt_id")
        if (
            isinstance(bound_attempt, str)
            and source.attempt_id is not None
            and bound_attempt != source.attempt_id
        ):
            continue
        record = collection_latest_metric(collection)
        if record:
            return [record], path, source.attempt_id
    return [], None, attempt_id


def evaluation_variants(
    run_dir: Path, *, attempt_id: Optional[str] = None, exact_attempt: bool = False,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Read unique evaluation variants from all applicable Attempt evidence.

    Collectors may place variants both directly under collected_run/ and under
    train_sampling_eval/. Duplicate names are merged before checkpoint
    projection so identical observations remain idempotent while conflicting
    cross-source rewrites remain visible.
    Each variant exposes a bounded, whitelisted epoch+step history; arbitrary
    JSONL fields are never copied into the read model. Exact Attempt roots are
    merged, while evidence is never merged across different Attempts.
    """
    selected_attempt = attempt_id or preferred_attempt_id(run_dir)
    by_name: dict[str, dict[str, Any]] = {}
    seen_paths: set[Path] = set()
    for source in evidence_sources(
        run_dir, attempt_id=attempt_id, exact_attempt=exact_attempt,
    ):
        # Run mirrors are applicable only when they bind to the selected
        # Attempt.  Exact queries already omit mirrors, but retain this guard so
        # future evidence-root additions cannot accidentally mix Attempts.
        if (
            selected_attempt is not None
            and source.attempt_id is not None
            and source.attempt_id != selected_attempt
        ):
            continue
        candidates = list((source.root / "train_sampling_eval").glob("*/metrics.jsonl"))
        candidates.extend(source.root.glob("*/metrics.jsonl"))
        for path in sorted(set(candidates)):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            records = read_jsonl(path)
            if not records:
                continue
            aggregate = by_name.setdefault(path.parent.name, {
                "records": [], "sources": [],
            })
            aggregate["records"].extend(records)
            aggregate["sources"].append(str(path))
    variants = []
    for name in sorted(by_name):
        aggregate = by_name[name]
        history = _evaluation_history(aggregate["records"])
        bounded_history = history["history"]
        sources = sorted(set(aggregate["sources"]))
        variants.append({
            "variant": name,
            "latest": bounded_history[-1] if bounded_history else {},
            "records": len(aggregate["records"]),
            "source": sources[0] if len(sources) == 1 else None,
            "sources": sources,
            "evaluation_family": _evaluation_variant_family(bounded_history),
            **history,
        })
    return variants, selected_attempt


def is_run_dir(path: Path) -> bool:
    """A run directory carries a manifest in either generation's spelling."""
    return (
        (path / "manifest.yaml").is_file()
        or (path / "manifest.json").is_file()
        or (path / "control_manifest.yaml").is_file()
        or (path / "collected_run" / "manifest.yaml").is_file()
    )


def _science_manifest(run_dir: Path) -> dict[str, Any]:
    """Prefer the science manifest; older dirs mirror it under collected_run/."""
    for candidate in (run_dir / "manifest.yaml", run_dir / "manifest.json",
                      run_dir / "collected_run" / "manifest.yaml",
                      run_dir / "control_manifest.yaml"):
        data = _load_json(candidate) if candidate.suffix == ".json" else _load_yaml(candidate)
        if data:
            return data
    return {}


def _infer_role(run_id: str) -> Optional[str]:
    match = _ROLE_PATTERN.match(run_id)
    return match.group(1) if match else None


def _mtime(path: Path) -> Optional[float]:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _autoresearch_source_parameters(run_dir: Path) -> dict[str, int]:
    """Read a tiny allowlist of immutable constants from a harness snapshot."""
    path = run_dir / "source" / "train.py"
    if not path.is_file():
        return {}
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    parameters = {}
    for authored, canonical in (("DEPTH", "depth"),
                                ("DEVICE_BATCH_SIZE", "device_batch_size")):
        match = re.search(rf"^\s*{authored}\s*=\s*(\d+)\b", source, re.MULTILINE)
        if match:
            parameters[canonical] = int(match.group(1))
    return parameters


def _scan_attempts(run_dir: Path) -> list[AttemptSummary]:
    attempts_dir = run_dir / "attempts"
    if not attempts_dir.is_dir():
        return []
    summaries = []
    for attempt_dir in sorted(attempts_dir.iterdir()):
        if not attempt_dir.is_dir():
            continue
        status = _load_json(attempt_dir / "status.json") or _load_json(attempt_dir / "attempt.json")
        backend = _load_json(attempt_dir / "backend.json")
        submission = _load_json(attempt_dir / "submission.json")
        summaries.append(AttemptSummary(
            attempt_id=attempt_dir.name,
            state=status.get("state"),
            backend=backend.get("backend") or status.get("backend")
            or ("local-cuda" if submission.get("gpu") is not None else None),
            backend_job_id=backend.get("backend_job_id") or status.get("backend_job_id"),
            decision=_observation_decision(_load_json(attempt_dir / "decision.json")),
            # ``job.sbatch`` is produced while preparing an Attempt and is
            # therefore not evidence that the scheduler mutation happened.
            # Only the durable submission receipt crosses that boundary.
            has_submission=(attempt_dir / "submission.json").is_file(),
        ))
    return summaries


def _observation_decision(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep operational decision metadata, not a hypothesis verdict."""
    return {
        key: payload[key]
        for key in (
            "action", "reason", "failure_class", "resume_checkpoint",
            "retries_allowed", "retries_used",
        )
        if payload.get(key) is not None
    }


def _decision_history(run_dir: Path) -> list[dict[str, Any]]:
    """Return durable decision snapshots with honest source timestamps.

    Newer producers may append decision events or write attempt-local decision
    files. Older runs only have the root mirror, so that remains a single
    snapshot rather than being presented as a fabricated lifecycle timeline.
    """
    history: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_payloads: set[str] = set()

    def add(decision: dict[str, Any], *, source: Path, attempt_id: Any = None,
            timestamp: Optional[float] = None, mirror: bool = False) -> None:
        decision = _observation_decision(decision)
        if not decision:
            return
        payload_identity = json.dumps(decision, sort_keys=True, default=str)
        if mirror and payload_identity in seen_payloads:
            return
        identity = json.dumps({
            "attempt_id": attempt_id,
            "decision": decision,
            "timestamp": timestamp,
        }, sort_keys=True, default=str)
        if identity in seen:
            return
        seen.add(identity)
        seen_payloads.add(payload_identity)
        history.append({
            "ts": timestamp if timestamp is not None else _mtime(source),
            "attempt_id": attempt_id,
            "source": str(source),
            "action": decision.get("action"),
            "reason": decision.get("reason"),
            "failure_class": decision.get("failure_class"),
        })

    events_path = run_dir / "events.jsonl"
    for event in read_jsonl(events_path):
        event_name = str(event.get("event") or event.get("event_type") or "")
        if "decision" not in event_name.lower():
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        decision = payload.get("decision", payload)
        if isinstance(decision, dict):
            add(
                decision,
                source=events_path,
                attempt_id=event.get("attempt_id"),
                timestamp=parse_iso_ts(event.get("timestamp")),
            )

    attempts_dir = run_dir / "attempts"
    if attempts_dir.is_dir():
        for attempt_dir in sorted(attempts_dir.iterdir()):
            path = attempt_dir / "decision.json"
            add(_load_json(path), source=path, attempt_id=attempt_dir.name)

    root_path = run_dir / "decision.json"
    add(_load_json(root_path), source=root_path, mirror=True)
    history.sort(key=lambda item: item.get("ts") or 0)
    return history


def _latest_train_record(
    run_dir: Path,
) -> tuple[dict[str, Any], Optional[Path], Optional[str]]:
    records, path, attempt_id = train_metric_records(run_dir)
    return (records[-1] if records else {}), path, attempt_id


def _eval_layer(
    variants: list[dict[str, Any]], attempt_id: Optional[str],
) -> EvidenceLayer:
    """Latest observed evaluation checkpoint, independent of model progress."""
    layer = EvidenceLayer()
    best_ts: Optional[float] = None
    layer.attempt_id = attempt_id
    for variant in variants:
        sources = variant.get("sources")
        source_paths = [Path(item) for item in sources if isinstance(item, str)] \
            if isinstance(sources, list) else []
        if not source_paths and isinstance(variant.get("source"), str):
            source_paths = [Path(variant["source"])]
        timestamps = [ts for path in source_paths if (ts := _mtime(path)) is not None]
        ts = max(timestamps) if timestamps else None
        layer.detail[variant["variant"]] = variant["latest"]
        if ts is not None and (best_ts is None or ts > best_ts):
            best_ts = ts
            layer.source = str(max(source_paths, key=lambda path: _mtime(path) or 0))
    current = evaluation_snapshot(variants)["current"]
    if current.get("step") is not None:
        layer.state = f"step {current['step']} · {current['state']}"
    elif variants:
        layer.state = "present"
    layer.as_of = best_ts
    return layer


def scan_run_dir(run_dir: Path, project: str, *, campaign: Optional[str] = None,
                 now: Optional[float] = None) -> RunIndexRow:
    """Build the read model for one run directory.

    Args:
        run_dir: the run directory (either file-name generation).
        project: project name the run root belongs to.
        campaign: campaign name; defaults to the parent directory name, which is
            the canonical `<local_root>/<campaign>/<run_id>` layout.
        now: injected clock for freshness (tests); defaults to time.time().
    """
    from .freshness import apply_freshness  # local import to avoid cycle

    manifest = _science_manifest(run_dir)
    status = _load_json(run_dir / "status.json")
    collection = _load_json(run_dir / "collection.json")
    decision = _observation_decision(_load_json(run_dir / "decision.json"))
    collected_status = _load_json(run_dir / "collected_run" / "status.json")
    run_id = manifest.get("run_id") or status.get("run_id") or run_dir.name
    selected_attempt = preferred_attempt_id(run_dir)
    attempt_dir = run_dir / "attempts" / selected_attempt if selected_attempt else None
    attempt_record = _load_json(attempt_dir / "attempt.json") if attempt_dir else {}
    attempt_summary = _load_json(attempt_dir / "summary.json") if attempt_dir else {}

    if manifest.get("campaign"):
        campaign_name = manifest["campaign"]
        campaign_source = "manifest"
    elif campaign:
        campaign_name = campaign
        campaign_source = "argument"
    else:
        campaign_name = run_dir.parent.name
        campaign_source = "directory"

    binding_issues: list[CampaignRelationship] = []
    manifest_project = manifest.get("project")
    if manifest_project and manifest_project != project:
        binding_issues.append(CampaignRelationship.PROJECT_MISMATCH)
    if campaign_source != "manifest" or not manifest_project:
        binding_issues.append(CampaignRelationship.LEGACY_INFERRED)
    initial_relationship = (
        CampaignRelationship.PROJECT_MISMATCH
        if CampaignRelationship.PROJECT_MISMATCH in binding_issues
        else CampaignRelationship.LEGACY_INFERRED
        if binding_issues
        else CampaignRelationship.UNRESOLVED
    )
    campaign_binding = CampaignBinding(
        relationship=initial_relationship,
        issues=binding_issues,
        origin_project=manifest_project,
        origin_campaign=manifest.get("campaign"),
        origin_revision=manifest.get("campaign_id"),
    )

    role = manifest.get("research_role")
    role_source = "manifest" if role else None
    if role is None:
        role = _infer_role(run_id)
        role_source = "heuristic" if role else None

    evidence = EvidenceLayers()

    # Scheduler layer: root status.json (mirror of current attempt).
    if status:
        evidence.scheduler = EvidenceLayer(
            state=status.get("state"),
            attempt_id=status.get("attempt_id"),
            as_of=parse_iso_ts(status.get("updated_at")) or _mtime(run_dir / "status.json"),
            source=str(run_dir / "status.json"),
            detail={k: status[k] for k in ("raw_state", "backend", "backend_job_id",
                                           "partition", "elapsed", "exit_code")
                    if status.get(k) is not None},
        )

    # Worker layer: remote-side status mirrored into collected_run/, or the
    # newer collection.json worker_state field.
    if collection.get("worker_state") is not None:
        evidence.worker = EvidenceLayer(
            state=collection.get("worker_state"),
            attempt_id=collection.get("attempt_id"),
            as_of=_mtime(run_dir / "collection.json"),
            source=str(run_dir / "collection.json"),
        )
    elif collected_status:
        evidence.worker = EvidenceLayer(
            state=collected_status.get("state"),
            attempt_id=collected_status.get("attempt_id"),
            as_of=parse_iso_ts(collected_status.get("updated_at"))
            or _mtime(run_dir / "collected_run" / "status.json"),
            source=str(run_dir / "collected_run" / "status.json"),
            detail={k: collected_status[k] for k in ("attempt_id",) if collected_status.get(k)},
        )

    # Process layer: only the newer collection format separates it.
    if collection.get("process_state") is not None:
        evidence.process = EvidenceLayer(
            state=collection.get("process_state"),
            attempt_id=collection.get("attempt_id"),
            as_of=_mtime(run_dir / "collection.json"),
            source=str(run_dir / "collection.json"),
        )
    elif attempt_record:
        evidence.worker = EvidenceLayer(
            state="LOCAL_GPU",
            attempt_id=selected_attempt,
            as_of=parse_iso_ts(attempt_record.get("finished_at")
                               or attempt_record.get("started_at")),
            source=str(attempt_dir / "attempt.json"),
        )
        evidence.process = EvidenceLayer(
            state=attempt_record.get("state"),
            attempt_id=selected_attempt,
            as_of=parse_iso_ts(attempt_record.get("finished_at")
                               or attempt_record.get("started_at")),
            source=str(attempt_dir / "attempt.json"),
            detail={key: attempt_record[key] for key in
                    ("return_code", "failure_class", "launch_error")
                    if attempt_record.get(key) is not None},
        )

    # Model layer: last train_metrics record; its own timestamp is the evidence time.
    train_record, train_path, train_attempt_id = _latest_train_record(run_dir)
    if train_record:
        ts = train_record.get("timestamp")
        evidence.model = EvidenceLayer(
            state=f"step {train_record.get('step')}" if train_record.get("step") is not None else "present",
            attempt_id=train_attempt_id,
            as_of=float(ts) if isinstance(ts, (int, float)) else (_mtime(train_path) if train_path else None),
            source=str(train_path) if train_path else None,
            detail={k: v for k, v in train_record.items() if k != "timestamp"},
        )
    elif collection.get("model_state") is not None:
        evidence.model = EvidenceLayer(
            state=collection.get("model_state"),
            attempt_id=collection.get("attempt_id"),
            as_of=_mtime(run_dir / "collection.json"),
            source=str(run_dir / "collection.json"),
        )

    contract = manifest.get("research_contract")
    if not isinstance(contract, dict):
        contract = None
    eval_variants, eval_attempt_id = evaluation_variants(run_dir)
    eval_snapshot = _bind_evaluation_snapshot_to_attempt(
        evaluation_snapshot(eval_variants, contract),
        attempt_id=eval_attempt_id,
        exact=(
            selected_attempt is not None
            and eval_attempt_id == selected_attempt
        ),
    )
    evidence.evaluation = _eval_layer(eval_variants, eval_attempt_id)
    harness_metrics = status.get("metrics") if isinstance(status.get("metrics"), dict) else {}
    if not harness_metrics and isinstance(attempt_summary.get("metrics"), dict):
        harness_metrics = attempt_summary["metrics"]
    if harness_metrics.get("val_bpb") is not None:
        evidence.evaluation = EvidenceLayer(
            state="OBSERVED",
            attempt_id=selected_attempt,
            as_of=parse_iso_ts(attempt_summary.get("collected_at"))
            or _mtime(attempt_dir / "summary.json"),
            source=str(attempt_dir / "summary.json"),
            detail={"val_bpb": harness_metrics.get("val_bpb")},
        )

    latest_metrics: dict[str, Any] = {}
    for key, value in collection.items():
        if key in _COLLECTION_NON_METRIC_KEYS or not isinstance(value, (int, float, str, bool)):
            continue
        if key not in _EVAL_METRIC_KEYS:
            latest_metrics[key] = value
    for key, value in collection_latest_metric(collection).items():
        if key != "timestamp" and key not in _EVAL_METRIC_KEYS:
            latest_metrics[key] = value
    # train_metrics.jsonl is fresher than collection.json when both exist.
    for key, value in train_record.items():
        if key != "timestamp":
            latest_metrics[key] = value
    for key, value in harness_metrics.items():
        if key not in _EVAL_METRIC_KEYS and isinstance(value, (int, float, str, bool)):
            latest_metrics[key] = value

    provenance = {k: manifest[k] for k in _PROVENANCE_KEYS if manifest.get(k) is not None}
    source_identity = manifest.get("source")
    if isinstance(source_identity, dict):
        provenance.update({
            "git_commit": source_identity.get("git_commit"),
            "source_id": source_identity.get("source_sha256"),
            "config_path": "train.py",
            "source_origin": source_identity.get("origin"),
            "train_py_sha256": source_identity.get("train_py_sha256"),
        })
        provenance = {key: value for key, value in provenance.items() if value is not None}
        source_parameters = _autoresearch_source_parameters(run_dir)
        if source_parameters:
            provenance["resolved_config_excerpt"] = source_parameters
    resolved = manifest.get("resolved_config")
    if isinstance(resolved, dict):
        provenance["resolved_config_excerpt"] = {
            k: resolved[k] for k in _CONFIG_EXCERPT_KEYS if k in resolved
        }
    wandb = _wandb_provenance(run_dir, attempt_dir, manifest, collection)
    if wandb is not None:
        provenance["wandb"] = wandb

    canonical_eval_variant_id = _canonical_eval_variant_id(eval_variants, contract)
    eval_metrics: dict[str, Any] = {}
    complete_eval = eval_snapshot.get("latest_metric_complete")
    if isinstance(complete_eval, dict):
        complete_metrics = complete_eval.get("metrics")
        if isinstance(complete_metrics, dict):
            eval_metrics = {
                key: complete_metrics[key] for key in _EVAL_METRIC_KEYS
                if complete_metrics.get(key) is not None
            }
    elif (
        selected_attempt is not None
        and not eval_variants
        and not collection.get("evidence_conflicts")
    ):
        # Legacy collectors sometimes provide one unnamed eval record only.
        eval_metrics = {
            key: collection[key] for key in _EVAL_METRIC_KEYS
            if isinstance(collection.get(key), (int, float, str, bool))
        }
        if harness_metrics.get("val_bpb") is not None:
            eval_metrics["val_bpb"] = harness_metrics["val_bpb"]
    conflicts, reclassified_conflicts = classify_evidence_conflicts(
        collection.get("evidence_conflicts"),
        project=project, run_id=str(run_id), attempt_id=selected_attempt,
    )
    warnings = [
        warning for warning in list(collection.get("warnings") or [])
        if warning not in conflicts
    ]
    if reclassified_conflicts:
        warnings.append(
            "cross-variant or cross-family evaluation values retained as "
            "distinct evidence, not conflicts"
        )
    if eval_variants and eval_snapshot.get("family_state") in {
        "UNRESOLVED", "CANONICAL_NOT_DECLARED", "CANONICAL_BINDING_CONFLICT",
        "CANONICAL_NOT_FOUND",
    }:
        warnings.append(
            "evaluation families are not canonical; flat eval_metrics suppressed: "
            f"{eval_snapshot.get('family_state')}"
        )
    checkpoint = {
        key: collection[key] for key in _CHECKPOINT_KEYS
        if collection.get(key) is not None
    }
    artifacts = collection.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    if attempt_summary:
        artifacts.setdefault("summary", {
            "records": 1,
            "nonempty_records": 1 if harness_metrics else 0,
        })
        integrity = attempt_summary.get("integrity")
        if isinstance(integrity, dict):
            artifacts["integrity"] = {
                "records": len(integrity),
                "nonempty_records": sum(value is True for value in integrity.values()),
            }
    if not manifest:
        warnings.append("no readable manifest found (manifest.yaml / manifest.json / control_manifest.yaml)")
    if CampaignRelationship.PROJECT_MISMATCH in binding_issues:
        warnings.append(
            f"manifest project {manifest_project!r} differs from configured project {project!r}"
        )

    row = RunIndexRow(
        project=project,
        campaign=campaign_name,
        campaign_source=campaign_source,
        campaign_binding=campaign_binding,
        run_id=run_id,
        role=role,
        role_source=role_source,
        run_dir=str(run_dir),
        scheduler_state=status.get("state"),
        evidence=evidence,
        latest_metrics=latest_metrics,
        eval_metrics=eval_metrics,
        eval_variants=eval_variants,
        evaluation_snapshot=eval_snapshot,
        canonical_eval_variant_id=canonical_eval_variant_id,
        decision=decision,
        decision_history=_decision_history(run_dir),
        research_contract=contract,
        research_contract_source=("manifest" if contract is not None else None),
        checkpoint=checkpoint,
        artifacts=artifacts,
        provenance=provenance,
        attempts=_scan_attempts(run_dir),
        warnings=warnings,
        evidence_conflicts=conflicts,
        scanned_at=now if now is not None else time.time(),
    )
    apply_freshness(row, now=row.scanned_at)
    return row


def discover_run_dirs(root: Path) -> list[Path]:
    """Find run directories below a root, including instance-scoped layouts.

    Controllers may insert durable namespaces such as
    ``state/<instance>/<campaign>/<run>`` below the Project run root.  Stop
    descending as soon as a Run is found so mirrored ``collected_run`` trees
    cannot become duplicate Runs.
    """
    if not root.is_dir():
        return []
    found: list[Path] = []
    for directory, children, _ in os.walk(root):
        path = Path(directory)
        children.sort()
        if is_run_dir(path):
            found.append(path)
            children[:] = []
    return sorted(found)
