"""Fail-closed classification of exact Attempt metric-conflict evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any


_IDENTITY_FIELDS = (
    "project", "run_id", "attempt_id", "epoch", "step",
    "variant_id", "family_id", "metric",
)
_SEMANTIC_SLOT_FIELDS = (
    "project", "run_id", "attempt_id", "epoch", "step", "family_id", "metric",
)
_REQUIRED_FIELDS = (
    "project", "run_id", "attempt_id", "epoch", "step", "variant_id",
    "family_id", "metric", "source", "sampling_dimensions",
)
_FAMILY_DIMENSION_FIELDS = (
    "sampling_method", "num_sampling_steps", "cfg", "self_cond_cfg_scale",
    "time_schedule", "time_warp_gamma",
)
_FAMILY_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


def _value_token(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _family_id(dimensions: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(dimensions), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _valid_dimensions(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == set(_FAMILY_DIMENSION_FIELDS)
        and all(
            not isinstance(item, bool)
            and isinstance(item, (str, int, float))
            and not (isinstance(item, float) and not math.isfinite(item))
            for item in value.values()
        )
    )


def _exact_binding(value: Mapping[str, Any]) -> bool:
    if any(value.get(key) is None for key in _REQUIRED_FIELDS):
        return False
    if any(
        not isinstance(value.get(key), str)
        or not value.get(key).strip()
        or value.get(key) != value.get(key).strip()
        for key in (
            "project", "run_id", "attempt_id", "variant_id", "family_id",
            "metric", "source",
        )
    ):
        return False
    if not all(
        isinstance(value.get(key), (int, float))
        and not isinstance(value.get(key), bool)
        and math.isfinite(float(value[key]))
        and float(value[key]) >= 0
        for key in ("epoch", "step")
    ):
        return False
    dimensions = value.get("sampling_dimensions")
    family_id = value.get("family_id")
    return (
        isinstance(family_id, str)
        and _FAMILY_ID_PATTERN.fullmatch(family_id) is not None
        and _valid_dimensions(dimensions)
        and _family_id(dimensions) == family_id
    )


def classify_evidence_conflicts(
    value: Any,
    *,
    project: str,
    run_id: str,
    attempt_id: str | None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Return blocking conflicts and safely reclassified cross-binding entries.

    Legacy strings and incomplete mappings remain blocking.  A mapping is
    reclassified only when every source carries an exact identity; path labels
    are never parsed to invent a variant or family binding.
    """
    raw_conflicts = value if isinstance(value, list) else []
    blocking: list[Any] = []
    reclassified: list[dict[str, Any]] = []
    for raw in raw_conflicts:
        if not isinstance(raw, Mapping):
            blocking.append(raw)
            continue
        sources = raw.get("sources")
        if not isinstance(sources, list) or len(sources) < 2 or any(
            not isinstance(source, Mapping) or "value" not in source
            for source in sources
        ):
            blocking.append(dict(raw))
            continue
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        malformed = False
        for source in sources:
            source = dict(source)
            authored = source.get("binding")
            authored = (
                dict(authored) if isinstance(authored, Mapping) else dict(source)
            )
            binding = {
                key: authored.get(key, raw.get(key)) for key in _IDENTITY_FIELDS
            }
            binding["source"] = source.get("source", source.get("path"))
            binding["sampling_dimensions"] = authored.get(
                "sampling_dimensions", raw.get("sampling_dimensions"),
            )
            if not _exact_binding(binding) or (
                binding["project"] != project
                or binding["run_id"] != run_id
                or attempt_id is not None and binding["attempt_id"] != attempt_id
            ):
                malformed = True
                break
            normalized_source = {
                "source": source.get("source", source.get("path")),
                "value": source["value"],
                "observed_at": source.get("observed_at"),
                "binding": {**authored, **binding},
            }
            groups.setdefault(
                tuple(binding[key] for key in _IDENTITY_FIELDS), [],
            ).append(normalized_source)
        if malformed:
            blocking.append(dict(raw))
            continue

        slot_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for group in groups.values():
            for source in group:
                binding = source["binding"]
                slot_groups.setdefault(
                    tuple(binding[key] for key in _SEMANTIC_SLOT_FIELDS), [],
                ).append(source)

        blocking_groups = 0
        for slot_identity, group in slot_groups.items():
            variants = {item["binding"]["variant_id"] for item in group}
            if len(variants) > 1:
                blocking_groups += 1
                blocking.append({
                    "type": "metric_semantic_slot_conflict",
                    **dict(zip(_SEMANTIC_SLOT_FIELDS, slot_identity)),
                    "variant_ids": sorted(variants),
                    "sources": group,
                })
                continue
            values = {_value_token(item["value"]) for item in group}
            if len(values) > 1:
                blocking_groups += 1
                binding = group[0]["binding"]
                blocking.append({
                    "type": "metric_value_conflict",
                    **{key: binding[key] for key in _IDENTITY_FIELDS},
                    "sources": group,
                })
        if blocking_groups:
            continue

        cross_family_identity = {
            tuple(
                source["binding"][key] for key in (
                    "project", "run_id", "attempt_id", "epoch", "step", "metric",
                )
            )
            for group in slot_groups.values() for source in group
        }
        family_ids = {identity[5] for identity in slot_groups}
        if (
            len(slot_groups) > 1
            and len(cross_family_identity) == 1
            and len(family_ids) == len(slot_groups)
        ):
            reclassified.append({
                "type": "cross_binding_values",
                "source_conflict_type": raw.get("type"),
                "bindings": [
                    dict(zip(_IDENTITY_FIELDS, identity))
                    for identity in groups
                ],
                "source_count": len(sources),
            })
        elif len(slot_groups) > 1:
            # Multiple non-conflicting exact slots are only safely separable
            # here when the family binding is the sole semantic difference.
            blocking.append(dict(raw))
    return blocking, reclassified
