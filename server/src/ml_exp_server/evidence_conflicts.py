"""Fail-closed classification of exact Attempt metric-conflict evidence."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any


_IDENTITY_FIELDS = (
    "project", "run_id", "attempt_id", "epoch", "step",
    "variant_id", "family_id", "metric",
)
_REQUIRED_FIELDS = (
    "project", "run_id", "attempt_id", "epoch", "step", "variant_id", "metric",
)


def _value_token(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _exact_binding(value: Mapping[str, Any]) -> bool:
    if any(value.get(key) is None for key in _REQUIRED_FIELDS):
        return False
    if any(
        not isinstance(value.get(key), str) or not value.get(key)
        for key in ("project", "run_id", "attempt_id", "variant_id", "metric")
    ):
        return False
    return all(
        isinstance(value.get(key), (int, float))
        and not isinstance(value.get(key), bool)
        and math.isfinite(float(value[key]))
        for key in ("epoch", "step")
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

        exact_conflicts = 0
        for identity, group in groups.items():
            if len(group) < 2 or len({_value_token(item["value"]) for item in group}) < 2:
                continue
            exact_conflicts += 1
            blocking.append({
                "type": "metric_value_conflict",
                **dict(zip(_IDENTITY_FIELDS, identity)),
                "sources": group,
            })
        if exact_conflicts == 0:
            reclassified.append({
                "type": "cross_binding_values",
                "source_conflict_type": raw.get("type"),
                "bindings": [
                    dict(zip(_IDENTITY_FIELDS, identity))
                    for identity in groups
                ],
                "source_count": len(sources),
            })
    return blocking, reclassified
