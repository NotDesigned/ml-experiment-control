"""Project-authored units and metric records; no scientific metric vocabulary."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MetricDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unit: str = Field(min_length=1, max_length=128)
    required: bool = False
    description: str = Field(default="", max_length=2000)
    aggregation: str = Field(default="scalar", min_length=1, max_length=128)
    numerator_unit: str | None = Field(default=None, min_length=1, max_length=128)
    denominator_unit: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("unit", "numerator_unit", "denominator_unit")
    @classmethod
    def units(cls, value):
        if value is not None and (not value.strip() or any(ord(c) < 32 for c in value)):
            raise ValueError("units must be nonempty printable strings")
        return value


class MetricSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    definitions: dict[str, MetricDefinition] = Field(default_factory=dict, max_length=256)

    @field_validator("definitions")
    @classmethod
    def names(cls, value):
        if any(not name.strip() or len(name) > 128 or any(ord(c) < 32 for c in name) for name in value):
            raise ValueError("metric names must be nonempty printable strings of at most 128 characters")
        return value


def finite(value):
    try:
        return type(value) in {int, float} and math.isfinite(value)
    except OverflowError:
        return False


def protocol_identity(context):
    return "sha256:" + hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":"),
                                               allow_nan=False).encode()).hexdigest()


def observations(record, schema, *, protocol_id=None, source=None):
    """Normalize typed or legacy wide JSONL without inventing units or zeros."""
    definitions = schema.get("definitions", {})
    common = {k: record[k] for k in ("step", "epoch") if finite(record.get(k))}
    common.update({k: record[k] for k in ("timestamp", "checkpoint_id", "dataset_id")
                   if k in record and (isinstance(record[k], str) and len(record[k]) <= 2000 or finite(record[k]))})
    context = record.get("protocol_id", protocol_id)
    bad_context = context is not None and (not isinstance(context, str) or len(context) > 2000)
    if bad_context:
        context = None
    variant = record.get("variant_id")
    bad_variant = variant is not None and (not isinstance(variant, str) or not variant.strip()
                                          or len(variant) > 2000 or any(ord(c) < 32 for c in variant))
    if "name" in record:
        if not isinstance(record["name"], str) or not record["name"].strip() or len(record["name"]) > 128 or any(ord(c) < 32 for c in record["name"]):
            return [{"name": None, "value": None, "unit": None, "status": "FAILED",
                     "error": "INVALID_METRIC_NAME", "errors": ["INVALID_METRIC_NAME"], "source": source}]
        values = [(record["name"], record.get("value"), record)]
    else:
        values = [(name, value, {}) for name, value in record.items()
                  if name not in {"step", "epoch", "timestamp", "checkpoint_id", "dataset_id", "protocol_id", "variant_id"}
                  and isinstance(value, (int, float)) and not isinstance(value, bool)]
    results = []
    for name, value, item in values:
        definition = definitions.get(name, {})
        unit = item.get("unit", definition.get("unit"))
        state = item.get("status", "VALID")
        error = item.get("error")
        errors = []
        if not isinstance(state, str) or state not in {"VALID", "PARTIAL", "FAILED", "MISSING"}:
            state, error = "FAILED", "INVALID_STATUS"
            errors.append(error)
        valid_number = finite(value)
        if state == "VALID" and not valid_number:
            state, error = "FAILED", "NONFINITE_OR_MISSING_VALUE"
            errors.append(error)
        if not isinstance(unit, str) or not unit.strip() or len(unit) > 128 or any(ord(c) < 32 for c in unit):
            state, error = "FAILED" if state == "FAILED" else "UNDECLARED", "UNIT_NOT_DECLARED"
            errors.append(error)
            unit = None
        elif definition.get("unit") is not None and unit != definition["unit"]:
            state, error = "FAILED", "UNIT_MISMATCH"
            errors.append(error)
        elif definitions and name not in definitions:
            state, error = "FAILED" if state == "FAILED" else "UNDECLARED", "METRIC_NOT_DECLARED"
            errors.append(error)
        if bad_context:
            state, error = "FAILED", "INVALID_PROTOCOL_ID"
            errors.append(error)
        if bad_variant:
            state, error = "FAILED", "INVALID_VARIANT_ID"
            errors.append(error)
        if protocol_id is not None and context != protocol_id:
            state, error = "FAILED", "PROTOCOL_MISMATCH"
            errors.append(error)
        sums = {}
        for key in ("numerator", "denominator"):
            raw = item.get(key)
            if raw is not None:
                sums[key] = raw if finite(raw) else None
                if not finite(raw):
                    state, error = "FAILED", "NONFINITE_AGGREGATION_COUNT"
                    errors.append(error)
        if definition.get("aggregation") == "ratio_of_sums" and (
                sums.get("numerator") is None or sums.get("denominator") is None or sums["denominator"] <= 0):
            state, error = "FAILED", "INVALID_AGGREGATION_COUNTS"
            errors.append(error)
        results.append({**common, "name": name, "value": value if valid_number else None, "unit": unit,
                        "invalid_value": str(value)[:128] if value is not None and not valid_number else None,
                        "variant_id": variant if not bad_variant else None,
                        "protocol_id": context, "status": state, "error": str(error)[:2000] if error is not None else None,
                        "source": source or record.get("_source_path"), "errors": errors, **sums})
    return results


def metric_result(records, schema=None, *, protocol_id=None, source=None):
    """Keep distinct contexts and mark conflicting rewrites instead of overriding."""
    schema = schema or {"definitions": {}}
    by_key = {}
    for record in records:
        for item in observations(record, schema, protocol_id=protocol_id, source=source):
            key = tuple(item.get(k) for k in ("name", "unit", "protocol_id", "checkpoint_id", "dataset_id", "variant_id", "epoch", "step"))
            previous = by_key.get(key)
            item["sources"] = sorted(set([*(previous.get("sources", []) if previous else []),
                                          *([item["source"]] if item["source"] else [])]))
            if previous is not None and any(previous.get(k) != item.get(k) for k in ("value", "status", "numerator", "denominator")):
                item.update(status="CONFLICTING", error="CONFLICTING_OBSERVATIONS", value=None)
            by_key[key] = item
    items = list(by_key.values())
    units_by_name = {}
    for item in items:
        identity = tuple(item.get(k) for k in ("name", "protocol_id", "checkpoint_id", "dataset_id", "variant_id", "epoch", "step"))
        units_by_name.setdefault(identity, set()).add(item["unit"])
    for item in items:
        identity = tuple(item.get(k) for k in ("name", "protocol_id", "checkpoint_id", "dataset_id", "variant_id", "epoch", "step"))
        if len(units_by_name[identity]) > 1:
            item.update(status="CONFLICTING", error="CONFLICTING_UNITS", value=None)
    required = [name for name, definition in schema.get("definitions", {}).items() if definition.get("required")]
    contexts = {}
    identity_fields = ("protocol_id", "checkpoint_id", "dataset_id", "variant_id", "epoch", "step")
    for item in items:
        identity = tuple(item.get(k) for k in identity_fields)
        contexts.setdefault(identity, []).append(item)
    groups = []
    for identity, members in contexts.items():
        published = {item["name"] for item in members if item["status"] == "VALID"}
        missing = [name for name in required if name not in published]
        failed = any(item["status"] in {"FAILED", "CONFLICTING"} for item in members)
        state = "FAILED" if failed else "PARTIAL" if missing or any(item["status"] != "VALID" for item in members) else "COMPLETE"
        groups.append({"identity": dict(zip(identity_fields, identity)), "state": state,
                       "missing_metrics": missing, "records": members})
    # A latest step is meaningful within one protocol/checkpoint/dataset, never
    # an automatic best-model selection across distinct evaluation contexts.
    bases = {tuple(group["identity"][k] for k in identity_fields[:4]) for group in groups}
    current = None
    if len(bases) == 1:
        if all(finite(group["identity"]["step"]) for group in groups):
            current = max(groups, key=lambda group: (group["identity"]["epoch"] or 0, group["identity"]["step"]))
        elif len(groups) == 1:
            current = groups[0]
    return {"state": current["state"] if current else "PARTIAL" if items else "NOT_OBSERVED",
            "required_metrics": required, "missing_metrics": current["missing_metrics"] if current else required,
            "records": items, "contexts": groups, "current": current, "protocol_id": protocol_id}
