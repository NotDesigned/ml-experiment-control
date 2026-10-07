"""Derived charts; immutable observations remain the scientific evidence."""
from __future__ import annotations

import hashlib
import json
import math
import re

VERSION = 2
FIELDS = ("name", "unit", "protocol_id", "dataset_id")


def identity(value, *, variant=False):
    fields = (*FIELDS, "variant_id") if variant else FIELDS
    return json.dumps({k: value.get(k) for k in fields}, sort_keys=True, separators=(",", ":"))


def fingerprint(value):
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def stem(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:64]


def number(value):
    return type(value) in {float, int} and math.isfinite(value)


class Projection:
    """Incremental, disposable view over the durable accepted event stream."""
    def __init__(self):
        self.cursor = 0
        self.series = {}

    def add(self, item, sequence):
        if not item.get("name") or not item.get("unit"):
            return
        key = identity(item, variant=True)
        if key not in self.series:
            if len(self.series) >= 512:
                raise ValueError("DISPLAY_CONTEXT_LIMIT")
            self.series[key] = {"context": {k: item.get(k) for k in (*FIELDS, "variant_id")},
                                "latest": None, "positions": {}}
        series = self.series[key]
        axis = "step" if number(item.get("step")) else "epoch" if number(item.get("epoch")) else None
        point = {**item, "axis": axis, "sequence": sequence}
        if axis and number(item.get("value")) and item["status"] in {"VALID", "PARTIAL"}:
            positions = series["positions"].setdefault(axis, [])
            if item[axis] not in positions and len(positions) < 2:
                positions.append(item[axis])
        old = series["latest"]
        if old and axis == old["axis"] and axis:
            if item[axis] < old[axis]:
                return
            if item[axis] == old[axis]:
                fields = ("value", "status", "checkpoint_id", "numerator", "denominator")
                if any(old.get(k) != item.get(k) for k in fields):
                    point.update(status="CONFLICTING", value=None, error="CONFLICTING_LATEST_OBSERVATIONS")
        elif old and axis is None and old["axis"] is None:
            if old.get("checkpoint_id") != item.get("checkpoint_id"):
                point.update(status="CONFLICTING", value=None, error="UNORDERED_CHECKPOINTS")
            elif any(old.get(k) != item.get(k) for k in ("value", "status", "numerator", "denominator")):
                point.update(status="CONFLICTING", value=None, error="CONFLICTING_LATEST_OBSERVATIONS")
        series["latest"] = point

    def document(self, names):
        variants = {}
        for series in self.series.values():
            base = identity(series["context"])
            variants.setdefault(base, set()).add(series["context"]["variant_id"])
        result = {}
        for key, series in self.series.items():
            base = identity(series["context"])
            name = names[base]
            suffix = "/variants/" + fingerprint(key)
            result[key] = {**series, "name": name, "multiple_variants": len(variants[base]) > 1,
                           "curve": "curves/" + name, "shadow": "curves/" + name + suffix,
                           "result": "results/" + name + (suffix if len(variants[base]) > 1 else "")}
        return {"contract": "metric-display.v2", "through": self.cursor, "series": result}


def curve_record(event, display):
    record = {}
    if event["payload"]["kind"] != "metrics":
        return record
    for item in event["payload"]["observations"]:
        series = display["series"].get(identity(item, variant=True))
        if series is None or not number(item.get("value")) or item["status"] not in {"VALID", "PARTIAL"}:
            continue
        axis = "step" if number(item.get("step")) else "epoch" if number(item.get("epoch")) else None
        if axis is None:
            continue
        for key in [series["shadow"], *([] if series["multiple_variants"] else [series["curve"]])]:
            key += "/by_epoch" if axis == "epoch" else ""
            record[key] = item["value"]
            record["axes/" + key.removeprefix("curves/") + "/" + axis] = item[axis]
    return record


def configure(handle, display):
    handle.define_metric("metrics/*", hidden=True, summary="none", overwrite=True)
    handle.define_metric("axes/*", hidden=True, summary="none", overwrite=True)
    handle.define_metric("ml_expd/*", hidden=True)
    for series in display["series"].values():
        for axis in ("step", "epoch"):
            for key, visible in [(series["curve"], not series["multiple_variants"]),
                                 (series["shadow"], series["multiple_variants"])]:
                key += "/by_epoch" if axis == "epoch" else ""
                axis_key = "axes/" + key.removeprefix("curves/") + "/" + axis
                handle.define_metric(axis_key, hidden=True, summary="none", overwrite=True)
                handle.define_metric(key, step_metric=axis_key, step_sync=False,
                                     hidden=not (visible and len(series["positions"].get(axis, [])) == 2),
                                     summary="none", overwrite=True)
        handle.define_metric(series["result"], hidden=False, overwrite=True)


def summaries(handle, remote_summary, display, *, terminal, coverage):
    values = {}
    for series in display["series"].values():
        latest = series["latest"]
        if terminal and latest["status"] == "VALID" and number(latest.get("value")):
            values[series["result"]] = latest["value"]
    # Helpers stay in immutable history, not in numeric result summaries.
    for key in remote_summary:
        if key.startswith(("metrics/", "axes/", "curves/", "results/")) and key not in values:
            try:
                del handle.summary[key]
            except KeyError:
                pass
    for key, value in values.items():
        handle.summary[key] = value
    definitions = {s["result"]: {**s["context"], "checkpoint_id": s["latest"].get("checkpoint_id"),
                   "step": s["latest"].get("step"), "epoch": s["latest"].get("epoch"),
                   "numerator": s["latest"].get("numerator"), "denominator": s["latest"].get("denominator"),
                   "status": s["latest"]["status"], "error": s["latest"].get("error")}
                   for s in display["series"].values()}
    # W&B flattens dict-valued summaries; JSON preserves literal metric keys.
    handle.summary["ml_expd/display_definitions_json"] = json.dumps(definitions, sort_keys=True, allow_nan=False)
    handle.summary["ml_expd/display_coverage"] = coverage
    handle.summary["ml_expd/display_through"] = display["through"]
    handle.summary["ml_expd/display_version"] = VERSION
