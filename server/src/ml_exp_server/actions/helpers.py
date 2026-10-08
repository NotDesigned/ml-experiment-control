"""Shared ActionService evidence and planning helpers."""
from __future__ import annotations
import difflib
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
import yaml
from ..controller_gateway import redact as _redact
from .errors import ActionError
from .policy import ExecutionDispatch

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


_CAMPAIGN_REVISION = re.compile(r"^campaign\.[0-9a-f]{64}$")


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _manifest_identity_digest(payload: dict[str, Any]) -> str:
    """Digest every execution-relevant manifest field except creation time."""
    identity = {key: value for key, value in payload.items() if key != "created_at"}
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return _sha256(encoded)


def _canonical_manifest_path(
    campaign: dict[str, Any], *, cwd: Path, run_id: str,
) -> Path | None:
    local_root = campaign.get("local_root")
    campaign_name = campaign.get("campaign")
    if not local_root or not campaign_name:
        return None
    root = Path(str(local_root))
    if not root.is_absolute():
        root = cwd / root
    return (root / str(campaign_name) / run_id / "manifest.yaml").resolve()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _parse_mapping(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1])
    try:
        payload = yaml.safe_load(stripped)
    except yaml.YAMLError as exc:
        raise ActionError(f"draft is not valid YAML/JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ActionError("draft must contain a mapping")
    def json_safe(value: Any) -> Any:
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return value
    return json_safe(payload)


def _unified_diff(path: Path, proposed: str) -> str:
    current = path.read_text(encoding="utf-8") if path.is_file() else ""
    return "".join(difflib.unified_diff(
        current.splitlines(keepends=True), proposed.splitlines(keepends=True),
        fromfile=str(path) if current else "/dev/null", tofile=str(path),
    ))


def _semantic_changes(before: Any, after: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(before, dict) and isinstance(after, dict):
        changes: list[dict[str, Any]] = []
        for key in sorted(set(before) | set(after)):
            path = f"{prefix}.{key}" if prefix else str(key)
            changes.extend(_semantic_changes(before.get(key), after.get(key), path))
        return changes
    if before == after:
        return []
    root = prefix.split(".", 1)[0]
    if root in {"links", "research_contract", "runs", "resolved_config"}:
        category = "EXPERIMENT_DESIGN"
    elif root in {"budget", "resources", "backend"}:
        category = "RESOURCE"
    elif root in {"storage", "command", "assets", "checkpoint", "resume_policy"}:
        category = "EXECUTION_IDENTITY"
    else:
        category = "METADATA"
    return [{
        "path": prefix, "category": category,
        "before": _redact(before), "after": _redact(after),
    }]


def _missing_frozen_match_fields(manifest: dict[str, Any]) -> list[str]:
    """Return comparison paths that claim frozen identity but are absent.

    Runtime-observed fields such as ``environment.torch`` are intentionally
    deferred until collection.  Fields rooted in the Run identity, however,
    must exist in the preview manifest before scheduler authorization.
    """
    contract = manifest.get("research_contract")
    contract = contract if isinstance(contract, dict) else {}
    comparison = contract.get("comparison")
    comparison = comparison if isinstance(comparison, dict) else {}
    fields = comparison.get("match_fields")
    if not isinstance(fields, list):
        return []
    frozen_roots = {
        "source_id", "image_id", "config_path", "git_commit",
        "resolved_config", "backend", "resources", "storage", "checkpoint",
    }
    missing: list[str] = []
    for value in fields:
        if not isinstance(value, str) or not value:
            continue
        parts = value.split(".")
        if parts[0] not in frozen_roots:
            continue
        current: Any = manifest
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                missing.append(value)
                break
            current = current[part]
    return missing


def _gate(name: str, passed: bool, detail: str, *, warning: bool = False) -> dict[str, Any]:
    return {
        "name": name,
        "status": "WARNING" if warning else ("PASS" if passed else "FAIL"),
        "detail": detail,
    }


@dataclass(frozen=True)
class PendingActionExecution:
    """A durably claimed Action whose slow executor may run in the background."""

    plan: dict[str, Any]
    execution: dict[str, Any]
    dispatch: ExecutionDispatch
