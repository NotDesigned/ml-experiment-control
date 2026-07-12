"""Durable exact-identity outboxes for scheduler mutations."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .manifest import append_event, atomic_create, atomic_write, utc_now


TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "CANCELLED", "PREEMPTED"})


def cancel_intent_path(run_dir: str | Path, attempt_id: str) -> Path:
    """Return the attempt-local durable cancel intent path."""
    if not attempt_id:
        raise ValueError("cancel requires an explicit attempt identity")
    return Path(run_dir) / "attempts" / attempt_id / "cancel_intent.json"


def execute_cancel_outbox(
    *,
    run_dir: str | Path,
    project: str,
    run_id: str,
    attempt_id: str,
    backend: str | None,
    backend_job_id: str,
    status_call: Callable[[], Mapping[str, Any]],
    cancel_call: Callable[[], Mapping[str, Any]],
    now: Callable[[], str] = utc_now,
) -> dict[str, Any]:
    """Cancel one exact scheduler job without permitting duplicate mutation.

    A create-once REQUESTED intent is written before the scheduler call. If a
    prior call crashed, reconciliation observes the exact job and never issues
    a second cancel mutation while that job remains nonterminal.
    """
    root = Path(run_dir)
    path = cancel_intent_path(root, attempt_id)
    identity = {
        "project": project,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "backend": backend,
        "backend_job_id": backend_job_id,
    }
    if path.is_file():
        intent = json.loads(path.read_text(encoding="utf-8"))
        conflicts = [
            key for key, expected in identity.items()
            if intent.get(key) != expected
        ]
        if conflicts:
            raise ValueError(
                "cancel intent conflicts with selected scheduler identity: "
                + ", ".join(conflicts)
            )
        if intent.get("state") == "VERIFIED":
            return dict(intent["result"])
        status = dict(status_call())
        _require_exact_job(status, backend_job_id, operation="cancel reconciliation")
        if str(status.get("state", "")).upper() in TERMINAL_STATES:
            verified_at = now()
            intent.update({
                "state": "VERIFIED",
                "verified_at": verified_at,
                "result": status,
            })
            atomic_write(path, intent)
            _append_cancel_event(
                root, identity, "cancel_verified", verified_at,
                {"state": status.get("state")},
            )
            return status
        raise RuntimeError(
            "cancel intent is unresolved and target is still nonterminal; "
            "do not issue a second cancel mutation"
        )

    requested_at = now()
    intent = {
        "schema_version": 1,
        "state": "REQUESTED",
        "requested_at": requested_at,
        **identity,
    }
    atomic_create(path, intent)
    _append_cancel_event(root, identity, "cancel_requested", requested_at, {})
    status = dict(cancel_call())
    _require_exact_job(status, backend_job_id, operation="cancel")
    verified_at = now()
    intent.update({
        "state": "VERIFIED",
        "verified_at": verified_at,
        "result": status,
    })
    atomic_write(path, intent)
    _append_cancel_event(
        root, identity, "cancel_verified", verified_at,
        {"state": status.get("state")},
    )
    return status


def _require_exact_job(
    status: Mapping[str, Any], expected_job_id: str, *, operation: str,
) -> None:
    if str(status.get("backend_job_id")) != expected_job_id:
        raise RuntimeError(f"{operation} returned a different backend job identity")


def _append_cancel_event(
    run_dir: Path,
    identity: Mapping[str, Any],
    event: str,
    timestamp: str,
    payload: Mapping[str, Any],
) -> None:
    append_event(run_dir / "events.jsonl", {
        "timestamp": timestamp,
        "project": identity["project"],
        "run_id": identity["run_id"],
        "attempt_id": identity["attempt_id"],
        "backend": identity["backend"],
        "backend_job_id": identity["backend_job_id"],
        "event": event,
        "payload": dict(payload),
    })
