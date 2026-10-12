"""Exact-Attempt reads and saved, explicitly executed training Actions."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
from urllib.parse import urlencode

from .api import ClientError, save, segment


_ATTEMPT = re.compile(r"attempt-[0-9]{3,}\Z")
_ACTION = re.compile(r"action-[a-f0-9]{16}\Z")
_DIGEST = re.compile(r"sha256:[a-f0-9]{64}\Z")
_OPERATIONS = {"cancel": "CANCEL_RUN", "retry": "RETRY_ATTEMPT"}
_CONTRACT = "training-operation.v1"


def _attempt(value):
    if not isinstance(value, str) or not _ATTEMPT.fullmatch(value):
        raise ClientError("choose an exact attempt-NNN identity")
    return value


def read_attempt(client, view, project, run, attempt=None, *, stream="both", lines=80,
                 keys=None, max_points=2000):
    """Read one exact Attempt; return metric dimensions without transformation."""
    project_path, run_path = segment(project), segment(run)
    if view == "logs":
        if stream not in {"both", "stdout", "stderr"} or not 1 <= lines <= 10000:
            raise ClientError("logs require stdout/stderr/both and 1..10000 lines")
        query = {"stream": stream, "lines": lines}
    elif view == "metrics":
        if max_points <= 0:
            raise ClientError("max-points must be positive")
        query = {"max_points": max_points}
        if keys is not None:
            query["keys"] = keys
    else:
        raise ClientError("unknown Attempt read operation")
    if attempt is None:
        listing = client.call(f"/api/runs/{project_path}/{run_path}/attempts")
        if not isinstance(listing, dict) or listing.get("project") != project or listing.get("run_id") != run:
            raise ClientError("Attempt listing differs from the requested Run")
        attempt = _attempt(listing.get("current_attempt_id"))
        entries = listing.get("attempts")
        if not isinstance(entries, list) or not any(isinstance(item, dict) and item.get("attempt_id") == attempt
                   and item.get("identity") == run + "::" + attempt
                   for item in entries):
            raise ClientError("current Attempt is absent from the Run listing")
    identity = segment(run + "::" + _attempt(attempt))
    return client.call(f"/api/attempts/{project_path}/{identity}/{view}?" + urlencode(query))


def _validate_action(saved, action, *, reviewed=True):
    if not isinstance(action, dict) or not isinstance(action.get("execution"), dict):
        raise ClientError("server returned an invalid Action snapshot")
    if action["execution"].get("status") not in {"BLOCKED", "PREPARED", "AUTHORIZED", "EXECUTING", "RECONCILE_REQUIRED", "VERIFIED", "FAILED"}:
        raise ClientError("server returned an invalid Action execution status")
    if (not _ACTION.fullmatch(str(action.get("action_id", "")))
            or action.get("operation") != _OPERATIONS[saved["operation"]]
            or action.get("scope") != {"project": saved["project"], "scope_type": "attempt",
                                       "object_id": saved["run_id"] + "::" + saved["source_attempt_id"]}
            or action.get("run_id") != saved["run_id"]):
        raise ClientError("Action identity differs from the saved training operation")
    target = _attempt(action.get("attempt_id"))
    if saved["operation"] == "cancel":
        expected = saved["source_attempt_id"]
    else:
        expected = saved["request"].get("new_attempt_id", target)
        if target == saved["source_attempt_id"]:
            raise ClientError("retry must create a new Attempt in the same frozen Run")
    if target != expected:
        raise ClientError("Action targets a different Attempt")
    if saved["operation"] == "retry" and action.get("ready"):
        summary = action.get("preflight_summary")
        summary = summary if isinstance(summary, dict) else {}
        requested = summary.get("requested_gpu_hours")
        cap = saved["request"]["max_gpu_hours"]
        if (not isinstance(requested, (int, float)) or isinstance(requested, bool)
                or not math.isfinite(requested) or requested < 0 or requested > cap):
            raise ClientError("retry Action resources exceed or lack the saved GPU-hour budget")
        if summary.get("resource_approval") == "budget_cap":
            approved = summary.get("max_gpu_hours")
            if (not isinstance(approved, (int, float)) or isinstance(approved, bool)
                    or not math.isfinite(approved) or not 0 < approved <= cap):
                raise ClientError("retry Action raises or lacks the saved GPU-hour budget cap")
    for key in ("intent_digest", "gate_bundle_digest"):
        if not _DIGEST.fullmatch(str(action.get(key, ""))):
            raise ClientError("Action lacks a sealed review digest")
    if reviewed and any(action.get(key) != saved["plan"].get(key)
                        for key in ("action_id", "attempt_id", "intent_digest", "gate_bundle_digest")):
        raise ClientError("Action changed after review; inspect and prepare a new plan")
    return action


def _result(saved, path):
    action = saved["action"]
    status = action["execution"].get("status")
    if status == "VERIFIED":
        next_action = "OBSERVE_RESULT"
    elif saved.get("execution_requested") or status in {"EXECUTING", "RECONCILE_REQUIRED"}:
        next_action = "RECONCILE" if status == "RECONCILE_REQUIRED" else "OBSERVE_ACTION"
    elif saved.get("authorization_requested") and status == "PREPARED":
        next_action = "OBSERVE_ACTION"
    elif status in {"PREPARED", "AUTHORIZED"} and action.get("ready"):
        next_action = "EXECUTE_SAVED_PLAN"
    else:
        next_action = "INSPECT_ACTION"
    result = {"state_file": str(path), "operation": saved["operation"],
              "project": saved["project"], "run_id": saved["run_id"],
              "source_attempt_id": saved["source_attempt_id"], "request": saved["request"],
              "action": action, "next_action": next_action}
    if next_action == "EXECUTE_SAVED_PLAN":
        result["execute_argv"] = ["ml-exp", "--url", saved["server"], saved["operation"],
                                  "--state", str(path), "--execute"]
    if saved["operation"] == "retry":
        result["retry_semantics"] = ("New Attempt in the same frozen Run; no checkpoint is selected or changed. "
                                     "Use create --resume-from to freeze a new Run with an explicit checkpoint reference.")
    return result


def training_operation(client, operation, path: Path, *, project=None, run=None,
                       attempt=None, max_gpu_hours=None, new_attempt=None, reason=None,
                       execute=False, reconcile=False, action_id=None):
    """Prepare once; a later explicit invocation can execute that reviewed plan."""
    if operation not in _OPERATIONS or (execute and reconcile):
        raise ClientError("choose one training operation and either execute or reconcile")
    if max_gpu_hours is not None and (not math.isfinite(max_gpu_hours) or max_gpu_hours <= 0):
        raise ClientError("retry requires a finite positive max-gpu-hours budget")
    if reason is not None and len(reason) > 4000:
        raise ClientError("reason must contain at most 4000 characters")
    if new_attempt is not None:
        _attempt(new_attempt)
    if not path.exists():
        if execute or reconcile or action_id:
            raise ClientError("prepare and review a saved plan before executing or reconciling it")
        if project is None or run is None or attempt is None:
            raise ClientError("preparation requires --project, --run and an exact --attempt")
        segment(project)
        segment(run)
        _attempt(attempt)
        if operation == "retry" and max_gpu_hours is None:
            raise ClientError("retry preparation requires --max-gpu-hours")
        if operation == "cancel" and (max_gpu_hours is not None or new_attempt is not None):
            raise ClientError("cancel has no retry budget or new Attempt")
        request = {"reason": reason or ""}
        if operation == "retry":
            request["max_gpu_hours"] = max_gpu_hours
            if new_attempt is not None:
                if new_attempt == attempt:
                    raise ClientError("retry must create a new Attempt")
                request["new_attempt_id"] = new_attempt
        saved = {"contract": _CONTRACT, "server": client.base, "operation": operation,
                 "project": project, "run_id": run, "source_attempt_id": attempt,
                 "request": request, "plan": None, "action": None,
                 "authorization_requested": False, "execution_requested": False,
                 "recovery_url": "/api/actions?" + urlencode({"project": project,
                     "scope_type": "attempt", "object_id": run + "::" + attempt})}
        # Keep intent before HTTP: an interrupted preparation is inspected, never replayed.
        save(path, saved)
        endpoint = f"/api/attempts/{segment(project)}/{segment(run + '::' + attempt)}/{operation}"
        response = client.call(endpoint, data=request)
        action = _validate_action(saved, response.get("action") if isinstance(response, dict) else None, reviewed=False)
        saved.update(plan=action, action=action)
        save(path, saved)
        return _result(saved, path)

    saved = json.loads(path.read_text())
    if (not isinstance(saved, dict) or saved.get("contract") != _CONTRACT
            or saved.get("server") != client.base or saved.get("operation") != operation):
        raise ClientError("state belongs to a different server or training operation")
    if (not isinstance(saved.get("request"), dict)
            or not isinstance(saved.get("authorization_requested"), bool)
            or not isinstance(saved.get("execution_requested"), bool)
            or not isinstance(saved.get("project"), str)
            or not isinstance(saved.get("run_id"), str)):
        raise ClientError("invalid saved training operation")
    if operation == "retry":
        cap = saved["request"].get("max_gpu_hours")
        if not isinstance(cap, (int, float)) or isinstance(cap, bool) or not math.isfinite(cap) or cap <= 0:
            raise ClientError("invalid saved retry budget")
    segment(saved["project"])
    segment(saved["run_id"])
    _attempt(saved["source_attempt_id"])
    for actual, expected in ((project, saved["project"]), (run, saved["run_id"]),
                             (attempt, saved["source_attempt_id"]),
                             (max_gpu_hours, saved["request"].get("max_gpu_hours")),
                             (new_attempt, saved["request"].get("new_attempt_id")),
                             (reason, saved["request"].get("reason"))):
        if actual is not None and actual != expected:
            raise ClientError("arguments differ from the saved plan; inspect it before preparing a new operation")
    if saved["plan"] is None:
        if not action_id or execute or reconcile:
            raise ClientError("preparation response is unknown; inspect " + saved["recovery_url"]
                              + " then attach its exact --action-id to review; no request was replayed")
        if not _ACTION.fullmatch(action_id):
            raise ClientError("invalid Action identity")
        action = _validate_action(saved, client.call("/api/actions/" + action_id), reviewed=False)
        if action["action_id"] != action_id:
            raise ClientError("server returned a different Action identity")
        saved.update(plan=action, action=action)
        save(path, saved)
        return _result(saved, path)
    _validate_action(saved, saved["plan"])
    aid = saved["plan"]["action_id"]
    if action_id is not None and action_id != aid:
        raise ClientError("Action identity differs from the saved plan")
    action = _validate_action(saved, client.call("/api/actions/" + aid))
    saved["action"] = action
    save(path, saved)
    status = action["execution"].get("status")
    if reconcile and status in {"EXECUTING", "RECONCILE_REQUIRED"}:
        action = _validate_action(saved, client.call("/api/actions/reconcile", data={"action_id": aid}))
    elif execute and not saved["execution_requested"] and action.get("ready"):
        if status == "PREPARED" and not saved["authorization_requested"]:
            saved["authorization_requested"] = True
            save(path, saved)
            action = _validate_action(saved, client.call("/api/actions/authorize", data={
                "action_id": aid, "note": "explicit CLI execution of the saved training plan"}))
            saved["action"] = action
            save(path, saved)
        if action["execution"].get("status") == "AUTHORIZED":
            # Durable dispatch marker must precede HTTP; even a lost reply cannot cause a replay.
            saved["execution_requested"] = True
            save(path, saved)
            action = _validate_action(saved, client.call("/api/actions/execute", data={
                "action_id": aid, "confirmation": "EXECUTE " + aid}))
    saved["action"] = action
    save(path, saved)
    return _result(saved, path)
