"""Exact-Attempt reads and saved, explicitly executed training Actions."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
from urllib.parse import urlencode

from .api import ClientError, attempt_id, segment
from .state import SavedPlan, save


_ACTION = re.compile(r"action-[a-f0-9]{16}\Z")
_DIGEST = re.compile(r"sha256:[a-f0-9]{64}\Z")
_OPERATIONS = {"cancel": "CANCEL_RUN", "retry": "RETRY_ATTEMPT"}
_CONTRACT = "training-operation.v1"





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
    target = attempt_id(action.get("attempt_id"))
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
        attempt_id(new_attempt)
    if not path.exists():
        if execute or reconcile or action_id:
            raise ClientError("prepare and review a saved plan before executing or reconciling it")
        if project is None or run is None or attempt is None:
            raise ClientError("preparation requires --project, --run and an exact --attempt")
        segment(project)
        segment(run)
        attempt_id(attempt)
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
        plan = SavedPlan(client, saved, path, persist=save)
        action = plan.request("plan", endpoint, data=request, validate=lambda response:
            _validate_action(saved, response.get("action") if isinstance(response, dict) else None, reviewed=False))
        plan.store("action", action)
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
    attempt_id(saved["source_attempt_id"])
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
    plan = SavedPlan(client, saved, path, persist=save)
    validate = lambda value: _validate_action(saved, value)
    action = plan.observe("action", "/api/actions/" + aid, validate=validate)
    status = action["execution"].get("status")
    if reconcile and status in {"EXECUTING", "RECONCILE_REQUIRED"}:
        action = plan.request("action", "/api/actions/reconcile", data={"action_id": aid}, validate=validate)
    elif execute and not saved["execution_requested"] and action.get("ready"):
        if status == "PREPARED" and not saved["authorization_requested"]:
            action = plan.request("action", "/api/actions/authorize", marker="authorization_requested",
                data={"action_id": aid, "note": "explicit CLI execution of the saved training plan"}, validate=validate)
        if action["execution"].get("status") == "AUTHORIZED":
            # Durable dispatch marker must precede HTTP; even a lost reply cannot cause a replay.
            action = plan.request("action", "/api/actions/execute", marker="execution_requested",
                data={"action_id": aid, "confirmation": "EXECUTE " + aid}, validate=validate)
    plan.store("action", action)
    return _result(saved, path)
