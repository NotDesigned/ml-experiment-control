"""Reviewed administrative deletion of one exact managed Run."""
from __future__ import annotations

import json
from pathlib import Path
import re
from urllib.parse import urlencode
from uuid import uuid4

from .api import ClientError, save, segment
from .saved_plan import SavedPlan, bound_response


_CONTRACT = "run-deletion.v1"
_ACTION = re.compile(r"action-[a-f0-9]{16}\Z")
_DIGEST = re.compile(r"sha256:[a-f0-9]{64}\Z")
_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}\Z")
_STATUSES = {"BLOCKED", "PREPARED", "AUTHORIZED", "EXECUTING", "RECONCILE_REQUIRED", "VERIFIED", "FAILED"}


def _validate_action(saved, action, *, reviewed=True):
    if (not isinstance(action, dict) or not isinstance(action.get("execution"), dict)
            or action["execution"].get("status") not in _STATUSES):
        raise ClientError("server returned an invalid deletion Action snapshot")
    if (not _ACTION.fullmatch(str(action.get("action_id", "")))
            or action.get("operation") != "DELETE_RUN_RESOURCES"
            or action.get("scope") != {"project": saved["project"], "scope_type": "run", "object_id": saved["run_id"]}
            or action.get("run_id") != saved["run_id"]
            or action.get("intent_id") != saved["request"]["idempotency_key"]
            or action.get("deletion_request") != saved["request"]
            or action.get("reason") != saved["request"]["reason"]):
        raise ClientError("Action identity or request differs from the saved Run deletion")
    if (not isinstance(action.get("target_manifest"), (dict, list))
            or not isinstance(action.get("gates"), list)
            or not isinstance(action.get("ready"), bool)):
        raise ClientError("deletion Action lacks its reviewable targets and gates")
    for key in ("intent_digest", "gate_bundle_digest", "target_digest"):
        if not _DIGEST.fullmatch(str(action.get(key, ""))):
            raise ClientError("deletion Action lacks a sealed review digest")
    if reviewed and any(action.get(key) != saved["plan"].get(key) for key in
                        ("action_id", "intent_digest", "gate_bundle_digest", "target_digest", "target_manifest")):
        raise ClientError("deletion targets changed after review; inspect and prepare a new plan")
    if action["execution"]["status"] == "VERIFIED":
        bound_response(action["execution"].get("result"), {
            "status": "DELETED", "project": saved["project"], "run_id": saved["run_id"],
            "target_digest": action["target_digest"],
        }, message="deletion completion lacks the exact Run identity and reviewed target digest")
    return action


def _result(saved, path):
    action = saved["action"]
    status = action["execution"]["status"]
    deleted = status == "VERIFIED"
    if deleted:
        next_action = "DELETION_COMPLETE"
    elif status == "RECONCILE_REQUIRED":
        next_action = "RECONCILE"
    elif saved["execution_requested"] or status == "EXECUTING" or (
            saved["authorization_requested"] and status == "PREPARED"):
        next_action = "OBSERVE_ACTION"
    elif status in {"PREPARED", "AUTHORIZED"} and action["ready"]:
        next_action = "EXECUTE_SAVED_PLAN"
    else:
        next_action = "INSPECT_ACTION"
    value = {"state_file": str(path), "operation": "delete", "project": saved["project"],
             "run_id": saved["run_id"], "request": saved["request"], "action": action,
             "deleted": deleted, "next_action": next_action}
    if next_action == "EXECUTE_SAVED_PLAN":
        value["execute_argv"] = ["ml-exp", "--url", saved["server"], "delete", "--state", str(path), "--execute"]
    return value


def run_deletion(client, path: Path, *, project=None, run=None, reason=None,
                 execute=False, reconcile=False, action_id=None):
    """Save intent before preparing; execute once after a separate review."""
    if execute and reconcile:
        raise ClientError("choose either execute or reconcile for Run deletion")
    if reason is not None and (not isinstance(reason, str) or not reason.strip() or len(reason) > 4000):
        raise ClientError("deletion requires a nonempty reason of at most 4000 characters")
    if not path.exists():
        if execute or reconcile or action_id:
            raise ClientError("prepare and review a saved deletion plan before executing or reconciling it")
        if project is None or run is None or reason is None:
            raise ClientError("deletion preparation requires --project, --run and --reason")
        segment(project)
        segment(run)
        request = {"reason": reason, "idempotency_key": "deletion-" + uuid4().hex}
        saved = {"contract": _CONTRACT, "server": client.base, "operation": "delete",
                 "project": project, "run_id": run, "request": request,
                 "plan": None, "action": None, "authorization_requested": False,
                 "execution_requested": False, "recovery_url": "/api/actions?" + urlencode({
                     "project": project, "scope_type": "run", "object_id": run})}
        # An unknown preparation is never replayed, including after a save failure.
        save(path, saved)
        plan = SavedPlan(client, saved, path, persist=save)
        action = plan.request("plan", f"/api/runs/{segment(project)}/{segment(run)}/deletion",
            data=request, validate=lambda value: _validate_action(saved, value, reviewed=False))
        plan.store("action", action)
        return _result(saved, path)

    saved = json.loads(path.read_text())
    if (not isinstance(saved, dict) or saved.get("contract") != _CONTRACT
            or saved.get("server") != client.base or saved.get("operation") != "delete"):
        raise ClientError("state belongs to a different server or Run deletion")
    request = saved.get("request")
    if (not isinstance(request, dict) or set(request) != {"reason", "idempotency_key"}
            or not isinstance(request.get("reason"), str) or not request["reason"].strip()
            or len(request["reason"]) > 4000
            or not isinstance(request.get("idempotency_key"), str)
            or not _KEY.fullmatch(request["idempotency_key"])
            or not isinstance(saved.get("authorization_requested"), bool)
            or not isinstance(saved.get("execution_requested"), bool)
            or not isinstance(saved.get("project"), str) or not isinstance(saved.get("run_id"), str)
            or "plan" not in saved or "action" not in saved or not isinstance(saved.get("recovery_url"), str)):
        raise ClientError("invalid saved Run deletion")
    segment(saved["project"])
    segment(saved["run_id"])
    for actual, expected in ((project, saved["project"]), (run, saved["run_id"]), (reason, request["reason"])):
        if actual is not None and actual != expected:
            raise ClientError("arguments differ from the saved deletion plan")
    if saved["plan"] is None:
        if not action_id or execute or reconcile:
            raise ClientError("deletion preparation response is unknown; inspect " + saved["recovery_url"]
                + " for intent_id=" + request["idempotency_key"]
                + " then attach its exact --action-id to review; no request was replayed")
        if not _ACTION.fullmatch(action_id):
            raise ClientError("invalid deletion Action identity")
        action = _validate_action(saved, client.call("/api/actions/" + action_id), reviewed=False)
        if action["action_id"] != action_id:
            raise ClientError("server returned a different deletion Action identity")
        saved.update(plan=action, action=action)
        save(path, saved)
        return _result(saved, path)
    _validate_action(saved, saved["plan"])
    aid = saved["plan"]["action_id"]
    if action_id is not None and action_id != aid:
        raise ClientError("Action identity differs from the saved deletion plan")
    plan = SavedPlan(client, saved, path, persist=save)
    validate = lambda value: _validate_action(saved, value)
    action = plan.observe("action", "/api/actions/" + aid, validate=validate)
    status = action["execution"]["status"]
    if reconcile and status in {"EXECUTING", "RECONCILE_REQUIRED"}:
        action = plan.request("action", "/api/actions/reconcile", data={"action_id": aid}, validate=validate)
    elif execute and not saved["execution_requested"] and action["ready"]:
        if status == "PREPARED" and not saved["authorization_requested"]:
            action = plan.request("action", "/api/actions/authorize", marker="authorization_requested",
                data={"action_id": aid, "note": "explicit CLI execution of the saved Run deletion plan"}, validate=validate)
        if action["execution"]["status"] == "AUTHORIZED":
            action = plan.request("action", "/api/actions/execute", marker="execution_requested",
                data={"action_id": aid, "confirmation": "EXECUTE " + aid}, validate=validate)
    plan.store("action", action)
    return _result(saved, path)
