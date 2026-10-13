"""Read-only compact experiment status, preserving separate evidence layers."""
from __future__ import annotations

import json

from .api import ClientError, segment
from .operations import _attempt
from .saved_plan import bound_response


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _bound_layer(run, name, attempt):
    layer = _mapping(_mapping(run.get("evidence")).get(name))
    if attempt is None or layer.get("attempt_id") != attempt:
        return {}
    return layer


def _identity(value, project, run, *, attempt=None):
    expected = {"project": project, "run_id": run}
    if attempt is not None:
        expected["attempt_id"] = attempt
    return bound_response(value, expected, message="status response differs from the requested exact Run/Attempt")


def run_status(client, health, project, run, attempt=None, *, state_path=None, details=False):
    """Observe existing resources only; never prepare, collect or refresh providers."""
    project_path, run_path = segment(project), segment(run)
    if attempt is not None:
        _attempt(attempt)
    evidence = {}
    preparation = {"status": "NOT_OBSERVED", "phase": None}
    saved = {}
    if state_path is not None:
        saved = json.loads(state_path.read_text())
        if not isinstance(saved, dict) or any(saved.get(key, value) != value for key, value in (
                ("server", client.base), ("project", project), ("run_id", run))):
            raise ClientError("status state belongs to a different server or Run")
        # Read-only status accepts experiment state, not an unrelated operation plan.
        if "config_sha256" not in saved:
            raise ClientError("status --state requires an experiment state file")
        if any(saved.get(key) is not None and not isinstance(saved[key], dict)
               for key in ("preparation", "submission")):
            raise ClientError("invalid saved preparation or Submission identity")
        if saved.get("preparation"):
            pid = segment(saved["preparation"]["preparation_id"])
            current = _identity(client.call(f"/api/projects/{project_path}/experiment-preparations/{pid}"), project, run)
            if current.get("preparation_id") != saved["preparation"]["preparation_id"]:
                raise ClientError("Preparation identity differs from the saved experiment")
            evidence["preparation"] = current
            preparation = {key: current.get(key) for key in ("status", "phase", "error_code")}
        if saved.get("submission"):
            original = saved["submission"]
            sid = segment(original["submission_id"])
            current = _identity(client.call("/api/submissions/" + sid), project, run)
            if (current.get("submission_id"), current.get("attempt_id")) != (original["submission_id"], original["attempt_id"]):
                raise ClientError("Submission identity differs from the saved exact Attempt")
            if attempt is not None and attempt != current["attempt_id"]:
                raise ClientError("status Attempt differs from the saved experiment")
            attempt = _attempt(current["attempt_id"])
            evidence["submission"] = current
            if "execution-progress.v1" in health.get("capabilities", []):
                progress = client.call("/api/submissions/" + sid + "/progress")
                if not isinstance(progress, dict) or any(progress.get(key) != current[key]
                        for key in ("submission_id", "run_id", "attempt_id")):
                    raise ClientError("progress identity differs from the saved exact Submission")
                evidence["progress"] = progress
            if preparation["status"] == "NOT_OBSERVED":
                preparation = {"status": current["status"], "phase": None}
    endpoint = f"/api/runs/{project_path}/{run_path}"
    try:
        run_value = _identity(client.call(endpoint), project, run)
    except ClientError as exc:
        # A preparation can exist before its Run has been frozen. Other 404s
        # and all unexpected failures remain errors instead of guessed states.
        if exc.status != 404 or "preparation" not in evidence or saved.get("submission"):
            raise
        run_value = {}
    listing = {}
    selected = {}
    if run_value:
        listing = _identity(client.call(endpoint + "/attempts"), project, run)
        entries = listing.get("attempts")
        if not isinstance(entries, list):
            raise ClientError("invalid Run Attempt listing")
        attempt = attempt or listing.get("current_attempt_id")
        if attempt is not None:
            _attempt(attempt)
            selected = next((item for item in entries if isinstance(item, dict)
                and item.get("attempt_id") == attempt and item.get("identity") == run + "::" + attempt), None)
            if selected is None:
                raise ClientError("requested exact Attempt is absent from the Run listing")
        elif entries:
            raise ClientError("Run has Attempts but no current Attempt; choose --attempt")
    scheduler = _bound_layer(run_value, "scheduler", attempt)
    process = _bound_layer(run_value, "process", attempt)
    scheduler_detail = _mapping(scheduler.get("detail"))
    collection = {}
    collection_error = None
    if attempt is not None and "result-collection.v1" in health.get("capabilities", []):
        try:
            collection = _identity(client.call(endpoint + f"/attempts/{segment(attempt)}/collection"),
                                   project, run, attempt=attempt)
        except ClientError as exc:
            # Keep other layers useful when an old/non-container Attempt has no
            # collection binding. A failed read never implies publication success.
            if exc.status not in {404, 409}:
                raise
            collection_error = {"status": exc.status, "error_code": exc.error_code}
            evidence["collection_error"] = {**collection_error, "details": exc.details}
    training = _mapping(collection.get("training"))
    result = {
        "project": project, "run_id": run, "attempt_id": attempt,
        "preparation": preparation,
        "scheduler": {"state": scheduler.get("state") or selected.get("state") or ("NOT_SUBMITTED" if attempt is None else "UNKNOWN"),
                      "observed_at": scheduler.get("as_of"), "stale": scheduler.get("stale", True),
                      "reason": scheduler_detail.get("reason") or scheduler_detail.get("pending_reason")},
        "training": {"status": training.get("status", "UNKNOWN"), "exit_code": training.get("exit_code"),
                     "process_state": process.get("state"), "observed_at": process.get("as_of"),
                     "stale": process.get("stale", True)},
        "publication": {"status": collection.get("status", "NOT_OBSERVED"), "phase": collection.get("phase"),
                        "download_available": collection.get("status") == "AVAILABLE"
                            and collection.get("verification") == "SHA256_VERIFIED"
                            and collection.get("download_available") is True,
                        "diagnostic": collection.get("diagnostic"), "error": collection_error},
        "verification": {"status": collection.get("verification", "NOT_OBSERVED")},
    }
    if details:
        evidence.update(run=run_value, attempts=listing, collection=collection)
        if attempt is not None and run_value:
            shown = client.call(f"/api/attempts/{project_path}/{segment(run + '::' + attempt)}/show")
            if (not isinstance(shown, dict) or shown.get("run_id") != run
                    or _mapping(shown.get("attempt")).get("attempt_id") != attempt
                    or shown.get("scope") != {"project": project, "scope_type": "attempt", "object_id": run + "::" + attempt}):
                raise ClientError("expanded evidence differs from the requested exact Attempt")
            evidence["attempt"] = shown
        result["evidence"] = evidence
    return result
