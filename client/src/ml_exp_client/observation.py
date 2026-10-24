"""Resolve saved exact identities for observation without local experiment source."""
from __future__ import annotations

import json
import re

from .api import ClientError, attempt_id, segment
from .state import bound_response

_ACTION = re.compile(r"action-[a-f0-9]{16}\Z")
_PREPARATION = re.compile(r"preparation-[a-f0-9]{32}\Z")
_DIGEST = re.compile(r"sha256:[a-f0-9]{64}\Z")
_OPERATIONS = {"cancel": "CANCEL_RUN", "retry": "RETRY_ATTEMPT"}


def _read_saved(state_path):
    try:
        saved = json.loads(state_path.read_text())
    except (OSError, ValueError):
        raise ClientError("cannot read observation state; use the original saved JSON file") from None
    if not isinstance(saved, dict):
        raise ClientError("observation requires a saved experiment or training-operation state")
    if "server" in saved and (not isinstance(saved["server"], str) or not saved["server"]):
        raise ClientError("invalid saved server or Run identity")
    return saved


def observation_server(state_path):
    """Read only the URL hint, including an uncertain operation awaiting attachment."""
    return _read_saved(state_path).get("server")


def observation_reference(state_path):
    """Read identity hints before connecting; never resolve an unknown plan as latest."""
    saved = _read_saved(state_path)
    if saved.get("contract") == "training-operation.v1":
        kind = "training-operation"
        plan = saved.get("plan")
        if saved.get("operation") not in _OPERATIONS or not isinstance(plan, dict):
            raise ClientError("training preparation reply is unknown; inspect its recovery URL and attach the original Action; no Attempt was guessed")
        target = attempt_id(plan.get("attempt_id"))
        attempt_id(saved.get("source_attempt_id"))
        if not _ACTION.fullmatch(str(plan.get("action_id", ""))):
            raise ClientError("invalid saved training Action identity")
    elif "config_sha256" in saved:
        kind = "experiment"
        for name in ("preparation", "submission"):
            if saved.get(name) is not None and not isinstance(saved[name], dict):
                raise ClientError("invalid saved preparation or Submission identity")
        submission = saved.get("submission") or (saved.get("preparation") or {}).get("submission") or {}
        if not isinstance(submission, dict):
            raise ClientError("invalid saved Submission identity")
        target = attempt_id(submission.get("attempt_id")) if submission else None
        if submission and not _ACTION.fullmatch(str(submission.get("submission_id", ""))):
            raise ClientError("invalid saved Submission identity")
    else:
        raise ClientError("observation requires an experiment state or training-operation.v1 plan")
    for key in ("server", "project", "run_id"):
        if key in saved and (not isinstance(saved[key], str) or not saved[key]):
            raise ClientError("invalid saved server or Run identity")
    for key in ("project", "run_id"):
        if saved.get(key) is not None:
            segment(saved[key])
    return {"server": saved.get("server"), "project": saved.get("project"),
            "run_id": saved.get("run_id"), "attempt_id": target,
            "state_kind": kind, "saved": saved}


def _action_identity(action, saved, project, run, target):
    plan = saved["plan"]
    source = attempt_id(saved["source_attempt_id"])
    if (not isinstance(action, dict)
            or action.get("operation") != _OPERATIONS[saved["operation"]]
            or action.get("scope") != {"project": project, "scope_type": "attempt", "object_id": run + "::" + source}
            or action.get("run_id") != run or action.get("attempt_id") != target
            or not isinstance(action.get("execution"), dict)
            or any(not _DIGEST.fullmatch(str(action.get(key, "")))
                   for key in ("intent_digest", "gate_bundle_digest"))
            or any(action.get(key) != plan.get(key) for key in
                   ("action_id", "intent_digest", "gate_bundle_digest"))):
        raise ClientError("Action identity differs from the saved exact training operation")
    if (saved["operation"] == "cancel" and target != source
            or saved["operation"] == "retry" and target == source):
        raise ClientError("saved training Action targets an invalid Attempt")
    request = saved.get("request")
    if not isinstance(request, dict) or request.get("new_attempt_id", target) != target:
        raise ClientError("Action target differs from the saved retry request")
    return action


def resolve_observation(client, project=None, run=None, attempt=None, state_path=None):
    """Bind reads to one saved identity; only GET original Submission/Action records."""
    reference = (observation_reference(state_path) if state_path is not None else
                 {"server": None, "project": None, "run_id": None, "attempt_id": None,
                  "state_kind": None, "saved": {}})
    base = getattr(client, "base", None)
    if reference["server"] is not None and (base is None or reference["server"].rstrip("/") != base.rstrip("/")):
        raise ClientError("observation state belongs to a different server or Run")
    for supplied, key in ((project, "project"), (run, "run_id"), (attempt, "attempt_id")):
        if supplied is not None and reference[key] is not None and supplied != reference[key]:
            raise ClientError("requested identity differs from the saved exact Run/Attempt")
    project, run = project or reference["project"], run or reference["run_id"]
    if project is None or run is None:
        raise ClientError("provide --project and --run for old state without a complete Run identity")
    segment(project)
    segment(run)
    attempt = attempt or reference["attempt_id"]
    if attempt is not None:
        attempt_id(attempt)
    saved = reference["saved"]
    reference.update(server=base, project=project, run_id=run, attempt_id=attempt,
                     submission=None, action=None, preparation=None)
    original = saved.get("submission")
    if reference["state_kind"] == "experiment" and not original and saved.get("preparation"):
        previous = saved["preparation"]
        pid = previous.get("preparation_id")
        if (not _PREPARATION.fullmatch(str(pid))
                or any(previous.get(key, value) != value for key, value in (("project", project), ("run_id", run)))):
            raise ClientError("Preparation identity differs from the saved experiment")
        current = bound_response(
            client.call(f"/api/projects/{segment(project)}/experiment-preparations/{pid}"),
            {"project": project, "run_id": run, "preparation_id": pid},
            message="Preparation identity differs from the saved experiment")
        reference["preparation"] = current
        original = current.get("submission")
        if original is not None:
            original = bound_response(original, {"project": project, "run_id": run},
                                      message="prepared Submission belongs to a different Run")
            target = attempt_id(original.get("attempt_id"))
            if attempt is not None and target != attempt:
                raise ClientError("prepared Submission differs from the saved exact Run/Attempt")
            previous_submission = previous.get("submission")
            if previous_submission and original.get("submission_id") != previous_submission.get("submission_id"):
                raise ClientError("prepared Submission differs from the saved exact identity")
            attempt = reference["attempt_id"] = target
    if reference["state_kind"] == "experiment" and original:
        if not _ACTION.fullmatch(str(original.get("submission_id", ""))):
            raise ClientError("invalid saved Submission identity")
        for key in ("project", "run_id", "attempt_id"):
            if original.get(key, reference[key]) != reference[key]:
                raise ClientError("saved Submission belongs to a different Run/Attempt")
        current = bound_response(client.call("/api/submissions/" + segment(original["submission_id"])),
                                 {"project": project, "run_id": run, "attempt_id": attempt,
                                  "submission_id": original["submission_id"]},
                                 message="Submission identity differs from the saved exact Attempt")
        reference["submission"] = current
    elif reference["state_kind"] == "training-operation":
        _action_identity(saved["plan"], saved, project, run, attempt)
        if saved.get("action") is not None:
            _action_identity(saved["action"], saved, project, run, attempt)
        reference["action"] = _action_identity(
            client.call("/api/actions/" + segment(saved["plan"]["action_id"])),
            saved, project, run, attempt,
        )
    return reference
