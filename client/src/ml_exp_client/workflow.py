"""One JSON experiment, resumable API preparation, explicit scheduler execution."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import time
from urllib.parse import urlencode

from .api import ClientError, segment
from .archives import data_archive, source_archive
from .artifacts import acknowledge, download
from .assets import upload_asset_parts
from .config import read_config
from .state import SavedPlan, bound_response, save


TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED", "PREEMPTED", "TIMEOUT"}


def report(value, *, json_output=False):
    progress = value.get("progress", value)
    from .presentation import progress_summary
    print(json.dumps(progress) if json_output else progress_summary(progress), file=sys.stderr)


def wait_resource(client, endpoint, seconds, *, pending=("EXECUTING",), validate=None, progress_validate=None, json_output=False):
    deadline = time.monotonic() + seconds
    while True:
        value = client.call(endpoint)
        if validate is not None:
            value = validate(value)
        if value["status"] not in pending:
            return value
        progress = client.call(endpoint + "/progress")
        report(progress_validate(progress) if progress_validate is not None else progress, json_output=json_output)
        if time.monotonic() >= deadline:
            raise ClientError("waiting timed out; execution continues; resume to inspect saved IDs, never resubmit")
        time.sleep(5)


def wait_download(client, project, run, attempt, state, state_path, seconds, *, json_output=False):
    """Observe publication for the original Attempt; never request collection."""
    endpoint = f"/api/runs/{segment(project)}/{segment(run)}/attempts/{segment(attempt)}/collection"
    deadline = time.monotonic() + seconds
    while True:
        value = client.call(endpoint)
        if not isinstance(value, dict) or (value.get("project"), value.get("run_id"), value.get("attempt_id")) != (project, run, attempt):
            raise ClientError("collection identity differs from the saved exact Attempt")
        state["collection"] = value
        save(state_path, state)
        status = value.get("status")
        if status == "AVAILABLE" and value.get("verification") == "SHA256_VERIFIED" and value.get("download_available") is True:
            return value
        if status == "RELEASED":
            raise ClientError("original Attempt archive was RELEASED; collection will not recreate it",
                              status=410, error_code="ARTIFACT_RELEASED")
        if status in ("FAILED", "RECONCILE_REQUIRED"):
            raise ClientError(f"original Attempt collection is {status}; inspect saved collection diagnostics before explicit recovery; no collection was requested",
                              error_code="RESULT_COLLECTION_" + status)
        if status not in ("NOT_COLLECTED", "QUEUED", "COLLECTING", "AVAILABLE"):
            raise ClientError("unknown collection status; inspect the saved exact Attempt before downloading")
        report(value, json_output=json_output)
        if time.monotonic() >= deadline:
            raise ClientError("result publication wait timed out; resume the saved state to observe the same Attempt; no collection was requested",
                              error_code="RESULT_WAIT_TIMEOUT")
        time.sleep(5)


def retry_download_acknowledgement(client, project, run, attempt, state, state_path, directory):
    """Retry a saved PENDING ACK after verifying its local identity and bytes."""
    previous = state.get("download", {})
    if previous.get("acknowledgement", {}).get("status") != "PENDING":
        return
    if directory is None:
        raise ClientError("download acknowledgement is PENDING; resume with --download-to pointing to the original directory, or use ml-exp run acknowledge")
    local = json.loads((directory / "verification.json").read_text())
    identity = {"project": project, "run_id": run, "attempt_id": attempt}
    if any(local.get(key) != value for key, value in identity.items()) or any(
            local.get(key) != previous.get(key) for key in ("archive_sha256", "archive_bytes", "files")):
        raise ClientError("local download proof differs from the saved exact Attempt; acknowledgement was not sent")
    # acknowledge() rehashes the archive/every file and preserves the locally
    # saved KEEP/release policy. A default CLI flag must not override that choice.
    state["download"] = acknowledge(client, directory)
    save(state_path, state)


def experiment(client, health, config_path, state_path, *, resume=False, execute=False, seconds=1800, out=None, continue_preparation=False, keep_server_copy=False, json_output=False):
    config, source = read_config(config_path)
    if continue_preparation and not resume:
        raise ClientError("--continue-preparation requires --resume and inspection of saved dependencies")
    if "server-experiment-preparation.v1" not in health.get("capabilities", []):
        raise ClientError("server lacks server-experiment-preparation.v1; upgrade for server-owned preparation")
    if "dockerfile-only.v1" not in health.get("capabilities", []):
        raise ClientError("server lacks dockerfile-only.v1; upgrade before using the experiment workflow")
    if (any("required_when" in definition
            for definition in config.get("metrics_schema", {}).get("definitions", {}).values())
            and "metric-records.v1" not in health.get("capabilities", [])):
        raise ClientError("server lacks metric-records.v1; upgrade before scoping metric requirements")
    if config.get("data_preparation") is not None and "data-preparation.v1" not in health.get("capabilities", []):
        raise ClientError("server lacks data-preparation.v1; upgrade before using a download script")
    if ("wandb" in config or "parameters" in config) and "wandb-sync.v1" not in health.get("capabilities", []):
        raise ClientError("server lacks wandb-sync.v1; upgrade before specifying publication options")
    if ("preparation_enabled" in config.get("wandb", {}) and
            "wandb-preparation-sync.v1" not in health.get("capabilities", [])):
        raise ClientError("server lacks wandb-preparation-sync.v1; upgrade before selecting preparation publication")
    if (any(config.get("resources", {}).get(key) is not None for key in ("allocation_policy", "priority")) and
            "pai-idle-resources.v1" not in health.get("capabilities", [])):
        raise ClientError("server lacks pai-idle-resources.v1; upgrade before selecting DLC resource policy")
    if (config.get("checkpoint_persistence") is not None or config.get("resume_from") is not None) and "persistent-checkpoints.v1" not in health.get("capabilities", []):
        raise ClientError("server lacks persistent-checkpoints.v1; upgrade before using backend checkpoint state")
    if state_path.resolve().is_relative_to(source):
        raise ClientError("save workflow state outside the uploaded source directory")
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    state = json.loads(state_path.read_text()) if state_path.exists() else {"config_sha256": fingerprint}
    if state_path.exists() and not resume:
        raise ClientError("state file exists; use --resume to inspect and continue the same experiment")
    if state.get("config_sha256") != fingerprint:
        raise ClientError("resume configuration changed; use a new Run and state file")
    identity = {"project": config["project"], "run_id": config["run_id"]}
    if getattr(client, "base", None):
        identity["server"] = client.base
    if any(state.get(key, value) != value for key, value in identity.items()):
        raise ClientError("workflow state belongs to a different server or Run")
    for marker in ("scheduler_requested", "scheduler_authorization_requested"):
        if marker in state and type(state[marker]) is not bool:
            raise ClientError("invalid saved workflow effect marker")
    state.update(identity)
    plan = SavedPlan(client, state, state_path, persist=save)
    binding = {"project": config["project"], "run_id": config["run_id"]}
    def validate_submission(value, original=None):
        expected = dict(binding)
        if original is not None:
            expected.update(submission_id=original[0], attempt_id=original[1])
        value = bound_response(value, expected, message="Submission identity differs from the saved exact Run/Attempt")
        if (not re.fullmatch(r"action-[a-f0-9]{16}", str(value.get("submission_id", "")))
                or not re.fullmatch(r"attempt-[0-9]{3,}", str(value.get("attempt_id", "")))):
            raise ClientError("Submission lacks an exact scheduler identity")
        return value
    def validate_preparation(value):
        expected = dict(binding)
        previous = state.get("preparation")
        if previous:
            expected["preparation_id"] = previous["preparation_id"]
        value = bound_response(value, expected, message="Preparation identity differs from the saved experiment Run")
        if not re.fullmatch(r"preparation-[a-f0-9]{32}", str(value.get("preparation_id", ""))):
            raise ClientError("Preparation lacks an exact identity")
        if value.get("run") is not None:
            bound_response(value["run"], binding, message="prepared Run identity differs from the experiment")
        if value.get("submission") is not None:
            validate_submission(value["submission"])
        return value
    if state.get("preparation"):
        validate_preparation(state["preparation"])
    if state.get("submission"):
        validate_submission(state["submission"])
    if "preparation_request" in state:
        request = state["preparation_request"]
        if not isinstance(request, dict) or not isinstance(request.get("run"), dict) or request["run"].get("run_id") != config["run_id"]:
            raise ClientError("saved preparation request differs from the experiment Run")
    if out is not None:
        if state.get("download_directory") and Path(state["download_directory"]).resolve() != out.resolve():
            raise ClientError("download directory differs from saved workflow state; resume with the original directory")
        state["download_directory"] = str(out.resolve())
        # A resumed command's default False must not release a copy originally
        # requested as KEEP. Explicit release remains an acknowledge operation.
        state["download_keep_server_copy"] = state.get("download_keep_server_copy", False) or keep_server_copy
    download_out = Path(state["download_directory"]) if state.get("download_directory") else None
    definition = {key: config[key] for key in ("run_id", "executor", "arguments", "resources", "env", "outputs", "required_outputs", "checkpoint_upload", "metrics_schema", "evaluation", "data_preparation", "checkpoint_persistence", "resume_from", "wandb", "parameters") if key in config}
    requirements = dict(config.get("requirements", {}))
    if config.get("inputs"):
        requirements["data_assets"] = True
    selection = {"requirements": requirements}
    if "executor_selector" in config:
        selection["executor_selector"] = config["executor_selector"]
    frozen_upload_target = state.get("upload_target") if "target-backed-data-upload.v1" in health.get("capabilities", []) else None
    if "submission" not in state and frozen_upload_target is None:
        state["matching_preview"] = client.call("/api/executors/match", data={"project": config["project"], "run": definition, **selection})
    upload_target = None
    if "target-backed-data-upload.v1" in health.get("capabilities", []):
        chosen = (frozen_upload_target.get("executor") if isinstance(frozen_upload_target, dict)
                  else state.get("matching_preview", {}).get("executor"))
        if not isinstance(chosen, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", chosen):
            raise ClientError("executor matching did not return a data upload target")
        upload_target = {"executor": chosen}
        if state.get("upload_target", upload_target) != upload_target:
            raise ClientError("resume executor changed; inspect the saved data target")
        state["upload_target"] = upload_target
        definition["executor"] = chosen
        selection.pop("executor_selector", None)
    payload = source_archive(source)
    source_hash = hashlib.sha256(gzip.decompress(payload)).hexdigest()
    if state.get("source_archive_content_sha256", source_hash) != source_hash:
        raise ClientError("resume source changed; use a new Run and state file")
    state["source_archive_content_sha256"] = source_hash
    save(state_path, state)
    project = segment(config["project"])
    if "source_id" not in state:
        query = urlencode({"project": config["project"], "sha256": hashlib.sha256(payload).hexdigest()})
        state["source_id"] = client.call("/api/source-imports/archive?" + query, raw=payload)["source_id"]
        save(state_path, state)
    bindings = []
    for number, item in enumerate(config.get("inputs", [])):
        asset_id = item.get("asset_id")
        if not asset_id:
            upload_state = state_path.with_name(state_path.name + f".asset-{number}.json")
            with tempfile.TemporaryFile() as stream:
                digest, length = data_archive((config_path.parent / item["directory"]).resolve(), stream)
                if "input_bindings" in state and state["input_bindings"][number]["asset_id"] != "asset." + digest:
                    raise ClientError("resume data changed; use a new Run and state file")
                limits = client.call("/api/storage-limits")
                if limits["asset_archive_bytes"] is not None and length > limits["asset_archive_bytes"]:
                    raise ClientError("data archive exceeds server upload limit")
                asset_id = upload_asset_parts(client, config["project"], stream, digest, length, upload_state, target=upload_target)["asset_id"]
        bindings.append({"asset_id": asset_id, "mount_path": item["mount_path"]})
    state["input_bindings"] = bindings
    save(state_path, state)
    if "submission" not in state:
        if "preparation_request" not in state:
            run = {**definition, "inputs": bindings}
            request = {"run": run, **selection, "max_gpu_hours": config["max_gpu_hours"]}
            if "runtime" in state:
                existing = state["runtime"]
                endpoint = f"/api/projects/{project}/runtimes/{segment(existing['runtime_id'])}"
                existing = client.call(endpoint)
                if state.get("build_requested") and existing["status"] == "PREPARED":
                    raise ClientError("saved build request outcome is uncertain; inspect Runtime before continuing")
                request["run"]["runtime_id"] = existing["runtime_id"]
            else:
                runtime_spec = {key: config[key] for key in ("dockerfile", "entrypoint", "workdir") if key in config}
                runtime_spec.update(source_id=state["source_id"], entrypoint=config.get("entrypoint", ["python3", "train.py"]))
                request["runtime"] = runtime_spec
            state["preparation_request"] = request
            save(state_path, state)
        if "preparation" not in state:
            # The server binds one request to project/run_id before starting
            # effects. A lost POST response can be safely retrieved by the same
            # request; it never creates another build or copy task.
            plan.request("preparation", f"/api/projects/{project}/experiment-preparations", data=state["preparation_request"], validate=validate_preparation)
        endpoint = f"/api/projects/{project}/experiment-preparations/{segment(state['preparation']['preparation_id'])}"
        if continue_preparation:
            current = validate_preparation(client.call(endpoint))
            if current["status"] == "RECONCILE_REQUIRED":
                client.call(endpoint + "/continue", data={})
        preparation = wait_resource(client, endpoint, seconds, validate=validate_preparation, json_output=json_output)
        plan.store("preparation", preparation)
        if preparation["status"] != "READY":
            raise ClientError("server preparation requires dependency inspection; saved Runtime/delivery IDs must be reconciled, never resubmitted")
        state.update(run=preparation["run"], submission=preparation["submission"], matching=preparation["matching"])
        state["runtime"] = client.call(f"/api/projects/{project}/runtimes/{segment(preparation['runtime_id'])}")
        save(state_path, state)
    submission = state["submission"]
    endpoint = "/api/submissions/" + segment(submission["submission_id"])
    original = (submission["submission_id"], submission["attempt_id"])
    validate = lambda value: validate_submission(value, original)
    def validate_progress(value):
        return bound_response(value, {"submission_id": original[0], "run_id": config["run_id"], "attempt_id": original[1]},
                              message="Submission progress identity differs from the saved exact Run/Attempt")
    submission = plan.observe("submission", endpoint, validate=validate)
    if execute:
        if state.get("scheduler_requested") and submission["status"] in {"PREPARED", "AUTHORIZED"}:
            raise ClientError("scheduler request outcome is uncertain; inspect saved Submission before explicit execution")
        if submission["status"] == "PREPARED" and submission["ready"]:
            submission = plan.request("submission", endpoint + "/authorize",
                data={"note": "explicit ml-exp run submit --execute"}, marker="scheduler_authorization_requested",
                validate=validate,
                uncertain="scheduler authorization outcome is uncertain; inspect saved Submission before explicit execution")
        if submission["status"] == "AUTHORIZED":
            plan.request("submission", endpoint + "/execute", data={"confirmation": submission["confirmation"]},
                marker="scheduler_requested", validate=validate,
                uncertain="scheduler request outcome is uncertain; inspect saved Submission before explicit execution")
        submission = wait_resource(client, endpoint, seconds, validate=validate, progress_validate=validate_progress, json_output=json_output)
    plan.store("submission", submission, validate=validate)
    if resume:
        retry_download_acknowledgement(client, config["project"], config["run_id"], submission["attempt_id"],
                                       state, state_path, download_out)
    if not execute:
        return state
    if submission["status"] != "VERIFIED":
        raise ClientError("submission was not verified; inspect gates/progress or reconcile; do not resubmit")
    endpoint = f"/api/runs/{project}/{segment(config['run_id'])}"
    deadline = time.monotonic() + seconds
    while True:
        result = bound_response(client.call(endpoint), binding, message="training Run response differs from the experiment")
        report(validate_progress(client.call("/api/submissions/" + segment(submission["submission_id"]) + "/progress")), json_output=json_output)
        if result["scheduler_state"] in TERMINAL:
            attempts = bound_response(client.call(endpoint + "/attempts"), binding,
                                      message="training Attempt listing differs from the experiment")
            if attempts["current_attempt_id"] != submission["attempt_id"]:
                raise ClientError("Run now refers to a different Attempt; inspect the saved exact submission")
            state["result"] = result
            save(state_path, state)
            break
        if time.monotonic() >= deadline:
            raise ClientError("training wait timed out; resume observes the same job and never creates a new Attempt")
        time.sleep(5)
    if download_out is not None and "download" not in state:
        if "result-collection.v1" in health.get("capabilities", []):
            wait_download(client, config["project"], config["run_id"], submission["attempt_id"], state, state_path, seconds, json_output=json_output)
        state["download"] = download(client, config["project"], config["run_id"], submission["attempt_id"], download_out,
                                     keep_server_copy=state.get("download_keep_server_copy", keep_server_copy))
        save(state_path, state)
    if result["scheduler_state"] != "SUCCEEDED":
        raise ClientError("training ended unsuccessfully; saved state and artifacts are available for diagnosis")
    return state
