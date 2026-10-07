"""One JSON experiment, resumable API preparation, explicit scheduler execution."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import time
from urllib.parse import urlencode

from .api import ClientError, data_archive, download, save, segment, source_archive, upload_asset_parts


TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED", "PREEMPTED", "TIMEOUT"}


def validate_dockerfile(source: Path, name: str):
    relative = PurePosixPath(name)
    path = source / name
    if (relative.is_absolute() or ".." in relative.parts or "\\" in name or
            not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(source.resolve())):
        raise ClientError("Dockerfile must be a regular file inside the source directory")
    stages = set()
    number = 0
    for line in path.read_text().splitlines():
        if re.match(r"(?i)^\s*FROM\s", line):
            parts = line.split()
            image = parts[1] if len(parts) > 1 else ""
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}", image) and image not in stages:
                raise ClientError("Dockerfile FROM must use an approved sha256-pinned base image or an earlier stage")
            stages.add(str(number))
            number += 1
            if len(parts) == 4 and parts[2].upper() == "AS":
                stages.add(parts[3])
    if not stages:
        raise ClientError("Dockerfile needs a digest-pinned FROM instruction")


def read_config(path: Path):
    value = json.loads(path.read_text())
    allowed = {"project", "run_id", "source", "dockerfile", "entrypoint", "workdir", "executor",
               "arguments", "env", "resources", "outputs", "inputs", "checkpoint_upload", "max_gpu_hours", "metrics_schema", "evaluation", "data_preparation", "checkpoint_persistence", "resume_from", "executor_selector", "requirements"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ClientError("experiment config contains unknown fields")
    evaluation = value.get("evaluation", {})
    try:
        valid_evaluation = isinstance(evaluation, dict) and len(json.dumps(evaluation, allow_nan=False).encode()) <= 16384
    except ValueError:
        valid_evaluation = False
    if not valid_evaluation:
        raise ClientError("evaluation must be a finite JSON object of at most 16 KiB")
    schema = value.get("metrics_schema", {})
    if not isinstance(schema, dict) or set(schema) - {"schema_version", "definitions"} or schema.get("schema_version", 1) != 1:
        raise ClientError("metrics_schema must use schema_version 1 and definitions")
    definitions = schema.get("definitions", {})
    if not isinstance(definitions, dict) or len(definitions) > 256:
        raise ClientError("metrics_schema supports at most 256 definitions")
    for name, definition in definitions.items():
        if (not isinstance(name, str) or not name.strip() or len(name) > 128 or any(ord(c) < 32 for c in name) or
                not isinstance(definition, dict) or set(definition) - {"unit", "required", "description", "aggregation", "numerator_unit", "denominator_unit"} or
                not isinstance(definition.get("unit"), str) or not definition["unit"].strip() or len(definition["unit"]) > 128 or any(ord(c) < 32 for c in definition["unit"]) or
                "required" in definition and not isinstance(definition["required"], bool)):
            raise ClientError("each metric definition needs a valid name, explicit unit and optional boolean required")
    for key in ("project", "run_id"):
        if not isinstance(value.get(key), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value[key]):
            raise ClientError("experiment needs valid project, run_id and executor")
    executor = value.get("executor")
    selector = value.get("executor_selector")
    if ((executor is None) == (selector is None) or executor is not None and
            (not isinstance(executor, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", executor))):
        raise ClientError("choose an executor or an executor_selector")
    if selector is not None and (not isinstance(selector, dict) or set(selector) - {"backend", "candidates"} or
            not isinstance(selector.get("backend"), (str, type(None))) or selector.get("backend") not in {None, "slurm", "sensecore"} or
            not isinstance(selector.get("candidates", []), list) or len(selector.get("candidates", [])) > 64 or
            any(not isinstance(i, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", i) for i in selector.get("candidates", [])) or
            not (selector.get("backend") or selector.get("candidates"))):
        raise ClientError("executor_selector needs a backend or valid candidate IDs")
    requirements = value.get("requirements", {})
    flags = {"data_assets", "data_preparation", "persistent_checkpoints", "queue_reason", "exit_code", "preemption_reason", "scheduler_walltime", "offline_output_export"}
    if (not isinstance(requirements, dict) or set(requirements) - flags - {"platform"} or
            not isinstance(requirements.get("platform", "linux/amd64"), str) or
            any(type(item) is not bool for key, item in requirements.items() if key in flags)):
        raise ClientError("requirements need a platform and boolean capability flags")
    if not isinstance(value.get("source"), str) or type(value.get("max_gpu_hours")) not in (int, float) or not math.isfinite(value["max_gpu_hours"]) or value["max_gpu_hours"] <= 0:
        raise ClientError("experiment needs source and a positive max_gpu_hours budget")
    source = (path.parent / value["source"]).resolve()
    validate_dockerfile(source, value.get("dockerfile", "Dockerfile"))
    preparation = value.get("data_preparation")
    if preparation is not None:
        if not isinstance(preparation, dict) or set(preparation) - {"script", "interpreter", "arguments", "timeout_seconds", "expected_content_sha256"}:
            raise ClientError("invalid data_preparation definition")
        name = preparation.get("script", "")
        if not isinstance(name, str) or not name or PurePosixPath(name).is_absolute() or any(part.startswith(".") for part in name.split("/")) or "\\" in name or "\x00" in name:
            raise ClientError("data preparation script must stay inside source")
        script = source / name
        args = preparation.get("arguments", [])
        timeout = preparation.get("timeout_seconds", 1200)
        sha = preparation.get("expected_content_sha256")
        interpreter = preparation.get("interpreter", "python3")
        if (not script.is_file() or script.is_symlink() or not script.resolve().is_relative_to(source) or
                not isinstance(interpreter, str) or interpreter not in {"python3", "/bin/sh"} or
                not isinstance(args, list) or len(args) > 128 or any(not isinstance(arg, str) or not arg or "\x00" in arg or len(arg) > 8192 for arg in args) or
                type(timeout) is not int or not 5 <= timeout <= 86400 or
                sha is not None and (not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha))):
            raise ClientError("data preparation needs an existing script, valid argv, interpreter and bounded timeout")
    workdir = value.get("workdir", "/workspace")
    if not isinstance(workdir, str) or PurePosixPath(workdir).parts[:2] != ("/", "workspace") or ".." in PurePosixPath(workdir).parts or "\x00" in workdir:
        raise ClientError("workdir must stay within /workspace")
    for key in ("entrypoint", "arguments"):
        items = value.get(key, ["python3", "train.py"] if key == "entrypoint" else [])
        if not isinstance(items, list) or key == "entrypoint" and not items or any(not isinstance(i, str) or not i or "\x00" in i for i in items):
            raise ClientError("entrypoint and arguments must be argv arrays")
    env = value.get("env", {})
    if not isinstance(env, dict) or any(not isinstance(item, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", key) or
            re.search(r"TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY|PROXY|AUTHORIZATION", key) or key.startswith("ML_EXPD_") or
            key in {"OUTPUT_DIR", "INPUTS_DIR", "DATA_DIR", "STATE_DIR", "RESUME_DIR", "PROJECT_NAME", "RUN_ID", "ATTEMPT_ID", "SOURCE_ID", "BACKEND_JOB_ID"} for key, item in env.items()):
        raise ClientError("environment contains a reserved or credential-bearing field")
    inputs = value.get("inputs", [])
    if not isinstance(inputs, list) or len(inputs) > 32:
        raise ClientError("inputs must be an array of at most 32 data bindings")
    mounts = set()
    for item in inputs:
        if (not isinstance(item, dict) or set(item) - {"directory", "asset_id", "mount_path"} or
                ("directory" in item) == ("asset_id" in item) or
                not re.fullmatch(r"/inputs/[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", str(item.get("mount_path", ""))) or item["mount_path"] in mounts):
            raise ClientError("each input needs directory or asset_id and a distinct /inputs/name mount_path")
        mounts.add(item["mount_path"])
        if "asset_id" in item and not re.fullmatch(r"asset\.[0-9a-f]{64}", str(item["asset_id"])):
            raise ClientError("invalid data asset identity")
        if "directory" in item:
            if not isinstance(item["directory"], str):
                raise ClientError("input directory must be a path")
            directory = (path.parent / item["directory"]).resolve()
            if not directory.is_dir() or directory.is_relative_to(source):
                raise ClientError("data must be a separate existing directory outside source")
    resources = value.get("resources", {})
    if not isinstance(resources, dict) or set(resources) - {"gpus", "cpus", "memory_gb", "max_time"}:
        raise ClientError("invalid resources")
    for key, limit in (("gpus", 64), ("cpus", 512), ("memory_gb", 4096)):
        if key in resources and (type(resources[key]) is not int or not 1 <= resources[key] <= limit):
            raise ClientError("resources exceed supported bounds")
    if "max_time" in resources and not re.fullmatch(r"[0-9]{2,3}:[0-5][0-9]:[0-5][0-9]", str(resources["max_time"])):
        raise ClientError("max_time must be HH:MM:SS")
    checkpoint = value.get("checkpoint_upload")
    if checkpoint is not None and (not isinstance(checkpoint, dict) or set(checkpoint) != {"interval_seconds"} or type(checkpoint["interval_seconds"]) is not int or not 5 <= checkpoint["interval_seconds"] <= 3600):
        raise ClientError("checkpoint_upload requires interval_seconds between 5 and 3600")
    persistence = value.get("checkpoint_persistence")
    if persistence is not None and (not isinstance(persistence, dict) or set(persistence) - {"interval_seconds"} or type(persistence.get("interval_seconds", 60)) is not int or not 5 <= persistence.get("interval_seconds", 60) <= 3600):
        raise ClientError("checkpoint_persistence requires an interval between 5 and 3600")
    resume = value.get("resume_from")
    if resume is not None:
        if (not isinstance(resume, dict) or set(resume) != {"run_id", "attempt_id", "checkpoint_id"} or
                any(not isinstance(resume[key], str) or not re.fullmatch(pattern, resume[key]) for key, pattern in (
                    ("run_id", r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}"), ("attempt_id", r"attempt-[0-9]{3,}"),
                    ("checkpoint_id", r"checkpoint\.[0-9a-f]{64}"))) or "/inputs/resume" in mounts):
            raise ClientError("resume_from needs an exact Run, Attempt and checkpoint identity; /inputs/resume is reserved")
    return value, source


def report(value):
    progress = value.get("progress", value)
    print(json.dumps({key: progress.get(key) for key in ("phase", "message", "seconds_since_progress", "diagnostic", "queue")}), file=sys.stderr)


def wait_resource(client, endpoint, seconds, *, pending=("EXECUTING",)):
    deadline = time.monotonic() + seconds
    while True:
        value = client.call(endpoint)
        if value["status"] not in pending:
            return value
        report(client.call(endpoint + "/progress"))
        if time.monotonic() >= deadline:
            raise ClientError("waiting timed out; execution continues; resume to inspect saved IDs, never resubmit")
        time.sleep(5)


def experiment(client, health, config_path, state_path, *, resume=False, execute=False, seconds=1800, out=None, continue_preparation=False):
    config, source = read_config(config_path)
    if continue_preparation and not resume:
        raise ClientError("--continue-preparation requires --resume and inspection of saved dependencies")
    if "server-experiment-preparation.v1" not in health.get("capabilities", []):
        raise ClientError("server lacks server-experiment-preparation.v1; upgrade for server-owned preparation")
    if "dockerfile-only.v1" not in health.get("capabilities", []):
        raise ClientError("server lacks dockerfile-only.v1; upgrade before using the experiment workflow")
    if config.get("data_preparation") is not None and "data-preparation.v1" not in health.get("capabilities", []):
        raise ClientError("server lacks data-preparation.v1; upgrade before using a download script")
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
    definition = {key: config[key] for key in ("run_id", "executor", "arguments", "resources", "env", "outputs", "checkpoint_upload", "metrics_schema", "evaluation", "data_preparation", "checkpoint_persistence", "resume_from") if key in config}
    requirements = dict(config.get("requirements", {}))
    if config.get("inputs"):
        requirements["data_assets"] = True
    selection = {"requirements": requirements}
    if "executor_selector" in config:
        selection["executor_selector"] = config["executor_selector"]
    if "submission" not in state:
        state["matching_preview"] = client.call("/api/executors/match", data={"project": config["project"], "run": definition, **selection})
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
                asset_id = upload_asset_parts(client, config["project"], stream, digest, length, upload_state)["asset_id"]
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
            state["preparation"] = client.call(f"/api/projects/{project}/experiment-preparations", data=state["preparation_request"])
            save(state_path, state)
        endpoint = f"/api/projects/{project}/experiment-preparations/{segment(state['preparation']['preparation_id'])}"
        if continue_preparation:
            current = client.call(endpoint)
            if current["status"] == "RECONCILE_REQUIRED":
                client.call(endpoint + "/continue", data={})
        preparation = wait_resource(client, endpoint, seconds)
        state["preparation"] = preparation
        save(state_path, state)
        if preparation["status"] != "READY":
            raise ClientError("server preparation requires dependency inspection; saved Runtime/delivery IDs must be reconciled, never resubmitted")
        state.update(run=preparation["run"], submission=preparation["submission"], matching=preparation["matching"])
        state["runtime"] = client.call(f"/api/projects/{project}/runtimes/{segment(preparation['runtime_id'])}")
        save(state_path, state)
    submission = state["submission"]
    endpoint = "/api/submissions/" + segment(submission["submission_id"])
    submission = client.call(endpoint)
    if execute:
        if submission["status"] == "PREPARED" and submission["ready"]:
            submission = client.call(endpoint + "/authorize", data={"note": "explicit ml-exp experiment --execute"})
        if submission["status"] == "AUTHORIZED":
            if state.get("scheduler_requested"):
                raise ClientError("scheduler request outcome is uncertain; inspect saved Submission before explicit execution")
            state["scheduler_requested"] = True
            save(state_path, state)
            client.call(endpoint + "/execute", data={"confirmation": submission["confirmation"]})
        submission = wait_resource(client, endpoint, seconds)
    state["submission"] = submission
    save(state_path, state)
    if not execute:
        return state
    if submission["status"] != "VERIFIED":
        raise ClientError("submission was not verified; inspect gates/progress or reconcile; do not resubmit")
    endpoint = f"/api/runs/{project}/{segment(config['run_id'])}"
    deadline = time.monotonic() + seconds
    while True:
        result = client.call(endpoint)
        report(client.call("/api/submissions/" + segment(submission["submission_id"]) + "/progress"))
        if result["scheduler_state"] in TERMINAL:
            attempts = client.call(endpoint + "/attempts")
            if attempts["current_attempt_id"] != submission["attempt_id"]:
                raise ClientError("Run now refers to a different Attempt; inspect the saved exact submission")
            state["result"] = result
            save(state_path, state)
            break
        if time.monotonic() >= deadline:
            raise ClientError("training wait timed out; resume observes the same job and never creates a new Attempt")
        time.sleep(5)
    if out is not None and "download" not in state:
        state["download"] = download(client, config["project"], config["run_id"], submission["attempt_id"], out)
        save(state_path, state)
    if result["scheduler_state"] != "SUCCEEDED":
        raise ClientError("training ended unsuccessfully; saved state and artifacts are available for diagnosis")
    return state
