"""Validate one Dockerfile experiment before making API requests."""
from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath
import re

from .api import ClientError


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
               "arguments", "env", "resources", "outputs", "required_outputs", "inputs", "checkpoint_upload", "max_gpu_hours", "metrics_schema", "evaluation", "data_preparation", "checkpoint_persistence", "resume_from", "executor_selector", "requirements", "wandb", "parameters"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ClientError("experiment config contains unknown fields")
    wandb = value.get("wandb")
    if wandb is not None and (not isinstance(wandb, dict) or set(wandb) - {"enabled", "preparation_enabled", "entity", "project"} or
            type(wandb.get("enabled", True)) is not bool or type(wandb.get("preparation_enabled", False)) is not bool or any(item is not None and
                (not isinstance(item, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", item))
                for key, item in wandb.items() if key in {"entity", "project"})):
        raise ClientError("wandb needs boolean enabled/preparation_enabled and optional entity/project names")
    parameters = value.get("parameters")
    try:
        valid_parameters = parameters is None or isinstance(parameters, dict) and len(json.dumps(parameters, allow_nan=False).encode()) <= 16384
    except ValueError:
        valid_parameters = False
    if not valid_parameters:
        raise ClientError("parameters must be a finite JSON object of at most 16 KiB")
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
            not isinstance(selector.get("backend"), (str, type(None))) or selector.get("backend") not in {None, "slurm", "sensecore", "pai-dlc"} or
            not isinstance(selector.get("candidates", []), list) or len(selector.get("candidates", [])) > 64 or
            any(not isinstance(i, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", i) for i in selector.get("candidates", [])) or
            not (selector.get("backend") or selector.get("candidates"))):
        raise ClientError("executor_selector needs a backend or valid candidate IDs")
    requirements = value.get("requirements", {})
    flags = {"data_assets", "data_preparation", "data_preparation_before_job", "persistent_checkpoints", "queue_reason", "exit_code", "preemption_reason", "scheduler_walltime", "offline_output_export"}
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
    if not isinstance(resources, dict) or set(resources) - {"gpus", "cpus", "memory_gb", "max_time", "allocation_policy", "priority"}:
        raise ClientError("invalid resources")
    if resources.get("allocation_policy") is not None and resources["allocation_policy"] not in ("guaranteed", "allow_idle", "idle_only"):
        raise ClientError("allocation_policy must be guaranteed, allow_idle or idle_only")
    if resources.get("priority") is not None and (type(resources["priority"]) is not int or not 1 <= resources["priority"] <= 9):
        raise ClientError("priority must be an integer from 1 to 9")
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
