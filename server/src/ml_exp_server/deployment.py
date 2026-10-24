"""Read-only deployment checks, separate from protocol support and job preflight."""
from pathlib import Path
from http.client import HTTPException
import json

import yaml

from .builds.builder_client import builder_request


STORAGE_CAPABILITIES = {
    "target-backed-data-upload.v1", "fixed-transfer-buffer.v1", "ccr-streamed-data-image.v1",
    "artifact-retention.v1", "oss-datasets.v1", "result-collection.v1", "training-result.v1",
    "experiment-records.v1", "input-delivery-progress.v1", "desktop-data-upload.v1",
    "ccr-data-delivery.v1", "nas-data-ready-reuse.v1", "managed-worker.v1",
    "data-preparation.v1", "persistent-checkpoints.v1", "multipart-upload.v1",
    "direct-object-download.v1", "data-assets.v1", "checkpoint-upload.v1",
}


def configuration_file(path, loader, valid):
    if not path:
        return "NOT_CONFIGURED", None
    try:
        value = loader(Path(path).read_text())
        if not valid(value):
            return "INVALID", None
        return "CONFIGURED", value
    except (OSError, ValueError, yaml.YAMLError):
        # Paths, provider credentials and parser excerpts are private.
        return "INVALID", None


def deployment_status(config, *, probe=False):
    execution, policy = config.container_execution, config.action_runtime
    profiles_status, profiles = configuration_file(execution.profiles_file, yaml.safe_load,
        lambda v: isinstance(v, dict) and isinstance(v.get("executors"), dict)
        and all(isinstance(p, dict) and isinstance(p.get("backend"), dict)
                and p["backend"].get("kind") in {"local", "slurm", "sensecore", "pai-dlc"}
                for p in v["executors"].values()))
    if profiles_status == "CONFIGURED" and not profiles["executors"]:
        profiles_status = "NOT_CONFIGURED"
    storage_status, storage = configuration_file(execution.artifact_store_file, json.loads,
        lambda v: isinstance(v, dict) and all(isinstance(v.get(k), str) and v[k].strip()
            for k in ("endpoint", "bucket", "access_key", "secret_key", "public_transfer_base")))
    components = {
        "executors": {"status": profiles_status, "setting": "container_execution.profiles_file"},
        "image_builder": {"status": "CONFIGURED" if execution.builder_socket else "NOT_CONFIGURED",
                          "setting": "container_execution.builder_socket"},
        "artifact_store": {"status": storage_status, "setting": "container_execution.artifact_store_file"},
    }
    for key in ("project_writes", "source_imports", "scheduler_mutations"):
        components[key] = {"status": "CONFIGURED" if getattr(policy, "allow_" + key) else "DISABLED",
                           "setting": "action_runtime.allow_" + key}
    if probe and execution.builder_socket:
        try:
            result = builder_request(execution.builder_socket, {"operation": "health"}, timeout=2)
            components["image_builder"]["status"] = (
                "AVAILABLE" if isinstance(result, dict) and result.get("status") == "ok"
                and result.get("dockerfile_builds") is True else "UNAVAILABLE")
        except (OSError, ValueError, HTTPException):
            components["image_builder"]["status"] = "UNAVAILABLE"
    issues = [name for name, value in components.items() if value["status"] not in {"CONFIGURED", "AVAILABLE"}]
    capabilities = {"storage": storage_status == "CONFIGURED",
                    "oss_datasets": storage_status == "CONFIGURED" and bool(storage.get("oss_datasets")),
                    "image_builder": bool(execution.builder_socket)}
    return {"api": "READY", "training": "CONFIGURED" if not issues else "INCOMPLETE",
            "scope": "configuration_and_builder" if probe else "configuration",
            "backend_admission_checked": False, "components": components, "issues": issues}, capabilities


def configured_capabilities(supported, available):
    disabled = set() if available["storage"] else set(STORAGE_CAPABILITIES)
    if not available["oss_datasets"]:
        disabled.add("oss-datasets.v1")
    if not available["image_builder"]:
        disabled.add("dockerfile-build.v1")
    return tuple(name for name in supported if name not in disabled)
