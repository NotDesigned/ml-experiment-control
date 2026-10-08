"""Configured executor contracts and deterministic matching, without live probes."""
from __future__ import annotations

import copy

from pydantic import BaseModel, ConfigDict, Field

from experiment_control.backends.capabilities import DECLARATIONS

from ..application_errors import ApplicationError
from ..results.checkpoint_registry import storage_scope
from ..workers.worker_http import CONTRACT as RELAY_CONTRACT, relay_environment


CONTRACT = "backend-capabilities.v1"


class ExecutionRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    platform: str = Field(default="linux/amd64", max_length=128)
    data_assets: bool = False
    data_preparation: bool = False
    persistent_checkpoints: bool = False
    queue_reason: bool = False
    exit_code: bool = False
    preemption_reason: bool = False
    scheduler_walltime: bool = False
    offline_output_export: bool = False


class ExecutorSelector(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: str | None = Field(default=None, pattern=r"^(slurm|sensecore)$")
    candidates: list[str] = Field(default_factory=list, max_length=64)


def declaration(profile, *, artifact_store: bool, desktop: bool) -> dict:
    kind = profile["backend"]["kind"]
    adapter = DECLARATIONS[kind]
    persistent = bool(adapter.persistent_filesystem and profile.get("storage_root")
                      and (kind != "sensecore" or profile["backend"].get("storage_mount")))
    storage_id = storage_scope(profile["backend"], profile["storage_root"]) if persistent else None
    relay = relay_environment(profile["backend"].get("api_relay"))
    if relay and kind != "slurm":
        raise ApplicationError("API TCP relay requires a Slurm gateway", code="EXECUTOR_TRANSPORT_INVALID")
    return {
        "contract": CONTRACT,
        "runtime": {"platforms": ["linux/amd64"] if kind != "local" else [],
                    "materialization": adapter.runtime_materialization},
        "resources": {"allocation_mode": adapter.allocation_mode,
                      "inventory_query": adapter.resource_inventory_query},
        "network": {"compute_internet": profile.get("compute_internet"),
                    "api_transport": "tcp_relay" if relay else "direct_https"},
        "storage": {"persistent": persistent, "storage_id": storage_id,
                    "checkpoint_registration": persistent and artifact_store},
        "data": {"asset_preparation": "before_job" if desktop and kind in {"slurm", "sensecore"} else "inside_job",
                 "assets": persistent and artifact_store,
                 "download_script": ("before_job" if relay else "inside_job") if persistent and artifact_store else None},
        "observability": {"logs": list(adapter.logs), "queue_reason": adapter.queue_reason,
                          "exit_code": adapter.exit_code, "preemption_reason": adapter.preemption_reason},
        "jobs": {"exact_submission_lookup": adapter.exact_submission_lookup,
                 "walltime_enforcement": list(adapter.walltime_enforcement)},
        "outputs": {"transfer": bool(artifact_store or profile.get("artifact_ssh") or kind == "slurm"),
                    "offline_export": adapter.offline_output_export},
    }


def mismatches(capabilities, profile, request, requirements) -> list[str]:
    problems = []
    if requirements.platform not in capabilities["runtime"]["platforms"]:
        problems.append("runtime.platforms")
    if capabilities["runtime"]["materialization"] not in {"oci", "oci_to_sif"}:
        problems.append("runtime.materialization")
    for name in ("queue_reason", "exit_code", "preemption_reason"):
        if getattr(requirements, name) and not capabilities["observability"][name]:
            problems.append("observability." + name)
    if requirements.scheduler_walltime and "scheduler" not in capabilities["jobs"]["walltime_enforcement"]:
        problems.append("jobs.scheduler_walltime")
    if requirements.offline_output_export and not capabilities["outputs"]["offline_export"]:
        problems.append("outputs.offline_export")
    if (request.inputs or requirements.data_assets) and not capabilities["data"]["assets"]:
        problems.append("data.assets")
    if (request.data_preparation is not None or requirements.data_preparation) and capabilities["data"]["download_script"] is None:
        problems.append("data.download_script")
    if (request.checkpoint_persistence is not None or request.resume_from is not None or requirements.persistent_checkpoints) and not capabilities["storage"]["checkpoint_registration"]:
        problems.append("storage.checkpoint_registration")
    if request.checkpoint_upload is not None and not capabilities["data"]["assets"]:
        problems.append("checkpoint_upload")
    capacity = profile.get("capacity") or {}
    for key in ("gpus", "cpus", "memory_gb"):
        if key in capacity and getattr(request.resources, key) > capacity[key]:
            problems.append("resources." + key)
    if capabilities["resources"]["allocation_mode"] == "fixed_spec" and request.resources.gpus != profile.get("gpus", 1):
        problems.append("resources.fixed_gpu_count")
    return problems


def match(profiles, declarations, request, requirements, selector, *, resume_scope=None, project=None):
    if request.executor is not None and selector is not None:
        raise ApplicationError("choose an executor or an executor_selector, not both", code="EXECUTOR_SELECTION_INVALID")
    if request.executor is None and (selector is None or not (selector.backend or selector.candidates)):
        raise ApplicationError("automatic matching needs an explicit backend or candidate list", code="EXECUTOR_SELECTION_INVALID")
    candidates = [request.executor] if request.executor else (selector.candidates or sorted(profiles))
    rejected = {}
    accepted = []
    for name in candidates:
        profile = profiles.get(name)
        if profile is None:
            rejected[name] = ["unknown_executor"]
            continue
        if selector is not None and selector.backend is not None and profile["backend"]["kind"] != selector.backend:
            rejected[name] = ["backend"]
            continue
        reasons = mismatches(declarations[name], profile, request, requirements)
        if resume_scope is not None and storage_scope(profile["backend"], profile["storage_root"].rstrip("/") + "/" + project) != resume_scope:
            reasons.append("checkpoint.storage_scope")
        if reasons:
            rejected[name] = reasons
        else:
            accepted.append(name)
    if not accepted:
        raise ApplicationError("no executor matches the request: " + str(rejected), code="NO_MATCHING_EXECUTOR")
    # Explicit candidate order is preference. Otherwise prefer the smallest
    # fitting fixed allocation, then stable ID; this is never a free-card claim.
    if request.executor is None and not selector.candidates:
        accepted.sort(key=lambda name: (*[profiles[name].get("capacity", {}).get(k, getattr(request.resources, k))
                                         for k in ("gpus", "cpus", "memory_gb")], name))
    name = accepted[0]
    return {"executor": name, "capabilities": copy.deepcopy(declarations[name]),
            "profile": copy.deepcopy(profiles[name]), "compatible_executors": accepted,
            "rejected": rejected, "basis": "configured_capabilities_and_capacity",
            "live_availability_checked": False}


def validate_worker(request, bundle, capability, requirements=None, *, cache_required=False):
    required = set()
    if capability.get("network", {}).get("api_transport") == "tcp_relay":
        required.add(RELAY_CONTRACT)
    if request.inputs or request.checkpoint_upload is not None:
        required.add("data-assets.v1")
    if cache_required:
        required.add("data-cache-required.v1")
    if request.data_preparation is not None:
        required.add("data-preparation.v1")
    if request.checkpoint_persistence is not None or request.resume_from is not None:
        required.add("persistent-checkpoints.v1")
    if requirements is not None:
        for flag, feature in (("data_assets", "data-assets.v1"), ("data_preparation", "data-preparation.v1"),
                              ("persistent_checkpoints", "persistent-checkpoints.v1")):
            if getattr(requirements, flag):
                required.add(feature)
    missing = sorted(required - set(bundle.get("capabilities", [])))
    if missing:
        raise ApplicationError("Runtime worker lacks " + ", ".join(missing) + "; prepare a capable Dockerfile Runtime", code="RUNTIME_CAPABILITY_MISMATCH")
