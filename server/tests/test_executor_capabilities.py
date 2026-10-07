"""Declarations describe adapters; matching never queries quotas or free cards."""
import copy
import json

import pytest

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.container_execution import ContainerExecutionService, RunRequest
from ml_exp_server.experiment_preparation import PreparationRun
from ml_exp_server.executor_capabilities import (
    ExecutionRequirements, ExecutorSelector, declaration, match, mismatches, validate_worker,
)
from tests.test_container_api import client, runtime


def request(**fields):
    return PreparationRun(run_id="trial", **fields)


def test_public_capabilities_are_private_readonly_and_not_inventory(client):
    values = client.get("/api/executors").json()["executors"]
    gpu, cloud = (next(v for v in values if v["id"] == n) for n in ("gpu", "cloud"))
    assert gpu["capabilities"]["runtime"]["materialization"] == "oci_to_sif"
    assert cloud["capabilities"]["resources"]["allocation_mode"] == "fixed_spec"
    for v in values:
        cap = v["capabilities"]
        assert cap["contract"] == "backend-capabilities.v1"
        assert not cap["resources"]["inventory_query"]
        assert not cap["outputs"]["offline_export"]
        assert not cap["storage"]["checkpoint_registration"]
        assert cap["storage"]["storage_id"].startswith("storage.")
    assert gpu["capabilities"]["observability"]["exit_code"]
    assert not cloud["capabilities"]["observability"]["exit_code"]
    assert "ssh_alias" not in json.dumps(values)


def test_local_and_unmounted_storage_are_not_declared_container_capable():
    local = declaration({"backend": {"kind": "local"}}, artifact_store=False, desktop=False)
    assert local["runtime"]["platforms"] == [] and local["storage"]["storage_id"] is None
    cloud = declaration({"backend": {"kind": "sensecore"}, "storage_root": "/workspace"}, artifact_store=True, desktop=True)
    assert not cloud["storage"]["persistent"] and cloud["data"]["download_script"] is None
    assert cloud["data"]["asset_preparation"] == "before_job"


@pytest.mark.parametrize("fields, requirements, expected", [
    ({}, {"platform": "linux/arm64"}, "runtime.platforms"),
    ({}, {"queue_reason": True}, "observability.queue_reason"),
    ({}, {"exit_code": True}, "observability.exit_code"),
    ({}, {"preemption_reason": True}, "observability.preemption_reason"),
    ({}, {"scheduler_walltime": True}, "jobs.scheduler_walltime"),
    ({}, {"offline_output_export": True}, "outputs.offline_export"),
    ({"inputs": [{"asset_id": "asset." + "a" * 64, "mount_path": "/inputs/data"}]}, {}, "data.assets"),
    ({}, {"data_assets": True}, "data.assets"),
    ({"data_preparation": {"script": "download.py"}}, {}, "data.download_script"),
    ({}, {"data_preparation": True}, "data.download_script"),
    ({"checkpoint_persistence": {}}, {}, "storage.checkpoint_registration"),
    ({}, {"persistent_checkpoints": True}, "storage.checkpoint_registration"),
    ({"checkpoint_upload": {}}, {}, "checkpoint_upload"),
    ({"resources": {"gpus": 2}}, {}, "resources.fixed_gpu_count"),
    ({"resources": {"cpus": 9}}, {}, "resources.cpus"),
    ({"resources": {"memory_gb": 33}}, {}, "resources.memory_gb"),
])
def test_mismatches_are_explicit_without_building(client, fields, requirements, expected):
    service = ContainerExecutionService(client.app.state.runtime)
    profile = service.profiles()["cloud"]
    profile["capacity"] = {"gpus": 1, "cpus": 8, "memory_gb": 32}
    cap = declaration(profile, artifact_store=False, desktop=False)
    assert expected in mismatches(cap, profile, request(**fields), ExecutionRequirements(**requirements))
    assert mismatches(cap, profile, request(), ExecutionRequirements()) == []


def test_native_process_does_not_satisfy_oci():
    profile = {"backend": {"kind": "local"}}
    cap = declaration(profile, artifact_store=False, desktop=False)
    assert "runtime.materialization" in mismatches(cap, profile, request(), ExecutionRequirements())


def test_matching_preserves_preferences_and_uses_configured_smallest_shape(client):
    service = ContainerExecutionService(client.app.state.runtime)
    profiles = service.profiles()
    profiles["large"] = copy.deepcopy(profiles["cloud"])
    profiles["cloud"]["capacity"] = {"gpus": 1, "cpus": 16, "memory_gb": 64}
    profiles["large"]["capacity"] = {"gpus": 1, "cpus": 32, "memory_gb": 128}
    caps = service.profile_capabilities(profiles)
    selected = match(profiles, caps, request(), ExecutionRequirements(), ExecutorSelector(backend="sensecore"))
    assert selected["executor"] == "cloud" and not selected["live_availability_checked"]
    preferred = match(profiles, caps, request(), ExecutionRequirements(), ExecutorSelector(candidates=["missing", "gpu", "large", "cloud"], backend="sensecore"))
    assert preferred["executor"] == "large"
    assert preferred["rejected"] == {"missing": ["unknown_executor"], "gpu": ["backend"]}
    selected["profile"]["backend"]["workspace"] = "different"
    assert profiles["cloud"]["backend"]["workspace"] != "different"


@pytest.mark.parametrize("run, selector, code", [
    ({}, None, "EXECUTOR_SELECTION_INVALID"),
    ({}, {}, "EXECUTOR_SELECTION_INVALID"),
    ({"executor": "gpu"}, {"backend": "slurm"}, "EXECUTOR_SELECTION_INVALID"),
    ({"executor": "missing"}, None, "NO_MATCHING_EXECUTOR"),
    ({"executor": "cloud", "resources": {"gpus": 2}}, None, "NO_MATCHING_EXECUTOR"),
])
def test_matching_rejects_ambiguous_or_impossible_selection(client, run, selector, code):
    service = ContainerExecutionService(client.app.state.runtime)
    with pytest.raises(ApplicationError) as exc:
        match(service.profiles(), service.profile_capabilities(), request(**run), ExecutionRequirements(),
              ExecutorSelector(**selector) if selector is not None else None)
    assert exc.value.code == code


def test_resume_matching_requires_same_project_storage(client):
    from ml_exp_server.checkpoint_registry import storage_scope
    service = ContainerExecutionService(client.app.state.runtime)
    profiles = service.profiles()
    scope = storage_scope(profiles["gpu"]["backend"], profiles["gpu"]["storage_root"] + "/demo")
    selected = match(profiles, service.profile_capabilities(), request(), ExecutionRequirements(),
                     ExecutorSelector(candidates=["cloud", "gpu"]), resume_scope=scope, project="demo")
    assert selected["executor"] == "gpu" and selected["rejected"]["cloud"] == ["checkpoint.storage_scope"]


@pytest.mark.parametrize("fields, requirements, expected", [
    ({"inputs": [{"asset_id": "asset." + "a" * 64, "mount_path": "/inputs/data"}]}, {}, "data-cache-required.v1"),
    ({"checkpoint_upload": {}}, {}, "data-assets.v1"),
    ({"data_preparation": {"script": "download.py"}}, {}, "data-preparation.v1"),
    ({"checkpoint_persistence": {}}, {}, "persistent-checkpoints.v1"),
    ({}, {"data_assets": True}, "data-assets.v1"),
    ({}, {"data_preparation": True}, "data-preparation.v1"),
    ({}, {"persistent_checkpoints": True}, "persistent-checkpoints.v1"),
])
def test_runtime_worker_support_is_checked_separately(client, fields, requirements, expected):
    profile = ContainerExecutionService(client.app.state.runtime).profiles()["cloud"]
    cap = declaration(profile, artifact_store=True, desktop=True)
    with pytest.raises(ApplicationError, match=expected):
        validate_worker(request(**fields), {"capabilities": []}, cap, ExecutionRequirements(**requirements), cache_required=expected == "data-cache-required.v1")
    validate_worker(request(**fields), {"capabilities": ["data-cache-required.v1", "data-assets.v1", "data-preparation.v1", "persistent-checkpoints.v1"]}, cap, ExecutionRequirements(**requirements), cache_required=expected == "data-cache-required.v1")


def test_lowlevel_create_cannot_bypass_capacity_check(client):
    bundle = runtime(client)
    service = ContainerExecutionService(client.app.state.runtime)
    profile = service.profiles()["gpu"]
    profile["capacity"] = {"gpus": 1}
    with pytest.raises(ApplicationError, match="resources.gpus"):
        service.create_run("demo", RunRequest(run_id="big", runtime_id=bundle["runtime_id"], executor="gpu", resources={"gpus": 2}), profile_snapshot=profile)
