"""Static server adapter contract and shared CPU identity safety."""
from types import SimpleNamespace

import pytest
import yaml

from ml_exp_server.backends.operations import backend_operations, SlurmOperations, SenseCoreOperations
from ml_exp_server.backends.sensecore_cpu import (
    checked_job, describe_job, find_job, stop_job, validate_profile,
)
from tests.test_artifact_store import storage
from tests.test_result_collection import recovery, IDENTITY, output


def profile():
    return {"aec2": "debug-pool", "worker_spec": "cpu.2c4g", "gpus": 0,
            "cpus": 2, "memory_gb": 4, "quota_type": "reserved", "queue_timeout_seconds": 5}


@pytest.mark.parametrize("key,value", [("gpus", True), ("worker_spec", "gpu.8c"), ("aec2", "bad/pool"),
                                       ("quota_type", "other"), ("queue_timeout_seconds", True),
                                       ("queue_timeout_seconds", 3601)])
def test_cpu_profiles_reject_ambiguous_resources_and_unbounded_times(key, value):
    with pytest.raises(ValueError):
        validate_profile({**profile(), key: value}, timeouts=("queue_timeout_seconds",), debug=True)


def test_shared_validation_retains_valid_historical_cpu_pool_and_spot():
    old = {**profile(), "aec2": "cpu-pool", "quota_type": "spot"}
    original = dict(old)
    validate_profile(old, timeouts=("queue_timeout_seconds",))
    assert old == original
    with pytest.raises(ValueError):
        validate_profile(old, debug=True)


@pytest.mark.parametrize("job", [None, {}, {"name": "other", "uid": "one"},
                                 {"name": "job", "uid": ""}, {"name": "job", "uid": 123},
                                 {"name": "job", "uid": "replacement"}])
def test_confirmed_uid_cannot_be_rebound(job):
    with pytest.raises(ValueError):
        checked_job(job, "job", "original")


def test_legacy_unbound_job_adopts_only_one_exact_name_and_then_refuses_replacement():
    rest = SimpleNamespace(find=lambda *_: [{"name": "job", "uid": "original"}])
    original = find_job(rest, profile(), "job")
    assert original["uid"] == "original"
    rest.find = lambda *_: [{"name": "job", "uid": "replacement"}]
    with pytest.raises(ValueError):
        find_job(rest, profile(), "job", original["uid"])
    rest.find = lambda *_: []
    assert find_job(rest, profile(), "job", original["uid"]) is None
    rest.find = lambda *_: [original, original]
    with pytest.raises(ValueError, match="ambiguous"):
        find_job(rest, profile(), "job")


def test_stop_rechecks_original_uid_before_name_based_provider_request():
    stopped = []
    rest = SimpleNamespace(describe=lambda *_: {"name": "job", "uid": "replacement"},
                           stop=lambda *args: stopped.append(args))
    with pytest.raises(ValueError):
        stop_job(rest, profile(), "job", "original")
    assert not stopped
    rest.describe = lambda *_: {"name": "job", "uid": "original"}
    assert describe_job(rest, profile(), "job", "original")["uid"] == "original"
    stop_job(rest, profile(), "job", "original")
    assert len(stopped) == 1


def test_server_boundary_is_static_and_rejects_uninstalled_kinds():
    assert isinstance(backend_operations("slurm"), SlurmOperations)
    assert isinstance(backend_operations("sensecore"), SenseCoreOperations)
    with pytest.raises(ValueError):
        backend_operations("missing")
    gateway = SlurmOperations()
    assert gateway.prepare_script(None, None, None, None, None) is None
    marker = {"operation_id": "gateway"}
    assert gateway.script_progress(None, None, marker) == marker


def test_third_fake_adapter_can_collect_without_provider_branch_or_client_changes(recovery):
    """Tests the server boundary only, not a production third scheduler."""
    service, root, _, _, objects, definition, row = recovery
    definition["backend"]["kind"] = "fake"
    (root / "attempts/attempt-001/attempt.yaml").write_text(yaml.safe_dump(definition))
    output(root)
    calls = []
    class Fake:
        def confirm_terminal(self, owner, backend, name):
            calls.append(("terminal", backend["kind"], name))
        def collect(self, owner, identity, root, frozen, transfer, operation, local):
            assert identity == IDENTITY and frozen["backend"]["kind"] == "fake"
            assert operation["attempts"] == 1
            calls.append(("collect", identity))
            owner.publish_local(identity, local, transfer)
            return True
    service.backends = {"fake": Fake()}
    _, pending = service.begin(*IDENTITY)
    service.finish(pending)
    assert service.read(*IDENTITY)["status"] == "AVAILABLE"
    assert calls == [("terminal", "fake", "24830"), ("collect", IDENTITY)]
    assert row.attempts[0].state == "FAILED" and len(objects) == 1


def test_inside_job_script_mode_keeps_legacy_worker_preparation():
    assert SenseCoreOperations().prepare_script(None, None, None, None,
        {'data_preparation_execution': {'phase': 'inside_job'}}) is None
