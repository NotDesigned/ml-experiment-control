"""CPU download preparation is exact, bounded, and cannot replay a create."""
from copy import deepcopy
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.backends.sensecore_preparation import PREFIX, SenseCoreScriptPreparation
from ml_exp_server.workers.data_preparation import identity


@pytest.fixture
def script(tmp_path):
    copy = {"aec2": "debug-cluster-01e", "worker_spec": "N6lS.Iu.I10.2c4g", "gpus": 0,
            "cpus": 2, "memory_gb": 4, "queue_timeout_seconds": 5}
    config = tmp_path / "store.json"
    config.write_text(json.dumps({"data_delivery": copy}))
    definition = {"script": "download.py", "script_sha256": "a" * 64,
                  "source_id": "source." + "b" * 64, "image": "registry/image@sha256:" + "c" * 64,
                  "workdir": "/workspace", "env": {}, "inputs": [], "interpreter": "python3",
                  "arguments": [], "timeout_seconds": 5}
    definition["preparation_id"] = "preparation." + identity(definition)
    run = {"backend": {"kind": "sensecore", "workspace": "lab", "storage_mount": "volume/subdir:/data"},
           "storage": {"project_data_root": "/data/lab/demo"},
           "container": {"image": definition["image"]}, "data_preparation": definition}
    receipt = {"preparation_id": definition["preparation_id"], "content_sha256": "d" * 64,
               "dataset_id": "dataset." + "d" * 64, "bytes": 17}
    creates, stops = [], []
    rest = SimpleNamespace(
        specs=lambda _: [{"name": copy["worker_spec"], "device": {"number": 0},
                          "cpu": {"vcpu_allocatable": 2}, "memory": {"allocatable": 4}}],
        create=lambda *a, **kw: creates.append((a, kw)) or {"name": a[1]["name"], "uid": "e" * 32},
        find=lambda *a: [{"uid": "e" * 32}],
        describe=lambda *a: {"state": "SUCCEEDED", "uid": "e" * 32},
        logs=lambda *a: {"text": "noise\n" + PREFIX + json.dumps(receipt)},
        stop=lambda *a: stops.append(a),
    )
    service = SenseCoreScriptPreparation(config, tmp_path / "registry", rest=rest)
    return service, run, receipt, creates, stops


def test_success_cpu_profile_exact_receipt_and_reuse(script):
    service, run, receipt, creates, _ = script
    reference = service.reference(run)
    assert service.read("demo", reference["operation_id"]) == {}
    with pytest.raises(ApplicationError, match="not verified"):
        service.ensure("demo", run, require_cached=True)
    value = service.ensure("demo", run)
    assert value["status"] == "READY" and value["receipt"] == receipt and "definition" not in value
    assert len(creates) == 1
    profile, body = creates[0][0]
    assert (profile["gpus"], profile["cpus"], profile["memory_gb"], profile["quota_type"]) == (0, 2, 4, "reserved")
    assert body["resource_pool"]["name"].startswith("debug")
    assert body["roles"][0]["image_path"] == run["container"]["image"]
    assert body["fault_tolerance"]["backoff_limit"] == 0
    assert service.ensure("demo", run, require_cached=True) == value
    assert service.read("demo", reference["operation_id"]) == value
    assert service.update("demo", reference["operation_id"], status="READY") ["updated_at"] == value["updated_at"]


def test_command_executes_frozen_helper_and_only_reports_verified_summary(script, tmp_path, monkeypatch, capsys):
    service, run, receipt, _, _ = script
    args = shlex.split(service.command(service.definition(run)))
    assert args[:4] == ["timeout", "--signal=TERM", "--kill-after=30s", "65s"]
    called = []
    import ml_exp_server.workers.data_preparation as preparation
    monkeypatch.setitem(__import__("sys").modules, "data_preparation", SimpleNamespace(prepare=lambda d, p: called.append((d, p)) or (p, receipt)))
    monkeypatch.setitem(__import__("sys").modules, "data_input", SimpleNamespace(deliver=lambda *a: tmp_path))
    monkeypatch.setitem(__import__("sys").modules, "worker", SimpleNamespace(link_path=lambda *a: None))
    monkeypatch.setattr("sys.argv", ["-c", args[-1]])
    exec(args[-2], {})
    assert called[0][0] == run["data_preparation"]
    assert PREFIX in capsys.readouterr().out


@pytest.mark.parametrize("change", ["backend", "config", "pool", "cache"])
def test_invalid_scope_rejected_before_create(script, change):
    service, run, _, creates, _ = script
    if change == "backend":
        run["backend"]["kind"] = "slurm"
    elif change == "config":
        service.config.clear()
    elif change == "pool":
        service.config["data_delivery"]["aec2"] = "compute"
    else:
        run["storage"]["project_data_root"] = "/other/lab"
    with pytest.raises((ApplicationError, ValueError)):
        service.ensure("demo", run)
    assert not creates


@pytest.mark.parametrize("project, operation", [("../bad", "script." + "a" * 64), ("demo", "wrong")])
def test_state_identity_rejected(script, project, operation):
    with pytest.raises(ValueError):
        script[0].read(project, operation)


def test_rest_created_lazily(script, monkeypatch):
    service = script[0]
    service._rest = None
    sentinel = object()
    monkeypatch.setattr("ml_exp_server.backends.sensecore_preparation.SenseCoreREST.from_environment", lambda: sentinel)
    assert service.rest is sentinel and service.rest is sentinel


def test_unknown_create_only_observes_original_job_on_continue(script):
    service, run, _, creates, _ = script
    original = service.rest.create
    def uncertain(*a, **kw):
        original(*a, **kw)
        raise RuntimeError("private provider error")
    service.rest.create = uncertain
    with pytest.raises(ApplicationError) as error:
        service.ensure("demo", run)
    assert error.value.code == "DATA_SCRIPT_PROVIDER_ERROR" and "private" not in str(error.value)
    service.rest.create = lambda *a, **kw: pytest.fail("do not replay create")
    assert service.ensure("demo", run)["status"] == "READY" and len(creates) == 1


@pytest.mark.parametrize("jobs", [[], [{"uid": "a"}, {"uid": "b"}], [{}], [{"uid": 12}]])
def test_ambiguous_or_missing_job_never_replayed(script, jobs):
    service, run, _, creates, _ = script
    service.rest.find = lambda *a: jobs
    for _ in range(2):
        with pytest.raises(ApplicationError):
            service.ensure("demo", run)
    assert len(creates) == 1


def test_changed_job_uid_refused(script):
    service, run, _, _, _ = script
    service.rest.describe = lambda *a: {"uid": "other", "state": "SUCCEEDED"}
    with pytest.raises(ApplicationError):
        service.ensure("demo", run)
    service.rest.find = lambda *a: [{"uid": "replacement"}]
    with pytest.raises(ApplicationError):
        service.ensure("demo", run)


def test_terminal_failure_preserved_and_not_restarted(script):
    service, run, _, creates, _ = script
    service.rest.describe = lambda *a: {"uid": "e" * 32, "state": "FAILED"}
    for _ in range(2):
        with pytest.raises(ApplicationError) as error:
            service.ensure("demo", run)
        assert error.value.code == "DATA_SCRIPT_FAILED"
    assert len(creates) == 1


@pytest.mark.parametrize("state, stopped", [("RUNNING", True), ("SUCCEEDED", False)])
def test_deadline_stops_only_original_active_cpu(script, monkeypatch, state, stopped):
    service, run, _, _, stops = script
    service.rest.describe = lambda *a: {"uid": "e" * 32, "state": state}
    service.rest.logs = lambda *a: {"text": "not indexed yet"}
    times = iter([0, 1000])
    monkeypatch.setattr("ml_exp_server.backends.sensecore_preparation.time.time", lambda: next(times))
    with pytest.raises(ApplicationError) as error:
        service.ensure("demo", run)
    assert error.value.code == "DATA_SCRIPT_TIMEOUT" and bool(stops) is stopped


def test_delayed_offline_receipt_is_observed(script, monkeypatch):
    service, run, receipt, _, _ = script
    responses = iter([{"text": ""}, {"text": PREFIX + json.dumps(receipt)}])
    service.rest.logs = lambda *a: next(responses)
    waited = []
    monkeypatch.setattr("ml_exp_server.backends.sensecore_preparation.time.sleep", lambda n: waited.append(n))
    assert service.ensure("demo", run)["status"] == "READY" and waited == [2]


@pytest.mark.parametrize("key,value", [("preparation_id", "wrong"), ("content_sha256", "x"), ("dataset_id", "wrong"), ("bytes", True), ("bytes", -1)])
def test_invalid_receipt_not_published(script, key, value):
    service, run, receipt, _, _ = script
    receipt[key] = value
    with pytest.raises(ApplicationError):
        service.ensure("demo", run)
    assert service.read("demo", service.reference(run)["operation_id"])["status"] == "RECONCILE_REQUIRED"


def test_active_operation_is_not_submitted_twice(script):
    import fcntl
    service, run, _, creates, _ = script
    reference = service.reference(run)
    service.read("demo", reference["operation_id"])
    path = service.root / "demo" / (reference["operation_id"] + ".worker.lock")
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ApplicationError) as error:
            service.ensure("demo", run)
        assert error.value.code == "DATA_SCRIPT_ACTIVE"
    assert not creates


def test_ready_after_lock_acquisition_is_reused(script, monkeypatch):
    service, run, _, creates, _ = script
    operation = service.reference(run)["operation_id"]
    value = service.ensure("demo", run)
    original = service.state
    calls = []
    from contextlib import contextmanager
    @contextmanager
    def raced(*args):
        with original(*args) as (store, snapshot):
            calls.append(True)
            yield store, SimpleNamespace(value={}) if len(calls) == 1 else snapshot
    monkeypatch.setattr(service, "state", raced)
    assert service.ensure("demo", run) == value and len(creates) == 1


@pytest.mark.parametrize("field,value", [("aec2", "debug/../pool"), ("worker_spec", "one-gpu"),
    ("gpus", True), ("gpus", 1), ("cpus", True), ("cpus", 3), ("memory_gb", True), ("memory_gb", 8),
    ("queue_timeout_seconds", True), ("queue_timeout_seconds", 4), ("queue_timeout_seconds", 3601),
    ("queue_timeout_seconds", 5.0)])
def test_cpu_profile_is_strict_and_bounded_before_any_provider_call(script, field, value):
    service, run, _, creates, _ = script
    service.config["data_delivery"][field] = value
    service.rest.specs = lambda *a: pytest.fail("invalid frozen profile must not query the provider")
    with pytest.raises(ApplicationError) as error:
        service.ensure("demo", run)
    assert error.value.code == "DATA_SCRIPT_CPU_PROFILE" and not creates


@pytest.mark.parametrize("seconds", [True, 4, 3601, 5.0])
def test_download_script_timeout_is_bounded_before_provider(script, seconds):
    service, run, _, creates, _ = script
    run["data_preparation"]["timeout_seconds"] = seconds
    service.rest.specs = lambda *a: pytest.fail("invalid timeout must not query the provider")
    with pytest.raises(ApplicationError) as error:
        service.ensure("demo", run)
    assert error.value.code == "DATA_SCRIPT_CPU_PROFILE" and not creates


@pytest.mark.parametrize("seconds", [5, 3600])
def test_queue_and_script_timeout_boundaries_are_accepted(script, seconds):
    service, run, _, creates, _ = script
    service.config["data_delivery"]["queue_timeout_seconds"] = seconds
    run["data_preparation"]["timeout_seconds"] = seconds
    assert service.ensure("demo", run)["status"] == "READY"
    assert creates[0][0][0]["queue_timeout_seconds"] == seconds
    assert shlex.split(creates[0][0][1]["roles"][0]["startup_script"])[3] == str(seconds + 60) + "s"


@pytest.mark.parametrize("cache", ["data/lab", "/data/../outside"])
def test_relative_or_parent_traversing_cache_is_refused(script, cache):
    service, run, _, creates, _ = script
    run["storage"]["project_data_root"] = cache
    with pytest.raises(ValueError, match="outside the frozen NAS mount"):
        service.ensure("demo", run)
    assert not creates


@pytest.mark.parametrize("missing", [False, True])
def test_command_mounts_only_verified_cached_inputs_before_downloading(script, tmp_path, monkeypatch, capsys, missing):
    import sys
    service, run, receipt, _, _ = script
    run["data_preparation"]["inputs"] = [{"asset_id": "asset." + "1" * 64, "mount_path": "/inputs/tokens", "require_cached": False}]
    definition = service.definition(run)
    calls = []
    def deliver(item, token, cache):
        assert item["require_cached"] is True and token == ""
        assert cache == Path(definition["cache"]).parent / "data-assets"
        calls.append("verify-input")
        if missing:
            raise ValueError("DATA_CACHE_MISSING")
        return tmp_path / "verified-tree"
    def link_path(source, target):
        assert source == tmp_path / "verified-tree" and target == Path("/inputs/tokens")
        calls.append("mount-input")
    def prepare(download, cache):
        assert download == run["data_preparation"] and cache == Path(definition["cache"])
        calls.append("download-script")
        return cache, receipt
    for name, module in (("data_input", SimpleNamespace(deliver=deliver)),
                         ("worker", SimpleNamespace(link_path=link_path)),
                         ("data_preparation", SimpleNamespace(prepare=prepare))):
        monkeypatch.setitem(sys.modules, name, module)
    argv = shlex.split(service.command(definition))
    monkeypatch.setattr(sys, "argv", ["-c", argv[-1]])
    if missing:
        with pytest.raises(ValueError, match="DATA_CACHE_MISSING"):
            exec(argv[-2], {})
        assert calls == ["verify-input"] and PREFIX not in capsys.readouterr().out
    else:
        exec(argv[-2], {})
        assert calls == ["verify-input", "mount-input", "download-script"]
        result = json.loads(capsys.readouterr().out.split(PREFIX, 1)[1])
        assert result == receipt


@pytest.mark.parametrize("confirmation", [{"name": "other", "uid": "e" * 32},
    {"name": None, "uid": "e" * 32}, {"uid": None}, {"uid": ""}, {"uid": 17}])
def test_invalid_create_confirmation_is_inspected_without_replaying(script, confirmation):
    service, run, _, creates, _ = script
    def create(profile, body, **kwargs):
        creates.append((profile, body))
        return {"name": body["name"], **confirmation}
    service.rest.create = create
    with pytest.raises(ApplicationError) as error:
        service.ensure("demo", run)
    assert error.value.code == "DATA_SCRIPT_PROVIDER_ERROR"
    service.rest.create = lambda *a, **kw: pytest.fail("do not replay an uncertain create")
    assert service.ensure("demo", run)["status"] == "READY"
    assert len(creates) == 1


def test_confirmed_create_uid_is_durable_before_lookup_and_cannot_be_replaced(script):
    service, run, _, creates, stops = script
    operation = service.reference(run)["operation_id"]
    def find(*args):
        with service.state("demo", operation) as (_, snapshot):
            assert snapshot.value["job_uid"] == "e" * 32
        return [{"uid": "f" * 32}]
    service.rest.find = find
    service.rest.describe = lambda *a: pytest.fail("must not observe or stop a replacement UID")
    for _ in range(2):
        with pytest.raises(ApplicationError):
            service.ensure("demo", run)
    assert len(creates) == 1 and not stops


def test_frozen_cpu_profile_survives_config_change_and_unknown_create_without_replay(script, tmp_path):
    service, run, _, creates, _ = script
    run["data_preparation_cpu"] = deepcopy(service.config["data_delivery"])
    reference = service.reference(run)
    create = service.rest.create
    def uncertain(*args, **kwargs):
        create(*args, **kwargs)
        raise RuntimeError("lost response")
    service.rest.create = uncertain
    with pytest.raises(ApplicationError):
        service.ensure("demo", run)
    changed = tmp_path / "changed-store.json"
    changed.write_text(json.dumps({"data_delivery": {"aec2": "wrong-pool", "gpus": 8}}))
    resumed = SenseCoreScriptPreparation(changed, service.root.parent, rest=service.rest)
    assert resumed.reference(run) == reference
    resumed.rest.create = lambda *a, **kw: pytest.fail("frozen operation must not submit again")
    assert resumed.ensure("demo", run)["status"] == "READY" and len(creates) == 1
    assert resumed.definition(run)["cpu_profile"]["aec2"] == "debug-cluster-01e"


@pytest.mark.parametrize("field,value", [("operation_id", "script." + "f" * 64),
    ("preparation_id", "preparation." + "f" * 64), ("definition", {})])
def test_stored_operation_identity_cannot_be_changed(script, field, value):
    service, run, _, creates, _ = script
    operation = service.reference(run)["operation_id"]
    service.ensure("demo", run)
    service.update("demo", operation, **{field: value})
    with pytest.raises(ValueError, match="stored script preparation identity differs"):
        service.ensure("demo", run)
    assert len(creates) == 1


def test_cpu_spec_rejection_never_creates_a_job_or_exposes_provider_error(script):
    service, run, _, creates, _ = script
    service.rest.specs = lambda *a: []
    with pytest.raises(ApplicationError) as error:
        service.ensure("demo", run)
    assert error.value.code == "DATA_SCRIPT_PROVIDER_ERROR" and not creates
    assert service.read("demo", service.reference(run)["operation_id"])["status"] == "RECONCILE_REQUIRED"
