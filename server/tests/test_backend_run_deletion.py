"""Managed Run deletion uses only fake providers and test-owned directories."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import runpy

import pytest

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.backends import run_deletion as backend
from ml_exp_server.workers import delete_run as worker


def instruction(tmp_path, **extra):
    root = tmp_path / "demo"
    return {"project": "demo", "run_id": "trial", "project_root": str(root),
            "target": str(root / "runs/trial"), "plan_sha256": "a" * 64,
            "deletion_id": "action-test", **extra}


def tree(tmp_path):
    payload = instruction(tmp_path)
    root = Path(payload["target"])
    (root / "attempts/attempt-001/state/checkpoints/step-4").mkdir(parents=True)
    (root / "attempts/attempt-001/state/checkpoints/step-4/state.bin").write_bytes(b"state")
    (root / "attempts/attempt-001/outputs").mkdir()
    (root / "attempts/attempt-001/outputs/weights.bin").write_bytes(b"weights")
    (root / ".hidden").write_bytes(b"hidden")
    return payload, root


def test_helper_deletes_exact_run_sealed_and_hidden_preserves_shared(tmp_path):
    payload, root = tree(tmp_path)
    shared = root.parent.parent / "sources/source.bin"
    shared.parent.mkdir(); shared.write_bytes(b"shared")
    sibling = root.parent / "other"; sibling.mkdir()
    sealed = root / "attempts/attempt-001/state/checkpoints/step-4"
    sealed.chmod(0o555)
    result = worker.delete(payload, mounts=lambda: set())
    assert result["files_deleted"] == 3 and result["bytes_deleted"] == 18
    assert result["target_absent"] is True and not root.exists()
    assert shared.read_bytes() == b"shared" and sibling.is_dir()
    assert worker.delete(payload, mounts=lambda: set())["files_deleted"] == 0
    assert worker.delete(instruction(tmp_path / "missing"), mounts=lambda: set())["target_absent"] is True


@pytest.mark.parametrize("fields", [
    {"project": "../demo"}, {"run_id": "."}, {"target": "/tmp/demo/runs/trial/"},
    {"project_root": "/demo", "target": "/demo/runs/trial"}, {"project_root": "/tmp/other"},
    {"target": "/tmp/demo/runs/../trial"}, {"target": "relative"},
    {"target": "/tmp/demo\\bad/runs/trial"}, {"target": "/tmp/demo/runs/tri\x00al"},
    {"plan_sha256": "invalid"}, {"mode": "force"}, {"deletion_id": "../bad"},
])
def test_helper_rejects_unbound_paths_and_identities_before_effect(tmp_path, fields):
    payload, root = tree(tmp_path)
    with pytest.raises(ValueError): worker.delete({**payload, **fields}, mounts=lambda: set())
    assert (root / ".hidden").read_bytes() == b"hidden"


@pytest.mark.parametrize("unsafe", ["fifo", "mount", "root-link", "ancestor-link", "root-file"])
def test_helper_refuses_unsafe_tree_without_partial_deletion(tmp_path, unsafe):
    payload, root = tree(tmp_path)
    outside = tmp_path / "outside"; outside.mkdir(); (outside / "keep").write_text("keep")
    mounts = lambda: set()
    if unsafe == "fifo": os.mkfifo(root / "pipe")
    elif unsafe == "mount": mounts = lambda: {str(root / "attempts")}
    else:
        renamed = root.with_name("retained"); root.rename(renamed)
        if unsafe == "root-link": root.symlink_to(outside, target_is_directory=True)
        elif unsafe == "root-file": root.write_text("not a directory")
        else:
            runs = root.parent; moved = runs.with_name("retained-runs"); runs.rename(moved)
            runs.symlink_to(moved, target_is_directory=True)
    with pytest.raises((ValueError, OSError)): worker.delete(payload, mounts=mounts)
    assert (outside / "keep").read_text() == "keep"
    assert (root if unsafe in {"fifo", "mount"} else renamed).joinpath(".hidden").read_bytes() == b"hidden"


def test_helper_unlinks_internal_links_without_following_outside_or_cycles(tmp_path):
    payload, root = tree(tmp_path)
    outside = tmp_path / "outside"; outside.mkdir(); (outside / "keep").write_text("keep")
    (root / "outside").symlink_to(outside, target_is_directory=True)
    (root / "cycle").symlink_to(root, target_is_directory=True)
    (root / "dangling").symlink_to("missing")
    result = worker.delete(payload, mounts=lambda: set())
    assert result["files_deleted"] == 6 and result["bytes_deleted"] == 18
    assert (outside / "keep").read_text() == "keep" and not root.exists()
    with pytest.raises(ValueError): worker.remove_tree("/")


def test_helper_check_and_deadline_do_not_remove_present_data(tmp_path):
    payload, root = tree(tmp_path)
    with pytest.raises(ValueError, match="still present"): worker.delete({**payload, "mode": "check"})
    with pytest.raises(ValueError, match="budget"): worker.delete({**payload, "seconds": 901})
    times = iter([0, 2])
    with pytest.raises(TimeoutError): worker.delete({**payload, "seconds": 1}, clock=lambda: next(times), mounts=lambda: set())
    assert (root / ".hidden").exists()
    assert "/" in worker.mountpoints()


class Connection:
    statuses = []
    requests = []
    closed = 0

    def __init__(self, *args, **kwargs): pass
    def request(self, *args, **kwargs): self.requests.append((args, kwargs))
    def getresponse(self):
        status = self.statuses.pop(0)
        if isinstance(status, Exception): raise status
        return SimpleNamespace(status=status, read=lambda count: b"{}")
    def close(self): self.__class__.closed += 1


def callback_payload(tmp_path, **fields):
    return instruction(tmp_path, receipt_url="https://example.invalid/api/run-deletion-transfers/action-test/receipt",
                       capability="t" * 43, **fields)


def test_helper_durable_receipt_callback_retry_and_safe_binding(tmp_path):
    payload = callback_payload(tmp_path); receipt = worker.delete(payload)
    Connection.statuses = [OSError("offline"), 503, 200]; Connection.requests = []; Connection.closed = 0
    worker.report(payload, receipt, connection=Connection)
    assert Connection.closed == 3
    assert Connection.requests[-1][1]["headers"]["Authorization"] == "Bearer " + payload["capability"]
    assert json.loads(Connection.requests[-1][1]["body"]) == receipt
    Connection.statuses = [400, 400, 400]
    with pytest.raises(ValueError, match="reconciliation"): worker.report(payload, receipt, connection=Connection)
    for fields in ({"receipt_url": "http://bad/receipt"}, {"receipt_url": payload["receipt_url"] + "?secret=1"},
                   {"receipt_url": "https://user:pass@example.invalid" + "/api/run-deletion-transfers/action-test/receipt"},
                   {"receipt_url": payload["receipt_url"] + "#bad"}, {"capability": "short"}, {"capability": "t " * 32}):
        with pytest.raises(ValueError): worker.report({**payload, **fields}, receipt, connection=Connection)


def test_helper_main_stdout_callback_and_input_limits(tmp_path, monkeypatch, capsys):
    payload = instruction(tmp_path)
    assert worker.main(payload) == 0
    assert json.loads(capsys.readouterr().out)["target_absent"] is True
    seen = []; monkeypatch.setattr(worker, "report", lambda p, r: seen.append(r))
    assert worker.main(callback_payload(tmp_path)) == 0 and len(seen) == 1
    monkeypatch.setattr(worker.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(json.dumps(payload).encode())))
    assert worker.main() == 0
    monkeypatch.setattr(worker.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"x" * 65537)))
    with pytest.raises(ValueError, match="too large"): worker.main()
    monkeypatch.setattr(worker.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(json.dumps(payload).encode())))
    with pytest.raises(SystemExit) as result: runpy.run_path(worker.__file__, run_name="__main__")
    assert result.value.code == 0


@pytest.mark.parametrize("race", ["root-inode", "root-device", "child-device", "child-open-inode",
                                  "child-remove-inode", "file-inode", "final-root-inode", "recreated-root"])
def test_helper_rejects_filesystem_replacements_and_device_changes(tmp_path, monkeypatch, race):
    payload, root = tree(tmp_path)
    original_stat, original_fstat, original_rmdir = os.stat, os.fstat, os.rmdir
    seen = {}

    def changed(value, field):
        values = list(value); values[field] += 1
        return os.stat_result(values)

    def stat_call(name, *args, **kwargs):
        result = original_stat(name, *args, **kwargs)
        if kwargs.get("dir_fd") is not None:
            seen[("stat", name)] = seen.get(("stat", name), 0) + 1
            count = seen[("stat", name)]
            if race == "child-device" and name == "attempts": return changed(result, 2)
            if race == "child-remove-inode" and name == "attempts" and count == 3: return changed(result, 1)
            if race == "file-inode" and name == ".hidden" and count == 3: return changed(result, 1)
            if race == "final-root-inode" and name == "trial" and count == 2: return changed(result, 1)
        return result

    def fstat_call(fd):
        result = original_fstat(fd)
        path = os.readlink("/proc/self/fd/" + str(fd))
        seen[("fstat", path)] = seen.get(("fstat", path), 0) + 1
        count = seen[("fstat", path)]
        if race == "root-inode" and path == str(root) and count == 1: return changed(result, 1)
        if race == "root-device" and path == str(root) and count == 2: return changed(result, 2)
        if race == "child-open-inode" and path == str(root / "attempts"): return changed(result, 1)
        return result

    def rmdir_call(name, *args, **kwargs):
        original_rmdir(name, *args, **kwargs)
        if race == "recreated-root" and name == "trial": root.mkdir()

    with monkeypatch.context() as patch:
        patch.setattr(os, "stat", stat_call); patch.setattr(os, "fstat", fstat_call); patch.setattr(os, "rmdir", rmdir_call)
        with pytest.raises(ValueError): worker.delete(payload, mounts=lambda: set())
    assert root.is_dir()


class Rest:
    def __init__(self):
        self.created = []; self.stopped = []; self.state = "RUNNING"; self.uid = "original-uid"; self.found = []
        self.create_error = False; self.describe_error = False
        self.worker_rows = []

    def specs(self, profile):
        return [{"name": profile["worker_spec"], "device": {"number": 0},
                 "cpu": {"vcpu_allocatable": 2}, "memory": {"allocatable": 4}}]
    def find(self, profile, name): return self.found
    def create(self, profile, body, **kwargs):
        self.created.append(body)
        if self.create_error: raise OSError("secret provider error")
        return {"name": body["name"], "uid": self.uid}
    def describe(self, profile, name):
        if self.describe_error: raise OSError("provider unavailable")
        return {"name": name, "uid": self.uid, "state": self.state}
    def stop(self, profile, name): self.stopped.append(name); self.state = "STOPPED"
    def workers(self, profile, name): return self.worker_rows


def setup(tmp_path, *, kind="slurm", runner=None, rest=None, clock=None):
    payload = instruction(tmp_path)
    config = tmp_path / "objects.json"
    config.write_text(json.dumps({"data_delivery": {"aec2": "debug-cluster", "worker_spec": "cpu.2c4g",
                                                     "gpus": 0, "cpus": 2, "memory_gb": 4}}))
    runtime = SimpleNamespace(config=SimpleNamespace(container_execution=SimpleNamespace(artifact_store_file=config),
                                                     action_runtime=SimpleNamespace(allow_scheduler_mutations=True)))
    run = {"project": "demo", "run_id": "trial", "storage": {"project_data_root": payload["project_root"],
                                                                 "run_dir": payload["target"]},
           "backend": {"kind": kind, "ssh_alias": "owned-gateway", "workspace": "workspace",
                       "storage_mount": "afs:" + str(tmp_path), "image": "runtime@sha256:" + "a" * 64}}
    service = backend.BackendRunDeletion(runtime, runner=runner, rest=rest, clock=clock or (lambda: 100))
    targets = service.plan(run, [])
    state = {"deletion_id": "action-test", "plan_sha256": backend.digest(targets),
             "receipt_url": callback_payload(tmp_path)["receipt_url"], "capability": "t" * 43}
    return service, run, targets, state


def test_backend_plan_exact_roots_profiles_and_shared_exclusion(tmp_path):
    service, run, targets, state = setup(tmp_path)
    assert targets["target"] == run["storage"]["run_dir"]
    assert targets["estimated_registered_bytes"] is None and targets["provider_history"] == {"status": "UNSUPPORTED"}
    assert "capability" not in json.dumps(targets) and "receipt_url" not in targets
    del run["project"]; assert service.plan(run, [])["project"] == "demo"
    run["backend"]["ssh_alias"] = "bad; rm"
    with pytest.raises(ValueError): service.plan(run, [])
    run["backend"]["kind"] = "local"
    with pytest.raises(ApplicationError): service.plan(run, [])
    service, run, targets, state = setup(tmp_path, kind="sensecore")
    assert targets["cpu_profile"]["quota_type"] == "reserved"
    assert targets["cpu_profile"]["gpus"] == 0
    run["container"] = {"image": "pinned-runtime"}; assert service.plan(run, [])["image"] == "pinned-runtime"
    run["backend"]["storage_mount"] = "afs:/different"
    with pytest.raises(ValueError): service.plan(run, [])
    run["backend"]["storage_mount"] = "afs:" + str(tmp_path)
    run["container"] = {}; run["backend"].pop("image")
    with pytest.raises(ApplicationError): service.plan(run, [])
    Path(service.runtime.config.container_execution.artifact_store_file).write_text("{}")
    with pytest.raises(ApplicationError): service.plan(run, [])


@pytest.mark.parametrize("output", ["", "42|RUNNING|", "43|COMPLETED|", "42|", "42|UNKNOWN|", "invalid"])
def test_fresh_slurm_terminal_requires_exact_positive_accounting(tmp_path, output):
    service, run, _, _ = setup(tmp_path, runner=lambda *args, **kwargs: SimpleNamespace(stdout=output))
    with pytest.raises(ApplicationError): service.terminal(run, [{"attempt_id": "attempt-001", "backend_job_id": "42"}])


def test_fresh_terminal_all_attempts_uid_and_observation_failures(tmp_path):
    calls = []
    def runner(args, **kwargs): calls.append(args); return SimpleNamespace(stdout="42|COMPLETED|\n")
    service, run, _, _ = setup(tmp_path, runner=runner)
    attempts = [{"attempt_id": "attempt-001", "backend_job_id": "42"},
                {"attempt_id": "attempt-002", "state": "NOT_SUBMITTED", "has_submission": False}]
    result = service.terminal(run, attempts)
    assert result["attempts"][0]["states"] == ["COMPLETED"] and len(calls) == 1
    with pytest.raises(ApplicationError): service.terminal(run, [{"attempt_id": "attempt-001", "state": "UNKNOWN"}])
    with pytest.raises(ValueError): service.terminal(run, [{"attempt_id": "../escape"}])
    with pytest.raises(ValueError): service.terminal(run, [{"attempt_id": "attempt-001", "backend_job_id": "bad"}])
    service.runner = lambda *args, **kwargs: (_ for _ in ()).throw(OSError("secret"))
    with pytest.raises(ApplicationError, match="unavailable"): service.terminal(run, attempts)
    rest = Rest(); service, run, _, _ = setup(tmp_path, kind="sensecore", rest=rest)
    attempt = {"attempt_id": "attempt-001", "backend_job_id": "original", "backend_job_uid": rest.uid}
    with pytest.raises(ApplicationError): service.terminal(run, [attempt])


def test_original_worker_observation_must_remain_terminal_and_rest_lazy(tmp_path, monkeypatch):
    rest = Rest(); rest.state = "SUCCEEDED"
    service, run, _, _ = setup(tmp_path, kind="sensecore")
    monkeypatch.setattr(backend.SenseCoreREST, "from_environment", lambda: rest)
    assert service.rest is rest and service.rest is rest
    attempt = {"attempt_id": "attempt-001", "backend_job_id": "original", "backend_job_uid": rest.uid}
    def workers(profile, name): rest.state = "RUNNING"; return []
    rest.workers = workers
    with pytest.raises(ApplicationError, match="changed"): service.terminal(run, [attempt])
    rest.workers = lambda profile, name: rest.worker_rows
    rest.state = "SUCCEEDED"
    assert service.terminal(run, [attempt])["attempts"][0]["job_uid"] == rest.uid
    rest.state = "SUSPENDED"; rest.worker_rows = [{"phase": "Running"}]
    with pytest.raises(ApplicationError, match="workers"): service.terminal(run, [attempt])
    rest.worker_rows = []
    with pytest.raises(ApplicationError, match="workers"): service.terminal(run, [attempt])
    attempt["worker_deleted"] = True
    assert service.terminal(run, [attempt])["attempts"][0]["workers_confirmed_terminal"] is True
    rest.worker_rows = [{"phase": "Succeeded", "containers": [{"state": {"terminated": {"exit_code": 0}}}]}]
    assert service.terminal(run, [attempt])["attempts"][0]["workers_observed"] == 1
    assert backend.worker_terminal({"status": {"phase": "Failed"}})
    assert backend.worker_terminal({"workerDeleted": True})
    assert not backend.worker_terminal("unknown")
    rest.uid = "replacement"
    with pytest.raises(ValueError): service.terminal(run, [attempt])
    attempt.pop("backend_job_uid")
    with pytest.raises(ApplicationError, match="UID"): service.terminal(run, [attempt])
    run["backend"]["kind"] = "local"
    with pytest.raises(ApplicationError): service.terminal(run, [attempt])


def test_ssh_executes_fixed_helper_then_durable_receipt_no_replay(tmp_path):
    payload, root = tree(tmp_path); calls = []
    def runner(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout=json.dumps(worker.delete(json.loads(kwargs["input"]), mounts=lambda: set())))
    service, run, targets, state = setup(tmp_path, runner=runner)
    persisted = []
    result = service.execute(targets, state, persisted.append)
    assert result["status"] == "DELETED" and not root.exists() and len(calls) == 1
    assert persisted[0]["stage"] == "SSH_EXECUTING" and persisted[-1]["receipt"] == result["receipt"]
    assert service.execute(targets, state, persisted.append) == result and len(calls) == 1
    state["stage"] = "SSH_EXECUTING"
    assert service.execute(targets, state, persisted.append) == result and len(calls) == 1
    assert calls[0][1]["timeout"] == 630 and "python3" in calls[0][0][-1]


def test_ssh_uncertain_response_observes_absence_only(tmp_path):
    payload, root = tree(tmp_path)
    def lost(args, **kwargs):
        worker.delete(json.loads(kwargs["input"]), mounts=lambda: set())
        raise OSError("reply lost")
    service, run, targets, state = setup(tmp_path, runner=lost)
    with pytest.raises(ApplicationError): service.execute(targets, state, lambda item: None)
    assert state["stage"] == "RECONCILE_REQUIRED" and not root.exists()
    modes = []
    def check(args, **kwargs):
        payload = json.loads(kwargs["input"]); modes.append(payload["mode"])
        return SimpleNamespace(stdout=json.dumps(worker.delete(payload, mounts=lambda: set())))
    service.runner = check
    assert service.execute(targets, state, lambda item: None)["status"] == "DELETED" and modes == ["check"]
    state.pop("receipt"); root.mkdir(); (root / "new").write_text("retained")
    with pytest.raises(ApplicationError): service.execute(targets, state, lambda item: None)
    assert (root / "new").read_text() == "retained"


def test_execute_receipt_binding_digest_oversize_and_unsupported(tmp_path):
    service, _, targets, state = setup(tmp_path, runner=lambda *a, **k: SimpleNamespace(stdout="x" * 65537))
    with pytest.raises(ApplicationError): service.execute(targets, state, lambda item: None)
    with pytest.raises(ValueError): service.payload(targets, {**state, "plan_sha256": "b" * 64})
    payload = service.payload(targets, state)
    receipt = worker.delete(payload)
    for fields in ({"target": "/different"}, {"target_absent": 1}, {"bytes_deleted": True}, {"files_deleted": -1}, {"extra": 1}):
        with pytest.raises(ValueError): service.receipt(payload, {**receipt, **fields})
    with pytest.raises(ValueError): service.receipt(payload, None)
    changed = {**targets, "backend": "local"}; state["plan_sha256"] = backend.digest(changed)
    with pytest.raises(ApplicationError): service.execute(changed, state, lambda item: None)


@pytest.fixture
def simple_document(monkeypatch):
    monkeypatch.setattr(backend, "document", lambda profile, name, image, command: {"name": name, "image": image, "command": command})


def test_cpu_deletion_delivers_frozen_mirror_before_creating_only_the_cpu_job(tmp_path,simple_document,monkeypatch):
    from tests.test_image_mirror import DELIVERY,SOURCE,TARGET,TAG
    rest=Rest();service,run,_,state=setup(tmp_path,kind='sensecore',rest=rest)
    run['container']={'image':SOURCE};run['backend'].update(image=TAG,image_delivery=DELIVERY)
    targets=service.plan(run,[]);state['plan_sha256']=backend.digest(targets);calls=[]
    monkeypatch.setattr('ml_exp_server.builds.image_mirror.ensure_delivery',lambda d:calls.append(d))
    assert service.execute(targets,state,lambda v:None)['status']=='PENDING'
    assert targets['image']==TARGET and calls==[DELIVERY] and rest.created[0]['image']==TARGET


def test_cpu_single_create_callback_required_and_confirmed_uid_terminal(tmp_path, simple_document):
    rest = Rest(); service, _, targets, state = setup(tmp_path, kind="sensecore", rest=rest)
    persisted = []
    assert service.execute(targets, state, persisted.append)["status"] == "PENDING"
    assert len(rest.created) == 1 and persisted[0]["stage"] == "CPU_SUBMITTING"
    rest.state = "SUCCEEDED"
    assert service.execute(targets, state, persisted.append)["status"] == "PENDING"  # stdout success cannot release state
    state["receipt"] = worker.delete(service.payload(targets, state))
    assert service.execute(targets, state, persisted.append)["status"] == "DELETED"
    assert len(rest.created) == 1 and state["stage"] == "DELETED"
    rest.describe_error = True
    assert service.execute(targets, state, persisted.append)["status"] == "DELETED"
    # Embedded fixed helper receives only the one-time callback capability.
    import base64, gzip, shlex
    encoded = shlex.split(rest.created[0]["command"])[-1].split("b64decode(", 1)[1].split(")", 1)[0]
    import ast
    script = gzip.decompress(base64.b64decode(ast.literal_eval(encoded))).decode()
    assert "def checked_target" in script and state["capability"] in script


def test_cpu_create_unknown_replacement_and_disabled_never_recreate(tmp_path, simple_document):
    rest = Rest(); rest.create_error = True
    service, _, targets, state = setup(tmp_path, kind="sensecore", rest=rest)
    for _ in range(2):
        with pytest.raises(ApplicationError, match="unconfirmed"): service.execute(targets, state, lambda item: None)
    assert len(rest.created) == 1 and "job_uid" not in state
    service.runtime.config.action_runtime.allow_scheduler_mutations = False
    with pytest.raises(ApplicationError, match="disabled"): service.execute(targets, state, lambda item: None)
    service.runtime.config.action_runtime.allow_scheduler_mutations = True
    state.clear(); state.update(deletion_id="action-test", plan_sha256=backend.digest(targets), receipt_url="url", capability="t" * 43)
    rest.create_error = False; rest.found = [{"name": "collision"}]
    with pytest.raises(ApplicationError, match="already exists"): service.execute(targets, state, lambda item: None)
    assert len(rest.created) == 1
    rest.found = []; assert service.execute(targets, state, lambda item: None)["status"] == "PENDING"
    rest.uid = "replacement"
    with pytest.raises(ApplicationError, match="identity"): service.execute(targets, state, lambda item: None)
    assert len(rest.created) == 2


def test_cpu_deadline_stops_only_original_uid_and_waits_for_receipt(tmp_path, simple_document):
    now = [100]; rest = Rest()
    service, _, targets, state = setup(tmp_path, kind="sensecore", rest=rest, clock=lambda: now[0])
    assert service.execute(targets, state, lambda item: None)["status"] == "PENDING"
    now[0] = 1001
    with pytest.raises(ApplicationError, match="deadline"): service.execute(targets, state, lambda item: None)
    assert rest.stopped == [state["scheduler_name"]]
    with pytest.raises(ApplicationError): service.execute(targets, state, lambda item: None)
    assert len(rest.stopped) == 1 and len(rest.created) == 1
    state["receipt"] = worker.delete(service.payload(targets, state))
    assert service.execute(targets, state, lambda item: None)["status"] == "DELETED"
