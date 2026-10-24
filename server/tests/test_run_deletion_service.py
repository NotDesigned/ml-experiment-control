"""Whole-Run deletion exercises real isolated stores with a fake backend only."""
from copy import deepcopy
import errno
import io
import json
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest
import yaml

from ml_exp_server.actions.errors import ActionError
from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.backends.run_deletion import BackendRunDeletion, digest as backend_digest
from ml_exp_server.deletion_state import RunFence
from ml_exp_server.projects.project_config import load_research_project
from ml_exp_server.records.store import RecordStore
from ml_exp_server.results.artifact_store import ArtifactStore
from ml_exp_server.runs.deletion import RunDeletionService, digest, mapping, replace_yaml, unlink_file
from ml_exp_server.runs.deletion_guards import action_guard, request_run
from ml_exp_server.runtime import ExperimentServerRuntime
from ml_exp_server.schemas import OperationScope, OperationScopeType, RunIndexRow, ServerConfig
from ml_exp_server.storage import atomic_json
from ml_exp_worker.delete_run import remove_tree
from tests.test_artifact_store import archive


class Backend:
    """Never contacts SSH or a provider; it only removes this test's temp tree."""
    payload = staticmethod(BackendRunDeletion.payload)
    receipt = staticmethod(BackendRunDeletion.receipt)

    def __init__(self, env):
        self.env = env
        self.calls = []
        self.fail = None
        self.pending = False
        self.callback = False

    def plan(self, run, attempts):
        plan = BackendRunDeletion(self.env.runtime).plan(run, attempts)
        if self.callback:
            plan["backend"] = "sensecore"
        return plan

    def terminal(self, run, attempts):
        self.calls.append(("terminal", deepcopy(attempts)))
        for attempt in attempts:
            if attempt["state"] not in {"SUCCEEDED", "FAILED", "CANCELLED", "NOT_SUBMITTED"}:
                raise ApplicationError("job not positively terminal", code="RUN_DELETION_JOB_NOT_TERMINAL")
        return {"attempts": deepcopy(attempts)}

    def execute(self, targets, state, persist):
        self.calls.append(("execute", deepcopy(state)))
        assert self.env.service.fence.is_deleted(targets["project"], targets["run_id"])
        if self.fail:
            raise self.fail
        if self.pending:
            self.pending = False
            persist({**state, "stage": "WAITING"})
            return {"status": "PENDING"}
        if not state.get("receipt"):
            removed = remove_tree(targets["target"])
            payload = self.payload(targets, state)
            receipt = {key: payload[key] for key in ("project", "run_id", "target", "deletion_id", "plan_sha256")}
            receipt.update(contract="run-deletion.v1", status="DELETED", **removed)
            if self.callback:
                self.env.service.receive(state["deletion_id"], state["capability"], receipt)
                # A stale backend update must not lose the independently saved receipt.
                persist({**state, "stage": "DELETED"})
            else:
                persist({**state, "stage": "DELETED", "receipt": receipt})
        return {"status": "DELETED", "receipt": state.get("receipt") or receipt}


@pytest.fixture
def deletion_env(tmp_path, monkeypatch):
    root = tmp_path / "registry"
    config_file = tmp_path / "objects.json"
    atomic_json(config_file, {"endpoint": "http://localhost:3900", "bucket": "isolated-results",
        "access_key": "fixture", "secret_key": "fixture", "local_storage_headroom_bytes": 0,
        "public_transfer_base": "https://example.test/api/artifact-transfers"})
    config = ServerConfig(index_db=str(tmp_path / "index.sqlite"), action_root=str(tmp_path / "actions"),
        project_registry_root=str(root), run_root=str(tmp_path / "local-runs"), collector_enabled=False,
        telemetry={"enabled": False}, action_runtime={"allow_resource_deletions": True},
        container_execution={"artifact_store_file": str(config_file)})
    managed = root / "managed" / "demo"
    project_file = managed / "experiments" / "research_project.yaml"
    project_file.parent.mkdir(parents=True)
    project = {"schema_version": 1, "project": "demo", "title": "Isolated deletion fixture", "run_roots": [],
        "controller": {"python": "python3", "experimentctl": "tools/experimentctl.py", "workdir": str(managed),
                       "capabilities": {"container_execution": True}}, "campaigns": []}
    project_file.write_text(yaml.safe_dump(project))
    env = SimpleNamespace(root=root, config=config, project_file=project_file, remote={}, deleted=[], tokens={},
                          local={}, backend_roots={}, definitions={}, sleeps=[])

    class S3:
        def upload_fileobj(self, stream, bucket, key, **kwargs):
            env.remote[(bucket, key)] = stream.read()

        def delete_object(self, *, Bucket, Key):
            assert env.service.fence.is_deleted(*Key.split("/")[:2])
            env.deleted.append(Key)
            env.remote.pop((Bucket, Key), None)

    monkeypatch.setattr(ArtifactStore, "client", lambda self: S3())
    monkeypatch.setattr(ArtifactStore, "_transfer_config", staticmethod(lambda: None))
    monkeypatch.setattr("ml_exp_server.backends.run_deletion.run_process", lambda *a, **k: pytest.fail("live backend transport forbidden"))

    def add_run(run="run-a", *, resume_from=None, state="SUCCEEDED", submitted=True):
        local = config.project_run_root_path("demo") / ("run-" + run) / run
        attempt = local / "attempts" / "attempt-001"
        attempt.mkdir(parents=True)
        atomic_json(attempt / "status.json", {"state": state})
        if submitted:
            atomic_json(attempt / "backend.json", {"backend_job_id": "123", "job_uid": "original-uid"})
        (attempt / "outputs").mkdir()
        (attempt / "outputs" / "weights.bin").write_bytes(b"local-original")
        (local / "manifest.yaml").write_text(yaml.safe_dump({"project": "demo", "run_id": run}))
        (attempt / "attempt.yaml").write_text(yaml.safe_dump({"project": "demo", "run_id": run, "attempt_id": "attempt-001"}))
        backend_root = tmp_path / "backend" / "demo" / "runs" / run
        (backend_root / "state" / "checkpoints" / "step-4").mkdir(parents=True)
        (backend_root / "state" / "checkpoints" / "step-4" / "weights.bin").write_bytes(b"persistent weights")
        frozen = {"project": "demo", "run_id": run, "backend": {"kind": "slurm", "ssh_alias": "isolated"},
            "storage": {"project_data_root": str(backend_root.parent.parent), "run_dir": str(backend_root)}}
        if resume_from is not None:
            frozen["resume_from"] = resume_from
        campaign = {"schema_version": 1, "project": "demo", "campaign": "run-" + run, "runs": [frozen]}
        definition = managed / "experiments" / "campaigns" / ("run-" + run + ".yaml")
        definition.parent.mkdir(exist_ok=True)
        definition.write_text(yaml.safe_dump(campaign))
        value = yaml.safe_load(project_file.read_text())
        value["campaigns"].append({"name": "run-" + run, "file": str(definition)})
        project_file.write_text(yaml.safe_dump(value))
        env.local[run], env.backend_roots[run], env.definitions[run] = local, backend_root, definition
        return frozen

    env.add_run = add_run
    add_run()
    with ExperimentServerRuntime.create(config, projects=[load_research_project(project_file)]) as runtime:
        def install(active):
            env.runtime = active
            env.backend = Backend(env)
            env.service = RunDeletionService(active, backend=env.backend, sleep=lambda seconds: env.sleeps.append(seconds))
            active.action_service.deletion_service = env.service
        env.install = install
        install(runtime)
        env.records = RecordStore(root)
        env.records.bind("demo", "run-a", {"enabled": False}, {})
        scope = env.records.scope("demo", "run-a", "attempt-001")
        env.records.append(scope, [("original", {"kind": "metrics", "observations": [{
            "name": "loss", "value": 1.25, "unit": "nats", "status": "VALID", "step": 4}]})])
        runtime.index.upsert_run(RunIndexRow(project="demo", run_id="run-a", run_dir=str(env.local["run-a"])))
        yield env


def prepare(env, *, key="review-key", run="run-a", reason="delete completed isolated experiment"):
    return env.service.prepare("demo", run, reason, key)


def execute(env, plan):
    service = env.runtime.action_service
    service.authorize(plan["action_id"], "reviewed exact isolated targets")
    return service.execute(plan["action_id"], "EXECUTE " + plan["action_id"])


def publish(env, *, repair=True, keep=False):
    objects = env.runtime.storage_service.objects
    _, token, _ = objects.issue("demo", "run-a", "attempt-001", env.local["run-a"], ["**/*"])
    body = archive({"weights.bin": b"original"})
    first = objects.receive("demo", "run-a", "attempt-001", token, io.BytesIO(body), len(body))
    if repair:
        body = archive({"weights.bin": b"original", "extra.bin": b"recovered"})
        objects.receive("demo", "run-a", "attempt-001", token, io.BytesIO(body), len(body), repair_sha256=first["sha256"])
    with objects.record("demo", "run-a", "attempt-001") as (path, value):
        if keep:
            value["retention"] = {"status": "KEEP", "sha256": value["receipt"]["sha256"]}
            value["download_acknowledgement"] = {"original": "must remain byte-identical"}
            atomic_json(path, value)
        original = path.read_bytes()
    env.tokens["run-a"] = token
    return path, original


def test_preparation_is_reviewable_idempotent_and_does_not_delete(deletion_env):
    env = deletion_env
    receipt, original = publish(env)
    plan = prepare(env)
    assert plan["operation"] == "DELETE_RUN_RESOURCES" and plan["ready"]
    assert plan["scope"] == {"project": "demo", "scope_type": "run", "object_id": "run-a"}
    assert plan["intent_id"] == "review-key" and plan["target_digest"] == digest(plan["target_manifest"])
    assert len(plan["target_manifest"]["objects"]) == 2
    assert plan["target_manifest"]["retained"]["metric_definitions"] is True
    assert prepare(env) == plan
    with pytest.raises(ActionError, match="different operation intent"):
        prepare(env, reason="changed reason")
    assert not env.deleted and not env.backend.calls and env.backend_roots["run-a"].exists()
    assert env.local["run-a"].exists() and receipt.read_bytes() == original
    assert not env.service.fence.is_deleted("demo", "run-a")
    with pytest.raises(ActionError, match="separate execution authorization"):
        env.runtime.action_service.execute(plan["action_id"], "EXECUTE " + plan["action_id"])


@pytest.mark.parametrize("keep", [False, True])
def test_whole_run_purge_keeps_audit_receipts_shared_assets_and_real_suffix_run(deletion_env, keep):
    env = deletion_env
    receipt, original = publish(env, keep=keep)
    env.add_run("run-a--preparation")
    env.runtime._replace_active_project(load_research_project(env.project_file))
    env.records.bind("demo", "run-a--preparation", {"enabled": False}, {})
    real_scope = env.records.scope("demo", "run-a--preparation", "attempt-001")
    env.records.append(real_scope, [("real", {"kind": "event", "message": "independent real Run"})])
    env.records.bind("demo", "run-a--backfill", {"enabled": False}, {
        "ml_expd": {"run_id": "run-a", "record_type": "backfill"}})
    registry = env.root / "persistent-checkpoints" / "demo" / "run-a" / "attempt-001"
    registry.mkdir(parents=True)
    atomic_json(registry / "checkpoint.fixture.json", {"retained": "backend registration"})
    shared = env.root / "assets" / "demo" / "checkpoint-upload.asset"
    shared.parent.mkdir(parents=True)
    shared.write_bytes(b"shared uploaded checkpoint snapshot")
    plan = prepare(env)
    result = execute(env, plan)
    assert result["execution"]["status"] == "VERIFIED"
    completion = result["execution"]["result"]
    assert {key: completion[key] for key in ("status", "project", "run_id", "target_digest")} == {
        "status": "DELETED", "project": "demo", "run_id": "run-a", "target_digest": plan["target_digest"]}
    assert env.service.fence.is_deleted("demo", "run-a") and env.service.fence.is_deleted("demo", "run-a--backfill")
    assert not env.service.fence.is_deleted("demo", "run-a--preparation")
    assert env.records.binding("demo", "run-a") is None and env.records.binding("demo", "run-a--backfill") is None
    assert env.records.events(real_scope)[0]["payload"]["message"] == "independent real Run"
    assert not env.local["run-a"].exists() and not env.backend_roots["run-a"].exists()
    assert not registry.exists() and not env.definitions["run-a"].exists()
    assert env.local["run-a--preparation"].exists() and env.backend_roots["run-a--preparation"].exists()
    assert shared.read_bytes() == b"shared uploaded checkpoint snapshot"
    assert receipt.read_bytes() == original
    assert len(env.deleted) == 2 and not env.remote
    assert completion["retained"]["object_receipts"] and completion["retained"]["provider_history"] == "UNSUPPORTED"
    assert completion["retained"]["metric_definitions"] is True
    assert env.runtime.action_store.snapshot(plan["action_id"])["target_manifest"] == plan["target_manifest"]
    with pytest.raises(ApplicationError) as late:
        env.runtime.storage_service.objects.authorize("demo", "run-a", "attempt-001", env.tokens["run-a"])
    assert late.value.status_code == 410
    with pytest.raises(ApplicationError): env.records.scope("demo", "run-a", "attempt-001")
    before = deepcopy(env.backend.calls)
    assert env.runtime.action_service.execute(plan["action_id"], "ignored after verified") == result
    assert env.runtime.action_service.reconcile(plan["action_id"]) == result and env.backend.calls == before


def test_disabled_policy_blocks_prepare_and_runtime_dispatch(deletion_env):
    env = deletion_env
    env.config.action_runtime.allow_resource_deletions = False
    plan = prepare(env)
    assert not plan["ready"] and plan["execution"]["status"] == "BLOCKED"
    assert next(item for item in plan["gates"] if item["name"] == "deletion_policy")["status"] == "FAIL"
    with pytest.raises(ActionError): env.runtime.action_service.authorize(plan["action_id"], "no override")
    with pytest.raises(ApplicationError) as caught: env.service.execute(plan)
    assert caught.value.code == "RUN_DELETION_DISABLED" and not env.backend.calls
    env.config.action_runtime.allow_resource_deletions = True
    ready = prepare(env, key="enabled-plan")
    env.runtime.action_service.authorize(ready["action_id"], "reviewed")
    env.config.action_runtime.allow_resource_deletions = False
    with pytest.raises(ActionError, match="disabled"):
        env.runtime.action_service.execute(ready["action_id"], "EXECUTE " + ready["action_id"])


@pytest.mark.parametrize("state", ["RUNNING", "UNKNOWN"])
def test_unresolved_backend_never_crosses_fence_or_deletes(deletion_env, state):
    env = deletion_env
    atomic_json(env.local["run-a"] / "attempts/attempt-001/status.json", {"state": state})
    plan = prepare(env)
    assert not plan["ready"] and plan["execution"]["status"] == "BLOCKED"
    assert any(gate["status"] == "FAIL" for gate in plan["gates"])
    with pytest.raises(ActionError): execute(env, plan)
    assert not env.service.fence.is_deleted("demo", "run-a") and env.local["run-a"].exists()
    assert not env.deleted and not env.backend.calls


@pytest.mark.parametrize("source", ["campaign", "preparation"])
def test_checkpoint_reference_blocks_deletion_until_consumer_is_deleted(deletion_env, source):
    env = deletion_env
    reference = {"run_id": "run-a", "attempt_id": "attempt-001", "checkpoint_id": "checkpoint.original"}
    if source == "campaign":
        env.add_run("consumer", resume_from=reference)
    else:
        atomic_json(env.root / "experiment-preparations/demo/consumer.json", {
            "run_id": "consumer", "request": {"run": {"resume_from": reference}}, "status": "PREPARED"})
    plan = prepare(env)
    assert not plan["ready"] and plan["target_manifest"]["dependencies"][0]["run_id"] == "consumer"
    env.service.fence.mark("demo", "consumer")
    assert prepare(env, key="consumer-deleted")["ready"]
    assert not env.deleted


@pytest.mark.parametrize("source", ["action", "preparation", "collection", "upload"])
def test_active_work_is_reviewed_as_a_blocking_gate(deletion_env, source):
    env = deletion_env
    if source == "action":
        other = prepare(env, key="already-authorized")
        env.runtime.action_service.authorize(other["action_id"], "already in use")
    elif source == "preparation":
        atomic_json(env.root / "experiment-preparations/demo/active.json", {
            "run_id": "run-a", "status": "EXECUTING", "preparation_id": "preparation-active"})
    elif source == "collection":
        atomic_json(env.root / "result-collections/demo/run-a/attempt-001.json", {"status": "COLLECTING"})
    else:
        atomic_json(env.root / "multipart-uploads/upload-active/upload.json", {"status": "UPLOADING", "upload_id": "upload-active",
            "binding": {"project": "demo", "run_id": "run-a"}})
    plan = prepare(env, key="blocked-review")
    assert not plan["ready"] and plan["execution"]["status"] == "BLOCKED"
    gate = next(item for item in plan["gates"] if item["name"] == "no_active_or_uncertain_operations")
    assert json.loads(gate["detail"])[0]["kind"] == source
    assert not env.deleted


@pytest.mark.parametrize("change", ["definition", "dependency", "busy", "new-attempt"])
def test_review_drift_is_rejected_before_any_external_effect(deletion_env, change):
    env = deletion_env
    plan = prepare(env)
    if change == "definition":
        campaign = mapping(env.definitions["run-a"])
        campaign["runs"][0]["arguments"] = ["changed"]
        env.definitions["run-a"].write_text(yaml.safe_dump(campaign))
    elif change == "dependency":
        env.add_run("consumer", resume_from={"run_id": "run-a"})
    elif change == "busy":
        atomic_json(env.root / "result-collections/demo/run-a/attempt-001.json", {"status": "WAITING"})
    else:
        (env.local["run-a"] / "attempts" / "attempt-002").mkdir()
    result = execute(env, plan)
    assert result["execution"]["status"] == "RECONCILE_REQUIRED"
    assert not env.service.fence.is_deleted("demo", "run-a")
    assert not env.deleted and not any(call[0] == "execute" for call in env.backend.calls)


def test_crash_after_intent_before_fence_reconciles_same_targets_without_recreation(deletion_env, monkeypatch):
    env = deletion_env
    publish(env)
    plan = prepare(env)
    original = env.service.fence.mark
    monkeypatch.setattr(env.service.fence, "mark", lambda *a, **k: (_ for _ in ()).throw(OSError("injected fence crash")))
    uncertain = execute(env, plan)
    assert uncertain["execution"]["status"] == "RECONCILE_REQUIRED" and not env.service.fence.is_deleted("demo", "run-a")
    monkeypatch.setattr(env.service.fence, "mark", original)
    recovered = env.runtime.action_service.reconcile(plan["action_id"])
    assert recovered["execution"]["status"] == "VERIFIED"
    assert len([call for call in env.backend.calls if call[0] == "terminal"]) == 2
    assert len(env.deleted) == 2 and len(env.runtime.action_store.list_all()) == 1


def test_backend_failure_fences_writers_and_reconcile_resumes_original_action(deletion_env):
    env = deletion_env
    plan = prepare(env)
    env.backend.fail = OSError("private backend failure must not leak")
    failed = execute(env, plan)
    assert failed["execution"]["status"] == "RECONCILE_REQUIRED"
    assert failed["execution"]["error"] == "RUN_DELETION_RECONCILE_REQUIRED"
    assert failed["execution"]["safe_to_retry"] is False
    assert env.service.fence.is_deleted("demo", "run-a")
    with pytest.raises(ApplicationError): prepare(env, key="never-create-another-action")
    env.config.action_runtime.allow_resource_deletions = False
    with pytest.raises(ActionError, match="disabled"): env.runtime.action_service.reconcile(plan["action_id"])
    env.config.action_runtime.allow_resource_deletions = True
    env.backend.fail = None
    recovered = env.runtime.action_service.reconcile(plan["action_id"])
    assert recovered["execution"]["status"] == "VERIFIED" and len(env.runtime.action_store.list_all()) == 1


def test_pending_backend_sleep_and_callback_receipt_survive_stale_updates(deletion_env):
    env = deletion_env
    env.backend.pending = True
    env.backend.callback = True
    plan = prepare(env)
    result = execute(env, plan)
    assert result["execution"]["status"] == "VERIFIED" and env.sleeps == [2]
    with env.service.state(plan["action_id"]) as (_, snapshot):
        state = snapshot.value
    assert state["backend"]["receipt"]["target_absent"] is True
    receipt = state["backend"]["receipt"]
    assert env.service.receive(plan["action_id"], state["backend"]["capability"], receipt) == {"accepted": True}
    changed = {**receipt, "bytes_deleted": receipt["bytes_deleted"] + 1}
    with pytest.raises(ApplicationError) as caught:
        env.service.receive(plan["action_id"], state["backend"]["capability"], changed)
    assert caught.value.code == "DELETION_RECEIPT_CONFLICT"
    for token in (None, "", "wrong"):
        with pytest.raises(ApplicationError) as caught: env.service.receive(plan["action_id"], token, receipt)
        assert caught.value.status_code == 403
    assert env.service.execute(plan)["status"] == "DELETED"


def test_exact_identity_and_damaged_local_evidence_fail_closed(deletion_env, tmp_path):
    env = deletion_env
    with pytest.raises(ApplicationError) as caught: prepare(env, run="missing")
    assert caught.value.code == "UNKNOWN_RUN"
    with pytest.raises(ApplicationError) as caught: prepare(env, reason=" ")
    assert caught.value.status_code == 422
    with pytest.raises(ValueError): env.service.manifest("demo", "../outside")
    configured = env.runtime.project("demo")
    original = configured.base_dir
    configured.base_dir = tmp_path / "unmanaged"
    with pytest.raises(ApplicationError) as caught: prepare(env)
    assert caught.value.code == "RUN_DELETION_UNSUPPORTED"
    configured.base_dir = original
    bad = env.local["run-a"] / "attempts" / "attempt-invalid"
    bad.mkdir()
    with pytest.raises((ValueError, ApplicationError)): prepare(env)
    bad.rmdir()
    with pytest.raises(ValueError):
        with env.service.state("invalid"):
            pytest.fail("invalid deletion journal identity")


def test_attempt_inventory_preserves_exact_uid_and_positive_worker_delete_evidence(deletion_env):
    env = deletion_env
    attempt = env.local["run-a"] / "attempts/attempt-001"
    atomic_json(attempt / "backend.json", {"backend_job_id": "123"})
    objects = env.runtime.storage_service.objects
    objects.issue("demo", "run-a", "attempt-001", env.local["run-a"], ["**/*"])
    with objects.record("demo", "run-a", "attempt-001") as (path, transfer):
        transfer["training_diagnostics"] = {"scheduler_name": "123", "job_uid": "uid-exact", "events": [
            {"reason": "SuccessfulDelete", "message": "pod pt-uid-exact-worker-0 was deleted"}]}
        atomic_json(path, transfer)
    value = env.service._attempts(env.local["run-a"])[0]
    assert value["backend_job_uid"] == "uid-exact" and value["worker_deleted"]
    (attempt / "backend.json").unlink()
    (attempt / "status.json").unlink()
    with pytest.raises(ApplicationError) as caught:
        env.service._attempts(env.local["run-a"])
    assert caught.value.code == "RUN_DELETION_ATTEMPT_EVIDENCE_INCOMPLETE"
    with objects.record("demo", "run-a", "attempt-001") as (path, transfer):
        transfer.pop("training_diagnostics")
        atomic_json(path, transfer)
    assert env.service._attempts(env.local["run-a"])[0]["state"] == "NOT_SUBMITTED"
    atomic_json(attempt / "submission.json", {"status": "UNCONFIRMED"})
    assert env.service._attempts(env.local["run-a"])[0]["state"] == "UNKNOWN"


def test_mapping_rejects_links_and_nonmapping_evidence(tmp_path):
    assert mapping(tmp_path / "missing.json") == {}
    data = tmp_path / "evidence.json"
    atomic_json(data, {"exact": True})
    assert mapping(data) == {"exact": True}
    link = tmp_path / "linked.json"
    link.symlink_to(data)
    with pytest.raises((ValueError, OSError)): mapping(link)
    data.write_text("[]")
    with pytest.raises(ValueError, match="mapping"): mapping(data)


def test_creation_and_action_guards_refuse_tombstoned_exact_run(deletion_env):
    env = deletion_env
    calls = []
    class Creator:
        runtime = env.runtime
        @request_run
        def create(self, project, request):
            calls.append((project, request))
            return "created"
    direct = SimpleNamespace(run_id="run-a")
    nested = SimpleNamespace(run=direct)
    assert Creator().create("demo", direct) == "created"
    assert Creator().create("demo", nested) == "created"
    env.service.fence.mark("demo", "run-a")
    with pytest.raises(ApplicationError): Creator().create("demo", nested)
    for scope in ("run", "attempt"):
        object_id = "run-a::attempt-001" if scope == "attempt" else "run-a"
        with pytest.raises(ApplicationError):
            with action_guard(env.runtime.action_service, {"operation": "CANCEL_RUN", "scope": {
                "project": "demo", "scope_type": scope, "object_id": object_id}}):
                pytest.fail("a new Action crossed the permanent fence")
    for service, snapshot in [(SimpleNamespace(), {}), (env.runtime.action_service, {"scope": {"scope_type": "project"}}),
                               (env.runtime.action_service, {"operation": "DELETE_RUN_RESOURCES"})]:
        with action_guard(service, snapshot):
            pass
    assert len(calls) == 2


def test_idle_operations_are_not_busy_and_matching_preparation_is_in_manifest(deletion_env):
    env = deletion_env
    idle = env.root / "experiment-preparations/demo" / ("preparation-" + "a" * 32 + ".json")
    atomic_json(idle, {"run_id": "run-a", "status": "READY"})
    atomic_json(env.root / "experiment-preparations/demo/unrelated.json", {"run_id": "other", "status": "EXECUTING"})
    atomic_json(env.root / "result-collections/demo/run-a/attempt-001.json", {"status": "AVAILABLE"})
    atomic_json(env.root / "multipart-uploads/upload-finished/upload.json", {
        "binding": {"project": "demo", "run_id": "run-a"}, "status": "COMPLETED"})
    atomic_json(env.root / "multipart-uploads/upload-other/upload.json", {
        "binding": {"project": "other", "run_id": "run-a"}, "status": "UPLOADING"})
    assert not env.service._busy("demo", "run-a")
    plan = prepare(env)
    assert str(idle) in plan["target_manifest"]["local_paths"]
    result = execute(env, plan)
    assert result["execution"]["status"] == "VERIFIED" and not idle.exists()
    assert (env.root / "experiment-preparations/demo/unrelated.json").exists()


def test_modified_plan_and_durable_target_are_rejected(deletion_env):
    env = deletion_env
    plan = prepare(env)
    changed = deepcopy(plan)
    changed["target_manifest"]["objects"].append({"unreviewed": True})
    with pytest.raises(ValueError, match="manifest changed"):
        env.service.execute(changed)
    env.service.update(plan["action_id"], status="DELETING", target_digest="sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="durable deletion target differs"):
        env.service.execute(plan)
    assert not env.deleted and not env.backend.calls


def test_recovery_before_fence_revalidates_changed_resources(deletion_env, monkeypatch):
    env = deletion_env
    plan = prepare(env)
    original = env.service.fence.mark
    monkeypatch.setattr(env.service.fence, "mark", lambda *a, **k: (_ for _ in ()).throw(OSError("crash before fence")))
    assert execute(env, plan)["execution"]["status"] == "RECONCILE_REQUIRED"
    monkeypatch.setattr(env.service.fence, "mark", original)
    atomic_json(env.root / "result-collections/demo/run-a/attempt-001.json", {"status": "WAITING"})
    result = env.runtime.action_service.reconcile(plan["action_id"])
    assert result["execution"]["error"] == "RUN_DELETION_PLAN_CONFLICT"
    assert not env.service.fence.is_deleted("demo", "run-a") and not env.deleted


def test_local_target_swapped_after_backend_cleanup_is_not_followed(deletion_env, tmp_path, monkeypatch):
    env = deletion_env
    plan = prepare(env)
    protected = tmp_path / "independent"
    protected.mkdir()
    (protected / "research.bin").write_bytes(b"preserve")
    target = env.root / "persistent-checkpoints/demo/run-a"
    original = env.backend.execute
    def swapped(*args):
        result = original(*args)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(protected, target_is_directory=True)
        return result
    monkeypatch.setattr(env.backend, "execute", swapped)
    result = execute(env, plan)
    assert result["execution"]["status"] == "RECONCILE_REQUIRED"
    assert target.is_symlink() and (protected / "research.bin").read_bytes() == b"preserve"


def test_deletion_action_requires_service_and_reconcile_state(deletion_env):
    env = deletion_env
    plan = prepare(env)
    with pytest.raises(ActionError, match="not awaiting reconciliation"):
        env.runtime.action_service.reconcile(plan["action_id"])
    env.runtime.action_service.authorize(plan["action_id"], "reviewed")
    started, pending = env.runtime.action_service.begin_execute(plan["action_id"], "EXECUTE " + plan["action_id"])
    assert env.runtime.action_service.reconcile(plan["action_id"]) == started
    env.runtime.action_service.deletion_service = None
    with pytest.raises(ActionError, match="unavailable"):
        env.runtime.action_service.finish_execute(pending)


def test_deleted_run_cannot_prepare_authorize_or_claim_other_actions(deletion_env):
    env = deletion_env
    scope = OperationScope(project="demo", scope_type=OperationScopeType.RUN, object_id="run-a")
    plan = {"action_id": "action-" + "f" * 16, "scope": scope.model_dump(mode="json"),
            "operation": "CANCEL_RUN", "ready": True, "request_digest": "fixture"}
    env.runtime.action_store.save_plan(plan)
    env.service.fence.mark("demo", "run-a")
    for operation in [
        lambda: env.runtime.action_service.prepare(scope, env.runtime.project("demo"), {
            "kind": "ARCHIVE_RUN", "title": "late archive", "draft": "{}"}),
        lambda: env.runtime.action_service.authorize(plan["action_id"], "late"),
        lambda: env.runtime.action_service.begin_execute(plan["action_id"], "late"),
    ]:
        with pytest.raises(ApplicationError) as caught: operation()
        assert (caught.value.code, caught.value.status_code) == ("RUN_DELETED", 410)


def test_catalog_can_reload_in_same_runtime_after_deleting_one_run(deletion_env):
    env = deletion_env
    env.add_run("other")
    env.runtime._replace_active_project(load_research_project(env.project_file))
    plan = prepare(env)
    assert execute(env, plan)["execution"]["status"] == "VERIFIED"
    reloaded = load_research_project(env.project_file)
    assert [item.name for item in reloaded.campaigns] == ["run-other"]
    assert [item.name for item in env.runtime.project("demo").campaigns] == ["run-other"]
    assert prepare(env, run="other", key="next-run")["ready"]
    with sqlite3.connect(env.config.index_db_path()) as conn:
        assert conn.execute("SELECT COUNT(*) FROM runs WHERE project='demo' AND run_id='run-a'").fetchone()[0] == 0


def test_same_action_has_one_executor_lease_and_other_calls_do_not_dispatch(deletion_env):
    env = deletion_env
    plan = prepare(env)
    with env.service.execution_lease(plan["action_id"]):
        with pytest.raises(ApplicationError) as caught: env.service.execute(plan)
        assert caught.value.code == "RUN_DELETION_BUSY"
    assert not env.backend.calls and not env.deleted
    with pytest.raises(ValueError):
        with env.service.execution_lease("invalid"):
            pytest.fail("invalid lease identity")


def test_descriptor_helpers_refuse_links_and_nonregular_files(tmp_path):
    missing = tmp_path / "missing-parent" / "record.json"
    assert mapping(missing) == {}
    unlink_file(missing)
    unlink_file(tmp_path / "missing-leaf")
    directory = tmp_path / "safe"
    directory.mkdir()
    file = directory / "record.json"
    atomic_json(file, {"keep": True})
    linked = tmp_path / "parent-link"
    linked.symlink_to(directory, target_is_directory=True)
    for call in (lambda: mapping(linked / "record.json"), lambda: unlink_file(linked / "record.json")):
        with pytest.raises((OSError, ValueError)): call()
    fifo = tmp_path / "record.fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular file"): mapping(fifo)
    with pytest.raises(ValueError, match="not regular"): unlink_file(directory)
    unlink_file(file)
    assert not file.exists()


def test_catalogue_replace_failure_preserves_old_content_and_removes_own_temp(tmp_path, monkeypatch):
    path = tmp_path / "research_project.yaml"
    path.write_text("original: true\n")
    def fail(*args, **kwargs):
        raise OSError("injected replace failure")
    monkeypatch.setattr("ml_exp_server.runs.deletion.os.replace", fail)
    with pytest.raises(OSError, match="replace failure"):
        replace_yaml(path, {"replacement": True})
    assert path.read_text() == "original: true\n" and not list(tmp_path.glob(".deletion-*"))


@pytest.mark.parametrize("damage", ["duplicate", "outside", "omitted", "foreign-preparation"])
def test_local_path_allowlist_is_exact_even_when_manifest_digest_is_recomputed(deletion_env, damage):
    env = deletion_env
    plan = prepare(env)
    changed = deepcopy(plan)
    paths = changed["target_manifest"]["local_paths"]
    if damage == "duplicate": paths.append(paths[0])
    elif damage == "outside": paths.append(str(env.root / "shared-private-record.json"))
    elif damage == "omitted": paths.pop()
    else:
        path = env.root / "experiment-preparations/demo" / ("preparation-" + "c" * 32 + ".json")
        atomic_json(path, {"run_id": "other"})
        paths.append(str(path))
    paths.sort()
    changed["target_digest"] = digest(changed["target_manifest"])
    with pytest.raises(ValueError): env.service.execute(changed)
    assert not env.deleted and not env.backend.calls


@pytest.mark.parametrize("damage", ["name", "path"])
def test_catalogue_identity_mismatch_is_never_silently_detached(deletion_env, damage):
    env = deletion_env
    value = mapping(env.project_file)
    if damage == "name": value["campaigns"][0]["name"] = "other"
    else: value["campaigns"][0]["file"] = "experiments/campaigns/other.yaml"
    env.project_file.write_text(yaml.safe_dump(value))
    original = env.project_file.read_bytes()
    with pytest.raises(ValueError, match="catalogue binding differs"):
        env.service._detach_catalogue("demo", "run-a")
    assert env.project_file.read_bytes() == original and env.definitions["run-a"].exists()


def test_relative_catalogue_reference_is_detached_by_exact_canonical_path(deletion_env):
    env = deletion_env
    value = mapping(env.project_file)
    value["campaigns"][0]["file"] = "experiments/campaigns/run-run-a.yaml"
    env.project_file.write_text(yaml.safe_dump(value))
    plan = prepare(env)
    assert execute(env, plan)["execution"]["status"] == "VERIFIED"
    assert not mapping(env.project_file)["campaigns"]


@pytest.mark.parametrize("damage", ["scope", "run", "operation"])
def test_action_scope_is_rechecked_before_any_executor_effect(deletion_env, damage):
    env = deletion_env
    plan = deepcopy(prepare(env))
    if damage == "scope": plan["scope"]["object_id"] = "other"
    elif damage == "run": plan["run_id"] = "other"
    else: plan["operation"] = "CANCEL_RUN"
    with pytest.raises(ValueError, match="Action scope differs"):
        env.service.execute(plan)
    assert not env.backend.calls and not env.deleted


@pytest.mark.parametrize("damage", ["status", "action", "digest"])
def test_durable_recovery_rejects_wrong_action_identity_or_stage(deletion_env, damage):
    env = deletion_env
    plan = prepare(env)
    state = {"deletion_id": plan["action_id"], "plan_sha256": backend_digest(plan["target_manifest"]["backend"])}
    status = "DELETING"
    if damage == "status": status = "PREPARED"
    elif damage == "action": state["deletion_id"] = "action-" + "f" * 16
    else: state["plan_sha256"] = "0" * 64
    env.service.update(plan["action_id"], status=status, backend=state,
        target_manifest=plan["target_manifest"], target_digest=plan["target_digest"])
    with pytest.raises(ValueError, match="durable deletion identity or state differs"):
        env.service.execute(plan)
    assert not env.backend.calls and not env.deleted


def test_recovery_before_fence_reobserves_provider_terminal_state(deletion_env, monkeypatch):
    env = deletion_env
    plan = prepare(env)
    original = env.service.fence.mark
    monkeypatch.setattr(env.service.fence, "mark", lambda *a, **k: (_ for _ in ()).throw(OSError("injected crash")))
    assert execute(env, plan)["execution"]["status"] == "RECONCILE_REQUIRED"
    monkeypatch.setattr(env.service.fence, "mark", original)
    def now_running(*args):
        raise ApplicationError("original provider job is running", code="RUN_DELETION_JOB_NOT_TERMINAL")
    monkeypatch.setattr(env.backend, "terminal", now_running)
    result = env.runtime.action_service.reconcile(plan["action_id"])
    assert result["execution"]["status"] == "RECONCILE_REQUIRED"
    assert result["execution"]["error"] == "RUN_DELETION_JOB_NOT_TERMINAL"
    assert not env.service.fence.is_deleted("demo", "run-a") and not env.deleted


def test_cleanup_crash_after_catalogue_detach_continues_without_new_delete_authorization(deletion_env, monkeypatch):
    env = deletion_env
    receipt, original_receipt = publish(env)
    plan = prepare(env)
    import ml_exp_server.runs.deletion as module
    original = module.unlink_file
    def fail_once(path):
        raise OSError("crash after catalogue detach")
    monkeypatch.setattr(module, "unlink_file", fail_once)
    failed = execute(env, plan)
    assert failed["execution"]["status"] == "RECONCILE_REQUIRED"
    assert not mapping(env.project_file)["campaigns"] and not env.local["run-a"].exists()
    assert env.definitions["run-a"].exists() and len(env.deleted) == 2
    monkeypatch.setattr(module, "unlink_file", original)
    recovered = env.runtime.action_service.reconcile(plan["action_id"])
    assert recovered["execution"]["status"] == "VERIFIED"
    assert len(env.deleted) == 2 and receipt.read_bytes() == original_receipt
    assert not env.definitions["run-a"].exists() and len(env.runtime.action_store.list_all()) == 1


def test_snapshot_asset_inventory_is_reviewed_and_retained(deletion_env):
    env = deletion_env
    objects = env.runtime.storage_service.objects
    objects.issue("demo", "run-a", "attempt-001", env.local["run-a"], ["**/*"])
    asset = "asset." + "1" * 64
    with objects.record("demo", "run-a", "attempt-001") as (path, value):
        value["snapshots"] = {asset: {"kind": "uploaded-checkpoint-snapshot"}}
        atomic_json(path, value)
    snapshot = env.root / "assets/demo" / asset / "manifest.json"
    atomic_json(snapshot, {"kind": "checkpoint", "sha256": "1" * 64})
    plan = prepare(env)
    assert plan["target_manifest"]["retained_snapshot_assets"] == [asset]
    original = snapshot.read_bytes()
    assert execute(env, plan)["execution"]["status"] == "VERIFIED"
    assert snapshot.read_bytes() == original


@pytest.mark.parametrize("stage", ["parent", "leaf"])
def test_evidence_io_errors_propagate_instead_of_becoming_missing_records(tmp_path, monkeypatch, stage):
    import ml_exp_server.runs.deletion as module
    path = tmp_path / "record.json"
    atomic_json(path, {"retain": "unreadable evidence is not absence"})
    original = module.os.open
    def unavailable(*args, **kwargs):
        if stage == "parent" or args[0] == path.name:
            raise PermissionError(errno.EACCES, "injected unreadable evidence")
        return original(*args, **kwargs)
    if stage == "parent": monkeypatch.setattr(module, "open_directory", unavailable)
    else: monkeypatch.setattr(module.os, "open", unavailable)
    with pytest.raises(PermissionError): mapping(path)


@pytest.mark.parametrize("damage", ["result-status", "stage", "receipt", "missing-receipt"])
def test_completed_journal_requires_matching_positive_backend_receipts(deletion_env, damage):
    env = deletion_env
    plan = prepare(env)
    assert execute(env, plan)["execution"]["status"] == "VERIFIED"
    before = deepcopy(env.backend.calls)
    with env.service.state(plan["action_id"]) as (store, snapshot):
        value = deepcopy(snapshot.value)
        if damage == "result-status": value["backend_result"]["status"] = "PENDING"
        elif damage == "stage": value["backend"]["stage"] = "WAITING"
        elif damage == "receipt": value["backend_result"]["receipt"]["bytes_deleted"] += 1
        else: value["backend_result"]["receipt"] = None
        store.commit(value, expected_revision=snapshot.revision, event={"event": "injected journal corruption"})
    with pytest.raises(ValueError): env.service.execute(plan)
    assert env.backend.calls == before


@pytest.mark.parametrize("field", ["project", "run_id", "deletion_id", "target_digest", "status"])
def test_recovery_requires_tombstone_bound_to_the_same_action_and_review(deletion_env, field):
    env = deletion_env
    plan = prepare(env)
    assert execute(env, plan)["execution"]["status"] == "VERIFIED"
    before = deepcopy(env.backend.calls)
    path = env.service.fence.path("demo", "run-a")
    changed = mapping(path)
    changed[field] = "different-original-identity"
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="fence identity differs"):
        env.service.execute(plan)
    assert env.backend.calls == before
