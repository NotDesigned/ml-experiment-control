"""Independent recovery, exact identities, interruption and CPU-only budgets."""
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from fastapi.testclient import TestClient

from ml_exp_server.api.app import create_app
from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.results.artifacts import ArtifactService
from ml_exp_server.results.artifact_store import ArtifactStore
from ml_exp_server.results.result_collection import ResultCollectionService, ACTIVE
from ml_exp_server.schemas import ServerConfig, RunIndexRow, AttemptSummary
from ml_exp_server.storage import atomic_json
from experiment_control.backends.sensecore_rest import RESTError
from tests.test_artifact_store import storage, archive

IDENTITY = ("demo", "run-a", "attempt-001")


@pytest.fixture
def recovery(storage, tmp_path, monkeypatch):
    objects, root, token, config, remote = storage
    backend = {"kind": "slurm", "ssh_alias": "wyd-l40s", "workspace": "ws", "aec2": "training",
               "storage_mount": "volume/user:/data", "image": "registry.example/base@sha256:" + "a" * 64}
    definition = {"project": "demo", "run_id": "run-a", "attempt_id": "attempt-001",
                  "backend": backend, "storage": {"run_dir": str(tmp_path / "backend")}}
    (root / "manifest.yaml").write_text(yaml.safe_dump(definition))
    attempt = root / "attempts/attempt-001"
    (attempt / "attempt.yaml").write_text(yaml.safe_dump(definition))
    (attempt / "backend.json").write_text(json.dumps({"backend_job_id": "24830"}))
    data = json.loads(config.read_text())
    data["data_delivery"] = {"aec2": "debug-cluster", "gpus": 0, "cpus": 2, "memory_gb": 4,
                            "worker_spec": "cpu.2c4g", "quota_type": "reserved"}
    config.write_text(json.dumps(data))
    cfg = ServerConfig(index_db=str(tmp_path / "index.sqlite"), project_registry_root=str(objects.root.parent),
                       collector_enabled=False, telemetry={"enabled": False},
                       action_runtime={"allow_project_writes": True, "allow_scheduler_mutations": True},
                       container_execution={"artifact_store_file": str(config)})
    row = RunIndexRow(project="demo", run_id="run-a", run_dir=str(root),
                      attempts=[AttemptSummary(attempt_id="attempt-001", state="FAILED")])
    index = SimpleNamespace(get_run=lambda p, r: row if (p, r) == ("demo", "run-a") else None)
    service = ResultCollectionService(SimpleNamespace(config=cfg, index=index))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="FAILED|\n"))
    return service, root, token, config, remote, definition, row


def output(root):
    path = root / "attempts/attempt-001/outputs"
    path.mkdir()
    (path / "weights.bin").write_bytes(b"weights")
    (path / "metrics.jsonl").write_bytes(b'{"arbitrary_metric":3}\n')
    return path


def test_terminal_local_results_published_without_duplicate_cache_and_original_state_preserved(recovery):
    service, root, token, config, remote, _, row = recovery
    output(root)
    first, pending = service.begin(*IDENTITY)
    assert first["status"] == "QUEUED"
    assert service.begin(*IDENTITY)[1] is None
    service.finish(pending)
    status = service.read(*IDENTITY)
    assert status["status"] == "AVAILABLE" and status["verification"] == "SHA256_VERIFIED"
    assert status["training"]["status"] == "UNKNOWN" and row.attempts[0].state == "FAILED"
    assert len(remote) == 1 and not (root / "attempts/attempt-001/uploaded_outputs").exists()
    listing = ArtifactService(service.runtime).list(*IDENTITY, checksums=True)
    assert listing["available"] and len([v for v in listing["files"] if v["path"].startswith("outputs/")]) == 2
    assert service.begin(*IDENTITY)[1] is None
    with service.objects.record(*IDENTITY) as (_, value):
        assert value["token"] == token and value["receipt"]["files"][0]["sha256"]
    # Cold files listing is metadata-only; it must not materialize the archive.
    for file in (root / "attempts/attempt-001/outputs").iterdir(): file.unlink()
    (root / "attempts/attempt-001/outputs").rmdir()
    assert ArtifactService(service.runtime).list(*IDENTITY)["available"]
    assert not (root / "attempts/attempt-001/uploaded_outputs").exists()


def test_training_result_is_scoped_frozen_and_separate_from_scheduler_status(recovery):
    service, root, token, _, _, _, _ = recovery
    value = dict(zip(("project", "run_id", "attempt_id"), IDENTITY), contract="training-result.v1", exit_code=0)
    assert service.register_training(*IDENTITY, token, value)["accepted"]
    assert service.register_training(*IDENTITY, token, value)["accepted"]
    assert service.read(*IDENTITY)["training"]["status"] == "COMPLETED"
    assert ArtifactService(service.runtime).list(*IDENTITY)["training"]["exit_code"] == 0
    with pytest.raises(ValueError, match="frozen"):
        service.register_training(*IDENTITY, token, {**value, "exit_code": 1})
    with pytest.raises(ValueError, match="capability"):
        service.register_training(*IDENTITY, "wrong", value)
    for bad in ([], {**value, "exit_code": True}, {**value, "run_id": "other"}, {**value, "contract": "unknown"}):
        with pytest.raises(ApplicationError):
            service.register_training(*IDENTITY, token, bad)


def test_unknown_active_attempt_policy_and_retry_limits(recovery):
    service, _, _, _, _, _, row = recovery
    with pytest.raises(ApplicationError): service.read("other", "run-a", "attempt-001")
    row.attempts[0].state = "RUNNING"
    with pytest.raises(ApplicationError, match="terminal"): service.begin(*IDENTITY)
    row.attempts[0].state = "FAILED"
    for count in range(1, 4):
        result, _ = service.begin(*IDENTITY, retry=True)
        assert result["attempts"] == count
        service.update(IDENTITY, status="FAILED")
        assert service.begin(*IDENTITY)[1] is None
    with pytest.raises(ApplicationError, match="budget"): service.begin(*IDENTITY, retry=True)
    service.runtime.config.action_runtime.allow_project_writes = False
    with pytest.raises(ApplicationError): service.begin(*IDENTITY)


def test_daemon_interruption_requires_reconcile_and_never_replays_submit(recovery):
    service, _, _, _, _, _, _ = recovery
    service.begin(*IDENTITY)
    assert service.recover_interrupted() == 1
    assert service.recover_interrupted() == 0
    assert service.begin(*IDENTITY, retry=True)[1] is None
    _, pending = service.begin(*IDENTITY, reconcile=True)
    assert pending == IDENTITY
    assert service.read(*IDENTITY)["attempts"] == 1


@pytest.mark.parametrize("case", ["run", "attempt", "transfer", "unknown-attempt", "bad-state", "bad-job", "bad-alias", "unsupported"])
def test_invalid_scope_or_unconfirmed_terminal_never_publishes(recovery, monkeypatch, case):
    service, root, _, _, remote, definition, row = recovery
    if case == "run":
        (root / "manifest.yaml").write_text("project: other\nrun_id: run-a\n")
    elif case == "attempt":
        definition["attempt_id"] = "attempt-002"
        (root / "attempts/attempt-001/attempt.yaml").write_text(yaml.safe_dump(definition))
    elif case == "transfer":
        with service.objects.record(*IDENTITY) as (path, value):
            value["run_dir"] = "/other"; atomic_json(path, value)
    elif case == "unknown-attempt":
        row.attempts = []
    else:
        if case == "bad-state": monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="RUNNING|\n"))
        if case == "bad-job": (root / "attempts/attempt-001/backend.json").write_text('{"backend_job_id":"; unsafe"}')
        if case == "bad-alias": definition["backend"]["ssh_alias"] = "unsafe alias"
        if case == "unsupported": definition["backend"]["kind"] = "unsupported"
        (root / "attempts/attempt-001/attempt.yaml").write_text(yaml.safe_dump(definition))
        service.begin(*IDENTITY)
        service.finish(IDENTITY)
        assert service.read(*IDENTITY)["status"] == "FAILED"
        assert not remote
        return
    with pytest.raises((ApplicationError, ValueError)): service.begin(*IDENTITY)
    assert not remote


def cpu(recovery, monkeypatch):
    service, root, _, _, _, definition, _ = recovery
    definition["backend"]["kind"] = "sensecore"
    (root / "attempts/attempt-001/attempt.yaml").write_text(yaml.safe_dump(definition))
    rest = Mock()
    rest.describe.return_value = {"state": "FAILED"}
    rest.specs.return_value = [{"name": "cpu.2c4g", "device": {"number": 0},
                               "cpu": {"vcpu_allocatable": 2}, "memory": {"allocatable": 4}}]
    monkeypatch.setattr("ml_exp_server.results.result_collection.SenseCoreREST.from_environment", lambda: rest)
    return rest


def test_uncertain_cpu_request_reconciles_exact_job_and_does_not_restart_training(recovery, monkeypatch):
    service, _, _, _, _, _, _ = recovery
    rest = cpu(recovery, monkeypatch)
    rest.create.side_effect = RESTError("create", uncertain=True)
    service.begin(*IDENTITY); service.finish(IDENTITY)
    assert service.read(*IDENTITY)["status"] == "RECONCILE_REQUIRED"
    assert service.begin(*IDENTITY, retry=True)[1] is None
    rest.find.return_value = []
    _, pending = service.begin(*IDENTITY, reconcile=True); service.finish(pending)
    assert service.read(*IDENTITY)["status"] == "RECONCILE_REQUIRED" and rest.create.call_count == 1
    rest.find.return_value = [{"state": "FAILED"}]
    _, pending = service.begin(*IDENTITY, reconcile=True); service.finish(pending)
    assert service.read(*IDENTITY)["status"] == "FAILED" and rest.create.call_count == 1
    assert rest.stop.call_count == 0


def test_cpu_recovery_reuses_data_image_and_verifies_zero_gpu_and_archive(recovery, monkeypatch):
    service, root, _, _, remote, _, _ = recovery
    rest = cpu(recovery, monkeypatch)
    folder = service.objects.root.parent / "data-deliveries/demo"; folder.mkdir(parents=True)
    atomic_json(folder / "delivery.a.json", {"status": "READY", "image": "wrong",
               "copy_profile": {"workspace": "other", "storage_mount": "other:/data"}})
    atomic_json(folder / "delivery.example.json", {"status": "READY", "image": "registry.example/cpu@sha256:" + "b" * 64,
               "copy_profile": {"workspace": "ws", "storage_mount": "volume/user:/data"}})
    def create_unlocked(profile, body, **kwargs):
        assert profile["gpus"] == 0 and body["roles"][0]["image_path"].startswith("registry.example/cpu@")
        _, _, value, _ = service.evidence(*IDENTITY)
        data = archive({"any-name.bin": b"payload"})
        service.objects.receive(*IDENTITY, value["token"], io.BytesIO(data), len(data))
    rest.create.side_effect = create_unlocked
    service.begin(*IDENTITY); service.finish(IDENTITY)
    assert service.read(*IDENTITY)["status"] == "AVAILABLE" and len(remote) == 1
    body = rest.create.call_args.args[1]
    profile = rest.create.call_args.args[0]
    assert (profile["gpus"], profile["cpus"], profile["memory_gb"], profile["quota_type"]) == (0, 2, 4, "reserved")
    assert body["roles"][0]["resource_spec"][0]["limits"] == {"cpu": "2", "memory": "4Gi"}
    assert body["roles"][0]["total_replicas"] == 1
    assert not (root / "attempts/attempt-001/uploaded_outputs").exists()


def test_disabled_storage_and_invalid_collection_paths_fail_closed(recovery):
    service, _, _, _, _, _, _ = recovery
    with pytest.raises(ValueError):
        with service.state("../outside", "run", "attempt-001"): pass
    service.runtime.config.container_execution.artifact_store_file = None
    with pytest.raises(ApplicationError): ResultCollectionService(service.runtime)
    for root in ("relative", "/data/../outside"):
        with pytest.raises(ValueError): service.remote_outputs({"storage": {"run_dir": root}}, "attempt-001")


def test_shared_storage_transfer_is_atomic_and_bounded(recovery, monkeypatch):
    service, root, _, _, remote, _, _ = recovery
    commands = []
    def run(command, **kwargs):
        commands.append((command, kwargs))
        if command[0] == "rsync":
            Path(command[-1]).joinpath("chosen.bin").write_bytes(b"weights")
        return SimpleNamespace(stdout="FAILED|\n")
    monkeypatch.setattr(subprocess, "run", run)
    service.begin(*IDENTITY); service.finish(IDENTITY)
    assert service.read(*IDENTITY)["status"] == "AVAILABLE" and remote
    assert commands[1][0][0] == "rsync" and commands[1][1]["timeout"] == 300
    assert not list((root / "attempts/attempt-001").glob(".result-recovery-*"))


@pytest.mark.parametrize("case", ["empty", "changed", "matching", "conflicting"])
def test_unfinished_archive_identity_cannot_be_silently_replaced(recovery, case):
    service, root, _, _, remote, _, _ = recovery
    path = output(root)
    raw = io.BytesIO()
    from ml_exp_server.workers.worker_artifacts import archive_outputs
    archive_outputs(path, raw, 0, ["**/*"])
    digest = hashlib.sha256(raw.getvalue()).hexdigest()
    folder = service.objects.root.parent / "multipart-uploads/upload.first"; folder.mkdir(parents=True)
    value = {"binding": dict(zip(("project", "run_id", "attempt_id"), IDENTITY), kind="artifacts"),
             "status": "UPLOADING", "sha256": digest, "bytes": len(raw.getvalue())}
    atomic_json(folder / "upload.json", value)
    unrelated = folder.parent / "upload.unrelated"; unrelated.mkdir()
    atomic_json(unrelated / "upload.json", {**value, "status": "COMPLETED"})
    atomic_json(folder / "other.json", {})
    if case == "empty":
        for file in path.iterdir(): file.unlink()
    if case == "changed": (path / "weights.bin").write_bytes(b"changed")
    if case == "conflicting":
        other = folder.parent / "upload.second"; other.mkdir()
        atomic_json(other / "upload.json", {**value, "sha256": "f" * 64})
    service.begin(*IDENTITY); service.finish(IDENTITY)
    assert service.read(*IDENTITY)["status"] == ("AVAILABLE" if case == "matching" else "FAILED")
    assert bool(remote) == (case == "matching")


@pytest.mark.parametrize("case", ["running-original", "blocked", "bad-pool", "known-request-error", "stopping", "deadline", "poll-then-upload"])
def test_cpu_recovery_limits_and_provider_failures(recovery, monkeypatch, case):
    service, root, _, _, _, _, _ = recovery
    rest = cpu(recovery, monkeypatch)
    service.begin(*IDENTITY)
    rest.describe.return_value = {"state": "RUNNING"}
    if case != "running-original":
        seen = []
        def describe(profile, name):
            seen.append(name)
            if name == "24830": return {"state": "FAILED"}
            if case == "poll-then-upload" and len(seen) > 2:
                _, _, value, _ = service.evidence(*IDENTITY)
                data = archive({"chosen.bin": b"ok"})
                service.objects.receive(*IDENTITY, value["token"], io.BytesIO(data), len(data))
            return {"state": "RUNNING"}
        rest.describe.side_effect = describe
    if case == "blocked": service.runtime.config.action_runtime.allow_scheduler_mutations = False
    if case == "bad-pool": service.objects.config["data_delivery"]["aec2"] = "compute-cluster"
    if case == "known-request-error": rest.describe.side_effect = RESTError("query", status=403)
    if case == "stopping": service.stopping = lambda: True
    if case == "deadline":
        rest.create.side_effect = lambda *a, **kw: service.update(IDENTITY, deadline_at=-1)
    monkeypatch.setattr("ml_exp_server.results.result_collection.time.sleep", lambda s: None)
    service.finish_one(IDENTITY)
    result = service.read(*IDENTITY)
    assert result["status"] == ("AVAILABLE" if case == "poll-then-upload" else "RECONCILE_REQUIRED" if case in {"stopping", "deadline"} else "FAILED")
    assert rest.stop.call_count == (1 if case == "deadline" else 0)
    if case in {"running-original", "blocked", "bad-pool", "known-request-error"}:
        assert not rest.create.called


def test_automatic_waits_for_original_attempt_to_end_and_retains_failed_training(recovery):
    service, _, token, _, _, _, row = recovery
    value = dict(zip(("project", "run_id", "attempt_id"), IDENTITY), contract="training-result.v1", exit_code=1)
    service.register_training(*IDENTITY, token, value)
    assert service.read(*IDENTITY)["training"]["status"] == "FAILED"
    assert ArtifactService(service.runtime).list(*IDENTITY)["training"]["status"] == "FAILED"
    calls = []
    service.automatic(lambda *args: calls.append(args))
    assert not calls
    with service.objects.record(*IDENTITY) as (path, transfer):
        transfer["results_ready"]["exit_code"] = 0; atomic_json(path, transfer)
    row.attempts[0].state = "RUNNING"
    service.automatic(lambda *args: calls.append(args))
    assert not calls


def test_only_new_successful_training_records_auto_enroll(recovery):
    service, _, token, _, _, _, _ = recovery
    submitted = []
    service.automatic(lambda *args: submitted.append(args))
    assert not submitted
    value = dict(zip(("project", "run_id", "attempt_id"), IDENTITY), contract="training-result.v1", exit_code=0)
    service.register_training(*IDENTITY, token, value)
    service.automatic(lambda *args: submitted.append(args))
    service.automatic(lambda *args: submitted.append(args))
    assert len(submitted) == 1 and submitted[0][1] == IDENTITY


def test_recovery_queue_survives_executor_failure_and_responds_to_shutdown(recovery, monkeypatch):
    import fcntl
    service, _, token, _, _, _, _ = recovery
    value = dict(zip(("project", "run_id", "attempt_id"), IDENTITY), contract="training-result.v1", exit_code=0)
    service.register_training(*IDENTITY, token, value)
    def fail(*args): raise RuntimeError("stopped executor")
    service.automatic(fail)
    assert service.read(*IDENTITY)["diagnostic"] == "EXECUTOR_UNAVAILABLE"
    service.begin(*IDENTITY, reconcile=True)
    original = fcntl.flock
    seen = []
    def busy_once(file, flags):
        if flags & fcntl.LOCK_NB and not seen:
            seen.append(True)
            raise BlockingIOError()
        return original(file, flags)
    monkeypatch.setattr(fcntl, "flock", busy_once)
    service.stopping = lambda: bool(seen)
    service.finish(IDENTITY)
    assert service.read(*IDENTITY)["diagnostic"] == "DAEMON_INTERRUPTED"


def test_http_collection_control_and_worker_capability_boundaries(recovery, tmp_path):
    service, _, token, _, _, _, row = recovery
    bearer = tmp_path / "bearer"; bearer.write_text("a" * 40); bearer.chmod(0o600)
    cfg = service.runtime.config
    cfg.http_auth.bearer_token_file = str(bearer)
    with TestClient(create_app(cfg, poll=False)) as client:
        client.app.state.runtime.index.upsert_run(row)
        client.app.state.submit_job = lambda *args: None
        endpoint = "/api/runs/demo/run-a/attempts/attempt-001/collection"
        headers = {"Authorization": "Bearer " + "a" * 40}
        assert client.get(endpoint).status_code == 401
        assert client.get(endpoint, headers=headers).json()["status"] == "NOT_COLLECTED"
        assert client.post(endpoint, headers=headers, json={}).json()["status"] == "QUEUED"
        assert client.post(endpoint, headers=headers, json={}).json()["status"] == "QUEUED"
        assert client.post(endpoint, headers={"Authorization": "Bearer " + token}, json={}).status_code == 401
        ready = "/api/result-ready-transfers/demo/run-a/attempt-001"
        value = dict(zip(("project", "run_id", "attempt_id"), IDENTITY), contract="training-result.v1", exit_code=0)
        assert client.put(ready, headers={"Authorization": "Bearer wrong"}, json=value).status_code == 401
        assert client.put(ready, headers={"Authorization": "Bearer " + token}, json=value).json()["accepted"]
        assert client.get(endpoint, headers=headers).json()["training"]["status"] == "COMPLETED"
        assert client.get(ready, headers={"Authorization": "Bearer " + token}).status_code == 401
        assert client.put(ready, headers={"Authorization": "Bearer " + token}, content="x" * 4097).status_code == 413
        assert client.put(ready, headers={"Authorization": "Bearer " + token}, content="invalid").status_code == 422
        assert client.put(ready, headers={"Authorization": "Bearer " + token}, json={**value, "exit_code": 1}).status_code == 409
        def stopped(*args): raise RuntimeError("executor stopped")
        client.app.state.submit_job = stopped
        service.update(IDENTITY, status="FAILED")
        assert client.post(endpoint, headers=headers, json={"retry": True}).status_code == 503
        assert client.get(endpoint, headers=headers).json()["diagnostic"] == "EXECUTOR_UNAVAILABLE"


@pytest.mark.parametrize("broken", [False, True])
def test_submission_progress_includes_collection_without_hiding_bad_metadata(recovery, monkeypatch, tmp_path, broken):
    from ml_exp_server.runs.submissions import ExperimentSubmissionService
    service, _, _, _, _, _, _ = recovery
    service.runtime.action_store = SimpleNamespace(directory=lambda _: tmp_path)
    submissions = ExperimentSubmissionService(None, service.runtime)
    monkeypatch.setattr(submissions, "get", lambda _: {"status": "VERIFIED", "project": "demo", "run_id": "run-a",
                                                      "attempt_id": "attempt-001", "execution": {}})
    if broken:
        monkeypatch.setattr(ResultCollectionService, "read", lambda *args: (_ for _ in ()).throw(ValueError()))
    result = submissions.progress("submission")
    assert result["result_collection"]["status"] == ("UNAVAILABLE" if broken else "NOT_COLLECTED")


def test_receipt_only_files_listing_is_bounded_without_restoring_objects(recovery):
    service, _, _, _, _, _, _ = recovery
    with service.objects.record(*IDENTITY) as (path, value):
        value["receipt"] = {"files": [{"path": str(i), "bytes": 1, "sha256": "a" * 64} for i in range(10001)]}
        atomic_json(path, value)
    result = ArtifactService(service.runtime).list(*IDENTITY)
    assert result["truncated"] and len(result["files"]) == 10000


def test_files_listing_without_a_transfer_record_is_uncollected(recovery):
    service, _, _, _, _, _, _ = recovery
    with service.objects.record(*IDENTITY) as (path, _):
        path.unlink()
    result = ArtifactService(service.runtime).list(*IDENTITY)
    assert result["files"] == [] and result["collection_required"]
    assert result["training"]["status"] == "UNKNOWN"


@pytest.mark.parametrize("enough", [False, True])
def test_artifact_completion_reserves_validation_and_colocated_object_storage(recovery, monkeypatch, enough):
    service, _, token, _, remote, _, _ = recovery
    data = archive({"chosen.bin": b"weights"})
    checksum = hashlib.sha256(data).hexdigest()
    headers = {"Authorization": "Bearer " + token}
    with TestClient(create_app(service.runtime.config, poll=False)) as client:
        base = "/api/attempt-uploads/demo/run-a/attempt-001/artifacts"
        upload = client.post(base, json={"sha256": checksum, "bytes": len(data)}, headers=headers).json()
        endpoint = base + "/" + upload["upload_id"]
        assert client.put(endpoint + "/parts/0", params={"sha256": checksum}, content=data, headers=headers).status_code == 200
        # Parts already exist; validation and a local object store both need space.
        free = 64 * 1024 ** 2 + 2 * len(data) - (0 if enough else 1)
        monkeypatch.setattr("ml_exp_server.api.upload_routes.shutil.disk_usage", lambda p: SimpleNamespace(free=free))
        response = client.post(endpoint + "/complete", headers=headers)
        assert response.status_code == (200 if enough else 507)
        assert bool(remote) == enough
        assert client.get(endpoint, headers=headers).json()["status"] == ("COMPLETED" if enough else "UPLOADING")


def test_poll_loop_automatically_enqueues_new_results(recovery):
    from fastapi import FastAPI
    from ml_exp_server.api.app import _poll_loop
    service, _, token, _, _, _, _ = recovery
    value = dict(zip(("project", "run_id", "attempt_id"), IDENTITY), contract="training-result.v1", exit_code=0)
    service.register_training(*IDENTITY, token, value)
    checks = iter([False, True])
    app = FastAPI()
    app.state.runtime = service.runtime
    app.state.index = SimpleNamespace(set_meta=lambda *args: None)
    app.state._stop = SimpleNamespace(is_set=lambda: next(checks), wait=lambda _: None)
    submitted = []
    app.state.submit_job = lambda *args: submitted.append(args)
    _poll_loop(app, SimpleNamespace(run_cycle=lambda: None, config=SimpleNamespace(poll_interval_seconds=1)))
    assert len(submitted) == 1 and submitted[0][1] == IDENTITY


@pytest.mark.parametrize("changed", [False, True])
def test_file_checksum_refuses_special_or_changed_files(recovery, monkeypatch, changed):
    import os
    import stat
    service, root, _, _, _, _, _ = recovery
    output(root)
    original = os.fstat
    def altered(fd):
        before = original(fd)
        if stat.S_ISREG(before.st_mode):
            return SimpleNamespace(st_mode=before.st_mode if changed else stat.S_IFIFO,
                                   st_ino=before.st_ino, st_size=before.st_size + 1,
                                   st_mtime_ns=before.st_mtime_ns)
        return before
    monkeypatch.setattr(os, "fstat", altered)
    with pytest.raises(ValueError, match="changed" if changed else "regular"):
        ArtifactService(service.runtime).list(*IDENTITY, checksums=True)
