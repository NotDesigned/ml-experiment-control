import json

import pytest

import experiment_control.manifest as manifest_module
from experiment_control.manifest import (
    ExperimentStateStore,
    RunState,
    append_event_once,
    atomic_create,
    atomic_write,
    sanitize_command,
)
from experiment_control.run_manifest import build_run_manifest, comparable_manifest


def run_manifest(tmp_path, **overrides):
    arguments = {
        "project": "project",
        "run_id": "run-s0",
        "created_at": "2026-07-12T00:00:00Z",
        "config_path": "config.yml",
        "resolved_config": {"seed": 7, "resume": "/old/checkpoint"},
        "source_id": "source-immutable",
        "runtime_tree_id": "runtime-immutable",
        "git_commit": "deadbeef",
        "campaign_id": "campaign-digest",
        "campaign": "campaign",
        "image_id": "sha256:" + "a" * 64,
        "run_dir": str(tmp_path),
        "max_infra_retries": 1,
        "backend": {"kind": "test-backend"},
        "resources": {"gpus": 1},
        "storage": {"run_dir": str(tmp_path), "checkpoint_dir": str(tmp_path)},
        "command": ["python", "train.py"],
        "execution": {"source_mount": "/workspace", "workdir": "/workspace"},
    }
    return build_run_manifest(**{**arguments, **overrides})


def prepare_store(tmp_path, attempt_id="attempt-001"):
    store = ExperimentStateStore(tmp_path)
    manifest = store.ensure_manifest(run_manifest(tmp_path))
    store.create_attempt({
        "schema_version": 1,
        "project": manifest["project"],
        "run_id": manifest["run_id"],
        "attempt_id": attempt_id,
        "created_at": "2026-07-12T00:00:01Z",
        "backend": "test-backend",
        "backend_job_id": None,
        "source_id": manifest["source_id"],
        "image_id": manifest["image_id"],
        "command": manifest["command"],
        "resources": manifest["resources"],
        "resume_from": None,
    })
    store.initialize_attempt_records(attempt_id)
    return store


def attempt_manifest(store, attempt_id="attempt-002", **overrides):
    manifest = store.load_manifest()
    payload = {
        "schema_version": 1,
        "project": manifest["project"],
        "run_id": manifest["run_id"],
        "attempt_id": attempt_id,
        "created_at": "2026-07-12T00:01:00Z",
        "backend": "test-backend",
        "backend_job_id": None,
        "source_id": manifest["source_id"],
        "image_id": manifest["image_id"],
        "command": manifest["command"],
        "resources": manifest["resources"],
        "resume_from": None,
    }
    return {**payload, **overrides}


def test_run_manifest_excludes_attempt_resume_from_scientific_identity(tmp_path):
    manifest = run_manifest(tmp_path)
    assert "resume" not in manifest["resolved_config"]
    changed = {**manifest, "created_at": "later"}
    changed["resolved_config"] = {**manifest["resolved_config"], "resume": "/new"}
    assert comparable_manifest(changed) == comparable_manifest(manifest)


def test_store_creates_canonical_run_attempt_and_current_mirrors(tmp_path):
    store = prepare_store(tmp_path)
    assert store.manifest_path.is_file()
    assert store.attempt_path("attempt-001").is_file()
    assert store.read_status().state is RunState.CREATED
    assert store.load_backend()["attempt_id"] == "attempt-001"
    assert store.read_status("attempt-999").state is RunState.NOT_SUBMITTED


def test_submission_outbox_is_idempotent_redacted_and_reconciled(tmp_path):
    store = prepare_store(tmp_path)
    request = {
        "argv": ["submit", "run.job"],
        "environment": {"API_TOKEN": "secret", "SEED": "7"},
        "callback": "https://user:password@example.invalid/status",
    }
    first = store.begin_submission(
        project="project",
        run_id="run-s0",
        attempt_id="attempt-001",
        backend="test-backend",
        request=request,
    )
    second = store.begin_submission(
        project="project",
        run_id="run-s0",
        attempt_id="attempt-001",
        backend="test-backend",
        request=request,
    )
    assert first == second
    assert first["request"]["environment"]["API_TOKEN"] == "<redacted>"
    assert first["request"]["callback"] == "https://<redacted>@example.invalid/status"
    assert store.read_status().state is RunState.SUBMITTING

    reconciled = store.reconcile_submission(
        project="project",
        run_id="run-s0",
        attempt_id="attempt-001",
        backend_job_id="job-123",
    )
    assert reconciled["state"] == "SUBMITTED"
    assert store.read_status().state is RunState.QUEUED
    assert store.load_backend()["backend_job_id"] == "job-123"


def test_repeating_submission_intent_repairs_derived_state_after_crash(tmp_path):
    store = prepare_store(tmp_path)
    kwargs = {
        "project": "project",
        "run_id": "run-s0",
        "attempt_id": "attempt-001",
        "backend": "test-backend",
        "request": {"argv": ["submit"]},
    }
    store.begin_submission(**kwargs)
    store.status_path.unlink()
    store.backend_path.unlink()
    store.attempt_status_path("attempt-001").unlink()
    store.attempt_backend_path("attempt-001").unlink()
    store.events_path.write_text(
        "\n".join(
            line for line in store.events_path.read_text().splitlines()
            if json.loads(line)["event"] != "submission_intent_created"
        ) + "\n",
        encoding="utf-8",
    )
    store.begin_submission(**kwargs)
    assert store.read_status().state is RunState.SUBMITTING
    assert store.load_backend()["attempt_id"] == "attempt-001"
    events = [json.loads(line) for line in store.events_path.read_text().splitlines()]
    assert [event["event"] for event in events].count("submission_intent_created") == 1


def test_reconcile_is_idempotent_and_rejects_different_job(tmp_path):
    store = prepare_store(tmp_path)
    store.begin_submission(
        project="project", run_id="run-s0", attempt_id="attempt-001",
        backend="test-backend", request={"argv": ["submit"]},
    )
    kwargs = {
        "project": "project",
        "run_id": "run-s0",
        "attempt_id": "attempt-001",
        "backend_job_id": "job-1",
    }
    assert store.reconcile_submission(**kwargs) == store.reconcile_submission(**kwargs)
    with pytest.raises(ValueError, match="already reconciled"):
        store.reconcile_submission(**{**kwargs, "backend_job_id": "job-2"})


def test_transition_event_id_is_idempotent(tmp_path):
    store = prepare_store(tmp_path)
    kwargs = {
        "project": "project",
        "run_id": "run-s0",
        "attempt_id": "attempt-001",
        "state": RunState.RUNNING,
        "event": "worker_observed",
        "event_id": "worker-observed:attempt-001",
    }
    store.transition(**kwargs)
    store.transition(**kwargs)
    events = [json.loads(line) for line in store.events_path.read_text().splitlines()]
    assert [event["event"] for event in events].count("worker_observed") == 1


def test_new_attempt_preserves_old_attempt_state_and_updates_root_mirror(tmp_path):
    store = prepare_store(tmp_path)
    store.begin_submission(
        project="project", run_id="run-s0", attempt_id="attempt-001",
        backend="test-backend", request={"argv": ["submit"]},
    )
    store.reconcile_submission(
        project="project", run_id="run-s0", attempt_id="attempt-001",
        backend_job_id="job-1",
    )
    manifest = store.load_manifest()
    store.create_attempt({
        "schema_version": 1,
        "project": "project",
        "run_id": "run-s0",
        "attempt_id": "attempt-002",
        "created_at": "2026-07-12T00:01:00Z",
        "backend": "test-backend",
        "backend_job_id": None,
        "source_id": manifest["source_id"],
        "image_id": manifest["image_id"],
        "command": manifest["command"],
        "resources": manifest["resources"],
        "resume_from": "/checkpoint-10",
    })
    store.initialize_attempt_records("attempt-002")
    assert store.read_status("attempt-001").state is RunState.QUEUED
    assert store.read_status("attempt-002").state is RunState.CREATED
    assert store.load_backend()["attempt_id"] == "attempt-002"


def test_new_attempt_fails_closed_when_root_mirror_drifted(tmp_path):
    store = prepare_store(tmp_path)
    drifted = json.loads(store.status_path.read_text())
    drifted["state"] = "RUNNING"
    store.status_path.write_text(json.dumps(drifted), encoding="utf-8")
    manifest = store.load_manifest()
    store.create_attempt({
        "schema_version": 1,
        "project": "project",
        "run_id": "run-s0",
        "attempt_id": "attempt-002",
        "created_at": "2026-07-12T00:01:00Z",
        "backend": "test-backend",
        "backend_job_id": None,
        "source_id": manifest["source_id"],
        "image_id": manifest["image_id"],
        "command": manifest["command"],
        "resources": manifest["resources"],
        "resume_from": None,
    })
    with pytest.raises(ValueError, match="root mirror conflicts"):
        store.initialize_attempt_records("attempt-002")


def test_atomic_create_and_event_idempotency_fail_closed(tmp_path):
    immutable = tmp_path / "immutable.json"
    atomic_create(immutable, {"value": 1})
    with pytest.raises(FileExistsError):
        atomic_create(immutable, {"value": 2})

    events = tmp_path / "events.jsonl"
    assert append_event_once(events, {"event": "created"}, "event-1") is True
    assert append_event_once(events, {"event": "created"}, "event-1") is False
    assert len(events.read_text().splitlines()) == 1


def test_secret_bearing_command_arguments_are_redacted():
    assert sanitize_command([
        "train", "API_TOKEN=secret", "--password", "hidden", "seed=7",
        "tokenizer_path=tokenizer.model", "train_batch_tokens=65536",
    ]) == [
        "train", "API_TOKEN=<redacted>", "--password", "<redacted>", "seed=7",
        "tokenizer_path=tokenizer.model", "train_batch_tokens=65536",
    ]


def test_legacy_manifest_is_observable_but_not_mutable(tmp_path):
    legacy = {"project": "project", "run_id": "legacy", "attempt_id": "attempt-001"}
    (tmp_path / "control_manifest.yaml").write_text(
        "project: project\nrun_id: legacy\n", encoding="utf-8"
    )
    attempt_dir = tmp_path / "attempts" / "attempt-001"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "control_attempt.yaml").write_text(
        "project: project\nrun_id: legacy\nattempt_id: attempt-001\n",
        encoding="utf-8",
    )
    store = ExperimentStateStore(tmp_path)
    assert store.load_manifest()["run_id"] == "legacy"
    assert store.load_attempt("attempt-001")["attempt_id"] == "attempt-001"
    with pytest.raises(ValueError, match="observation-only"):
        store.ensure_manifest(legacy)


def test_manifest_and_attempt_publication_fail_closed_on_identity_drift(tmp_path):
    store = ExperimentStateStore(tmp_path)
    with pytest.raises(FileNotFoundError, match="run manifest"):
        store.load_manifest()
    with pytest.raises(FileNotFoundError, match="before the run manifest"):
        store.create_attempt({"attempt_id": "attempt-001"})

    original = store.ensure_manifest(run_manifest(tmp_path))
    assert store.ensure_manifest({**original, "created_at": "later"}) == original
    with pytest.raises(ValueError, match="existing run manifest conflicts"):
        store.ensure_manifest({**original, "source_id": "different-source"})
    with pytest.raises(ValueError, match="attempt_id=.*invalid"):
        store.create_attempt(attempt_manifest(store, "../escape"))
    with pytest.raises(ValueError, match="attempt identity conflicts"):
        store.create_attempt(attempt_manifest(store, project="other"))
    with pytest.raises(FileNotFoundError, match="attempt manifest"):
        store.load_attempt("attempt-missing")


def test_state_records_reject_non_objects_and_attempt_identity_drift(tmp_path):
    store = prepare_store(tmp_path)
    store.backend_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="state record is not an object"):
        store.load_backend()

    store = prepare_store(tmp_path / "canonical")
    backend = json.loads(store.attempt_backend_path("attempt-001").read_text())
    backend["attempt_id"] = "attempt-other"
    store.attempt_backend_path("attempt-001").write_text(json.dumps(backend))
    with pytest.raises(ValueError, match="backend record conflicts"):
        store.load_backend("attempt-001")

    status = json.loads(store.attempt_status_path("attempt-001").read_text())
    status["attempt_id"] = "attempt-other"
    store.attempt_status_path("attempt-001").write_text(json.dumps(status))
    with pytest.raises(ValueError, match="status record conflicts"):
        store.load_status_payload("attempt-001")


def test_root_mirrors_must_select_one_current_attempt(tmp_path):
    store = prepare_store(tmp_path)
    backend = json.loads(store.backend_path.read_text())
    backend["attempt_id"] = "attempt-other"
    store.backend_path.write_text(json.dumps(backend), encoding="utf-8")
    with pytest.raises(ValueError, match="root backend/status mirrors disagree"):
        store.write_status_payload("attempt-001", {"state": "RUNNING"})


def test_noncurrent_attempt_updates_do_not_replace_current_root_mirror(tmp_path):
    store = prepare_store(tmp_path)
    store.create_attempt(attempt_manifest(store))
    store.initialize_attempt_records("attempt-002")

    updated = store.write_status_payload("attempt-001", {"state": "FAILED"})
    assert updated == {"attempt_id": "attempt-001", "state": "FAILED"}
    assert store.load_status_payload("attempt-001")["state"] == "FAILED"
    assert store.load_status_payload()["attempt_id"] == "attempt-002"
    with pytest.raises(ValueError, match="status payload conflicts"):
        store.write_status_payload(
            "attempt-001", {"attempt_id": "attempt-002", "state": "FAILED"}
        )


def test_attempt_manifest_without_derived_records_reads_as_created(tmp_path):
    store = ExperimentStateStore(tmp_path)
    store.ensure_manifest(run_manifest(tmp_path))
    store.create_attempt(attempt_manifest(store, "attempt-001"))
    assert store.read_status("attempt-001").state is RunState.CREATED
    assert store.load_backend("attempt-001") is None


def test_submission_rejects_wrong_identity_unsafe_payload_and_conflicts(tmp_path):
    store = prepare_store(tmp_path)
    with pytest.raises(ValueError, match="submission identity conflicts"):
        store.begin_submission(
            project="other", run_id="run-s0", attempt_id="attempt-001",
            backend="test-backend", request={},
        )
    with pytest.raises(TypeError, match="cannot serialize submission request"):
        store.begin_submission(
            project="project", run_id="run-s0", attempt_id="attempt-001",
            backend="test-backend", request={"opaque": object()},
        )

    store.begin_submission(
        project="project", run_id="run-s0", attempt_id="attempt-001",
        backend="test-backend", request={"argv": ["submit", "one"]},
    )
    with pytest.raises(ValueError, match="existing submission intent conflicts in request"):
        store.begin_submission(
            project="project", run_id="run-s0", attempt_id="attempt-001",
            backend="test-backend", request={"argv": ["submit", "two"]},
        )


def test_reconcile_requires_an_intent_and_nonempty_scheduler_identity(tmp_path):
    store = prepare_store(tmp_path)
    kwargs = {
        "project": "project", "run_id": "run-s0", "attempt_id": "attempt-001",
    }
    with pytest.raises(ValueError, match="backend_job_id must not be empty"):
        store.reconcile_submission(**kwargs, backend_job_id="")
    with pytest.raises(FileNotFoundError, match="before submission intent"):
        store.reconcile_submission(**kwargs, backend_job_id="job-1")


def test_transition_without_id_appends_redacted_event_and_exit_code(tmp_path):
    store = prepare_store(tmp_path)
    status = store.transition(
        project="project", run_id="run-s0", attempt_id="attempt-001",
        state=RunState.FAILED, event="worker_failed",
        payload={"api_token": "secret", "reason": "model"}, exit_code=9,
    )
    assert status.exit_code == 9
    event = json.loads(store.events_path.read_text().splitlines()[-1])
    assert event["event"] == "worker_failed"
    assert event["payload"] == {"api_token": "<redacted>", "reason": "model"}
    assert "event_id" not in event


def test_event_idempotency_ignores_interrupted_invalid_json_line(tmp_path):
    events = tmp_path / "events.jsonl"
    events.write_text('{"event_id": "interrupted"\n', encoding="utf-8")
    assert append_event_once(events, {"event": "created"}, "event-1") is True
    assert json.loads(events.read_text().splitlines()[-1])["event_id"] == "event-1"


def test_research_contract_is_part_of_run_identity(tmp_path):
    manifest = run_manifest(
        tmp_path,
        research_contract={"metric": "loss", "direction": "minimize"},
        research_role="baseline",
    )
    assert manifest["research_contract"]["metric"] == "loss"
    assert manifest["research_role"] == "baseline"
    assert comparable_manifest({"resolved_config": "legacy"}) == {
        "resolved_config": "legacy"
    }


def test_atomic_write_removes_temporary_file_when_publication_fails(
    tmp_path, monkeypatch,
):
    target = tmp_path / "status.json"
    temporary = None

    def fail_replace(source, destination):
        nonlocal temporary
        temporary = source
        raise OSError("simulated rename failure")

    monkeypatch.setattr(manifest_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="rename failure"):
        atomic_write(target, {"state": "RUNNING"})
    assert temporary is not None
    assert not manifest_module.os.path.exists(temporary)
    assert not target.exists()


def test_submission_request_preserves_json_primitives(tmp_path):
    store = prepare_store(tmp_path)
    intent = store.begin_submission(
        project="project", run_id="run-s0", attempt_id="attempt-001",
        backend="test-backend",
        request={"none": None, "flag": True, "count": 3, "ratio": 0.5},
    )
    assert intent["request"] == {
        "none": None, "flag": True, "count": 3, "ratio": 0.5,
    }


def test_snapshot_repairs_missing_canonical_records_and_skips_missing_root(tmp_path):
    store = prepare_store(tmp_path)
    store.attempt_backend_path("attempt-001").unlink()
    store.attempt_status_path("attempt-001").unlink()
    store.backend_path.unlink()
    store.create_attempt(attempt_manifest(store))

    store.initialize_attempt_records("attempt-002")

    assert not store.attempt_backend_path("attempt-001").exists()
    repaired = json.loads(store.attempt_status_path("attempt-001").read_text())
    assert repaired["attempt_id"] == "attempt-001"


def test_explicit_attempt_load_falls_back_to_matching_root_records(tmp_path):
    store = prepare_store(tmp_path)
    expected_backend = store.load_backend()
    expected_status = store.load_status_payload()
    store.attempt_backend_path("attempt-001").unlink()
    store.attempt_status_path("attempt-001").unlink()
    assert store.load_backend("attempt-001") == expected_backend
    assert store.load_status_payload("attempt-001") == expected_status


def test_reinitializing_attempt_reuses_existing_derived_records(tmp_path):
    store = prepare_store(tmp_path)
    before = store.read_status("attempt-001")
    after = store.initialize_attempt_records("attempt-001")
    assert after == before


def test_manifest_publication_recovers_when_concurrent_writer_wins(
    tmp_path, monkeypatch,
):
    store = ExperimentStateStore(tmp_path)
    candidate = run_manifest(tmp_path)
    original_create = manifest_module.atomic_create
    calls = 0

    def concurrent_create(path, payload, **kwargs):
        nonlocal calls
        calls += 1
        original_create(path, payload, **kwargs)
        raise FileExistsError(path)

    monkeypatch.setattr(manifest_module, "atomic_create", concurrent_create)
    assert store.ensure_manifest(candidate) == candidate
    assert calls == 1


def test_submission_publication_recovers_when_concurrent_writer_wins(
    tmp_path, monkeypatch,
):
    store = prepare_store(tmp_path)
    original_create = manifest_module.atomic_create
    calls = 0

    def concurrent_create(path, payload, **kwargs):
        nonlocal calls
        calls += 1
        original_create(path, payload, **kwargs)
        raise FileExistsError(path)

    monkeypatch.setattr(manifest_module, "atomic_create", concurrent_create)
    intent = store.begin_submission(
        project="project", run_id="run-s0", attempt_id="attempt-001",
        backend="test-backend", request={"argv": ["submit"]},
    )
    assert intent["state"] == "SUBMITTING"
    assert calls == 1
