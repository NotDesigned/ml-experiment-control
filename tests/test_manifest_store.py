import json

import pytest

from experiment_control.manifest import (
    ExperimentStateStore,
    RunState,
    append_event_once,
    atomic_create,
    sanitize_command,
)
from experiment_control.run_manifest import build_run_manifest, comparable_manifest


def run_manifest(tmp_path):
    return build_run_manifest(
        project="project",
        run_id="run-s0",
        created_at="2026-07-12T00:00:00Z",
        config_path="config.yml",
        resolved_config={"seed": 7, "resume": "/old/checkpoint"},
        source_id="source-immutable",
        runtime_tree_id="runtime-immutable",
        git_commit="deadbeef",
        campaign_id="campaign-digest",
        campaign="campaign",
        image_id="sha256:" + "a" * 64,
        run_dir=str(tmp_path),
        max_infra_retries=1,
        backend={"kind": "test-backend"},
        resources={"gpus": 1},
        storage={"run_dir": str(tmp_path), "checkpoint_dir": str(tmp_path)},
        command=["python", "train.py"],
        execution={"source_mount": "/workspace", "workdir": "/workspace"},
    )


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
    ]) == [
        "train", "API_TOKEN=<redacted>", "--password", "<redacted>", "seed=7",
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
