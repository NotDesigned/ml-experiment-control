"""Run purge refuses missing original Attempts and mismatched scheduler evidence."""
import json
import shutil
from types import SimpleNamespace

import pytest

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.schemas import AttemptSummary, RunIndexRow
from ml_exp_server.storage import atomic_json
from tests.test_run_deletion_service import deletion_env, prepare, publish


def indexed(env, **fields):
    attempt = {"attempt_id": "attempt-001", "state": "SUCCEEDED", **fields}
    env.runtime.index.upsert_run(RunIndexRow(project="demo", run_id="run-a", run_dir=str(env.local["run-a"]),
        attempts=[AttemptSummary(**attempt)]))


def saved_action(env, *, job=None):
    plan = {"action_id": "action-" + "b" * 16, "request_digest": "fake", "ready": True,
            "scope": {"project": "demo", "scope_type": "attempt", "object_id": "run-a::attempt-001"},
            "operation": "SUBMIT_RUN", "run_id": "run-a", "attempt_id": "attempt-001"}
    snapshot = env.runtime.action_store.save_plan(plan)
    if job is not None:
        execution = snapshot["execution"]
        execution.update(status="VERIFIED", result={"submission": {
            "run_id": "run-a", "attempt_id": "attempt-001", "backend_job_id": job}})
        env.runtime.action_store.set_execution(plan["action_id"], execution, event="fixture")


@pytest.mark.parametrize("source", ["index", "transfer", "action", "root-mirror"])
def test_missing_canonical_attempt_blocks_every_independent_history_source(deletion_env, source):
    env = deletion_env
    if source == "index": indexed(env, backend_job_id="123", has_submission=True)
    elif source == "transfer": env.runtime.storage_service.objects.issue("demo", "run-a", "attempt-001", env.local["run-a"], ["**/*"])
    elif source == "action": saved_action(env)
    else: atomic_json(env.local["run-a"] / "backend.json", {"attempt_id": "attempt-001", "backend_job_id": "123"})
    shutil.rmtree(env.local["run-a"] / "attempts/attempt-001")
    with pytest.raises(ApplicationError) as error: prepare(env)
    assert error.value.code == "RUN_DELETION_ATTEMPT_EVIDENCE_INCOMPLETE"
    assert not env.backend.calls and not env.deleted and not env.service.fence.is_deleted("demo", "run-a")


@pytest.mark.parametrize("source", ["index", "action", "root-mirror", "outbox"])
def test_differing_known_job_cannot_replace_canonical_attempt(deletion_env, source):
    env = deletion_env
    if source == "index": indexed(env, backend_job_id="different-job")
    elif source == "action": saved_action(env, job="different-job")
    elif source == "root-mirror": atomic_json(env.local["run-a"] / "backend.json", {"attempt_id": "attempt-001", "backend_job_id": "different-job"})
    else: atomic_json(env.local["run-a"] / "attempts/attempt-001/submission.json", {"backend_job_id": "different-job"})
    with pytest.raises(ApplicationError) as error: prepare(env)
    assert error.value.code == "RUN_DELETION_ATTEMPT_EVIDENCE_INCOMPLETE" and not env.deleted


def test_canonical_manifest_missing_run_directory_and_identity_changes_are_refused(deletion_env):
    env = deletion_env
    path = env.local["run-a"] / "attempts/attempt-001/attempt.yaml"
    original = path.read_bytes(); path.unlink()
    with pytest.raises(ApplicationError, match="manifest"): prepare(env)
    path.write_bytes(original)
    atomic_json(path.parent / "status.json", {"state": "SUCCEEDED", "attempt_id": "attempt-002"})
    with pytest.raises(ValueError, match="provenance"): env.service._attempts(env.local["run-a"])
    shutil.rmtree(env.local["run-a"])
    with pytest.raises(ApplicationError, match="canonical Run"): prepare(env)


def diagnostics(env, *, uid="a" * 32, job="123", events=()):
    objects = env.runtime.storage_service.objects
    objects.issue("demo", "run-a", "attempt-001", env.local["run-a"], ["**/*"])
    with objects.record("demo", "run-a", "attempt-001") as (path, value):
        value["training_diagnostics"] = {"scheduler_name": job, "job_uid": uid, "events": list(events)}
        atomic_json(path, value)
    atomic_json(env.local["run-a"] / "attempts/attempt-001/backend.json", {"backend_job_id": "123"})


def test_training_uid_comes_from_exact_artifact_transfer_not_obsolete_side_file(deletion_env):
    env = deletion_env
    uid = "a" * 32
    diagnostics(env, uid=uid, events=[{"reason": "SuccessfulDelete", "message": 'Deleted pod "pt-' + uid + '-worker-0"'}])
    result = env.service._attempts(env.local["run-a"])
    assert result[0]["backend_job_uid"] == uid and result[0]["worker_deleted"] is True
    obsolete = env.root / "result-collections/demo/run-a/attempt-001.training-diagnostics.json"
    atomic_json(obsolete, {"scheduler_name": "123", "job_uid": "replacement"})
    assert env.service._attempts(env.local["run-a"])[0]["backend_job_uid"] == uid
    atomic_json(env.local["run-a"] / "attempts/attempt-001/backend.json", {"backend_job_id": "123", "job_uid": "original"})
    with pytest.raises(ApplicationError, match="UID evidence conflicts"): prepare(env)


@pytest.mark.parametrize("event,job,expected", [
    ({"reason": "SuccessfulDelete", "message": 'Deleted pod "pt-' + "a" * 32 + '-worker-0"'}, "123", True),
    ({"reason": "SuccessfulDelete", "message": 'Deleted pod "pt-' + "b" * 32 + '-worker-0"'}, "123", False),
    ({"reason": "SuccessfulDelete", "message": "the UID " + "a" * 32 + " is mentioned"}, "123", False),
    ({"reason": "SuccessfulDelete", "message": "pt-" + "a" * 32 + "-worker-0-other"}, "123", False),
    ({"reason": "JobRunning", "message": "pt-" + "a" * 32 + "-worker-0"}, "123", False),
    ({"reason": "SuccessfulDelete", "message": "pt-" + "a" * 32 + "-worker-0"}, "other-job", False),
    ("not-an-event", "123", False),
])
def test_worker_deleted_requires_exact_original_job_uid_and_pod(deletion_env, event, job, expected):
    service = deletion_env.service
    value = {"scheduler_name": "123", "job_uid": "a" * 32, "events": [event]}
    assert service._worker_deleted(value, job, "a" * 32) is expected


def test_transfer_diagnostics_cannot_invent_a_different_original_job(deletion_env):
    env = deletion_env
    diagnostics(env, job="replacement")
    with pytest.raises(ApplicationError, match="scheduler jobs differ"): prepare(env)
    objects = env.runtime.storage_service.objects
    with objects.record("demo", "run-a", "attempt-001") as (path, value):
        value["run_dir"] = "/different"; atomic_json(path, value)
    with pytest.raises(ValueError, match="provenance"): prepare(env)


@pytest.mark.parametrize("state", ["RUNNING", "STARTING", "UNKNOWN", "SUBMITTING"])
def test_cached_nonterminal_gate_blocks_plan_without_provider_call(deletion_env, state):
    env = deletion_env
    atomic_json(env.local["run-a"] / "attempts/attempt-001/status.json", {"state": state})
    plan = prepare(env)
    assert not plan["ready"] and plan["execution"]["status"] == "BLOCKED" and not env.backend.calls
    assert next(gate for gate in plan["gates"] if gate["name"] == "original_attempts_terminal")["status"] == "FAIL"


def test_created_unsubmitted_attempt_is_not_confused_with_submission_intent(deletion_env):
    env = deletion_env
    path = env.local["run-a"] / "attempts/attempt-001"
    atomic_json(path / "backend.json", {"backend": "slurm", "backend_job_id": None, "attempt_id": "attempt-001"})
    atomic_json(path / "status.json", {"state": "CREATED"})
    plan = prepare(env)
    attempt = plan["target_manifest"]["attempts"][0]
    assert plan["ready"] and attempt["state"] == "NOT_SUBMITTED" and not attempt["has_submission"]
    indexed(env, has_submission=True)
    blocked = prepare(env, key="uncertain-index")
    assert not blocked["ready"] and blocked["target_manifest"]["attempts"][0]["has_submission"]


def test_cached_sensecore_requires_durable_uid_before_approval(deletion_env):
    env = deletion_env
    env.backend.callback = True
    atomic_json(env.local["run-a"] / "attempts/attempt-001/backend.json", {"backend_job_id": "123"})
    plan = prepare(env)
    assert not plan["ready"]
    assert next(gate for gate in plan["gates"] if gate["name"] == "original_job_identities_known")["status"] == "FAIL"
    diagnostics(env)
    assert prepare(env, key="durable-uid")["ready"] and not env.backend.calls


@pytest.mark.parametrize("kind", ["link", "file"])
def test_attempt_inventory_refuses_noncanonical_directory_nodes(deletion_env, kind):
    env = deletion_env
    path = env.local["run-a"] / "attempts/attempt-002"
    if kind == "link": path.symlink_to(env.local["run-a"], target_is_directory=True)
    else: path.write_text("not an Attempt directory")
    with pytest.raises(ValueError, match="canonical Attempt directory"): env.service._attempts(env.local["run-a"])


def test_nonstring_recorded_scheduler_identity_fails_before_provider_lookup(deletion_env, monkeypatch):
    env = deletion_env
    monkeypatch.setattr(env.runtime.index, "get_run", lambda *args: SimpleNamespace(
        run_dir=str(env.local["run-a"]), attempts=[{"attempt_id": "attempt-001", "backend_job_id": 123}]))
    with pytest.raises(ValueError, match="scheduler job identity"): env.service._attempts(env.local["run-a"])
    assert not env.backend.calls


def test_action_scope_only_identity_legacy_result_and_invalid_observation(deletion_env, monkeypatch):
    env = deletion_env
    action = {"scope": {"project": "demo", "scope_type": "attempt", "object_id": "run-a::attempt-001"},
              "execution": {"result": ["historical nonmapping result"]}}
    monkeypatch.setattr(env.runtime.action_store, "list_all", lambda: [action])
    assert env.service._attempts(env.local["run-a"])[0]["backend_job_id"] == "123"
    action["execution"]["result"] = {"observation": {"run_id": "different-run", "backend_job_id": "123"}}
    with pytest.raises(ValueError, match="Action scheduler evidence"): env.service._attempts(env.local["run-a"])


def test_malformed_diagnostics_are_not_interpreted_as_original_uid_evidence(deletion_env):
    env = deletion_env
    diagnostics(env)
    objects = env.runtime.storage_service.objects
    with objects.record("demo", "run-a", "attempt-001") as (path, value):
        value["training_diagnostics"] = ["malformed"]; atomic_json(path, value)
    with pytest.raises(ValueError, match="training diagnostics"): env.service._attempts(env.local["run-a"])
