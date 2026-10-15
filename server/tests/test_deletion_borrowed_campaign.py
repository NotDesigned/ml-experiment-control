"""Cancellation reuses a submission Campaign without creating a new consumer."""
import hashlib

import pytest
import yaml

from tests.test_run_deletion_service import deletion_env, prepare

REF = {"run_id": "run-a", "attempt_id": "attempt-001", "checkpoint_id": "checkpoint.original"}


def frozen_action(env, aid, operation, *, run="consumer", attempt="attempt-001", resume=None,
                  borrowed=None, summary=None):
    directory = env.runtime.action_store.directory(aid)
    directory.mkdir(parents=True, exist_ok=True)
    path = borrowed or directory / "campaign.execution.yml"
    if borrowed is None:
        definition = {"run_id": run}
        if resume is not None:
            definition["resume_from"] = resume
        path.write_text(yaml.safe_dump({"schema_version": 1, "project": "demo",
                                       "campaign": "run-" + run, "runs": [definition]}))
    attempt_scope = operation in {"CANCEL_RUN", "RETRY_ATTEMPT"}
    plan = {"action_id": aid, "request_digest": aid, "ready": True, "operation": operation,
            "run_id": run, "attempt_id": attempt, "backend_job_id": "123",
            "scope": {"project": "demo", "scope_type": "attempt" if attempt_scope else "run",
                      "object_id": run + "::attempt-001" if attempt_scope else run},
            "execution_campaign_file": str(path),
            "execution_campaign_sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            "preflight_summary": summary or {}}
    saved = env.runtime.action_store.save_plan(plan)
    env.runtime.action_store.set_execution(aid, {**saved["execution"], "status": "VERIFIED", "result": {
        "submission": {"run_id": run, "attempt_id": attempt, "backend_job_id": "123"}}}, event="isolated_completed_action")
    return path


def test_real_cancel_plan_borrows_original_submission_campaign_without_blocking_unrelated_run(deletion_env):
    env = deletion_env
    env.add_run("cancel-collect")
    source = frozen_action(env, "action-" + "1" * 16, "SUBMIT_RUN", run="cancel-collect")
    frozen_action(env, "action-" + "2" * 16, "CANCEL_RUN", run="cancel-collect", borrowed=source)
    plan = prepare(env)
    assert plan["ready"]
    assert plan["target_manifest"]["dependencies"] == []
    assert source.is_file()
    assert not env.service.fence.is_deleted("demo", "run-a")


def test_cancel_summary_scientific_reference_remains_protected(deletion_env):
    env = deletion_env
    env.add_run("consumer")
    source = frozen_action(env, "action-" + "1" * 16, "SUBMIT_RUN")
    frozen_action(env, "action-" + "2" * 16, "CANCEL_RUN", borrowed=source,
                  summary={"run_id": "consumer", "resolved_config": {"resume_from": REF}})
    plan = prepare(env)
    assert not plan["ready"]
    assert plan["target_manifest"]["dependencies"] == [
        {"run_id": "consumer", "kind": "action_resume", "checkpoint_id": REF["checkpoint_id"]}]


def test_real_retry_and_borrowed_cancel_keep_original_resume_dependency_once(deletion_env):
    env = deletion_env
    env.add_run("consumer")
    source = frozen_action(env, "action-" + "1" * 16, "SUBMIT_RUN", resume=REF)
    frozen_action(env, "action-" + "2" * 16, "CANCEL_RUN", borrowed=source)
    frozen_action(env, "action-" + "3" * 16, "RETRY_ATTEMPT", attempt="attempt-002", resume=REF,
                  summary={"run_id": "consumer", "resolved_config": {"resume_from": REF}})
    plan = prepare(env)
    assert not plan["ready"]
    assert plan["target_manifest"]["dependencies"] == [
        {"run_id": "consumer", "kind": "action_resume", "checkpoint_id": REF["checkpoint_id"]}]


@pytest.mark.parametrize("operation", ["SUBMIT_RUN", "RETRY_ATTEMPT", "RUN_EVALUATION"])
def test_each_submission_consumer_remains_bound_to_its_own_action_directory(deletion_env, operation):
    env = deletion_env
    env.add_run("consumer")
    source = frozen_action(env, "action-" + "1" * 16, "SUBMIT_RUN")
    # Even a matching real Campaign digest cannot authorize a consumer to borrow
    # another Action's path. Only cancellation has that existing contract.
    frozen_action(env, "action-" + "2" * 16, operation, borrowed=source)
    with pytest.raises(ValueError, match="outside its Action"):
        prepare(env)


@pytest.mark.parametrize("operation", ["LEGACY_UNKNOWN_SUBMIT", None])
def test_unknown_frozen_campaign_consumer_never_silently_disappears(deletion_env, operation):
    env = deletion_env
    env.add_run("consumer")
    frozen_action(env, "action-" + "1" * 16, operation, resume=REF)
    with pytest.raises(ValueError, match="unknown frozen Action Campaign consumer"):
        prepare(env)
