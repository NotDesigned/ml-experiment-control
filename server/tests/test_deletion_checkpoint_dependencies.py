"""Checkpoint consumers remain protected after authored Campaign changes."""
import json

import pytest
import yaml

from ml_exp_server.storage import atomic_json
from tests.test_run_deletion_service import deletion_env, prepare

REF = {"run_id": "run-a", "attempt_id": "attempt-001", "checkpoint_id": "checkpoint.original"}


def write(path, value):
    path.write_text(yaml.safe_dump(value))


def action(env, *, summary=None, frozen=None, project="demo"):
    aid = "action-" + "c" * 16
    root = env.runtime.action_store.directory(aid)
    root.mkdir(parents=True, exist_ok=True)
    value = {"action_id": aid, "request_digest": "fixture", "ready": True, "operation": "SUBMIT_RUN",
             "scope": {"project": project, "scope_type": "run", "object_id": "consumer"}, "run_id": "consumer"}
    if summary is not None: value["preflight_summary"] = summary
    if frozen is not None:
        path = root / "campaign.execution.yml"; write(path, frozen)
        value["execution_campaign_file"] = str(path)
    env.runtime.action_store.save_plan(value)


def test_canonical_restore_reference_survives_authored_campaign_removal(deletion_env):
    env = deletion_env
    env.add_run("consumer", resume_from=REF)
    manifest = env.local["consumer"] / "manifest.yaml"
    write(manifest, {"project": "demo", "run_id": "consumer", "resolved_config": {"resume_from": REF}})
    current = yaml.safe_load(env.definitions["consumer"].read_text())
    current["runs"][0].pop("resume_from"); write(env.definitions["consumer"], current)
    plan = prepare(env)
    assert not plan["ready"]
    assert plan["target_manifest"]["dependencies"] == [{
        "run_id": "consumer", "kind": "canonical_resume", "checkpoint_id": REF["checkpoint_id"]}]
    env.service.fence.mark("demo", "consumer")
    assert prepare(env, key="consumer-retired")["ready"]


@pytest.mark.parametrize("source", ["direct-manifest", "legacy-manifest", "attempt", "legacy-attempt", "resolved-attempt"])
def test_scientific_manifest_locations_block_exact_original_checkpoint(deletion_env, source):
    env = deletion_env; env.add_run("consumer")
    local = env.local["consumer"]
    value = {"project": "demo", "run_id": "consumer", "resume_from": REF}
    if source == "direct-manifest": target = local / "manifest.yaml"
    elif source == "legacy-manifest": target = local / "control_manifest.yaml"
    elif source == "legacy-attempt": target = local / "attempts/attempt-001/control_attempt.yaml"
    else: target = local / "attempts/attempt-001/attempt.yaml"
    if source == "resolved-attempt": value.update(resume_from="checkpoint.original", resolved_config={"resume_from": REF})
    write(target, value)
    assert not prepare(env)["ready"]
    assert env.service._dependencies("demo", "run-a")[0]["run_id"] == "consumer"


@pytest.mark.parametrize("source", ["frozen-action", "action-summary"])
def test_frozen_action_keeps_dependency_without_current_authored_reference(deletion_env, source):
    env = deletion_env; env.add_run("consumer")
    if source == "frozen-action": action(env, frozen={"project": "demo", "runs": [{"run_id": "consumer", "resume_from": REF}]})
    else: action(env, summary={"run_id": "consumer", "resolved_config": {"resume_from": REF}})
    assert env.service._dependencies("demo", "run-a") == [{
        "run_id": "consumer", "kind": "action_resume", "checkpoint_id": REF["checkpoint_id"]}]
    assert not prepare(env)["ready"]


def test_same_consumer_generation_is_deduplicated_across_all_sources(deletion_env):
    env = deletion_env; env.add_run("consumer", resume_from=REF)
    write(env.local["consumer"] / "manifest.yaml", {"project": "demo", "run_id": "consumer", "resume_from": REF})
    write(env.local["consumer"] / "attempts/attempt-001/attempt.yaml", {
        "project": "demo", "run_id": "consumer", "resume_from": REF, "resolved_config": {"resume_from": REF}})
    atomic_json(env.root / "experiment-preparations/demo/preparation.json", {
        "project": "demo", "run_id": "consumer", "request": {"run": {"resume_from": REF}}})
    action(env, summary={"run_id": "consumer", "resolved_config": {"resume_from": REF}})
    assert len(env.service._dependencies("demo", "run-a")) == 1


def test_generic_yml_campaign_keeps_same_project_restore_reference(deletion_env):
    env = deletion_env
    path = env.definitions["run-a"].parent / "authored.yml"
    write(path, {"project": "demo", "runs": [{"run_id": "consumer", "resume_from": REF}]})
    assert env.service._dependencies("demo", "run-a")[0]["run_id"] == "consumer"


def test_other_projects_self_references_and_arbitrary_log_json_are_not_dependencies(deletion_env):
    env = deletion_env; env.add_run("consumer")
    write(env.local["consumer"] / "manifest.yaml", {"project": "other", "run_id": "consumer", "resume_from": REF})
    write(env.local["run-a"] / "manifest.yaml", {"project": "demo", "run_id": "run-a", "resume_from": REF})
    attempt = env.local["consumer"] / "attempts/attempt-001"
    write(attempt / "attempt.yaml", {"project": "demo", "run_id": "consumer", "resume_from": "checkpoint.original",
        "logs": {"resume_from": REF}, "message": json.dumps({"resume_from": REF})})
    write(attempt / "unrelated.yaml", {"project": "demo", "run_id": "consumer", "resume_from": REF})
    action(env, project="other", summary={"run_id": "consumer", "resolved_config": {"resume_from": REF}})
    atomic_json(env.root / "experiment-preparations/demo/other.json", {"project": "other", "run_id": "consumer", "request": {"run": {"resume_from": REF}}})
    assert env.service._dependencies("demo", "run-a") == []


def test_foreign_checkpoint_scope_invalid_generation_and_action_path_fail_closed(deletion_env, monkeypatch):
    env = deletion_env; env.add_run("consumer")
    path = env.local["consumer"] / "manifest.yaml"
    write(path, {"project": "demo", "run_id": "consumer", "resume_from": {**REF, "project": "other"}})
    assert env.service._dependencies("demo", "run-a") == []
    write(path, {"project": "demo", "run_id": "consumer", "resume_from": {**REF, "checkpoint_id": []}})
    with pytest.raises(ValueError, match="dependency identity"): env.service._dependencies("demo", "run-a")
    write(path, {"project": "demo", "run_id": "consumer"})
    monkeypatch.setattr(env.runtime.action_store, "list_all", lambda: [{
        "action_id": "action-" + "d" * 16, "scope": {"project": "demo"}, "run_id": "consumer",
        "execution_campaign_file": str(env.definitions["consumer"])}])
    with pytest.raises(ValueError, match="outside its Action"): env.service._dependencies("demo", "run-a")


def test_foreign_campaign_documents_do_not_expand_the_deletion_dependency_scope(deletion_env):
    env = deletion_env; env.add_run("consumer")
    foreign = {"project": "other", "runs": [{"run_id": "consumer", "resume_from": REF}]}
    write(env.definitions["run-a"].parent / "foreign.yml", foreign)
    action(env, frozen=foreign)
    assert env.service._dependencies("demo", "run-a") == []
