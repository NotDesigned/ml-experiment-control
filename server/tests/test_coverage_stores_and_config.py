"""Focused branch coverage for durable stores and authored configuration failures."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
import yaml

from ml_exp_server.actions.store import ActionStore
from ml_exp_server.project_config import ConfigError, load_server_config, load_projects, load_research_project, load_research_question
from ml_exp_server.schemas import OperationScope, OperationScopeType, ServerConfig, ProjectRef
from ml_exp_server.storage import StorageError, atomic_json, atomic_text, read_json


def scope(object_id: str = "demo") -> OperationScope:
    return OperationScope(project="demo", scope_type=OperationScopeType.PROJECT, object_id=object_id)


def test_action_store_is_immutable_claimed_once_and_fails_closed_on_corruption(tmp_path):
    store = ActionStore(tmp_path / "actions")
    operation_scope = scope()
    action_id = store.action_id(operation_scope, "intent-a")
    assert action_id == store.action_id(operation_scope, "intent-a")
    for invalid in ("wrong", "action-bad/name", "action-"):
        with pytest.raises(ValueError, match="invalid action_id"):
            store.directory(invalid)

    plan = {
        "action_id": action_id,
        "scope": operation_scope.model_dump(mode="json"),
        "ready": False,
        "operation": "BLOCKED_TEST",
    }
    saved = store.save_plan(plan)
    assert saved["execution"]["status"] == "BLOCKED"
    assert store.save_plan({**plan, "operation": "MUTATED"})["operation"] == "BLOCKED_TEST"

    store.claim_execution(action_id)
    with pytest.raises(RuntimeError, match="already been claimed"):
        store.claim_execution(action_id)

    directory = store.directory(action_id)
    (directory / "journal.jsonl").write_text(
        "not-json\n" + json.dumps(["not", "mapping"]) + "\n" +
        json.dumps({"event": "valid"}) + "\n", encoding="utf-8",
    )
    with pytest.raises(StorageError, match="invalid complete record"):
        store.snapshot(action_id)
    (directory / "journal.jsonl").write_text(
        json.dumps({"event": "valid"}) + "\n", encoding="utf-8",
    )
    repaired = store.snapshot(action_id)["journal"]
    assert repaired[0] == {"event": "valid"}
    assert repaired[-1]["event"] == "action_prepared"
    assert repaired[-1]["revision"] == 1
    updated = store.set_execution(
        action_id, {
            **store.execution(action_id), "status": "FAILED", "error": "boom",
        }, event="failed",
    )
    assert updated["execution"]["status"] == "FAILED"
    (directory / "execution.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(StorageError, match="durable JSON is unreadable"):
        store.execution(action_id)
    with pytest.raises(StorageError, match="durable JSON is unreadable"):
        store.set_execution(
            action_id, {"revision": 2, "status": "FAILED", "error": "again"},
            event="failed",
        )

    foreign = scope("other")
    foreign_id = store.action_id(foreign, "intent-b")
    store.save_plan({
        "action_id": foreign_id, "scope": foreign.model_dump(mode="json"),
        "ready": True, "operation": "OTHER",
    })
    malformed = store.root / "action-malformed"
    malformed.mkdir()
    (malformed / "plan.json").write_text(json.dumps({
        "action_id": "not-valid", "scope": operation_scope.model_dump(mode="json"),
    }))
    with pytest.raises(StorageError, match="durable JSON is unreadable"):
        store.list_for_scope(operation_scope)
    atomic_json(directory / "execution.json", {"status": "FAILED", "error": "repaired"})
    assert [item["action_id"] for item in store.list_for_scope(operation_scope)] == [action_id]
    with pytest.raises(FileNotFoundError):
        store.snapshot("action-missing")

    missing = tmp_path / "missing.json"
    assert read_json(missing, {"fallback": True}) == {"fallback": True}
    missing.write_text("[]")
    assert read_json(missing, {}) == []


def test_action_store_recovers_plan_written_before_initial_execution(tmp_path):
    store = ActionStore(tmp_path / "actions")
    operation_scope = scope()
    action_id = store.action_id(operation_scope, "recover-initial-state")
    directory = store.directory(action_id)
    directory.mkdir(parents=True)
    plan = {
        "action_id": action_id,
        "scope": operation_scope.model_dump(mode="json"),
        "ready": True,
        "operation": "RECOVER_TEST",
    }
    (directory / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    recovered = store.snapshot(action_id)
    assert recovered["execution"]["status"] == "PREPARED"
    assert recovered["journal"][-1]["event"] == "action_prepared"


def test_begin_execution_uses_cas_state_if_claim_artifact_write_fails(monkeypatch, tmp_path):
    from ml_exp_server.actions import store as store_module

    store = ActionStore(tmp_path / "actions")
    action_id = store.action_id(scope(), "intent-execute")
    store.save_plan({
        "action_id": action_id, "scope": scope().model_dump(mode="json"),
        "ready": True, "operation": "SUBMIT_RUN", "request_digest": "sha256:req",
    })
    store.set_execution(
        action_id, {**store.execution(action_id), "status": "AUTHORIZED"},
        event="authorized",
    )
    real_atomic = store_module.atomic_json

    def fail_claim(path, payload):
        if path.name == "execution.claim":
            raise OSError("crash")
        return real_atomic(path, payload)

    monkeypatch.setattr(store_module, "atomic_json", fail_claim)
    store.begin_execution(
        action_id, {**store.execution(action_id), "status": "EXECUTING"},
        intent_digest="sha256:intent",
    )
    assert store.execution(action_id)["status"] == "EXECUTING"
    assert store.snapshot(action_id)["journal"][-1]["payload"][
        "intent_digest"
    ] == "sha256:intent"


def test_action_store_serializes_plan_creation_across_daemon_instances(tmp_path):
    root = tmp_path / "actions"
    stores = [ActionStore(root), ActionStore(root)]
    operation_scope = scope()
    action_id = stores[0].action_id(operation_scope, "shared-intent")
    barrier = Barrier(2)

    def save(index):
        barrier.wait(timeout=5)
        return stores[index].save_plan({
            "action_id": action_id,
            "scope": operation_scope.model_dump(mode="json"),
            "ready": True,
            "operation": f"PLAN_{index}",
        })

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save, range(2)))

    assert results[0]["operation"] == results[1]["operation"]
    assert ActionStore(root).snapshot(action_id)["operation"] == results[0]["operation"]

    prepared = ActionStore(root).execution(action_id)
    winner = ActionStore(root).set_execution(
        action_id, {**prepared, "status": "AUTHORIZED", "actor": "first"},
        event="authorized", expected_status="PREPARED",
    )
    assert winner["execution"]["actor"] == "first"
    with pytest.raises(RuntimeError, match="expected PREPARED, found AUTHORIZED"):
        ActionStore(root).set_execution(
            action_id, {**prepared, "status": "AUTHORIZED", "actor": "second"},
            event="authorized", expected_status="PREPARED",
        )

    first_pending = winner["execution"]
    stale_pending = dict(first_pending)
    ActionStore(root).set_execution(
        action_id, {**first_pending, "status": "RECONCILE_REQUIRED", "actor": "winner"},
        event="pending", expected_status="AUTHORIZED",
    )
    with pytest.raises(RuntimeError, match="expected revision"):
        ActionStore(root).set_execution(
            action_id, {
                **stale_pending, "status": "RECONCILE_REQUIRED", "actor": "stale",
            },
            event="pending", expected_status="RECONCILE_REQUIRED",
        )


def test_action_store_rejects_concurrent_different_request_digests(tmp_path):
    root = tmp_path / "actions"
    stores = [ActionStore(root), ActionStore(root)]
    action_id = stores[0].action_id(scope(), "shared-key")
    barrier = Barrier(2)

    def save(index):
        barrier.wait(timeout=5)
        try:
            stores[index].save_plan({
                "action_id": action_id, "scope": scope().model_dump(mode="json"),
                "ready": True, "operation": f"PLAN_{index}",
                "request_digest": f"sha256:{index}",
            })
            return "saved"
        except RuntimeError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save, range(2)))

    assert results.count("saved") == 1
    assert sum("already bound" in result for result in results) == 1


def test_atomic_json_fsync_path_is_private_and_cleans_failed_temporary(
    tmp_path, monkeypatch,
):
    target = tmp_path / "state" / "record.json"
    atomic_json(target, {"value": 1})
    assert (target.stat().st_mode & 0o777) == 0o600
    assert read_json(target, {}) == {"value": 1}

    monkeypatch.setattr(
        "ml_exp_server.storage.os.replace",
        lambda *_args: (_ for _ in ()).throw(OSError("disk failure")),
    )
    with pytest.raises(OSError, match="disk failure"):
        atomic_json(target, {"value": 2})
    assert read_json(target, {}) == {"value": 1}
    assert not list(target.parent.glob(".record.json.*.tmp"))


def test_atomic_text_writes_private_utf8_and_replaces(tmp_path):
    target = tmp_path / "campaign.yml"

    atomic_text(target, "project: elf\nvalue: 1.0e-08\n")
    atomic_text(target, "project: elf\nvalue: 2.0e-08\n")

    assert target.read_text(encoding="utf-8") == (
        "project: elf\nvalue: 2.0e-08\n"
    )
    assert target.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".campaign.yml.*.tmp"))


def write_project(tmp_path: Path, project_body: str,
                  campaigns: dict[str, str] | None = None) -> Path:
    experiments = tmp_path / "experiments"
    experiments.mkdir(parents=True, exist_ok=True)
    for name, body in (campaigns or {}).items():
        target = experiments / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    target = experiments / "research_project.yaml"
    target.write_text(project_body)
    return target


@pytest.mark.parametrize(("body", "message"), [
    ("[", "invalid YAML"),
    ("- item\n", "expected a mapping"),
    ("schema_version: 2\nprojects: []\n", "unsupported schema_version"),
    ("schema_version: 1\nprojects: wrong\n", "invalid daemon config"),
])
def test_server_config_failure_matrix(tmp_path, body, message):
    path = tmp_path / "console.yaml"
    path.write_text(body)
    with pytest.raises(ConfigError, match=message):
        load_server_config(path)


def test_project_and_question_schema_version_and_missing_file_failures(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_research_project(tmp_path / "missing.yml")
    question = tmp_path / "q.yml"
    question.write_text("schema_version: 2\nid: Q\ntitle: Question\n")
    with pytest.raises(ConfigError, match="unsupported schema_version"):
        load_research_question(question)
    question.write_text("schema_version: 1\nid: []\n")
    with pytest.raises(ConfigError, match="invalid research question"):
        load_research_question(question)
    project = write_project(tmp_path, "schema_version: 2\nproject: demo\ntitle: Demo\nrun_roots: []\n")
    with pytest.raises(ConfigError, match="unsupported schema_version"):
        load_research_project(project)
    project.write_text("schema_version: 1\nproject: []\n")
    with pytest.raises(ConfigError, match="invalid research project"):
        load_research_project(project)


@pytest.mark.parametrize(("campaign", "message"), [
    ("schema_version: 1\nproject: demo\ncampaign: study\nruns: nope\n", "runs must be a list"),
    ("schema_version: 1\nproject: demo\ncampaign: study\nrun_refs: nope\n", "run_refs must be a list"),
    ("schema_version: 1\nproject: demo\ncampaign: study\nruns: []\n", "requires runs or run_refs"),
    ("schema_version: 1\nproject: demo\ncampaign: study\nrun_refs: [bad]\n", "entries must be mappings"),
    ("schema_version: 1\nproject: demo\ncampaign: study\nrun_refs: [{run_id: 'run-{x}'}]\n", "concrete run_id"),
    ("schema_version: 1\nproject: demo\ncampaign: study\nruns: [{run_id: same}]\nrun_refs: [{run_id: same}]\n", "duplicate campaign run_id"),
])
def test_campaign_authored_shape_failure_matrix(tmp_path, campaign, message):
    project = write_project(
        tmp_path,
        "schema_version: 1\nproject: demo\ntitle: Demo\nrun_roots: []\n"
        "campaigns: [{name: study, file: experiments/study.yml}]\n",
        {"study.yml": campaign},
    )
    with pytest.raises(ConfigError, match=message):
        load_research_project(project)


def test_duplicate_campaigns_materializers_and_projects_are_rejected(tmp_path):
    duplicate_names = write_project(
        tmp_path / "names",
        "schema_version: 1\nproject: demo\ntitle: Demo\nrun_roots: []\n"
        "campaigns: [{name: study}, {name: study}]\n",
    )
    with pytest.raises(ConfigError, match="duplicate campaign name"):
        load_research_project(duplicate_names)

    common = "schema_version: 1\nproject: demo\ncampaign: {name}\nruns: [{{run_id: shared}}]\n"
    duplicate_runs = write_project(
        tmp_path / "runs",
        "schema_version: 1\nproject: demo\ntitle: Demo\nrun_roots: []\ncampaigns:\n"
        "  - {name: a, file: experiments/a.yml}\n  - {name: b, file: experiments/b.yml}\n",
        {"a.yml": common.format(name="a"), "b.yml": common.format(name="b")},
    )
    with pytest.raises(ConfigError, match="materialized by both"):
        load_research_project(duplicate_runs)

    first = write_project(tmp_path / "first", "schema_version: 1\nproject: same\ntitle: A\nrun_roots: []\n")
    second = write_project(tmp_path / "second", "schema_version: 1\nproject: same\ntitle: B\nrun_roots: []\n")
    config = ServerConfig(projects=[ProjectRef(project_file=str(first)), ProjectRef(project_file=str(second))])
    with pytest.raises(ConfigError, match="duplicate project name"):
        load_projects(config)


def test_yaml_unhashable_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yml"
    path.write_text("? [a, b]\n: value\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unhashable mapping key"):
        load_server_config(path)


def test_server_config_resolves_relative_project_and_token_paths(tmp_path):
    path = tmp_path / "console.yml"
    path.write_text(
        "schema_version: 1\n"
        "projects: [{project_file: experiments/project.yml}]\n"
        "http_auth: {bearer_token_file: secrets/token}\n",
        encoding="utf-8",
    )
    config = load_server_config(path)
    assert config.projects[0].project_file == str(
        (tmp_path / "experiments" / "project.yml").resolve()
    )
    assert config.http_auth.bearer_token_file == str(
        (tmp_path / "secrets" / "token").resolve()
    )


def test_missing_campaign_and_invalid_memberships_are_rejected(tmp_path):
    missing = write_project(
        tmp_path / "missing",
        "schema_version: 1\nproject: demo\ntitle: Demo\nrun_roots: []\n"
        "campaigns: [{name: study, file: experiments/missing.yml}]\n",
    )
    with pytest.raises(ConfigError, match="campaign file not found"):
        load_research_project(missing)

    duplicate = write_project(
        tmp_path / "duplicate",
        "schema_version: 1\nproject: demo\ntitle: Demo\nrun_roots: []\n"
        "campaigns: [{name: study, file: experiments/study.yml}]\n",
        {"study.yml": (
            "schema_version: 1\nproject: demo\ncampaign: study\n"
            "runs: [{run_id: same}, {run_id: same}]\n"
        )},
    )
    with pytest.raises(ConfigError, match="duplicate campaign run_id"):
        load_research_project(duplicate)

    for kind, entry in (
        ("runs", "{run_id: 'bad/id'}"),
        ("run_refs", "{run_id: 'bad/id'}"),
    ):
        invalid = write_project(
            tmp_path / kind,
            "schema_version: 1\nproject: demo\ntitle: Demo\nrun_roots: []\n"
            "campaigns: [{name: study, file: experiments/study.yml}]\n",
            {"study.yml": (
                "schema_version: 1\nproject: demo\ncampaign: study\n"
                f"{kind}: [{entry}]\n"
            )},
        )
        with pytest.raises(ConfigError, match="invalid campaign"):
            load_research_project(invalid)


def test_load_projects_accepts_empty_catalog():
    assert load_projects(ServerConfig(projects=[])) == []


def test_absolute_empty_research_question_directory_is_supported(tmp_path):
    questions = tmp_path / "questions"
    questions.mkdir()
    project = write_project(
        tmp_path,
        "schema_version: 1\nproject: demo\ntitle: Demo\nrun_roots: []\n"
        f"research_questions_dir: {questions}\n",
    )
    assert load_research_project(project).research_questions == []


def test_missing_research_question_directory_is_ignored(tmp_path):
    project = write_project(
        tmp_path,
        "schema_version: 1\nproject: demo\ntitle: Demo\nrun_roots: []\n"
        "research_questions_dir: missing-questions\n",
    )
    assert load_research_project(project).research_questions == []
