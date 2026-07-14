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
from ml_exp_server.storage import StorageError, read_json


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
    assert store.snapshot(action_id)["journal"] == [{"event": "valid"}]
    (directory / "execution.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(StorageError, match="durable JSON is unreadable"):
        store.execution(action_id)
    updated = store.set_execution(action_id, {"status": "FAILED", "error": "boom"}, event="failed")
    assert updated["execution"]["status"] == "FAILED"

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
    assert [item["action_id"] for item in store.list_for_scope(operation_scope)] == [action_id]
    with pytest.raises(FileNotFoundError):
        store.snapshot("action-missing")

    missing = tmp_path / "missing.json"
    assert read_json(missing, {"fallback": True}) == {"fallback": True}
    missing.write_text("[]")
    assert read_json(missing, {}) == []


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

    winner = ActionStore(root).set_execution(
        action_id, {"status": "AUTHORIZED", "actor": "first"},
        event="authorized", expected_status="PREPARED",
    )
    assert winner["execution"]["actor"] == "first"
    with pytest.raises(RuntimeError, match="expected PREPARED, found AUTHORIZED"):
        ActionStore(root).set_execution(
            action_id, {"status": "AUTHORIZED", "actor": "second"},
            event="authorized", expected_status="PREPARED",
        )


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
