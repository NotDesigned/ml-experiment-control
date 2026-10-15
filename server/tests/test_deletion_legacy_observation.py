"""Legacy observation identities stay readable while managed mutations stay strict."""
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ml_exp_server.application import ExperimentServerApplication
from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.collectord import Collector, CollectorConfig
from ml_exp_server.container_execution import ContainerExecutionService
from ml_exp_server.controller_gateway import ControllerCall
from ml_exp_server.data.multipart_upload import UploadStore
from ml_exp_server.deletion_state import RunFence
from ml_exp_server.ingest.indexer import index_project
from ml_exp_server.records.store import RecordStore
from ml_exp_server.results.checkpoint_registry import CheckpointRegistry
from ml_exp_server.schemas import AttemptSummary, RunIndexRow
from ml_exp_server.storage import atomic_json
from ml_exp_server.tracking.tracking_contract import WandbOptions
from ml_exp_server.tracking.tracking_service import backfill, observe
from ml_exp_server.tracking.tracking_store import TrackingStore
from tests.test_run_deletion_service import deletion_env


LEGACY = ["r" * 129, "legacy+run", "旧实验", "legacy:run", "run-a::literal"]


@pytest.fixture(params=LEGACY, ids=["long", "plus", "unicode", "colon", "literal-separator"])
def legacy_env(deletion_env, request):
    env = deletion_env
    env.legacy_id = request.param
    directory = env.config.project_run_root_path("demo") / "legacy-campaign" / env.legacy_id
    attempt = directory / "attempts" / "attempt-001"
    attempt.mkdir(parents=True)
    (directory / "manifest.yaml").write_text(yaml.safe_dump({
        "project": "demo", "campaign": "legacy-campaign", "run_id": env.legacy_id,
        "source_id": "legacy-source", "image_id": "sha256:" + "c" * 64}))
    (attempt / "attempt.yaml").write_text(yaml.safe_dump({
        "project": "demo", "run_id": env.legacy_id, "attempt_id": "attempt-001"}))
    atomic_json(attempt / "status.json", {"state": "SUCCEEDED", "attempt_id": "attempt-001"})
    metric = {"name": "loss", "value": 0.75, "unit": "nats/token", "step": 4,
              "protocol_id": "legacy-validation", "dataset_id": "retained-corpus"}
    (attempt / "train_metrics.jsonl").write_text(json.dumps(metric) + "\n")
    env.legacy_directory = directory
    env.legacy_row = RunIndexRow(project="demo", run_id=env.legacy_id, run_dir=str(directory),
        campaign="legacy-campaign", attempts=[AttemptSummary(attempt_id="attempt-001", state="SUCCEEDED")])
    return env


def test_legacy_index_upsert_read_listing_scan_and_poll_survive_deletion_fence(legacy_env):
    env = legacy_env
    index = env.runtime.index
    assert index.upsert_run(env.legacy_row)
    assert index.get_run("demo", env.legacy_id).run_dir == str(env.legacy_directory)
    assert env.legacy_id in {row.run_id for row in index.list_runs("demo")}
    index.record_poll("demo", env.legacy_id, "observe", outcome="succeeded")
    assert env.legacy_id in {row.run_id for row in index.collector_statuses("demo")}
    assert index_project(index, env.runtime.project("demo")) == 2
    assert index.get_run("demo", env.legacy_id).run_id == env.legacy_id
    assert env.legacy_id in {row.run_id for row in index.list_runs()}
    fence = RunFence(env.root)
    assert not fence.is_deleted("demo", env.legacy_id)
    assert fence.require_observed_live("demo", env.legacy_id) is None


def test_legacy_run_and_attempt_contexts_remain_observable(legacy_env):
    env = legacy_env
    env.runtime.index.upsert_run(env.legacy_row)
    if env.legacy_id == "run-a::literal":
        RunFence(env.root).mark("demo", "run-a")
    application = ExperimentServerApplication(env.runtime)
    scope, _, row = application.resolve_scope("demo", "run", env.legacy_id)
    assert scope.object_id == row.run_id == env.legacy_id
    exact = env.legacy_id + "::attempt-001"
    scope, _, attempt = application.resolve_scope("demo", "attempt", exact)
    assert scope.object_id == exact and attempt.attempt_id == "attempt-001"
    assert not env.backend.calls and not env.deleted


def test_collector_legacy_observe_holds_shared_fence_for_readonly_controller_call(legacy_env, monkeypatch):
    env = legacy_env
    index = env.runtime.index
    index.upsert_run(env.legacy_row)
    calls = [ControllerCall(project="demo", run_id=env.legacy_id, verb="observe", argv=["fake-observe"], cwd=env.legacy_directory)]
    observed = []
    def execute(call, *, timeout):
        assert call.verb == "observe" and timeout > 0
        # A separate descriptor must not delete any Run while the observation
        # process is still touching its canonical files.
        with pytest.raises(ApplicationError) as caught:
            with RunFence(env.root).exclusive():
                pytest.fail("deletion crossed an active legacy observation")
        assert caught.value.code == "RUN_DELETION_BUSY"
        observed.append(call.run_id)
        return {"returncode": 0, "stdout": "", "stderr": ""}
    collector = Collector(index=index, projects=[env.runtime.project("demo")], config=CollectorConfig(),
                          controller=SimpleNamespace(execute=execute))
    monkeypatch.setattr(collector, "plan_cycle", lambda: calls)
    original = (env.legacy_directory / "manifest.yaml").read_bytes()
    assert collector.run_cycle() == calls
    assert observed == [env.legacy_id]
    assert env.legacy_id in {row.run_id for row in index.collector_statuses("demo")}
    assert (env.legacy_directory / "manifest.yaml").read_bytes() == original
    assert not env.backend.calls and not env.deleted


def test_legacy_record_observation_and_backfill_keep_numeric_evidence(legacy_env):
    env = legacy_env
    env.runtime.index.upsert_run(env.legacy_row)
    store = TrackingStore(env.root)
    store.bind("demo", env.legacy_id, {"enabled": False}, {})
    observe(env.runtime, store)
    scope = store.scope("demo", env.legacy_id, "attempt-001")
    assert store.metric_records("demo", env.legacy_id, "attempt-001")[0]["value"] == 0.75
    assert any(item["payload"]["kind"] == "lifecycle" for item in store.events(scope))
    atomic_json(store.root / "settings.json", {"enabled": True, "api_key": "isolated-not-a-real-key", "resolved_entity": "fixture"})
    result = backfill(env.runtime, "demo", env.legacy_id, WandbOptions(entity="fixture", project="retained-evidence"))
    identity = env.legacy_id + "--backfill"
    alias = store.binding("demo", identity)
    assert result["wandb"]["enabled"] and alias["config"]["ml_expd"]["run_id"] == env.legacy_id
    assert alias["config"]["ml_expd"]["record_type"] == "backfill"
    assert store._fence_names("demo", identity) == [identity, env.legacy_id]
    assert store.metric_records("demo", identity, "attempt-001")[0]["value"] == 0.75
    original = deepcopy(store.metric_records("demo", identity, "attempt-001"))
    backfill(env.runtime, "demo", env.legacy_id, WandbOptions(entity="fixture", project="retained-evidence"))
    assert store.metric_records("demo", identity, "attempt-001") == original
    assert not env.backend.calls and not env.deleted


def test_publication_parent_fence_requires_proven_alias_binding(deletion_env):
    env = deletion_env
    records = RecordStore(env.root)
    route = {"enabled": False}
    records.bind("demo", "independent--backfill", route, {})
    original = records.scope("demo", "independent--backfill", "attempt-001")
    fence = RunFence(env.root)
    fence.mark("demo", "independent")
    records.append(original, [("real-suffix-run", {"kind": "event", "data": {"retained": True}})])
    assert not records.is_deleted("demo", "independent--backfill")
    for suffix, kind in [("--preparation", "preparation"), ("--backfill", "backfill")]:
        parent = "publication-" + kind
        alias = parent + suffix
        records.bind("demo", alias, route, {"ml_expd": {"run_id": parent, "record_type": kind}})
        scope = records.scope("demo", alias, "attempt-001")
        fence.mark("demo", parent)
        with pytest.raises(ApplicationError) as caught:
            records.append(scope, [("late", {"kind": "event", "data": {}})])
        assert caught.value.code == "RUN_DELETED" and records.events(scope) == []
    assert records.events(original)[0]["event_id"] == "real-suffix-run"


def test_legacy_observation_does_not_relax_managed_storage_or_mutation_identities(legacy_env):
    env = legacy_env
    fence = RunFence(env.root)
    run = env.legacy_id
    for effect in [lambda: fence.path("demo", run), lambda: fence.mark("demo", run),
                   lambda: fence.require_live("demo", run)]:
        with pytest.raises(ValueError): effect()
    with pytest.raises(ValueError):
        with fence.guard("demo", run):
            pytest.fail("legacy observation identity entered a managed mutation")
    objects = env.runtime.storage_service.objects
    checkpoints = CheckpointRegistry(Path(env.config.container_execution.artifact_store_file), env.root)
    uploads = UploadStore(env.root, {"local_storage_headroom_bytes": 0})
    containers = ContainerExecutionService(env.runtime)
    original_uploads = list(uploads.root.iterdir())
    original_actions = env.runtime.action_store.list_all()
    original_catalogue = env.project_file.read_bytes()
    for effect in [
        lambda: objects.issue("demo", run, "attempt-001", env.legacy_directory, ["**/*"]),
        lambda: objects.receive("demo", run, "attempt-001", "never-issued", io.BytesIO(b"x"), 1),
        lambda: checkpoints.receive("demo", run, "attempt-001", "never-issued", {}),
        lambda: uploads.create({"project": "demo", "run_id": run, "attempt_id": "attempt-001", "kind": "artifacts"}, "a" * 64, 1, None),
        lambda: containers.create_run("demo", SimpleNamespace(run_id=run)),
        lambda: env.service.prepare("demo", run, "unsupported legacy deletion", "never-delete-legacy"),
    ]:
        with pytest.raises((ValueError, ApplicationError)): effect()
    assert list(uploads.root.iterdir()) == original_uploads
    assert env.runtime.action_store.list_all() == original_actions
    assert env.project_file.read_bytes() == original_catalogue
    assert env.legacy_directory.exists() and not env.backend.calls and not env.deleted


@pytest.mark.parametrize("scope_type", ["run", "attempt"])
@pytest.mark.parametrize("top_level_run", [False, True])
def test_legacy_literal_separator_action_does_not_belong_to_another_run(
    deletion_env, scope_type, top_level_run,
):
    env = deletion_env
    literal = "run-a::literal"
    original = env.service._attempts(env.local["run-a"])
    plan = {"action_id": "action-" + "d" * 16, "request_digest": "legacy-literal-action", "ready": True,
        "scope": {"project": "demo", "scope_type": scope_type,
                  "object_id": literal + ("::attempt-099" if scope_type == "attempt" else "")},
        "operation": "SUBMIT_RUN", "attempt_id": "attempt-099"}
    if top_level_run:
        plan["run_id"] = literal
    saved = env.runtime.action_store.save_plan(plan)
    env.runtime.action_store.set_execution(plan["action_id"], {**saved["execution"], "status": "AUTHORIZED"},
                                           event="isolated-legacy-authorization")
    assert env.service._attempts(env.local["run-a"]) == original
    assert env.service._busy("demo", "run-a") == []
    reviewed = env.service.prepare("demo", "run-a", "exact scope must exclude literal separator Run", "scope-isolation")
    assert reviewed["ready"] and reviewed["target_manifest"]["attempts"] == original
    assert not env.backend.calls and not env.deleted
