"""Administrator deletion is durable, explicit, and fences late producers."""
from copy import deepcopy
import io
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.deletion_state import RunFence
from ml_exp_server.data.data_assets import AssetStore
from ml_exp_server.data.multipart_upload import UploadStore
from ml_exp_server.ingest.indexer import RunIndex, index_project
from ml_exp_server.records.store import RecordStore
from ml_exp_server.results.admin_deletion import AdminArtifacts
from ml_exp_server.results.artifact_lifecycle import ArtifactLifecycle
from ml_exp_server.results.checkpoint_registry import CheckpointRegistry
from ml_exp_server.schemas import ResearchProject, RunIndexRow
from ml_exp_server.storage import atomic_json
from ml_exp_server.tracking.tracking_service import observe
from ml_exp_server.tracking.tracking_store import TrackingStore
from tests.test_artifact_store import archive, storage

IDENTITY = ("demo", "run-a", "attempt-001")


def test_fence_cross_process_nonblocking_claim_and_permanent_identity(tmp_path):
    fence = RunFence(tmp_path)
    script = """from pathlib import Path
from ml_exp_server.deletion_state import RunFence
from ml_exp_server.application_errors import ApplicationError
import sys
try:
    RunFence(Path(sys.argv[1])).mark('demo', 'run-a')
except ApplicationError as error:
    assert error.code == 'RUN_DELETION_BUSY'
else:
    raise AssertionError('exclusive deletion crossed a shared reader')
"""
    with fence.guard("demo", "run-a"):
        result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        with pytest.raises(ApplicationError, match="busy"):
            with RunFence(tmp_path).exclusive():
                pytest.fail("a second descriptor crossed a shared reader")
        with fence.shared():
            with pytest.raises(ApplicationError, match="busy"):
                fence.mark("demo", "run-a")
    with fence.exclusive():
        with fence.shared():
            original = fence.mark("demo", "run-a", {"deletion_id": "deletion-original"})
    assert RunFence(tmp_path).mark("demo", "run-a", {"deletion_id": "changed"}) == original
    with pytest.raises(ApplicationError) as error:
        with fence.guard("demo", "run-a"):
            pytest.fail("deleted Run was reopened")
    assert (error.value.status_code, error.value.code) == (410, "RUN_DELETED")
    with pytest.raises(ValueError):
        fence.path("demo", "../outside")


def test_records_delete_aliases_and_shared_display_without_resurrection(tmp_path):
    store = TrackingStore(tmp_path)
    route = {"enabled": True, "entity": "person", "project": "science"}
    payload = {"kind": "metrics", "observations": [
        {"name": "loss", "unit": "nats", "value": 1, "status": "VALID", "step": 1}]}
    scopes = {}
    for run in (*store.run_identities("run-a"), "other"):
        config = {"ml_expd": {"run_id": "run-a", "record_type": run.rsplit("--", 1)[1]}} if "--" in run else {}
        store.bind("demo", run, route, config)
        scope = store.scope("demo", run, "attempt-001")
        store.append(scope, [("event", payload)])
        store.display(scope, 1)
        scopes[run] = scope
    private = {"kind": "metrics", "observations": [
        {"name": "private", "unit": "nats", "value": 2, "status": "VALID", "step": 1}]}
    store.append(scopes["run-a"], [("private", private)])
    store.display(scopes["run-a"], 2)
    with pytest.raises(ApplicationError, match="fenced first"):
        store.delete_run("demo", "run-a")
    RunFence(tmp_path).mark("demo", "run-a")
    assert {row["run"] for row in store.pending()} == {"other"}
    assert store.delete_run("demo", "run-a")["scopes_removed"] == 3
    assert store.delete_run("demo", "run-a")["scopes_removed"] == 0
    for run in store.run_identities("run-a"):
        assert store.binding("demo", run) is None
        for operation in [lambda: store.bind("demo", run, route, {}),
                          lambda: store.scope("demo", run, "attempt-001"),
                          lambda: store.append(scopes[run], [("late", payload)])]:
            with pytest.raises(ApplicationError) as error:
                operation()
            assert error.value.code == "RUN_DELETED"
    with store.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM event_ids").fetchone()[0] == 1
        assert [row[0] for row in conn.execute("SELECT public_name FROM display_metrics")] == ["loss"]
    assert set(store._display_cache) == {scopes["other"]}
    assert len(store.events(scopes["other"])) == 1


@pytest.mark.parametrize("retention", [None, "KEEP", "RELEASED"])
def test_admin_exact_objects_retain_ack_history_and_block_late_upload(storage, monkeypatch, retention):
    objects, root, token, _, remote = storage
    body = archive({"weights.bin": b"original"})
    old = objects.receive(*IDENTITY, token, io.BytesIO(body), len(body))
    if retention is None:
        repaired = archive({"weights.bin": b"original", "extra.bin": b"recovered"})
        current = objects.receive(*IDENTITY, token, io.BytesIO(repaired), len(repaired), repair_sha256=old["sha256"])
    else:
        current = old
    with objects.record(*IDENTITY) as (path, value):
        if retention:
            value["retention"] = {"status": retention, "sha256": current["sha256"]}
            value["download_acknowledgement"] = {"original": "retained exactly"}
        atomic_json(path, value)
        original = path.read_bytes()
    cache = root / "attempts/attempt-001/uploaded_outputs"
    cache.mkdir()
    (cache / "weights.bin").write_bytes(b"original")
    (cache / "unique.bin").write_bytes(b"unique local data")
    deletes = []
    def delete(**arguments):
        deletes.append(arguments["Key"])
        assert state["artifact_objects"][arguments["Key"]]["status"] == "DELETING"
        remote.pop((arguments["Bucket"], arguments["Key"]), None)
    monkeypatch.setattr(objects, "client", lambda: SimpleNamespace(delete_object=delete))
    service = AdminArtifacts(objects)
    with RunFence(objects.root.parent).exclusive():
        manifest = service.plan_objects(*IDENTITY[:2])
    assert len(manifest) == (2 if retention is None else 1)
    state, saved = {}, []
    with pytest.raises(ApplicationError, match="fenced first"):
        service.delete_objects(manifest, state, lambda value: saved.append(deepcopy(value)))
    RunFence(objects.root.parent).mark(*IDENTITY[:2])
    service.delete_objects(manifest, state, lambda value: saved.append(deepcopy(value)))
    service.delete_objects(manifest, state, lambda value: saved.append(deepcopy(value)))
    assert len(deletes) == (0 if retention == "RELEASED" else len(manifest))
    assert path.read_bytes() == original
    assert not (cache / "weights.bin").exists()
    assert (cache / "unique.bin").read_bytes() == b"unique local data"
    assert ArtifactLifecycle(objects).run_due() == 0
    with pytest.raises(ApplicationError) as error:
        objects.authorize(*IDENTITY, token)
    assert error.value.code == "RUN_DELETED"
    with pytest.raises(ApplicationError): objects.issue(*IDENTITY, root, ["**/*"])
    with pytest.raises(ApplicationError): objects.receive(*IDENTITY, token, io.BytesIO(body), len(body))
    with pytest.raises(ApplicationError): objects.artifact_download(*IDENTITY)


def test_admin_external_failure_retries_exact_key_and_cache_without_redelete(storage, monkeypatch):
    objects, _, token, _, remote = storage
    body = archive({"weights.bin": b"original"})
    objects.receive(*IDENTITY, token, io.BytesIO(body), len(body))
    service = AdminArtifacts(objects)
    manifest = service.plan_objects(*IDENTITY[:2])
    RunFence(objects.root.parent).mark(*IDENTITY[:2])
    state, saved, calls = {}, [], []
    def uncertain(**arguments):
        calls.append(arguments["Key"])
        remote.pop((arguments["Bucket"], arguments["Key"]), None)
        raise OSError("private remote diagnostic")
    monkeypatch.setattr(objects, "client", lambda: SimpleNamespace(delete_object=uncertain))
    with pytest.raises(ApplicationError) as error:
        service.delete_objects(manifest, state, lambda value: saved.append(deepcopy(value)))
    assert error.value.code == "ADMIN_OBJECT_DELETE_RETRY_REQUIRED"
    state = saved[-1]
    monkeypatch.setattr(objects, "client", lambda: SimpleNamespace(delete_object=lambda **args: calls.append(args["Key"])))
    import ml_exp_server.results.admin_deletion as module
    original = module.evict_caches
    monkeypatch.setattr(module, "evict_caches", lambda value: (_ for _ in ()).throw(OSError("private local diagnostic")))
    with pytest.raises(ApplicationError) as error:
        service.delete_objects(manifest, state, lambda value: saved.append(deepcopy(value)))
    assert error.value.code == "ADMIN_CACHE_CLEANUP_RETRY_REQUIRED"
    monkeypatch.setattr(module, "evict_caches", original)
    service.delete_objects(manifest, state, lambda value: saved.append(deepcopy(value)))
    assert calls == [manifest[0]["object_key"]] * 2
    assert "private" not in json.dumps(saved)


def test_index_and_tracking_ignore_permanent_tombstone(tmp_path, storage):
    objects, root, _, _, _ = storage
    index = RunIndex(tmp_path / "index.sqlite")
    index.deletion_fence = RunFence(objects.root.parent)
    row = RunIndexRow(project="demo", run_id="run-a", run_dir=str(root))
    assert index.upsert_run(row)
    other = RunIndexRow(project="demo", run_id="other", run_dir=str(root.parent / "other"))
    index.upsert_run(other)
    index.record_poll("demo", "other", "observe")
    index.record_poll(*IDENTITY[:2], "observe")
    index.deletion_fence.mark(*IDENTITY[:2])
    assert index.get_run(*IDENTITY[:2]) is None
    assert [item.run_id for item in index.list_runs()] == ["other"]
    assert not index.upsert_run(row)
    index.record_poll(*IDENTITY[:2], "observe")
    updates = []
    index.on_update = lambda *args: updates.append(args)
    assert index.delete_run(*IDENTITY[:2])
    assert not index.delete_run(*IDENTITY[:2])
    assert updates == [IDENTITY[:2]]
    assert [item.run_id for item in index.collector_statuses()] == ["other"]
    store = RecordStore(objects.root.parent)
    runtime = SimpleNamespace(index=SimpleNamespace(list_runs=lambda: [row]),
                              config=SimpleNamespace(project_registry_root_path=lambda: objects.root.parent))
    observe(runtime, store)
    project = ResearchProject(project="demo", title="Demo", run_roots=[str(root.parent)])
    assert index_project(index, project) == 0
    index.close()


def test_checkpoint_registration_and_queries_reject_late_worker(storage):
    objects, _, token, config, _ = storage
    registry = CheckpointRegistry(config, objects.root.parent)
    RunFence(objects.root.parent).mark(*IDENTITY[:2])
    for operation in [lambda: registry.read(*IDENTITY, "checkpoint." + "a" * 64),
                      lambda: registry.list(*IDENTITY),
                      lambda: registry.receive(*IDENTITY, token, {})]:
        with pytest.raises(ApplicationError) as error:
            operation()
        assert error.value.code == "RUN_DELETED"


@pytest.mark.parametrize("damage", ["sha", "bytes", "key", "attempt", "provenance", "release", "history"])
def test_corrupt_archive_inventory_never_reaches_object_delete(storage, monkeypatch, damage):
    objects, _, token, _, _ = storage
    body = archive({"weights.bin": b"original"})
    objects.receive(*IDENTITY, token, io.BytesIO(body), len(body))
    with objects.record(*IDENTITY) as (path, value):
        if damage == "sha": value["receipt"]["sha256"] = "invalid"
        elif damage == "bytes": value["receipt"]["bytes"] = True
        elif damage == "key": value["receipt"]["object_key"] = "other/research.tar"
        elif damage == "provenance": value["run_id"] = "other"
        elif damage == "release": value["retention"] = {"status": "RELEASED", "sha256": "f" * 64}
        elif damage == "history": value["receipt_history"] = [{**value["receipt"], "bytes": value["receipt"]["bytes"] + 1}]
        atomic_json(path, value)
        if damage == "attempt": atomic_json(path.with_name("attempt-wrong.json"), value)
    monkeypatch.setattr(objects, "client", lambda: pytest.fail("invalid inventory reached external storage"))
    with pytest.raises(ValueError):
        AdminArtifacts(objects).plan_objects(*IDENTITY[:2])


@pytest.mark.parametrize("damage", ["attempt", "provenance", "bytes", "release", "missing"])
def test_execution_revalidates_entire_sealed_manifest_before_first_delete(storage, monkeypatch, damage):
    objects, _, token, _, _ = storage
    body = archive({"weights.bin": b"original"})
    objects.receive(*IDENTITY, token, io.BytesIO(body), len(body))
    service = AdminArtifacts(objects)
    manifest = service.plan_objects(*IDENTITY[:2])
    invalid = deepcopy(manifest[0])
    if damage == "attempt": invalid["attempt_id"] = "attempt-wrong"
    elif damage == "bytes": invalid["bytes"] += 1
    elif damage == "release": invalid["already_released"] = True
    elif damage == "missing": invalid["object_key"] = invalid["object_key"].replace(invalid["sha256"], "f" * 64)
    elif damage == "provenance":
        with objects.record(*IDENTITY) as (path, value):
            value["project"] = "other"; atomic_json(path, value)
    RunFence(objects.root.parent).mark(*IDENTITY[:2])
    monkeypatch.setattr(objects, "client", lambda: pytest.fail("invalid manifest reached external storage"))
    with pytest.raises(ValueError):
        service.delete_objects([manifest[0], invalid], {}, lambda value: pytest.fail("invalid manifest changed progress"))


def test_duplicate_archive_history_is_one_object_and_empty_attempt_is_valid(storage):
    objects, _, token, _, _ = storage
    service = AdminArtifacts(objects)
    assert service.plan_objects(*IDENTITY[:2]) == []
    body = archive({"weights.bin": b"original"})
    receipt = objects.receive(*IDENTITY, token, io.BytesIO(body), len(body))
    with objects.record(*IDENTITY) as (path, value):
        value["receipt_history"] = [deepcopy(receipt)]
        value["retention"] = {"status": "RELEASED", "sha256": receipt["sha256"]}
        atomic_json(path, value)
    manifest = service.plan_objects(*IDENTITY[:2])
    assert len(manifest) == 1 and manifest[0]["already_released"]


def test_collector_rechecks_stale_planned_call_and_holds_shared_fence(tmp_path, monkeypatch):
    from ml_exp_server.collectord import Collector
    index = RunIndex(tmp_path / "index.sqlite")
    index.deletion_fence = RunFence(tmp_path / "registry")
    row = RunIndexRow(project="demo", run_id="run-a", run_dir=str(tmp_path / "run-a"))
    index.upsert_run(row)
    call = SimpleNamespace(project="demo", run_id="run-a", verb="observe")
    invoked = []
    def execute(call, **options):
        with pytest.raises(ApplicationError, match="busy"):
            RunFence(tmp_path / "registry").mark(*IDENTITY[:2])
        invoked.append(call)
        return {"returncode": 0}
    collector = Collector(index, [], controller=SimpleNamespace(execute=execute))
    monkeypatch.setattr(collector, "plan_cycle", lambda: [call])
    collector.run_cycle()
    assert invoked == [call]
    def failed(*args, **options):
        raise ApplicationError("observation failed", code="PROBE_FAILED")
    monkeypatch.setattr(collector.controller, "execute", failed)
    with pytest.raises(ApplicationError, match="observation failed"):
        collector.run_cycle()
    index.deletion_fence.mark(*IDENTITY[:2])
    collector.run_cycle()
    assert invoked == [call]
    index.close()


def test_publisher_rechecks_stale_scope_and_holds_shared_fence(tmp_path, monkeypatch):
    import ml_exp_server.tracking.tracking_service as module
    runtime = SimpleNamespace(index=SimpleNamespace(list_runs=lambda: []),
                              config=SimpleNamespace(project_registry_root_path=lambda: tmp_path))
    publisher = module.TrackingPublisher(runtime)
    store = publisher.store
    store.bind("demo", "run-a", {"enabled": True, "entity": "person", "project": "science"}, {})
    scope = store.scope(*IDENTITY)
    store.append(scope, [("event", {"kind": "lifecycle", "data": {"state": "SUCCEEDED"}})])
    atomic_json(store.root / "settings.json", {"api_key": "local-fixture-only"})
    invoked = []
    def export(store, row, transport):
        with pytest.raises(ApplicationError, match="busy"):
            RunFence(tmp_path).mark(*IDENTITY[:2])
        invoked.append(row["id"])
    monkeypatch.setattr(module, "export", export)
    pending = store.pending()
    publisher.cycle()
    assert invoked == [scope]
    def failed(*args):
        raise ApplicationError("publication failed", code="PROBE_FAILED")
    monkeypatch.setattr(module, "export", failed)
    with pytest.raises(ApplicationError, match="publication failed"):
        publisher.cycle()
    RunFence(tmp_path).mark(*IDENTITY[:2])
    module.preparation_record(runtime, {"project": "demo", "run_id": "run-a", "tracking": {"enabled": True}})
    monkeypatch.setattr(store, "pending", lambda: pending)
    publisher.cycle()
    assert invoked == [scope]


def test_snapshot_upload_and_link_are_one_fenced_publication(storage, monkeypatch):
    objects, root, token, config, _ = storage
    objects.issue(*IDENTITY, root, ["**/*"], checkpoint_upload=True)
    assets = AssetStore(config, objects.root.parent)
    invoked = []
    def receive(project, stream, digest, size, **options):
        with pytest.raises(ApplicationError, match="busy"):
            RunFence(objects.root.parent).mark(*IDENTITY[:2])
        invoked.append(digest)
        return {"asset_id": "asset." + digest}
    monkeypatch.setattr(assets, "receive", receive)
    first = assets.snapshot(*IDENTITY, token, io.BytesIO(b"checkpoint"), 10)
    assert assets.snapshot(*IDENTITY, token, io.BytesIO(b"checkpoint"), 10) == first
    with objects.record(*IDENTITY) as (_, value):
        assert value["snapshots"] == [first["asset_id"]]
    RunFence(objects.root.parent).mark(*IDENTITY[:2])
    with pytest.raises(ApplicationError):
        assets.snapshot(*IDENTITY, token, io.BytesIO(b"checkpoint"), 10)
    assert len(invoked) == 2


def test_local_only_maximum_length_alias_records_are_deleted(tmp_path):
    store = RecordStore(tmp_path)
    run = "r" * 128
    for name in store.run_identities(run):
        config = {"ml_expd": {"run_id": run, "record_type": name.rsplit("--", 1)[1]}} if name != run else {}
        store.bind("demo", name, {"enabled": False}, config)
        scope = store.scope("demo", name, "attempt-001")
        store.append(scope, [("event", {"kind": "metrics", "observations": []})])
    store.bind("demo", "other", {"enabled": False}, {})
    other = store.scope("demo", "other", "attempt-001")
    store.append(other, [("event", {"kind": "lifecycle", "data": {}})])
    RunFence(tmp_path).mark("demo", run)
    assert store.delete_run("demo", run)["scopes_removed"] == 3


def test_independent_suffix_run_records_survive_parent_deletion(tmp_path):
    store = RecordStore(tmp_path)
    for run in ("trial", "trial--preparation", "trial--backfill"):
        store.bind("demo", run, {"enabled": False}, {"ml_expd": {"run_id": run}})
        scope = store.scope("demo", run, "attempt-001")
        store.append(scope, [("event", {"kind": "lifecycle", "data": {"state": "SUCCEEDED"}})])
    RunFence(tmp_path).mark("demo", "trial")
    assert not store.is_deleted("demo", "trial--preparation")
    assert store.delete_run("demo", "trial")["scopes_removed"] == 1
    for run in ("trial--preparation", "trial--backfill"):
        scope = store.scope("demo", run, "attempt-001")
        store.append(scope, [("later", {"kind": "lifecycle", "data": {"state": "SUCCEEDED"}})])
        assert len(store.events(scope)) == 2


def test_ack_maintenance_preserves_unrelated_application_errors(storage, monkeypatch):
    objects, _, _, _, _ = storage
    def failed(*args):
        raise ApplicationError("storage failure", code="PROBE_FAILED")
    monkeypatch.setattr(objects, "record", failed)
    with pytest.raises(ApplicationError, match="storage failure"):
        ArtifactLifecycle(objects).run_due()


@pytest.mark.parametrize("kind", ["artifacts", "checkpoint"])
def test_multipart_late_run_capabilities_cannot_rebuild_staging_or_publish(tmp_path, kind):
    uploads = UploadStore(tmp_path, {})
    binding = dict(zip(("project", "run_id", "attempt_id"), IDENTITY), kind=kind)
    data = b"original bytes"
    sha = hashlib.sha256(data).hexdigest()
    value = uploads.create(binding, sha, len(data), None)
    upload_id = value["upload_id"]
    uploads.abort(upload_id, binding)
    RunFence(tmp_path).mark(*IDENTITY[:2])
    temporary = tmp_path / "uncommitted-part"
    temporary.write_bytes(data)
    before = (uploads.root / upload_id / "upload.json").read_bytes()
    for operation in [lambda: uploads.create(binding, sha, len(data), None),
                      lambda: uploads.part(upload_id, binding, 0, temporary, sha, len(data)),
                      lambda: uploads.complete(upload_id, binding, lambda *args: pytest.fail("late publication"), 100),
                      lambda: uploads.abort(upload_id, binding)]:
        with pytest.raises(ApplicationError) as error:
            operation()
        assert error.value.code == "RUN_DELETED"
    assert temporary.read_bytes() == data
    assert (uploads.root / upload_id / "upload.json").read_bytes() == before
    assert uploads.create({"project": "demo", "kind": "asset"}, sha, len(data), None)["status"] == "UPLOADING"


def test_multipart_publication_holds_fence_through_external_effect_and_commit(tmp_path):
    uploads = UploadStore(tmp_path, {})
    binding = dict(zip(("project", "run_id", "attempt_id"), IDENTITY), kind="artifacts")
    data = b"original bytes"
    sha = hashlib.sha256(data).hexdigest()
    value = uploads.create(binding, sha, len(data), None)
    temporary = tmp_path / "part"; temporary.write_bytes(data)
    uploads.part(value["upload_id"], binding, 0, temporary, sha, len(data))
    def publish(stream, size, digest):
        with pytest.raises(ApplicationError, match="busy"):
            RunFence(tmp_path).mark(*IDENTITY[:2])
        assert stream.read() == data
        return {"sha256": digest, "bytes": size}
    assert uploads.complete(value["upload_id"], binding, publish, 100) == {"sha256": sha, "bytes": len(data)}
    RunFence(tmp_path).mark(*IDENTITY[:2])
