"""Every image recipe delivers the same frozen data and checkpoint contract."""
from pathlib import Path

import pytest
import yaml

from ml_exp_server.container_controller import Controller
from ml_exp_server.container_execution import ContainerExecutionService
from ml_exp_server.image_builder import ImageBuilder
from ml_exp_server.worker_contract import CAPABILITIES, WORKER_CONTRACT, managed_io
from tests.test_container_api import legacy_prepare, archive, client, import_source, runtime, wait_runtime
from tests.test_sensecore_data_workflow import stored, put_asset


@pytest.mark.parametrize("recipe", ["source", "requirements", "dockerfile"])
def test_all_recipes_share_worker_receipt_data_mounts_and_checkpoint_support(client, stored, monkeypatch, tmp_path, recipe):
    base = "registry.example/base@sha256:" + "a" * 64
    source = import_source(client, archive({"train.py": b"print('training')\n", "requirements.txt": b"colorama==0.4.6\n",
                                          "Dockerfile": (f"FROM {base}\nRUN echo configured\n").encode()}))
    root = client.app.state.runtime.config.project_registry_root_path()
    builder = ImageBuilder({"source_root": str(root / "source-revisions/sources"), "state_root": str(tmp_path / "builds"),
                            "repository": "registry.example/runtime", "publisher": "buildkit",
                            "allow_dependency_builds": True, "allow_dockerfile_builds": True})
    packaged = []
    def publish(tag, context):
        internal = context / "ml-expd-build-internal" if recipe == "dockerfile" else context
        package = Path(__file__).parents[1] / "src/ml_exp_server"
        assert (internal / "worker.py").read_bytes() == (package / "managed_worker.py").read_bytes()
        assert (internal / "legacy_worker.py").read_bytes() == (package / "container_worker.py").read_bytes()
        assert (internal / "launch.py").read_bytes() == (package / "worker_launcher.py").read_bytes()
        assert "COPY --chmod=0555" in (context / "Dockerfile").read_text()
        assert (internal / "worker.py").stat().st_mode & 0o777 == 0o444
        assert (internal / "legacy_worker.py").stat().st_mode & 0o777 == 0o444
        packaged.append((context / "Dockerfile").read_text())
        return "registry.example/runtime@sha256:" + "b" * 64
    monkeypatch.setattr(builder, "_publish_buildkit", publish)
    monkeypatch.setattr("ml_exp_server.container_execution.builder_request", lambda socket, payload: builder.request(payload))
    spec = {"source_id": source["source_id"], "entrypoint": ["python3", "train.py"]}
    spec.update({"dockerfile": "Dockerfile"} if recipe == "dockerfile" else {"image": base})
    if recipe == "requirements": spec["requirements"] = "requirements.txt"
    value = (client.post("/api/projects/demo/runtimes/prepare", json=spec) if recipe == "dockerfile" else legacy_prepare(client, spec)).json()
    endpoint = "/api/projects/demo/runtimes/" + value["runtime_id"]
    assert client.post(endpoint + "/execute", json={"confirmation": value["confirmation"]}).status_code == 202
    ready = wait_runtime(client, endpoint)
    assert ready["status"] == "READY" and ready["capabilities"] == CAPABILITIES
    assert ready["worker_contract"] == WORKER_CONTRACT and packaged == [value["dockerfile"]]
    asset = put_asset(client).json()
    for executor in ("gpu", "cloud"):
        result = client.post("/api/projects/demo/runs", json={
            "run_id": executor, "executor": executor, "runtime_id": ready["runtime_id"],
            "inputs": [{"asset_id": asset["asset_id"], "mount_path": "/inputs/data"}],
            "checkpoint_upload": {"interval_seconds": 5}})
        assert result.status_code == 200, result.text
        project = Path(client.app.state.runtime.project("demo").base_dir)
        campaign = yaml.safe_load((project / f"experiments/campaigns/run-{executor}.yaml").read_text())
        campaign["local_root"] = str(client.app.state.runtime.config.project_run_root_path("demo"))
        ctl = Controller(campaign, executor, "attempt-001")
        ctl.prepare()
        manifest = ctl.store.load_manifest()
        assert manifest["command"] == ["ml-exp-worker"]
        assert manifest["resolved_config"]["container"]["worker_contract"] == WORKER_CONTRACT
        assert manifest["execution"].get("managed_io", False) == (executor == "gpu")
        dispatch = ctl.dispatch_command(ctl.store.load_attempt("attempt-001"))
        from ml_exp_server.artifact_store import ArtifactStore
        transfer = ArtifactStore(stored[0], root)
        with transfer.record("demo", executor, "attempt-001") as (_, record):
            environment = record["launch_manifest"]["environment"]
            assert environment["INPUTS_DIR"] == "/inputs"
            assert "ML_EXPD_INPUT_ASSETS" in environment and environment["ML_EXPD_SNAPSHOT_INTERVAL"] == "5"
            assert "ML_EXPD_UPLOAD_TOKEN" not in environment
        assert "--manifest-url" in dispatch
        changed = dict(campaign["runs"][0]["container"])
        changed.pop("worker_contract")
        campaign["runs"][0]["container"] = changed
        with pytest.raises(ValueError, match="immutable definition"):
            Controller(campaign, executor, "attempt-002")


@pytest.mark.parametrize("tamper", ["worker_contract", "worker_sha256", "capabilities", "dockerfile_sha256"])
def test_generated_recipe_rejects_forged_worker_capabilities(client, monkeypatch, tamper):
    source = import_source(client)
    service = ContainerExecutionService(client.app.state.runtime)
    original = __import__("ml_exp_server.container_execution", fromlist=["builder_request"]).builder_request
    def forged(socket, payload):
        result = original(socket, payload)
        result[tamper] = "wrong"
        return result
    monkeypatch.setattr("ml_exp_server.container_execution.builder_request", forged)
    value = legacy_prepare(client, {"source_id": source["source_id"],
        "image": "registry.example/python@sha256:" + "a" * 64, "entrypoint": ["python3", "train.py"]}).json()
    assert service.execute("demo", value["runtime_id"], value["confirmation"])["status"] == "RECONCILE_REQUIRED"


def test_legacy_ready_image_keeps_original_command_and_final_only_capability(client):
    ready = runtime(client)
    service = ContainerExecutionService(client.app.state.runtime)
    with service.state("demo", ready["runtime_id"]) as (store, snapshot):
        old = dict(snapshot.value)
        old.pop("worker_contract")
        old.pop("capabilities")
        store.commit(old, expected_revision=snapshot.revision, event={"event": "legacy_fixture"})
    assert service.execute("demo", old["runtime_id"], old["confirmation"]) == old
    assert client.post("/api/projects/demo/runs", json={"run_id": "legacy", "runtime_id": old["runtime_id"], "executor": "gpu"}).status_code == 200
    project = Path(client.app.state.runtime.project("demo").base_dir)
    campaign = yaml.safe_load((project / "experiments/campaigns/run-legacy.yaml").read_text())
    campaign["local_root"] = str(client.app.state.runtime.config.project_run_root_path("demo"))
    ctl = Controller(campaign, "legacy", "attempt-001")
    ctl.prepare()
    assert "INPUTS_DIR=/inputs" not in ctl.command("attempt-001")
    assert "managed_io" not in ctl.store.load_manifest()["execution"]
    assert "worker_contract" not in campaign["runs"][0]["container"]
    assert managed_io({"dockerfile": "Dockerfile"}) and not managed_io({})


def test_previous_managed_worker_keeps_full_dispatch_without_new_entrypoint(client, stored, tmp_path):
    from ml_exp_server.artifact_store import ArtifactStore
    from tests.test_persistent_checkpoints import controller, ready, token
    source = import_source(client, archive({"train.py": b"pass", "download.py": b"pass"}))
    bundle = runtime(client, source)
    service = ContainerExecutionService(client.app.state.runtime)
    with service.state("demo", bundle["runtime_id"]) as (store, snapshot):
        old = dict(snapshot.value)
        old["capabilities"] = [cap for cap in old["capabilities"] if cap != "launcher-manifest.v1"]
        store.commit(old, expected_revision=snapshot.revision, event={"event": "previous_launcher_fixture"})
    first = controller(client, bundle, name="old-first", checkpoint_persistence={})
    response = client.put("/api/checkpoint-transfers/demo/old-first/attempt-001", json=ready(tmp_path / "state"),
                          headers={"Authorization": "Bearer " + token(first, stored[0])})
    checkpoint = response.json()
    assert response.status_code == 200
    asset = put_asset(client).json()
    resumed = controller(client, bundle, name="old-resumed", checkpoint_persistence={},
        checkpoint_upload={"interval_seconds": 5}, data_preparation={"script": "download.py"},
        inputs=[{"asset_id": asset["asset_id"], "mount_path": "/inputs/data"}],
        resume_from={key: checkpoint[key] for key in ("run_id", "attempt_id", "checkpoint_id")})
    frozen = resumed.store.load_attempt("attempt-001")
    assert "/usr/local/lib/ml-expd/worker.py" in frozen["command"]
    dispatch = resumed.dispatch_command(frozen)
    assert any(arg.startswith("ML_EXPD_UPLOAD_TOKEN=") for arg in dispatch)
    assert any(arg.startswith("ML_EXPD_CHECKPOINT_RESTORE=") for arg in dispatch)
    assert "ml-exp-worker" not in dispatch
    transfer = ArtifactStore(stored[0], Path(resumed.campaign["source_store"]))
    with transfer.record("demo", "old-resumed", "attempt-001") as (_, record):
        assert "launch_manifest" not in record
