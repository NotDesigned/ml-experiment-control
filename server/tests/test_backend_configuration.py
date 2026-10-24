"""Standalone operator profiles work without unrelated provider credentials."""
from pathlib import Path

import pytest
import yaml

from ml_exp_server.data.data_assets import AssetStore
from tests.test_container_api import client, runtime
from tests.test_data_location_integration import publish
from tests.test_sensecore_data_workflow import stored


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("kind", ["slurm", "sensecore", "pai-dlc"])
def test_single_backend_template_discovers_and_freezes_without_other_credentials(client, monkeypatch, kind):
    from experiment_control.backends.pai_api import PaiAPI
    from experiment_control.backends.sensecore_rest import SenseCoreREST
    from ml_exp_server.backends.slurm import SlurmDataStager

    def unrelated(*a, **kw):
        pytest.fail("catalogue and metadata-only freezing must not authenticate another backend")

    monkeypatch.setattr(SenseCoreREST, "from_environment", unrelated)
    monkeypatch.setattr(PaiAPI, "from_environment", unrelated)
    monkeypatch.setattr(SlurmDataStager, "process", unrelated)
    monkeypatch.setenv("EXPERIMENTCTL_SENSECORE_REST_CONFIG", "/nonexistent/sensecore.json")
    monkeypatch.setenv("EXPERIMENTCTL_PAI_CONFIG", "/nonexistent/pai.json")

    content = (ROOT / "server/examples/executors" / (kind + ".yaml")).read_text()
    content = (content.replace("REPLACE_REGION", "us-east-1").replace("REPLACE_TARGET", "example")
               .replace("REPLACE_USER", "researcher").replace("REPLACE_ACCOUNT", "1234567890123456"))
    profiles = yaml.safe_load(content)["executors"]
    name, selected = next(iter(profiles.items()))
    if kind == "sensecore":
        selected["backend"]["pool_selection"] = "fixed"
    path = Path(client.app.state.runtime.config.container_execution.profiles_file)
    path.write_text(yaml.safe_dump({"executors": profiles}))

    response = client.get("/api/executors")
    assert response.status_code == 200, response.text
    exposed = response.json()["executors"]
    assert [(item["id"], item["kind"]) for item in exposed] == [(name, kind)]
    assert exposed[0]["capabilities"]["storage"]["persistent"]

    ready = runtime(client)
    request = {"run_id": "standalone", "runtime_id": ready["runtime_id"], "executor": name,
               "resources": {"gpus": 1, "cpus": 8, "memory_gb": 32, "max_time": "00:01:00"}}
    response = client.post("/api/projects/demo/runs", json=request)
    assert response.status_code == 200, response.text
    # This creates immutable metadata only; no scheduling/GPU request is made.
    assert client.post("/api/projects/demo/runs", json=request).json() == response.json()
    project = Path(client.app.state.runtime.project("demo").base_dir)
    frozen = yaml.safe_load((project / "experiments/campaigns/run-standalone.yaml").read_text())["runs"][0]
    assert frozen["backend"]["kind"] == kind
    assert not list(project.glob("**/attempts/attempt-*"))
    if kind == "slurm":
        assert frozen["backend"]["ssh_alias"] == "hpc-login"
        assert frozen["backend"]["mount_root"] == "/shared"
        assert frozen["backend"]["gres"] == "gpu:a100:1"


def test_legacy_location_is_read_without_rewriting_and_new_alias_cannot_change_origin(client, stored):
    original = publish(client, stored[0], "gpu", "wyd")
    root = client.app.state.runtime.config.project_registry_root_path()
    store = AssetStore(stored[0], root)
    record = store.root / "demo" / original["asset_id"] / "asset.json"
    locator = store.root.parent / "remote-data-assets/demo" / (original["asset_id"] + ".json")
    before = record.read_bytes(), locator.read_bytes()
    assert store.read("demo", original["asset_id"])["remote_storage"] == "wyd"
    renamed = {**original, "remote_storage": "slurm"}
    assert store.publish_remote("demo", renamed)["remote_storage"] == "wyd"
    assert (record.read_bytes(), locator.read_bytes()) == before
    with pytest.raises(ValueError, match="storage scope is already bound"):
        store.publish_remote("demo", {**renamed, "remote_locator": {"data_path": "/different/origin"}})
    assert (record.read_bytes(), locator.read_bytes()) == before
