"""Checkpoint metadata queries are independent of result archive availability."""
from copy import deepcopy
import errno
import io
import json

import pytest

from ml_exp_server.results.artifact_store import ArtifactStore
from ml_exp_server.results.checkpoint_registry import CheckpointRegistry
from tests.test_container_api import archive, client, import_source, runtime
from tests.test_persistent_checkpoints import controller, ready, token
from tests.test_sensecore_data_workflow import stored


@pytest.fixture
def registered_metadata(client, stored, tmp_path):
    bundle = runtime(client, import_source(client))
    ctl = controller(client, bundle, checkpoint_persistence={},
                     checkpoint_upload={"interval_seconds": 5})
    headers = {"Authorization": "Bearer " + token(ctl, stored[0])}
    response = client.put("/api/checkpoint-transfers/demo/trial/attempt-001",
                          json=ready(tmp_path / "state"), headers=headers)
    assert response.status_code == 200, response.text
    checkpoint = response.json()
    snapshot_body = archive({"state.bin": b"uploaded checkpoint"})
    response = client.put("/api/snapshot-transfers/demo/trial/attempt-001",
                          content=snapshot_body, headers=headers)
    assert response.status_code == 200, response.text
    snapshot = response.json()
    objects = ArtifactStore(stored[0], client.app.state.runtime.config.project_registry_root_path())
    body = archive({"metrics.jsonl": b"{}\n"})
    receipt = objects.receive("demo", "trial", "attempt-001", token(ctl, stored[0]),
                              io.BytesIO(body), len(body))
    return ctl, objects, receipt, checkpoint, snapshot


@pytest.mark.parametrize("released", [False, True])
def test_metadata_queries_never_restore_results_or_contact_object_storage(
        client, stored, registered_metadata, monkeypatch, released):
    ctl, objects, receipt, checkpoint, snapshot = registered_metadata
    # The published result object is unusable and no expanded cache exists.
    # Metadata must remain readable even without disk space or S3 connectivity.
    stored[1][receipt["object_key"]] = b"corrupt result archive"
    with objects.record("demo", "trial", "attempt-001") as (path, value):
        if released:
            value["retention"] = {"status": "RELEASED"}
            path.write_text(json.dumps(value))
            stored[1].pop(receipt["object_key"])
        record_before = path.read_bytes()

    def cannot_restore(*args, **kwargs):
        raise OSError(errno.ENOSPC, "metadata reads must not restore result archives")

    monkeypatch.setattr(ArtifactStore, "restore_cache", cannot_restore)
    monkeypatch.setattr(ArtifactStore, "client", lambda *args, **kwargs:
                        pytest.fail("metadata reads must not contact object storage"))
    endpoint = "/api/runs/demo/trial/attempts/attempt-001"
    for suffix, expected in [("/snapshots", {"snapshots": [snapshot]}),
                             ("/checkpoints", {"checkpoints": [checkpoint]}),
                             ("/checkpoints/" + checkpoint["checkpoint_id"], checkpoint)]:
        response = client.get(endpoint + suffix)
        assert response.status_code == 200, response.text
        assert response.json() == expected
    assert not (ctl.root / "attempts/attempt-001/uploaded_outputs").exists()
    assert path.read_bytes() == record_before


def test_metadata_queries_keep_exact_attempt_and_checkpoint_hash_guards(
        client, stored, registered_metadata, monkeypatch):
    _, _, _, checkpoint, _ = registered_metadata
    monkeypatch.setattr(ArtifactStore, "restore_cache", lambda *args, **kwargs:
                        pytest.fail("metadata guards must not restore result archives"))
    endpoint = "/api/runs/demo/trial/attempts/attempt-001"
    suffixes = ["/snapshots", "/checkpoints", "/checkpoints/" + checkpoint["checkpoint_id"]]
    for unknown in [endpoint.replace("demo/", "other/"), endpoint.replace("trial/", "other/"),
                    endpoint.replace("attempt-001", "attempt-002")]:
        for suffix in suffixes:
            response = client.get(unknown + suffix)
            assert response.status_code == 404
            assert response.headers["X-ML-Expd-Error-Code"] == "UNKNOWN_ATTEMPT"
    assert client.get(endpoint + "/checkpoints/checkpoint." + "f" * 64).status_code == 404

    registry = CheckpointRegistry(stored[0], client.app.state.runtime.config.project_registry_root_path())
    path = registry.directory("demo", "trial", "attempt-001") / (checkpoint["checkpoint_id"] + ".json")
    original = path.read_bytes()
    for field in ["project", "run_id", "attempt_id", "source_id", "file_sha256"]:
        changed = deepcopy(checkpoint)
        if field == "file_sha256":
            changed["files"][0]["sha256"] = "f" * 64
        elif field == "source_id":
            changed[field] = "source." + "f" * 64
        else:
            changed[field] = "other" if field != "attempt_id" else "attempt-002"
        path.write_text(json.dumps(changed))
        for suffix in suffixes[1:]:
            assert client.get(endpoint + suffix).status_code == 409
        path.write_bytes(original)
    assert client.get(endpoint + suffixes[-1]).json() == checkpoint
