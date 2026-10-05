"""Artifact HTTP reads, transfer limits, and idempotent async execution."""

import io
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from ml_exp_server.artifact_store import ArtifactStore
from ml_exp_server.schemas import AttemptSummary, RunIndexRow
from tests.test_artifact_store import archive, storage
from tests.test_container_api import client
from tests.test_submissions import _app, _action


def configure(client, storage):
    store, root, token, config, objects = storage
    client.app.state.runtime.config.container_execution.artifact_store_file = str(config)
    client.app.state.runtime.index.upsert_run(RunIndexRow(project="demo", run_id="run-a", run_dir=str(root),
        attempts=[AttemptSummary(attempt_id="attempt-001", state="SUCCEEDED")]))
    return store, root, token, objects


def test_transfer_configuration_authentication_and_size_are_checked(client, storage):
    endpoint = "/api/artifact-transfers/demo/run-a/attempt-001"
    assert client.put(endpoint, content=b"data").status_code == 404
    store, root, token, objects = configure(client, storage)
    assert client.put(endpoint, headers={"Authorization": "Basic test"}, content=b"data").status_code == 401
    original = storage[3]
    import json
    config = json.loads(Path(original).read_text())
    config["max_archive_bytes"] = 10
    Path(original).write_text(json.dumps(config))
    assert client.put(endpoint, headers={"Authorization": "Bearer " + token}, content=b"x" * 11).status_code == 413
    assert not objects


@pytest.mark.parametrize("uploaded", [False, True])
def test_archive_download_requires_receipt_and_closes_s3_body(client, storage, monkeypatch, uploaded):
    store, root, token, objects = configure(client, storage)
    endpoint = "/api/runs/demo/run-a/attempts/attempt-001/artifacts/archive"
    if not uploaded:
        assert client.get(endpoint).status_code == 404
        return
    data = archive({"result.bin": b"result"})
    receipt = store.receive("demo", "run-a", "attempt-001", token, io.BytesIO(data), len(data))
    bodies = []
    class Body(io.BytesIO):
        def iter_chunks(self, chunk_size):
            while chunk := self.read(chunk_size):
                yield chunk
    class S3:
        def get_object(self, *, Bucket, Key):
            body = Body(objects[(Bucket, Key)])
            bodies.append(body)
            return {"Body": body}
    monkeypatch.setattr(ArtifactStore, "client", lambda self: S3())
    response = client.get(endpoint)
    assert response.status_code == 200 and response.content == data
    assert response.headers["etag"] == '"' + receipt["sha256"] + '"'
    assert bodies[0].closed
    client.app.state.runtime.config.container_execution.artifact_store_file = None
    assert client.get(endpoint).status_code == 404


@pytest.mark.parametrize("selected,expected", [("bytes=-", 416), ("invalid", 416), ("bytes=5-3", 416), ("bytes=-2", 206)])
def test_artifact_suffix_and_invalid_ranges(client, storage, selected, expected):
    store, root, token, _ = configure(client, storage)
    data = archive({"result.bin": b"0123456789"})
    store.receive("demo", "run-a", "attempt-001", token, io.BytesIO(data), len(data))
    endpoint = "/api/runs/demo/run-a/attempts/attempt-001/files/outputs/result.bin"
    response = client.get(endpoint, headers={"Range": selected})
    assert response.status_code == expected
    if expected == 206:
        assert response.content == b"89"
    full = client.get(endpoint, headers={"Range": "bytes=2-5", "If-Range": '"old-etag"'})
    assert full.status_code == 200 and full.content == b"0123456789"


@pytest.mark.parametrize("route", ["action", "submission"])
def test_verified_async_execution_is_idempotent_without_new_scheduler_call(tmp_path, route):
    app, runner = _app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/api/actions/action-" + "0" * 16).status_code == 404
        response = client.post("/api/experiments/demo/run-a/submissions/prepare", json={"max_gpu_hours": 2})
        assert response.status_code == 200, response.text
        prepared = response.json()
        sid = prepared["submission_id"]
        authorized = client.post(f"/api/submissions/{sid}/authorize", json={}).json()
        if route == "action":
            path = "/api/actions/execute"
            body = {"action_id": sid, "confirmation": authorized["confirmation"]}
        else:
            path = f"/api/submissions/{sid}/execute"
            body = {"confirmation": authorized["confirmation"]}
        first = client.post(path, json=body)
        assert first.status_code == 200
        assert _action(client, sid)["execution"]["status"] == "VERIFIED"
        second = client.post(path, json=body).json()
        assert second["execution"]["status"] == "VERIFIED"
        assert len([call for call in runner.calls if call[3] == "submit" and "--dry-run" not in call]) == 1
