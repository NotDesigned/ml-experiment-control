"""Publication failure evidence and exact CPU observation without compute."""
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.data.multipart_upload import UploadStore
from ml_exp_server.storage import atomic_json
from tests.test_artifact_store import storage
from tests.test_result_collection import recovery, IDENTITY, output


def test_capacity_failure_is_durable_then_cleared_only_on_success(tmp_path, monkeypatch):
    store = UploadStore(tmp_path, {"upload_part_bytes": 1024})
    binding = dict(zip(("project", "run_id", "attempt_id"), IDENTITY), kind="artifacts")
    data = b"state"
    value = store.create(binding, hashlib.sha256(data).hexdigest(), len(data), None)
    temporary = tmp_path / "part"
    temporary.write_bytes(data)
    store.part(value["upload_id"], binding, 0, temporary, value["sha256"], len(data))
    with pytest.raises(ApplicationError) as error:
        store.complete(value["upload_id"], binding, Mock(), 10**20)
    assert error.value.status_code == 507
    failed = store.read(value["upload_id"], binding)
    diagnostic = failed["last_error"]
    assert diagnostic["code"] == "UPLOAD_STORAGE" and diagnostic["http_status"] == 507
    assert diagnostic["required_free_bytes"] == 10**20 and diagnostic["available_bytes"] >= 0
    assert failed["status"] == "UPLOADING" and len(failed["parts"]) == 1
    publish = Mock(return_value={"sha256": value["sha256"]})
    store.complete(value["upload_id"], binding, publish, 0)
    assert "last_error" not in store.read(value["upload_id"], binding)
    assert publish.call_count == 1


def test_publication_snapshot_is_exact_and_does_not_take_publisher_lock(recovery):
    service, _, _, _, _, _, _ = recovery
    uploads = service.objects.root.parent / "multipart-uploads"
    good = uploads / ("upload." + "a" * 64) / "upload.json"
    good.parent.mkdir(parents=True)
    binding = dict(zip(("project", "run_id", "attempt_id"), IDENTITY), kind="artifacts")
    value = {"upload_id": good.parent.name, "binding": binding, "bytes": 4, "status": "UPLOADING",
             "parts": {"0": {"bytes": 4, "sha256": "x"}}, "part_count": 1,
             "last_error": {"http_status": 507, "code": "UPLOAD_STORAGE"}}
    atomic_json(good, value)
    for name, content in (("broken", "{"), ("list", "[]"), ("foreign", json.dumps({"binding": {}, "status": "UPLOADING"})),
                          ("legacy", json.dumps({"binding": binding, "status": "UPLOADING", "bytes": 4}))):
        path = uploads / name / "upload.json"
        path.parent.mkdir(); path.write_text(content)
    result = service.read(*IDENTITY)["publication"]
    assert result["upload_id"] == good.parent.name and result["all_parts_received"]
    assert result["diagnostic"]["code"] == "UPLOAD_STORAGE"
    good.unlink()
    assert service.publication_diagnostics(IDENTITY) is None


def test_cpu_diagnostics_cache_is_bound_to_the_current_job(recovery, monkeypatch):
    service, _, _, _, _, definition, _ = recovery
    assert service.diagnostics(*IDENTITY)["cpu"] is None
    definition["backend"]["kind"] = "sensecore"
    root = service.evidence(*IDENTITY)[0]
    import yaml
    (root / "attempts/attempt-001/attempt.yaml").write_text(yaml.safe_dump(definition))
    observer = Mock(return_value={"scheduler_name": None, "logs": {"status": "NOT_APPLICABLE"}})
    monkeypatch.setattr(service, "result_job_diagnostics", observer)
    service.diagnostics(*IDENTITY, refresh=True)
    with service.state(*IDENTITY) as (_, snapshot):
        assert snapshot.value == {}
    service.update(IDENTITY, scheduler_name="cpu-exact", status="FAILED")
    cpu = {"scheduler_name": "cpu-exact", "job_exit_code": None, "logs": {"status": "PENDING"}}
    observer = Mock(return_value=cpu)
    monkeypatch.setattr(service, "result_job_diagnostics", observer)
    assert service.diagnostics(*IDENTITY, refresh=True)["cpu"] == cpu
    assert service.read(*IDENTITY)["cpu_diagnostics"] == cpu
    observer.assert_called_once_with(IDENTITY, refresh=True)
    assert service.diagnostics(*IDENTITY)["cpu"] == cpu
    observer.assert_called_with(IDENTITY, refresh=False)
    # A concurrent new recovery must not inherit the previous job's observation.
    service.update(IDENTITY, scheduler_name="cpu-new", cpu_diagnostics=None)
    service.diagnostics(*IDENTITY, refresh=True)
    assert service.read(*IDENTITY)["cpu_diagnostics"] is None
    with pytest.raises(ApplicationError):
        service.diagnostics("foreign", "run-a", "attempt-001")


def test_http_diagnostics_requires_bearer_and_never_starts_jobs(recovery, tmp_path):
    from starlette.testclient import TestClient
    from ml_exp_server.api.app import create_app
    service, _, token, _, _, _, row = recovery
    bearer = tmp_path / "bearer"
    bearer.write_text("a" * 40); bearer.chmod(0o600)
    service.runtime.config.http_auth.bearer_token_file = str(bearer)
    with TestClient(create_app(service.runtime.config, poll=False)) as client:
        client.app.state.runtime.index.upsert_run(row)
        submit = Mock()
        client.app.state.submit_job = submit
        endpoint = "/api/runs/demo/run-a/attempts/attempt-001/collection/diagnostics"
        headers = {"Authorization": "Bearer " + "a" * 40}
        assert client.get(endpoint).status_code == 401
        assert client.get(endpoint, headers={"Authorization": "Bearer " + token}).status_code == 401
        assert client.get(endpoint, headers=headers).json()["cpu"] is None
        assert client.get(endpoint, params={"refresh": True}, headers=headers).status_code == 200
        assert client.get(endpoint.replace('run-a', 'foreign'), headers=headers).status_code == 404
        submit.assert_not_called()
