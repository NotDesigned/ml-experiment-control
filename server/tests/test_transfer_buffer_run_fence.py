"""A late worker transfer cannot retain scratch for an explicitly deleted Run."""
import hashlib

import pytest

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.data.multipart_upload import UploadStore
from ml_exp_server.deletion_state import RunFence
from ml_exp_server.results.artifact_store import ArtifactStore
from tests.test_multipart_upload import client, stored, policy, begin, archive
from tests.test_sensecore_data_workflow import custom_runtime, controller


@pytest.mark.parametrize("code,mark,retained", [("RUN_DELETED", True, False),
                                              ("UPLOAD_EXPIRED", True, True),
                                              ("RUN_DELETED", False, True)])
def test_inflight_worker_scratch_requires_exact_deleted_fence(client, stored, monkeypatch, code, mark, retained):
    policy(stored)
    ctl = controller(client, custom_runtime(client, monkeypatch))
    ctl.dispatch_command(ctl.store.load_attempt("attempt-001"))
    registry = client.app.state.runtime.config.project_registry_root_path()
    objects = ArtifactStore(stored[0], registry)
    with objects.record("demo", "trial", "attempt-001") as (_, record):
        capability = record["token"]
    endpoint = "/api/attempt-uploads/demo/trial/attempt-001/artifacts"
    headers = {"Authorization": "Bearer " + capability}
    data = archive({"weights.pt": b"weights"})
    value = begin(client, data, endpoint, headers=headers)
    def late(*args):
        if mark:
            RunFence(registry).mark("demo", "trial")
        raise ApplicationError("upload unavailable", status_code=410, code=code)
    monkeypatch.setattr(UploadStore, "part", late)
    chunk = data[:value["part_bytes"]]
    response = client.put(endpoint + "/" + value["upload_id"] + "/parts/0",
                          params={"sha256": hashlib.sha256(chunk).hexdigest()}, headers=headers, content=chunk)
    assert response.status_code == 410 and response.headers["X-ML-Expd-Error-Code"] == code
    pending = list((registry / "transfer-buffer").glob("*.part"))
    assert bool(pending) is retained
    if retained:
        assert pending[0].read_bytes() == chunk
