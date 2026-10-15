"""A late worker receives the durable deletion status across launch paths."""
import pytest

from ml_exp_server.results.artifact_store import ArtifactStore
from tests.test_run_deletion_api import deletion_api
from tests.test_run_deletion_service import deletion_env, publish


@pytest.mark.parametrize("method", ["get", "put"])
def test_deleted_launch_authorization_returns_gone_not_server_error(deletion_api, method):
    api, env = deletion_api
    publish(env)
    env.service.fence.mark("demo", "run-a")
    response = getattr(api, method)("/api/launch-transfers/demo/run-a/attempt-001/" + "a" * 64,
        headers={"Authorization": "Bearer " + env.tokens["run-a"]})
    assert response.status_code == 410
    assert response.headers["X-ML-Expd-Error-Code"] == "RUN_DELETED"


def test_deletion_between_launch_authorization_and_report_returns_gone(deletion_api, monkeypatch):
    api, env = deletion_api
    path, original = publish(env)
    def authorize(self, *args):
        env.service.fence.mark("demo", "run-a")
        return b"{}"
    monkeypatch.setattr(ArtifactStore, "launch", authorize)
    response = api.put("/api/launch-transfers/demo/run-a/attempt-001/" + "a" * 64,
        json={"phase": "WORKER_EXITED", "status": "COMPLETED", "exit_code": 0},
        headers={"Authorization": "Bearer " + env.tokens["run-a"]})
    assert response.status_code == 410 and response.headers["X-ML-Expd-Error-Code"] == "RUN_DELETED"
    assert path.read_bytes() == original
