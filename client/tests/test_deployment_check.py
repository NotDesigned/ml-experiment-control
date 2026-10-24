"""CLI readiness stays useful for minimal APIs and fails closed for training."""
import json

import pytest

from ml_exp_client.api import ClientError
from ml_exp_client.cli import main


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("configured", [False, True])
def test_base_and_training_check_have_distinct_requirements(monkeypatch, capsys, training, configured):
    calls = []
    class API:
        def negotiate(self):
            return {"capabilities": ["deployment-readiness.v1"], "deployment": {"training": "CONFIGURED"}}
        def call(self, path):
            calls.append(path)
            if path == "/api/readiness":
                return {"api": "READY", "training": "CONFIGURED" if configured else "INCOMPLETE",
                        "issues": [] if configured else ["artifact_store", "image_builder"]}
            return {}
    monkeypatch.setattr("ml_exp_client.cli.Client", lambda *a, **k: API())
    assert main(["--url", "https://api.example", "check", *(["--training"] if training else [])]) == (2 if training and not configured else 0)
    result = json.loads(capsys.readouterr().out)
    assert result["readiness"]["training"] == ("CONFIGURED" if configured else "INCOMPLETE")
    assert calls == ["/api/actions/policy", "/api/executors", "/api/readiness"]


@pytest.mark.parametrize("training", [False, True])
def test_older_server_unconfigured_storage_is_explicitly_optional(monkeypatch, capsys, training):
    class API:
        def negotiate(self):return {"capabilities": ["data-assets.v1"]}
        def call(self, path):
            if path == "/api/storage-limits":
                raise ClientError("not configured", status=404, details="data asset storage is not configured")
            return {}
    monkeypatch.setattr("ml_exp_client.cli.Client", lambda *a, **k: API())
    assert main(["--url", "https://api.example", "check", *(["--training"] if training else [])]) == (2 if training else 0)
    result = json.loads(capsys.readouterr().out)
    assert result["storage_limits"] == {"status": "NOT_CONFIGURED"}
    assert "storage_limits_error" not in result
    if training:
        assert result["readiness"]["training"] == "UNKNOWN"


@pytest.mark.parametrize("status,detail", [(404, "route not found"), (503, "data asset storage is not configured")])
def test_other_optional_service_errors_still_fail(monkeypatch, capsys, status, detail):
    class API:
        def negotiate(self):return {"capabilities": ["data-assets.v1"]}
        def call(self, path):
            if path == "/api/storage-limits":raise ClientError("failed", status=status, details=detail)
            return {}
    monkeypatch.setattr("ml_exp_client.cli.Client", lambda *a, **k: API())
    assert main(["--url", "https://api.example", "check"]) == 2
    assert json.loads(capsys.readouterr().out)["storage_limits_error"]["status"] == status


def test_deployment_field_works_without_live_readiness_capability(monkeypatch, capsys):
    class API:
        def negotiate(self):return {"capabilities": [], "deployment": {"api": "READY", "training": "CONFIGURED"}}
        def call(self, path):return {}
    monkeypatch.setattr("ml_exp_client.cli.Client", lambda *a, **k: API())
    assert main(["--url", "https://api.example", "check", "--training"]) == 0
    assert json.loads(capsys.readouterr().out)["readiness"]["training"] == "CONFIGURED"


@pytest.mark.parametrize("readiness", [None, [], {}, {"api": "READY", "training": "unexpected"},
                                      {"api": "FAILED", "training": "CONFIGURED"}])
def test_bad_readiness_response_is_not_success(monkeypatch, capsys, readiness):
    class API:
        def negotiate(self):return {"capabilities": ["deployment-readiness.v1"]}
        def call(self, path):return readiness if path == "/api/readiness" else {}
    monkeypatch.setattr("ml_exp_client.cli.Client", lambda *a, **k: API())
    assert main(["--url", "https://api.example", "check", "--training"]) == 2
    result = capsys.readouterr()
    assert not result.out and json.loads(result.err)["error"] == "invalid deployment readiness response"
