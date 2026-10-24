"""Minimal/complete deployments and private builder liveness, without cloud writes."""
from contextlib import contextmanager
import json
import os
import threading

from fastapi.testclient import TestClient
import pytest
import yaml

from ml_exp_server.api.app import create_app
from ml_exp_server.api.contract import SERVER_CAPABILITIES
from ml_exp_server.deployment import deployment_status, configured_capabilities
from ml_exp_server.image_builder import BuilderServer, Handler, ImageBuilder
from ml_exp_server.schemas import ServerConfig


@pytest.fixture
def config(tmp_path):
    profiles = tmp_path / "executors.yaml"
    profiles.write_text(yaml.safe_dump({"executors": {"local-cluster": {
        "backend": {"kind": "slurm"}, "storage_root": "/data"}}}))
    objects = tmp_path / "objects.json"
    objects.write_text(json.dumps({"endpoint": "http://127.0.0.1:3900", "bucket": "test",
        "access_key": "private-access-marker", "secret_key": "private-secret-marker",
        "public_transfer_base": "https://api.example/api/artifact-transfers"}))
    return ServerConfig(index_db=str(tmp_path / "index.sqlite"), collector_enabled=False,
        telemetry={"enabled": False}, action_runtime={"allow_project_writes": True,
        "allow_source_imports": True, "allow_scheduler_mutations": True},
        container_execution={"profiles_file": str(profiles), "artifact_store_file": str(objects),
                             "builder_socket": str(tmp_path / "builder.sock")})


@contextmanager
def builder(config, tmp_path, **settings):
    value = ImageBuilder({"state_root": str(tmp_path / "builder"), "client_uid": os.getuid(),
                          "publisher": "buildkit", "allow_dockerfile_builds": True, **settings})
    with BuilderServer(config.container_execution.builder_socket, Handler) as server:
        server.builder = value
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield value
        finally:
            server.shutdown()
            worker.join(timeout=5)


def test_minimal_api_is_healthy_without_optional_training_services(tmp_path):
    config = ServerConfig(index_db=str(tmp_path / "index.sqlite"), collector_enabled=False,
                          telemetry={"enabled": False})
    with TestClient(create_app(config)) as client:
        health = client.get("/api/health").json()
        assert health["status"] == "ok" and health["deployment"]["api"] == "READY"
        assert health["deployment"]["training"] == "INCOMPLETE"
        assert health["supported_capabilities"] == list(SERVER_CAPABILITIES)
        assert "deployment-readiness.v1" in health["capabilities"]
        assert "actions.v1" in health["capabilities"]
        for name in ("data-assets.v1", "dockerfile-build.v1", "oss-datasets.v1"):
            assert name in health["supported_capabilities"] and name not in health["capabilities"]
        readiness = client.get("/api/readiness").json()
        assert readiness["api"] == "READY" and readiness["training"] == "INCOMPLETE"
        assert set(readiness["issues"]) == {"executors", "image_builder", "artifact_store",
                                            "project_writes", "source_imports", "scheduler_mutations"}
        assert readiness["components"]["artifact_store"]["setting"] == "container_execution.artifact_store_file"
        assert client.get("/api/storage-limits").status_code == 404


def test_configured_service_probes_real_authorized_builder_without_building(config, tmp_path):
    with builder(config, tmp_path) as image_builder:
        with TestClient(create_app(config)) as client:
            health = client.get("/api/health").json()
            assert health["deployment"]["training"] == "CONFIGURED"
            assert health["deployment"]["scope"] == "configuration"
            assert "data-assets.v1" in health["capabilities"]
            assert "dockerfile-build.v1" in health["capabilities"]
            assert "oss-datasets.v1" not in health["capabilities"]
            readiness = client.get("/api/readiness").json()
            assert readiness["training"] == "CONFIGURED" and readiness["issues"] == []
            assert readiness["components"]["image_builder"]["status"] == "AVAILABLE"
            assert readiness["scope"] == "configuration_and_builder"
            assert readiness["backend_admission_checked"] is False
            assert "private-access-marker" not in json.dumps([health, readiness])
            assert "private-secret-marker" not in json.dumps([health, readiness])
            assert list(image_builder.root.iterdir()) == []
    assert deployment_status(config, probe=True)[0]["components"]["image_builder"]["status"] == "UNAVAILABLE"


@pytest.mark.parametrize("settings", [{"allow_dockerfile_builds": False}, {"publisher": "disabled"}])
def test_connected_builder_with_disabled_publisher_is_not_ready(config, tmp_path, settings):
    with builder(config, tmp_path, **settings):
        result, _ = deployment_status(config, probe=True)
        assert result["training"] == "INCOMPLETE"
        assert result["issues"] == ["image_builder"]


@pytest.mark.parametrize("response", [None, [], {"status": "bad", "dockerfile_builds": True},
                                      {"status": "ok", "dockerfile_builds": 1}])
def test_invalid_builder_response_fails_closed(config, monkeypatch, response):
    monkeypatch.setattr("ml_exp_server.deployment.builder_request", lambda *a, **k: response)
    assert deployment_status(config, probe=True)[0]["issues"] == ["image_builder"]


def test_builder_probe_has_short_timeout_and_does_not_expose_errors(config, monkeypatch):
    def rejected(path, payload, *, timeout):
        assert path == config.container_execution.builder_socket
        assert payload == {"operation": "health"} and timeout == 2
        raise ValueError("private-password and /private/config")
    monkeypatch.setattr("ml_exp_server.deployment.builder_request", rejected)
    result, _ = deployment_status(config, probe=True)
    assert result["issues"] == ["image_builder"]
    assert "private-password" not in json.dumps(result) and "/private/config" not in json.dumps(result)


@pytest.mark.parametrize("content", ["[", "[]", "{}", '{"endpoint":false}'])
def test_bad_storage_configuration_is_reported_without_exposing_contents(config, content):
    from pathlib import Path
    Path(config.container_execution.artifact_store_file).write_text(content)
    result, available = deployment_status(config)
    assert result["components"]["artifact_store"]["status"] == "INVALID"
    assert not available["storage"] and not available["oss_datasets"]


@pytest.mark.parametrize("content", ["[", "[]", "{}", "executors: []", "executors: {bad: []}",
                                     "executors: {bad: {backend: []}}",
                                     "executors: {bad: {backend: {kind: unsupported}}}"])
def test_bad_executor_configuration_is_reported(config, content):
    from pathlib import Path
    Path(config.container_execution.profiles_file).write_text(content)
    assert deployment_status(config)[0]["components"]["executors"]["status"] == "INVALID"


def test_empty_missing_and_unreadable_optional_configs(config, tmp_path):
    from pathlib import Path
    Path(config.container_execution.profiles_file).write_text("executors: {}")
    config.container_execution.artifact_store_file = str(tmp_path / "absent")
    result, _ = deployment_status(config)
    assert result["components"]["executors"]["status"] == "NOT_CONFIGURED"
    assert result["components"]["artifact_store"]["status"] == "INVALID"
    config.container_execution.profiles_file = str(tmp_path)
    assert deployment_status(config)[0]["components"]["executors"]["status"] == "INVALID"


def test_oss_capability_requires_its_independent_storage_configuration(config):
    from pathlib import Path
    path = Path(config.container_execution.artifact_store_file)
    value = json.loads(path.read_text());value["oss_datasets"] = {"enabled": True}
    path.write_text(json.dumps(value))
    assert deployment_status(config)[1]["oss_datasets"] is True
    assert "oss-datasets.v1" in configured_capabilities(SERVER_CAPABILITIES, deployment_status(config)[1])


def test_readiness_requires_the_same_bearer_as_other_api_routes(config, tmp_path):
    path = tmp_path / "http.token"
    path.write_text("private-api-marker-" + "x" * 32);path.chmod(0o600)
    config.http_auth.bearer_token_file = str(path)
    with TestClient(create_app(config)) as client:
        assert client.get("/api/readiness").status_code == 401
        response = client.get("/api/readiness", headers={"Authorization": "Bearer " + path.read_text()})
        assert response.status_code == 200
        assert "private-api-marker" not in response.text
