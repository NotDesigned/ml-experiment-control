from pathlib import Path

from fastapi.testclient import TestClient

from ml_exp_server.api.app import create_app
from ml_exp_server.schemas import (
    ObservabilityConfig, ServerConfig, WandbCloudConfig,
)


def test_observability_api_exposes_no_credential_reference_or_local_paths(tmp_path: Path):
    config = ServerConfig(
        index_db=str(tmp_path / "index.sqlite"),
        action_root=str(tmp_path / "actions"),
        collector_enabled=False,
        projects=[],
        observability=ObservabilityConfig(
            credential_root=str(tmp_path / "credentials"),
            log_archive_root=str(tmp_path / "logs"),
            wandb_cloud=WandbCloudConfig(
                enabled=True, default_credential_ref="private-production-reference",
            ),
        ),
    )

    with TestClient(create_app(config)) as client:
        response = client.get("/api/observability")
        assert response.status_code == 200
        payload = response.json()
        assert payload["cloud"] == {
            "publisher_available": False,
            "state": "UNAVAILABLE",
        }
        encoded = repr(payload)
        assert "private-production-reference" not in encoded
        assert str(tmp_path / "credentials") not in encoded
        assert str(tmp_path / "logs") not in encoded
        assert "credential_ref" not in encoded
        assert "log_archive_root" not in encoded

        # Until a real publisher and auth boundary exist, secret provisioning
        # remains out of the HTTP surface entirely.
        write = client.post("/api/observability/wandb-credential", json={
            "credential_ref": "private-production-reference",
            "api_key": "secret-value",
        })
        assert write.status_code == 405
        assert write.json() == {"detail": "Method Not Allowed"}
