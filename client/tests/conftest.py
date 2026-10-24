"""A developer's saved connection must never influence the client suite."""
import pytest


@pytest.fixture(autouse=True)
def isolated_client_config(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_EXPD_CLIENT_CONFIG", str(tmp_path / "client.json"))
