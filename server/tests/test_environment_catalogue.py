"""Approved FROM catalogue and Dockerfile-only request boundaries."""
import pytest
import yaml
from tests.test_container_api import client, import_source, prepare_runtime
BASE = "registry.example/base@sha256:" + "a" * 64

def catalogue(client, tmp_path, payload=None):
    path = tmp_path / "environments.yaml"
    path.write_text(yaml.safe_dump(payload if payload is not None else {"environments": {
        "torch": {"image": BASE, "versions": {"torch": "2.10.0+cu126"}, "private_field": "not-public"}}}))
    client.app.state.runtime.config.container_execution.environments_file = str(path)
    return path



def test_catalogue_only_publishes_approved_from_metadata(client, tmp_path):
    assert client.get("/api/environments").json() == {"environments": []}
    catalogue(client, tmp_path)
    assert client.get("/api/environments").json()["environments"] == [
        {"id": "torch", "image": BASE, "versions": {"torch": "2.10.0+cu126"}}]

@pytest.mark.parametrize("payload", [[], {"environments": []}, {"environments": {"../bad": {"image": BASE}}},
                                    {"environments": {"bad": "string"}}, {"environments": {"bad": {"image": "latest"}}}])
def test_invalid_catalogue_cannot_publish_unvalidated_entries(client, tmp_path, payload):
    catalogue(client, tmp_path, payload)
    response = client.get("/api/environments")
    assert response.status_code == (200 if payload == [] else 409)


@pytest.mark.parametrize("extra", [{"image": BASE}, {"image": None}, {"environment_id": "torch"},
    {"environment_id": None}, {"requirements": "requirements.txt"}, {"requirements": None},
    {"packaging_revision": "source-copy-docker-v2-v2"}])
def test_retired_build_selectors_fail_before_dispatch(client, extra):
    source = import_source(client)
    response = prepare_runtime(client, {"source_id": source["source_id"], "entrypoint": ["python3", "train.py"], **extra})
    assert response.status_code == 422

@pytest.mark.parametrize("path", ["../Dockerfile", "/Dockerfile", "a\\b", "a\x00b", ".private/Dockerfile", "", "a/./Dockerfile"])
def test_dockerfile_path_cannot_escape_source(client, path):
    source = import_source(client)
    assert prepare_runtime(client, {"source_id": source["source_id"], "entrypoint": ["python3"], "dockerfile": path}).status_code == 422
