"""Historical reads remain scoped without keeping executable retired recipes."""
import json
import pytest
from ml_exp_server.container_execution import ContainerExecutionService
from ml_exp_server.application_errors import ApplicationError
from tests.test_container_api import client, runtime
from tests.test_image_builder_boundary import builder


def test_retired_unbuilt_runtime_never_replays_publisher(client, monkeypatch):
    ready = runtime(client)
    service = ContainerExecutionService(client.app.state.runtime)
    with service.state("demo", ready["runtime_id"]) as (store, snapshot):
        old = dict(snapshot.value)
        old["status"] = "PREPARED"
        old["spec"] = {"source_id": old["spec"]["source_id"], "image": old["base_image"],
                       "entrypoint": ["python3"], "packaging_revision": "source-copy-docker-v2-v2"}
        store.commit(old, expected_revision=snapshot.revision, event={"event": "historical_fixture"})
    monkeypatch.setattr("ml_exp_server.container_execution.builder_request", lambda *args: pytest.fail("retired recipe dispatched"))
    with pytest.raises(ApplicationError, match="new Dockerfile Runtime"):
        service.execute("demo", old["runtime_id"], old["confirmation"])
    assert service.read("demo", old["runtime_id"]) == old


@pytest.mark.parametrize("field", ["project", "source_id", "base_image"])
def test_historical_log_identity_cannot_cross_bindings(builder, field):
    value, request, _ = builder
    identity = "e" * 64
    (value.root / (identity + ".definition.json")).write_text(json.dumps({
        "bundle_id": identity, **{key: request[key] for key in ("project", "source_id", "base_image")}}))
    (value.root / (identity + ".log")).write_text("historical log")
    read = {**request, "operation": "logs", "bundle_id": identity}
    read.pop("dockerfile")
    assert value.request(read)["lines"] == ["historical log"]
    read[field] = {"project": "other", "source_id": "source." + "c" * 64,
                   "base_image": "registry.example/other@sha256:" + "c" * 64}[field]
    with pytest.raises(ValueError, match="identity mismatch"):
        value.request(read)


def test_private_builder_requires_dockerfile_for_new_publication(builder):
    value, request, _ = builder
    request.pop("dockerfile")
    with pytest.raises(ValueError, match="Dockerfile recipe"):
        value.request(request)


@pytest.mark.parametrize("field,value", [("project", "../other"), ("source_id", "source.bad"),
    ("base_image", "registry.example/base:latest")])
def test_invalid_packaging_identity_is_rejected_before_execution(builder, field, value):
    service, request, _ = builder
    with pytest.raises(ValueError, match="invalid image packaging identity"):
        service.request({**request, field: value})
