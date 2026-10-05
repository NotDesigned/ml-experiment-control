"""Transient build isolation and authenticated object-download capabilities."""
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from ml_exp_server.artifact_store import ArtifactStore
from ml_exp_server.image_builder import ImageBuilder
from tests.test_artifact_store import archive, storage
from tests.test_container_api import client
from tests.test_container_http_boundaries import configure
from tests.test_sensecore_data_workflow import stored, put_asset

REAL_CLIENT = ArtifactStore.client


@pytest.mark.parametrize("failure", [None, "create", "build", "cleanup"])
def test_ephemeral_builders_never_load_or_prune_shared_images(tmp_path, failure):
    builder = ImageBuilder({"state_root": str(tmp_path), "buildkit_image": "moby/buildkit@sha256:" + "a" * 64})
    builder.config["ephemeral_buildkit"] = True
    calls = []
    def docker(args, **kwargs):
        calls.append(args)
        if failure == "create" and args[1] == "create" or failure == "cleanup" and args[1] == "rm":
            raise ValueError("injected failure")
        if args[1] == "ls":
            return json.loads((tmp_path / "ephemeral-builder.json").read_text())["name"]
    def publish(tag, context, name):
        assert name == calls[0][calls[0].index("--name") + 1]
        assert (tmp_path / "ephemeral-builder.json").exists()
        if failure == "build":
            raise ValueError("injected failure")
        return "registry/image@sha256:" + "b" * 64
    builder._docker = docker
    builder._buildkit_image = publish
    if failure:
        with pytest.raises(ValueError, match="injected"):
            builder._publish_buildkit("registry/image:bundle", tmp_path)
    else:
        assert builder._publish_buildkit("registry/image:bundle", tmp_path).endswith("b" * 64)
    assert calls[0][:2] == ["buildx", "create"] and "default-load=false" in calls[0]
    assert calls[-1] == ["buildx", "rm", "--force", calls[0][3]]
    assert (tmp_path / "ephemeral-builder.json").exists() == (failure == "cleanup")


def test_recovery_removes_only_recorded_builder_and_rejects_invalid_owner(tmp_path, monkeypatch):
    calls = []
    def docker(self, args, **kwargs):
        calls.append(args)
        return "ml-expd-" + "a" * 32 if args[1] == "ls" else ""
    monkeypatch.setattr(ImageBuilder, "_docker", docker)
    path = tmp_path / "ephemeral-builder.json"
    path.write_text(json.dumps({"name": "ml-expd-" + "a" * 32}))
    ImageBuilder({"state_root": str(tmp_path), "ephemeral_buildkit": True})
    assert calls[-1] == ["buildx", "rm", "--force", "ml-expd-" + "a" * 32] and not path.exists()
    path.write_text('{"name":"default"}')
    with pytest.raises(ValueError, match="recovery"):
        ImageBuilder({"state_root": str(tmp_path), "ephemeral_buildkit": True})
    assert len(calls) == 2


def test_recovery_is_idempotent_when_builder_was_already_removed(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ImageBuilder, "_docker", lambda self, args, **kwargs: calls.append(args) or "default\n")
    path = tmp_path / "ephemeral-builder.json"
    path.write_text(json.dumps({"name": "ml-expd-" + "a" * 32}))
    ImageBuilder({"state_root": str(tmp_path), "ephemeral_buildkit": True})
    assert not path.exists() and calls == [["buildx", "ls", "--format", "{{.Name}}"]]


def test_ephemeral_builder_rejects_mutable_runtime_before_creation(tmp_path):
    value = ImageBuilder({"state_root": str(tmp_path), "ephemeral_buildkit": True})
    with pytest.raises(ValueError, match="pinned"):
        value._publish_buildkit("registry/image:bundle", tmp_path)


def test_legacy_publisher_removes_only_verified_output_tag(tmp_path):
    from ml_exp_server.image_builder import MANIFEST_TYPE
    builder = ImageBuilder({"state_root": str(tmp_path), "repository": "registry/image", "cleanup_published_image": True})
    raw = json.dumps({"mediaType": MANIFEST_TYPE, "config": {"digest": "sha256:" + "a" * 64}})
    digest = "sha256:" + hashlib.sha256(raw.encode()).hexdigest()
    calls = []
    builder._docker = lambda args, **kwargs: calls.append(args)
    def skopeo(args, **kwargs):
        if args[0] == "copy":
            Path(args[args.index("--digestfile") + 1]).write_text(digest)
        return raw
    builder._skopeo = skopeo
    assert builder._publish("registry/image:bundle", tmp_path) == "registry/image@" + digest
    assert calls[-1] == ["image", "rm", "registry/image:bundle"]


@pytest.mark.parametrize("failed", [False, True])
def test_wyd_conversion_drops_temporary_oci_cache_and_preserves_sif(tmp_path, failed):
    import os
    import shlex
    import subprocess
    from experiment_control.backends.wyd import WydSlurmBackend
    tools = tmp_path / "tools"; tools.mkdir()
    tool = tools / "apptainer"
    tool.write_text('#!/bin/sh\nprintf blob > "$APPTAINER_CACHEDIR/blob"\n' +
                    ('exit 1\n' if failed else 'printf verified-sif > "$3"\n'))
    tool.chmod(0o700)
    backend = object.__new__(WydSlurmBackend)
    backend.remote_exec = lambda alias, cmd: subprocess.run(shlex.split(cmd), check=True,
        env={**os.environ, "PATH": str(tools) + ":" + os.environ["PATH"]})
    sif = tmp_path / "images/test.sif"
    run = {"backend": {"ssh_alias": "cluster", "oci_image": "registry/image@sha256:" + "a" * 64, "sif_path": str(sif)}}
    if failed:
        with pytest.raises(subprocess.CalledProcessError):
            backend._stage_oci_image(run)
    else:
        backend._stage_oci_image(run)
        assert sif.read_text() == "verified-sif" and Path(str(sif) + ".oci").exists()
    assert not list(sif.parent.glob("*.cache.*")) and not list(sif.parent.glob("*.tmp.*"))


@pytest.mark.parametrize("endpoint", ["", "http://public", "https://user:pass@public", "https://public/path", "https://public?q=x", "https://public#x", "https://"])
def test_direct_signer_requires_public_https_endpoint(storage, endpoint):
    store = storage[0]
    store.config["public_endpoint"] = endpoint
    with pytest.raises(ValueError, match="HTTPS"):
        store.download("key", "a" * 64, 1)


@pytest.mark.parametrize("seconds", [29, 3601])
def test_download_lifetime_is_bounded(storage, seconds):
    store = storage[0]
    store.config.update(public_endpoint="https://objects.example", download_url_seconds=seconds)
    with pytest.raises(ValueError, match="lifetime"):
        store.download("key", "a" * 64, 1)


def test_public_presigner_uses_exact_object_and_does_not_contact_s3(storage, monkeypatch):
    store = storage[0]
    store.config["public_endpoint"] = "https://objects.example"
    monkeypatch.setattr(ArtifactStore, "client", REAL_CLIENT)
    ticket = store.download("demo/run/attempt.tar", "a" * 64, 10)
    target = urlsplit(ticket["url"])
    query = parse_qs(target.query)
    assert target.netloc == "objects.example" and target.path == "/results/demo/run/attempt.tar"
    assert query["X-Amz-Expires"] == ["300"] and query["X-Amz-SignedHeaders"] == ["host"]
    assert ticket["sha256"] == "a" * 64 and ticket["bytes"] == 10
    assert "secret_key" not in ticket


def signer(monkeypatch):
    calls = []
    class S3:
        def generate_presigned_url(self, method, **kwargs):
            calls.append((method, kwargs))
            return "https://objects.example/results/exact?signature=test"
    monkeypatch.setattr(ArtifactStore, "client", lambda self, **kwargs: S3())
    return calls


def test_download_link_authenticates_exact_attempt_without_restoring_cache(client, storage, monkeypatch):
    store, root, token, _ = configure(client, storage)
    endpoint = "/api/runs/demo/run-a/attempts/attempt-001/artifacts/download"
    assert client.get(endpoint).status_code == 404
    config = json.loads(storage[3].read_text()); config["public_endpoint"] = "https://objects.example"
    storage[3].write_text(json.dumps(config))
    assert client.get(endpoint).status_code == 404  # No receipt.
    body = archive({"checkpoint.bin": b"weights"})
    store.receive("demo", "run-a", "attempt-001", token, io.BytesIO(body), len(body))
    calls = signer(monkeypatch)
    monkeypatch.setattr(ArtifactStore, "restore_cache", lambda *args: pytest.fail("direct reads must not hydrate local cache"))
    response = client.get(endpoint)
    ticket = response.json()
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    assert ticket["files"] == [{"path": "checkpoint.bin", "bytes": 7, "sha256": hashlib.sha256(b"weights").hexdigest()}]
    assert calls[0][1]["Params"]["Key"].startswith("demo/run-a/attempt-001/")
    assert client.get(endpoint.replace("001", "002")).status_code == 404
    client.app.state.runtime.config.container_execution.artifact_store_file = None
    assert client.get(endpoint).status_code == 404


def test_signed_links_require_global_auth_not_worker_capability(storage, tmp_path):
    from fastapi.testclient import TestClient
    from ml_exp_server.api.app import create_app
    from ml_exp_server.schemas import ServerConfig
    token_file = tmp_path / "bearer"; token_file.write_text("a" * 40); token_file.chmod(0o600)
    config = ServerConfig(index_db=str(tmp_path / "index.sqlite"), collector_enabled=False,
        project_registry_root=str(tmp_path / "registry"), http_auth={"bearer_token_file": str(token_file)}, telemetry={"enabled": False})
    with TestClient(create_app(config, poll=False)) as api:
        endpoint = "/api/runs/demo/run-a/attempts/attempt-001/artifacts/download"
        assert api.get(endpoint, headers={"Authorization": "Bearer " + storage[2]}).status_code == 401
        assert api.get(endpoint).status_code == 401


def test_data_link_uses_only_registered_asset(client, stored, monkeypatch):
    asset = put_asset(client).json()
    endpoint = "/api/projects/demo/assets/" + asset["asset_id"] + "/download"
    assert client.get(endpoint).status_code == 404
    config = json.loads(stored[0].read_text()); config["public_endpoint"] = "https://objects.example"
    stored[0].write_text(json.dumps(config))
    calls = signer(monkeypatch)
    ticket = client.get(endpoint).json()
    assert ticket["files"] == asset["files"] and ticket["sha256"] == asset["sha256"]
    assert calls[0][1]["Params"]["Key"] == "data-assets/demo/" + asset["asset_id"] + ".tar"
    assert client.get(endpoint.replace(asset["asset_id"], "asset." + "f" * 64)).status_code == 404
