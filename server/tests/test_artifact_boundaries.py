"""Attempt-bound capabilities, corrupt-store recovery, and no-follow downloads."""

from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_exp_server import artifacts as module
from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.artifact_store import ArtifactStore
from ml_exp_server.artifacts import ArtifactService, open_directory, safe_parts
from ml_exp_server.schemas import AttemptSummary, RunIndexRow, ServerConfig
from ml_exp_server.source_imports import remove_staging
from tests.test_artifact_store import archive, storage


@pytest.mark.parametrize("identity", [("../project", "run-a", "attempt-001"), ("demo", "../run", "attempt-001"), ("demo", "run-a", "attempt-1")])
def test_transfer_rejects_escaping_or_noncanonical_identity(storage, identity):
    store, root, token, _, _ = storage
    with pytest.raises(ValueError, match="invalid transfer identity"):
        store.authorize(*identity, token)


def test_capability_is_idempotent_but_not_rebindable_or_renewable(storage):
    store, root, token, _, _ = storage
    assert store.issue("demo", "run-a", "attempt-001", root, ["**/*"])[1] == token
    with pytest.raises(ValueError, match="already bound"):
        store.issue("demo", "run-a", "attempt-001", root, ["*.pt"])
    with store.record("demo", "run-a", "attempt-001") as (path, value):
        value["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="expired"):
        store.authorize("demo", "run-a", "attempt-001", token)
    data = archive({"result.bin": b"result"})
    with pytest.raises(ValueError, match="capability"):
        store.receive("demo", "run-a", "attempt-001", token, io.BytesIO(data), len(data))


@pytest.mark.parametrize("failure", ["zero", "over-limit", "unknown-attempt", "token", "manifest", "undeclared-output"])
def test_upload_rejection_never_writes_s3(storage, failure):
    store, root, token, _, objects = storage
    data = archive({"nested/result.bin": b"result"})
    attempt = "attempt-001"
    size = len(data)
    if failure == "zero":
        size = 0
    elif failure == "over-limit":
        store.limit = 4 * 1024 ** 3
        size = store.limit + 1
    elif failure == "unknown-attempt":
        attempt = "attempt-002"
    elif failure == "token":
        token = "wrong"
    elif failure == "manifest":
        (root / "manifest.json").write_text('{"project":"other","run_id":"run-a"}')
    else:
        with store.record("demo", "run-a", attempt) as (path, value):
            value["outputs"] = ["*.pt"]
            path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        store.receive("demo", "run-a", attempt, token, io.BytesIO(data), size)
    assert not objects


@pytest.mark.parametrize("failure", ["size", "digest", "unavailable"])
def test_s3_corruption_never_publishes_restored_cache(storage, failure):
    store, root, token, _, objects = storage
    data = archive({"nested/result.bin": b"result"})
    receipt = store.receive("demo", "run-a", "attempt-001", token, io.BytesIO(data), len(data))
    remove_staging(root / "attempts/attempt-001/uploaded_outputs")
    key = (store.config["bucket"], receipt["object_key"])
    if failure == "size":
        objects[key] = b"truncated"
    elif failure == "digest":
        objects[key] = b"x" * len(data)
    else:
        del objects[key]
    with pytest.raises((ValueError, KeyError)):
        store.restore_cache("demo", "run-a", "attempt-001")
    parent = root / "attempts/attempt-001"
    assert not (parent / "uploaded_outputs").exists() and not list(parent.glob(".restore-*"))


def test_cache_restore_noop_and_interrupted_rename_recovery(storage):
    store, root, token, _, _ = storage
    store.restore_cache("demo", "run-a", "attempt-002")
    store.restore_cache("demo", "run-a", "attempt-001")
    destination = root / "attempts/attempt-001/uploaded_outputs"
    destination.mkdir()
    (destination / "stale.bin").write_bytes(b"stale")
    data = archive({"nested/result.bin": b"new"})
    store.receive("demo", "run-a", "attempt-001", token, io.BytesIO(data), len(data))
    store.restore_cache("demo", "run-a", "attempt-001")
    assert (destination / "nested/result.bin").read_bytes() == b"new" and not (destination / "stale.bin").exists()


def test_s3_client_and_multipart_config_do_not_require_network(tmp_path):
    path = tmp_path / "s3.json"
    path.write_text(json.dumps({"endpoint": "http://127.0.0.1:3900", "access_key": "test", "secret_key": "test"}))
    store = ArtifactStore(path, tmp_path)
    client = store.client()
    assert client.meta.endpoint_url == "http://127.0.0.1:3900"
    assert store._transfer_config().max_concurrency == 2
    client.close()


def service(tmp_path, config=None):
    row = RunIndexRow(project="demo", run_id="run-a", run_dir=str(tmp_path), attempts=[AttemptSummary(attempt_id="attempt-001")])
    runtime = SimpleNamespace(index=SimpleNamespace(get_run=lambda *args: row), config=config or ServerConfig())
    return ArtifactService(runtime), tmp_path / "attempts/attempt-001"


@pytest.mark.parametrize("path", ["", "/absolute", "../escape", ".env", "a\\b", "nul\x00value"])
def test_download_rejects_unsafe_relative_paths(tmp_path, path):
    with pytest.raises(ValueError):
        safe_parts(path)
    value, _ = service(tmp_path)
    with pytest.raises(ApplicationError, match="invalid artifact path"):
        value.open("demo", "run-a", "attempt-001", path)


def test_download_root_rejects_relative_paths_and_symlink_components(tmp_path):
    with pytest.raises(ValueError):
        open_directory(Path("relative"))
    with pytest.raises(ValueError):
        open_directory(tmp_path / "..")
    target = tmp_path / "target"
    target.mkdir()
    (tmp_path / "link").symlink_to(target, target_is_directory=True)
    with pytest.raises(OSError):
        open_directory(tmp_path / "link")


def test_download_handles_fifo_missing_namespace_and_truncated_file(tmp_path):
    value, attempt = service(tmp_path)
    outputs = attempt / "outputs/nested"
    outputs.mkdir(parents=True)
    (outputs / "result.bin").write_bytes(b"result")
    os.mkfifo(outputs / "fifo")
    for relative in ["unknown/result.bin", "outputs", "outputs/nested/fifo"]:
        with pytest.raises(ApplicationError):
            value.open("demo", "run-a", "attempt-001", relative)
    opened = value.open("demo", "run-a", "attempt-001", "outputs/nested/result.bin")
    assert b"".join(opened.chunks(1, 3)) == b"esu"
    opened.close()
    opened = value.open("demo", "run-a", "attempt-001", "outputs/nested/result.bin")
    (outputs / "result.bin").write_bytes(b"")
    with pytest.raises(OSError, match="truncated"):
        list(opened.chunks())
    assert opened.descriptor == -1


def test_download_listing_prefers_uploaded_file_and_tolerates_disappearing_files(tmp_path, monkeypatch):
    value, attempt = service(tmp_path)
    uploaded = attempt / "uploaded_outputs"
    local = attempt / "outputs"
    uploaded.mkdir(parents=True)
    local.mkdir()
    (uploaded / "result.bin").write_bytes(b"uploaded")
    (local / "result.bin").write_bytes(b"old")
    (uploaded / ".env").write_bytes(b"private")
    (uploaded / "vanished").write_bytes(b"race")
    original = module.os.stat
    def stat(path, *args, **kwargs):
        if path == "vanished":
            raise FileNotFoundError("project removed file")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(module.os, "stat", stat)
    listing = value.list("demo", "run-a", "attempt-001")
    assert listing["files"][0]["bytes"] == 8 and len(listing["files"]) == 1


def test_listing_truncates_at_documented_file_limit(tmp_path):
    value, attempt = service(tmp_path)
    outputs = attempt / "outputs"
    outputs.mkdir(parents=True)
    for index in range(10001):
        (outputs / f"{index:05}.bin").touch()
    listing = value.list("demo", "run-a", "attempt-001")
    assert listing["truncated"] and len(listing["files"]) == 10000


def test_artifact_reads_restore_exact_attempt_cache(storage, monkeypatch):
    store, root, token, config, _ = storage
    data = archive({"result.bin": b"remote"})
    store.receive("demo", "run-a", "attempt-001", token, io.BytesIO(data), len(data))
    remove_staging(root / "attempts/attempt-001/uploaded_outputs")
    runtime_config = ServerConfig(project_registry_root=str(store.root.parent), container_execution={"artifact_store_file": str(config)})
    value, _ = service(root, runtime_config)
    assert value.list("demo", "run-a", "attempt-001")["available"]
