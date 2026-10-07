"""Private builder authorization and immutable source/publication boundaries."""

import json
import os
from pathlib import Path
import runpy
import socketserver
import subprocess
import sys
import threading

import pytest

from ml_exp_server import image_builder as module
from ml_exp_server.image_builder import BuilderServer, ImageBuilder, builder_request
from ml_exp_server.source_imports import seal_tree
from ml_exp_server.source_revisions import _tree_digest


@pytest.fixture
def builder(tmp_path):
    source = tmp_path / "sources/demo/pending"
    tree = source / "tree"
    (tree / "nested").mkdir(parents=True)
    (tree / "nested/run.sh").write_text("echo training\n")
    (tree / "nested/run.sh").chmod(0o755)
    (tree / "Dockerfile").write_text("FROM registry.example/base@sha256:" + "a" * 64 + "\n")
    digest = _tree_digest(tree)
    source_id = "source." + digest[7:]
    published = source.with_name(source_id)
    source.rename(published)
    (published / "source.json").write_text(json.dumps({"project": "demo", "tree_digest": digest}))
    seal_tree(published)
    value = ImageBuilder({"state_root": str(tmp_path / "builder"), "source_root": str(tmp_path / "sources"),
                          "repository": "registry.example/results", "publisher": "buildkit", "allow_dockerfile_builds": True})
    value._publish_buildkit = lambda *args: "registry.example/results@sha256:" + "b" * 64
    request = {"operation": "build", "project": "demo", "source_id": source_id,
               "base_image": "registry.example/base@sha256:" + "a" * 64, "dockerfile": "Dockerfile"}
    return value, request, published


@pytest.mark.parametrize("failure", ["get-before-build", "unknown-operation", "recipe", "base-policy", "registry", "source-identity", "source-symlink", "tree-symlink", "metadata-symlink", "modified-source", "copy-race"])
def test_builder_rejects_invalid_or_changed_definitions(builder, tmp_path, monkeypatch, failure):
    value, request, source = builder
    if failure == "get-before-build":
        request["operation"] = "get"
    elif failure == "unknown-operation":
        request["operation"] = "execute-project-dockerfile"
    elif failure == "recipe":
        request["packaging_revision"] = "unreviewed"
    elif failure == "base-policy":
        value.config["base_image_prefixes"] = ["approved.example/"]
    elif failure == "registry":
        value.config["repository"] = "registry example"
    elif failure == "source-identity":
        source.chmod(0o700)
        p = source / "source.json"
        p.chmod(0o600)
        p.write_text('{"project":"other","tree_digest":"sha256:wrong"}')
    elif failure in {"source-symlink", "tree-symlink", "metadata-symlink"}:
        target = source if failure == "source-symlink" else source / ("tree" if failure == "tree-symlink" else "source.json")
        target.parent.chmod(0o700)
        if target.is_dir():
            target.chmod(0o700)
        moved = tmp_path / "moved"
        target.rename(moved)
        target.symlink_to(moved, target_is_directory=moved.is_dir())
    elif failure == "modified-source":
        path = source / "tree/nested/run.sh"
        path.chmod(0o700)
        path.write_text("changed\n")
        seal_tree(source)
    else:
        original = module.shutil.copytree
        def changed_copy(src, dest, *args, **kwargs):
            result = original(src, dest, *args, **kwargs)
            if Path(src) == source / "tree":
                path = Path(dest) / "nested/run.sh"
                path.chmod(0o700)
                path.write_text("changed during copy\n")
            return result
        monkeypatch.setattr(module.shutil, "copytree", changed_copy)
    with pytest.raises((ValueError, FileNotFoundError)):
        value.request(request)
    assert not [path for path in value.root.glob("*.json") if not path.name.endswith(".progress.json")]


def test_buildkit_source_packaging_and_receipt_binding(builder):
    value, request, source = builder
    result = value.request(request)
    assert result["source_id"] == request["source_id"] and result["image"].endswith("b" * 64)
    receipt = value.root / (result["bundle_id"] + ".json")
    data = json.loads(receipt.read_text())
    data["bundle_id"] = "wrong"
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="metadata mismatch"):
        value.request(request)


def test_frozen_log_lookup_survives_worker_recipe_upgrade(builder, monkeypatch):
    value, request, _ = builder
    old = value.request(request)
    (value.root / (old["bundle_id"] + ".log")).write_text("original build output\n")
    monkeypatch.setattr(module, "bundle_id", lambda *args, **kwargs: "e" * 64)
    assert value.request({**request, "operation": "logs"})["lines"] == []
    logs = value.request({**request, "operation": "logs", "bundle_id": old["bundle_id"]})
    assert logs["bundle_id"] == old["bundle_id"] and logs["lines"] == ["original build output"]
    assert not logs["truncated"]


@pytest.mark.parametrize("pinned", [None, "../private", "a" * 64])
def test_frozen_log_lookup_rejects_invalid_or_unpublished_identity(builder, pinned):
    value, request, _ = builder
    with pytest.raises(ValueError):
        value.request({**request, "operation": "logs", "bundle_id": pinned})


@pytest.mark.parametrize("field", ["bundle_id", "project", "source_id", "base_image"])
def test_frozen_log_lookup_checks_published_receipt_identity(builder, monkeypatch, field):
    value, request, _ = builder
    old = value.request(request)
    receipt = value.root / (old["bundle_id"] + ".json")
    receipt.write_text(json.dumps({**old, field: "wrong"}))
    monkeypatch.setattr(module, "bundle_id", lambda *args, **kwargs: "e" * 64)
    with pytest.raises(ValueError, match="identity mismatch"):
        value.request({**request, "operation": "logs", "bundle_id": old["bundle_id"]})


@pytest.mark.parametrize("failure", ["missing-command", "nonzero", "timeout"])
def test_builder_command_failures_never_echo_command_or_credentials(builder, monkeypatch, failure):
    value, _, _ = builder
    if failure == "timeout":
        monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("private-command", 1)))
        command = ["private-command", "test-private-credential"]
    else:
        command = ["/definitely/missing" if failure == "missing-command" else "/bin/false", "test-private-credential"]
    with pytest.raises(ValueError) as error:
        value._command(command)
    assert "test-private-credential" not in str(error.value)


def test_builder_command_capture_and_configured_tools(builder):
    value, _, _ = builder
    assert value._command(["/bin/echo", "result"], capture=True) == "result\n"
    assert value._command(["/bin/true"]) == ""
    value.config.update(docker="/bin/echo", skopeo="/bin/echo")
    assert value._docker(["version"], capture=True).strip() == "version"
    assert "inspect" in value._skopeo(["inspect"], capture=True)


@pytest.mark.parametrize("case", ["authorized", "storage", "wrong-uid", "rejected-request", "bad-path", "empty", "oversized", "invalid-json"])
def test_unix_builder_accepts_only_authorized_bounded_requests(tmp_path, case):
    path = str(tmp_path / "builder.sock")
    server = BuilderServer(path, module.Handler)
    class Builder:
        config = {"client_uid": os.getuid() + (case == "wrong-uid")}
        def request(self, payload):
            if case == "storage":
                raise module.BuildStorageError("BUILD_STORAGE_INSUFFICIENT", {"available_bytes": 7, "required_bytes": 20})
            if case == "rejected-request":
                raise ValueError("invalid packaging operation")
            return {"accepted": payload["operation"]}
    server.builder = Builder()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        if case in {"authorized", "rejected-request", "storage"}:
            if case == "authorized":
                assert builder_request(path, {"operation": "get"}) == {"accepted": "get"}
            elif case == "storage":
                with pytest.raises(module.BuildStorageError) as error:
                    builder_request(path, {"operation": "build"})
                assert error.value.code == "BUILD_STORAGE_INSUFFICIENT"
                assert error.value.details == {"available_bytes": 7, "required_bytes": 20}
            else:
                with pytest.raises(ValueError, match="packaging request failed"):
                    builder_request(path, {"operation": "get"})
        else:
            conn = module.UnixConnection(path, timeout=5)
            body = b"invalid" if case == "invalid-json" else b""
            length = "16385" if case == "oversized" else "2" if case in {"wrong-uid", "bad-path"} else str(len(body))
            conn.request("POST", "/other" if case == "bad-path" else "/", body=body,
                         headers={"Content-Length": length})
            response = conn.getresponse()
            assert response.status == 409 and json.loads(response.read())["error"] == "image packaging request failed"
            conn.close()
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_builder_entrypoint_creates_private_socket_and_serves_request(tmp_path, monkeypatch):
    config = tmp_path / "builder.json"
    socket = tmp_path / "builder.sock"
    socket.write_text("stale socket path")
    config.write_text(json.dumps({"socket": str(socket), "state_root": str(tmp_path / "state"),
                                 "client_uid": os.getuid(), "client_gid": os.getgid()}))
    results = []
    ownership = []
    monkeypatch.setattr(os, "chown", lambda path, uid, gid: ownership.append((Path(path), uid, gid)))
    def one_request(server):
        def client_request():
            try:
                builder_request(str(socket), {"operation": "get", "project": "demo", "source_id": "source." + "a" * 64, "base_image": "registry.example/image@sha256:" + "b" * 64})
            except ValueError:
                results.append("unpublished")
        thread = threading.Thread(target=client_request)
        thread.start()
        server.handle_request()
        thread.join(5)
        assert not thread.is_alive()
    monkeypatch.setattr(socketserver.BaseServer, "serve_forever", one_request)
    monkeypatch.setattr(sys, "argv", ["image-builder", "--config", str(config)])
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        runpy.run_module(module.__name__, run_name="__main__")
    assert results == ["unpublished"] and socket.stat().st_mode & 0o777 == 0o660
    assert ownership == [(socket.parent, 0, os.getgid()), (socket, 0, os.getgid())]
