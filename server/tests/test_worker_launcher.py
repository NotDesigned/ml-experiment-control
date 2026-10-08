"""Sealed launch, exact capability scope, and real bounded CPU execution."""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import signal
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from ml_exp_server.workers import worker_launcher as launcher
from ml_exp_server.api.app import create_app
from ml_exp_server.container_execution import Resources
from ml_exp_server.schemas import ServerConfig
from tests.test_artifact_store import storage


@pytest.fixture
def document(tmp_path):
    return {"contract": launcher.CONTRACT, "identity": {"project": "demo", "run_id": "run-a", "attempt_id": "attempt-001"},
            "environment": {"OUTPUT_DIR": str(tmp_path / "outputs")}, "workdir": str(tmp_path),
            "argv": [sys.executable, "-c", "print('training')"], "timeout_seconds": 10}


@pytest.mark.parametrize("change", [
    {"contract": "unknown"}, {"extra": True}, {"identity": []}, {"identity": {"project": "../escape"}},
    {"identity": {"project": 1, "run_id": "run-a", "attempt_id": "attempt-001"}},
    {"environment": []}, {"environment": {"bad=name": "x"}}, {"environment": {"OK": 1}},
    {"environment": {"OK": "\x00"}}, {"environment": {"ML_EXPD_UPLOAD_TOKEN": "private"}},
    {"environment": {"ML_EXPD_BOOTSTRAP_TOKEN": "private"}}, {"environment": {}},
    {"argv": []}, {"argv": "shell"}, {"argv": [1]}, {"argv": ["python", "\x00"]}, {"argv": [""]},
    {"workdir": "relative"}, {"workdir": 1}, {"workdir": "/\x00"},
    {"timeout_seconds": 0}, {"timeout_seconds": True}, {"timeout_seconds": 3600000},
    {"environment": {"OUTPUT_DIR": "/out", "LARGE": "x" * launcher.MAX_BYTES}},
])
def test_invalid_launch_never_becomes_executable(document, change):
    with pytest.raises(ValueError):
        launcher.encode({**document, **change})


def test_invalid_type_and_unbounded_api_budget_rejected():
    with pytest.raises(ValueError): launcher.encode([])
    with pytest.raises(ValueError, match="positive duration"): Resources(max_time="00:00:00")
    assert Resources().max_time == "00:10:00"
    assert Resources(max_time="999:59:59").max_time == "999:59:59"


class Connection:
    def __init__(self, body, status=200):
        self.body, self.status = io.BytesIO(body), status
        self.sock = SimpleNamespace(settimeout=lambda value: None)
        self.closed = False
        self.close_on_eof = False

    def request(self, method, path, headers):
        self.requested = (method, path, headers)

    def getresponse(self): return self
    def read1(self, size): return self.body.read(size)
    def isclosed(self): return self.close_on_eof and self.body.tell() == len(self.body.getvalue())
    def close(self): self.closed = True


def fetch_body(monkeypatch, body, *, status=200, digest=None):
    connection = Connection(body, status)
    monkeypatch.setattr(launcher.http.client, "HTTPSConnection", lambda *args, **kwargs: connection)
    digest = hashlib.sha256(body).hexdigest() if digest is None else digest
    return connection, "https://example/api/launch-transfers/demo/run-a/attempt-001/" + digest


def test_fetch_verifies_hash_scope_and_authorization(document, monkeypatch):
    body = launcher.encode(document)
    connection, url = fetch_body(monkeypatch, body)
    assert launcher.fetch(url, "private-capability") == document
    assert connection.closed
    assert connection.requested == ("GET", url.removeprefix("https://example"), {"Authorization": "Bearer private-capability"})
    connection, url = fetch_body(monkeypatch, body)
    connection.close_on_eof = True
    assert launcher.fetch(url, "private-capability") == document
    connection, url = fetch_body(monkeypatch, body)
    assert launcher.fetch(url.replace("/api/", "/ml-expd/api/"), "private-capability") == document
    assert connection.requested[1].startswith("/ml-expd/api/")


@pytest.mark.parametrize("suffix", ["?token=private", "#fragment", "/extra"])
def test_fetch_rejects_nonexact_url(document, suffix):
    url = "https://example/api/launch-transfers/demo/run-a/attempt-001/" + "a" * 64
    with pytest.raises(ValueError): launcher.fetch(url + suffix, "capability")


@pytest.mark.parametrize("url", ["http://example", "https:///", "https://user:password@example"])
def test_fetch_rejects_unsafe_origin(url):
    with pytest.raises(ValueError): launcher.fetch(url, "capability")


@pytest.mark.parametrize("mode", ["status", "digest", "scope", "noncanonical", "json", "oversize", "timeout", "body_timeout", "token"])
def test_fetch_failure_closes_transport(document, monkeypatch, mode):
    body = launcher.encode(document)
    if mode == "scope":
        changed = deepcopy(document); changed["identity"]["run_id"] = "other"
        body = launcher.encode(changed)
    if mode == "noncanonical": body = json.dumps(document).encode()
    if mode == "json": body = b"not json"
    if mode == "oversize": body = b"x" * (launcher.MAX_BYTES + 1)
    connection, url = fetch_body(monkeypatch, body, status=503 if mode == "status" else 200,
                                 digest="0" * 64 if mode == "digest" else None)
    if mode in {"timeout", "body_timeout"}:
        times = iter([0, 31] if mode == "timeout" else [0, 0, 31])
        monkeypatch.setattr(launcher.time, "monotonic", lambda: next(times))
    with pytest.raises((ValueError, TimeoutError)):
        launcher.fetch(url, "\nprivate" if mode == "token" else "private-capability")
    assert connection.closed == (mode != "token")


def test_manifest_is_immutable_and_capability_cannot_read_other_attempt(storage, document, tmp_path):
    store, _, token, config, _ = storage
    url = store.seal_launch("demo", "run-a", "attempt-001", document)
    assert store.seal_launch("demo", "run-a", "attempt-001", document) == url
    digest = url.rsplit("/", 1)[1]
    assert store.launch("demo", "run-a", "attempt-001", digest, token) == launcher.encode(document)
    changed = deepcopy(document); changed["argv"].append("--changed")
    with pytest.raises(ValueError, match="already sealed"): store.seal_launch("demo", "run-a", "attempt-001", changed)
    with pytest.raises(ValueError, match="identity differs"): store.seal_launch("demo", "other", "attempt-001", document)
    changed["identity"]["attempt_id"] = "attempt-002"
    with pytest.raises(ValueError, match="not been issued"): store.seal_launch("demo", "run-a", "attempt-002", changed)
    _, second_token, _ = store.issue("demo", "run-a", "attempt-002", tmp_path, ["**/*"])
    with pytest.raises(ValueError, match="not available"):
        store.launch("demo", "run-a", "attempt-002", digest, second_token)
    bearer = tmp_path / "bearer"; bearer.write_text("a" * 40); bearer.chmod(0o600)
    server = ServerConfig(index_db=str(tmp_path / "index.sqlite"), project_registry_root=str(tmp_path / "projects"),
        collector_enabled=False, telemetry={"enabled": False}, http_auth={"bearer_token_file": str(bearer)},
        container_execution={"artifact_store_file": str(config)})
    with TestClient(create_app(server, poll=False)) as client:
        path = url.removeprefix("https://example")
        headers = {"Authorization": "Bearer " + token}
        response = client.get(path, headers=headers)
        assert response.status_code == 200 and response.content == launcher.encode(document)
        assert response.headers["cache-control"] == "no-store"
        assert token.encode() not in response.content
        assert client.get(path, headers={"Authorization": "Basic " + token}).status_code == 401
        assert client.get(path.replace("001", "002"), headers=headers).status_code == 401
        assert client.get(path[:-64] + "f" * 64, headers=headers).status_code == 401
        assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.post(path, headers=headers).status_code == 401
        assert client.get("/api/projects", headers=headers).status_code == 401


def test_unconfigured_launch_returns_404(tmp_path):
    server = ServerConfig(index_db=str(tmp_path / "index.sqlite"), collector_enabled=False, telemetry={"enabled": False})
    with TestClient(create_app(server, poll=False)) as client:
        path = "/api/launch-transfers/demo/run-a/attempt-001/" + "a" * 64
        assert client.get(path).status_code == 404


def test_invalid_public_origin_fails_before_sealing(storage, document):
    store = storage[0]
    store.config["public_transfer_base"] = "http://example/api/artifact-transfers"
    with pytest.raises(ValueError, match="exact HTTPS"):
        store.seal_launch("demo", "run-a", "attempt-001", document)
    with store.record("demo", "run-a", "attempt-001") as (_, record):
        assert "launch_manifest" not in record


@pytest.mark.parametrize("mode, expected", [("success", 0), ("failure", 7), ("timeout", 124)])
def test_real_worker_preserves_argv_budget_and_hides_capability(document, tmp_path, monkeypatch, mode, expected):
    wrapper = tmp_path / "worker.py"
    wrapper.write_text("from ml_exp_server.workers import managed_worker as w\n"
                       "w.link_path = lambda target, path: None\n"
                       "w.upload = lambda url, token, stream, size: None\nraise SystemExit(w.main())\n")
    code = "import json, os, pathlib; pathlib.Path(os.environ['OUTPUT_DIR'], 'proof.json').write_text(json.dumps({'bootstrap': 'ML_EXPD_BOOTSTRAP_TOKEN' in os.environ, 'upload': 'ML_EXPD_UPLOAD_TOKEN' in os.environ}))"
    if mode == "failure": code += "; raise SystemExit(7)"
    if mode == "timeout": code += "; import time; time.sleep(60)"
    document["argv"] = [sys.executable, "-c", code]
    document["environment"].update(ML_EXPD_UPLOAD_URL="https://example/upload", ML_EXPD_UPLOAD_LIMIT=str(4 * 1024 * 1024), ML_EXPD_OUTPUT_PATTERNS='["**/*"]')
    document["timeout_seconds"] = 1 if mode == "timeout" else 10
    monkeypatch.setenv("ML_EXPD_BOOTSTRAP_TOKEN", "private-capability")
    assert launcher.execute(document, "private-capability", worker=str(wrapper)) == expected
    proof = json.loads((Path(document["environment"]["OUTPUT_DIR"]) / "proof.json").read_text())
    assert proof == {"bootstrap": False, "upload": False}


@pytest.mark.parametrize("gone", [False, True])
def test_cancellation_forwarded_and_handlers_restored(document, monkeypatch, gone):
    previous = signal.getsignal(signal.SIGTERM)
    forwarded = []
    def kill(pid, sig):
        forwarded.append((pid, sig))
        if gone: raise ProcessLookupError()
    def wait():
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return 143
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=123, wait=wait))
    monkeypatch.setattr(launcher.os, "killpg", kill)
    assert launcher.execute(document, "private") == 143
    assert forwarded == [(123, signal.SIGTERM)]
    assert signal.getsignal(signal.SIGTERM) == previous


def test_cli_main_does_not_print_secrets_on_failure(document, monkeypatch, capsys):
    monkeypatch.setenv("ML_EXPD_BOOTSTRAP_TOKEN", "private-capability")
    monkeypatch.setattr(launcher, "fetch", lambda url, token: document)
    monkeypatch.setattr(launcher, "execute", lambda spec, token: 0)
    assert launcher.main(["--manifest-url", "https://example/private-url"]) == 0
    assert "private" not in capsys.readouterr().out
    def fail(*args): raise ValueError("private-capability private-url")
    monkeypatch.setattr(launcher, "fetch", fail)
    assert launcher.main(["--manifest-url", "https://example/private-url"]) == 78
    assert "private" not in capsys.readouterr().err
    def invalid(*args): raise launcher.LaunchError("launch manifest checksum differs")
    monkeypatch.setattr(launcher, "fetch", invalid)
    assert launcher.main(["--manifest-url", "https://example/private-url"]) == 78
    assert "checksum differs" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", ["ml-exp-worker", "--help"])
    with pytest.raises(SystemExit) as exc: runpy.run_path(launcher.__file__, run_name="__main__")
    assert exc.value.code == 0
