"""NAS-sealed script readiness survives missing logs and fast/uncertain creates."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.backends.sensecore_preparation import script_receipt_worker_sha256
from ml_exp_worker import script_preparation, data_preparation
from ml_exp_worker.data_preparation import identity
from tests.test_cpu_script_preparation import script


@pytest.fixture
def ready(script):
    service, run, _summary, creates, stops = script
    run.update(data_preparation_receipt_mode="callback_v1", data_preparation_receipt_worker_sha256=script_receipt_worker_sha256())
    service.config["public_transfer_base"] = "https://api.example.invalid/ml-expd/api/artifact-transfers"
    files = [{"path": "dataset.bin", "bytes": 17, "sha256": "f" * 64}]
    receipt = {"schema_version": 1, "status": "READY", "definition": run["data_preparation"],
               "preparation_id": run["data_preparation"]["preparation_id"], "files": files,
               "content_sha256": identity(files), "dataset_id": "dataset." + identity(files), "bytes": 17}
    service.rest.logs = lambda *a: pytest.fail("new script readiness cannot depend on provider logs")
    return service, run, receipt, creates, stops


def request_from_body(body):
    argv = shlex.split(body["roles"][0]["startup_script"])
    environment = dict(value.split("=", 1) for value in argv[5:7])
    request = json.loads(environment[script_preparation.REQUEST_ENV])
    return environment[script_preparation.TOKEN_ENV], request


def payload(request, receipt):
    return {key: value for key, value in request.items() if key != "url"} | {"receipt": receipt}


def test_fast_callback_waits_for_confirmed_uid_then_ready_without_any_logs(ready):
    service, run, receipt, creates, _ = ready
    original = service.rest.create
    observations = []

    def create(profile, body, **kw):
        token, request = request_from_body(body)
        operation = request["operation_id"]
        response = service.callback("demo", operation, token, payload(request, receipt))
        assert response["status"] == "PENDING_UID"
        with service.state("demo", operation) as (_, snapshot):
            assert "job_uid" not in snapshot.value
            assert snapshot.value["status"] == "SUBMITTING"
            assert snapshot.value["callback_receipt"] == receipt
            assert token not in json.dumps(snapshot.value)
        observations.append((token, request))
        return original(profile, body, **kw)

    service.rest.create = create
    result = service.ensure("demo", run)
    assert result["status"] == "READY" and len(creates) == 1
    operation = result["operation_id"]
    with service.state("demo", operation) as (_, snapshot):
        assert snapshot.value["job_uid"] == "e" * 32
        assert snapshot.value["receipt_source"] == "NAS_SEALED_CALLBACK"
    token, request = observations[0]
    service.rest.describe = lambda *a: pytest.fail("same sealed READY callback is idempotent without another provider read")
    assert service.callback("demo", operation, token, payload(request, receipt))["status"] == "READY"
    assert service.update("demo", operation, status="WAITING")["status"] == "READY"


def test_pending_callback_survives_lost_create_response_without_replay(ready):
    service, run, receipt, creates, _ = ready
    original = service.rest.create

    def create(profile, body, **kw):
        token, request = request_from_body(body)
        service.callback("demo", request["operation_id"], token, payload(request, receipt))
        original(profile, body, **kw)
        raise OSError("create response lost")

    service.rest.create = create
    with pytest.raises(ApplicationError):
        service.ensure("demo", run)
    service.rest.create = lambda *a, **kw: pytest.fail("never recreate uncertain original CPU task")
    assert service.ensure("demo", run)["status"] == "READY"
    assert len(creates) == 1


def seeded(ready):
    service, run, receipt, _, _ = ready
    definition = service.definition(run)
    operation = service.reference(run)["operation_id"]
    token = "private-token"
    name = "ml-expd-data-script-" + operation.split(".", 1)[1][:32]
    request = service.callback_request("demo", operation, definition, name, token)["request"]
    service.update("demo", operation, status="WAITING", definition=definition,
                   operation_id=operation, preparation_id=receipt["preparation_id"], scheduler_name=name,
                   job_uid="e" * 32, callback_token_sha256=hashlib.sha256(token.encode()).hexdigest(),
                   callback_expires_at=10**12)
    return service, operation, token, request, receipt


@pytest.mark.parametrize("change", ["token", "expired", "operation", "definition", "source", "profile", "files", "bytes", "uid"])
def test_callback_refuses_wrong_identity_expiry_or_replacement(ready, change, monkeypatch):
    service, operation, token, request, receipt = seeded(ready)
    document = deepcopy(payload(request, receipt))
    if change == "token": token = "wrong"
    elif change == "expired": monkeypatch.setattr("ml_exp_server.backends.sensecore_preparation.time.time", lambda: 10**12)
    elif change == "operation": document["operation_id"] = "script." + "0" * 64
    elif change == "definition": document["definition_sha256"] = "0" * 64
    elif change == "source": document["receipt"]["definition"]["source_id"] = "source." + "0" * 64
    elif change == "profile": document["scheduler_name"] = "different"
    elif change == "files": document["receipt"]["files"][0]["sha256"] = "0" * 64
    elif change == "bytes": document["receipt"]["bytes"] = True
    elif change == "uid": service.rest.describe = lambda *a: {"name": a[1], "uid": "replacement", "state": "SUCCEEDED"}
    with pytest.raises(ValueError):
        service.callback("demo", operation, token, document)
    assert service.read("demo", operation)["status"] != "READY"


def test_conflicting_second_receipt_cannot_replace_first(ready):
    service, operation, token, request, receipt = seeded(ready)
    assert service.callback("demo", operation, token, payload(request, receipt))["status"] == "READY"
    changed = deepcopy(receipt)
    changed["files"][0]["sha256"] = "0" * 64
    changed["content_sha256"] = identity(changed["files"])
    changed["dataset_id"] = "dataset." + changed["content_sha256"]
    with pytest.raises(ValueError, match="already sealed"):
        service.callback("demo", operation, token, payload(request, changed))


def test_wrapper_rechecks_readonly_receipt_and_does_not_expose_token_to_prepare(tmp_path, monkeypatch):
    tree = tmp_path / "preparation" / "tree"
    tree.mkdir(parents=True)
    (tree / "dataset.bin").write_bytes(b"data")
    definition = {"preparation_id": "preparation.test"}
    files = data_preparation.inventory(tree)
    receipt = {"schema_version": 1, "status": "READY", "definition": definition,
               "preparation_id": "preparation.test", "files": files, "bytes": 4,
               "content_sha256": identity(files), "dataset_id": "dataset." + identity(files)}
    path = tree.parent / "receipt.json"
    path.write_text(json.dumps(receipt)); path.chmod(0o444)
    request = {"url": "https://example.invalid/callback", "operation_id": "op"}
    monkeypatch.setenv(script_preparation.TOKEN_ENV, "secret")
    monkeypatch.setenv(script_preparation.REQUEST_ENV, json.dumps(request))

    def prepare(*a):
        assert script_preparation.TOKEN_ENV not in os.environ
        assert script_preparation.REQUEST_ENV not in os.environ
        return tree, {"untrusted_summary": True}

    monkeypatch.setitem(sys.modules, "data_preparation", SimpleNamespace(
        prepare=prepare, inventory=data_preparation.inventory, identity=identity))
    monkeypatch.setitem(sys.modules, "data_input", SimpleNamespace(deliver=lambda *a: None))
    monkeypatch.setitem(sys.modules, "worker", SimpleNamespace(link_path=lambda *a: None))
    reported = []
    monkeypatch.setattr(script_preparation, "report", lambda *a: reported.append(a))
    frozen = {"definition": definition | {"inputs": []}, "cache": str(tmp_path)}
    receipt["definition"] = frozen["definition"]
    path.chmod(0o644); path.write_text(json.dumps(receipt)); path.chmod(0o444)
    script_preparation.main(["wrapper", json.dumps(frozen)])
    assert reported == [(request, "secret", receipt)]
    (tree / "dataset.bin").write_bytes(b"evil")
    with pytest.raises(ValueError, match="IDENTITY"):
        script_preparation.sealed_receipt(tree, frozen["definition"])


def test_narrow_callback_route_requires_its_own_capability_and_bounds_body(ready, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ml_exp_server.api.boundary import install_http_boundary
    from ml_exp_server.api import script_preparation_routes
    service, operation, token, request, receipt = seeded(ready)
    app = FastAPI()
    app.state.workspace_owner = True
    app.state.runtime = SimpleNamespace(config=SimpleNamespace(
        container_execution=SimpleNamespace(artifact_store_file="private.json"),
        project_registry_root_path=lambda: service.root.parent,
    ))
    monkeypatch.setattr(script_preparation_routes, "SenseCoreScriptPreparation", lambda *a: service)
    install_http_boundary(app, "admin-private")
    app.include_router(script_preparation_routes.router)
    path = "/api/data-script-transfers/demo/" + operation
    headers = {"Authorization": "Bearer " + token}
    with TestClient(app) as client:
        assert client.put(path, json={}).status_code == 401
        assert client.put(path, json={}, headers={"Authorization": "Bearer admin-private"}).status_code == 401
        assert client.post(path, json={}).status_code == 401
        assert client.put(path + "/extra", json={}, headers=headers).status_code == 401
        assert client.put(path, content=b"not json", headers=headers).status_code == 422
        assert client.put(path, json=[], headers=headers).status_code == 422
        assert client.put(path, content=b"x" * (8 * 1024 ** 2 + 16385), headers=headers).status_code == 413
        response = client.put(path, json=payload(request, receipt), headers=headers)
        assert response.status_code == 200 and response.json()["status"] == "READY"
        app.state.runtime.config.container_execution.artifact_store_file = None
        assert client.put(path, json={}, headers=headers).status_code == 404


@pytest.mark.parametrize("responses,success", [
    ([503, 200], True), ([302] * 5, False), (["invalid", 200], True), (["different", 200], True), (["list", 200], True), (["network"] * 5, False),
])
def test_wrapper_callback_retries_only_same_receipt_and_never_follows_redirects(monkeypatch, responses, success):
    from io import BytesIO
    values = iter(responses)
    sent, closed = [], []
    request = {"url": "https://example.invalid/api/data-script-transfers/demo/op", "operation_id": "op"}
    receipt = {"preparation_id": "prep"}

    class Connection:
        def request(self, method, path, *, body, headers):
            sent.append((method, path, body, headers))
        def getresponse(self):
            value = next(values)
            if value == "network": raise OSError("private network details")
            raw = b"not json" if value == "invalid" else b"[]" if value == "list" else json.dumps({
                "operation_id": "other" if value == "different" else "op", "preparation_id": "prep", "status": "PENDING_UID",
            }).encode()
            return SimpleNamespace(status=200 if value in {"invalid", "different", "list"} else value, read=BytesIO(raw).read)
        def close(self): closed.append(True)

    monkeypatch.setitem(sys.modules, "worker_http", SimpleNamespace(https_connection=lambda *a, **kw: Connection()))
    monkeypatch.setattr(script_preparation.time, "sleep", lambda seconds: None)
    if success:
        script_preparation.report(request, "secret", receipt)
    else:
        with pytest.raises(ValueError, match="DATA_SCRIPT_CALLBACK_PENDING"):
            script_preparation.report(request, "secret", receipt)
    assert len(closed) == len(sent)
    assert len({body for _, _, body, _ in sent}) == 1
    assert all(headers["Authorization"] == "Bearer secret" for *_, headers in sent)
    assert all(path == "/api/data-script-transfers/demo/op" for _, path, _, _ in sent)


@pytest.mark.parametrize("url", ["http://example.invalid/path", "https://user:password@example.invalid/path", "https://example.invalid/path?token=secret", "https://example.invalid/path#fragment"])
def test_wrapper_rejects_untrusted_callback_origins(monkeypatch, url):
    monkeypatch.setitem(sys.modules, "worker_http", SimpleNamespace(https_connection=lambda *a, **kw: pytest.fail("invalid origin")))
    with pytest.raises(ValueError, match="DATA_SCRIPT_CALLBACK_ORIGIN"):
        script_preparation.report({"url": url}, "secret", {})


def test_v2_missing_worker_pin_or_changed_wrapper_cannot_create(ready):
    service, run, _, creates, _ = ready
    run.pop("data_preparation_receipt_worker_sha256")
    with pytest.raises(ValueError, match="must be frozen"):
        service.ensure("demo", run)
    run["data_preparation_receipt_worker_sha256"] = "0" * 64
    with pytest.raises(ApplicationError):
        service.ensure("demo", run)
    assert not creates


@pytest.mark.parametrize("kind", ["nonobject", "empty", "parent", "duplicate", "origin"])
def test_callback_rejects_invalid_inventory_and_failed_operations(ready, kind):
    service, operation, token, request, receipt = seeded(ready)
    document = deepcopy(payload(request, receipt))
    if kind == "nonobject": document = []
    elif kind == "empty": document["receipt"]["files"] = []
    elif kind == "parent": document["receipt"]["files"][0]["path"] = "../dataset.bin"
    elif kind == "duplicate": document["receipt"]["files"] *= 2
    elif kind == "origin":
        service.config["public_transfer_base"] = "http://untrusted.invalid/path"
        with pytest.raises(ValueError):
            service.callback_request("demo", operation, service.definition(ready[1]), "name", token)
        return
    with pytest.raises(ValueError):
        service.callback("demo", operation, token, document)


def test_valid_callback_wins_a_concurrent_observation_failure(ready):
    service, run, receipt, _, _ = ready
    original = service.rest.create

    def create(profile, body, **kw):
        created = original(profile, body, **kw)
        token, request = request_from_body(body)
        service.update("demo", request["operation_id"], job_uid=created["uid"])
        service.callback("demo", request["operation_id"], token, payload(request, receipt))
        raise OSError("late create transport exception")

    service.rest.create = create
    assert service.ensure("demo", run)["status"] == "READY"


def test_receipt_must_be_readonly_and_stable(tmp_path, monkeypatch):
    tree = tmp_path / "tree"
    tree.mkdir()
    path = tmp_path / "receipt.json"
    path.write_text("{}")
    monkeypatch.setitem(sys.modules, "data_preparation", data_preparation)
    with pytest.raises(ValueError, match="NOT_SEALED"):
        script_preparation.sealed_receipt(tree, {})
    path.chmod(0o444)
    original = script_preparation.os.fstat
    calls = []

    def changed(fd):
        result = original(fd)
        calls.append(True)
        if len(calls) == 2:
            return SimpleNamespace(st_size=result.st_size, st_mtime_ns=result.st_mtime_ns,
                                   st_ctime_ns=result.st_ctime_ns + 1)
        return result

    monkeypatch.setattr(script_preparation.os, "fstat", changed)
    with pytest.raises(ValueError, match="CHANGED"):
        script_preparation.sealed_receipt(tree, {})


def test_inline_wrapper_entrypoint_success_and_sanitized_failure(tmp_path, monkeypatch, capsys):
    import runpy
    from io import BytesIO
    tree = tmp_path / "prep" / "tree"
    tree.mkdir(parents=True)
    (tree / "data").write_bytes(b"data")
    download = {"preparation_id": "prep", "inputs": [{"asset_id": "asset", "mount_path": "/inputs/data"}]}
    files = data_preparation.inventory(tree)
    receipt = {"schema_version": 1, "status": "READY", "definition": download,
               "preparation_id": "prep", "files": files, "bytes": 4,
               "content_sha256": identity(files), "dataset_id": "dataset." + identity(files)}
    path = tree.parent / "receipt.json"
    path.write_text(json.dumps(receipt)); path.chmod(0o444)
    request = {"url": "https://example.invalid/path", "operation_id": "operation"}
    monkeypatch.setitem(sys.modules, "data_preparation", SimpleNamespace(
        prepare=lambda *a: (tree, {}), inventory=data_preparation.inventory, identity=identity))
    links = []
    monkeypatch.setitem(sys.modules, "data_input", SimpleNamespace(deliver=lambda *a: tree))
    monkeypatch.setitem(sys.modules, "worker", SimpleNamespace(link_path=lambda *a: links.append(a)))
    connection = SimpleNamespace(
        request=lambda *a, **kw: None, close=lambda: None,
        getresponse=lambda: SimpleNamespace(status=200, read=BytesIO(json.dumps({
            "operation_id": "operation", "preparation_id": "prep", "status": "READY",
        }).encode()).read),
    )
    monkeypatch.setitem(sys.modules, "worker_http", SimpleNamespace(https_connection=lambda *a, **kw: connection))
    monkeypatch.setenv(script_preparation.TOKEN_ENV, "private-token")
    monkeypatch.setenv(script_preparation.REQUEST_ENV, json.dumps(request))
    monkeypatch.setattr(sys, "argv", ["wrapper", json.dumps({"definition": download, "cache": str(tmp_path)})])
    runpy.run_module("ml_exp_worker.script_preparation", run_name="__main__")
    assert links == [(tree, Path("/inputs/data"))]
    assert "ACCEPTED" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        runpy.run_module("ml_exp_worker.script_preparation", run_name="__main__")
    assert error.value.code == 74
    output = capsys.readouterr().out
    assert "FAILED" in output and "private-token" not in output


@pytest.mark.parametrize("state", ["FAILED", "SUSPENDED"])
def test_sealed_data_readiness_survives_control_job_failure_without_rewriting_job_state(ready, state):
    service, operation, token, request, receipt = seeded(ready)
    service.update("demo", operation, status="FAILED", diagnostic="CPU_DATA_SCRIPT_EXITED", job_state=state)
    service.rest.describe = lambda *a: {"name": a[1], "uid": "e" * 32, "state": state, "exit_code": 74}
    assert service.callback("demo", operation, token, payload(request, receipt))["status"] == "READY"
    value = service.read("demo", operation)
    assert value["status"] == "READY" and value["job_state"] == state and value["job_exit_code"] == 74
    assert value["control_diagnostic"] == "CPU_DATA_SCRIPT_EXITED"
    events = [json.loads(line) for line in (service.root / "demo" / (operation + ".jsonl")).read_text().splitlines()]
    assert "FAILED" in [event.get("status") for event in events]
