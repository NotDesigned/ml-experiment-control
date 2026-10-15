"""Reconciliation preserves only positive evidence that a build never started."""
from copy import deepcopy

import pytest

from ml_exp_server import container_execution
from ml_exp_server.builds.builder_client import (
    BuildReceiptMissingError, BuildStorageError, BuildTransportError,
)
from ml_exp_server.builds.runtime_jobs import recover_interrupted_builds
from ml_exp_server.container_execution import ContainerExecutionService
from tests.test_container_api import client, import_source, prepare_runtime, wait_runtime


def prepared(api):
    source = import_source(api)
    return prepare_runtime(api, {"source_id": source["source_id"], "entrypoint": ["python3"]}).json()


def failed_preflight(api, monkeypatch, code="BUILD_STORAGE_INSUFFICIENT"):
    value = prepared(api)
    endpoint = "/api/projects/demo/runtimes/" + value["runtime_id"]
    original = container_execution.builder_request
    operations = []

    def fail(_socket, payload):
        operations.append(payload["operation"])
        raise BuildStorageError(code, {"scheduler_submitted": False})

    monkeypatch.setattr(container_execution, "builder_request", fail)
    assert api.post(endpoint + "/execute", json={"confirmation": value["confirmation"]}).status_code == 202
    failed = wait_runtime(api, endpoint)
    assert failed["build_error"]["retry_safe"] is True
    return value, endpoint, original, operations, failed["build_error"]


@pytest.mark.parametrize("code", ["BUILD_STORAGE_INSUFFICIENT", "BUILD_STORAGE_UNCHECKED"])
def test_missing_receipt_preserves_durable_safe_evidence_and_explicit_retry(client, monkeypatch, code):
    value, endpoint, original, operations, evidence = failed_preflight(client, monkeypatch, code)

    def missing(_socket, payload):
        operations.append(payload["operation"])
        raise BuildReceiptMissingError("image bundle has not been published")

    monkeypatch.setattr(container_execution, "builder_request", missing)
    service = ContainerExecutionService(client.app.state.runtime)
    pending = service.begin_execute("demo", value["runtime_id"], value["confirmation"], reconcile=True)
    assert service.read("demo", value["runtime_id"])["build_error"] == evidence
    assert service.finish_execute(pending)["build_error"] == evidence
    # Reopen the actual durable store, rather than relying on the pending copy.
    reopened = ContainerExecutionService(client.app.state.runtime)
    assert reopened.read("demo", value["runtime_id"])["build_error"] == evidence

    def successful(socket, payload):
        operations.append(payload["operation"])
        return original(socket, payload)

    monkeypatch.setattr(container_execution, "builder_request", successful)
    assert client.post(endpoint + "/execute", json={"confirmation": value["confirmation"]}).status_code == 202
    result = wait_runtime(client, endpoint)
    assert result["status"] == "READY" and result["build_error"] is None
    assert operations == ["build", "get", "build"]


def test_daemon_recovery_keeps_preflight_evidence_during_readonly_reconcile(client, monkeypatch):
    value, _, _, _, evidence = failed_preflight(client, monkeypatch)
    service = ContainerExecutionService(client.app.state.runtime)
    service.begin_execute("demo", value["runtime_id"], value["confirmation"], reconcile=True)
    assert recover_interrupted_builds(service) == 1
    assert service.read("demo", value["runtime_id"])["build_error"] == evidence


def test_successful_receipt_reconcile_clears_prior_preflight_evidence(client, monkeypatch):
    value, endpoint, original, operations, _ = failed_preflight(client, monkeypatch)

    def published(socket, payload):
        operations.append(payload["operation"])
        return original(socket, payload)

    monkeypatch.setattr(container_execution, "builder_request", published)
    result = ContainerExecutionService(client.app.state.runtime).execute(
        "demo", value["runtime_id"], value["confirmation"], reconcile=True)
    assert result["status"] == "READY" and result["build_error"] is None
    assert operations == ["build", "get"]
    assert client.get(endpoint).json()["build_error"] is None


@pytest.mark.parametrize("failure", ["identity", "worker", "dockerfile", "shape", "unknown", "transport", "storage"])
def test_contradictory_or_unknown_receipt_lookup_never_authorizes_new_build(client, monkeypatch, failure):
    value, endpoint, original, operations, _ = failed_preflight(client, monkeypatch)

    def reconcile(socket, payload):
        operations.append(payload["operation"])
        if failure == "unknown":
            raise ValueError("unknown private failure")
        if failure == "transport":
            raise BuildTransportError("BUILD_SESSION_TIMEOUT", {"publication_uncertain": True})
        if failure == "storage":
            raise BuildStorageError("BUILD_STORAGE_UNCHECKED", {"scheduler_submitted": False})
        if failure == "shape":
            return None
        receipt = deepcopy(original(socket, payload))
        if failure == "identity":
            receipt["project"] = "other"
        elif failure == "worker":
            receipt["worker_sha256"] = "0" * 64
        else:
            receipt["dockerfile_sha256"] = "0" * 64
        return receipt

    monkeypatch.setattr(container_execution, "builder_request", reconcile)
    result = ContainerExecutionService(client.app.state.runtime).execute(
        "demo", value["runtime_id"], value["confirmation"], reconcile=True)
    assert result["status"] == "RECONCILE_REQUIRED"
    assert not (result.get("build_error") or {}).get("retry_safe", False)
    assert client.post(endpoint + "/execute", json={"confirmation": value["confirmation"]}).status_code == 409
    assert operations == ["build", "get"]


def test_preflight_error_on_receipt_lookup_cannot_override_uncertain_publication(client, monkeypatch):
    value = prepared(client)
    service = ContainerExecutionService(client.app.state.runtime)
    operations = []

    def uncertain_then_preflight(_socket, payload):
        operations.append(payload["operation"])
        if payload["operation"] == "build":
            raise ValueError("unknown publication effect")
        raise BuildStorageError("BUILD_STORAGE_INSUFFICIENT", {"scheduler_submitted": False})

    monkeypatch.setattr(container_execution, "builder_request", uncertain_then_preflight)
    service.execute("demo", value["runtime_id"], value["confirmation"])
    result = service.execute("demo", value["runtime_id"], value["confirmation"], reconcile=True)
    assert result["build_error"]["retry_safe"] is False
    endpoint = "/api/projects/demo/runtimes/" + value["runtime_id"]
    assert client.post(endpoint + "/execute", json={"confirmation": value["confirmation"]}).status_code == 409
    assert operations == ["build", "get"]


@pytest.mark.parametrize("failure", ["unknown", "transport", "disk"])
def test_missing_receipt_cannot_turn_uncertain_publication_into_safe_retry(client, monkeypatch, failure):
    value = prepared(client)
    endpoint = "/api/projects/demo/runtimes/" + value["runtime_id"]
    operations = []

    def uncertain(_socket, payload):
        operations.append(payload["operation"])
        if payload["operation"] == "get":
            raise BuildReceiptMissingError("image bundle has not been published")
        if failure == "transport":
            raise BuildTransportError("BUILD_SESSION_TIMEOUT", {"publication_uncertain": True})
        if failure == "disk":
            raise BuildStorageError("BUILD_DISK_EXHAUSTED", {"publication_uncertain": True})
        raise ValueError("unknown publication effect")

    monkeypatch.setattr(container_execution, "builder_request", uncertain)
    assert client.post(endpoint + "/execute", json={"confirmation": value["confirmation"]}).status_code == 202
    failed = wait_runtime(client, endpoint)
    evidence = failed.get("build_error")
    result = ContainerExecutionService(client.app.state.runtime).execute(
        "demo", value["runtime_id"], value["confirmation"], reconcile=True)
    assert result.get("build_error") == evidence
    assert not (result.get("build_error") or {}).get("retry_safe", False)
    assert client.post(endpoint + "/execute", json={"confirmation": value["confirmation"]}).status_code == 409
    assert operations == ["build", "get"]
