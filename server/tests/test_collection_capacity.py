"""Result observation cannot bypass peak admission, including archive repair."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ml_exp_server.storage_capacity import CapacityError
from tests.test_result_collection import recovery, IDENTITY, output
from tests.test_artifact_store import storage


@pytest.mark.parametrize("remote", [False, True])
def test_capacity_rejection_precedes_copy_and_archive_and_keeps_scientific_files(recovery, monkeypatch, remote):
    service, root, _, _, objects, *_ = recovery
    local = None if remote else output(root)
    copy = Mock()
    monkeypatch.setattr(service, "read_wyd_outputs", copy)
    monkeypatch.setattr(service, "wyd_output_bytes", lambda *a: 1024 ** 3)
    monkeypatch.setattr("shutil.disk_usage", lambda _: SimpleNamespace(free=1024 ** 3))
    service.begin(*IDENTITY); service.finish(IDENTITY)
    state = service.read(*IDENTITY)
    assert state["status"] == "FAILED" and state["diagnostic"] == "UPLOAD_STORAGE"
    assert state["storage_capacity"]["available_bytes"] == 1024 ** 3
    assert state["storage_capacity"]["required_free_bytes"] > 1024 ** 3
    assert not objects and not copy.called
    assert not list(service.capacity().root.glob("claim-*"))
    if local: assert (local / "weights.bin").read_bytes() == b"weights"


def test_remote_size_requires_valid_stats_and_handles_locale_separators(recovery, monkeypatch):
    from ml_exp_server.results.errors import RecoveryError
    service, _, _, _, _, definition, _ = recovery
    monkeypatch.setattr("ml_exp_server.backends.wyd.run_process", lambda *a, **kw: SimpleNamespace(stdout="Total file size: 1,234,567 bytes\n"))
    assert service.wyd_output_bytes(definition["backend"], definition, "attempt-001") == 1234567
    monkeypatch.setattr("ml_exp_server.backends.wyd.run_process", lambda *a, **kw: SimpleNamespace(stdout="unknown"))
    with pytest.raises(RecoveryError, match="cannot determine"):
        service.wyd_output_bytes(definition["backend"], definition, "attempt-001")


def test_archive_peak_ignores_hidden_undeclared_and_linked_files(recovery):
    service, root, *_ = recovery
    local = output(root)
    (local / "hidden").mkdir(); (local / "hidden/.secret").write_bytes(b"secret")
    (local / ".part").write_bytes(b"partial")
    (local / "link.bin").symlink_to(local / "weights.bin")
    assert service.archive_peak(local, ["**/weights.bin"]) == 3 * (512 + 8192 + 10240) + 4096
    assert service.archive_peak(local, ["**/link.bin"]) == 3 * 10240


def test_transfer_failure_removes_only_its_owned_scratch(recovery, monkeypatch):
    service, root, _, _, _, definition, _ = recovery
    attempt = root / "attempts/attempt-001"
    unknown = attempt / ".result-recovery-existing"; unknown.mkdir(); (unknown / "unique").write_bytes(b"keep")
    def fail(command, **options):
        Path(command[-1], "partial").write_bytes(b"bytes")
        raise TimeoutError()
    monkeypatch.setattr("ml_exp_server.backends.wyd.run_process", fail)
    with pytest.raises(TimeoutError):
        service.read_wyd_outputs(definition["backend"], definition, "attempt-001", attempt / "recovered_outputs")
    assert list(attempt.glob(".result-recovery-*")) == [unknown]
    assert (unknown / "unique").read_bytes() == b"keep"


@pytest.mark.parametrize("verification", ["valid", "corrupt", "new-receipt"])
def test_published_copy_eviction_requires_complete_object_and_matching_files(recovery, monkeypatch, verification):
    service, root, _, _, objects, *_ = recovery
    recovered = root / "attempts/attempt-001/recovered_outputs"
    recovered.mkdir(); (recovered / "weights.bin").write_bytes(b"weights")
    (recovered / ".partial").write_bytes(b"unique")
    if verification == "corrupt":
        monkeypatch.setattr(type(service.objects.client()), "download_fileobj", lambda self, bucket, key, sink, **kw: sink.write(b"corrupt"))
    service.begin(*IDENTITY); service.finish(IDENTITY)
    assert service.read(*IDENTITY)["status"] == "AVAILABLE" and len(objects) == 1
    assert (recovered / ".partial").read_bytes() == b"unique"
    with service.objects.record(*IDENTITY) as (path, value):
        assert not value.get("retention")  # Server verification never fabricates client ACK.
        receipt = value["receipt"]
        if verification == "corrupt":
            assert value["cache_cleanup"]["error"] == "ValueError"
        if verification == "new-receipt":
            (recovered / "weights.bin").write_bytes(b"late-changed")
            value["receipt"] = {**receipt, "received_at": "new-generation"}
            path.write_text(json.dumps(value))
    if verification == "new-receipt": service.evict_published_copies(IDENTITY, receipt)
    assert (recovered / "weights.bin").exists() is (verification != "valid")


def test_concurrent_collection_requests_share_one_attempt_operation(recovery):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    service, root, *_ = recovery
    output(root)
    barrier = threading.Barrier(8)
    def begin(_):
        barrier.wait(timeout=5)
        return service.begin(*IDENTITY)
    with ThreadPoolExecutor(8) as pool: results = list(pool.map(begin, range(8)))
    assert sum(pending is not None for _, pending in results) == 1
    assert service.read(*IDENTITY)["attempts"] == 1


def test_cache_verification_uses_official_s3_download_without_materializing_archive(recovery, monkeypatch):
    import boto3
    from botocore.response import StreamingBody
    from botocore.stub import Stubber
    import io
    from tests.test_artifact_store import archive
    service, root, token, _, objects, *_ = recovery
    data = archive({"weights.bin": b"weights"})
    receipt = service.objects.receive(*IDENTITY, token, io.BytesIO(data), len(data))
    cache = root / "attempts/attempt-001/recovered_outputs"; cache.mkdir()
    (cache / "weights.bin").write_bytes(b"weights")
    client = boto3.client("s3", endpoint_url="https://unused.example", region_name="us-east-1",
                          aws_access_key_id="test", aws_secret_access_key="test")
    seen = []
    def parameters(params, **kwargs): seen.append((params["Bucket"], params["Key"]))
    client.meta.events.register("before-parameter-build.s3", parameters)
    download = client.download_fileobj
    def checked_download(bucket, key, sink, **options):
        assert sink.writable()
        return download(bucket, key, sink, **options)
    monkeypatch.setattr(client, "download_fileobj", checked_download)
    with Stubber(client) as stub:
        stub.add_response("head_object", {"ContentLength": len(data)})
        stub.add_response("get_object", {"ContentLength": len(data), "Body": StreamingBody(io.BytesIO(data), len(data))})
        monkeypatch.setattr(service.objects, "client", lambda: client)
        service.evict_published_copies(IDENTITY, receipt)
        stub.assert_no_pending_responses()
    assert seen == [("results", receipt["object_key"])] * 2
    assert not cache.exists() and len(objects) == 1


def test_cold_file_restoration_reserves_space_without_affecting_metadata(recovery, monkeypatch):
    import io
    from tests.test_artifact_store import archive
    service, root, token, *_ = recovery
    data = archive({"weights.bin": b"weights"})
    service.objects.receive(*IDENTITY, token, io.BytesIO(data), len(data))
    monkeypatch.setattr("shutil.disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(CapacityError): service.objects.restore_cache(*IDENTITY)
    assert service.read(*IDENTITY)["status"] == "AVAILABLE"
    assert not (root / "attempts/attempt-001/uploaded_outputs").exists()


def test_legacy_chunked_upload_grows_capacity_claim_and_cleans_it(recovery):
    from fastapi.testclient import TestClient
    from ml_exp_server.api.app import create_app
    from tests.test_artifact_store import archive
    service, _, token, _, objects, *_ = recovery
    data = archive({"weights.bin": b"weights"})
    with TestClient(create_app(service.runtime.config, poll=False)) as client:
        response = client.put("/api/artifact-transfers/demo/run-a/attempt-001",
                              content=iter([data[:100], data[100:]]),
                              headers={"Authorization": "Bearer " + token})
        assert response.status_code == 200
    assert len(objects) == 1 and not list(service.capacity().root.glob("claim-*"))
