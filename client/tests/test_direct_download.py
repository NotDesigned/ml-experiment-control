"""Object downloads keep API credentials private and validate before extraction."""
import hashlib
import io
import json
import tarfile
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from ml_exp_client.api import Client, ClientError, download


def archive(files):
    body = io.BytesIO()
    with tarfile.open(fileobj=body, mode="w") as tar:
        for name, content in files.items():
            member = tarfile.TarInfo(name); member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
    return body.getvalue()


@pytest.mark.parametrize("historical", [False, True])
def test_direct_download_transfers_archive_once_and_verifies_on_client(tmp_path, historical):
    files = {"checkpoint.pt": b"weights", "metrics.jsonl": b'{"step":8}\n'}
    body = archive(files)
    ticket = {"transport": "s3-presigned-get", "url": "https://objects.example/exact?signed=private",
              "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
              "files": [{"path": name, "bytes": len(data), **({} if historical else {"sha256": hashlib.sha256(data).hexdigest()})}
                        for name, data in files.items()]}
    calls = []
    class API:
        def call(self, path):
            calls.append(path); assert path.endswith("/artifacts/download")
            return ticket
        def open_object(self, url):
            calls.append(url); return io.BytesIO(body)
    report = download(API(), "demo", "run-a", "attempt-001", tmp_path / "out")
    assert len(calls) == 2 and report["transport"] == "s3-presigned-get"
    assert report["archive_sha256"] == ticket["sha256"]
    assert (tmp_path / "out/outputs/checkpoint.pt").read_bytes() == b"weights"
    assert "private" not in (tmp_path / "out/verification.json").read_text()


def test_signed_download_never_forwards_bearer_or_echoes_signed_urls():
    client = Client("https://api.example", "private-bearer")
    requests = []
    def open(request, **kwargs):
        requests.append(request)
        return io.BytesIO(b"object")
    client.opener = SimpleNamespace(open=open)
    with client.open_object("https://objects.example/exact?signature=private") as body:
        assert body.read() == b"object"
    assert requests[0].header_items() == []
    client.opener.open = lambda *a, **kw: (_ for _ in ()).throw(HTTPError("https://private-url", 403, "secret-signature", {}, None))
    with pytest.raises(ClientError) as error:
        client.open_object("https://objects.example/exact?signature=private")
    assert "private" not in str(error.value) and "secret" not in str(error.value)


@pytest.mark.parametrize("url", ["http://objects.example/x", "https://user:pass@objects.example/x", "https://objects.example/x#f", "https://"])
def test_signed_download_rejects_unsafe_transport(url):
    with pytest.raises(ClientError, match="HTTPS"):
        Client("https://api.example", "token").open_object(url)


@pytest.mark.parametrize("status", [401, 404, 409])
def test_download_fallback_occurs_only_for_missing_feature(tmp_path, monkeypatch, status):
    class API:
        def call(self, path):
            raise ClientError("HTTP error", status=status)
    monkeypatch.setattr("ml_exp_client.api._download_proxy", lambda *args: "legacy-download")
    if status == 404:
        assert download(API(), "demo", "run-a", "attempt-001", tmp_path) == "legacy-download"
    else:
        with pytest.raises(ClientError):
            download(API(), "demo", "run-a", "attempt-001", tmp_path)


@pytest.mark.parametrize("failure", ["archive-sha", "archive-size", "file-sha", "file-size", "unexpected", "missing", "escape", "duplicate", "transport"])
def test_direct_download_rejects_corrupt_receipts_and_contents(tmp_path, failure):
    body = archive({"checkpoint.pt": b"weights"})
    ticket = {"transport": "s3-presigned-get", "url": "https://objects.example/exact?signature=private",
              "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
              "files": [{"path": "checkpoint.pt", "bytes": 7}]}
    if failure == "archive-sha": ticket["sha256"] = "a" * 64
    if failure == "archive-size": ticket["bytes"] += 1
    if failure == "file-sha": ticket["files"][0]["sha256"] = "a" * 64
    if failure == "file-size": ticket["files"][0]["bytes"] += 1
    if failure == "unexpected": ticket["files"] = []
    if failure == "missing": ticket["files"].append({"path": "missing", "bytes": 0})
    if failure == "escape": ticket["files"][0]["path"] = "../outside"
    if failure == "duplicate": ticket["files"] *= 2
    if failure == "transport": ticket["transport"] = "untrusted"
    class API:
        def call(self, path): return ticket
        def open_object(self, url): return io.BytesIO(body)
    with pytest.raises(ClientError):
        download(API(), "demo", "run-a", "attempt-001", tmp_path / "out")
    assert not (tmp_path / "outside").exists()
