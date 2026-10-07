"""Files-only recovery is complete and hash-verified; controls never train."""
import hashlib
import io
import json
from types import SimpleNamespace
import pytest
from ml_exp_client.api import ClientError, _download_proxy
from ml_exp_client.cli import main


@pytest.mark.parametrize("case", ["success", "bad-sha", "no-proof", "server-error", "empty", "truncated"])
def test_files_only_download_requires_complete_sha_evidence(tmp_path, case):
    content = b"final weight"
    file = {"path": "outputs/chosen.bin", "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    if case == "bad-sha": file["sha256"] = "a" * 64
    if case == "no-proof": file.pop("sha256")
    listing = {"files": [] if case == "empty" else [file], "truncated": case == "truncated"}
    class API:
        def call(self, path):
            assert path.endswith("/files?checksums=true"); return listing
        def open(self, path):
            if path.endswith("/artifacts/archive"):
                raise ClientError("archive missing", status=503 if case == "server-error" else 404)
            return io.BytesIO(content)
    if case != "success":
        with pytest.raises(ClientError):
            _download_proxy(API(), "demo", "run", "attempt-001", tmp_path / "out")
    else:
        result = _download_proxy(API(), "demo", "run", "attempt-001", tmp_path / "out")
        assert result["transport"] == "http-files" and result["archive_sha256"] is None
        assert json.loads((tmp_path / "out/verification.json").read_text()) == result


@pytest.mark.parametrize("flag", [None, "--start", "--retry", "--reconcile"])
def test_client_collection_queries_or_authorizes_only_original_attempt(monkeypatch, capsys, flag):
    calls = []
    class API:
        def __init__(self, *args): pass
        def negotiate(self): return {"capabilities": ["result-collection.v1"]}
        def call(self, path, **kwargs):
            calls.append((path, kwargs)); return {"status": "QUEUED"}
    monkeypatch.setattr("ml_exp_client.cli.Client", API)
    monkeypatch.setenv("ML_EXPD_API_URL", "https://api.example")
    monkeypatch.setenv("ML_EXPD_API_TOKEN", "private")
    args = ["collect", "--project", "demo", "--run", "run", "--attempt", "attempt-001"]
    if flag is not None: args.append(flag)
    assert main(args) == 0
    assert calls[0][0] == "/api/runs/demo/run/attempts/attempt-001/collection"
    assert ("data" in calls[0][1]) == (flag is not None)
    assert "private" not in capsys.readouterr().out


def test_collection_requires_server_capability(monkeypatch, capsys):
    monkeypatch.setattr("ml_exp_client.cli.Client", lambda *args: SimpleNamespace(negotiate=lambda: {"capabilities": []}))
    assert main(["collect", "--project", "demo", "--run", "run", "--attempt", "attempt-001"]) == 2
    assert "result-collection.v1" in capsys.readouterr().err
