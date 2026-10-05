"""Actual producer exit semantics and deterministic HTTPS transport failures."""

import http.client
import io
import json
import os
import runpy
import signal
import sys
import tarfile
from types import SimpleNamespace

import pytest

from ml_exp_server import container_worker as worker


@pytest.mark.parametrize("url", ["http://example/upload", "https://user:pass@example/upload", "https://example/upload?key=value", "https://example/upload#fragment"])
def test_worker_rejects_nonfixed_https_upload_destination(url):
    with pytest.raises(ValueError, match="fixed HTTPS"):
        worker.upload(url, "test-capability", io.BytesIO(b"payload"), 7)


@pytest.mark.parametrize("responses,error", [
    ([503, 200], None), ([OSError, 200], None), ([http.client.HTTPException, 200], None),
    ([403], ValueError), ([503, 503, 503], ValueError), ([OSError] * 3, OSError),
])
def test_https_upload_retries_same_bytes_and_closes_every_connection(monkeypatch, responses, error):
    connections = []
    class Connection:
        def __init__(self, *args, **kwargs):
            self.response = responses[len(connections)]
            self.headers = {}
            self.body = b""
            self.closed = False
            connections.append(self)
        def putrequest(self, method, path):
            assert method == "PUT" and path == "/upload"
        def putheader(self, key, value):
            self.headers[key] = value
        def endheaders(self):
            pass
        def send(self, chunk):
            self.body += chunk
        def getresponse(self):
            if isinstance(self.response, type):
                raise self.response("transport unavailable")
            return SimpleNamespace(status=self.response, read=lambda limit: b"")
        def close(self):
            self.closed = True
    monkeypatch.setattr(worker.http.client, "HTTPSConnection", Connection)
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: None)
    data = io.BytesIO(b"same artifact bytes")
    if error:
        with pytest.raises(error):
            worker.upload("https://upload.example:443/upload", "test-capability", data, 19)
    else:
        worker.upload("https://upload.example:443/upload", "test-capability", data, 19)
    assert all(connection.closed and connection.body == b"same artifact bytes" for connection in connections)
    assert all(connection.headers["Authorization"] == "Bearer test-capability" for connection in connections)


def configure(monkeypatch, tmp_path, limit=2 * 1024 * 1024):
    output = tmp_path / "outputs"
    monkeypatch.setenv("OUTPUT_DIR", str(output))
    monkeypatch.setenv("ML_EXPD_UPLOAD_URL", "https://upload.example/upload")
    monkeypatch.setenv("ML_EXPD_UPLOAD_TOKEN", "test-capability")
    monkeypatch.setenv("ML_EXPD_UPLOAD_LIMIT", str(limit))
    monkeypatch.setenv("ML_EXPD_OUTPUT_PATTERNS", '["**/*"]')
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    return output


@pytest.mark.parametrize("program_code,upload_failure,expected", [(0, False, 0), (3, False, 3), (-15, False, 143), (0, True, 74), (3, True, 3)])
def test_producer_preserves_program_failure_and_hides_upload_capability(monkeypatch, tmp_path, capsys, program_code, upload_failure, expected):
    output = configure(monkeypatch, tmp_path)
    uploaded = []
    def upload(url, token, stream, length):
        assert token == "test-capability"
        if upload_failure:
            raise OSError("upload unavailable")
        stream.seek(0)
        with tarfile.open(fileobj=stream) as archive:
            content = json.loads(archive.extractfile("environment.json").read())
            assert content == []
        uploaded.append(length)
    monkeypatch.setattr(worker, "upload", upload)
    command = [sys.executable, "-c", "import json,os,signal;from pathlib import Path;"
               "Path(os.environ['OUTPUT_DIR'],'environment.json').write_text(json.dumps([k for k in os.environ if k.startswith('ML_EXPD_UPLOAD') or k=='ML_EXPD_OUTPUT_PATTERNS']));"
               + ("os.kill(os.getpid(),signal.SIGTERM)" if program_code < 0 else f"raise SystemExit({program_code})")]
    assert worker.main(command) == expected
    text = capsys.readouterr()
    assert ("ML_EXPD_ARTIFACT_UPLOAD=FAILED" in text.err) == upload_failure
    assert "test-capability" not in text.out + text.err


def test_empty_archive_still_obeys_transport_limit(monkeypatch, tmp_path, capsys):
    configure(monkeypatch, tmp_path, limit=5000)
    assert worker.main([sys.executable, "-c", "pass"]) == 74
    assert "FAILED" in capsys.readouterr().err


def test_launcher_forwards_signals_only_to_running_child_group(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    handlers = {}
    sent = []
    class Child:
        pid = 12345
        running = True
        def poll(self):
            return None if self.running else 0
        def wait(self):
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            self.running = False
            handlers[signal.SIGINT](signal.SIGINT, None)
            return 0
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(worker.signal, "signal", lambda signum, handler: handlers.update({signum: handler}))
    monkeypatch.setattr(worker.os, "killpg", lambda pid, signum: sent.append((pid, signum)))
    monkeypatch.setattr(worker, "upload", lambda *args: None)
    assert worker.main(["test-program"]) == 0
    assert sent == [(12345, signal.SIGTERM)]


def test_archiver_ignores_special_files_and_a_regular_to_fifo_race(monkeypatch, tmp_path):
    normal = tmp_path / "race.txt"
    normal.write_text("content")
    os.mkfifo(tmp_path / "fifo")
    (tmp_path / "link").symlink_to(normal)
    original_open = os.open
    def replace_before_open(path, flags, *args, **kwargs):
        if path == "race.txt" and not flags & os.O_DIRECTORY:
            os.unlink(path, dir_fd=kwargs["dir_fd"])
            os.mkfifo(path, dir_fd=kwargs["dir_fd"])
        return original_open(path, flags, *args, **kwargs)
    monkeypatch.setattr(worker.os, "open", replace_before_open)
    assert worker.archive_outputs(tmp_path, io.BytesIO(), 100, ["**/*"]) == 0


def test_worker_module_entrypoint_executes_and_uploads(monkeypatch, tmp_path, capsys):
    configure(monkeypatch, tmp_path)
    class Connection:
        def __getattr__(self, name):
            return lambda *args: None
        def getresponse(self):
            return SimpleNamespace(status=200, read=lambda limit: b"")
    monkeypatch.setattr(http.client, "HTTPSConnection", lambda *args, **kwargs: Connection())
    monkeypatch.setattr(sys, "argv", ["worker", sys.executable, "-c", "pass"])
    with pytest.warns(RuntimeWarning, match="found in sys.modules"), pytest.raises(SystemExit) as error:
        runpy.run_module(worker.__name__, run_name="__main__")
    assert error.value.code == 0 and "COMPLETE" in capsys.readouterr().out
