"""Exercise the installed daemon entry point as a real subprocess."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_real_daemon_process_serves_versioned_health(tmp_path):
    port = _free_port()
    config = tmp_path / "server.yaml"
    config.write_text(
        "schema_version: 1\n"
        f"index_db: {tmp_path / 'index.sqlite'}\n"
        f"action_root: {tmp_path / 'actions'}\n"
        "collector_enabled: false\n"
        "projects: []\n",
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [
            sys.executable, "-m", "ml_exp_server.cli",
            "--config", str(config), "--host", "127.0.0.1",
            "--port", str(port), "--snapshot", "--log-level", "warning",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 10
    payload = None
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise AssertionError(
                    f"ml-expd exited early ({process.returncode}): {stdout}\n{stderr}"
                )
            try:
                request = Request(
                    f"http://127.0.0.1:{port}/api/health",
                    headers={"X-ML-Expd-Client-Protocol": "1"},
                )
                with urlopen(request, timeout=0.5) as response:
                    payload = json.loads(response.read())
                break
            except (OSError, URLError):
                time.sleep(0.05)
        assert payload is not None, "ml-expd did not become healthy"
        assert payload["status"] == "ok"
        assert payload["api_protocol_version"] == 1
        assert "terminal-snapshot.v1" in payload["capabilities"]
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
