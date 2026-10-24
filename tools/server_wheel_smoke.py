"""Run the installed basic API and canonical metrics without the optional W&B SDK."""
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


def main():
    assert importlib.util.find_spec("wandb") is None
    import ml_exp_server
    from ml_exp_server.api.contract import API_PROTOCOL_VERSION, CLIENT_PROTOCOL_HEADER
    from ml_exp_server.records.ingest import normalized_metric
    from ml_exp_server.records.store import RecordStore
    assert ml_exp_server.__version__ == importlib.metadata.version("ml-experiment-server")
    with tempfile.TemporaryDirectory(prefix="ml-exp-server-wheel-") as directory:
        root = Path(directory)
        store = RecordStore(root / "projects")
        store.bind("demo", "train", {"enabled": False}, {})
        scope = store.scope("demo", "train", "attempt-001")
        record = {"name": "loss", "value": 1.25, "unit": "nats/token", "step": 2}
        store.append(scope, [("metric-1", normalized_metric(record, {}))])
        assert store.metric_records("demo", "train", "attempt-001")[0]["value"] == 1.25
        token = "wheel-smoke-local-token-not-a-real-secret"
        token_path = root / "token"
        token_path.write_text(token)
        token_path.chmod(0o600)
        config = {"schema_version": 1, "index_db": str(root / "index.sqlite"),
                  "run_root": str(root / "runs"), "project_registry_root": str(root / "projects"),
                  "action_root": str(root / "actions"), "projects": [], "collector_enabled": False,
                  "telemetry": {"enabled": False}, "http_auth": {"bearer_token_file": str(token_path)}}
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config))
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        env = {key: value for key, value in os.environ.items()
               if key != "PYTHONPATH" and not key.startswith(("ML_EXPD_", "WANDB_"))}
        command = [str(Path(sys.executable).parent / "ml-expd"), "--config", str(config_path),
                   "--host", "127.0.0.1", "--port", str(port), "--snapshot"]
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def request(path, *, authenticated=True):
            headers = {"Authorization": "Bearer " + token} if authenticated else {}
            headers[CLIENT_PROTOCOL_HEADER] = str(API_PROTOCOL_VERSION)
            with opener.open(urllib.request.Request(f"http://127.0.0.1:{port}" + path, headers=headers), timeout=2) as response:
                return json.load(response)
        with (root / "server.log").open("w+") as log:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 30
                while True:
                    try:
                        health = request("/api/health")
                        break
                    except (OSError, urllib.error.URLError):
                        if process.poll() is not None or time.monotonic() >= deadline:
                            log.seek(0)
                            raise AssertionError("basic server did not start: " + log.read())
                        time.sleep(0.1)
                assert health["status"] == "ok"
                settings = request("/api/tracking/wandb")
                assert settings["publisher"] == {"available": False, "error": "WANDB_SDK_NOT_INSTALLED"}
                try:
                    request("/api/health", authenticated=False)
                except urllib.error.HTTPError as error:
                    assert error.code == 401
                else:
                    raise AssertionError("basic API accepted an unauthenticated request")
            finally:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    print("Basic server wheel: authenticated API and metrics work without W&B SDK")


if __name__ == "__main__":
    main()
