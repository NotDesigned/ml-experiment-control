"""Real downloads, cache identity, fail-closed preparation and both backend commands."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
from types import SimpleNamespace

import pytest
import yaml

from ml_exp_server import data_preparation as data
from ml_exp_server import managed_worker as worker
from ml_exp_server.container_controller import Controller
from ml_exp_server.container_execution import ContainerExecutionService
from tests.test_container_api import archive, client, import_source, runtime
from tests.test_sensecore_data_workflow import stored


def definition(workspace, text=None, **changes):
    workspace.mkdir(exist_ok=True)
    script = workspace / "download.py"
    script.write_text(text or "import os\nfrom pathlib import Path\nPath(os.environ['DATA_DIR'], 'tokens.bin').write_bytes(b'tokens')\n")
    spec = {"script": "download.py", "interpreter": "python3", "arguments": [], "timeout_seconds": 5,
            "source_id": "source." + "a" * 64, "image": "registry.example/base@sha256:" + "b" * 64,
            "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(), "workdir": "/workspace", "env": {}, "inputs": [], **changes}
    return {**spec, "preparation_id": "preparation." + data.identity(spec)}


def test_real_http_download_hashes_files_and_reuses_without_network(tmp_path, monkeypatch):
    requests = []
    payload = b"training data\n" * 1000
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(200); self.end_headers(); self.wfile.write(payload)
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        workspace = tmp_path / "source"
        spec = definition(workspace, "import os, sys, urllib.request\nfrom pathlib import Path\nassert 'OUTPUT_DIR' not in os.environ\nassert not any(k.startswith('ML_EXPD_') for k in os.environ)\nPath(os.environ['DATA_DIR'], 'tokens.bin').write_bytes(urllib.request.urlopen(sys.argv[1]).read())\n",
                          arguments=[f"http://127.0.0.1:{server.server_port}/dataset"], env={"DATA_VERSION": "1"})
        monkeypatch.setenv("ML_EXPD_UPLOAD_TOKEN", "do-not-pass")
        monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
        tree, receipt = data.prepare(spec, tmp_path / "cache", workspace=workspace)
        assert receipt["files"] == [{"path": "tokens.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}]
        assert receipt["dataset_id"] == "dataset." + data.identity(receipt["files"])
        assert not receipt["cache_reused"] and tree.parent.parent == tmp_path / "cache"
        assert data.prepare(spec, tmp_path / "cache", workspace=workspace)[1]["cache_reused"]
        assert requests == ["/dataset"] and (tree / "tokens.bin").stat().st_mode & 0o777 == 0o444
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_parallel_jobs_prepare_one_shared_tree_and_reuse_the_verified_receipt(tmp_path):
    workspace = tmp_path / "source"
    counter = tmp_path / "downloads.txt"
    spec = definition(workspace, "import os, sys, time\nfrom pathlib import Path\nwith open(sys.argv[1], 'a') as count: count.write('download\\n')\ntime.sleep(0.2)\nPath(os.environ['DATA_DIR'], 'tokens.bin').write_bytes(b'tokens')",
                      arguments=[str(counter)])
    import sys
    code = "import json,sys; from pathlib import Path; from ml_exp_server.data_preparation import prepare; print(json.dumps(prepare(json.loads(sys.stdin.read()),Path(sys.argv[1]),workspace=Path(sys.argv[2]))[1]))"
    children = [subprocess.Popen([sys.executable, "-c", code, str(tmp_path / "cache"), str(workspace)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    for child in children:
        child.stdin.write(json.dumps(spec)); child.stdin.close(); child.stdin = None
    receipts = []
    for child in children:
        stdout, stderr = child.communicate(timeout=15)
        assert child.returncode == 0, stderr
        receipts.append(json.loads(stdout.splitlines()[-1]))
    assert counter.read_text() == "download\n"
    assert sum(item["cache_reused"] for item in receipts) == 1
    assert receipts[0]["dataset_id"] == receipts[1]["dataset_id"]


@pytest.mark.parametrize("case", ["identity", "script-sha", "script-escape", "exit", "empty", "expected-hash", "symlink", "fifo"])
def test_preparation_failure_never_publishes_or_reuses_partial_data(tmp_path, case):
    workspace = tmp_path / "source"
    text = {"exit": "raise SystemExit(7)", "empty": "pass", "symlink": "import os\nfrom pathlib import Path\nPath(os.environ['DATA_DIR'], 'link').symlink_to('/etc/passwd')",
            "fifo": "import os\nos.mkfifo(os.path.join(os.environ['DATA_DIR'], 'fifo'))"}.get(case)
    spec = definition(workspace, text, **({"expected_content_sha256": "0" * 64} if case == "expected-hash" else {}))
    if case == "identity": spec["preparation_id"] = "preparation." + "0" * 64
    if case in {"script-sha", "script-escape"}:
        (workspace / "download.py").unlink()
        if case == "script-sha": (workspace / "download.py").write_text("changed")
        else: (workspace / "download.py").symlink_to("/etc/passwd")
    with pytest.raises((ValueError, OSError)):
        data.prepare(spec, tmp_path / "cache", workspace=workspace)
    assert not list((tmp_path / "cache").glob(".preparation-*"))
    assert not (tmp_path / "cache" / spec["preparation_id"]).exists()


@pytest.mark.parametrize("case", ["file", "definition", "files", "content", "status", "dataset", "id", "version", "bytes", "root-link", "broken-link", "receipt-link", "receipt-large"])
def test_corrupt_cache_fails_closed_without_rerunning_script(tmp_path, monkeypatch, case):
    workspace = tmp_path / "source"; spec = definition(workspace)
    tree, _ = data.prepare(spec, tmp_path / "cache", workspace=workspace)
    receipt = tree.parent / "receipt.json"
    if case == "file": (tree / "tokens.bin").chmod(0o644); (tree / "tokens.bin").write_bytes(b"bad")
    elif case in {"root-link", "broken-link"}:
        destination = tree.parent; saved = destination.with_name("saved"); destination.rename(saved); destination.symlink_to(saved, target_is_directory=True)
        if case == "broken-link": saved.rename(saved.with_name("hidden"))
    elif case == "receipt-link":
        receipt.unlink(); receipt.symlink_to("/etc/passwd")
    elif case == "receipt-large":
        receipt.chmod(0o644); receipt.write_bytes(b"x" * (8 * 1024 ** 2 + 1))
    else:
        value = json.loads(receipt.read_text())
        key = {"content": "content_sha256", "dataset": "dataset_id", "id": "preparation_id", "version": "schema_version"}.get(case, case)
        value[key] = "broken"
        receipt.chmod(0o644); receipt.write_text(json.dumps(value))
    monkeypatch.setattr(data, "run_script", lambda *args: pytest.fail("corrupt cache must not silently redownload"))
    with pytest.raises(ValueError): data.prepare(spec, tmp_path / "cache", workspace=workspace)


def test_expected_identity_and_nested_files_are_sealed(tmp_path):
    workspace = tmp_path / "source"
    spec = definition(workspace, "import os\nfrom pathlib import Path\np = Path(os.environ['DATA_DIR'], 'nested'); p.mkdir(); (p / 'tokens').write_bytes(b'tokens')")
    records = [{"path": "nested/tokens", "bytes": 6, "sha256": hashlib.sha256(b"tokens").hexdigest()}]
    spec["expected_content_sha256"] = data.identity(records)
    spec["preparation_id"] = "preparation." + data.identity({k: v for k, v in spec.items() if k != "preparation_id"})
    tree, receipt = data.prepare(spec, tmp_path / "cache", workspace=workspace)
    assert receipt["content_sha256"] == spec["expected_content_sha256"]
    assert (tree / "nested").stat().st_mode & 0o777 == 0o555


def test_file_inventory_rejects_invalid_trees_count_and_mid_read_mutation(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="DIRECTORY_INVALID"): data.inventory(tmp_path / "missing")
    path = tmp_path / "tokens"; path.write_bytes(b"tokens")
    original = os.fstat; calls = []
    def changed(fd):
        record = original(fd); calls.append(1)
        return record if len(calls) == 1 else SimpleNamespace(st_size=record.st_size, st_mtime_ns=record.st_mtime_ns + 1, st_ctime_ns=record.st_ctime_ns)
    monkeypatch.setattr(data.os, "fstat", changed)
    with pytest.raises(ValueError, match="FILE_CHANGED"): data.file_record(path, "tokens")
    monkeypatch.setattr(data.os, "fstat", original)
    monkeypatch.setattr(Path, "rglob", lambda *args: [path] * 20001)
    with pytest.raises(ValueError, match="COUNT_LIMIT"): data.inventory(tmp_path)


@pytest.mark.parametrize("kill", [False, True])
def test_script_timeout_kills_process_group_and_restores_handlers(monkeypatch, kill):
    calls = []; handlers = {}; signals = []
    class Child:
        pid = 1234
        def poll(self): return None if not calls else 0
        def wait(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1 or kill and len(calls) == 2: raise subprocess.TimeoutExpired("script", 1)
            return 0
    monkeypatch.setattr(data.subprocess, "Popen", lambda *a, **k: Child())
    def register(sig, fn):
        old = handlers.get(sig); handlers[sig] = fn; return old
    monkeypatch.setattr(data.signal, "signal", register)
    monkeypatch.setattr(data.os, "killpg", lambda pid, sig: signals.append(sig))
    with pytest.raises(ValueError, match="TIMEOUT"): data.run_script(["script"], {}, ".", 1)
    assert signals == ([signal.SIGTERM, signal.SIGKILL] if kill else [signal.SIGTERM])
    assert all(handler is None for handler in handlers.values())


@pytest.mark.parametrize("kill", [False, True])
def test_download_signals_are_forwarded_to_the_active_group(monkeypatch, kill):
    handlers = {}; forwarded = []
    class Child:
        pid = 1234; active = True; calls = 0
        def poll(self): return None if self.active else 0
        def wait(self, **kwargs):
            self.calls += 1
            if self.calls == 1: handlers[signal.SIGTERM](signal.SIGTERM, None)
            if kill and self.calls == 2: raise subprocess.TimeoutExpired("script", 1)
            self.active = False; return 0
    monkeypatch.setattr(data.subprocess, "Popen", lambda *a, **k: Child())
    monkeypatch.setattr(data.signal, "signal", lambda sig, fn: handlers.setdefault(sig, fn))
    monkeypatch.setattr(data.os, "killpg", lambda pid, sig: forwarded.append(sig))
    with pytest.raises(ValueError, match="CANCELLED"): data.run_script(["script"], {}, ".", 1)
    handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert forwarded == ([signal.SIGTERM, signal.SIGKILL] if kill else [signal.SIGTERM])


def test_receipt_size_is_bounded_and_partial_files_cleaned(tmp_path, monkeypatch):
    workspace = tmp_path / "source"; spec = definition(workspace)
    real = data.json.dumps
    monkeypatch.setattr(data.json, "dumps", lambda value, **kwargs: "x" * (8 * 1024 ** 2 + 1) if isinstance(value, dict) and value.get("status") == "READY" else real(value, **kwargs))
    with pytest.raises(ValueError, match="SIZE_LIMIT"): data.prepare(spec, tmp_path / "cache", workspace=workspace)
    assert not list((tmp_path / "cache").glob(".preparation-*"))


@pytest.mark.parametrize("executor", ["gpu", "cloud"])
def test_api_freezes_script_argv_and_backend_data_dir(client, stored, executor):
    source = import_source(client, archive({"train.py": b"print('training')", "download.py": b"print('download')"}))
    ready = runtime(client, source)
    request = {"run_id": "script-data", "runtime_id": ready["runtime_id"], "executor": executor,
               "data_preparation": {"script": "download.py", "arguments": ["--version", "1"]},
               "metrics_schema": {"definitions": {"loss": {"unit": "nats/token"}}}, "outputs": ["model.pt"]}
    response = client.post("/api/projects/demo/runs", json=request)
    assert response.status_code == 200, response.text
    spec = response.json()["data_preparation"]
    assert spec["script_sha256"] == hashlib.sha256(b"print('download')").hexdigest()
    assert spec["arguments"] == ["--version", "1"]
    assert client.post("/api/projects/demo/runs", json=request).json() == response.json()
    project = Path(client.app.state.runtime.project("demo").base_dir)
    campaign = yaml.safe_load((project / "experiments/campaigns/run-script-data.yaml").read_text())
    campaign["local_root"] = str(client.app.state.runtime.config.project_run_root_path("demo"))
    ctl = Controller(campaign, "script-data", "attempt-001"); ctl.prepare()
    manifest = ctl.store.load_manifest()
    assert manifest["resolved_config"]["data_preparation"] == spec
    assert manifest["evaluation"]["protocol"]["data_preparation"] == spec
    assert manifest["assets"][-1]["identity"] == spec["preparation_id"]
    assert any(arg.startswith("DATA_DIR=/data/lab/demo/data-preparations/") for arg in manifest["command"])
    assert any(arg.startswith("ML_EXPD_DATA_PREPARATION=") for arg in ctl.dispatch_command(ctl.store.load_attempt("attempt-001")))
    assert campaign["runs"][0]["outputs"] == ["model.pt", "data-preparation.json"]
    dispatch = ctl.dispatch_command(ctl.store.load_attempt("attempt-001"))
    token = next(arg.split("=", 1)[1] for arg in dispatch if arg.startswith("ML_EXPD_UPLOAD_TOKEN="))
    assert client.put("/api/artifact-transfers/demo/script-data/attempt-001", content=archive({"data-preparation.json": b'{"status":"READY"}'}), headers={"Authorization": "Bearer " + token}).status_code == 200
    changed = {**request, "data_preparation": {"script": "download.py", "arguments": ["--version", "2"]}}
    assert client.post("/api/projects/demo/runs", json=changed).status_code == 409
    changed["run_id"] = "changed"; other = client.post("/api/projects/demo/runs", json=changed).json()
    assert other["data_preparation"]["preparation_id"] != spec["preparation_id"]
    request.update(run_id="explicit-receipt", outputs=["data-preparation.json"])
    assert client.post("/api/projects/demo/runs", json=request).status_code == 200


@pytest.mark.parametrize("changes,status", [({"script": "../download.py"}, 422), ({"script": "missing.py"}, 409),
    ({"script": "download.py", "arguments": [""]}, 422), ({"script": "download.py", "timeout_seconds": 0}, 422),
    ({"script": "download.py", "arguments": ["x" * 8192] * 5}, 409)])
def test_api_rejects_bad_scripts_or_oversized_scheduler_commands(client, stored, changes, status):
    source = import_source(client, archive({"download.py": b"pass"})); ready = runtime(client, source)
    response = client.post("/api/projects/demo/runs", json={"run_id": "bad", "runtime_id": ready["runtime_id"], "executor": "gpu", "data_preparation": changes})
    assert response.status_code == status


def test_old_runtime_and_reserved_data_dir_are_rejected(client, stored):
    ready = runtime(client)
    service = ContainerExecutionService(client.app.state.runtime)
    with service.state("demo", ready["runtime_id"]) as (store, snapshot):
        old = dict(snapshot.value); old["capabilities"] = ["data-assets.v1"]
        store.commit(old, expected_revision=snapshot.revision, event={"event": "historical"})
    request = {"run_id": "old", "runtime_id": ready["runtime_id"], "executor": "cloud", "data_preparation": {"script": "train.py"}}
    assert client.post("/api/projects/demo/runs", json=request).status_code == 409
    request.pop("data_preparation"); request["env"] = {"DATA_DIR": "/tmp/wrong"}
    assert client.post("/api/projects/demo/runs", json=request).status_code == 422


@pytest.mark.parametrize("failed_upload", [False, True])
@pytest.mark.parametrize("known", [False, True])
def test_worker_returns_preparation_failure_receipt_without_starting_training(tmp_path, monkeypatch, failed_upload, known):
    root = tmp_path / "project/runs/trial/attempts/attempt-001/outputs"; root.mkdir(parents=True)
    for key, value in {"OUTPUT_DIR": str(root), "ML_EXPD_UPLOAD_URL": "https://api.example/output", "ML_EXPD_UPLOAD_TOKEN": "private",
                       "ML_EXPD_UPLOAD_LIMIT": str(2 * 1024 ** 2), "ML_EXPD_DATA_PREPARATION": '{"script":"download.py"}'}.items(): monkeypatch.setenv(key, value)
    monkeypatch.setattr(worker, "link_path", lambda *args: None)
    def prepare(*args):
        if known: raise ValueError("DATA_SCRIPT_EXIT_7")
        raise OSError("sensitive signed URL must not enter receipt")
    monkeypatch.setattr(worker, "prepare_data", prepare)
    published = []
    def upload(url, token, stream, size):
        if failed_upload: raise OSError("unavailable")
        published.append(stream.getvalue() if isinstance(stream, io.BytesIO) else size)
    monkeypatch.setattr(worker, "upload", upload)
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **k: pytest.fail("training must not run"))
    assert worker.main(["train"]) == 65
    receipt = json.loads((root / "data-preparation.json").read_text())
    assert receipt["status"] == "FAILED" and receipt["error"] == ("DATA_SCRIPT_EXIT_7" if known else "OSError")
    assert bool(published) != failed_upload


def test_worker_passes_verified_data_dir_and_always_returns_metadata(tmp_path, monkeypatch):
    root = tmp_path / "project/runs/trial/attempts/attempt-001/outputs"; root.mkdir(parents=True)
    workspace = tmp_path / "source"; spec = definition(workspace)
    for key, value in {"OUTPUT_DIR": str(root), "ML_EXPD_UPLOAD_URL": "https://api.example/output", "ML_EXPD_UPLOAD_TOKEN": "private",
                       "ML_EXPD_UPLOAD_LIMIT": str(2 * 1024 ** 2), "ML_EXPD_OUTPUT_PATTERNS": '["metrics.json"]', "ML_EXPD_DATA_PREPARATION": json.dumps(spec)}.items(): monkeypatch.setenv(key, value)
    monkeypatch.setattr(worker, "link_path", lambda *args: None)
    monkeypatch.setattr(worker, "prepare_data", lambda definition, cache: data.prepare(definition, cache, workspace=workspace))
    real = subprocess.Popen
    def popen(*args, **kwargs):
        assert "ML_EXPD_UPLOAD_TOKEN" not in os.environ
        if "env" not in kwargs:
            assert Path(os.environ["DATA_DIR"], "tokens.bin").read_bytes() == b"tokens"
        return real(*args, **kwargs)
    monkeypatch.setattr(worker.subprocess, "Popen", popen)
    target = root.parents[4] / "data-preparations" / spec["preparation_id"] / "tree"
    patterns = []
    original = worker.archive_outputs
    def capture(root, stream, limit, selected):
        patterns.extend(selected)
        return original(root, stream, limit, selected)
    monkeypatch.setattr(worker, "archive_outputs", capture)
    monkeypatch.setattr(worker, "upload", lambda *args: None)
    assert worker.main(["python3", "-c", "pass"]) == 0
    assert "data-preparation.json" in patterns
    assert (root / "data-preparation.json").is_file() and Path(os.environ["DATA_DIR"]) == target
