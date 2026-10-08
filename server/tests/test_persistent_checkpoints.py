"""Backend-local recovery keeps full state off the upload path."""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import pytest
import yaml

from ml_exp_server.workers import persistent_state as state
from ml_exp_server.workers import managed_worker as worker
from ml_exp_server.results.artifact_store import ArtifactStore
from ml_exp_server.results.checkpoint_registry import CheckpointRegistry, storage_scope
from ml_exp_server.container_controller import Controller
from ml_exp_server.schemas import RunIndexRow, AttemptSummary
from tests.test_container_api import client, import_source, runtime
from tests.test_sensecore_data_workflow import stored


def ready(root, *, step=4, generation="step-0004", body=b"model optimizer scheduler RNG data-cursor"):
    path = root / "checkpoints" / generation / "state.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    value = {"step": step, "files": [{"path": path.relative_to(root).as_posix(), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}]}
    (root / "checkpoint.ready.json").write_text(json.dumps(value))
    return value


def context(root):
    return {"project": "demo", "run_id": "trial", "attempt_id": "attempt-001", "source_id": "source." + "a" * 64,
            "image_id": "sha256:" + "b" * 64, "storage_scope": "storage." + "c" * 64, "state_root": str(root)}


def controller(api, bundle, name="trial", executor="cloud", **extra):
    result = api.post("/api/projects/demo/runs", json={"run_id": name, "runtime_id": bundle["runtime_id"], "executor": executor, **extra})
    assert result.status_code == 200, result.text
    root = Path(api.app.state.runtime.project("demo").base_dir)
    campaign = yaml.safe_load((root / f"experiments/campaigns/run-{name}.yaml").read_text())
    campaign["local_root"] = str(api.app.state.runtime.config.project_run_root_path("demo"))
    ctl = Controller(campaign, name, "attempt-001"); ctl.prepare()
    api.app.state.runtime.index.upsert_run(RunIndexRow(project="demo", run_id=name, run_dir=str(ctl.root),
        attempts=[AttemptSummary(attempt_id="attempt-001", state="RUNNING")]))
    ctl.dispatch_command(ctl.store.load_attempt("attempt-001"))
    return ctl


def token(ctl, config):
    objects = ArtifactStore(config, Path(ctl.campaign["source_store"]))
    with objects.record("demo", ctl.run["run_id"], "attempt-001") as (_, transfer): return transfer["token"]


@pytest.mark.parametrize("executor", ["gpu", "cloud"])
def test_registration_frozen_restore_and_storage_scope(client, stored, tmp_path, executor):
    source = import_source(client); bundle = runtime(client, source)
    ctl = controller(client, bundle, executor=executor, checkpoint_persistence={}, metrics_schema={"definitions": {"loss": {"unit": "nats/token"}}})
    manifest = ctl.store.load_manifest()
    assert manifest["resolved_config"]["checkpoint_persistence"]["storage_scope"]
    assert ctl.command("attempt-001") == ["ml-exp-worker"]
    assert ctl.environment("attempt-001")["STATE_DIR"].endswith("/attempts/attempt-001/state")
    value = ready(tmp_path / "state")
    endpoint = "/api/checkpoint-transfers/demo/trial/attempt-001"
    assert client.put(endpoint, json=value, headers={"Authorization": "Bearer wrong"}).status_code == 401
    headers = {"Authorization": "Bearer " + token(ctl, stored[0])}
    result = client.put(endpoint, json=value, headers=headers)
    assert result.status_code == 200, result.text
    checkpoint = result.json()
    assert checkpoint["state_root"].endswith("/runs/trial/attempts/attempt-001/state")
    assert checkpoint["source_id"] == source["source_id"] and checkpoint["verification"] == "worker-sha256"
    assert client.put(endpoint, json=value, headers=headers).json() == checkpoint
    assert client.get(endpoint.replace("checkpoint-transfers", "runs")).status_code == 404
    path = "/api/runs/demo/trial/attempts/attempt-001/checkpoints"
    assert client.get(path).json()["checkpoints"] == [checkpoint]
    assert client.get(path + "/" + checkpoint["checkpoint_id"]).json() == checkpoint
    assert client.get(path + "/checkpoint." + "e" * 64).status_code == 404
    assert client.get(path + "/invalid").status_code == 409
    ref = {k: checkpoint[k] for k in ("run_id", "attempt_id", "checkpoint_id")}
    restored = controller(client, bundle, name="resumed", executor=executor, resume_from=ref, metrics_schema={"definitions": {}})
    assert restored.run["resume_from"] == checkpoint
    assert restored.store.load_attempt("attempt-001")["resume_from"] == checkpoint["checkpoint_id"]
    assert restored.store.load_manifest()["evaluation"]["protocol"]["resume_from"] == checkpoint
    assert restored.store.load_manifest()["assets"][-1]["kind"] == "persistent_checkpoint"
    assert restored.environment("attempt-001")["RESUME_DIR"] == "/inputs/resume"
    request = {"run_id": "wrong-store", "runtime_id": bundle["runtime_id"], "executor": "gpu" if executor == "cloud" else "cloud", "resume_from": ref}
    response = client.post("/api/projects/demo/runs", json=request)
    assert response.status_code == 409 and "different backend storage" in response.text
    request.update(run_id="collision", executor=executor, inputs=[{"asset_id": "asset." + "f" * 64, "mount_path": "/inputs/resume"}])
    # Asset identity must exist before the reserved mount can be checked.
    from tests.test_sensecore_data_workflow import put_asset
    request["inputs"][0]["asset_id"] = put_asset(client).json()["asset_id"]
    assert client.post("/api/projects/demo/runs", json=request).status_code == 409
    changed = deepcopy(value); changed["step"] = 5
    assert client.put(endpoint, json=changed, headers=headers).status_code == 409
    next_value = ready(tmp_path / "state", step=8, generation="step-0008")
    assert client.put(endpoint, json=next_value, headers=headers).status_code == 200
    assert len(client.get(path).json()["checkpoints"]) == 2
    assert len(stored[1]) == 1  # Only the explicit mount-collision fixture asset was uploaded.


def test_registration_limits_capabilities_and_missing_store(client, stored, tmp_path):
    source = import_source(client); bundle = runtime(client, source)
    ctl = controller(client, bundle)
    value = ready(tmp_path)
    path = "/api/checkpoint-transfers/demo/trial/attempt-001"
    headers = {"Authorization": "Bearer " + token(ctl, stored[0])}
    assert client.put(path, json=value, headers=headers).status_code == 401
    registry = CheckpointRegistry(stored[0], client.app.state.runtime.config.project_registry_root_path())
    with pytest.raises(ValueError, match="not enabled"):
        registry.receive("demo", "trial", "attempt-001", token(ctl, stored[0]), value)
    ctl = controller(client, bundle, name="persistent", checkpoint_persistence={})
    path = path.replace("trial", "persistent"); headers["Authorization"] = "Bearer " + token(ctl, stored[0])
    assert client.put(path, content=b"x" * (state.MANIFEST_LIMIT + 1), headers=headers).status_code == 413
    assert client.put(path, content=b"invalid", headers=headers).status_code == 422
    assert client.put(path, json={}, headers=headers).status_code == 409
    assert client.put(path, json=value, headers={}).status_code == 401
    assert client.put(path, json=value, headers={"Authorization": "Basic invalid"}).status_code == 401
    assert client.put(path.replace("persistent", "other"), json=value, headers=headers).status_code == 401
    objects = ArtifactStore(stored[0], client.app.state.runtime.config.project_registry_root_path())
    with objects.record("demo", "persistent", "attempt-001") as (p, transfer):
        transfer["expires_at"] = "2000-01-01T00:00:00+00:00"; p.write_text(json.dumps(transfer))
    assert client.put(path, json=value, headers=headers).status_code == 401
    service = CheckpointRegistry(stored[0], client.app.state.runtime.config.project_registry_root_path())
    with pytest.raises(ValueError): service.directory("../escape", "trial", "attempt-001")
    with pytest.raises(ValueError): service.receive("demo", "trial", "attempt-001", token(ctl, stored[0]), value)
    bundle_path = client.app.state.runtime.config.project_registry_root_path() / "runtime-bundles/demo" / (bundle["runtime_id"] + ".json")
    frozen = json.loads(bundle_path.read_text()); frozen["capabilities"] = []; bundle_path.write_text(json.dumps(frozen))
    assert client.post("/api/projects/demo/runs", json={"run_id":"old-worker", "executor":"cloud", "runtime_id":bundle["runtime_id"], "checkpoint_persistence":{}}).status_code == 409
    client.app.state.runtime.config.container_execution.artifact_store_file = None
    assert client.get("/api/runs/demo/trial/attempts/attempt-001/checkpoints").status_code == 404


@pytest.mark.parametrize("case", ["shape", "step", "empty", "fields", "path-type", "path", "duplicate", "bytes", "sha", "mixed", "many", "large"])
def test_manifest_rejects_invalid_complete_state(tmp_path, case):
    value = ready(tmp_path)
    if case == "shape": value = []
    if case == "step": value["step"] = True
    if case == "empty": value["files"] = []
    if case == "fields": value["files"][0]["other"] = 1
    if case == "path-type": value["files"][0]["path"] = []
    if case == "path": value["files"][0]["path"] = "../escape"
    if case == "duplicate": value["files"] *= 2
    if case == "bytes": value["files"][0]["bytes"] = -1
    if case == "sha": value["files"][0]["sha256"] = "wrong"
    if case == "mixed": value["files"].append({**value["files"][0], "path": "checkpoints/other/state.bin"})
    if case == "many": value["files"] *= 20001
    if case == "large": value["files"][0]["path"] = "checkpoints/step/" + "x" * state.MANIFEST_LIMIT
    with pytest.raises(ValueError): state.manifest(value)


@pytest.mark.parametrize("case", ["missing", "extra", "fifo", "link", "parent-link", "wrong-sha", "wrong-size", "changing", "large-marker", "wrong-id"])
def test_restore_checks_every_file_and_refuses_corruption(tmp_path, monkeypatch, case):
    value = ready(tmp_path); path = tmp_path / value["files"][0]["path"]
    if case == "missing": path.unlink()
    if case == "extra": path.with_name("extra.bin").write_bytes(b"extra")
    if case in {"fifo", "link"}:
        path.unlink()
        if case == "fifo": os.mkfifo(path)
        else: path.symlink_to(tmp_path / "checkpoint.ready.json")
    if case == "parent-link":
        moved = path.parent.with_name("moved"); path.parent.rename(moved); path.parent.symlink_to(moved)
    if case == "wrong-sha": value["files"][0]["sha256"] = "a" * 64
    if case == "wrong-size": value["files"][0]["bytes"] = 1
    if case == "changing":
        original = state.os.fstat; calls = []
        def changed(fd):
            actual = original(fd); calls.append(1)
            return SimpleNamespace(st_size=actual.st_size, st_mtime_ns=actual.st_mtime_ns + 1, st_ctime_ns=actual.st_ctime_ns) if len(calls) == 3 else actual
        monkeypatch.setattr(state.os, "fstat", changed)
    if case == "large-marker":
        (tmp_path / "checkpoint.ready.json").write_bytes(b"x" * (state.MANIFEST_LIMIT + 1))
        with pytest.raises(ValueError): state.read_ready(tmp_path)
        return
    value = state.document(context(tmp_path), value)
    if case == "wrong-id": value["checkpoint_id"] = "checkpoint." + "f" * 64
    with pytest.raises((ValueError, OSError)): state.restore(value)


def test_open_file_rejects_special_file_and_metadata_accepts_large_state(tmp_path):
    os.mkfifo(tmp_path / "fifo")
    with pytest.raises(ValueError, match="regular"): state.open_file(tmp_path, "fifo")
    value = ready(tmp_path); value["files"][0]["bytes"] = 16 * 1024 ** 3
    assert state.manifest(value)["files"][0]["bytes"] > 4 * 1024 ** 3


@pytest.mark.parametrize("case", ["bad-url", "status", "oversize", "identity", "success"])
def test_registration_transport_is_fixed_https_bounded_and_exact(tmp_path, monkeypatch, case):
    value = ready(tmp_path); expected = state.document(context(tmp_path), value); closed = []
    raw = json.dumps({**expected, "status":"REGISTERED"}).encode()
    if case == "oversize": raw = b"x" * (state.MANIFEST_LIMIT * 2 + 1)
    if case == "identity": raw = b"{}"
    class Connection:
        def __init__(self, *args, **kwargs): pass
        def request(self, method, path, body, headers):
            assert method == "PUT" and json.loads(body) == value and headers["Authorization"] == "Bearer exact"
        def getresponse(self): return SimpleNamespace(status=401 if case == "status" else 200, read=lambda n: raw[:n])
        def close(self): closed.append(True)
    monkeypatch.setattr(state.http.client, "HTTPSConnection", Connection)
    if case == "success": assert state.register("https://api.example/register", "exact", value, expected)["checkpoint_id"] == expected["checkpoint_id"]
    else:
        with pytest.raises(ValueError): state.register("http://api.example/register" if case == "bad-url" else "https://api.example/register", "exact", value, expected)
    assert closed == ([] if case == "bad-url" else [True])


def test_registered_manifest_corruption_and_transfer_rebinding(client, stored, tmp_path):
    bundle = runtime(client, import_source(client)); ctl = controller(client, bundle, checkpoint_persistence={})
    objects = ArtifactStore(stored[0], client.app.state.runtime.config.project_registry_root_path())
    with pytest.raises(ValueError, match="already bound"):
        objects.issue("demo", "trial", "attempt-001", ctl.root, ctl.run["outputs"], checkpoint_state={"changed":True})
    registry = CheckpointRegistry(stored[0], client.app.state.runtime.config.project_registry_root_path())
    value = registry.receive("demo", "trial", "attempt-001", token(ctl, stored[0]), ready(tmp_path))
    path = registry.directory("demo", "trial", "attempt-001") / (value["checkpoint_id"] + ".json")
    value["files"][0]["bytes"] += 1; path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="identity differs"): registry.read("demo", "trial", "attempt-001", value["checkpoint_id"])


def test_real_cpu_jobs_restore_without_state_archive_or_reupload(tmp_path, monkeypatch, client, stored):
    profiles_path = Path(client.app.state.runtime.config.container_execution.profiles_file)
    profiles = yaml.safe_load(profiles_path.read_text())
    for profile in profiles["executors"].values(): profile["storage_root"] = str(tmp_path / "backend")
    profiles_path.write_text(yaml.safe_dump(profiles))
    bundle = runtime(client, import_source(client))
    ctl = controller(client, bundle, checkpoint_persistence={"interval_seconds":5})
    script = tmp_path / "train.py"
    script.write_text('''import os,json,hashlib
from pathlib import Path
assert not any(k.startswith("ML_EXPD_") for k in os.environ)
resume=os.environ.get("RESUME_DIR")
step=int((Path(resume)/"state.bin").read_text())+4 if resume else 4
root=Path(os.environ["STATE_DIR"]); path=root/"checkpoints"/f"step-{step:04d}"/"state.bin"
path.parent.mkdir(parents=True); body=str(step).encode()+b" "*(3*1024**2); path.write_bytes(body)
ready={"step":step,"files":[{"path":path.relative_to(root).as_posix(),"bytes":len(body),"sha256":hashlib.sha256(body).hexdigest()}]}
(root/"checkpoint.ready.tmp").write_text(json.dumps(ready)); (root/"checkpoint.ready.tmp").replace(root/"checkpoint.ready.json")
Path(os.environ["OUTPUT_DIR"],"weights.bin").write_bytes(str(step).encode())
''')
    connections, uploads = [], []
    class Connection:
        def __init__(self, *args, **kwargs): pass
        def request(self, method, path, body, headers):
            connections.append(path); self.response = client.request(method, path, content=body, headers=headers)
        def getresponse(self): return SimpleNamespace(status=self.response.status_code, read=lambda n:self.response.content[:n])
        def close(self): pass
    monkeypatch.setattr(state.http.client, "HTTPSConnection", Connection)
    bound = {}
    monkeypatch.setattr(worker, "link_path", lambda target, path: bound.update({str(path): target}))
    real_popen = subprocess.Popen
    def child(*args, **kwargs):
        environment = dict(os.environ)
        if "RESUME_DIR" in environment: environment["RESUME_DIR"] = str(bound["/inputs/resume"])
        return real_popen(*args, **kwargs, env=environment)
    monkeypatch.setattr(worker.subprocess, "Popen", child)
    monkeypatch.setattr(worker.time, "sleep", lambda n:None)
    def upload(url, cap, stream, length):
        stream.seek(0)
        with tarfile.open(fileobj=stream) as archive: uploads.append(archive.getnames())
    monkeypatch.setattr(worker, "upload", upload)
    def run(ctl):
        output = Path(ctl.run["storage"]["run_dir"]) / "attempts/attempt-001/outputs"
        ctx = ctl.state_context("attempt-001")
        for key,value in {"OUTPUT_DIR":str(output), "ML_EXPD_UPLOAD_URL":"https://api.example/final", "ML_EXPD_UPLOAD_TOKEN":token(ctl,stored[0]),
             "ML_EXPD_UPLOAD_LIMIT":str(2*1024**2), "ML_EXPD_CHECKPOINT_STATE":json.dumps(ctx), "ML_EXPD_CHECKPOINT_STATE_URL":f"https://api.example/api/checkpoint-transfers/demo/{ctl.run['run_id']}/attempt-001",
             "ML_EXPD_CHECKPOINT_STATE_INTERVAL":"5", "ML_EXPD_OUTPUT_PATTERNS":'["weights.bin","checkpoints.json"]',
             "ML_EXPD_CHECKPOINT_RESTORE":json.dumps(ctl.run.get("resume_from"))}.items(): monkeypatch.setenv(key,value)
        assert worker.main([os.sys.executable,str(script)]) == 0
        return client.get(f"/api/runs/demo/{ctl.run['run_id']}/attempts/attempt-001/checkpoints").json()["checkpoints"][0]
    first = run(ctl); assert first["step"] == 4
    ref = {k:first[k] for k in ("run_id","attempt_id","checkpoint_id")}
    resumed = controller(client,bundle,name="resumed",resume_from=ref)
    second = run(resumed); assert second["step"] == 8
    assert first["state_root"] != second["state_root"]
    assert connections == ["/api/checkpoint-transfers/demo/trial/attempt-001","/api/checkpoint-transfers/demo/resumed/attempt-001"]
    assert len(uploads)==2 and all(set(names)=={"weights.bin","checkpoints.json"} for names in uploads)
    assert not stored[1]


def test_nested_generation_sealing_and_oversized_frozen_restore(client, stored, tmp_path):
    value = ready(tmp_path)
    extra = tmp_path / "checkpoints/step-0004/nested/config.json"; extra.parent.mkdir(); extra.write_bytes(b"config")
    value["files"].append({"path":extra.relative_to(tmp_path).as_posix(),"bytes":6,"sha256":hashlib.sha256(b"config").hexdigest()})
    state.verify(tmp_path,value,seal=True)
    assert extra.parent.stat().st_mode & 0o777 == 0o555
    bundle = runtime(client, import_source(client)); ctl = controller(client,bundle,checkpoint_persistence={})
    registry = CheckpointRegistry(stored[0], client.app.state.runtime.config.project_registry_root_path())
    value = {"step":1,"files":[{"path":"checkpoints/large/"+str(i)+"x"*100,"bytes":1,"sha256":"a"*64} for i in range(600)]}
    registered = registry.receive("demo","trial","attempt-001",token(ctl,stored[0]),value)
    result = client.post("/api/projects/demo/runs",json={"run_id":"large-resume","runtime_id":bundle["runtime_id"],"executor":"cloud",
        "resume_from":{k:registered[k] for k in ("run_id","attempt_id","checkpoint_id")}})
    assert result.status_code == 409 and "scheduler command limit" in result.text


@pytest.mark.parametrize("case", ["repeat", "retry", "state-mismatch", "corrupt-restore", "backup"])
def test_worker_state_retry_failed_restore_and_optional_backup(tmp_path, monkeypatch, case, capsys):
    root=tmp_path/"project/runs/trial/attempts/attempt-001/outputs";root.mkdir(parents=True)
    state_root=root.parent/"state";ready(state_root)
    ctx=context(state_root)
    restore=None
    if case=="state-mismatch":ctx["state_root"]=str(tmp_path/"another-state")
    if case=="corrupt-restore":
        restore=state.document(ctx,state.read_ready(state_root));restore["files"][0]["sha256"]="f"*64
    for key,value in {"OUTPUT_DIR":str(root),"ML_EXPD_UPLOAD_URL":"https://api.example/final","ML_EXPD_UPLOAD_TOKEN":"private",
        "ML_EXPD_UPLOAD_LIMIT":str(2*1024**2),"ML_EXPD_CHECKPOINT_STATE":json.dumps(ctx),"ML_EXPD_CHECKPOINT_STATE_URL":"https://api.example/register",
        "ML_EXPD_CHECKPOINT_STATE_INTERVAL":"5","ML_EXPD_CHECKPOINT_RESTORE":json.dumps(restore),
        "ML_EXPD_SNAPSHOT_URL":"https://api.example/backup" if case=="backup" else ""}.items():monkeypatch.setenv(key,value)
    monkeypatch.setattr(worker,"link_path",lambda *args:None)
    class Child:
        pid=12345;calls=0
        def poll(self):self.calls+=1;return None if self.calls<3 else 0
    started=[]
    monkeypatch.setattr(worker.subprocess,"Popen",lambda *a,**k:started.append(1) or Child())
    ticks=iter(range(0,10000,10));monkeypatch.setattr(worker.time,"monotonic",lambda:next(ticks))
    monkeypatch.setattr(worker.time,"sleep",lambda *a:None)
    registrations=[]
    def register(url,cap,manifest,expected):
        registrations.append(1)
        if case=="retry" and len(registrations)==1:raise OSError("unavailable")
        return {**expected,"status":"REGISTERED"}
    monkeypatch.setattr(state,"register",register)
    uploaded=[];monkeypatch.setattr(worker,"upload",lambda url,*args:uploaded.append(url))
    code=worker.main(["fake-training"])
    if case in {"state-mismatch","corrupt-restore"}:
        assert code==65 and not started and not uploaded
    else:
        assert code==0 and len(registrations)==(2 if case=="retry" else 1)
        assert len(json.loads((root/"checkpoints.json").read_text())["checkpoints"])==1
        assert uploaded==(["https://api.example/backup","https://api.example/final"] if case=="backup" else ["https://api.example/final"])
        if case=="retry":assert "REGISTRATION_FAILED" in capsys.readouterr().err
