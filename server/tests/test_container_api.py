"""Real HTTP-to-store contracts; no live scheduler or registry mutations."""

import hashlib
import io
import json
from pathlib import Path
import tarfile

from fastapi.testclient import TestClient
import pytest
import yaml

from ml_exp_server.api.app import create_app
from ml_exp_server.container_controller import Controller
from ml_exp_server.image_builder import bundle_id
from ml_exp_server.schemas import ServerConfig, RunIndexRow, AttemptSummary


def archive(files=None):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as output:
        for name, body in (files or {"train.py": b"print('training')\n"}).items():
            member = tarfile.TarInfo(name)
            member.size = len(body)
            output.addfile(member, io.BytesIO(body))
    return stream.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    profiles = tmp_path / "profiles.yaml"
    profiles.write_text(yaml.safe_dump({"executors": {
        "gpu": {"storage_root": "/data/lab", "backend": {"kind": "slurm", "ssh_alias": "cluster",
                 "partition": "gpu", "account": "lab", "qos": "normal", "gres": "gpu:l40s:1", "mount_root": "/data"}},
        "cloud": {"storage_root": "/data/lab", "backend": {"kind": "sensecore", "workspace": "lab",
                   "aec2": "compute", "worker_spec": "one-gpu", "quota_type": "spot", "storage_mount": "volume:/data"}},
    }}))
    config = ServerConfig(index_db=str(tmp_path / "index.sqlite"),
                          action_root=str(tmp_path / "actions"), run_root=str(tmp_path / "runs"),
                          project_registry_root=str(tmp_path / "projects"), telemetry={"enabled": False},
                          collector_enabled=False, action_runtime={"allow_source_imports": True,
                          "allow_project_writes": True, "allow_scheduler_mutations": True},
                          container_execution={"profiles_file": str(profiles), "builder_socket": "test.sock"})
    def build(_socket, request):
        return {"project": request["project"], "source_id": request["source_id"],
                "base_image": request["base_image"], "image": "registry.example/lab/project@sha256:" + "b" * 64,
                "bundle_id": bundle_id(request["project"], request["source_id"], request["base_image"])}
    monkeypatch.setattr("ml_exp_server.container_execution.builder_request", build)
    with TestClient(create_app(config, poll=False)) as value:
        yield value


def import_source(client, data=None, project="demo"):
    data = archive() if data is None else data
    response = client.post("/api/source-imports/archive", params={"project": project, "sha256": hashlib.sha256(data).hexdigest()}, content=data)
    assert response.status_code == 200, response.text
    return response.json()


def runtime(client, source=None):
    source = import_source(client) if source is None else source
    prepared = client.post("/api/projects/demo/runtimes/prepare", json={
        "source_id": source["source_id"], "image": "registry.example/python@sha256:" + "a" * 64,
        "entrypoint": ["python", "train.py"]})
    assert prepared.status_code == 200, prepared.text
    value = prepared.json()
    endpoint = "/api/projects/demo/runtimes/" + value["runtime_id"]
    assert client.post(endpoint + "/execute", json={"confirmation": value["confirmation"]}).status_code == 202
    completed = client.get(endpoint).json()
    assert completed["status"] == "READY"
    return completed


def test_import_runtime_and_backends_share_frozen_execution(client):
    source = import_source(client)
    assert source["source_id"] == import_source(client)["source_id"]
    bundle = runtime(client, source)
    for executor in ("gpu", "cloud"):
        response = client.post("/api/projects/demo/runs", json={"run_id": executor, "runtime_id": bundle["runtime_id"], "executor": executor})
        assert response.status_code == 200, response.text
        frozen = response.json()
        assert frozen["source_id"] == source["source_id"]
        assert frozen["entrypoint"] == ["python", "train.py"]
        assert frozen["image"] == bundle["image"]
    assert client.post("/api/projects/demo/runs", json={"run_id": "gpu", "runtime_id": bundle["runtime_id"], "executor": "cloud"}).status_code == 409
    root = Path(client.app.state.runtime.project("demo").base_dir)
    campaign = yaml.safe_load((root / "experiments/campaigns/run-gpu.yaml").read_text())
    campaign["local_root"] = str(client.app.state.runtime.config.project_run_root_path("demo"))
    controller = Controller(campaign, "gpu", "attempt-001")
    prepared = controller.prepare("campaign." + "c" * 64)
    assert controller.backend.s.run_manifest_path(campaign, controller.run) == controller.store.manifest_path
    assert controller.backend.s.run_manifest_path(campaign, controller.run).is_file()
    manifest = controller.store.load_manifest()
    assert manifest["identity_version"] == 2 and manifest["source_id"] == source["source_id"]
    assert manifest["image_id"] == "sha256:" + "b" * 64
    assert "{attempt_id}" in " ".join(manifest["command"])
    import subprocess, sys
    preview = dict(campaign)
    preview.pop("local_root")
    preview_path = root / "preview.yml"
    preview_path.write_text(yaml.safe_dump(preview))
    response = subprocess.run([sys.executable, str(root / "tools/experimentctl.py"), str(preview_path),
                               "submit", "--run", "gpu", "--attempt-id", "attempt-001", "--dry-run",
                               "--local-root", str(root / "preview-runs"), "--campaign-id", "campaign." + "c" * 64],
                              capture_output=True, text=True)
    assert response.returncode == 0, response.stderr
    assert json.loads(response.stdout)[0]["source_id"] == source["source_id"]
    controller2 = Controller(campaign, "gpu", "attempt-002")
    controller2.prepare("campaign." + "d" * 64)
    assert controller2.store.load_manifest() == manifest
    assert controller2.store.load_attempt("attempt-002")["attempt_id"] == "attempt-002"
    campaign["runs"][0]["container"]["entrypoint"] = ["sh", "different.sh"]
    with pytest.raises(ValueError, match="immutable definition"):
        Controller(campaign, "gpu", "attempt-003")


@pytest.mark.parametrize('job', [None, '1234'])
def test_status_recovers_pending_outbox_without_submitting(client, job):
    from types import SimpleNamespace
    bundle = runtime(client)
    client.post('/api/projects/demo/runs', json={'run_id':'gpu','runtime_id':bundle['runtime_id'],'executor':'gpu'})
    root = Path(client.app.state.runtime.project('demo').base_dir)
    campaign = yaml.safe_load((root/'experiments/campaigns/run-gpu.yaml').read_text())
    campaign['local_root'] = str(client.app.state.runtime.config.project_run_root_path('demo'))
    controller = Controller(campaign, 'gpu', 'attempt-001')
    controller.prepare()
    intent = controller.store.begin_submission(project='demo',run_id='gpu',attempt_id='attempt-001',backend='slurm',request={'scheduler_name':'gpu--attempt-001'})
    recovered = []
    def recover(run, observed, attempt):
        assert observed == intent and attempt == 'attempt-001'
        recovered.append(True)
        return job
    controller.backend = SimpleNamespace(kind='slurm',recover_submission=recover,status=lambda *args: {'state':'SUCCEEDED','backend_job_id':job})
    status = controller.status()
    assert recovered == [True]
    assert status['backend_job_id'] == job
    assert status['state'] == ('SUCCEEDED' if job else 'SUBMITTING')
    if job is None:assert status['submission_recovery'] == 'NOT_FOUND'


def test_slurm_submission_uploads_canonical_run_manifest_from_attempt_scope(client):
    from experiment_control.runner import CommandResult
    bundle = runtime(client)
    client.post('/api/projects/demo/runs',json={'run_id':'gpu','runtime_id':bundle['runtime_id'],'executor':'gpu'})
    root = Path(client.app.state.runtime.project('demo').base_dir)
    campaign = yaml.safe_load((root/'experiments/campaigns/run-gpu.yaml').read_text())
    campaign['local_root'] = str(client.app.state.runtime.config.project_run_root_path('demo'))
    calls = []
    class Runner:
        def run(self, command, **kwargs):
            calls.append(command)
            if command[0] == 'rsync':assert Path(command[-2]).is_file()
            return CommandResult(tuple(command),0,'1234\n' if 'sbatch --parsable' in command[-1] else '')
    controller = Controller(campaign,'gpu','attempt-001',runner=Runner())
    controller.prepare()
    controller.backend.validate_live = lambda run: {}
    intent = controller.store.begin_submission(project='demo',run_id='gpu',attempt_id='attempt-001',backend='slurm',request=controller.backend.submission_request(campaign,controller.run,'attempt-001'))
    result = controller.backend.submit(campaign,controller.run,controller.store.load_attempt('attempt-001'),dry_run=False,intent=intent)
    assert result == '1234'
    assert [command[-2] for command in calls if command[0]=='rsync'][0] == str(controller.store.manifest_path)


@pytest.mark.parametrize('failure', [None,'manifest-conflict','missing-prior-job','new-job-exists'])
def test_retry_identity_allows_only_matching_manifest_and_terminal_prior_attempt(client, failure):
    from types import SimpleNamespace
    from experiment_control.identity import IdentityReport
    bundle = runtime(client)
    client.post('/api/projects/demo/runs',json={'run_id':'gpu','runtime_id':bundle['runtime_id'],'executor':'gpu'})
    root = Path(client.app.state.runtime.project('demo').base_dir)
    campaign = yaml.safe_load((root/'experiments/campaigns/run-gpu.yaml').read_text())
    campaign['local_root'] = str(client.app.state.runtime.config.project_run_root_path('demo'))
    prior = Controller(campaign,'gpu','attempt-001')
    prior.prepare()
    prior.store.begin_submission(project='demo',run_id='gpu',attempt_id='attempt-001',backend='slurm',request={'scheduler_name':'gpu--attempt-001'})
    if failure != 'missing-prior-job':prior.store.reconcile_submission(project='demo',run_id='gpu',attempt_id='attempt-001',backend_job_id='1234',state='FAILED')
    controller = Controller(campaign,'gpu','attempt-002')
    report = IdentityReport(available=False,ambiguous=False,remote_manifest_exists=True,remote_manifest_matches=failure!='manifest-conflict',scheduler_job_ids=('5555',) if failure=='new-job-exists' else ())
    controller.backend = SimpleNamespace(identity=lambda *args:report)
    if failure:
        with pytest.raises(ValueError):controller.check_identity()
    else:controller.check_identity()


@pytest.mark.parametrize("name", ["../escape", "/absolute", ".env", ".ssh/id_rsa", "nested/../escape"])
def test_archive_paths_fail_before_registration(client, name):
    data = archive({name: b"value"})
    response = client.post("/api/source-imports/archive", params={"project": "blocked", "sha256": hashlib.sha256(data).hexdigest()}, content=data)
    assert response.status_code == 409
    assert all(p["project"] != "blocked" for p in client.get("/api/project-lifecycle").json()["projects"])


def test_archive_digest_limits_and_links(client):
    data = archive()
    assert client.post("/api/source-imports/archive", params={"project": "bad", "sha256": "0" * 64}, content=data).status_code == 409
    client.app.state.runtime.config.container_execution.max_archive_bytes = 10
    assert client.post("/api/source-imports/archive", params={"project": "bad", "sha256": hashlib.sha256(data).hexdigest()}, content=data).status_code == 413
    client.app.state.runtime.config.container_execution.max_archive_bytes = 1024 * 1024
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as output:
        member = tarfile.TarInfo("link")
        member.type, member.linkname = tarfile.SYMTYPE, "/etc/passwd"
        output.addfile(member)
    data = stream.getvalue()
    assert client.post("/api/source-imports/archive", params={"project": "bad", "sha256": hashlib.sha256(data).hexdigest()}, content=data).status_code == 409


def test_source_updates_preserve_old_versions_and_runtime_pins(client):
    first = import_source(client)
    second = import_source(client, archive({"train.py": b"print('new')\n"}))
    assert first["source_id"] != second["source_id"]
    assert client.get("/api/projects/demo/sources/" + first["source_id"]).status_code == 200
    assert client.post("/api/projects/demo/runtimes/prepare", json={"source_id": second["source_id"], "image": "python:latest", "entrypoint": ["python", "train.py"]}).status_code == 422
    assert client.post("/api/source-imports/git", json={"project": "remote", "url": "https://localhost/private.git", "commit": "a" * 40}).status_code == 409


def test_artifact_download_is_exact_attempt_and_supports_ranges(client, tmp_path):
    root = tmp_path / "result"
    outputs = root / "attempts/attempt-001/outputs"
    outputs.mkdir(parents=True)
    (outputs / "result.bin").write_bytes(b"0123456789")
    (outputs / "escape").symlink_to("/etc/passwd")
    client.app.state.runtime.index.upsert_run(RunIndexRow(project="demo", run_id="r", run_dir=str(root),
        attempts=[AttemptSummary(attempt_id="attempt-001", state="SUCCEEDED")]))
    endpoint = "/api/runs/demo/r/attempts/attempt-001/files"
    listing = client.get(endpoint).json()
    assert [item["path"] for item in listing["files"]] == ["outputs/result.bin"]
    response = client.get(endpoint + "/outputs/result.bin", headers={"Range": "bytes=2-5"})
    assert response.status_code == 206 and response.content == b"2345"
    assert response.headers["Content-Range"] == "bytes 2-5/10"
    assert client.get(endpoint + "/outputs/result.bin", headers={"Range": "bytes=90-"}).status_code == 416
    assert client.get(endpoint + "/outputs/escape").status_code == 404
    assert client.get(endpoint.replace("attempt-001", "attempt-002") + "/outputs/result.bin").status_code == 404


def test_removed_tracking_and_old_protocol_are_explicit(client):
    health = client.get("/api/health").json()
    assert health["api_protocol_version"] == 2
    assert "tracking.v1" not in health["capabilities"]
    assert client.get("/api/tracking").status_code == 404
    assert client.get("/api/observability").status_code == 404
    assert client.get("/api/projects", headers={"X-ML-Expd-Client-Protocol": "1"}).status_code == 426
    schema = client.get("/api/v2/openapi.json").json()
    assert "/api/source-imports/archive" in schema["paths"]
    assert not any("tracking" in key or "observability" in key for key in schema["paths"])
