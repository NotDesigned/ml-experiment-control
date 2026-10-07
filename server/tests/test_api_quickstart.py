"""Run the shipped stdlib CLI against real loopback HTTP, never live schedulers."""
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time

import pytest
import uvicorn
import yaml

from ml_exp_client import api as client_module
from ml_exp_server.api.app import create_app
from ml_exp_server.artifact_store import ArtifactStore
from ml_exp_server.container_controller import Controller
from ml_exp_server.container_worker import archive_outputs
from ml_exp_server.image_builder import bundle_id
from ml_exp_server.environment_build import dockerfile
from ml_exp_server.dockerfile_build import managed_dockerfile, inspect_dockerfile
from ml_exp_server.worker_contract import WORKER_CONTRACT, CAPABILITIES, worker_digest
from ml_exp_server.schemas import AttemptSummary, RunIndexRow, ServerConfig
from tests.test_submissions import _app


ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def http_server(app):
    # Exercise the same stripped proxy prefix used by a real external client.
    async def prefixed(scope, receive, send):
        if scope["type"] == "http":
            assert scope["path"].startswith("/ml-expd/")
            scope = dict(scope, path=scope["path"][len("/ml-expd"):])
            scope["raw_path"] = scope["raw_path"][len(b"/ml-expd"):]
        await app(scope, receive, send)

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(prefixed, log_level="error", lifespan="on"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert thread.is_alive() and time.monotonic() < deadline, "HTTP daemon did not start"
            time.sleep(0.01)
        yield f"http://127.0.0.1:{port}/ml-expd"
    finally:
        server.should_exit = True
        thread.join(10)
        listener.close()
        assert not thread.is_alive(), "HTTP daemon did not stop"


def cli(base, token_file, *arguments, success=True):
    env = dict(os.environ, ML_EXPD_API_URL=base, ML_EXPD_API_TOKEN_FILE=str(token_file))
    env.pop("ML_EXPD_API_TOKEN", None)
    result = subprocess.run([sys.executable, "-m", "ml_exp_client", *map(str, arguments)],
                            env=env, capture_output=True, text=True, timeout=30)
    assert token_file.read_text().strip() not in result.stdout + result.stderr
    if success:
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    assert result.returncode == 2
    return result


def token(tmp_path):
    path = tmp_path / "token"
    path.write_text("quickstart-test-token-" + "x" * 32)
    path.chmod(0o600)
    return path


def test_client_import_pack_two_profiles_execute_program_upload_download(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiment_control.backends.sensecore_rest import SenseCoreREST
    def selected(backend, *, gpus):
        return {**backend, "pool_selection_evidence": {
            "policy": "highest_spot", "configured_aec2": backend["aec2"], "selected": backend["aec2"]}}
    monkeypatch.setattr(SenseCoreREST, "from_environment", lambda: SimpleNamespace(select_pool=selected))
    auth = token(tmp_path)
    s3_config = json.loads((ROOT / "server/examples/artifact-store.json").read_text())
    s3_file = tmp_path / "s3.json"
    s3_file.write_text(json.dumps(s3_config))
    config = ServerConfig(index_db=str(tmp_path / "index.sqlite"), run_root=str(tmp_path / "runs"),
                          action_root=str(tmp_path / "actions"), project_registry_root=str(tmp_path / "projects"),
                          http_auth={"bearer_token_file": str(auth)}, collector_enabled=False,
                          action_runtime={"allow_source_imports": True, "allow_project_writes": True},
                          container_execution={"profiles_file": str(ROOT / "server/examples/executors.yaml"),
                          "artifact_store_file": str(s3_file), "builder_socket": "injected-builder.sock"})
    calls = []
    def build(_socket, request):
        if request["operation"] == "progress":
            return {"progress": {"phase": "BUILDING_AND_PUSHING"}, "lines": []}
        calls.append(request)
        return {"project": request["project"], "source_id": request["source_id"],
                "base_image": request["base_image"], "image": "registry.example/team/run@sha256:" + "b" * 64,
                "bundle_id": bundle_id(request["project"], request["source_id"], request["base_image"], dockerfile_path=request.get("dockerfile")),
                "worker_contract": WORKER_CONTRACT, "worker_sha256": worker_digest(), "capabilities": list(CAPABILITIES),
                "dockerfile_sha256": hashlib.sha256(managed_dockerfile(inspect_dockerfile(config.project_registry_root_path() / "source-revisions/sources" / request["project"] / request["source_id"] / "tree", request["dockerfile"]), request["source_id"]).encode()).hexdigest(), "dockerfile": {k: v for k, v in inspect_dockerfile(config.project_registry_root_path() / "source-revisions/sources" / request["project"] / request["source_id"] / "tree", request["dockerfile"]).items() if k != "text"}}
    monkeypatch.setattr("ml_exp_server.container_execution.builder_request", build)
    monkeypatch.setattr("ml_exp_server.api.container_routes.builder_request", build)
    objects = {}
    class Body(io.BytesIO):
        def iter_chunks(self, chunk_size):
            while part := self.read(chunk_size):
                yield part
    class S3:
        def upload_fileobj(self, stream, bucket, key, **kwargs):
            objects[bucket, key] = stream.read()
        def get_object(self, *, Bucket, Key):
            return {"Body": Body(objects[Bucket, Key])}
    monkeypatch.setattr(ArtifactStore, "client", lambda self: S3())
    app = create_app(config, poll=False)
    with http_server(app) as base:
        schema = tmp_path / "openapi.json"
        check = cli(base, auth, "check", "--schema", schema)
        assert check["health"]["api_protocol_version"] == 2
        assert len(check["executors"]["executors"]) == 2
        assert "/api/source-imports/archive" in json.loads(schema.read_text())["paths"]
        source = tmp_path / "source"
        initialized = cli(base, auth, "init", source, "--base-image", "registry.example/team/base@sha256:" + "a" * 64)
        assert initialized["entrypoint"] == ["python", "train.py"]
        state = tmp_path / "runtime.json"
        runtime = cli(base, auth, "pack", "--project", "quickstart", "--source", source,
                      "--dockerfile", "Dockerfile", "--state", state)
        assert runtime["status"] == "READY" and len(calls) == 1
        assert cli(base, auth, "runtime", "--state", state)["runtime_id"] == runtime["runtime_id"]
        assert "state file exists" in cli(base, auth, "pack", "--project", "quickstart", "--source", source,
                                        "--dockerfile", "unused", "--state", state, success=False).stderr
        for name in ("wyd-l40s", "sensecore-1gpu"):
            frozen = cli(base, auth, "create", "--runtime-state", state, "--run", name, "--executor", name)
            assert frozen["state"] == "NOT_SUBMITTED" and frozen["image"] == runtime["image"]
        root = Path(app.state.runtime.project("quickstart").base_dir)
        campaign = yaml.safe_load((root / "experiments/campaigns/run-wyd-l40s.yaml").read_text())
        campaign["local_root"] = str(config.project_run_root_path("quickstart"))
        controller = Controller(campaign, "wyd-l40s", "attempt-001")
        controller.prepare("campaign." + "c" * 64)
        run_root = controller.root
        outputs = run_root / "attempts/attempt-001/outputs"
        env = dict(os.environ, OUTPUT_DIR=str(outputs), PROJECT_NAME="quickstart", RUN_ID="wyd-l40s",
                   ATTEMPT_ID="attempt-001", SOURCE_ID=runtime["spec"]["source_id"])
        program = subprocess.run([sys.executable, str(source / "train.py"), "--steps", "4"],
                                 env=env, capture_output=True, text=True, check=True)
        assert json.loads(program.stdout)["steps"] == 4
        assert len((outputs / "metrics.jsonl").read_text().splitlines()) == 4
        app.state.runtime.index.upsert_run(RunIndexRow(project="quickstart", run_id="wyd-l40s", run_dir=str(run_root),
            scheduler_state="SUCCEEDED", attempts=[AttemptSummary(attempt_id="attempt-001", state="SUCCEEDED")]))
        store = ArtifactStore(s3_file, config.project_registry_root_path())
        _, capability, limit = store.issue("quickstart", "wyd-l40s", "attempt-001", run_root, ["**/*"])
        archive = io.BytesIO()
        assert archive_outputs(outputs, archive, limit, ["**/*"]) == 2
        data = archive.getvalue()
        from urllib.request import Request, urlopen
        with urlopen(Request(base + "/api/artifact-transfers/quickstart/wyd-l40s/attempt-001", data=data,
                             method="PUT", headers={"Authorization": "Bearer " + capability})) as response:
            assert response.status == 200
        watched = cli(base, auth, "watch", "--project", "quickstart", "--run", "wyd-l40s", "--seconds", "1")
        assert watched["attempts"]["current_attempt_id"] == "attempt-001"
        destination = tmp_path / "download"
        report = cli(base, auth, "download", "--project", "quickstart", "--run", "wyd-l40s", "--attempt", "attempt-001", "--out", destination)
        assert report["archive_sha256"] == hashlib.sha256(data).hexdigest()
        assert (destination / "outputs/summary.json").read_bytes() == (outputs / "summary.json").read_bytes()
        bad = cli(base, auth, "download", "--project", "quickstart", "--run", "wyd-l40s", "--attempt", "attempt-002",
                  "--out", tmp_path / "wrong", success=False)
        assert "HTTP 404" in bad.stderr and not (tmp_path / "wrong").exists()
        key = next(iter(objects))
        objects[key] = b"corrupt stored archive"
        bad = cli(base, auth, "download", "--project", "quickstart", "--run", "wyd-l40s", "--attempt", "attempt-001",
                  "--out", tmp_path / "corrupt", success=False)
        assert "SHA256" in bad.stderr or "connection failed" in bad.stderr


@pytest.mark.parametrize("synchronous", [False, True])
def test_client_submission_confirmation_polling_and_no_duplicate_submit(tmp_path, synchronous):
    app, runner = _app(tmp_path)
    auth = token(tmp_path)
    app.state.config.http_auth.bearer_token_file = str(auth)
    with http_server(app) as base:
        if synchronous:
            service = app.state.submission_service
            original = service.begin_execute
            def sync_execute(sid, confirmation):
                result, pending = original(sid, confirmation)
                if pending is not None:
                    app.state.application.finish_action_execution(pending)
                return service.get(sid), None
            service.begin_execute = sync_execute
        state = tmp_path / "submission.json"
        prepared = cli(base, auth, "prepare", "--project", "demo", "--run", "run-a", "--max-gpu-hours", "2", "--state", state)
        assert prepared["status"] == "PREPARED"
        assert not any(call[3] == "submit" and "--dry-run" not in call for call in runner.calls)
        cli(base, auth, "execute", "--state", state, "--confirm", "EXECUTE wrong", success=False)
        assert not any(call[3] == "submit" and "--dry-run" not in call for call in runner.calls)
        result = cli(base, auth, "execute", "--state", state, "--confirm", prepared["confirmation"])
        assert result["status"] == "VERIFIED"
        cli(base, auth, "execute", "--state", state, "--confirm", prepared["confirmation"])
        cli(base, auth, "submission", "--id", prepared["submission_id"], "--reconcile")
        assert len([call for call in runner.calls if call[3] == "submit" and "--dry-run" not in call]) == 1


def test_client_unknown_submission_reconciles_without_replaying(tmp_path):
    app, runner = _app(tmp_path)
    runner.status_visible = False
    auth = token(tmp_path)
    app.state.config.http_auth.bearer_token_file = str(auth)
    with http_server(app) as base:
        state = tmp_path / "submission.json"
        prepared = cli(base, auth, "prepare", "--project", "demo", "--run", "run-a", "--max-gpu-hours", "2", "--state", state)
        cli(base, auth, "execute", "--state", state, "--confirm", prepared["confirmation"], success=False)
        assert json.loads(state.read_text())["status"] == "RECONCILE_REQUIRED"
        runner.status_visible = True
        result = cli(base, auth, "submission", "--id", prepared["submission_id"], "--reconcile")
        assert result["status"] == "VERIFIED"
        assert len([call for call in runner.calls if call[3] == "submit" and "--dry-run" not in call]) == 1


def test_examples_and_new_documentation_are_consistent():
    config = ServerConfig.model_validate(yaml.safe_load((ROOT / "server/examples/source-api.yaml").read_text()))
    assert not config.action_runtime.allow_scheduler_mutations
    builder = json.loads((ROOT / "server/examples/image-builder.json").read_text())
    assert builder["source_root"] == config.project_registry_root + "/source-revisions/sources"
    assert builder["socket"] == config.container_execution.builder_socket
    for name in ("README.md", "client/README.md", "docs/api-quickstart.md", "docs/operator-guide.md", "docs/library-integration.md",
                 "docs/source-api.md", "docs/http_contract.md", "docs/development.md", "CONTRIBUTING.md"):
        path = ROOT / name
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" not in link and not link.startswith("#"):
                assert (path.parent / link.split("#")[0]).exists(), (name, link)
    with pytest.raises(client_module.ClientError, match="HTTPS"):
        client_module.Client("http://remote.example", "test")
    with pytest.raises(client_module.ClientError, match="set ML_EXPD"):
        client_module.Client("https://api.example", "")


def test_client_rejects_credential_paths_links_and_redirects(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_text("pass\n")
    (source / ".env").write_text("private")
    with pytest.raises(client_module.ClientError, match="credential"):
        client_module.source_archive(source)
    (source / ".env").unlink()
    (source / "link").symlink_to(source / "train.py")
    with pytest.raises(client_module.ClientError, match="link"):
        client_module.source_archive(source)
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, RedirectResponse
    from starlette.routing import Route
    forwarded = []
    async def redirect(request):
        return RedirectResponse("/ml-expd/api/other")
    async def other(request):
        forwarded.append(request.headers.get("Authorization"))
        return JSONResponse({"status": "unexpected"})
    app = Starlette(routes=[Route("/api/health", redirect), Route("/api/other", other)])
    with http_server(app) as base:
        with pytest.raises(client_module.ClientError, match="HTTP 307"):
            client_module.Client(base, "never-forward-this-token").negotiate()
    assert forwarded == []


def test_lost_execute_response_is_observed_without_replaying_submit(tmp_path):
    app, runner = _app(tmp_path)
    auth = token(tmp_path)
    app.state.config.http_auth.bearer_token_file = str(auth)
    with http_server(app) as base:
        client = client_module.Client(base, auth.read_text().strip(), timeout=5)
        prepared = client.call("/api/experiments/demo/run-a/submissions/prepare", data={"max_gpu_hours": 2})
        endpoint = "/api/submissions/" + prepared["submission_id"]
        client.call(endpoint + "/authorize", data={})
        service = app.state.submission_service
        original = service.begin_execute
        def execute_with_delayed_response(sid, confirmation):
            result, pending = original(sid, confirmation)
            if pending is not None:
                app.state.application.finish_action_execution(pending)
            time.sleep(0.3)  # Exact scheduler effect exists before the client times out.
            return service.get(sid), None
        service.begin_execute = execute_with_delayed_response
        client.timeout = 0.1
        with pytest.raises(client_module.ClientError, match="inspect saved IDs"):
            client.call(endpoint + "/execute", data={"confirmation": prepared["confirmation"]})
        client.timeout = 5
        assert client.wait(endpoint, interval=0.01)["status"] == "VERIFIED"
        state = tmp_path / "submission.json"
        client_module.save(state, prepared)
        assert cli(base, auth, "execute", "--state", state, "--confirm", prepared["confirmation"])["status"] == "VERIFIED"
        assert len([call for call in runner.calls if call[3] == "submit" and "--dry-run" not in call]) == 1
