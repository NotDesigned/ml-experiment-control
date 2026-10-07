"""Shipped client over real HTTP: server preparation, dispatch and verified output.

Only the GPU scheduler and image publisher are injected; durable Actions,
controller/outbox, source/Run identities and artifact transfer are real.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from experiment_control.preflight import PreflightReport
from ml_exp_server.actions.service import ActionService
from ml_exp_server.api.app import create_app
from ml_exp_server.artifact_store import ArtifactStore
from ml_exp_server.container_controller import Controller
from ml_exp_server.dockerfile_build import inspect_dockerfile, managed_dockerfile
from ml_exp_server.image_builder import bundle_id
from ml_exp_server.schemas import ServerConfig
from ml_exp_server.worker_artifacts import archive_outputs
from ml_exp_server.worker_contract import CAPABILITIES, WORKER_CONTRACT, worker_digest
from tests.test_api_quickstart import cli, http_server, token
from tests.test_container_api import client
from tests.test_sensecore_data_workflow import stored


def test_stdlib_client_matches_both_backends_and_recovers_exact_outputs(client, stored, tmp_path, monkeypatch):
    auth = token(tmp_path)
    config = ServerConfig(index_db=str(tmp_path / "http/index.sqlite"), run_root=str(tmp_path / "http/runs"),
        action_root=str(tmp_path / "http/actions"), project_registry_root=str(tmp_path / "http/projects"),
        collector_enabled=False, telemetry={"enabled": False}, http_auth={"bearer_token_file": str(auth)},
        action_runtime={"allow_source_imports": True, "allow_project_writes": True, "allow_scheduler_mutations": True},
        container_execution={"profiles_file": client.app.state.config.container_execution.profiles_file,
            "builder_socket": "fake.sock", "artifact_store_file": str(stored[0])})
    builds, launches = [], []
    def build(socket, request):
        builds.append(request)
        tree = config.project_registry_root_path() / "source-revisions/sources" / request["project"] / request["source_id"] / "tree"
        inspection = inspect_dockerfile(tree, request["dockerfile"])
        return {"project": request["project"], "source_id": request["source_id"], "base_image": request["base_image"],
            "image": "registry.example/test@sha256:" + "b" * 64,
            "bundle_id": bundle_id(request["project"], request["source_id"], request["base_image"], dockerfile_path=request["dockerfile"]),
            "worker_contract": WORKER_CONTRACT, "worker_sha256": worker_digest(), "capabilities": list(CAPABILITIES),
            "dockerfile": {k: v for k, v in inspection.items() if k != "text"},
            "dockerfile_sha256": hashlib.sha256(managed_dockerfile(inspection, request["source_id"]).encode()).hexdigest()}
    monkeypatch.setattr("ml_exp_server.container_execution.builder_request", build)
    app = create_app(config, poll=False)

    def runner(command, *, cwd, timeout):
        campaign = yaml.safe_load(Path(command[2]).read_text())
        verb = command[3]
        run_id = command[command.index("--run") + 1]
        attempt = command[command.index("--attempt-id") + 1] if "--attempt-id" in command else "attempt-001"
        controller = Controller(campaign, run_id, attempt)
        campaign_id = command[command.index("--campaign-id") + 1] if "--campaign-id" in command else None
        if verb == "submit" and "--dry-run" in command:
            payload = [{**controller.prepare(campaign_id), "scheduler_mutated": False}]
        elif verb == "submit":
            kind = controller.backend.kind
            controller.backend.preflight = lambda *a, **kw: PreflightReport(kind, "submit", ())
            controller.backend.recover_submission = lambda *a: None
            def submit(*a, **kw):
                launches.append(kind)
                outputs = controller.attempt / "outputs"
                outputs.mkdir(parents=True, exist_ok=True)
                # CPU execution stands in for the remote container. The client
                # still retrieves files through real exact-Attempt transfer.
                subprocess.run([sys.executable, str(controller.source() / "train.py")], check=True,
                               env={**os.environ, "OUTPUT_DIR": str(outputs)}, capture_output=True)
                store = ArtifactStore(stored[0], config.project_registry_root_path())
                _, capability, limit = store.issue(campaign["project"], run_id, attempt, controller.root, ["**/*"])
                archive = io.BytesIO()
                archive_outputs(outputs, archive, limit, ["**/*"])
                size = archive.tell(); archive.seek(0)
                store.receive(campaign["project"], run_id, attempt, capability, archive, size)
                return "mock-" + kind + "-" + run_id
            controller.backend.submit = submit
            payload = [controller.submit(campaign_id)]
        elif verb == "status":
            record = controller.store.load_backend(attempt)
            controller.backend.status = lambda *a: {"state": "SUCCEEDED", "backend_job_id": record["backend_job_id"]}
            payload = [controller.status()]
        else:
            payload = [{"ready": True}]
        return {"returncode": 0, "timeout": False, "payload": payload, "stdout": "", "stderr": ""}

    def initialize(runtime):
        runtime.action_service = ActionService(runtime.action_store, config.action_runtime, runner,
                                              actor_provider=lambda: "trusted:operator")
    app.state.runtime_initializers.append(initialize)
    source = tmp_path / "source"
    source.mkdir()
    (source / "Dockerfile").write_text("FROM registry.example/python@sha256:" + "a" * 64 + "\n")
    (source / "train.py").write_text("import os,json,pathlib\np=pathlib.Path(os.environ['OUTPUT_DIR'])\n(p/'summary.json').write_text(json.dumps({'steps':3,'result':6}))\n")
    with http_server(app) as base:
        for backend in ("slurm", "sensecore"):
            experiment = tmp_path / (backend + ".json")
            experiment.write_text(json.dumps({"project": "http-study", "run_id": backend, "source": "source",
                "executor_selector": {"backend": backend}, "max_gpu_hours": 0.2}))
            state = tmp_path / (backend + ".state.json")
            prepared = cli(base, auth, "experiment", experiment, "--state", state, "--seconds", "10")
            assert prepared["preparation"]["status"] == "READY", prepared
            assert prepared["submission"]["ready"], json.dumps(prepared["submission"], indent=2)
            assert not launches or launches == ["slurm"]
            out = tmp_path / (backend + "-outputs")
            finished = cli(base, auth, "experiment", experiment, "--state", state, "--resume", "--execute", "--seconds", "10", "--download-to", out)
            assert finished["result"]["scheduler_state"] == "SUCCEEDED"
            assert json.loads((out / "outputs/summary.json").read_text()) == {"steps": 3, "result": 6}
            cli(base, auth, "experiment", experiment, "--state", state, "--resume", "--execute", "--seconds", "10", "--download-to", out)
        assert launches == ["slurm", "sensecore"] and len(builds) == 1
