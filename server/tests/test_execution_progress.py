"""Progress is nonblocking, identity-bound and independent of execution control."""
import hashlib
import json
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ml_exp_server.execution_progress import record_progress, progress_view
from ml_exp_server.image_builder import ImageBuilder, BUILD_CACHE, BUILD_PROGRESS, MANIFEST_TYPE
from ml_exp_server.schemas import RunIndexRow
from ml_exp_server.container_execution import RuntimeSpec
from tests.test_container_api import client, import_source, archive, wait_runtime, runtime
from tests.test_image_builder_boundary import builder
from tests.test_submissions import _app, _action


def test_progress_freshness_uses_real_log_activity_and_never_claims_failure(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    monkeypatch.setattr("ml_exp_server.execution_progress.time.time", lambda: 1000)
    assert progress_view(path, "PREPARED")["last_progress_unix"] is None
    assert progress_view(path, "EXECUTING", active=True, last_activity=999)["seconds_since_progress"] == 1
    record_progress(path, "STAGING", "Converting the image", timeout_seconds=1200)
    monkeypatch.setattr("ml_exp_server.execution_progress.time.time", lambda: 1201)
    delayed = progress_view(path, "EXECUTING", active=True)
    assert delayed["no_progress_warning"] and "do not resubmit" in delayed["diagnostic"]
    assert delayed["phase_timeout_seconds"] == 1200
    assert not progress_view(path, "EXECUTING", active=True, last_activity=1200)["no_progress_warning"]
    assert progress_view(path, "FAILED")["phase"] == "FAILED"
    assert not progress_view(path, "FAILED")["no_progress_warning"]
    for _ in range(70): record_progress(path, "STAGING", "Still the same owner")
    assert len(progress_view(path, "EXECUTING", active=True)["events"]) == 64


@pytest.mark.parametrize("change", [{"packaging_revision": "unsupported"}, {"image": "python:latest"}])
def test_historical_runtime_parser_still_validates_old_recipe_and_digest(change):
    with pytest.raises(ValueError):
        RuntimeSpec(source_id="source." + "a" * 64, entrypoint=["python3"],
                    **{"image": "registry.example/python@sha256:" + "a" * 64, **change})


@pytest.mark.parametrize("selector", ["image", "environment_id", "requirements"])
def test_new_runtime_api_rejects_ordinary_environment_selectors(client, selector):
    source = import_source(client)
    response = client.post("/api/projects/demo/runtimes/prepare", json={"source_id": source["source_id"],
                          selector: "registry.example/python@sha256:" + "a" * 64, "entrypoint": ["python3"]})
    assert response.status_code == 422
    assert not list((client.app.state.runtime.config.project_registry_root_path() / "runtime-bundles").glob("**/runtime.*.json"))


def test_dockerfile_build_progress_does_not_wait_for_builder_lock(client, tmp_path, monkeypatch):
    source = import_source(client, archive({"Dockerfile": ("FROM registry.example/python@sha256:" + "a" * 64 + "\n").encode(), "train.py": b"print('train')"}))
    root = client.app.state.runtime.config.project_registry_root_path()
    builder = ImageBuilder({"source_root": str(root / "source-revisions/sources"), "state_root": str(tmp_path / "builds"),
                            "repository": "registry.example/runtime", "publisher": "buildkit", "allow_dockerfile_builds": True})
    release, started = threading.Event(), threading.Event()
    def publish(tag, context):
        started.set()
        assert release.wait(5)
        return "registry.example/runtime@sha256:" + "b" * 64
    monkeypatch.setattr(builder, "_publish_buildkit", publish)
    monkeypatch.setattr("ml_exp_server.container_execution.builder_request", lambda socket, payload: builder.request(payload))
    monkeypatch.setattr("ml_exp_server.api.container_routes.builder_request", lambda socket, payload: builder.request(payload))
    value = client.post("/api/projects/demo/runtimes/prepare", json={"source_id": source["source_id"], "entrypoint": ["python3", "train.py"]}).json()
    endpoint = "/api/projects/demo/runtimes/" + value["runtime_id"]
    before = client.get(endpoint + "/progress").json()
    assert before["progress"]["phase"] == "PREPARED"
    try:
        assert client.post(endpoint + "/execute", json={"confirmation": value["confirmation"]}).status_code == 202
        assert started.wait(5)
        observed = client.get(endpoint + "/progress").json()
        assert observed["status"] == "EXECUTING" and observed["progress"]["phase"] == "PREPARING_CONTEXT"
        assert observed["progress"]["last_progress_unix"] is not None
    finally: release.set()
    assert wait_runtime(client, endpoint)["status"] == "READY"
    assert client.get(endpoint + "/progress").json()["progress"]["phase"] == "READY"


def test_builder_cache_is_remote_project_scoped_and_reuses_exact_receipt(builder):
    value, request, _ = builder
    value.config["registry_cache"] = True
    caches = []
    def publish(tag, context):
        assert not (context / "cache-ref").exists()
        caches.append(BUILD_CACHE.get())
        return "registry.example/runtime@sha256:" + "b" * 64
    value._publish_buildkit = publish
    result = value.request(request)
    assert value.request(request) == result and len(caches) == 1
    assert caches[0] == "registry.example/results:buildcache-" + hashlib.sha256(b"demo").hexdigest()[:24]
    before = value.request({**request, "operation": "progress"})
    assert before["progress"]["phase"] == "READY"
    assert value.request({**request, "operation": "progress", "source_id": "source." + "c" * 64})["progress"]["last_progress_unix"] is None


def test_registry_cache_flags_and_phase_steps_do_not_change_image_verification(tmp_path):
    builder = ImageBuilder({"state_root": str(tmp_path / "builds"), "repository": "registry.example/runtime"})
    config = "sha256:" + "a" * 64
    raw = json.dumps({"mediaType": MANIFEST_TYPE, "config": {"digest": config}})
    digest = "sha256:" + hashlib.sha256(raw.encode()).hexdigest()
    commands = []
    def docker(args, **kwargs):
        commands.append(args)
        (tmp_path / "buildkit.json").write_text(json.dumps({"containerimage.digest": digest, "containerimage.config.digest": config}))
    builder._docker = docker
    builder._skopeo = lambda *args, **kwargs: raw
    progress = BUILD_PROGRESS.set(tmp_path / "progress.json")
    cache = BUILD_CACHE.set("registry.example/runtime:cache")
    try:
        assert builder._buildkit_image("registry.example/runtime:bundle", tmp_path, "private").endswith(digest)
    finally:
        BUILD_CACHE.reset(cache); BUILD_PROGRESS.reset(progress)
    assert "--cache-from" in commands[0] and "--cache-to" in commands[0]
    assert "ignore-error=true" in commands[0][commands[0].index("--cache-to") + 1]
    assert [e["phase"] for e in progress_view(tmp_path / "progress.json", "EXECUTING", active=True)["events"]] == ["BUILDING_AND_PUSHING", "VERIFYING_IMAGE"]


def test_submission_progress_tracks_stage_and_keeps_exact_queue_evidence(tmp_path):
    app, runner = _app(tmp_path)
    with TestClient(app) as api:
        prepared = api.post("/api/experiments/demo/run-a/submissions/prepare", json={"max_gpu_hours": 2}).json()
        endpoint = "/api/submissions/" + prepared["submission_id"]
        assert api.get(endpoint + "/progress").json()["phase"] == "PREPARED"
        api.post(endpoint + "/authorize", json={"note": "test"})
        api.post(endpoint + "/execute", json={"confirmation": prepared["confirmation"]})
        _action(api, prepared["submission_id"])
        result = api.get(endpoint + "/progress").json()
        assert result["phase"] == "QUEUED" and result["queue"]["estimated_start_at"] is None
        assert [e["phase"] for e in result["events"]] == ["VALIDATING_SOURCE", "STAGING", "SCHEDULER_SUBMITTING", "VERIFYING_SUBMISSION"]
        for attempt, reason in (("attempt-002", None), ("attempt-001", "Resources")):
            api.app.state.index.upsert_run(RunIndexRow(project="demo", run_id="run-a", run_dir=str(tmp_path),
                scheduler_state="QUEUED", evidence={"scheduler": {"state": "QUEUED", "attempt_id": attempt, "as_of": 1000, "detail": {"reason": reason}}}))
            result = api.get(endpoint + "/progress").json()
            assert result["queue"]["reason"] == reason
        assert api.get("/api/submissions/action-unknown/progress").status_code == 404
