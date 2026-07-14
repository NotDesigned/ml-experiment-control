from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from ml_exp_server import project_imports
from ml_exp_server.api.app import create_app
from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.schemas import ActionRuntimeConfig, ServerConfig


def config(tmp_path: Path, *, writes: bool = False) -> ServerConfig:
    return ServerConfig(
        index_db=str(tmp_path / "state" / "index.sqlite"),
        action_root=str(tmp_path / "state" / "actions"),
        project_registry_root=str(tmp_path / "state" / "projects"),
        project_import_roots=[str(tmp_path)],
        action_runtime=ActionRuntimeConfig(allow_project_writes=writes),
        collector_enabled=False,
    )


def preview(client: TestClient, repository: Path, **extra):
    return client.post("/api/project-imports/preview", json={
        "source": {"kind": "daemon_path", "repository_root": str(repository)},
        **extra,
    })


def test_preview_discovers_minimal_manifest_without_writing(tmp_path):
    repository = tmp_path / "my-study"
    (repository / "runs").mkdir(parents=True)
    controller = repository / "tools" / "experimentctl.py"
    controller.parent.mkdir()
    controller.write_text("print('controller')\n", encoding="utf-8")

    with TestClient(create_app(config(tmp_path))) as client:
        response = preview(client, repository)

    assert response.status_code == 200
    plan = response.json()
    assert plan["operation"] == "GENERATE_AND_REGISTER"
    assert plan["manifest"] == {
        "schema_version": 1,
        "project": "my-study",
        "title": "My Study",
        "run_roots": ["runs"],
        "controller": {
            "python": "python3", "experimentctl": "tools/experimentctl.py",
            "workdir": ".", "capabilities": {},
        },
    }
    assert plan["confirmation"] == f"IMPORT {plan['import_id']}"
    assert not (repository / "experiments" / "research_project.yaml").exists()


def test_generated_manifest_requires_policy_and_exact_confirmation(tmp_path):
    repository = tmp_path / "demo"
    repository.mkdir()
    with TestClient(create_app(config(tmp_path))) as client:
        plan = preview(client, repository).json()
        wrong = client.post(
            f"/api/project-imports/{plan['import_id']}/execute",
            json={"confirmation": "IMPORT wrong"},
        )
        blocked = client.post(
            f"/api/project-imports/{plan['import_id']}/execute",
            json={"confirmation": plan["confirmation"]},
        )

    assert wrong.status_code == 409
    assert blocked.status_code == 409
    assert "disabled by daemon policy" in blocked.json()["detail"]
    assert not (repository / "experiments" / "research_project.yaml").exists()


def test_execute_generates_registers_and_indexes_project(tmp_path):
    repository = tmp_path / "demo"
    repository.mkdir()
    with TestClient(create_app(config(tmp_path, writes=True))) as client:
        plan = preview(client, repository, title="Demo Project").json()
        response = client.post(
            f"/api/project-imports/{plan['import_id']}/execute",
            json={"confirmation": plan["confirmation"]},
        )
        lifecycle = client.get("/api/project-lifecycle").json()

    assert response.status_code == 200
    registration = response.json()["registration"]
    assert registration["project"]["project"] == "demo"
    assert registration["initial_index"]["status"] == "COMPLETED"
    manifest = yaml.safe_load(
        (repository / "experiments" / "research_project.yaml").read_text()
    )
    assert manifest["title"] == "Demo Project"
    assert lifecycle["projects"][0]["project"] == "demo"


def test_execute_rejects_manifest_created_after_preview(tmp_path):
    repository = tmp_path / "demo"
    repository.mkdir()
    manifest = repository / "experiments" / "research_project.yaml"
    with TestClient(create_app(config(tmp_path, writes=True))) as client:
        plan = preview(client, repository).json()
        manifest.parent.mkdir()
        manifest.write_text("schema_version: 1\n", encoding="utf-8")
        response = client.post(
            f"/api/project-imports/{plan['import_id']}/execute",
            json={"confirmation": plan["confirmation"]},
        )

    assert response.status_code == 409
    assert response.headers["X-ML-Expd-Error-Code"] == "PROJECT_IMPORT_STALE"


def test_execute_rejects_repository_replaced_by_outside_symlink(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    config_value = config(tmp_path, writes=True)
    config_value.project_import_roots = [str(allowed)]
    repository = allowed / "demo"
    repository.mkdir()

    with TestClient(create_app(config_value)) as client:
        plan = preview(client, repository).json()
        repository.rename(allowed / "demo-moved")
        repository.symlink_to(outside, target_is_directory=True)
        response = client.post(
            f"/api/project-imports/{plan['import_id']}/execute",
            json={"confirmation": plan["confirmation"]},
        )

    assert response.status_code == 409
    assert response.headers["X-ML-Expd-Error-Code"] == "PROJECT_IMPORT_STALE"
    assert not (outside / "experiments" / "research_project.yaml").exists()


def test_execute_revalidates_current_import_allowlist(tmp_path):
    repository = tmp_path / "demo"
    repository.mkdir()
    config_value = config(tmp_path, writes=True)
    with TestClient(create_app(config_value)) as client:
        plan = preview(client, repository).json()
        config_value.project_import_roots = [str(tmp_path / "different")]
        response = client.post(
            f"/api/project-imports/{plan['import_id']}/execute",
            json={"confirmation": plan["confirmation"]},
        )

    assert response.status_code == 409
    assert response.headers["X-ML-Expd-Error-Code"] == "PROJECT_IMPORT_STALE"
    assert not (repository / "experiments" / "research_project.yaml").exists()


def test_completed_import_remains_idempotent_after_repository_and_policy_drift(tmp_path):
    repository = tmp_path / "demo"
    repository.mkdir()
    config_value = config(tmp_path, writes=True)
    with TestClient(create_app(config_value)) as client:
        plan = preview(client, repository).json()
        first = client.post(
            f"/api/project-imports/{plan['import_id']}/execute",
            json={"confirmation": plan["confirmation"]},
        ).json()
        repository.rename(tmp_path / "demo-moved")
        config_value.project_import_roots = []
        repeated = client.post(
            f"/api/project-imports/{plan['import_id']}/execute",
            json={"confirmation": plan["confirmation"]},
        )

    assert repeated.status_code == 200
    assert repeated.json() == first


def test_execute_anchors_manifest_write_against_post_identity_path_swap(
    tmp_path, monkeypatch,
):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    repository = allowed / "demo"
    repository.mkdir()
    moved = allowed / "demo-moved"
    config_value = config(tmp_path, writes=True)
    config_value.project_import_roots = [str(allowed)]
    real_identity = project_imports._repository_identity

    def swap_after_identity(root, **kwargs):
        identity = real_identity(root, **kwargs)
        if root == repository:
            repository.rename(moved)
            repository.symlink_to(outside, target_is_directory=True)
        return identity

    with TestClient(create_app(config_value)) as client:
        plan = preview(client, repository).json()
        monkeypatch.setattr(project_imports, "_repository_identity", swap_after_identity)
        response = client.post(
            f"/api/project-imports/{plan['import_id']}/execute",
            json={"confirmation": plan["confirmation"]},
        )

    assert response.status_code == 409
    assert response.headers["X-ML-Expd-Error-Code"] == "PROJECT_IMPORT_STALE"
    assert not (outside / "experiments" / "research_project.yaml").exists()


def test_execute_rejects_non_git_repository_content_change(tmp_path):
    repository = tmp_path / "plain"
    repository.mkdir()
    controller = repository / "experimentctl.py"
    controller.write_text("print('v1')\n", encoding="utf-8")
    with TestClient(create_app(config(tmp_path, writes=True))) as client:
        plan = preview(client, repository).json()
        controller.write_text("print('v2')\n", encoding="utf-8")
        response = client.post(
            f"/api/project-imports/{plan['import_id']}/execute",
            json={"confirmation": plan["confirmation"]},
        )

    assert response.status_code == 409
    assert response.headers["X-ML-Expd-Error-Code"] == "PROJECT_IMPORT_STALE"
    assert not (repository / "experiments" / "research_project.yaml").exists()


def test_execute_cleans_own_crash_leftover_before_identity_check(tmp_path):
    repository = tmp_path / "crash-recovery"
    repository.mkdir()
    with TestClient(create_app(config(tmp_path, writes=True))) as client:
        plan = preview(client, repository).json()
        experiments = repository / "experiments"
        experiments.mkdir()
        leftover = experiments / project_imports._manifest_temp_name(plan["import_id"])
        leftover.write_text("partially durable yaml", encoding="utf-8")
        response = client.post(
            f"/api/project-imports/{plan['import_id']}/execute",
            json={"confirmation": plan["confirmation"]},
        )

    assert response.status_code == 200
    assert response.json()["import"]["phase"] == "COMPLETED"
    assert not leftover.exists()


def test_existing_import_never_cleans_transaction_named_repository_file(tmp_path):
    repository = tmp_path / "existing-cleanup-scope"
    manifest = repository / "experiments" / "research_project.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(yaml.safe_dump({
        "schema_version": 1, "project": "existing-cleanup-scope",
        "title": "Existing", "run_roots": [],
    }), encoding="utf-8")
    with TestClient(create_app(config(tmp_path))) as client:
        plan = preview(client, repository).json()
        victim = manifest.parent / project_imports._manifest_temp_name(plan["import_id"])
        victim.write_text("repository-owned", encoding="utf-8")
        response = client.post(
            f"/api/project-imports/{plan['import_id']}/execute",
            json={"confirmation": plan["confirmation"]},
        )

    assert response.status_code == 409
    assert response.headers["X-ML-Expd-Error-Code"] == "PROJECT_IMPORT_STALE"
    assert victim.read_text(encoding="utf-8") == "repository-owned"


def test_closed_write_policy_never_cleans_generated_import_temp_name(tmp_path):
    repository = tmp_path / "closed-cleanup-scope"
    repository.mkdir()
    with TestClient(create_app(config(tmp_path))) as client:
        plan = preview(client, repository).json()
        experiments = repository / "experiments"
        experiments.mkdir()
        victim = experiments / project_imports._manifest_temp_name(plan["import_id"])
        victim.write_text("repository-owned", encoding="utf-8")
        response = client.post(
            f"/api/project-imports/{plan['import_id']}/execute",
            json={"confirmation": plan["confirmation"]},
        )

    assert response.status_code == 409
    assert "disabled by daemon policy" in response.json()["detail"]
    assert victim.read_text(encoding="utf-8") == "repository-owned"


def test_existing_manifest_preview_registers_without_project_write_policy(tmp_path):
    repository = tmp_path / "existing"
    manifest = repository / "experiments" / "research_project.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(yaml.safe_dump({
        "schema_version": 1, "project": "existing", "title": "Existing",
        "run_roots": [],
    }), encoding="utf-8")
    with TestClient(create_app(config(tmp_path))) as client:
        plan = preview(client, repository).json()
        response = client.post(
            f"/api/project-imports/{plan['import_id']}/execute",
            json={"confirmation": plan["confirmation"]},
        )
        repeated = preview(client, repository).json()

    assert plan["operation"] == "REGISTER_EXISTING"
    assert response.status_code == 200
    assert repeated["executed"] is True
    assert repeated["phase"] == "COMPLETED"


def test_existing_git_manifest_executes_with_anchored_filesystem_handle(tmp_path):
    repository = tmp_path / "existing-git"
    manifest = repository / "experiments" / "research_project.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(yaml.safe_dump({
        "schema_version": 1, "project": "existing-git", "title": "Existing Git",
        "run_roots": [],
    }), encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Test"], check=True,
    )
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-qm", "project"], check=True,
    )

    with TestClient(create_app(config(tmp_path))) as client:
        plan = preview(client, repository).json()
        response = client.post(
            f"/api/project-imports/{plan['import_id']}/execute",
            json={"confirmation": plan["confirmation"]},
        )

    assert plan["repository_identity"]["kind"] == "git"
    assert response.status_code == 200
    assert response.json()["import"]["phase"] == "COMPLETED"


def test_preview_rejects_ambiguous_or_unsupported_sources(tmp_path):
    with TestClient(create_app(config(tmp_path))) as client:
        relative = preview(client, Path("relative"))
        unsupported = client.post("/api/project-imports/preview", json={
            "source": {"kind": "git", "repository_root": "https://example/repo"},
        })

    assert relative.status_code == 409
    assert unsupported.status_code == 422


def test_preview_rejects_paths_outside_allowlist_and_skips_symlink_discovery(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    config_value = config(tmp_path)
    config_value.project_import_roots = [str(allowed)]
    repository = allowed / "repo"
    repository.mkdir()
    (repository / "runs").symlink_to(outside, target_is_directory=True)
    controller = outside / "experimentctl.py"
    controller.write_text("raise SystemExit\n", encoding="utf-8")
    (repository / "experimentctl.py").symlink_to(controller)
    outside_questions = outside / "questions"
    outside_questions.mkdir()
    (repository / "experiments").mkdir()
    (repository / "experiments" / "research_questions").symlink_to(
        outside_questions, target_is_directory=True,
    )
    outside_campaigns = outside / "campaigns"
    outside_campaigns.mkdir()
    (outside_campaigns / "escaped.yaml").write_text(
        "campaign: escaped\n", encoding="utf-8",
    )
    (repository / "experiments" / "campaigns").symlink_to(
        outside_campaigns, target_is_directory=True,
    )

    with TestClient(create_app(config_value)) as client:
        rejected = preview(client, outside)
        plan = preview(client, repository).json()

    assert rejected.status_code == 409
    assert "project_import_roots" in rejected.json()["detail"]
    assert plan["manifest"]["run_roots"] == []
    assert "controller" not in plan["manifest"]
    assert "research_questions_dir" not in plan["manifest"]
    assert "campaigns" not in plan["manifest"]


def test_preview_rejects_manifest_symlink_and_out_of_repository_references(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "experiments").symlink_to(outside, target_is_directory=True)

    with TestClient(create_app(config(tmp_path))) as client:
        response = preview(client, linked)
    assert response.status_code == 409
    assert "manifest_path may not traverse a symlink" in response.json()["detail"]

    cases = [
        ({"run_roots": ["../outside"]}, "run_roots[0]"),
        ({"research_questions_dir": str(outside)}, "research_questions_dir"),
        ({"campaigns": [{"name": "escaped", "file": str(outside / "x.yaml")}]},
         "campaigns[0].file"),
        ({"controller": {"python": "python3", "experimentctl": "tool.py",
                          "workdir": str(outside)}}, "controller.workdir"),
        ({"controller": {"python": "python3", "experimentctl": "../outside.py",
                          "workdir": "."}}, "controller.experimentctl"),
    ]
    for index, (extra, label) in enumerate(cases):
        repository = tmp_path / f"reference-{index}"
        manifest = repository / "experiments" / "research_project.yaml"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(yaml.safe_dump({
            "schema_version": 1,
            "project": f"reference-{index}",
            "title": "Reference",
            "run_roots": [],
            **extra,
        }), encoding="utf-8")
        with TestClient(create_app(config(tmp_path))) as client:
            response = preview(client, repository)
        assert response.status_code == 409
        assert label in response.json()["detail"]


def test_execute_rolls_forward_after_manifest_and_registry_crash_windows(
    tmp_path, monkeypatch,
):
    repository = tmp_path / "demo"
    repository.mkdir()
    with TestClient(create_app(config(tmp_path, writes=True))) as client:
        service = client.app.state.application.project_import_service
        plan = service.preview(repository)
        real_register = service.projects.register
        calls = 0

        def crash_before_register(path):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("crash after manifest")
            return real_register(path)

        monkeypatch.setattr(service.projects, "register", crash_before_register)
        try:
            service.execute(plan["import_id"], plan["confirmation"])
        except RuntimeError as exc:
            assert "crash after manifest" in str(exc)
        recovered = service.execute(plan["import_id"], plan["confirmation"])
        repeated = service.execute(plan["import_id"], plan["confirmation"])

    assert recovered == repeated
    assert recovered["import"]["phase"] == "COMPLETED"
    assert recovered["import"]["executed"] is True


def test_execute_reports_manifest_filesystem_failure_and_remains_retryable(
    tmp_path, monkeypatch,
):
    repository = tmp_path / "demo"
    repository.mkdir()
    with TestClient(create_app(config(tmp_path, writes=True))) as client:
        service = client.app.state.application.project_import_service
        plan = service.preview(repository)
        manifest = repository / "experiments" / "research_project.yaml"
        real_write_manifest = project_imports._atomic_write_manifest

        def fail_manifest(_root_fd, _payload, _import_id):
            raise OSError(30, "Read-only file system")

        monkeypatch.setattr(project_imports, "_atomic_write_manifest", fail_manifest)
        with pytest.raises(ApplicationError, match="could not be materialized") as caught:
            service.execute(plan["import_id"], plan["confirmation"])
        assert caught.value.code == "PROJECT_IMPORT_BLOCKED"
        stored = json.loads(
            (service.root / f"{plan['import_id']}.json").read_text(encoding="utf-8")
        )
        assert stored["phase"] == "PREPARED"
        assert not manifest.exists()

        monkeypatch.setattr(
            project_imports, "_atomic_write_manifest", real_write_manifest,
        )
        recovered = service.execute(plan["import_id"], plan["confirmation"])

    assert recovered["import"]["phase"] == "COMPLETED"
