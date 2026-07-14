"""Focused defensive coverage for Project import persistence and discovery."""

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from ml_exp_server.api.app import create_app
from ml_exp_server.application_errors import ApplicationError
from ml_exp_server import project_imports
from tests.test_project_imports import config


def git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True,
                   capture_output=True)


def service(client: TestClient):
    return client.app.state.application.project_import_service


def test_git_identity_ignore_and_discovery_variants(tmp_path):
    repository = tmp_path / "git-study"
    repository.mkdir()
    git(repository, "init", "-q")
    git(repository, "config", "user.email", "test@example.invalid")
    git(repository, "config", "user.name", "Test")
    (repository / "tracked.txt").write_text("base\n", encoding="utf-8")
    git(repository, "add", ".")
    git(repository, "commit", "-qm", "base")

    questions = repository / "experiments" / "research_questions"
    campaigns = repository / "experiments" / "campaigns"
    questions.mkdir(parents=True)
    campaigns.mkdir()
    (campaigns / "good.yml").write_text(
        "campaign: study\n", encoding="utf-8",
    )
    (campaigns / "bad.yml").write_text("[", encoding="utf-8")
    (campaigns / "list.yaml").write_text("- item\n", encoding="utf-8")
    (campaigns / "unsafe.yaml").write_text(
        "campaign: bad/id\n", encoding="utf-8",
    )
    outside = tmp_path / "outside.yml"
    outside.write_text("campaign: escaped\n", encoding="utf-8")
    (campaigns / "linked.yml").symlink_to(outside)

    with TestClient(create_app(config(tmp_path))) as client:
        plan = service(client).preview(repository)

    assert plan["repository_identity"]["kind"] == "git"
    assert plan["repository_identity_without_manifest"]["kind"] == "git"
    assert plan["manifest"]["research_questions_dir"].endswith(
        "research_questions"
    )
    assert plan["manifest"]["campaigns"] == [{
        "name": "study", "file": "experiments/campaigns/good.yml",
    }]
    assert plan["detected"]["campaigns"] == 1


def test_preview_rejects_resolve_root_project_and_existing_identity_edges(
    tmp_path, monkeypatch,
):
    with TestClient(create_app(config(tmp_path))) as client:
        import_service = service(client)
        real_resolve = Path.resolve
        broken = tmp_path / "broken"
        broken.mkdir()

        def fail_resolve(path, *args, **kwargs):
            if path == broken:
                raise OSError("gone")
            return real_resolve(path, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", fail_resolve)
        with pytest.raises(ApplicationError, match="gone"):
            import_service.preview(broken)
        monkeypatch.setattr(Path, "resolve", real_resolve)

        with pytest.raises(ApplicationError, match="non-root"):
            import_service.preview(Path("/"))

        odd = tmp_path / "..."
        odd.mkdir()
        with pytest.raises(ApplicationError, match="safe Project ID"):
            import_service.preview(odd)

        repository = tmp_path / "existing"
        manifest = repository / "experiments" / "research_project.yaml"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(yaml.safe_dump({
            "schema_version": 1, "project": "actual", "title": "Actual",
            "run_roots": [],
        }), encoding="utf-8")
        with pytest.raises(ApplicationError, match="conflicts with requested"):
            import_service.preview(repository, project="requested")

        with pytest.raises(ApplicationError, match="must use"):
            import_service.preview(repository, project="bad/id")


def test_locked_plan_validation_corruption_and_application_error_passthrough(
    tmp_path, monkeypatch,
):
    with TestClient(create_app(config(tmp_path))) as client:
        import_service = service(client)
        with pytest.raises(ApplicationError, match="invalid import identity"):
            import_service.execute("bad", "IMPORT bad")
        missing = "import-" + "a" * 24
        with pytest.raises(ApplicationError, match="not found"):
            import_service.execute(missing, f"IMPORT {missing}")

        import_service.root.mkdir(parents=True, exist_ok=True)
        corrupt = "import-" + "b" * 24
        (import_service.root / f"{corrupt}.json").write_text(
            "[]\n", encoding="utf-8",
        )
        with pytest.raises(ApplicationError, match="unreadable"):
            import_service.execute(corrupt, f"IMPORT {corrupt}")
        (import_service.root / f"{corrupt}.json").write_text(
            "{", encoding="utf-8",
        )
        with pytest.raises(ApplicationError, match="unreadable"):
            import_service.execute(corrupt, f"IMPORT {corrupt}")

        class CaughtApplicationError(ValueError):
            def __init__(self, message, **kwargs):
                super().__init__(message)

        monkeypatch.setattr(project_imports, "ApplicationError", CaughtApplicationError)
        passthrough = "import-" + "c" * 24
        with pytest.raises(CaughtApplicationError, match="not found"):
            import_service.execute(passthrough, f"IMPORT {passthrough}")


def test_execute_stale_repository_and_manifest_parse_edges(tmp_path):
    repository = tmp_path / "git-repository"
    repository.mkdir()
    git(repository, "init", "-q")
    git(repository, "config", "user.email", "test@example.invalid")
    git(repository, "config", "user.name", "Test")
    (repository / "base.txt").write_text("base\n", encoding="utf-8")
    git(repository, "add", ".")
    git(repository, "commit", "-qm", "base")

    with TestClient(create_app(config(tmp_path, writes=True))) as client:
        import_service = service(client)
        plan = import_service.preview(repository)
        (repository / "changed.txt").write_text("changed\n", encoding="utf-8")
        with pytest.raises(ApplicationError, match="repository changed"):
            import_service.execute(plan["import_id"], plan["confirmation"])

    plain = tmp_path / "plain"
    plain.mkdir()
    with TestClient(create_app(config(tmp_path, writes=True))) as client:
        import_service = service(client)
        plan = import_service.preview(plain)
        manifest = plain / "experiments" / "research_project.yaml"
        manifest.parent.mkdir()
        manifest.write_text("[", encoding="utf-8")
        with pytest.raises(ApplicationError, match="manifest changed"):
            import_service.execute(plan["import_id"], plan["confirmation"])

    existing = tmp_path / "existing-stale"
    manifest = existing / "experiments" / "research_project.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(yaml.safe_dump({
        "schema_version": 1, "project": "stale", "title": "Stale",
        "run_roots": [],
    }), encoding="utf-8")
    with TestClient(create_app(config(tmp_path))) as client:
        import_service = service(client)
        plan = import_service.preview(existing)
        manifest.write_text(yaml.safe_dump({
            "schema_version": 1, "project": "stale", "title": "Changed",
            "run_roots": [],
        }), encoding="utf-8")
        with pytest.raises(ApplicationError, match="manifest changed"):
            import_service.execute(plan["import_id"], plan["confirmation"])
