"""Focused defensive coverage for immutable source revision imports."""

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ml_exp_server.api.app import create_app
from ml_exp_server.application_errors import ApplicationError
from ml_exp_server import source_revisions
from tests.test_source_revisions import config, proposal, repository


def service(client: TestClient):
    return client.app.state.application.source_revision_service


@pytest.mark.parametrize("payload,message", [
    ({"base_commit": "bad", "patch": "x", "patch_digest": "x",
      "changed_files": []}, "invalid base commit"),
    ({"base_commit": "a" * 40, "patch": 1, "patch_digest": "x",
      "changed_files": []}, "patch must be text"),
    ({"base_commit": "a" * 40, "patch": "", "patch_digest": "x",
      "changed_files": []}, "must contain"),
    ({"base_commit": "a" * 40, "patch": "x", "patch_digest": "wrong",
      "changed_files": []}, "digest mismatch"),
    ({"base_commit": "a" * 40, "patch": "x",
      "patch_digest": "sha256:" + hashlib.sha256(b"x").hexdigest(),
      "changed_files": "file.py"}, "must be a list"),
])
def test_proposal_validation_edges(payload, message):
    with pytest.raises(ApplicationError, match=message):
        source_revisions.SourceRevisionService._proposal(payload)


def test_repository_unknown_non_git_and_nested_root_edges(tmp_path):
    root, _base = repository(tmp_path)
    with TestClient(create_app(config(
        tmp_path, root / "experiments" / "research_project.yaml",
    ))) as client:
        revision_service = service(client)
        with pytest.raises(ApplicationError, match="unknown project"):
            revision_service._repository("missing")

    non_git = tmp_path / "non-git"
    manifest = non_git / "experiments" / "research_project.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "schema_version: 1\nproject: plain\ntitle: Plain\nrun_roots: []\n",
        encoding="utf-8",
    )
    with TestClient(create_app(config(tmp_path / "plain-state", manifest))) as client:
        with pytest.raises(ApplicationError, match="Git-backed"):
            service(client)._repository("plain")

    nested_manifest = root / "nested" / "experiments" / "research_project.yaml"
    nested_manifest.parent.mkdir(parents=True)
    nested_manifest.write_text(
        "schema_version: 1\nproject: nested\ntitle: Nested\nrun_roots: []\n",
        encoding="utf-8",
    )
    with TestClient(create_app(config(
        tmp_path / "nested-state", nested_manifest,
    ))) as client:
        with pytest.raises(ApplicationError, match="repository root"):
            service(client)._repository("nested")


def _fake_clone(target: Path):
    def run(command, **kwargs):
        target.mkdir(parents=True, exist_ok=True)
        return type("Completed", (), {"stdout": b""})()
    return run


def test_materialize_no_change_gitlink_and_symlink_edges(tmp_path, monkeypatch):
    target = tmp_path / "target"
    monkeypatch.setattr(source_revisions.subprocess, "run", _fake_clone(target))
    monkeypatch.setattr(source_revisions, "_git", lambda *args, **kwargs: b"")
    with pytest.raises(ApplicationError, match="no source change"):
        source_revisions.SourceRevisionService._materialize(
            tmp_path, "a" * 40, b"patch", target,
        )

    target = tmp_path / "gitlink"
    calls = iter([b"", b"", b"", b"", b"file.py\0", b":100644 120000 x\0"])
    monkeypatch.setattr(source_revisions.subprocess, "run", _fake_clone(target))
    monkeypatch.setattr(
        source_revisions, "_git", lambda *args, **kwargs: next(calls),
    )
    with pytest.raises(ApplicationError, match="symlinks or Git links"):
        source_revisions.SourceRevisionService._materialize(
            tmp_path, "a" * 40, b"patch", target,
        )

    target = tmp_path / "symlink"
    outside = tmp_path / "outside"
    outside.write_text("outside\n", encoding="utf-8")

    def clone_with_symlink(command, **kwargs):
        target.mkdir(parents=True, exist_ok=True)
        (target / "file.py").symlink_to(outside)
        return type("Completed", (), {"stdout": b""})()

    calls = iter([b"", b"", b"", b"", b"file.py\0", b""])
    monkeypatch.setattr(source_revisions.subprocess, "run", clone_with_symlink)
    monkeypatch.setattr(
        source_revisions, "_git", lambda *args, **kwargs: next(calls),
    )
    with pytest.raises(ApplicationError, match="modify symlinks"):
        source_revisions.SourceRevisionService._materialize(
            tmp_path, "a" * 40, b"patch", target,
        )


def test_preview_binding_and_materialization_error_edges(tmp_path, monkeypatch):
    root, base = repository(tmp_path)
    with TestClient(create_app(config(
        tmp_path, root / "experiments" / "research_project.yaml", imports=True,
    ))) as client:
        revision_service = service(client)
        bound = proposal(base)
        bound["binding"] = {"project": "other"}
        with pytest.raises(ApplicationError, match="different Project"):
            revision_service.preview("demo", bound)

        monkeypatch.setattr(
            revision_service, "_materialize",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("apply failed")),
        )
        with pytest.raises(ApplicationError, match="does not apply cleanly"):
            revision_service.preview("demo", proposal(base))


def test_locked_plan_invalid_missing_corrupt_and_passthrough(tmp_path, monkeypatch):
    root, _base = repository(tmp_path)
    with TestClient(create_app(config(
        tmp_path, root / "experiments" / "research_project.yaml", imports=True,
    ))) as client:
        revision_service = service(client)
        with pytest.raises(ApplicationError, match="invalid source import identity"):
            revision_service.execute("bad", "bad")
        missing = "source-import-" + "a" * 24
        with pytest.raises(ApplicationError, match="not found"):
            revision_service.execute(missing, "bad")

        revision_service.plans.mkdir(parents=True, exist_ok=True)
        corrupt = "source-import-" + "b" * 24
        path = revision_service.plans / f"{corrupt}.json"
        path.write_text("[]\n", encoding="utf-8")
        with pytest.raises(ApplicationError, match="unreadable"):
            revision_service.execute(corrupt, "bad")
        path.write_text("{", encoding="utf-8")
        with pytest.raises(ApplicationError, match="unreadable"):
            revision_service.execute(corrupt, "bad")

        class CaughtApplicationError(ValueError):
            def __init__(self, message, **kwargs):
                super().__init__(message)

        monkeypatch.setattr(source_revisions, "ApplicationError", CaughtApplicationError)
        passthrough = "source-import-" + "c" * 24
        with pytest.raises(CaughtApplicationError, match="not found"):
            revision_service.execute(passthrough, "bad")


def test_execute_tampered_identity_collision_and_changed_materialization(
    tmp_path, monkeypatch,
):
    root, base = repository(tmp_path)
    with TestClient(create_app(config(
        tmp_path, root / "experiments" / "research_project.yaml", imports=True,
    ))) as client:
        revision_service = service(client)
        plan = revision_service.preview("demo", proposal(base))
        plan_path = revision_service.plans / f"{plan['import_id']}.json"
        stored = json.loads(plan_path.read_text())
        stored["source_id"] = "bad"
        plan_path.write_text(json.dumps(stored), encoding="utf-8")
        with pytest.raises(ApplicationError, match="planned source identity"):
            revision_service.execute(plan["import_id"], plan["confirmation"])

        plan = revision_service.preview("demo", proposal(base))
        monkeypatch.setattr(
            revision_service, "_proposal",
            lambda value: (base, b"patch", "sha256:" + "f" * 64, ["train.py"]),
        )
        with pytest.raises(ApplicationError, match="plan changed"):
            revision_service.execute(plan["import_id"], plan["confirmation"])
        monkeypatch.undo()

        plan = revision_service.preview("demo", proposal(base))
        first = revision_service.execute(plan["import_id"], plan["confirmation"])
        metadata = Path(first["source"]["metadata_path"])
        payload = json.loads(metadata.read_text())
        payload["patch_digest"] = "sha256:" + "0" * 64
        metadata.chmod(0o644)
        metadata.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ApplicationError, match="collision"):
            revision_service.execute(plan["import_id"], plan["confirmation"])

        other = proposal(base)
        other["patch"] = other["patch"].replace("+value = 2", "+value = 3")
        other["patch_digest"] = "sha256:" + hashlib.sha256(
            other["patch"].encode()
        ).hexdigest()
        plan = revision_service.preview("demo", other)
        monkeypatch.setattr(
            revision_service, "_materialize", lambda *args, **kwargs: ["other.py"],
        )
        with pytest.raises(ApplicationError, match="changed after preview"):
            revision_service.execute(plan["import_id"], plan["confirmation"])


def test_execute_cleanup_branches_and_get_errors(tmp_path, monkeypatch):
    root, base = repository(tmp_path)
    with TestClient(create_app(config(
        tmp_path, root / "experiments" / "research_project.yaml", imports=True,
    ))) as client:
        revision_service = service(client)
        plan = revision_service.preview("demo", proposal(base))

        def application_failure(*args, **kwargs):
            raise ApplicationError("materialize rejected", code="SOURCE_IMPORT_BLOCKED")

        monkeypatch.setattr(revision_service, "_materialize", application_failure)
        with pytest.raises(ApplicationError, match="materialize rejected"):
            revision_service.execute(plan["import_id"], plan["confirmation"])

        monkeypatch.setattr(
            revision_service, "_materialize",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk failed")),
        )
        with pytest.raises(ApplicationError, match="failed to materialize"):
            revision_service.execute(plan["import_id"], plan["confirmation"])

        with pytest.raises(ApplicationError, match="invalid source identity"):
            revision_service.get("bad/id", "bad")
        with pytest.raises(ApplicationError, match="not found"):
            revision_service.get("demo", "source." + "a" * 64)


def test_execute_chmod_skips_materialized_symlink(tmp_path, monkeypatch):
    root, base = repository(tmp_path)
    with TestClient(create_app(config(
        tmp_path, root / "experiments" / "research_project.yaml", imports=True,
    ))) as client:
        revision_service = service(client)
        plan = revision_service.preview("demo", proposal(base))

        def materialize(_repository, _base, _patch, tree):
            tree.mkdir(parents=True)
            (tree / ".git").mkdir()
            (tree / "train.py").write_text("value = 2\n", encoding="utf-8")
            (tree / "link.py").symlink_to("train.py")
            return ["train.py"]

        monkeypatch.setattr(revision_service, "_materialize", materialize)
        result = revision_service.execute(plan["import_id"], plan["confirmation"])

    assert Path(result["source"]["tree_path"]).joinpath("link.py").is_symlink()
