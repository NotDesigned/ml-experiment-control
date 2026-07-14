from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from ml_exp_server.api.app import create_app
from ml_exp_server.schemas import ActionRuntimeConfig, ProjectRef, ServerConfig


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=True,
        capture_output=True, text=True,
    ).stdout


def repository(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repository"
    manifest = root / "experiments" / "research_project.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(yaml.safe_dump({
        "schema_version": 1, "project": "demo", "title": "Demo",
        "run_roots": [],
    }), encoding="utf-8")
    (root / "train.py").write_text("value = 1\n", encoding="utf-8")
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "Test")
    git(root, "add", ".")
    git(root, "commit", "-qm", "base")
    return root, git(root, "rev-parse", "HEAD").strip()


def config(tmp_path: Path, manifest: Path, *, imports: bool = False) -> ServerConfig:
    return ServerConfig(
        index_db=str(tmp_path / "state" / "index.sqlite"),
        action_root=str(tmp_path / "state" / "actions"),
        project_registry_root=str(tmp_path / "state" / "projects"),
        projects=[ProjectRef(project_file=str(manifest))],
        action_runtime=ActionRuntimeConfig(allow_source_imports=imports),
        collector_enabled=False,
    )


def proposal(base: str, *, path: str = "train.py") -> dict:
    patch = (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1 +1 @@\n"
        "-value = 1\n"
        "+value = 2\n"
    )
    return {
        "base_commit": base,
        "patch": patch,
        "patch_digest": "sha256:" + hashlib.sha256(patch.encode()).hexdigest(),
        "changed_files": [path],
    }


def test_source_revision_preview_is_read_only_and_execute_is_policy_gated(tmp_path):
    root, base = repository(tmp_path)
    with TestClient(create_app(config(
        tmp_path, root / "experiments" / "research_project.yaml",
    ))) as client:
        response = client.post("/api/source-revisions/preview", json={
            "project": "demo", "proposal": proposal(base),
        })
        plan = response.json()
        blocked = client.post(
            f"/api/source-revisions/{plan['import_id']}/execute",
            json={"confirmation": plan["confirmation"]},
        )

    assert response.status_code == 200
    assert plan["source_id"].startswith("source.")
    assert plan["confirmation"] == f"IMPORT SOURCE {plan['source_id']}"
    assert blocked.status_code == 409
    assert "disabled by daemon policy" in blocked.json()["detail"]
    assert (root / "train.py").read_text(encoding="utf-8") == "value = 1\n"


def test_source_revision_execute_materializes_idempotent_read_only_tree(tmp_path):
    root, base = repository(tmp_path)
    with TestClient(create_app(config(
        tmp_path, root / "experiments" / "research_project.yaml", imports=True,
    ))) as client:
        plan = client.post("/api/source-revisions/preview", json={
            "project": "demo", "proposal": proposal(base),
        }).json()
        wrong = client.post(
            f"/api/source-revisions/{plan['import_id']}/execute",
            json={"confirmation": "IMPORT SOURCE wrong"},
        )
        first = client.post(
            f"/api/source-revisions/{plan['import_id']}/execute",
            json={"confirmation": plan["confirmation"]},
        )
        second = client.post(
            f"/api/source-revisions/{plan['import_id']}/execute",
            json={"confirmation": plan["confirmation"]},
        )
        fetched = client.get(
            f"/api/projects/demo/source-revisions/{plan['source_id']}"
        )

    assert wrong.status_code == 409
    assert first.status_code == second.status_code == fetched.status_code == 200
    source = first.json()["source"]
    tree = Path(source["tree_path"])
    assert (tree / "train.py").read_text(encoding="utf-8") == "value = 2\n"
    assert not (tree / ".git").exists()
    assert tree.stat().st_mode & 0o222 == 0
    assert (tree / "train.py").stat().st_mode & 0o222 == 0
    assert first.json()["source"] == second.json()["source"]
    assert fetched.json()["patch_digest"] == plan["patch_digest"]
    assert (root / "train.py").read_text(encoding="utf-8") == "value = 1\n"


def test_source_revision_rejects_forged_metadata_protected_paths_and_unknown_base(
    tmp_path,
):
    root, base = repository(tmp_path)
    configured = config(
        tmp_path, root / "experiments" / "research_project.yaml", imports=True,
    )
    with TestClient(create_app(configured)) as client:
        forged = proposal(base)
        forged["changed_files"] = ["other.py"]
        mismatch = client.post("/api/source-revisions/preview", json={
            "project": "demo", "proposal": forged,
        })
        protected = proposal(base, path=".env")
        protected["patch"] = (
            "diff --git a/.env b/.env\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/.env\n"
            "@@ -0,0 +1 @@\n"
            "+SECRET=nope\n"
        )
        protected["patch_digest"] = "sha256:" + hashlib.sha256(
            protected["patch"].encode()
        ).hexdigest()
        secret = client.post("/api/source-revisions/preview", json={
            "project": "demo", "proposal": protected,
        })
        missing = proposal("0" * 40)
        unknown = client.post("/api/source-revisions/preview", json={
            "project": "demo", "proposal": missing,
        })

    assert mismatch.status_code == 409
    assert "changed_files" in mismatch.json()["detail"]
    assert secret.status_code == 409
    assert "protected paths" in secret.json()["detail"]
    assert unknown.status_code == 409
    assert "base commit" in unknown.json()["detail"]


def test_source_revision_health_advertises_capability_and_closed_gate(tmp_path):
    root, _ = repository(tmp_path)
    with TestClient(create_app(config(
        tmp_path, root / "experiments" / "research_project.yaml",
    ))) as client:
        health = client.get("/api/health").json()

    assert "source-revision-import.v1" in health["capabilities"]
    assert health["source_imports"] is False
