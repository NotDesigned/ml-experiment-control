from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from ml_exp_server.controller_gateway import ProjectControllerGateway
from ml_exp_server.schemas import (
    ControllerConfig,
    ControllerExecutionBundle,
    ResearchProject,
)


def _project(
    root: Path, *, require_clean_git: bool = False,
    python: str = sys.executable,
) -> ResearchProject:
    return ResearchProject(
        project="demo",
        title="Demo",
        run_roots=[],
        base_dir=root,
        controller=ControllerConfig(
            python=python,
            experimentctl="unused.py",
            workdir=".",
            execution_bundle=ControllerExecutionBundle(
                entry_module="demo_controller",
                paths=["src/demo_controller"],
                require_clean_git=require_clean_git,
            ),
        ),
    )


def _source_tree(root: Path) -> Path:
    package = root / "src" / "demo_controller"
    package.mkdir(parents=True)
    (package / "__main__.py").write_text(
        """\
import hashlib
import experiment_control
import json
import os
import sys
from pathlib import Path

value = Path(sys.argv[1]).read_text(encoding="utf-8")
print(json.dumps({
    "value": value,
    "cwd": str(Path.cwd()),
    "snapshot": os.environ["ML_EXPD_CONTROLLER_SNAPSHOT_SHA256"],
    "experiment_control": experiment_control.__file__,
}))
""",
        encoding="utf-8",
    )
    reviewed = root / "reviewed.txt"
    reviewed.write_text("approved", encoding="utf-8")
    return reviewed


def test_snapshot_executes_private_code_and_reviewed_inputs(tmp_path: Path) -> None:
    root = tmp_path / "science"
    reviewed = _source_tree(root)
    gateway = ProjectControllerGateway()
    destination = tmp_path / "actions" / "controller-input"

    snapshot = gateway.snapshot_execution_bundle(
        _project(root), destination,
        reviewed_inputs={"inputs/value.txt": reviewed},
    )
    reviewed.write_text("mutated original", encoding="utf-8")
    (root / "src" / "demo_controller" / "__main__.py").write_text(
        "raise SystemExit('mutated original code ran')\n", encoding="utf-8",
    )

    result = gateway.execute_snapshot(
        snapshot, ["{controller_snapshot}/inputs/value.txt"], timeout=30,
    )

    assert result["returncode"] == 0, result
    assert result["payload"]["value"] == "approved"
    assert result["payload"]["snapshot"] == snapshot["manifest_sha256"]
    assert "/proc/self/fd/" in result["payload"]["experiment_control"]
    assert result["controller_snapshot_sha256"] == snapshot["manifest_sha256"]
    assert result["payload"]["cwd"] == str(destination)
    assert stat.S_IMODE(destination.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600
        for path in destination.rglob("*") if path.is_file()
    )


def test_snapshot_rejects_source_symlinks_and_cleans_partial_publish(
    tmp_path: Path,
) -> None:
    root = tmp_path / "science"
    reviewed = _source_tree(root)
    target = root / "target.py"
    target.write_text("pass\n", encoding="utf-8")
    (root / "src" / "demo_controller" / "linked.py").symlink_to(target)
    destination = tmp_path / "actions" / "controller-input"

    with pytest.raises(ValueError, match="rejects symlink"):
        ProjectControllerGateway().snapshot_execution_bundle(
            _project(root), destination,
            reviewed_inputs={"inputs/value.txt": reviewed},
        )

    assert not destination.exists()
    assert not list(destination.parent.glob(".controller-input.*.tmp"))


def test_snapshot_rejects_private_file_and_root_replacement(tmp_path: Path) -> None:
    root = tmp_path / "science"
    reviewed = _source_tree(root)
    gateway = ProjectControllerGateway()
    destination = tmp_path / "actions" / "controller-input"
    snapshot = gateway.snapshot_execution_bundle(
        _project(root), destination,
        reviewed_inputs={"inputs/value.txt": reviewed},
    )

    private_source = destination / "src" / "demo_controller" / "__main__.py"
    private_source.write_text("raise SystemExit(9)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Merkle inputs drifted"):
        gateway.execute_snapshot(snapshot, [], timeout=30)

    # Rebuild, then replace the sealed directory with byte-identical content.
    destination = tmp_path / "actions" / "controller-input-2"
    snapshot = gateway.snapshot_execution_bundle(
        _project(root), destination,
        reviewed_inputs={"inputs/value.txt": reviewed},
    )
    replacement = destination.with_name("replacement")
    subprocess.run(["cp", "-a", str(destination), str(replacement)], check=True)
    os.replace(destination, destination.with_name("old-snapshot"))
    os.replace(replacement, destination)
    with pytest.raises(ValueError, match="root inode drifted"):
        gateway.execute_snapshot(snapshot, [], timeout=30)


def test_snapshot_requires_git_identity_to_stay_clean(tmp_path: Path) -> None:
    root = tmp_path / "science"
    reviewed = _source_tree(root)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git", "-C", str(root), "-c", "user.name=Test", "-c",
            "user.email=test@example.invalid", "commit", "-qm", "initial",
        ],
        check=True,
    )
    (root / "untracked.txt").write_text("dirty", encoding="utf-8")

    with pytest.raises(ValueError, match="clean Git tree"):
        ProjectControllerGateway().snapshot_execution_bundle(
            _project(root, require_clean_git=True),
            tmp_path / "actions" / "controller-input",
            reviewed_inputs={"inputs/value.txt": reviewed},
        )


def test_execution_bundle_schema_rejects_undeclared_fields() -> None:
    with pytest.raises(ValueError):
        ControllerExecutionBundle.model_validate({
            "entry_module": "demo", "paths": [], "backend_collect": True,
        })


def test_snapshot_supplies_daemon_core_to_python_without_it_installed(
    tmp_path: Path,
) -> None:
    python = "/usr/bin/python3"
    unavailable = subprocess.run(
        [python, "-I", "-c", "import experiment_control"], check=False,
        text=True, capture_output=True,
    )
    assert unavailable.returncode != 0
    root = tmp_path / "science"
    reviewed = _source_tree(root)
    gateway = ProjectControllerGateway()
    snapshot = gateway.snapshot_execution_bundle(
        _project(root, python=python),
        tmp_path / "actions" / "controller-input",
        reviewed_inputs={"inputs/value.txt": reviewed},
    )

    result = gateway.execute_snapshot(
        snapshot, ["{controller_snapshot}/inputs/value.txt"], timeout=30,
    )

    assert result["returncode"] == 0, result
    assert result["payload"]["value"] == "approved"
    assert "/proc/self/fd/" in result["payload"]["experiment_control"]
    assert snapshot["manifest"]["trusted_core"]["content_sha256"].startswith(
        "sha256:"
    )
