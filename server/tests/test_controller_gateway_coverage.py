"""Adversarial branch coverage for immutable controller execution bundles."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_exp_server import controller_gateway as module
from ml_exp_server.controller_gateway import CommandRunner, ProjectControllerGateway
from ml_exp_server.schemas import (
    ControllerConfig,
    ControllerExecutionBundle,
    ResearchProject,
)


def _project(
    root: Path,
    *,
    python: str = sys.executable,
    paths: list[str] | None = None,
    packages: list[str] | None = None,
    clean_git: bool = False,
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
                paths=paths or [],
                python_packages=packages or [],
                require_clean_git=clean_git,
            ),
        ),
    )


def _minimal_snapshot(tmp_path: Path) -> dict[str, object]:
    root = tmp_path / "snapshot"
    root.mkdir(mode=0o700)
    payload = root / "payload.txt"
    payload.write_text("reviewed", encoding="utf-8")
    payload.chmod(0o600)
    python = module._file_identity(Path(sys.executable).resolve())
    manifest = {
        "schema_version": 1,
        "entry_module": "demo",
        "python": python,
        "snapshot_root": {
            "device": root.stat().st_dev,
            "inode": root.stat().st_ino,
        },
        "files": module._snapshot_files(root),
    }
    manifest_path = root / "manifest.json"
    encoded = json.dumps(manifest, sort_keys=True).encode()
    manifest_path.write_bytes(encoded)
    manifest_path.chmod(0o600)
    return {
        "root": str(root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": module._sha256_bytes(encoded),
        "manifest": manifest,
    }


def _install_trusted_core(gateway: ProjectControllerGateway, root: Path) -> Path:
    core = root / "trusted-core"
    core.mkdir(parents=True, exist_ok=True)
    (core / "__init__.py").write_text("", encoding="utf-8")
    identity = module._source_tree_identity(core)
    gateway._trusted_core_source = lambda: (core, identity)
    return core


@pytest.mark.parametrize("failure", ["timeout", "oserror"])
def test_pinned_runner_maps_process_failures(monkeypatch, tmp_path, failure):
    def fail(*_args, **_kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(["python"], 1)
        raise OSError("unavailable")

    monkeypatch.setattr(module.subprocess, "run", fail)
    result = CommandRunner().run_pinned(
        ["python"], cwd=tmp_path, timeout=1, env={}, pass_fds=(),
    )
    assert result["timeout"] is (failure == "timeout")
    assert result["returncode"] == (None if failure == "timeout" else 127)


def test_pinned_runner_tolerates_non_json_output(monkeypatch, tmp_path):
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="not-json", stderr="",
        ),
    )
    result = CommandRunner().run_pinned(
        ["python"], cwd=tmp_path, timeout=1, env={}, pass_fds=(),
    )
    assert result["payload"] is None

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="", stderr="",
        ),
    )
    assert CommandRunner().run_pinned(
        ["python"], cwd=tmp_path, timeout=1, env={}, pass_fds=(),
    )["payload"] is None


def test_ordinary_runner_maps_timeout_and_invalid_output(tmp_path):
    timeout = SimpleNamespace(
        run=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(["controller"], 1),
        ),
    )
    assert CommandRunner(timeout)(["controller"], cwd=tmp_path, timeout=1)[
        "timeout"
    ] is True
    invalid = SimpleNamespace(
        run=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1, stdout="not-json", stderr="failed",
        ),
    )
    result = CommandRunner(invalid)(["controller"], cwd=tmp_path, timeout=1)
    assert result["payload"] is None
    assert result["returncode"] == 1
    empty = SimpleNamespace(
        run=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="", stderr="",
        ),
    )
    assert CommandRunner(empty)(["controller"], cwd=tmp_path, timeout=1)[
        "payload"
    ] is None


def test_runner_and_call_builder_cover_transport_edges(tmp_path):
    core = SimpleNamespace(
        run=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout='{"api_token":"secret","url":"https://user:pass@example.test"}',
            stderr="",
        ),
    )
    result = CommandRunner(core)(["controller"], cwd=tmp_path, timeout=1)
    assert result["payload"]["api_token"] == "[REDACTED]"
    assert "[REDACTED]" in result["payload"]["url"]
    assert module.redact([{"safe": 1}, 2]) == [{"safe": 1}, 2]

    gateway = ProjectControllerGateway(runner=lambda *_args, **_kwargs: {"ok": True})
    project = _project(tmp_path)
    call = gateway.build(
        project,
        tmp_path / "campaign.yml",
        "collect",
        "run-a",
        attempt_id="a1",
        dry_run=True,
        extra=("--format", "json"),
    )
    assert call.argv[-5:] == ["--attempt-id", "a1", "--dry-run", "--format", "json"]
    assert gateway.execute(call, timeout=1) == {"ok": True}
    assert gateway.execute_command(["controller"], cwd=tmp_path, timeout=1) == {
        "ok": True,
    }

    no_controller = ResearchProject(project="demo", title="Demo", run_roots=[])
    with pytest.raises(ValueError, match="no controller"):
        gateway.build(no_controller, tmp_path / "campaign.yml", "collect", "run-a")

    absolute = ResearchProject(
        project="demo",
        title="Demo",
        run_roots=[],
        base_dir=tmp_path,
        controller=ControllerConfig(
            python=sys.executable,
            experimentctl=str(tmp_path / "experimentctl.py"),
            workdir=str(tmp_path),
        ),
    )
    direct = gateway.build(absolute, tmp_path / "campaign.yml", "collect", "run-a")
    assert direct.cwd == tmp_path
    assert "--attempt-id" not in direct.argv
    assert "--dry-run" not in direct.argv


def test_snapshot_file_helpers_reject_unsafe_inputs(monkeypatch, tmp_path):
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        module._open_regular_nofollow(directory)

    for relative in ("", "../escape", str(tmp_path / "absolute")):
        with pytest.raises(ValueError, match="must be relative"):
            module._safe_source(tmp_path, relative)

    target = tmp_path / "target"
    target.write_text("value", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="rejects symlink"):
        module._safe_source(tmp_path, "link")
    with pytest.raises(ValueError, match="rejects symlink"):
        module._copy_source(link, tmp_path / "copy")

    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="rejects special file"):
        module._copy_source(fifo, tmp_path / "fifo-copy")

    source = tmp_path / "source"
    source.write_text("content", encoding="utf-8")
    monkeypatch.setattr(module.os, "write", lambda *_args: 0)
    with pytest.raises(OSError, match="short write"):
        module._copy_regular(source, tmp_path / "short-write")


def test_seal_and_inventory_reject_mode_and_type_drift(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    regular = root / "regular"
    regular.write_text("value", encoding="utf-8")
    link = root / "link"
    link.symlink_to(regular)
    with pytest.raises(ValueError, match="contains a symlink"):
        module._seal_snapshot_modes(root)
    link.unlink()

    fifo = root / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="contains a special file"):
        module._seal_snapshot_modes(root)
    fifo.unlink()
    module._seal_snapshot_modes(root)

    subdir = root / "subdir"
    subdir.mkdir(mode=0o700)
    subdir.chmod(0o755)
    with pytest.raises(ValueError, match="directory mode drifted"):
        module._snapshot_files(root)
    subdir.chmod(0o700)
    regular.chmod(0o644)
    with pytest.raises(ValueError, match="file mode drifted"):
        module._snapshot_files(root)
    regular.chmod(0o600)
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="contains a special file"):
        module._snapshot_files(root)

    fifo.unlink()
    link.symlink_to(regular)
    with pytest.raises(ValueError, match="contains a symlink"):
        module._snapshot_files(root)


def test_source_identity_rejects_symlink_and_special_file(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    value = root / "value.py"
    value.write_text("pass\n", encoding="utf-8")
    link = root / "link.py"
    link.symlink_to(value)
    with pytest.raises(ValueError, match="contains a symlink"):
        module._source_tree_identity(root)
    link.unlink()
    fifo = root / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="contains a special file"):
        module._source_tree_identity(root)


def test_python_and_git_identity_validation(monkeypatch, tmp_path):
    gateway = ProjectControllerGateway()
    no_controller = ResearchProject(project="demo", title="Demo", run_roots=[])
    with pytest.raises(ValueError, match="no controller"):
        gateway._python_path(no_controller)

    monkeypatch.setattr(module.shutil, "which", lambda _value: None)
    with pytest.raises(ValueError, match="unavailable"):
        gateway._python_path(_project(tmp_path, python="missing-python"))
    with pytest.raises(ValueError, match="unavailable"):
        gateway._python_path(_project(tmp_path, python=str(tmp_path / "missing")))

    executable = Path(sys.executable).resolve()
    monkeypatch.setattr(module.shutil, "which", lambda _value: str(executable))
    assert gateway._python_path(_project(tmp_path, python="python")) == executable

    responses = iter([
        SimpleNamespace(returncode=0, stdout="", stderr=""),
        SimpleNamespace(returncode=1, stdout="", stderr=""),
    ])
    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs: next(responses))
    with pytest.raises(ValueError, match="no stable Git identity"):
        gateway._git_identity(tmp_path)

    responses = iter([
        SimpleNamespace(returncode=0, stdout="", stderr=""),
        SimpleNamespace(returncode=0, stdout="head\n", stderr=""),
        SimpleNamespace(returncode=0, stdout="tree\n", stderr=""),
    ])
    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs: next(responses))
    identity = gateway._git_identity(tmp_path)
    assert identity["head"] == "head"
    assert identity["tree"] == "tree"


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (SimpleNamespace(returncode=1, stdout=""), "unavailable"),
        (SimpleNamespace(returncode=0, stdout="not-json"), "lookup failed"),
        (SimpleNamespace(returncode=0, stdout="{}"), "no stable source root"),
    ],
)
def test_package_source_rejects_unstable_lookup(monkeypatch, tmp_path, result, message):
    gateway = ProjectControllerGateway()
    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs: result)
    with pytest.raises(ValueError, match=message):
        gateway._package_source(Path(sys.executable), "demo")
    with pytest.raises(ValueError, match="invalid"):
        gateway._package_source(Path(sys.executable), "../demo")


def test_package_source_accepts_package_and_module(monkeypatch, tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    source = tmp_path / "module.py"
    source.write_text("pass\n", encoding="utf-8")
    responses = iter([
        SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"locations": [str(package)], "origin": None}),
        ),
        SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"locations": [], "origin": str(source)}),
        ),
    ])
    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs: next(responses))
    gateway = ProjectControllerGateway()
    assert gateway._package_source(Path(sys.executable), "package") == package
    assert gateway._package_source(Path(sys.executable), "module") == source


def test_trusted_core_requires_one_source_and_allows_no_git_identity(monkeypatch, tmp_path):
    gateway = ProjectControllerGateway()
    monkeypatch.setattr(
        module.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(submodule_search_locations=[]),
    )
    with pytest.raises(ValueError, match="source root is unavailable"):
        gateway._trusted_core_source()

    package = tmp_path / "experiment_control"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        module.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(submodule_search_locations=[str(package)]),
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    source, identity = gateway._trusted_core_source()
    assert source == package
    assert identity["git_commit"] is None


def test_snapshot_rejects_contract_and_publish_conflicts(monkeypatch, tmp_path):
    gateway = ProjectControllerGateway()
    no_bundle = ResearchProject(
        project="demo",
        title="Demo",
        run_roots=[],
        base_dir=tmp_path,
        controller=ControllerConfig(
            python=sys.executable, experimentctl="unused.py", workdir=".",
        ),
    )
    with pytest.raises(ValueError, match="execution_bundle"):
        gateway.snapshot_execution_bundle(no_bundle, tmp_path / "out", reviewed_inputs={})

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(ValueError, match="incomplete"):
        gateway.snapshot_execution_bundle(_project(tmp_path), incomplete, reviewed_inputs={})

    with pytest.raises(ValueError, match="daemon-trusted"):
        gateway.snapshot_execution_bundle(
            _project(tmp_path, packages=["experiment_control"]),
            tmp_path / "reserved",
            reviewed_inputs={},
        )

    source = tmp_path / "reviewed"
    source.write_text("value", encoding="utf-8")
    with pytest.raises(ValueError, match="target must be relative"):
        gateway.snapshot_execution_bundle(
            _project(tmp_path),
            tmp_path / "unsafe",
            reviewed_inputs={"../reviewed": source},
        )

    trusted = tmp_path / "trusted-core"
    trusted.mkdir()
    (trusted / "__init__.py").write_text("", encoding="utf-8")
    trusted_identity = module._source_tree_identity(trusted)
    monkeypatch.setattr(
        gateway, "_trusted_core_source", lambda: (trusted, trusted_identity),
    )
    monkeypatch.setattr(
        module,
        "_source_tree_identity",
        lambda _source: {"files": [], "content_sha256": "changed"},
    )
    with pytest.raises(ValueError, match="changed while snapshotting"):
        gateway.snapshot_execution_bundle(
            _project(tmp_path), tmp_path / "drift", reviewed_inputs={},
        )


def test_existing_snapshot_is_verified_and_reused(tmp_path):
    snapshot = _minimal_snapshot(tmp_path)
    result = ProjectControllerGateway().snapshot_execution_bundle(
        _project(tmp_path), Path(str(snapshot["root"])), reviewed_inputs={},
    )
    assert result["manifest_sha256"] == snapshot["manifest_sha256"]

    gateway = ProjectControllerGateway()
    _install_trusted_core(gateway, tmp_path / "absolute-case")
    project = _project(tmp_path)
    assert project.controller is not None
    project.controller.workdir = str(tmp_path)
    created = gateway.snapshot_execution_bundle(
        project, tmp_path / "absolute-snapshot", reviewed_inputs={},
    )
    assert Path(str(created["root"])).is_dir()


def test_snapshot_copies_declared_package_shapes_and_rejects_duplicates(
    monkeypatch, tmp_path,
):
    package = tmp_path / "package-source"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    module_source = tmp_path / "module-source.py"
    module_source.write_text("VALUE = 1\n", encoding="utf-8")
    gateway = ProjectControllerGateway()
    core = _install_trusted_core(gateway, tmp_path)
    sources = {"package": package, "single.module": module_source}
    monkeypatch.setattr(gateway, "_package_source", lambda _python, name: sources[name])
    snapshot = gateway.snapshot_execution_bundle(
        _project(tmp_path, packages=list(sources)),
        tmp_path / "packages-snapshot",
        reviewed_inputs={},
    )
    root = Path(str(snapshot["root"]))
    assert (root / "src/package/__init__.py").is_file()
    assert (root / "src/single/module.py").is_file()

    reviewed = tmp_path / "reviewed"
    reviewed.write_text("duplicate", encoding="utf-8")
    gateway = ProjectControllerGateway()
    _install_trusted_core(gateway, tmp_path / "other")
    with pytest.raises(ValueError, match="duplicate controller snapshot target"):
        gateway.snapshot_execution_bundle(
            _project(tmp_path),
            tmp_path / "duplicate-snapshot",
            reviewed_inputs={"src/experiment_control/__init__.py": reviewed},
        )
    assert core.is_dir()


def test_snapshot_detects_git_python_and_manifest_write_races(monkeypatch, tmp_path):
    gateway = ProjectControllerGateway()
    _install_trusted_core(gateway, tmp_path)
    identities = iter([{"head": "one"}, {"head": "two"}])
    monkeypatch.setattr(gateway, "_git_identity", lambda _workdir: next(identities))
    with pytest.raises(ValueError, match="Git identity changed"):
        gateway.snapshot_execution_bundle(
            _project(tmp_path, clean_git=True), tmp_path / "git-race", reviewed_inputs={},
        )

    gateway = ProjectControllerGateway()
    _install_trusted_core(gateway, tmp_path / "python-case")
    original_identity = module._file_identity
    executable = Path(sys.executable).resolve()
    python_calls = 0

    def changing_identity(path):
        nonlocal python_calls
        identity = original_identity(path)
        if Path(path) == executable:
            python_calls += 1
            if python_calls == 2:
                identity = {**identity, "mtime_ns": 0}
        return identity

    monkeypatch.setattr(module, "_file_identity", changing_identity)
    with pytest.raises(ValueError, match="python changed"):
        gateway.snapshot_execution_bundle(
            _project(tmp_path), tmp_path / "python-race", reviewed_inputs={},
        )

    monkeypatch.setattr(module, "_file_identity", original_identity)
    gateway = ProjectControllerGateway()
    _install_trusted_core(gateway, tmp_path / "write-case")
    original_write = module.os.write

    def short_manifest_write(descriptor, value):
        target = os.readlink(f"/proc/self/fd/{descriptor}")
        if target.endswith("manifest.json"):
            return 0
        return original_write(descriptor, value)

    monkeypatch.setattr(module.os, "write", short_manifest_write)
    with pytest.raises(OSError, match="short write while sealing"):
        gateway.snapshot_execution_bundle(
            _project(tmp_path), tmp_path / "short-manifest", reviewed_inputs={},
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("root-mode", "root mode drifted"),
        ("manifest-link", "manifest is a symlink"),
        ("manifest-mode", "manifest mode drifted"),
        ("manifest-digest", "manifest digest drifted"),
        ("manifest-content", "manifest content drifted"),
        ("root-inode", "root inode drifted"),
        ("files", "Merkle inputs drifted"),
        ("python", "python identity drifted"),
    ],
)
def test_verify_snapshot_rejects_identity_drift(tmp_path, mutation, message):
    snapshot = _minimal_snapshot(tmp_path)
    root = Path(str(snapshot["root"]))
    manifest_path = Path(str(snapshot["manifest_path"]))
    if mutation == "root-mode":
        root.chmod(0o755)
    elif mutation == "manifest-link":
        original = root / "manifest-original"
        manifest_path.rename(original)
        manifest_path.symlink_to(original)
    elif mutation == "manifest-mode":
        manifest_path.chmod(0o644)
    elif mutation == "manifest-digest":
        snapshot["manifest_sha256"] = "sha256:wrong"
    elif mutation == "manifest-content":
        snapshot["manifest"] = {}
    elif mutation == "root-inode":
        snapshot["manifest"]["snapshot_root"] = {"device": 0, "inode": 0}
        encoded = json.dumps(snapshot["manifest"], sort_keys=True).encode()
        manifest_path.write_bytes(encoded)
        manifest_path.chmod(0o600)
        snapshot["manifest_sha256"] = module._sha256_bytes(encoded)
    elif mutation == "files":
        (root / "payload.txt").write_text("changed", encoding="utf-8")
    else:
        snapshot["manifest"]["python"]["mtime_ns"] = 0
        encoded = json.dumps(snapshot["manifest"], sort_keys=True).encode()
        manifest_path.write_bytes(encoded)
        manifest_path.chmod(0o600)
        snapshot["manifest_sha256"] = module._sha256_bytes(encoded)
    with pytest.raises(ValueError, match=message):
        ProjectControllerGateway().verify_execution_bundle(snapshot)


def test_verify_snapshot_rejects_invalid_root(tmp_path):
    with pytest.raises(ValueError, match="root identity is invalid"):
        ProjectControllerGateway().verify_execution_bundle({
            "root": str(tmp_path / "missing"),
            "manifest_path": str(tmp_path / "manifest.json"),
        })


def test_execute_snapshot_detects_inode_drift_and_supports_plain_runner(tmp_path):
    snapshot = _minimal_snapshot(tmp_path)
    gateway = ProjectControllerGateway(
        runner=lambda command, **_kwargs: {"returncode": 0, "command": command},
    )
    result = gateway.execute_snapshot(snapshot, ["value"], timeout=1)
    assert result["returncode"] == 0
    assert result["controller_snapshot_sha256"] == snapshot["manifest_sha256"]

    gateway.verify_execution_bundle = lambda _snapshot: None
    snapshot["manifest"]["python"]["inode"] = 0
    with pytest.raises(ValueError, match="python inode drifted"):
        gateway.execute_snapshot(snapshot, [], timeout=1)
