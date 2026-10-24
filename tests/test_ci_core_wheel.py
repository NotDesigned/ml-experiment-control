"""A cached native wheel cannot replace different core sources or build inputs."""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import sysconfig
import tarfile
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import ci_core_wheel as core


@pytest.fixture
def source(tmp_path, monkeypatch):
    contents = {"pyproject.toml": '[project]\nname="ml-experiment-control"\nversion="1.2.3"\n',
                "README.md": "core", "LICENSE": "MIT", "rust/Cargo.toml": "manifest",
                "rust/Cargo.lock": "locked", "rust/src/main.rs": "fn main() {}",
                "src/experiment_control/__init__.py": 'VERSION="1.2.3"\n',
                "src/experiment_control/backend.py": 'def state(): return "UNKNOWN"\n'}
    for name, value in contents.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    monkeypatch.setenv("CI_RUST_VERSION", "1.85.0")
    return tmp_path


def wheel(root, folder, *, extra=False, changed=False, version="1.2.3"):
    path = folder / f"ml_experiment_control-1.2.3-py3-none-{sysconfig.get_platform().replace('-', '_')}.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ml_experiment_control-1.2.3.dist-info/METADATA",
                         f"Metadata-Version: 2.1\nName: ml-experiment-control\nVersion: {version}\n")
        for source in (root / "src/experiment_control").rglob("*.py"):
            archive.writestr("experiment_control/" + source.name,
                             b"changed" if changed else source.read_bytes())
        if extra:
            archive.writestr("extra.py", "raise RuntimeError('never execute cached artifact code')")
        archive.writestr("ml_experiment_control-1.2.3.data/scripts/experiment-redact", b"native fixture")
    return path


def test_first_source_build_is_verified_and_matching_cache_avoids_rebuild(source, monkeypatch):
    calls = []
    def build(argv, **kwargs):
        calls.append(argv)
        assert kwargs["check"] and argv[:4] == ["uv", "build", "--package", "ml-experiment-control"]
        wheel(source, Path(argv[-1]))
    monkeypatch.setattr(core.subprocess, "run", build)
    first = core.prepare(source)
    assert len(calls) == 1 and first.is_file()
    assert core.prepare(source) == first and len(calls) == 1
    # A changed Rust source must cause a new build, even when Python matches.
    (source / "rust/src/main.rs").write_text("fn main() { println!(\"new\"); }")
    assert core.prepare(source) == first and len(calls) == 2
    monkeypatch.setenv("CI_RUST_VERSION", "1.99.0")
    assert core.prepare(source) == first and len(calls) == 3


@pytest.mark.parametrize("change", ["python", "extra", "version", "bytes", "manifest"])
def test_untrusted_cache_is_never_installed_and_requires_current_source_build(source, monkeypatch, change):
    cache = source / ".ops/core-wheel"
    cache.mkdir(parents=True)
    path = wheel(source, cache)
    manifest = {**core.identity(source), "wheel_sha256": core.wheel_source(source, path)}
    (cache / "manifest.json").write_text(json.dumps(manifest))
    if change in {"python", "extra", "version"}:
        wheel(source, cache, changed=change == "python", extra=change == "extra",
              version="9.9.9" if change == "version" else "1.2.3")
    elif change == "bytes":
        with path.open("ab") as stream:
            stream.write(b"changed native wheel")
    else:
        (cache / "manifest.json").write_text("{}")
    def failed(*args, **kwargs):
        raise RuntimeError("current source build was requested; no install")
    monkeypatch.setattr(core.subprocess, "run", failed)
    with pytest.raises(RuntimeError, match="current source build"):
        core.prepare(source)


def test_every_cached_python_module_must_match_the_current_checkout(source):
    path = wheel(source, source)
    assert len(core.wheel_source(source, path)) == 64
    (source / "src/experiment_control/backend.py").write_text("def state(): return 'RUNNING'\n")
    with pytest.raises(ValueError, match="sources or package metadata"):
        core.wheel_source(source, path)


def test_seed_never_installs_and_normal_install_preserves_plain_source_path(source, monkeypatch):
    package = source / 'verified.whl'
    monkeypatch.setattr(core, 'ROOT', source)
    monkeypatch.setattr(core, 'prepare', lambda: package)
    site = source / 'site-packages'
    site.mkdir()
    monkeypatch.setattr(core.sysconfig, 'get_path', lambda name: str(site))
    commands = []
    monkeypatch.setattr(core.subprocess, 'run', lambda command, **kwargs: commands.append(command))
    core.main(['--cache-only'])
    assert not commands and not list(site.iterdir())
    core.main([])
    assert commands[0][:5] == ['uv', 'pip', 'install', '--offline', '--no-deps']
    assert (site / '_ml_exp_core_checkout.pth').read_text() == str(source / 'src') + '\n'


def sdist(root, output, *, change=None):
    path = output / "ml_experiment_control-1.2.3.tar.gz"
    files = ["pyproject.toml", "README.md", "rust/Cargo.toml", "rust/Cargo.lock",
             "rust/src/main.rs", "src/experiment_control/__init__.py",
             "src/experiment_control/backend.py"]
    with tarfile.open(path, "w:gz") as archive:
        for name in files:
            if change == "missing" and name.endswith("backend.py"):
                continue
            data = (root / name).read_bytes()
            if name == change:
                data = b"changed source"
            member = tarfile.TarInfo("ml_experiment_control-1.2.3/" + name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        if change == "extra":
            member = tarfile.TarInfo("ml_experiment_control-1.2.3/src/extra.py")
            archive.addfile(member, io.BytesIO())
        if change == "symlink":
            member = tarfile.TarInfo("ml_experiment_control-1.2.3/link")
            member.type = tarfile.SYMTYPE
            member.linkname = "/tmp/untrusted"
            archive.addfile(member)
    return path


@pytest.mark.parametrize("change", [None, "missing", "src/experiment_control/backend.py",
                                  "rust/src/main.rs", "rust/Cargo.lock", "pyproject.toml",
                                  "README.md", "extra", "symlink"])
def test_new_source_archive_must_preserve_all_current_core_and_rust_sources(source, change):
    archive = sdist(source, source, change=change)
    if change is None:
        core.source_distribution(source, archive)
    else:
        with pytest.raises(ValueError, match="source archive"):
            core.source_distribution(source, archive)


def test_package_output_only_copies_verified_wheel_after_validating_current_sdist(source, monkeypatch):
    package = wheel(source, source)
    output = source / "dist"
    calls = []
    def build(command, **kwargs):
        calls.append(command)
        assert command == ["uv", "build", "--package", "ml-experiment-control", "--sdist",
                           "--out-dir", str(output)]
        sdist(source, output)
    monkeypatch.setattr(core.subprocess, "run", build)
    core.package_output(source, package, output)
    assert (output / package.name).read_bytes() == package.read_bytes()
    assert len(calls) == 1
    (output / package.name).unlink()
    monkeypatch.setattr(core.subprocess, "run", lambda *args, **kwargs: sdist(source, output, change="missing"))
    with pytest.raises(ValueError, match="source archive"):
        core.package_output(source, package, output)
    assert not (output / package.name).exists()


def test_packaging_requires_cache_only_and_never_installs(source, monkeypatch):
    monkeypatch.setattr(core, "ROOT", source)
    package = wheel(source, source)
    monkeypatch.setattr(core, "prepare", lambda: package)
    outputs = []
    monkeypatch.setattr(core, "package_output", lambda *args: outputs.append(args))
    monkeypatch.setattr(core.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not install"))
    with pytest.raises(SystemExit):
        core.main(["--package-output", str(source / "dist")])
    core.main(["--cache-only", "--package-output", str(source / "dist")])
    assert outputs == [(source, package, source / "dist")]


def test_unrelated_workspace_dependency_lock_does_not_change_native_core_build_identity(source):
    (source / "uv.lock").write_text("client and server version 1")
    before = core.identity(source)
    (source / "uv.lock").write_text("client and server version 2")
    assert core.identity(source) == before
    # The build system and native source remain mandatory, independent inputs.
    (source / "pyproject.toml").write_text((source / "pyproject.toml").read_text() + "# build recipe changed\n")
    assert core.identity(source) != before


def test_native_probe_rules_are_an_optional_but_strict_core_build_input(source):
    before = core.identity(source)
    probe = source / "tools/ci_native.py"
    probe.parent.mkdir()
    probe.write_text("selected compiler probe version 1")
    added = core.identity(source)
    assert before != added
    probe.write_text("selected compiler probe version 2")
    assert core.identity(source) != added
