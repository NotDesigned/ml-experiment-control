"""A cached native wheel cannot replace different core sources or build inputs."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import sysconfig
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
