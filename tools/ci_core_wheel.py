"""Install a CI core wheel only after matching every Python source to checkout.

The cache key includes compiler/platform and all core build inputs. The package
job still builds all source distributions and wheels on every full CI run.
"""
from __future__ import annotations

from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def identity(root):
    paths = [root / name for name in ("pyproject.toml", "README.md", "LICENSE", "rust/Cargo.toml", "rust/Cargo.lock")]
    paths += [root / name for name in ("uv.lock", ".github/workflows/quality.yml", ".cargo/config.toml",
                                      "rust/.cargo/config.toml", "rust-toolchain.toml") if (root / name).is_file()]
    paths += [path for folder in ("src", "rust/src") for path in (root / folder).rglob("*")
              if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"]
    sources = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
    sources["ci_core_wheel.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {"source_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
            "rustc": os.environ["CI_RUST_VERSION"], "platform": sysconfig.get_platform(),
            "python_abi": sys.implementation.cache_tag, "libc": list(platform.libc_ver())}


def wheel_source(root, wheel):
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    expected_name = f"ml_experiment_control-{version}-py3-none-{sysconfig.get_platform().replace('-', '_')}.whl"
    if wheel.name != expected_name or wheel.is_symlink() or wheel.stat().st_size > 16 * 1024**2:
        raise ValueError("core wheel identity or size differs")
    expected = {"experiment_control/" + path.relative_to(root / "src/experiment_control").as_posix(): path.read_bytes()
                for path in (root / "src/experiment_control").rglob("*.py")}
    with zipfile.ZipFile(wheel) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("core wheel contains duplicate entries")
        if sum(row.file_size for row in archive.infolist()) > 32 * 1024**2:
            raise ValueError("core wheel expanded size differs")
        metadata, = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        package = BytesParser().parsebytes(archive.read(metadata))
        actual = {name: archive.read(name) for name in archive.namelist() if name.endswith(".py")}
    if (package.get_all("Name") != ["ml-experiment-control"] or package.get_all("Version") != [version]
            or not expected or actual != expected):
        raise ValueError("core wheel Python sources or package metadata differ from checkout")
    return hashlib.sha256(wheel.read_bytes()).hexdigest()


def prepare(root=ROOT):
    cache = root / ".ops/core-wheel"
    expected = identity(root)
    try:
        manifest = cache / "manifest.json"
        if cache.is_symlink() or manifest.is_symlink() or manifest.stat().st_size > 65536:
            raise ValueError("cached core manifest is invalid")
        value = json.loads(manifest.read_text())
        wheel, = cache.glob("*.whl")
        if value != {**expected, "wheel_sha256": wheel_source(root, wheel)}:
            raise ValueError("cached core wheel build inputs differ")
        print("Exact-source core wheel cache verified")
        return wheel
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        print("Core wheel cache absent or mismatched; building current checkout")
    if cache.is_symlink():
        cache.unlink()
    elif cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True)
    subprocess.run(["uv", "build", "--package", "ml-experiment-control", "--wheel", "--out-dir", str(cache)],
                   cwd=root, check=True)
    wheel, = cache.glob("*.whl")
    value = {**expected, "wheel_sha256": wheel_source(root, wheel)}
    (cache / "manifest.json").write_text(json.dumps(value, sort_keys=True) + "\n")
    return wheel


def main():
    wheel = prepare()
    subprocess.run(["uv", "pip", "install", "--offline", "--no-deps", "--python", sys.executable, str(wheel)], check=True)


if __name__ == "__main__":
    main()
