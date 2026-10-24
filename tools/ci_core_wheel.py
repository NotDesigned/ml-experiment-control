"""Install a CI core wheel only after matching every Python source to checkout.

The cache key includes compiler/platform and all core build inputs. The package
job rebuilds source distributions and verifies their exact core source contents;
only an unchanged core wheel may be reused.
"""
from __future__ import annotations

import argparse
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
import tarfile
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def identity(root):
    paths = [root / name for name in ("pyproject.toml", "README.md", "LICENSE", "rust/Cargo.toml", "rust/Cargo.lock")]
    paths += [root / name for name in (".github/workflows/quality.yml", "tools/ci_native.py", ".cargo/config.toml",
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



def source_distribution(root, archive_path):
    """Check the newly built sdist without extracting or executing its contents."""
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    prefix = f"ml_experiment_control-{version}/"
    expected = {name: (root / name).read_bytes() for name in
                ("pyproject.toml", "README.md", "rust/Cargo.toml", "rust/Cargo.lock")}
    expected.update({path.relative_to(root).as_posix(): path.read_bytes()
                     for folder in ("src/experiment_control", "rust/src")
                     for path in (root / folder).rglob("*") if path.is_file()
                     and "__pycache__" not in path.parts and path.suffix != ".pyc"})
    if archive_path.is_symlink() or archive_path.stat().st_size > 16 * 1024**2:
        raise ValueError("core source archive identity or size differs")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or sum(member.size for member in members) > 32 * 1024**2:
            raise ValueError("core source archive has duplicate entries or excessive size")
        actual = {}
        for member in members:
            if (not member.name.startswith(prefix) or ".." in Path(member.name).parts
                    or not (member.isdir() or member.isfile())):
                raise ValueError("core source archive has unsafe entries")
            if member.isfile():
                name = member.name[len(prefix):]
                # All packaged source files must be from this exact checkout.
                if (name in expected or name.startswith("src/") or name.startswith("rust/src/")):
                    actual[name] = archive.extractfile(member).read()
        if actual != expected:
            raise ValueError("core source archive differs from checkout")


def package_output(root, wheel, output):
    """Build current sources and pair them only with a source-verified wheel."""
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(["uv", "build", "--package", "ml-experiment-control", "--sdist",
                    "--out-dir", str(output)], cwd=root, check=True)
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    source_distribution(root, output / f"ml_experiment_control-{version}.tar.gz")
    wheel_source(root, wheel)
    shutil.copyfile(wheel, output / wheel.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-only", action="store_true", help="verify/build the cache without installing a package")
    parser.add_argument("--package-output", type=Path, help="build and verify current core sdist, then copy the verified wheel")
    args = parser.parse_args(argv)
    if args.package_output and not args.cache_only:
        parser.error("--package-output requires --cache-only")
    wheel = prepare()
    if args.package_output:
        package_output(ROOT, wheel, args.package_output.resolve())
    if not args.cache_only:
        subprocess.run(["uv", "pip", "install", "--offline", "--no-deps", "--python", sys.executable, str(wheel)], check=True)
        # Preserve the original editable installation's source search context,
        # including mypy's classification of followed imports. This is a path,
        # never an executable .pth statement; source bytes were checked above.
        (Path(sysconfig.get_path("purelib")) / "_ml_exp_core_checkout.pth").write_text(str(ROOT / "src") + "\n")


if __name__ == "__main__":
    main()
