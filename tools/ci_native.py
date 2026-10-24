"""Identify the selected Rust compiler without loading LLVM on every CI runner.

Only rustup's selected installed compiler can use its local distribution
manifest. Custom compilers and unverifiable manifests execute --version instead.
"""
from __future__ import annotations

import argparse
from datetime import date
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tomllib


NUMBER = r"(?:0|[1-9][0-9]*)"
VERSION = re.compile(rf"({NUMBER}\.{NUMBER}\.{NUMBER}(?:-(?:nightly|dev|beta(?:\.{NUMBER})?))?) "
                     r"\(([0-9a-f]{7,40}) (\d{4}-\d{2}-\d{2})\)")


def version(value):
    match = VERSION.fullmatch(value)
    if match is None:
        raise ValueError("unrecognized Rust compiler version")
    date.fromisoformat(match[3])
    return value


def command(arguments):
    return subprocess.check_output(arguments, text=True, timeout=15).strip()


def manifest_version(compiler, rustup):
    selected = Path(command([str(rustup), "which", "rustc"]))
    if not selected.is_absolute():
        raise ValueError("rustup selected a relative compiler path")
    selected = selected.resolve(strict=True)
    if (selected.name != "rustc" or selected.parent.name != "bin"
            or not selected.is_file() or not os.access(selected, os.X_OK)):
        raise ValueError("rustup selected an unexpected compiler path")
    if not (compiler.samefile(rustup) or compiler.samefile(selected)):
        raise ValueError("PATH compiler is not the selected rustup compiler")
    path = selected.parent.parent / "lib/rustlib/multirust-channel-manifest.toml"
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 2 * 1024**2:
        raise ValueError("invalid installed Rust manifest")
    manifest = tomllib.loads(path.read_text())
    if manifest["manifest-version"] != "2":
        raise ValueError("unsupported Rust manifest")
    rustc = manifest["pkg"]["rustc"]
    value = version(rustc["version"])
    match = VERSION.fullmatch(value)
    # rustup's installed serialization can omit this distribution-only field.
    commit = rustc.get("git_commit_hash")
    if (manifest["pkg"]["rust"]["version"] != value
            or commit is not None and (not isinstance(commit, str)
                or re.fullmatch(r"[0-9a-f]{40}", commit) is None or not commit.startswith(match[2]))
            or date.fromisoformat(manifest["date"]) < date.fromisoformat(match[3])):
        raise ValueError("Rust manifest version identity differs")
    # The selected toolchain directory names its actual host; do not infer a
    # compiler from a different target's available package.
    hosts = [host for host, row in rustc["target"].items()
             if selected.parent.parent.name.endswith("-" + host)
             and row.get("available") is True]
    if len(hosts) != 1:
        raise ValueError("Rust manifest does not match the selected host")
    channel = selected.parent.parent.name.removesuffix("-" + hosts[0])
    release = match[1]
    expected = {release}
    if "-" not in release:
        expected |= {"stable", "stable-" + manifest["date"]}
    elif "-nightly" in release:
        expected |= {"nightly", "nightly-" + manifest["date"]}
    elif "-beta" in release:
        expected |= {"beta", "beta-" + manifest["date"]}
    if channel not in expected:
        raise ValueError("Rust manifest does not match the selected toolchain")
    return value


def real_version(compiler):
    value = command([str(compiler), "--version"])
    if not value.startswith("rustc "):
        raise ValueError("compiler did not report a rustc version")
    return version(value.removeprefix("rustc "))


def identify(*, check=False):
    compiler = Path(os.environ.get("RUSTC") or shutil.which("rustc") or "rustc")
    rustup = shutil.which("rustup")
    try:
        if os.environ.get("RUSTC") or rustup is None:
            raise ValueError("custom compiler requires direct version evidence")
        value = manifest_version(compiler, Path(rustup))
        source = "selected rustup manifest"
    except (OSError, ValueError, KeyError, TypeError, AttributeError, subprocess.SubprocessError):
        value = real_version(compiler)
        source = "compiler --version"
    if check and source == "selected rustup manifest" and real_version(compiler) != value:
        raise ValueError("selected compiler version disagrees with its manifest")
    print("Compiler identity: " + source, file=sys.stderr)
    match = VERSION.fullmatch(value)
    return {"toolchain": "-".join(match.groups()),
            "abi": sysconfig.get_platform() + "-" + "-".join(platform.libc_ver()) + "-" + sys.implementation.cache_tag}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="also execute the real selected compiler and require an exact match")
    args = parser.parse_args(argv)
    for key, value in identify(check=args.check).items():
        print(key + "=" + value)


if __name__ == "__main__":
    main()
