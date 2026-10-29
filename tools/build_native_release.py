#!/usr/bin/env python3
"""Package already-built native executables; never install or publish them."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
TARGET = "x86_64-unknown-linux-gnu"
BINARIES = ("ml-exp-rs", "ml-expd-rs", "ml-exp-worker-rs", "ml-expd-state-audit")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(binary_dir, output, version):
    if not re.fullmatch(r"(?:0|[1-9][0-9]{0,5})\.(?:0|[1-9][0-9]{0,5})\.(?:0|[1-9][0-9]{0,5})", version):
        raise ValueError("native publication requires a stable exact version")
    workspace = tomllib.loads((ROOT / "rust/Cargo.toml").read_text())
    if workspace["workspace"]["package"]["version"] != version:
        raise ValueError("workspace version differs from requested release")
    for name in BINARIES:
        path = binary_dir / name
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 128 * 1024**2:
            raise ValueError("native executable is missing or outside its size limit")
        result = subprocess.run([str(path.resolve()), "--version"], check=True,
                                capture_output=True, timeout=5, env={"LANG": "C"})
        if result.stdout.decode().strip() != f"{name} {version}":
            raise ValueError("native executable version differs")
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    assets = []
    for name in BINARIES:
        filename = f"{name}-{TARGET}"
        path = output / filename
        shutil.copyfile(binary_dir / name, path)
        path.chmod(0o755)
        assets.append({"name": filename, "target": TARGET, "executable": name,
                       "bytes": path.stat().st_size, "sha256": digest(path)})
    manifest = {"schema_version": 1, "version": version, "protocol_version": 2, "assets": assets}
    (output / "native-release.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copyfile(ROOT / "rust/install.sh", output / "install-native.sh")
    (output / "install-native.sh").chmod(0o755)
    # The server bundle contains no secrets, provider profiles, databases,
    # images or training payloads. Optional Python image effects are separate.
    archive = output / f"ml-expd-native-{version}-{TARGET}.tar.gz"
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))

    def metadata(info):
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = epoch
        return info

    with tarfile.open(archive, "w:gz") as tar:
        for name in BINARIES:
            tar.add(output / f"{name}-{TARGET}", arcname=f"bin/{name}", filter=metadata)
        for source, target in [(ROOT / "rust/README.md", "README.md"),
                               (ROOT / "docs/rust-rewrite.md", "MIGRATION.md"),
                               (output / "native-release.json", "native-release.json")]:
            tar.add(source, arcname=target, filter=metadata)
        for path in sorted((ROOT / "rust/deploy").rglob("*")):
            if path.is_file() and not path.is_symlink():
                tar.add(path, arcname="deploy/" + str(path.relative_to(ROOT / "rust/deploy")), filter=metadata)
    checksums = [f"{digest(path)}  {path.name}\n" for path in sorted(output.iterdir()) if path.is_file()]
    (output / "SHA256SUMS").write_text("".join(checksums))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    try:
        manifest = build(args.binary_dir, args.output, args.version)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"native packaging refused: {error}\n")
    print(json.dumps({"version": manifest["version"], "executables": len(manifest["assets"]), "published": False}))


if __name__ == "__main__":
    main()
