#!/usr/bin/env python3
"""Package already-built native executables; never install or publish them."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tarfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
TARGET = "x86_64-unknown-linux-gnu"
BINARIES = ("ml-exp", "ml-expd-rs", "ml-exp-worker-rs")
CLIENT_TARGETS = ("aarch64-unknown-linux-musl", "x86_64-apple-darwin",
                  "aarch64-apple-darwin", "x86_64-pc-windows-msvc")
# The image bridge the server's builder imports; not a Python service install.
BRIDGE = {"experiment_control": "src/experiment_control",
          "ml_exp_server": "server/src/ml_exp_server",
          "ml_exp_worker": "worker/src/ml_exp_worker"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def static_worker(path):
    """Require an x86_64 ELF worker with no dynamic-loader dependency."""
    with path.open("rb") as stream:
        header = stream.read(64)
        if len(header) != 64 or header[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<H", header, 18)[0] != 62:
            return False
        offset = struct.unpack_from("<Q", header, 32)[0]
        size, count = struct.unpack_from("<HH", header, 54)
        if size != 56 or not 1 <= count <= 128 or offset + size * count > path.stat().st_size:
            return False
        stream.seek(offset)
        types = [struct.unpack("<I", stream.read(size)[:4])[0] for _ in range(count)]
        return 1 in types and 3 not in types  # PT_LOAD present, PT_INTERP absent.


def client_format(path, target):
    """Reject artifacts whose executable format or CPU differs from their label."""
    with path.open('rb') as stream:
        header = stream.read(64)
        if len(header) != 64:
            return False
        if target == 'aarch64-unknown-linux-musl':
            return header[:6] == b'\x7fELF\x02\x01' and struct.unpack_from('<H', header, 18)[0] == 183
        if 'apple-darwin' in target:
            return header[:4] == b'\xcf\xfa\xed\xfe' and struct.unpack_from('<I', header, 4)[0] == (0x0100000c if target.startswith('aarch64') else 0x01000007)
        if target == 'x86_64-pc-windows-msvc':
            offset = struct.unpack_from('<I', header, 60)[0]
            if header[:2] != b'MZ' or not 64 <= offset <= 65536:
                return False
            stream.seek(offset)
            return stream.read(6) == b'PE\0\0\x64\x86'
        return False


def bridge_files():
    """Map bundle paths to sources of tracked bridge files, without build output."""
    files = {}
    for package, source in BRIDGE.items():
        tracked = subprocess.run(["git", "ls-files", "-z", source], cwd=ROOT, check=True,
                                 capture_output=True).stdout.decode().split("\0")
        for name in filter(None, tracked):
            files[f"{package}/{Path(name).relative_to(source)}"] = ROOT / name
    if not files:
        raise ValueError("image bridge sources are missing")
    return dict(sorted(files.items()))


def source_commit():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def build(binary_dir, output, version, client_dir=None):
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
    if not static_worker(binary_dir / "ml-exp-worker-rs"):
        raise ValueError("release worker must be a static Linux x86_64 ELF executable")
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    assets = []
    for name in BINARIES:
        filename = f"{name}-{TARGET}"
        path = output / filename
        shutil.copyfile(binary_dir / name, path)
        path.chmod(0o755)
        assets.append({"name": filename, "target": TARGET, "executable": name,
                       "bytes": path.stat().st_size, "sha256": digest(path)})
    if client_dir is None:
        raise ValueError("all native client platform artifacts are required")
    expected = {f"ml-exp-{target}" + (".exe" if "windows" in target else "")
                for target in CLIENT_TARGETS}
    if {p.name for p in client_dir.iterdir()} != expected:
        raise ValueError("client platform artifacts are incomplete or unexpected")
    for target in CLIENT_TARGETS:
        name = f"ml-exp-{target}" + (".exe" if "windows" in target else "")
        path = client_dir / name
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 128 * 1024**2 or not client_format(path, target):
            raise ValueError("invalid platform client executable")
        shutil.copyfile(path, output / name)
        (output / name).chmod(0o755)
        assets.append({"name": name, "target": target, "executable": "ml-exp",
                       "bytes": path.stat().st_size, "sha256": digest(path)})
    manifest = {"schema_version": 1, "version": version, "assets": assets}
    (output / "native-release.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copyfile(ROOT / "rust/install.sh", output / "install-native.sh")
    (output / "install-native.sh").chmod(0o755)
    shutil.copyfile(ROOT / "rust/install.ps1", output / "install-native.ps1")
    # The server bundle contains no secrets, provider profiles, databases,
    # images or training payloads. It carries the image bridge sources so one
    # release fully identifies what rust/deploy/deploy_native.py installs.
    bridge = bridge_files()
    release_info = output / "release-info.json"
    release_info.write_text(json.dumps({
        "schema_version": 1, "version": version, "source_commit": source_commit(),
        "python": {name: digest(path) for name, path in bridge.items()}}, indent=2) + "\n")
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
                               (ROOT / "docs/acceptance-status.md", "ACCEPTANCE-STATUS.md"),
                               (output / "native-release.json", "native-release.json"),
                               (release_info, "release-info.json")]:
            tar.add(source, arcname=target, filter=metadata)
        for name, path in bridge.items():
            tar.add(path, arcname=f"python/{name}", filter=metadata)
        for path in sorted((ROOT / "rust/deploy").rglob("*")):
            if path.is_file() and not path.is_symlink():
                tar.add(path, arcname="deploy/" + str(path.relative_to(ROOT / "rust/deploy")), filter=metadata)
    release_info.unlink()
    checksums = [f"{digest(path)}  {path.name}\n" for path in sorted(output.iterdir()) if path.is_file()]
    (output / "SHA256SUMS").write_text("".join(checksums))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--client-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = build(args.binary_dir, args.output, args.version, args.client_dir)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"native packaging refused: {error}\n")
    print(json.dumps({"version": manifest["version"], "executables": len(manifest["assets"]), "published": False}))


if __name__ == "__main__":
    main()
