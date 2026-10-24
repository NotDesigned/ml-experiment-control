"""Package reviewed wheels and locked dependencies for offline installation.

Run on the target Linux architecture/Python minor with uv, pip, setuptools and wheel.
This is a release tool; the installed service needs neither this tool nor Rust.
"""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile


PACKAGES = {"server": ("ml_experiment_control", "ml_experiment_server"),
            "client": ("ml_experiment_client",)}
LOCKFILE = Path(__file__).resolve().parents[1] / "uv.lock"


def wheel_identity(path):
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(members) != 1:
            raise ValueError("wheel must have one package metadata record")
        metadata = BytesParser().parsebytes(archive.read(members[0]))
    name, version = metadata["Name"], metadata["Version"]
    if (not name or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name)
            or not version or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", version)):
        raise ValueError("invalid wheel package identity")
    return re.sub(r"[-_.]+", "-", name).lower(), version


def build(component, wheels, output):
    if output.exists():
        raise ValueError("output already exists; choose a new bundle path")
    selected = []
    for package in PACKAGES[component]:
        matches = list(wheels.glob(package + "-*.whl"))
        if len(matches) != 1:
            raise ValueError("use a clean wheel directory with exactly one " + package + " wheel")
        selected.append(matches[0].resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ml-expd-bundle-", dir=output.parent) as temporary:
        root = Path(temporary) / ("ml-expd-" + component)
        house = root / "wheelhouse"
        house.mkdir(parents=True)
        if component == "server":
            inputs = Path(temporary) / "inputs.txt"
            subprocess.run(["uv", "export", "--project", str(LOCKFILE.parent), "--frozen", "--no-dev",
                            "--no-emit-workspace", "--package", "ml-experiment-server",
                            "--output-file", str(inputs)], check=True, stdout=subprocess.DEVNULL)
            with inputs.open("a") as stream:
                for path in selected:
                    stream.write(str(path) + " --hash=sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() + "\n")
            subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-cache-dir", "--require-hashes",
                            "--no-build-isolation", "--wheel-dir", str(house), "-r", str(inputs)], check=True)
        else:
            shutil.copyfile(selected[0], house / selected[0].name)
        locked, sources = {}, {}
        if component == "server":
            for package in tomllib.loads(LOCKFILE.read_text())["package"]:
                locked[package["name"], package["version"]] = {row["hash"] for row in package.get("wheels", [])}
                sources[package["name"], package["version"]] = package.get("sdist", {}).get("hash")
        local = {wheel.name: hashlib.sha256(wheel.read_bytes()).hexdigest() for wheel in selected}
        requirements, records, names = [], {}, set()
        for wheel in sorted(house.glob("*.whl")):
            name, version = wheel_identity(wheel)
            if name in names:
                raise ValueError("duplicate package in wheelhouse")
            names.add(name)
            sha = hashlib.sha256(wheel.read_bytes()).hexdigest()
            if wheel.name in local:
                if sha != local[wheel.name]:
                    raise ValueError("download changed a reviewed project wheel")
            elif ("sha256:" + sha not in locked.get((name, version), set())
                  and not sources.get((name, version))):
                raise ValueError("dependency wheel hash is not in uv.lock: " + name)
            requirements.append(f"{name}=={version} --hash=sha256:{sha}")
            records[wheel.name] = {"package": name, "version": version, "sha256": sha,
                                   "bytes": wheel.stat().st_size}
            if wheel.name not in local and "sha256:" + sha not in locked.get((name, version), set()):
                records[wheel.name]["built_from_locked_sdist"] = sources[name, version]
        if any(package.replace("_", "-") not in names for package in PACKAGES[component]):
            raise ValueError("wheel download omitted a selected package")
        (root / "requirements.txt").write_text("\n".join(requirements) + "\n")
        minor = f"{sys.version_info.major}.{sys.version_info.minor}"
        manifest = {"component": component, "python_minor": minor,
                    "platform": platform.platform(), "machine": platform.machine(),
                    "wheels": records}
        if component == "server":
            manifest["uv_lock_sha256"] = hashlib.sha256(LOCKFILE.read_bytes()).hexdigest()
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        repository = Path(__file__).resolve().parents[1]
        shutil.copyfile(repository / "docs/two-machine-quickstart.md", root / "two-machine-quickstart.md")
        if component == "server":
            shutil.copyfile(repository / "server/examples/minimal-api.yaml", root / "minimal-api.yaml")
        python_check = (f'actual=$("$python" -c \'import sys; print("%s.%s" % sys.version_info[:2])\')\n'
                        f'[ "$actual" = "{minor}" ] || {{ echo "This server bundle requires Python {minor}" >&2; exit 1; }}\n'
                        if component == "server" else "")
        installer = f'''#!/bin/sh
set -eu
base=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
venv=${{1:?Usage: ./install.sh /absolute/path/to/new-venv}}
case "$venv" in /*) ;; *) echo "Use an absolute venv path" >&2; exit 1;; esac
[ ! -e "$venv" ] || {{ echo "Destination exists; choose a new venv" >&2; exit 1; }}
(cd "$base" && sha256sum --check SHA256SUMS)
python=${{PYTHON:-python3}}
{python_check}"$python" -m venv "$venv"
"$venv/bin/python" -m pip --isolated --disable-pip-version-check install --no-index --no-deps --find-links "$base/wheelhouse" --require-hashes -r "$base/requirements.txt"
"$venv/bin/python" -m pip --isolated --disable-pip-version-check check
"$venv/bin/ml-exp{'d' if component == 'server' else ''}" --help >/dev/null
echo "Installed {component} into $venv; no service was started."
'''
        (root / "install.sh").write_text(installer)
        (root / "install.sh").chmod(0o755)
        (root / "README.txt").write_text(
            f"ML-Expd {component} offline installation\n\n"
            "Verify the archive SHA256 against the trusted release channel before extraction.\n"
            "Inspect manifest.json for exact package versions and build platform.\n"
            "Run: ./install.sh /absolute/path/to/new-venv\n"
            "Python with venv/ensurepip, /bin/sh and sha256sum must already be available.\n"
            + (f"Requires Python {minor} and a compatible Linux architecture/libc.\n"
               if component == "server" else "Requires Python 3.10 or newer.\n")
            + "No Rust, uv, package index or cloud credentials are needed for installation.\n"
            "Configure the service/client separately; see two-machine-quickstart.md.\n")
        checksums = [hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.relative_to(root).as_posix()
                     for path in sorted(root.rglob("*")) if path.is_file()]
        (root / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
        packed = Path(temporary) / "bundle.tar.gz"
        def normalize(info):
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            return info
        with packed.open("wb") as stream, gzip.GzipFile(fileobj=stream, mode="wb", mtime=0, filename="") as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                archive.add(root, arcname=root.name, filter=normalize)
        # Publish only the complete archive, and never replace an existing bundle.
        os.link(packed, output)
    return {"path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "bytes": output.stat().st_size, "packages": len(records)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=PACKAGES, required=True)
    parser.add_argument("--wheels", type=Path, required=True, help="clean directory of reviewed project wheels")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build(args.component, args.wheels, args.output), indent=2))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Bundle build failed: {error}\n")


if __name__ == "__main__":
    main()
