"""Deployment archives are complete, hash checked and never overwrite a target."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from build_deployment_bundle import build
import build_deployment_bundle as bundles


def wheel(directory, name, version="1.0"):
    directory.mkdir(exist_ok=True)
    path = directory / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{name}-{version}.dist-info/METADATA",
                         f"Metadata-Version: 2.1\nName: {name.replace('_', '-')}\nVersion: {version}\n")
    return path


def test_client_bundle_is_offline_and_corruption_prevents_install(tmp_path):
    wheels = tmp_path / "wheels"
    original = wheel(wheels, "ml_experiment_client")
    output = tmp_path / "client.tar.gz"
    result = build("client", wheels, output)
    assert result["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    with tarfile.open(output) as archive:
        archive.extractall(tmp_path / "extract", filter="data")
    root = tmp_path / "extract/ml-expd-client"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["wheels"][original.name]["sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
    assert len(manifest["wheels"]) == 1
    assert "ml-experiment-client==1.0 --hash=sha256:" in (root / "requirements.txt").read_text()
    assert (root / "two-machine-quickstart.md").is_file()
    verified = subprocess.run(["sha256sum", "--check", "SHA256SUMS"], cwd=root, capture_output=True)
    assert verified.returncode == 0
    (root / "wheelhouse" / original.name).write_bytes(b"corrupt")
    target = tmp_path / "venv"
    failed = subprocess.run([str(root / "install.sh"), str(target)], capture_output=True, text=True)
    assert failed.returncode != 0 and "FAILED" in failed.stdout and not target.exists()
    with pytest.raises(ValueError, match="output already exists"):
        build("client", wheels, output)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == result["sha256"]


def test_bundle_rejects_ambiguous_project_versions(tmp_path):
    for version in ["1.0", "1.1"]:wheel(tmp_path, "ml_experiment_client", version)
    with pytest.raises(ValueError, match="exactly one"):
        build("client", tmp_path, tmp_path / "result.tar.gz")


def test_server_bundle_contains_downloaded_dependency_closure(tmp_path, monkeypatch):
    wheels = tmp_path / "wheels"
    core = wheel(wheels, "ml_experiment_control")
    server = wheel(wheels, "ml_experiment_server")
    dependency = wheel(tmp_path / "dependencies", "PyYAML", "6.0.3")
    lock = tmp_path / "uv.lock"
    lock.write_text('[[package]]\nname = "pyyaml"\nversion = "6.0.3"\nwheels = [{hash = "sha256:'
                    + hashlib.sha256(dependency.read_bytes()).hexdigest() + '"}]\n')
    monkeypatch.setattr(bundles, "LOCKFILE", lock)
    def download(argv, **kwargs):
        if argv[0] == "uv":
            assert "--frozen" in argv and "--no-emit-workspace" in argv and "--no-hashes" not in argv
            Path(argv[argv.index("--output-file") + 1]).write_text("PyYAML==6.0.3 --hash=sha256:test\n")
            return
        assert argv[:4] == [sys.executable, "-m", "pip", "wheel"]
        assert "--require-hashes" in argv and "--no-build-isolation" in argv and kwargs["check"]
        target = Path(argv[argv.index("--wheel-dir") + 1])
        for item in [core, server, dependency]:shutil.copyfile(item, target / item.name)
    monkeypatch.setattr(subprocess, "run", download)
    output = tmp_path / "server.tar.gz"
    result = build("server", wheels, output)
    assert result["packages"] == 3
    with tarfile.open(output) as archive:
        requirements = archive.extractfile("ml-expd-server/requirements.txt").read().decode()
        assert len(requirements.splitlines()) == 3 and "pyyaml==6.0.3 --hash=sha256:" in requirements
        assert archive.extractfile("ml-expd-server/minimal-api.yaml").read().startswith(b"# Basic API")
        installer = archive.extractfile("ml-expd-server/install.sh").read().decode()
        assert "--no-index" in installer and "--require-hashes" in installer
    with dependency.open("ab") as stream:stream.write(b"unreviewed changed upstream wheel")
    rejected = tmp_path / "rejected.tar.gz"
    with pytest.raises(ValueError, match="not in uv.lock"):
        build("server", wheels, rejected)
    assert not rejected.exists()


def test_failed_dependency_download_leaves_no_published_archive(tmp_path, monkeypatch):
    for name in ["ml_experiment_control", "ml_experiment_server"]:wheel(tmp_path, name)
    target = tmp_path / "result.tar.gz"
    def failed(argv, **kwargs):raise subprocess.CalledProcessError(1, argv)
    monkeypatch.setattr(subprocess, "run", failed)
    with pytest.raises(subprocess.CalledProcessError):build("server", tmp_path, target)
    assert not target.exists() and not list(tmp_path.glob("ml-expd-bundle-*"))
