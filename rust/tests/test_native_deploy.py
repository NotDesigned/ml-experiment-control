#!/usr/bin/env python3
"""Exercise the deployment tool with fake services; no real unit is touched."""
import fcntl
import hashlib
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest

TOOL = Path(__file__).resolve().parents[1] / "deploy/deploy_native.py"
BINARIES = ("ml-exp", "ml-expd-rs", "ml-exp-worker-rs", "ml-expd-state-audit")
TARGET = "x86_64-unknown-linux-gnu"


def program(path, name, version, tag=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/sh\n# {tag}\n[ "$1" = --version ] && echo "{name} {version}"\nexit 0\n')
    path.chmod(0o755)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Deploy(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name)
        self.root = self.base / "root"
        self.site = self.base / "site"
        self.state = self.base / "builder"
        self.state.mkdir()
        for name in BINARIES:
            program(self.root / "bin" / name, name, "1.0.0")
        (self.root / "bin/ml-expd-rs.SOURCE-COMMIT").write_text("stale\n")
        for package in ("experiment_control", "ml_exp_server", "ml_exp_worker"):
            (self.site / package).mkdir(parents=True)
            (self.site / package / "old.py").write_text("old = 1\n")
            (self.site / package / "extra.dat").write_text("kept\n")
        (self.site / "ml_exp_server/image_builder.py").write_text("old = 1\n")
        self.cloud = self.base / "cloud.json"
        self.cloud.write_text(json.dumps({"builds": {"worker_sha256": "sha256:old"}}))
        self.cloud.chmod(0o600)
        self.lock = self.base / "publisher.lock"
        self.lock.touch()
        token = self.base / "token"
        token.write_text("secret\n")

        root, test = self.root, self

        class Health(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                version = subprocess.check_output([str(root / "bin/ml-expd-rs"), "--version"],
                                                  text=True).split()[1]
                if version in test.unhealthy:
                    version = "broken"
                body = json.dumps({"status": "ok", "server_version": version}).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.unhealthy = set()
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Health)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        client = self.base / "client.json"
        client.write_text(json.dumps({"url": f"http://127.0.0.1:{server.server_port}",
                                      "token_file": str(token)}))
        (self.root / "deployment.json").write_text(json.dumps({
            "api_unit": "api.service", "builder_unit": "builder.service",
            "server_config": str(self.base / "server.json"), "client_config": str(client),
            "cloud_config": str(self.cloud), "builder_state": str(self.state),
            "python": sys.executable, "python_site_packages": str(self.site),
            "publisher_locks": [str(self.lock)]}))
        self.fake = self.base / "fake-bin"
        self.fake.mkdir()
        self.log = self.base / "systemctl.log"
        self.status = self.base / "api-status"
        self.status.write_text("0\n")
        systemctl = self.fake / "systemctl"
        systemctl.write_text(
            f'#!/bin/sh\necho "$1 $2" >> {self.log}\n'
            f'[ "$1" = show ] && cat {self.status}\nexit 0\n')
        systemctl.chmod(0o755)

    def bundle(self, version, tag="new"):
        source = self.base / f"bundle-{version}-{tag}"
        assets = []
        for name in BINARIES:
            program(source / "bin" / name, name, version, tag)
            assets.append({"executable": name, "target": TARGET,
                           "sha256": digest(source / "bin" / name)})
        (source / "native-release.json").write_text(json.dumps({"version": version, "assets": assets}))
        python = {}
        for name in ("experiment_control/new.py", "ml_exp_server/image_builder.py", "ml_exp_worker/new.py"):
            path = source / "python" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"tag = {tag!r}\n")
            python[name] = digest(path)
        (source / "release-info.json").write_text(json.dumps(
            {"version": version, "source_commit": "c" * 40, "python": python}))
        archive = self.base / f"{source.name}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    tar.add(path, arcname=str(path.relative_to(source)))
        return archive

    def run_tool(self, *args):
        env = {**os.environ, "PATH": f"{self.fake}:{os.environ['PATH']}"}
        return subprocess.run([sys.executable, str(TOOL), "--root", str(self.root),
                               "--health-timeout", "2", *map(str, args)],
                              env=env, capture_output=True, text=True, timeout=60)

    def deploy(self, archive, *args):
        return self.run_tool("--bundle", archive, "--sha256", digest(archive), *args)

    def version(self):
        return subprocess.check_output([str(self.root / "bin/ml-expd-rs"), "--version"],
                                       text=True).split()[1]

    def calls(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def test_verification_only_changes_nothing(self):
        result = self.deploy(self.bundle("1.1.0"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)["applied"])
        self.assertEqual(self.version(), "1.0.0")
        self.assertEqual(self.calls(), [])
        self.assertFalse((self.root / "previous").exists())

    def test_apply_installs_release_and_pins_only_new_builds(self):
        result = self.deploy(self.bundle("1.1.0"), "--apply")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.version(), "1.1.0")
        worker = "sha256:" + digest(self.root / "bin/ml-exp-worker-rs")
        self.assertEqual(json.loads(self.cloud.read_text())["builds"]["worker_sha256"], worker)
        self.assertEqual(self.cloud.stat().st_mode & 0o777, 0o600)
        self.assertEqual(json.loads((self.root / "DEPLOYED.json").read_text())["source_commit"], "c" * 40)
        self.assertFalse((self.root / "bin/ml-expd-rs.SOURCE-COMMIT").exists())
        self.assertTrue((self.site / "ml_exp_worker/new.py").exists())
        self.assertFalse((self.site / "ml_exp_worker/old.py").exists())
        self.assertEqual((self.site / "ml_exp_worker/extra.dat").read_text(), "kept\n")
        calls = self.calls()
        self.assertLess(calls.index("stop api.service"), calls.index("stop builder.service"))
        self.assertLess(calls.index("start builder.service"), calls.index("start api.service"))

    def test_unhealthy_release_restores_previous_files_and_pin(self):
        before = self.cloud.read_text()
        self.unhealthy.add("1.1.0")
        result = self.deploy(self.bundle("1.1.0"), "--apply")
        self.assertEqual(result.returncode, 1)
        self.assertIn("restored 1.0.0", result.stderr)
        self.assertEqual(self.version(), "1.0.0")
        self.assertEqual(self.cloud.read_text(), before)
        self.assertTrue((self.site / "ml_exp_worker/old.py").exists())
        self.assertFalse((self.site / "ml_exp_worker/new.py").exists())
        self.assertEqual(self.calls()[-1], "start api.service")

    def test_explicit_rollback_restores_previous_release(self):
        self.assertEqual(self.deploy(self.bundle("1.1.0"), "--apply").returncode, 0)
        result = self.run_tool("--rollback")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.version(), "1.0.0")
        self.assertEqual(json.loads(self.cloud.read_text())["builds"]["worker_sha256"], "sha256:old")

    def test_rollback_without_retained_install_refuses(self):
        result = self.run_tool("--rollback")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.calls(), [])

    def test_active_build_refuses_without_stopping_anything(self):
        lock = self.state / "build.lock"
        with open(lock, "a") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            result = self.deploy(self.bundle("1.1.0"), "--apply")
        self.assertEqual(result.returncode, 1)
        self.assertIn("build.lock", result.stderr)
        self.assertEqual(self.version(), "1.0.0")
        self.assertEqual(self.calls(), [])

    def test_abnormal_api_exit_restarts_services_without_installing(self):
        self.status.write_text("1\n")
        result = self.deploy(self.bundle("1.1.0"), "--apply")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.version(), "1.0.0")
        self.assertEqual(self.calls()[-1], "start api.service")

    def test_downgrade_refuses(self):
        result = self.deploy(self.bundle("0.9.0"), "--apply")
        self.assertEqual(result.returncode, 1)
        self.assertIn("downgrade", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_changed_binary_or_checksum_refuses(self):
        archive = self.bundle("1.1.0")
        result = self.run_tool("--bundle", archive, "--sha256", "0" * 64, "--apply")
        self.assertEqual(result.returncode, 1)
        source = self.base / "bundle-1.1.0-new"
        (source / "bin/ml-expd-rs").write_text('#!/bin/sh\necho "ml-expd-rs 1.1.0"\n')
        tampered = self.base / "tampered.tar.gz"
        with tarfile.open(tampered, "w:gz") as tar:
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    tar.add(path, arcname=str(path.relative_to(source)))
        result = self.deploy(tampered, "--apply")
        self.assertEqual(result.returncode, 1)
        self.assertIn("does not match", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_escaping_bundle_member_refuses(self):
        archive = self.base / "escape.tar.gz"
        outside = self.base / "evil"
        outside.write_text("x")
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(outside, arcname="../../evil-copy")
        result = self.deploy(archive, "--apply")
        self.assertEqual(result.returncode, 1)
        self.assertIn("unsafe", result.stderr)
        self.assertFalse((self.root.parent / "evil-copy").exists())


if __name__ == "__main__":
    unittest.main()
