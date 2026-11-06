#!/usr/bin/env python3
"""Exercise the deployment tool with fake services; no real unit is touched."""
import fcntl
import hashlib
import http.server
import importlib.util
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

TOOL = Path(__file__).resolve().parents[1] / "deploy/deploy_native.py"
BINARIES = ("ml-exp", "ml-expd-rs", "ml-exp-worker-rs")
TARGET = "x86_64-unknown-linux-gnu"


def program(path, name, version, tag=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/sh\n# {tag}\n[ "$1" = --version ] && echo "{name} {version}" && exit 0\n'
                    '[ -n "$FAKE_SERVER_LOG" ] && echo "$3" >> "$FAKE_SERVER_LOG"\n'
                    '[ "$3" = "$FAKE_SERVER_REFUSES" ] && exit 1\nexit 0\n')
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
        self.server_log = self.base / "server.log"
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
            f'#!/bin/sh\n[ "$4" = User ] && echo {pwd.getpwuid(os.geteuid()).pw_name} && exit 0\n'
            f'echo "$1 $2" >> {self.log}\n'
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

    def run_tool(self, *args, refuses=""):
        env = {**os.environ, "PATH": f"{self.fake}:{os.environ['PATH']}",
               "FAKE_SERVER_LOG": str(self.server_log), "FAKE_SERVER_REFUSES": refuses}
        return subprocess.run([sys.executable, str(TOOL), "--root", str(self.root),
                               "--health-timeout", "2", *map(str, args)],
                              env=env, capture_output=True, text=True, timeout=60)

    def deploy(self, archive, *args, **kwargs):
        return self.run_tool("--bundle", archive, "--sha256", digest(archive), *args, **kwargs)

    def empty_host(self):
        shutil.rmtree(self.root / "bin")
        shutil.rmtree(self.site)

    def server_calls(self):
        return self.server_log.read_text().splitlines() if self.server_log.exists() else []

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

    def test_smaller_bridge_removes_nested_modules_and_rollback_restores_them(self):
        removed = [self.site / "experiment_control/backends/nested.py",
                   self.site / "ml_exp_server/builds/nested.py",
                   self.site / "ml_exp_worker/nested.py"]
        for path in removed:
            path.parent.mkdir(exist_ok=True)
            path.write_text("nested = 1\n")
        before = {path: path.read_text() for path in removed}
        self.assertEqual(self.deploy(self.bundle("1.1.0"), "--apply").returncode, 0)
        for path in removed:
            self.assertFalse(path.exists(), path)
            self.assertTrue((self.root / "previous/python" / path.relative_to(self.site)).is_file())
        result = self.run_tool("--rollback")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({path: path.read_text() for path in removed}, before)
        self.assertFalse((self.site / "experiment_control/new.py").exists())

    def test_other_files_in_bin_are_left_alone(self):
        other = self.root / "bin/operator-tool"
        program(other, other.name, "1.0.0")
        before = other.read_bytes()
        self.assertEqual(self.deploy(self.bundle("1.1.0"), "--apply").returncode, 0)
        self.assertEqual(other.read_bytes(), before)
        self.assertFalse((self.root / "previous/bin" / other.name).exists())
        self.assertEqual(self.run_tool("--rollback").returncode, 0)
        self.assertEqual(other.read_bytes(), before)

    def test_first_install_initializes_starts_and_records_on_an_empty_host(self):
        self.empty_host()
        archive = self.bundle("1.1.0")
        result = self.deploy(archive, "--first-install")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)["applied"])
        self.assertEqual(self.server_calls(), ["--check"])
        self.assertFalse((self.root / "bin").exists())
        self.assertEqual(self.calls(), [])

        result = self.deploy(archive, "--first-install", "--apply")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.server_calls(), ["--check", "--check", "--initialize"])
        self.assertEqual(self.version(), "1.1.0")
        self.assertTrue((self.site / "ml_exp_server/image_builder.py").exists())
        worker = "sha256:" + digest(self.root / "bin/ml-exp-worker-rs")
        self.assertEqual(json.loads(self.cloud.read_text())["builds"]["worker_sha256"], worker)
        self.assertEqual(self.calls(), ["start builder.service", "start api.service"])
        recorded = json.loads((self.root / "DEPLOYED.json").read_text())
        self.assertEqual((recorded["version"], recorded["action"]), ("1.1.0", "first install"))
        self.assertFalse((self.root / "previous").exists())

    def test_first_install_refuses_any_trace_of_an_earlier_install(self):
        archive = self.bundle("1.1.0")
        self.assertIn("bin/ml-expd-rs exists", self.deploy(archive, "--first-install", "--apply").stderr)
        (self.root / "bin/ml-expd-rs").unlink()
        for name, make in (("previous", Path.mkdir), ("DEPLOYED.json", Path.touch)):
            path = self.root / name
            make(path)
            result = self.deploy(archive, "--first-install", "--apply")
            self.assertEqual(result.returncode, 1)
            self.assertIn(f"{name} exists", result.stderr)
            path.rmdir() if name == "previous" else path.unlink()
        self.assertEqual(self.server_calls(), [])
        self.assertEqual(self.calls(), [])
        self.assertEqual(json.loads(self.cloud.read_text())["builds"]["worker_sha256"], "sha256:old")

    def test_refused_initialization_leaves_the_host_ready_to_repeat(self):
        self.empty_host()
        archive = self.bundle("1.1.0")
        result = self.deploy(archive, "--first-install", "--apply", refuses="--initialize")
        self.assertEqual(result.returncode, 1)
        self.assertFalse((self.root / "bin/ml-expd-rs").exists())
        self.assertEqual(self.calls(), [])
        self.assertEqual(json.loads(self.cloud.read_text())["builds"]["worker_sha256"], "sha256:old")
        result = self.deploy(archive, "--first-install", "--apply")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_failed_check_or_missing_cloud_config_installs_nothing(self):
        self.empty_host()
        archive = self.bundle("1.1.0")
        result = self.deploy(archive, "--first-install", "--apply", refuses="--check")
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("--initialize", self.server_calls())
        self.cloud.unlink()
        result = self.deploy(archive, "--first-install", "--apply")
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("--initialize", self.server_calls())
        self.assertFalse((self.root / "bin/ml-expd-rs").exists())

    def test_unhealthy_first_install_stops_units_and_an_ordinary_apply_records_it(self):
        self.empty_host()
        archive = self.bundle("1.1.0")
        self.unhealthy.add("1.1.0")
        result = self.deploy(archive, "--first-install", "--apply")
        self.assertEqual(result.returncode, 1)
        self.assertIn("stays installed", result.stderr)
        self.assertEqual(self.calls()[-2:], ["stop api.service", "stop builder.service"])
        self.assertFalse((self.root / "DEPLOYED.json").exists())
        self.assertFalse((self.root / "previous").exists())
        self.assertEqual(self.version(), "1.1.0")
        self.unhealthy.clear()
        result = self.deploy(archive, "--apply")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((self.root / "DEPLOYED.json").read_text())["action"], "deploy")

    def test_ordinary_deploy_on_an_empty_host_names_first_install(self):
        self.empty_host()
        result = self.deploy(self.bundle("1.1.0"), "--apply")
        self.assertEqual(result.returncode, 1)
        self.assertIn("--first-install", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_missing_publisher_lock_needs_no_proof_and_is_not_created(self):
        self.lock.unlink()
        result = self.deploy(self.bundle("1.1.0"), "--apply")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.version(), "1.1.0")
        self.assertFalse(self.lock.exists())

    def test_held_publisher_lock_refuses_and_restarts_services(self):
        with open(self.lock, "a") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            result = self.deploy(self.bundle("1.1.0"), "--apply")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.version(), "1.0.0")
        self.assertEqual(self.calls()[-1], "start api.service")

    def tool(self):
        spec = importlib.util.spec_from_file_location("deploy_native_under_test", TOOL)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_root_runs_the_server_as_the_units_account_and_refuses_without_one(self):
        tool = self.tool()
        settings = {"api_unit": "api.service"}
        with patch.object(os, "geteuid", return_value=1000), \
                patch.object(tool, "command", side_effect=AssertionError("not root")):
            self.assertEqual(tool.service_account(settings), {"cwd": "/"})
        account = SimpleNamespace(pw_uid=1234, pw_gid=5678)
        with patch.object(os, "geteuid", return_value=0), patch.object(tool, "command", return_value="svc"), \
                patch.object(pwd, "getpwnam", return_value=account) as lookup:
            become = tool.service_account(settings)["preexec_fn"]
            lookup.assert_called_once_with("svc")
            with patch.object(os, "setgroups") as groups, patch.object(os, "setgid") as gid, \
                    patch.object(os, "setuid") as uid, patch.object(os, "umask") as umask:
                become()
            groups.assert_called_once_with([])
            gid.assert_called_once_with(5678)
            uid.assert_called_once_with(1234)
            umask.assert_called_once_with(0o077)
        with patch.object(os, "geteuid", return_value=0), patch.object(tool, "command", return_value=""):
            with self.assertRaisesRegex(tool.Refused, "no service account"):
                tool.service_account(settings)

    def test_rewriting_the_cloud_config_keeps_its_owner(self):
        tool = self.tool()
        before = self.cloud.stat()
        with patch.object(os, "chown") as chown:
            tool.pin_worker({"cloud_config": str(self.cloud)}, "sha256:new")
        self.assertEqual(chown.call_args.args[1:], (before.st_uid, before.st_gid))
        self.assertEqual(self.cloud.stat().st_mode & 0o777, 0o600)
        self.assertEqual(json.loads(self.cloud.read_text())["builds"]["worker_sha256"], "sha256:new")

    def test_deployment_example_is_complete_and_locks_are_under_the_registry(self):
        tool = self.tool()
        deploy = TOOL.parent
        example = json.loads((deploy / "deployment.example.json").read_text())
        self.assertEqual(set(example), set(tool.SETTINGS))
        registry = json.loads((deploy / "reads.example.json").read_text())["registry_root"]
        self.assertEqual(example["publisher_locks"], [f"{registry}/submissions/.publisher.lock",
                                                      f"{registry}/cloud-submissions/.publisher.lock"])
        for name in ("ml-expd.service", "ml-expd-builder.service"):
            self.assertNotIn("ml-expd/", (deploy / name).read_text())

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
