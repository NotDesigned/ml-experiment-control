import fcntl
import hashlib
import importlib.util
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "rust/crates/server/src/cloud_runs/service/sync_diagnostic.py"
spec = importlib.util.spec_from_file_location("sync_diagnostic", SCRIPT)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class ReadonlyDiagnostic(unittest.TestCase):
    def test_scoped_probe_uses_get_and_does_not_copy_remote_error_or_extra_fields(self):
        requests = []
        secret = "synthetic-private-token-must-not-appear"

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append((self.command, self.path, self.headers.get("Authorization")))
                self.send_response(200 if self.path.endswith("/good/status") else 409)
                self.end_headers()
                self.wfile.write(json.dumps({"contract": "sync-diagnostic-status.v1", "secret": secret,
                    "exchange": {"observed": True, "sequence": 12, "phase": "FINISHED", "complete": True},
                    "artifacts": {"prepared": False, "published": False, "released": False}}).encode())

            def log_message(self, *_):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            base = "http://127.0.0.1:" + str(server.server_port) + "/api/worker-exchange/demo/run/"
            good = diagnostic.transport({"url": base + "good", "token": secret})
            failed = diagnostic.transport({"url": base + "failed", "token": secret})
            self.assertEqual(good["exchange"]["sequence"], 12)
            self.assertEqual(failed, {"stage": "HTTP_ERROR", "http_status": 409})
            self.assertNotIn(secret, json.dumps([good, failed]))
            self.assertEqual([r[0] for r in requests], ["GET", "GET"])
            self.assertTrue(all(r[2] == "Bearer " + secret for r in requests))
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_reads_metadata_and_payload_sizes_without_mutating_or_creating_locks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "output").mkdir()
            (root / "output/final.pt").write_bytes(b"weights-not-metadata")
            (root / ".managed-launch.lock").write_text('{"binding_sha256":"binding"}\n')
            (root / ".cloud-artifacts").mkdir()
            (root / ".cloud-artifacts/owner").write_text("sha256:package")
            manifest = b'{"identity":{"run_id":"retained"}}'
            (root / "launch-manifest.json").write_bytes(manifest)
            package = {
                "config": {"attempt_root": str(root), "output_directory": str(root / "output"),
                           "local_binding": {"sha256": hashlib.sha256(manifest).hexdigest()}},
                "manifest": json.loads(manifest), "artifacts": {"required_files": ["final.pt", "absent.json"]},
            }
            config = {"contract": diagnostic.CONTRACT, "action": "action-0123456789abcdef",
                      "request_id": "a" * 32, "package": package, "package_sha256": "sha256:package",
                      "config_sha256": "sha256:config", "config_binding_sha256": "binding", "channel": {}}

            def snapshot():
                return {str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None, p.stat().st_mode) for p in root.rglob("*")}

            before = snapshot()
            with (root / ".managed-launch.lock").open("rb") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(diagnostic, "transport", return_value={"stage": "HTTP_OK"}):
                    report = diagnostic.diagnose(config)
            self.assertEqual(before, snapshot())
            self.assertFalse((root / ".cloud-artifacts/.lock").exists())
            checks = {row["path"]: row for row in report["checks"]}
            self.assertEqual(checks[".managed-launch.lock"]["lock"], "HELD")
            self.assertEqual(checks["launch-manifest.json"]["identity"], "MATCH")
            self.assertEqual(checks[".cloud-artifacts/owner"]["identity"], "MATCH")
            self.assertEqual(checks["output/absent.json"]["status"], "MISSING")
            self.assertEqual(checks["output/final.pt"]["size"], 20)
            self.assertNotIn("sha256", checks["output/final.pt"])
            self.assertNotIn(str(root), json.dumps(report))

    def test_symlinks_are_not_followed_and_malformed_configuration_is_not_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "outside").write_text("private")
            (root / "link").symlink_to(root / "outside")
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                row, value = diagnostic.inspect(fd, "link", 100)
                self.assertEqual(row["status"], "UNSAFE_PATH")
                self.assertIsNone(value)
                self.assertEqual(diagnostic.inspect(fd, "../outside", 100)[0]["status"], "UNSAFE_PATH")
            finally:
                os.close(fd)
            secret = "synthetic-private-token-must-not-appear"
            result = subprocess.run([sys.executable, str(SCRIPT), json.dumps({"token": secret})],
                                    capture_output=True, text=True, check=True)
            self.assertNotIn(secret, result.stdout + result.stderr)
            self.assertEqual(result.stderr, "")
            self.assertIn('"error":"DIAGNOSTIC_UNAVAILABLE"', result.stdout)


if __name__ == "__main__":
    unittest.main()
