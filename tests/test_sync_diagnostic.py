import fcntl
import hashlib
import importlib.util
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "rust/crates/server/src/backends/cloud/service/engine/sync_diagnostic.py"
spec = importlib.util.spec_from_file_location("sync_diagnostic", SCRIPT)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)
# Also read by the Rust test in service/diagnostics.rs, so both sides pin one format.
PINNED_LINE = ('ML_EXPD_DIAGNOSTIC_PART=265ee923a61584279face2967b07f430b70f20d8d36ab06bc1ccf8d66cde6446'
               ':0:1:{"contract":"sync-diagnostic.v1"}')


def lines_of(stdout):
    return stdout.splitlines()


def reassembled(stdout):
    """Read a report back from its log lines the way the API does."""
    chunks = {}
    for line in lines_of(stdout):
        digest, index, count, chunk = line.split(diagnostic.PART_PREFIX, 1)[1].split(":", 3)
        chunks[int(index)] = chunk
    text = "".join(chunks[i] for i in range(int(count)))
    if hashlib.sha256(text.encode()).hexdigest() != digest:
        raise AssertionError("report digest does not match its parts")
    return json.loads(text)


class ReadonlyDiagnostic(unittest.TestCase):
    def run_script(self, config):
        result = subprocess.run([sys.executable, str(SCRIPT), json.dumps(config)],
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stderr, "")
        for line in lines_of(result.stdout):
            self.assertLessEqual(len(line), 2048 - 64, "keep room for a provider timestamp")
            self.assertTrue(line.startswith(diagnostic.PART_PREFIX))
            self.assertEqual(line, line.strip())
        return result.stdout

    def test_a_short_report_is_this_exact_line_which_the_api_test_also_reads(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            diagnostic.emit('{"contract":"sync-diagnostic.v1"}')
        self.assertEqual(stream.getvalue(), PINNED_LINE + "\n")

    def test_a_report_at_the_cap_fits_nineteen_lines_inside_the_log_tail(self):
        stream = io.StringIO()
        report = {"contract": diagnostic.CONTRACT, "filler": "x" * 40000}
        text = diagnostic.encoded(report)[:diagnostic.MAX_REPORT_BYTES]
        with redirect_stdout(stream):
            diagnostic.emit(text)
        lines = lines_of(stream.getvalue())
        self.assertEqual(len(lines), 19)
        self.assertTrue(all(len(line) < 2048 for line in lines))
        parts = [line.split(":", 3)[3] for line in lines]
        self.assertEqual("".join(parts), text)

    def test_text_has_no_whitespace_for_a_log_store_to_trim_and_means_the_same(self):
        report = {"path": "a b\tc\nd", "name": "\u4e2d\u6587"}
        text = diagnostic.encoded(report)
        self.assertFalse(any(c.isspace() for c in text))
        self.assertEqual(json.loads(text), report)

    def test_128_data_parts_stay_within_the_line_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            definition = {"archive_bytes": 128 * 16777216, "files": []}
            config = {"contract": "data-diagnostic.v1", "project": "demo", "asset_id": "asset." + "a" * 64,
                      "request_id": "b" * 32, "root": tmp, "definition": definition, "target": {}}
            stdout = self.run_script(config)
            self.assertGreater(len(lines_of(stdout)), 1)
            report = reassembled(stdout)
            self.assertEqual(len(report["parts"]), 128)
            self.assertEqual(report["request_id"], config["request_id"])

    def sync_config(self, root, required):
        package = {
            "config": {"attempt_root": root, "output_directory": root + "/output",
                       "local_binding": {"sha256": "0" * 64}},
            "manifest": {}, "artifacts": {"required_files": required},
        }
        return {"contract": diagnostic.CONTRACT, "action": "action-0123456789abcdef",
                "request_id": "a" * 32, "package": package, "package_sha256": "sha256:package",
                "config_sha256": "sha256:config", "config_binding_sha256": "binding",
                "channel": {"url": "http://127.0.0.1:1/api/worker-exchange/demo/run/attempt", "token": "t"}}

    def test_many_required_files_stay_within_the_line_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            required = ["outputs/file {:05}.bin".format(n) for n in range(200)]
            stdout = self.run_script(self.sync_config(tmp, required))
            self.assertGreater(len(lines_of(stdout)), 1)
            report = reassembled(stdout)
            paths = [row["path"] for row in report["checks"]]
            self.assertIn("output/outputs/file 00000.bin", paths)
            self.assertIn("output/outputs/file 00199.bin", paths)
            self.assertNotIn(tmp, stdout)

    def test_too_many_required_files_report_their_size_on_one_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            required = ["outputs/file-{:05}.bin".format(n) for n in range(2000)]
            stdout = self.run_script(self.sync_config(tmp, required))
            self.assertEqual(len(lines_of(stdout)), 1)
            self.assertEqual(reassembled(stdout), {"contract": diagnostic.CONTRACT, "action": "action-0123456789abcdef",
                                                   "request_id": "a" * 32, "error": "REPORT_TOO_LARGE"})

    def test_missing_data_root_reports_safe_errno_and_original_lineage_instead_of_hanging(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {"contract": "data-diagnostic.v1", "project": "demo", "asset_id": "asset." + "a" * 64,
                      "request_id": "b" * 32, "root": tmp + "/absent-private-path", "token": "synthetic-secret"}
            result = subprocess.run([sys.executable, str(SCRIPT), json.dumps(config)], capture_output=True, text=True, check=True)
            report = reassembled(result.stdout)
            self.assertEqual(report["contract"], config["contract"])
            self.assertEqual(report["request_id"], config["request_id"])
            self.assertEqual(report["errno"], 2)
            self.assertNotIn(tmp, result.stdout + result.stderr)
            self.assertNotIn(config["token"], result.stdout + result.stderr)

    def test_data_diagnostic_preserves_committed_and_uncommitted_parts_and_reports_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset = "asset." + "a" * 64
            slot = root / "demo" / asset
            (slot / "parts").mkdir(parents=True)
            definition = {"archive_bytes": 16777217, "files": []}
            target = {"backend_id": "pai-h200", "namespace": "own-fixture"}
            marker = {"contract": "native-dataset.v1", "target": target,
                      "identity": [root.stat().st_dev + 1, root.stat().st_ino]}
            (root / ".native-data.json").write_text(json.dumps(marker))
            (slot / "state.json").write_text(json.dumps({"contract": "native-dataset.v1", "project": "demo", "asset_id": asset,
                "target": target, "definition": definition, "parts": {"0": {"bytes": 3}}, "ready": None}))
            (slot / "parts/0").write_bytes(b"part0")
            (slot / "parts/1").write_bytes(b"uncommitted")
            def snapshot():
                return {str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None,
                        p.stat().st_mode, p.stat().st_mtime_ns) for p in root.rglob("*")}
            before = snapshot()
            config = {"contract": "data-diagnostic.v1", "project": "demo", "asset_id": asset,
                      "request_id": "b" * 32, "root": str(root), "definition": definition, "target": target}
            with patch.object(diagnostic, "inspect", wraps=diagnostic.inspect) as reads:
                report = diagnostic.diagnose(config)
            self.assertEqual(before, snapshot())
            lock = "demo/" + asset + ".lock"
            self.assertFalse((root / lock).exists())
            self.assertIn(lock, [row["path"] for row in report["checks"]])
            self.assertEqual(report["recorded_parts"], 1)
            self.assertFalse(report["ready_recorded"])
            self.assertEqual([part["size"] for part in report["parts"]], [5, 11])
            self.assertTrue(all("sha256" not in part for part in report["parts"]))
            for call in reads.call_args_list:
                if "/parts/" in call.args[1]:
                    self.assertEqual(len(call.args), 2, "part contents must never be read")
            self.assertGreater(report["capacity"]["available_bytes"], 0)
            self.assertFalse(report["capacity"]["user_quota_observed"])
            self.assertEqual(report["checks"][0]["identity"], "MATCH")
            self.assertFalse(report["checks"][0]["device_matches"], "device differences across CPFS mounts are informational")
            self.assertTrue(report["checks"][0]["inode_matches"])
            self.assertNotIn(str(root), json.dumps(report))

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
