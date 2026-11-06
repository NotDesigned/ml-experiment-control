#!/usr/bin/env python3
"""The probe's protocol and findings, with two real processes on one filesystem."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "tools/storage_probe/probe.py"
sys.path.insert(0, str(PROBE.parent))
import probe  # noqa: E402


class Probe(unittest.TestCase):
    def run_both(self, directory, output):
        children = [subprocess.Popen(
            [sys.executable, str(PROBE), "--dir", str(directory), "--role", role, "--timeout", "60"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**os.environ, "OUTPUT_DIR": str(output)}) for role in "ab"]
        results = []
        for child in children:
            stdout, stderr = child.communicate(timeout=120)
            self.assertEqual(child.returncode, 0, stderr)
            results.append(json.loads(stdout))
        return results

    def test_two_jobs_meet_and_report_what_they_saw(self):
        with tempfile.TemporaryDirectory() as temp:
            root, output = Path(temp, "shared"), Path(temp, "out")
            root.mkdir()
            output.mkdir()
            a, b = self.run_both(root, output)
            for result in (a, b):
                self.assertEqual(result["lock_held_by_a"], "refused")
                self.assertEqual(result["lock_held_by_b"], "refused")
                self.assertTrue(result["same_host"])
                self.assertIn("nodes", result["findings"], "one host proves nothing")
                self.assertEqual(result["findings"]["flock"], "excludes across clients")
                self.assertEqual(result["findings"]["inode"], "same")
                self.assertGreater(result["stat_per_second"], 0)
                self.assertTrue(result["command_chain"])
            self.assertTrue((output / "storage-probe-a.json").exists())
            self.assertTrue((output / "storage-probe-b.json").exists())
            leftovers = [path.name for path in (root / "ml-exp-storage-probe").iterdir()
                         if path.name.startswith("stat-")]
            self.assertEqual(leftovers, [], "the stat files are cleaned up")

    def base(self):
        same = {"inode": 1, "device": 1, "size": 1, "mtime_ns": 1, "ctime_ns": 1}
        return {"same_host": False, "lock_held_by_a": "refused", "lock_held_by_b": "refused",
                "same_file": {"mine": same, "theirs": dict(same)}, "euid": 1000, "stat_per_second": 5}

    def test_a_good_result_reads_as_good(self):
        findings = probe.verdict(self.base())
        self.assertNotIn("nodes", findings)
        self.assertEqual(findings["flock"], "excludes across clients")
        self.assertEqual(findings["root"], "runs as uid 1000")

    def test_a_lock_that_does_not_exclude_is_called_out_in_either_direction(self):
        for key in ("lock_held_by_a", "lock_held_by_b"):
            result = self.base()
            result[key] = "acquired"
            self.assertIn("DOES NOT EXCLUDE", probe.verdict(result)["flock"])

    def test_what_differs_between_clients_is_named(self):
        for key in ("inode", "device", "mtime_ns", "ctime_ns"):
            result = self.base()
            result["same_file"]["theirs"][key] = 2
            self.assertEqual(probe.verdict(result)[key], "DIFFERS")


if __name__ == "__main__":
    unittest.main()
