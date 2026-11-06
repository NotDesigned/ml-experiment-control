"""Exercise the embedded data launcher without cloud or shared storage."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


LAUNCHER = (Path(__file__).parents[1] / "crates/core/src/cloud/materialize.py").read_text()
STORAGE = json.dumps({"mount": "/mnt/workspace", "runs": "/mnt/workspace/runs", "datasets": "/mnt/workspace/data"})


class MaterializationLauncher(unittest.TestCase):
    def test_stage_failure_never_prints_scoped_credentials_and_removes_partial_archive(self):
        for stage in ("data-import", "data-copy", "success"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                staging = root / "data"
                token = "dataset-ready-token-never-log-this"
                config = json.dumps({"definition": {"sha256": "private-manifest"}})
                worker = root / "worker"
                worker.write_text(
                    "#!" + sys.executable + "\n"
                    "import pathlib,sys\n"
                    "if sys.argv[1]=='data-import':\n"
                    "    pathlib.Path(sys.argv[-1]).write_bytes(b'partial or verified archive')\n"
                    "else:\n"
                    "    pathlib.Path(sys.argv[0]+'.copy-args').write_text('\\n'.join(sys.argv))\n"
                    "sys.exit(1 if sys.argv[1]==" + repr(stage) + " else 0)\n"
                )
                worker.chmod(0o700)
                script = LAUNCHER.replace('directory = "/ml-expd/data"', "directory = " + repr(str(staging)))
                result = subprocess.run(
                    [sys.executable, "-c", script, str(worker), config, json.dumps({"token": token}), STORAGE],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 0 if stage == "success" else 1)
                output = result.stdout + result.stderr
                for secret in (token, config, "private-manifest", "--channel-json", "Traceback"):
                    self.assertNotIn(secret, output)
                if stage != "success":
                    self.assertIn(stage.upper().replace("-", "_") + "_FAILED", output)
                self.assertFalse((staging / "archive.tar").exists())
                if stage == "success":
                    arguments = (root / "worker.copy-args").read_text().splitlines()
                    self.assertEqual(arguments[arguments.index("--storage-json") + 1], STORAGE)

    def test_existing_archive_or_nonprivate_directory_is_not_overwritten(self):
        for mode in (0o700, 0o755):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                root.chmod(mode)
                archive = root / "archive.tar"
                archive.write_bytes(b"preexisting")
                script = LAUNCHER.replace('directory = "/ml-expd/data"', "directory = " + repr(str(root)))
                result = subprocess.run(
                    [sys.executable, "-c", script, "/never-called", "{}", "{}", STORAGE],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 1)
                self.assertEqual(archive.read_bytes(), b"preexisting")
                self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
