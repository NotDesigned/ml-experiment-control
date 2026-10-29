"""Native builder effect-boundary tests; no provider, registry or GPU writes.

Run with the existing builder dependencies and candidate Python source on
PYTHONPATH. All payloads and executable fixtures are test-owned temporary files.
"""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ml_exp_server.builds import native_data, native_image
from ml_exp_server.data import registry_transport


class NativeBridge(unittest.TestCase):
    def test_late_primary_or_fallback_publication_reconciles_without_starting_builder(self):
        for labels_changed in [False, True]:
            with self.subTest(labels_changed=labels_changed), tempfile.TemporaryDirectory() as directory:
                builder = SimpleNamespace(root=Path(directory), config={"repository": "registry.example/team/build"})
                value = {"operation": "native-get", "runtime_id": "runtime." + "1" * 64,
                         "project": "demo", "source_id": "source." + "2" * 64,
                         "base_image": "registry.example/base@sha256:" + "3" * 64,
                         "dockerfile": "Dockerfile", "worker_sha256": "sha256:" + "4" * 64,
                         "mirror": False}
                frozen = {k: v for k, v in value.items() if k != "operation"}
                (builder.root / ("1" * 64 + ".native-definition.json")).write_text(json.dumps(frozen))
                raw = json.dumps({"mediaType": registry_transport.MANIFEST, "config": {"digest": "sha256:" + "5" * 64}}).encode()
                labels = native_image.recipe_labels(value)
                if labels_changed:
                    labels["org.ml-expd.worker"] = "sha256:" + "6" * 64
                registry = SimpleNamespace(prefix="/v2/team/build", request=lambda *a, **k: (200, {}, raw),
                                           json_blob=lambda _: {"os": "linux", "architecture": "amd64", "config": {"Labels": labels}})
                with patch.object(registry_transport.Registry, "from_config", return_value=registry):
                    if labels_changed:
                        with self.assertRaisesRegex(ValueError, "recipe, source, worker"):
                            native_image.request(builder, value)
                        self.assertFalse((builder.root / ("1" * 64 + ".native.json")).exists())
                    else:
                        result = native_image.request(builder, value)
                        self.assertEqual(result["runtime"]["image"], builder.config["repository"] + "@sha256:" + hashlib.sha256(raw).hexdigest())
                        self.assertEqual(native_image.request(builder, value), result)
                        self.assertEqual(result["request"], frozen)

    def test_absent_publication_never_dispatches_a_replacement(self):
        builder = SimpleNamespace(config={"repository": "registry.example/team/build"})
        registry = SimpleNamespace(prefix="/v2/team/build", request=lambda *a, **k: (404, {}, b""))
        with patch.object(registry_transport.Registry, "from_config", return_value=registry):
            self.assertIsNone(native_image.published_image(builder, "bundle-" + "1" * 64, {}))

    def test_data_staging_verifies_control_and_excludes_capability_from_context(self):
        for broken in [False, True]:
            with self.subTest(broken=broken), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                # This local executable stands in only for the already tested
                # native importer. The stage program and its tar are real.
                worker = b'#!/bin/sh\nprintf "verified dataset fixture" > "$5"\n'
                with tarfile.open(root / "context.tar", "w") as archive:
                    info = tarfile.TarInfo("worker-rs")
                    info.size = len(worker)
                    info.mode = 0o500
                    archive.addfile(info, io.BytesIO(worker))
                sha = hashlib.sha256((root / "context.tar").read_bytes()).hexdigest()
                (root / "request.json").write_text(json.dumps({"context_sha256": "0" * 64 if broken else sha}))
                (root / "download.json").write_text('synthetic-capability-must-not-enter-image')
                script = native_data.stage_script(str(root), "sha256:" + hashlib.sha256(worker).hexdigest(), 1)
                result = subprocess.run([sys.executable, "-O", "-c", script], capture_output=True)
                status = json.loads((root / "data-stage-status.json").read_text())
                self.assertFalse((root / "download.json").exists())
                if broken:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(status, {"status": "FAILED"})
                else:
                    self.assertEqual(result.returncode, 0, result.stderr.decode())
                    self.assertEqual(status["status"], "SUCCEEDED")
                    self.assertEqual(status["control"], sha)
                    self.assertEqual(status["materialized"], hashlib.sha256((root / "context.tar").read_bytes()).hexdigest())
                    with tarfile.open(root / "context.tar") as archive:
                        self.assertEqual(archive.getnames(), ["worker-rs", "archive.tar"])
                        self.assertNotIn(b"synthetic-capability", archive.extractfile("archive.tar").read())


if __name__ == "__main__":
    unittest.main()
