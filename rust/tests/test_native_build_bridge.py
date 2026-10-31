"""Native builder effect-boundary tests; no provider, registry or GPU writes.

Run with the existing builder dependencies and candidate Python source on
PYTHONPATH. All payloads and executable fixtures are test-owned temporary files.
"""
import hashlib
import fcntl
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

from ml_exp_server.builds import native_data, native_image, native_worker
from ml_exp_server.data import registry_transport


class NativeBridge(unittest.TestCase):
    def test_data_publication_mirrors_exact_digest_and_reuses_receipt_without_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker"
            worker.write_bytes(b"owned worker fixture; must never execute")
            raw = json.dumps({"mediaType": registry_transport.MANIFEST, "layers": []}).encode()
            digest = "sha256:" + hashlib.sha256(raw).hexdigest()
            source = "registry.example/team/build"
            target = "mirror.example/team/runtime"
            key = "1" * 64
            recipe = {"project": "demo", "executor": "sensecore-test", "definition": {"archive_bytes": 17},
                      "base_image": "registry.example/base@sha256:" + "2" * 64,
                      "worker_sha256": "sha256:" + hashlib.sha256(worker.read_bytes()).hexdigest()}
            value = {"operation": "native-data-get", "data_image_id": "data-image." + key,
                     "recipe": recipe, "download": {"project": "demo", "definition": recipe["definition"],
                        "target": {"backend_id": "sensecore-test"}, "source": {"kind": "dataset_parts"}}}
            builder = SimpleNamespace(root=root, config={"repository": source, "native_worker_bin": str(worker),
                "sensecore_mirror": {"source_repository": source, "target_repository": target}})
            (root / (key + ".native-data-definition.json")).write_text(json.dumps(recipe))
            calls = []
            def observed(method, path, **kwargs):
                calls.append((method, path))
                return 200, {}, raw
            registry = SimpleNamespace(prefix="/v2/team/runtime", request=observed)
            with patch.object(native_image, "published_image", return_value=source + "@" + digest) as published, \
                    patch.object(registry_transport.Registry, "from_config", return_value=registry), \
                    patch.object(subprocess, "run", side_effect=AssertionError("verified mirror must not copy or create a job")):
                result = native_data.request(builder, value)
                self.assertEqual(result["image"], target + "@" + digest)
                self.assertEqual(result["recipe"], recipe)
                self.assertEqual(native_data.request(builder, value), result)
                published.assert_called_once()
            self.assertEqual(calls, [("GET", "/v2/team/runtime/manifests/data-" + key)])

    def test_progress_requires_a_live_lock_and_never_creates_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / ("1" * 64 + ".lock")
            self.assertFalse(native_image._active(lock))
            self.assertEqual(list(root.iterdir()), [])
            lock.write_bytes(b"original lock evidence")
            with lock.open("rb") as holder:
                self.assertFalse(native_image._active(lock))
                fcntl.flock(holder, fcntl.LOCK_EX)
                self.assertTrue(native_image._active(lock))
                fcntl.flock(holder, fcntl.LOCK_UN)
            self.assertFalse(native_image._active(lock))
            self.assertEqual(lock.read_bytes(), b"original lock evidence")
            link = root / "foreign-lock"
            link.symlink_to(lock)
            self.assertFalse(native_image._active(link))

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

    def test_worker_features_are_frozen_against_exact_binary_and_never_reprobed_on_read(self):
        for declared in [None, ["storage-layout.v1"], ["storage-layout.v1", "shared-cache-identity.v2"]]:
            supported = declared is not None
            with self.subTest(declared=declared), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                binary = root / "worker"
                response = {"contract": "native-worker-capabilities.v1", "features": declared or []}
                binary.write_text("#!/bin/sh\n" + ("printf '%s' '" + json.dumps(response) + "'\n" if supported else "exit 2\n"))
                binary.chmod(0o500)
                expected = "sha256:" + hashlib.sha256(binary.read_bytes()).hexdigest()
                marker = root / "features.json"
                features = native_worker.freeze_features(binary, expected, marker)
                self.assertEqual(features, response["features"] if supported else [])
                binary.unlink()
                with patch.object(subprocess, "run", side_effect=AssertionError("read must not execute worker")):
                    self.assertEqual(native_worker.freeze_features(binary, expected, marker), features)
                    self.assertEqual(native_worker.load_features(marker, expected), features)
                    with self.assertRaisesRegex(ValueError, "binding differs"):
                        native_worker.load_features(marker, "sha256:" + "0" * 64)
                self.assertEqual(native_worker.load_features(root / "old-missing.json", expected), [])

    def test_late_publication_carries_only_original_worker_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builder = SimpleNamespace(root=root, config={"repository": "registry.example/team/build"})
            value = {"runtime_id": "runtime." + "1" * 64, "project": "demo",
                     "source_id": "source." + "2" * 64, "worker_sha256": "sha256:" + "3" * 64, "mirror": False}
            (root / ("1" * 64 + ".native-worker.json")).write_text(json.dumps({
                "worker_sha256": value["worker_sha256"], "features": ["storage-layout.v1"]}))
            with patch.object(subprocess, "run", side_effect=AssertionError("reconciliation must not probe replacement")):
                result = native_image._finish(builder, value, "registry.example/team/build@sha256:" + "4" * 64,
                                              value, root / "receipt.json")
            self.assertEqual(result["runtime"]["worker_features"], ["storage-layout.v1"])

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
