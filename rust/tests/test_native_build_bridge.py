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
from ml_exp_server.dockerfile_build import inspect_dockerfile
from ml_exp_server.projects.source_revisions import _tree_digest


class NativeBridge(unittest.TestCase):
    def test_preferred_acr_skips_dsw_but_never_overlaps_an_unknown_primary(self):
        from ml_exp_server.builds import acr_fallback
        from ml_exp_server.builds.builder_client import BuildTransportError
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = {'bundle_id': 'a' * 64}
            builder = SimpleNamespace(root=root, config={'acr_fallback': {'prefer': True}})
            with patch.object(acr_fallback, 'publish', return_value='image') as publish, \
                    patch('ml_exp_server.builds.pai_dsw.DswPublisher') as dsw, \
                    patch('ml_exp_server.builds.pai_build_recovery.observe_dsw_absence', return_value=None):
                self.assertEqual(acr_fallback.publish_runtime(builder, 'tag', root, receipt), 'image')
                dsw.assert_not_called()
                publish.assert_called_once()
                publish.reset_mock()
                (root / ('a' * 64 + '.dsw.json')).write_text('{"phase":"CREATED"}')
                dsw.return_value.publish.side_effect = BuildTransportError('PRIMARY_UNRESOLVED', {})
                with self.assertRaises(BuildTransportError):
                    acr_fallback.publish_runtime(builder, 'tag', root, receipt)
                publish.assert_not_called()

    def test_dsw_empty_poll_and_lost_start_ack_do_not_restart_or_clean_the_builder(self):
        from ml_exp_server.builds.pai_dsw import DswPublisher
        from ml_exp_server.builds.image_build_context import BUILD_PROGRESS
        for lost_start in (False, True):
            with self.subTest(lost_start=lost_start), tempfile.TemporaryDirectory() as directory:
                builder = DswPublisher.__new__(DswPublisher)
                builder.timeout = 1200
                builder.instance = 'dsw-owned'
                calls = []
                responses = iter(['', 'ML_EXPD_DSW_STATUS_JSON={"status":"RUNNING","log_offset":4}',
                                  'ML_EXPD_DSW_STATUS_JSON={"status":"SUCCEEDED","log_offset":8,"receipt":{"ok":true}}'])
                def cli(args, *_args, **_kwargs):
                    calls.append(args[-1])
                    if 'Popen' in args[-1]:
                        if lost_start:
                            raise subprocess.TimeoutExpired('owned-start', 45)
                        return ''
                    return next(responses)
                builder.cli = cli
                builder.log_output = lambda value: value
                token = BUILD_PROGRESS.set(Path(directory) / 'progress.json')
                try:
                    with patch('ml_exp_server.builds.pai_dsw.time.sleep'):
                        self.assertEqual(builder.build('/tmp/ml-expd-build-' + 'a' * 32, Path(directory)), {'ok': True})
                finally:
                    BUILD_PROGRESS.reset(token)
                self.assertEqual(sum('Popen' in value for value in calls), 1)
                self.assertEqual(len(calls), 4)

    def test_native_reconcile_returns_fresh_retry_evidence_without_dispatch(self):
        from ml_exp_server.builds.builder_client import BuildReceiptMissingError
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = {'operation': 'native-reconcile', 'runtime_id': 'runtime.' + 'a' * 64,
                       'project': 'demo', 'source_id': 'source.' + 'b' * 64,
                       'base_image': 'registry.example/python@sha256:' + 'c' * 64,
                       'dockerfile': 'Dockerfile', 'worker_sha256': 'sha256:' + 'd' * 64, 'mirror': False}
            frozen = {k:v for k,v in request.items() if k != 'operation'}
            (root / ('a' * 64 + '.native-definition.json')).write_text(json.dumps(frozen))
            proof = {'source': 'pai-dsw-reconcile.v1', 'bundle_id': 'a' * 64}
            with patch.object(native_image, 'published_image', return_value=None), \
                    patch('ml_exp_server.builds.build_recovery.publication_absence', return_value=proof) as observe:
                with self.assertRaises(BuildReceiptMissingError) as error:
                    native_image.request(SimpleNamespace(root=root,config={}), request)
                self.assertEqual(error.exception.retry_proof, proof)
                observe.assert_called_once()

    def test_buildkit_uses_local_engine_even_with_remote_docker_environment(self):
        from ml_exp_server.image_builder import ImageBuilder
        with tempfile.TemporaryDirectory() as directory:
            builder = ImageBuilder({"state_root": directory})
            with patch.dict("os.environ", {"DOCKER_HOST": "tcp://remote.example:2375"}), \
                    patch.object(builder, "_command", return_value="ok") as command:
                self.assertEqual(builder._docker(["version"], capture=True), "ok")
                self.assertEqual(command.call_args.args[0], [
                    "/usr/bin/docker", "--host", "unix:///var/run/docker.sock", "version"])

    def test_removed_remote_docker_configuration_does_not_build_locally(self):
        from ml_exp_server.image_builder import ImageBuilder
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(subprocess, "run", side_effect=AssertionError("must not start a builder")):
            with self.assertRaisesRegex(ValueError, "remote Docker publishing was removed"):
                ImageBuilder({"state_root": directory, "docker_host": "unix:///old-desktop.sock"})

    def test_removed_data_staging_route_never_dispatches_a_build(self):
        import os
        import threading
        from ml_exp_server.image_builder import BuilderServer, Handler
        from ml_exp_server.builds.builder_client import UnixConnection
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "builder.sock")
            with BuilderServer(path, Handler) as server:
                server.builder = SimpleNamespace(config={"client_uid": os.getuid()},
                    request=lambda _: self.fail("removed data route dispatched a build"))
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                connection = UnixConnection(path, timeout=5)
                try:
                    connection.request("POST", "/data-stage/create", body=b"{}")
                    self.assertEqual(connection.getresponse().status, 409)
                finally:
                    connection.close()
                    server.shutdown()
                    thread.join()

    def test_multistage_registry_origins_are_unrestricted_but_frozen_base_is_checked(self):
        for old_filter in [None, ["unrelated.example/"]]:
            for wrong_base in [False, True]:
                with self.subTest(old_filter=old_filter, wrong_base=wrong_base), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    tree = root / "sources/demo/pending/tree"
                    tree.mkdir(parents=True)
                    base = "docker.io/library/python@sha256:" + "2" * 64
                    text = "FROM ghcr.io/astral-sh/uv@sha256:" + "3" * 64 + " AS uv\nFROM " + base + "\nCOPY --from=uv /uv /usr/local/bin/uv\n"
                    (tree / "Dockerfile").write_text(text)
                    digest = _tree_digest(tree)
                    source_id = "source." + digest[7:]
                    source = tree.parent.with_name(source_id)
                    tree.parent.rename(source)
                    (source / "source.json").write_text(json.dumps({"project": "demo", "tree_digest": digest}))
                    for path in source.rglob("*"):
                        path.chmod(0o555 if path.is_dir() else 0o444)
                    source.chmod(0o555)
                    worker = root / "worker"
                    worker.write_bytes(b"owned fixture; never execute")
                    config = {"native_worker_bin": str(worker), "native_source_root": str(root / "sources"),
                              "allow_dockerfile_builds": True, "publisher": "buildkit", "repository": "registry.example/build"}
                    if old_filter is not None:
                        config["base_image_prefixes"] = old_filter
                    calls = []
                    def publish(tag, context):
                        self.assertIn(text, (context / "Dockerfile").read_text())
                        self.assertEqual((context / "ml-expd-build-internal/worker-rs").read_bytes(), worker.read_bytes())
                        calls.append(tag)
                        return "registry.example/build@sha256:" + "4" * 64
                    builder = SimpleNamespace(root=root, config=config, _publish_buildkit=publish)
                    value = {"operation": "native-build", "runtime_id": "runtime." + "1" * 64,
                             "project": "demo", "source_id": source_id, "base_image": base if not wrong_base else "other.example/base@sha256:" + "5" * 64,
                             "dockerfile": "Dockerfile", "worker_sha256": "sha256:" + hashlib.sha256(worker.read_bytes()).hexdigest(), "mirror": False}
                    with patch.object(native_worker, "freeze_features", return_value=[]), \
                            patch.object(subprocess, "run", side_effect=AssertionError("test must not contact a provider")):
                        if wrong_base:
                            with self.assertRaisesRegex(ValueError, "base differs from the frozen request"):
                                native_image.request(builder, value)
                            self.assertEqual(calls, [])
                            self.assertFalse((root / ("1" * 64 + ".native-definition.json")).exists())
                        else:
                            result = native_image.request(builder, value)
                            self.assertEqual(result["runtime"]["source_id"], source_id)
                            self.assertEqual(native_image.request(builder, value), result)
                            self.assertEqual(len(calls), 1)

    def test_standard_dockerfile_instructions_and_literal_shift_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = "FROM docker.io/library/python@sha256:" + "1" * 64 + "\n"
            for instruction in ["ADD https://example.org/data.tar /tmp/", "ONBUILD RUN echo child",
                                "RUN echo 'literal << text'", 'RUN echo "literal << text"',
                                r"RUN echo \<\<", 'RUN ["python", "-c", "print(1 << 2)"]',
                                "# << in a comment", "RUN true # << in a shell comment"]:
                with self.subTest(instruction=instruction):
                    (root / "Dockerfile").write_text(base + instruction + "\n")
                    self.assertEqual(inspect_dockerfile(root, "Dockerfile")["base_images"], [base[5:].strip()])
            for instruction in ["RUN <<EOF\necho hello\nEOF", "RUN python <<'PY'\nprint('hello')\nPY", "COPY <<-EOF /tmp/file\nhello\nEOF"]:
                with self.subTest(instruction=instruction):
                    (root / "Dockerfile").write_text(base + instruction + "\n")
                    with self.assertRaisesRegex(ValueError, "heredocs.*Dockerfile parser"):
                        inspect_dockerfile(root, "Dockerfile")

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
            builder = SimpleNamespace(root=root, config={"repository": source, "native_worker_bin": str(worker), "base_image_prefixes": ["unrelated.example/"],
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
