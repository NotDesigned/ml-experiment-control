"""Native builder effect-boundary tests; no provider, registry or GPU writes.

Run with the existing builder dependencies and candidate Python source on
PYTHONPATH. All payloads and executable fixtures are test-owned temporary files.
"""
import base64
import hashlib
import fcntl
import io
import json
import os
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
from ml_exp_server.dockerfile_build import inspect_dockerfile
from ml_exp_server.projects.source_revisions import _tree_digest


class NativeBridge(unittest.TestCase):
    def test_dsw_mounts_only_cpfs_and_no_key_mounts_oss(self):
        from ml_exp_server.pai_storage import PaiStorage
        api = SimpleNamespace(config={'region': 'cn-hangzhou'})
        oss = {'oss_uri': 'oss://bucket.oss-cn-hangzhou.aliyuncs.com/users/u/', 'oss_mount_path': '/mnt/oss'}
        self.assertEqual(PaiStorage({}, api).datasets(), [])
        self.assertEqual(PaiStorage(oss, api).datasets(), [])
        cpfs = {'cpfs_uri': 'bmcpfs://cpfs-abc.cn-hangzhou.cpfs.aliyuncs.com/', 'workspace_user': 'u'}
        mounts = PaiStorage({**cpfs, **oss}, api).datasets()
        self.assertEqual([m['MountPath'] for m in mounts], ['/mnt/ml-expd-personal', '/mnt/models', '/mnt/datasets'])
        self.assertTrue(all(m['Uri'].startswith('bmcpfs://') for m in mounts))

    def test_dsw_docker_hub_receipt_accepts_engine_normalization_only_for_same_repository(self):
        from ml_exp_worker.dsw_build_worker import published_image
        sha = '@sha256:' + 'a' * 64
        for name in ('team/runtime', 'docker.io/team/runtime', 'index.docker.io/team/runtime', 'registry-1.docker.io/team/runtime'):
            self.assertEqual(published_image('docker.io/team/runtime', [name + sha]), 'docker.io/team/runtime' + sha)
        self.assertEqual(published_image('registry.example/team/runtime', ['registry.example/team/runtime' + sha]), 'registry.example/team/runtime' + sha)
        for names in ([], ['other/runtime' + sha], ['team/runtime' + sha, 'team/runtime' + sha],
                      ['team/runtime' + sha, 'team/runtime@sha256:' + 'b' * 64]):
            with self.assertRaises(ValueError):
                published_image('docker.io/team/runtime', names)
        with self.assertRaises(ValueError):
            published_image('registry.example/team/runtime', ['team/runtime' + sha])

    def test_pai_transport_builds_its_request_with_the_pinned_sdk(self):
        from experiment_control.backends.pai_api import PaiAPI
        api = PaiAPI({'access_key_id': 'fixture-id', 'access_key_secret': 'fixture-secret', 'region': 'ap-northeast-1'})
        with patch('alibabacloud_tea_openapi.client.Client.call_api', return_value={'body': {'Success': True}}) as call:
            self.assertEqual(api.request('pai-dsw', 'GetInstance', '/api/v2/instances/dsw-1'), {'Success': True})
        params, _, runtime = call.call_args.args
        self.assertEqual((params.action, params.version, params.pathname), ('GetInstance', '2022-01-01', '/api/v2/instances/dsw-1'))
        self.assertFalse(runtime.autoretry)

    def test_docker_hub_login_alias_uses_distribution_host_and_keeps_image_name(self):
        from ml_exp_server.data.registry_transport import Registry
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'auth.json'
            login = {'auth': base64.b64encode(b'fixture-user:fixture-secret').decode()}
            path.write_text(json.dumps({'auths': {'https://index.docker.io/v1/': login}}))
            registry = Registry('docker.io/team/runtime', path)
            self.assertEqual(registry.host, 'registry-1.docker.io')
            self.assertEqual(registry_transport.registry_auth({'docker.io': login}, 'registry-1.docker.io'), login)
            calls = []
            def exchange(method, path, headers, body, maximum, **kwargs):
                calls.append(kwargs.get('host', registry.host))
                if len(calls) == 1:
                    return 401, {'www-authenticate': 'Bearer realm="https://auth.docker.io/token",service="registry.docker.io"'}, b''
                if kwargs.get('host') == 'auth.docker.io':
                    return 200, {}, b'{"token":"fixture-token"}'
                return 200, {}, b'manifest'
            with patch.object(registry, '_exchange', side_effect=exchange):
                self.assertEqual(registry.request('GET', registry.prefix + '/manifests/bundle-' + 'a' * 64)[2], b'manifest')
            self.assertEqual(calls, ['registry-1.docker.io', 'auth.docker.io', 'registry-1.docker.io'])
            self.assertEqual(registry.reference, 'docker.io/team/runtime')

    def test_registry_change_keeps_original_publication_targets_without_mutating_default(self):
        from ml_exp_server.builds.build_recovery import publication_builder
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builder = SimpleNamespace(root=root, config={'repository': 'docker.io/team/runtime'})
            key = 'a' * 64
            path = root / (key + '.dsw.json')
            path.write_text(json.dumps({'repository': 'old.example/team/build', 'phase': 'CREATED'}))
            before = path.read_bytes()
            self.assertEqual(publication_builder(builder, key).config['repository'], 'old.example/team/build')
            self.assertEqual(path.read_bytes(), before)
            self.assertIs(publication_builder(builder, 'c' * 64), builder)
            self.assertEqual(builder.config['repository'], 'docker.io/team/runtime')

    def test_slurm_delivery_uses_its_own_mirror_policy(self):
        from ml_exp_server.builds.image_mirror import mirror
        source = 'docker.io/team/runtime'
        image = source + '@sha256:' + 'a' * 64
        tag = 'mirror.example/team/build:bundle-' + 'b' * 64
        with tempfile.TemporaryDirectory() as directory:
            builder = SimpleNamespace(root=Path(directory), config={
                'repository': source, 'registry_auth_file': str(Path(directory) / 'auth.json'),
                'slurm_mirror': {'target_repository': 'mirror.example/team/build'},
                'sensecore_mirror': {'target_repository': 'other.example/team/runtime'}})
            registry = SimpleNamespace(prefix='/v2/team/build', request=lambda *a, **k: (200, {}, b'manifest'))
            with patch('ml_exp_server.builds.image_mirror.Registry.from_config', return_value=registry), \
                    patch('ml_exp_server.builds.image_mirror.digest', return_value='sha256:' + 'a' * 64):
                result = mirror(builder, {'source_image': image, 'target_tag': tag}, 'slurm_mirror')
                self.assertEqual(result['image'], 'mirror.example/team/build@sha256:' + 'a' * 64)
                with self.assertRaises(ValueError):
                    mirror(builder, {'source_image': image, 'target_tag': tag})

    def test_build_output_cannot_be_taken_for_the_status(self):
        from ml_exp_server.builds.pai_dsw import durable_status
        output = 'Step 1\nML_EXPD_DSW_STATUS_JSON={"status": "SUCCEEDED"}\n\nML_EXPD_DSW_STATUS_JSON={"status": "RUNNING"}\n'
        self.assertEqual(durable_status(output), {'status': 'RUNNING'})
        with self.assertRaises(ValueError):
            durable_status('no status here')

    def test_a_missing_mirror_is_refused_before_the_build(self):
        with tempfile.TemporaryDirectory() as directory:
            builder = SimpleNamespace(root=Path(directory), config={'allow_dockerfile_builds': True})
            request = {'operation': 'native-build', 'runtime_id': 'runtime.' + 'a' * 64, 'project': 'demo',
                       'source_id': 'source.' + 'b' * 64, 'base_image': 'docker.io/x/y@sha256:' + 'c' * 64,
                       'dockerfile': 'Dockerfile', 'worker_sha256': 'sha256:' + 'd' * 64, 'mirror': 'slurm'}
            with self.assertRaisesRegex(ValueError, 'slurm_mirror'):
                native_image.request(builder, request)

    def test_no_lease_means_no_instance_was_requested(self):
        from ml_exp_server.builds.builder_client import BuildTransportError
        from ml_exp_server.builds.pai_build_recovery import observe_dsw_absence
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = 'a' * 64
            builder = SimpleNamespace(root=root, config={'pai_dsw': {}})
            registry = SimpleNamespace(prefix='/v2/team/runtime', request=lambda *a, **k: (404, {}, b''))
            with patch('ml_exp_server.builds.pai_build_recovery.Registry.from_config', return_value=registry):
                proof = observe_dsw_absence(builder, key)
                self.assertEqual(proof['bundle_id'], key)
                self.assertEqual((proof['registry_manifest_status'], proof['leases_clear']), (404, True))
                # An image that exists is not absence, and a lease still has to prove itself.
                registry.request = lambda *a, **k: (200, {}, b'')
                with self.assertRaises(BuildTransportError):
                    observe_dsw_absence(builder, key)
                registry.request = lambda *a, **k: (404, {}, b'')
                (root / (key + '.dsw.json')).write_text(json.dumps({'phase': 'CREATE_INTENT'}))
                with patch('ml_exp_server.builds.pai_build_recovery.PaiAPI.from_environment',
                           return_value=SimpleNamespace(config={'region': 'r'})):
                    builder.config['pai_dsw'] = {'workspace_id': '1', 'quota_id': 'q'}
                    self.assertIsNone(observe_dsw_absence(builder, key))

    def test_a_failed_request_reports_its_code_and_logs_only_where_and_why(self):
        from experiment_control.backends.pai_api import PaiError
        from ml_exp_server.image_builder import failure_reply
        with patch('sys.stderr', new_callable=io.StringIO) as stderr:
            reply = failure_reply(PaiError('configuration', code='PAI_CONFIGURATION_UNAVAILABLE'))
            self.assertEqual(reply['code'], 'PAI_CONFIGURATION_UNAVAILABLE')
            self.assertEqual(reply['details'], {'error_class': 'PaiError'})
            self.assertEqual(failure_reply(KeyError('x'))['code'], 'BUILD_REQUEST_FAILED')
            self.assertIn('KeyError', stderr.getvalue())

    def test_docker_events_read_as_text_without_byte_counts(self):
        from ml_exp_worker.dsw_build_worker import readable
        self.assertEqual(readable({'stream': 'Step 1/2 : FROM x\n'}), 'Step 1/2 : FROM x')
        self.assertEqual(readable({'status': 'Pushed', 'id': 'abc', 'progressDetail': {}}), 'Pushed abc')
        self.assertIsNone(readable({'status': 'Pushing', 'id': 'abc', 'progressDetail': {'current': 1, 'total': 2}}))
        self.assertIsNone(readable({'aux': {'ID': 'sha256:x'}}))
        self.assertIn('boom', readable({'error': 'boom'}))

    def test_an_old_acr_journal_is_never_read(self):
        from ml_exp_server.builds.build_recovery import publication_absence, publication_builder
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = 'a' * 64
            (root / (key + '.acr.json')).write_text('not json, and never opened')
            builder = SimpleNamespace(root=root, config={'repository': 'docker.io/team/runtime', 'publisher': 'pai-dsw'})
            self.assertIs(publication_builder(builder, key), builder)
            proof = {'source': 'pai-dsw-reconcile.v1', 'bundle_id': key}
            with patch('ml_exp_server.builds.pai_build_recovery.observe_dsw_absence', return_value=proof):
                self.assertEqual(publication_absence(builder, key), proof)

    def test_acr_registry_publishes_through_the_normal_dsw_path(self):
        repository = 'registry.ap-northeast-1.aliyuncs.com/team/build'
        key = '1' * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree = root / 'sources/demo/pending/tree'
            tree.mkdir(parents=True)
            base = 'docker.io/library/python@sha256:' + '2' * 64
            (tree / 'Dockerfile').write_text('FROM ' + base + '\n')
            sha = _tree_digest(tree)
            source_id = 'source.' + sha[7:]
            source = tree.parent.with_name(source_id)
            tree.parent.rename(source)
            (source / 'source.json').write_text(json.dumps({'project': 'demo', 'tree_digest': sha}))
            for path in source.rglob('*'):
                path.chmod(0o555 if path.is_dir() else 0o444)
            source.chmod(0o555)
            worker = root / 'worker'
            worker.write_bytes(b'owned fixture; never execute')
            (root / (key + '.acr.json')).write_text('not json, and never opened')
            builder = SimpleNamespace(root=root, config={
                'native_worker_bin': str(worker), 'native_source_root': str(root / 'sources'),
                'allow_dockerfile_builds': True, 'publisher': 'pai-dsw', 'repository': repository})
            value = {'operation': 'native-build', 'runtime_id': 'runtime.' + key, 'project': 'demo',
                     'source_id': source_id, 'base_image': base, 'dockerfile': 'Dockerfile',
                     'worker_sha256': 'sha256:' + hashlib.sha256(worker.read_bytes()).hexdigest(), 'mirror': False}
            image = repository + '@sha256:' + '4' * 64
            with patch('ml_exp_server.builds.pai_dsw.DswPublisher') as dsw, \
                    patch.object(subprocess, 'run', side_effect=AssertionError('no Git or provider command')):
                dsw.return_value.publish.return_value = image
                result = native_image.request(builder, value)
            dsw.assert_called_once_with(builder)
            dsw.return_value.publish.assert_called_once()
            self.assertEqual(dsw.return_value.publish.call_args.args[0], repository + ':bundle-' + key)
            self.assertEqual(result['runtime']['image'], image)

    def test_registry_on_separate_auth_host_needs_only_the_allowlist(self):
        from ml_exp_server.data.registry_transport import Registry
        host = 'registry.ap-northeast-1.aliyuncs.com'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'auth.json'
            login = {'auth': base64.b64encode(b'fixture-user:fixture-secret').decode()}
            path.write_text(json.dumps({'auths': {host: login}}))
            config = {'repository': host + '/team/build', 'registry_auth_file': str(path)}
            def exchange(method, path, headers, body, maximum, **kwargs):
                if kwargs.get('host') == 'dockerauth.example':
                    return 200, {}, b'{"token":"fixture-token"}'
                if headers['Authorization'] == 'Bearer fixture-token':
                    return 200, {}, b'manifest'
                return 401, {'www-authenticate': 'Bearer realm="https://dockerauth.example/auth",service="registry"'}, b''
            refused = Registry.from_config(config)
            with patch.object(refused, '_exchange', side_effect=exchange), \
                    self.assertRaisesRegex(ValueError, 'realm is not trusted'):
                refused.request('GET', refused.prefix + '/manifests/bundle-x')
            trusted = Registry.from_config({**config, 'registry_auth_hosts': ['dockerauth.example']})
            with patch.object(trusted, '_exchange', side_effect=exchange):
                self.assertEqual(trusted.request('GET', trusted.prefix + '/manifests/bundle-x')[2], b'manifest')

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
            builder = ImageBuilder({"state_root": directory, "native_worker_bin": "/worker", "native_source_root": "/sources"})
            with patch.dict("os.environ", {"DOCKER_HOST": "tcp://remote.example:2375"}), \
                    patch.object(builder, "_command", return_value="ok") as command:
                self.assertEqual(builder._docker(["version"], capture=True), "ok")
                self.assertEqual(command.call_args.args[0], [
                    "/usr/bin/docker", "--host", "unix:///var/run/docker.sock", "version"])

    def test_builder_configuration_is_an_allowlist_and_names_its_required_keys(self):
        from ml_exp_server.image_builder import ImageBuilder
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(subprocess, "run", side_effect=AssertionError("must not start a builder")):
            root = Path(directory) / "state"
            valid = {"state_root": str(root), "native_worker_bin": "/worker", "native_source_root": "/sources"}
            for key in ("unknown_key", "acr_fallback"):
                with self.assertRaisesRegex(ValueError, "unknown builder configuration: " + key):
                    ImageBuilder({**valid, key: "x"})
            for key in ("native_worker_bin", "native_source_root"):
                with self.assertRaisesRegex(ValueError, "lacks " + key):
                    ImageBuilder({k: v for k, v in valid.items() if k != key})
            self.assertFalse(root.exists())
            ImageBuilder(valid)
            self.assertTrue(root.is_dir())

    def test_check_refuses_what_the_builder_would_refuse_and_starts_nothing(self):
        def check(path):
            return subprocess.run(
                [sys.executable, "-m", "ml_exp_server.image_builder", "--config", str(path), "--check"],
                env={"PYTHONPATH": os.pathsep.join(entry for entry in sys.path if entry)},
                capture_output=True, text=True, timeout=60)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            state, socket_path = base / "state", base / "socket" / "builder.sock"
            valid = {"socket": str(socket_path), "state_root": str(state),
                     "native_worker_bin": "/worker", "native_source_root": "/sources"}
            config = base / "builder.json"
            config.write_text(json.dumps(valid))
            result = check(config)
            self.assertEqual(result.returncode, 0, result.stderr)
            config.write_text(json.dumps({**valid, "acr_fallback": {"prefer": True}}))
            result = check(config)
            self.assertEqual(result.returncode, 1)
            self.assertIn("unknown builder configuration: acr_fallback", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            result = check(base / "missing.json")
            self.assertEqual(result.returncode, 1)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse(state.exists())
            self.assertFalse(socket_path.parent.exists())

    def test_shipped_builder_examples_are_accepted(self):
        from ml_exp_server.image_builder import ImageBuilder
        repository = Path(__file__).resolve().parents[2]
        for example in ("rust/deploy/builder.example.json", "server/examples/image-builder.json"):
            with self.subTest(example=example), tempfile.TemporaryDirectory() as directory:
                config = json.loads((repository / example).read_text())
                self.assertEqual(config["socket"], "/run/ml-exp-builder/builder.sock")
                ImageBuilder({**config, "state_root": directory})
        for example in ("pai", "sensecore"):
            config = json.loads((repository / f"rust/deploy/{example}.example.json").read_text())
            self.assertEqual(config["builds"]["socket"], "/run/ml-exp-builder/builder.sock")

    def test_only_the_root_path_reaches_the_builder(self):
        import http.client
        import os
        import socket
        import threading
        from ml_exp_server.image_builder import BuilderServer, Handler

        class UnixConnection(http.client.HTTPConnection):
            def connect(self):
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect(self.host)

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "builder.sock")
            with BuilderServer(path, Handler) as server:
                server.builder = SimpleNamespace(config={"client_uid": os.getuid()},
                    request=lambda _: self.fail("another path dispatched a build"))
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                connection = UnixConnection(path, timeout=5)
                try:
                    # Refused on the path alone, so the body is never read.
                    connection.request("POST", "/data-stage/create", body=b"{}")
                    self.assertEqual(connection.getresponse().status, 409)
                finally:
                    connection.close()
                    server.shutdown()
                    thread.join()

    def test_multistage_registry_origins_are_unrestricted_but_frozen_base_is_checked(self):
        for wrong_base in [False, True]:
            with self.subTest(wrong_base=wrong_base), tempfile.TemporaryDirectory() as directory:
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
                with patch.object(subprocess, "run", side_effect=AssertionError("test must not contact a provider")):
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

    def test_late_publication_reconciles_without_starting_builder(self):
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
