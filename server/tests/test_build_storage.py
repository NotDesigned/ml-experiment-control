"""No huge pull before storage checks; public diagnostics contain no credentials."""
import errno
import json
import subprocess
from types import SimpleNamespace

import pytest

from ml_exp_server.image_builder import ImageBuilder, BuildStorageError, BUILD_LOG
from tests.test_image_builder_boundary import builder
from tests.test_container_api import client, import_source, legacy_prepare, wait_runtime


@pytest.mark.parametrize('case', ['enough', 'bytes', 'inodes', 'index', 'bad-index', 'manifest', 'stat'])
def test_storage_checks_actual_builder_mount_before_creating_builder(tmp_path, monkeypatch, case):
    value = ImageBuilder({'state_root': str(tmp_path), 'build_storage_path': '/actual-docker-state',
                          'ephemeral_buildkit': True, 'buildkit_image': 'moby/buildkit@sha256:'+'a'*64,
                          'build_reserve_bytes': 1000, 'build_min_free_inodes': 10})
    (tmp_path/'Dockerfile').write_text('FROM registry/base@sha256:'+'b'*64+'\nFROM scratch AS other\n')
    seen = []
    def skopeo(args, **kwargs):
        seen.append(args)
        if case == 'manifest': return '{}'
        if case in {'index', 'bad-index'} and len(seen) == 1:
            return json.dumps({'manifests': [] if case == 'bad-index' else [
                {'digest': 'sha256:'+'d'*64, 'platform': {'os': 'linux', 'architecture': 'arm64'}},
                {'digest': 'sha256:'+'c'*64, 'platform': {'os': 'linux', 'architecture': 'amd64'}}]})
        return json.dumps({'layers': [{'size': 100}]})
    value._skopeo = skopeo
    def stats(path):
        assert path == '/actual-docker-state'
        if case == 'stat': raise OSError('private filesystem')
        return SimpleNamespace(f_bavail=1 if case == 'bytes' else 10000, f_frsize=1,
                               f_favail=1 if case == 'inodes' else 100)
    monkeypatch.setattr('ml_exp_server.image_builder.os.statvfs', stats)
    value._docker = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('must not create a builder'))
    if case in {'enough', 'index'}:
        value._storage_preflight(tmp_path)
        assert len(seen) == (2 if case == 'index' else 1)
    else:
        with pytest.raises(BuildStorageError) as caught:
            value._publish_buildkit('registry/runtime:bundle', tmp_path)
        assert caught.value.code == ('BUILD_STORAGE_INSUFFICIENT' if case in {'bytes', 'inodes'} else 'BUILD_STORAGE_UNCHECKED')
        assert not (tmp_path/'ephemeral-builder.json').exists()
        assert 'private' not in str(caught.value)
        if case in {'bytes', 'inodes'}:
            assert caught.value.details['required_bytes'] > 1400
            assert caught.value.details['scheduler_submitted'] is False


@pytest.mark.parametrize('case', ['logged-enospc', 'errno-enospc', 'other', 'missing-log'])
def test_runtime_disk_exhaustion_is_distinct_and_never_echoes_log(tmp_path, monkeypatch, case):
    value = ImageBuilder({'state_root': str(tmp_path)})
    log = tmp_path/'build.log'
    if case != 'missing-log':
        log.write_text('secret-test-token '+('no space left on device' if case == 'logged-enospc' else 'registry unavailable'))
    token = BUILD_LOG.set(log)
    if case == 'logged-enospc':
        def failed(*args, **kwargs):
            kwargs['stdout'].write(b'no space left on device secret-current-token\n')
            kwargs['stdout'].flush()
            raise subprocess.CalledProcessError(1, 'private')
        monkeypatch.setattr('ml_exp_server.image_builder.subprocess.run', failed)
    if case == 'errno-enospc':
        monkeypatch.setattr('ml_exp_server.image_builder.subprocess.run', lambda *a, **kw: (_ for _ in ()).throw(OSError(errno.ENOSPC, 'private')))
    try:
        with pytest.raises(ValueError) as caught:
            value._command(['/bin/false'], capture=case != 'logged-enospc')
        if case in {'logged-enospc', 'errno-enospc'}:
            assert caught.value.code == 'BUILD_DISK_EXHAUSTED'
        assert 'secret-test-token' not in str(caught.value) and 'private' not in str(caught.value)
    finally: BUILD_LOG.reset(token)


def test_old_disk_error_cannot_misclassify_a_new_registry_failure(tmp_path):
    value = ImageBuilder({'state_root': str(tmp_path)})
    log = tmp_path/'build.log'; log.write_text('no space left on device\n')
    token = BUILD_LOG.set(log)
    try:
        with pytest.raises(ValueError) as caught: value._command(['/bin/false'])
        assert not isinstance(caught.value, BuildStorageError)
    finally: BUILD_LOG.reset(token)


def test_structured_builder_storage_failure_survives_runtime_api(client, monkeypatch):
    from ml_exp_server import container_execution
    ready = container_execution.builder_request
    def unavailable(*args, **kwargs):
        raise BuildStorageError('BUILD_STORAGE_INSUFFICIENT', {'available_bytes': 7, 'required_bytes': 20})
    monkeypatch.setattr('ml_exp_server.container_execution.builder_request', unavailable)
    source = import_source(client)
    value = legacy_prepare(client, {'source_id': source['source_id'],
        'image': 'registry/base@sha256:'+'a'*64, 'entrypoint': ['python3']}).json()
    endpoint = '/api/projects/demo/runtimes/'+value['runtime_id']
    client.post(endpoint+'/execute', json={'confirmation': value['confirmation']})
    result = wait_runtime(client, endpoint)
    assert result['error'] == 'BUILD_STORAGE_INSUFFICIENT'
    assert result['build_error']['details'] == {'available_bytes': 7, 'required_bytes': 20}
    assert result['build_error']['retry_safe'] is True
    monkeypatch.setattr('ml_exp_server.api.container_routes.builder_request', lambda *a: {'progress': {'events': []}})
    assert client.get(endpoint+'/progress').json()['build_error'] == result['build_error']
    monkeypatch.setattr('ml_exp_server.container_execution.builder_request', ready)
    client.post(endpoint+'/execute', json={'confirmation': value['confirmation']})
    result = wait_runtime(client, endpoint)
    assert result['status'] == 'READY' and result['build_error'] is None


def test_ready_receipt_reuse_does_not_require_rebuild_space(builder):
    value, request, _ = builder
    result = value.request(request)
    value._publish_buildkit = lambda *a: (_ for _ in ()).throw(AssertionError('must reuse ready receipt'))
    value.config['build_storage_path'] = '/missing-storage'
    assert value.request(request) == result
