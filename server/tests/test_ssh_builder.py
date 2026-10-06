"""Remote Docker never uses local capacity or falls back to a different engine."""
import json
from contextlib import nullcontext

import pytest

from ml_exp_server.image_builder import ImageBuilder, BuildStorageError


def config(tmp_path):
    return {'state_root': str(tmp_path), 'publisher': 'buildkit', 'ephemeral_buildkit': True,
            'docker_host': 'unix:///run/private/docker.sock', 'buildkit_image': 'moby/buildkit@sha256:'+'a'*64}


@pytest.mark.parametrize('change', [
    {'docker_host': None}, {'docker_host': 42}, {'docker_host': 'tcp://public:2375'}, {'docker_host': 'unix:///run/../other.sock'},
    {'publisher': 'archive'}, {'ephemeral_buildkit': False}, {'build_storage_path': '/local'},
    {'buildkit_image': 'moby/buildkit:latest'},
])
def test_remote_configuration_cannot_disable_real_storage_check(tmp_path, change):
    with pytest.raises(ValueError, match='remote Docker'):
        ImageBuilder({**config(tmp_path), **change})


def test_docker_calls_pin_remote_socket(tmp_path):
    value = ImageBuilder(config(tmp_path))
    seen = []
    value._command = lambda command, **kw: seen.append((command, kw)) or 'remote'
    assert value._docker(['version'], capture=True) == 'remote'
    assert seen == [(['/usr/bin/docker', '--host', 'unix:///run/private/docker.sock', 'version'], {'capture': True})]


@pytest.mark.parametrize('proxy', [None, 42, 'http://secret:password@proxy:3128', 'invalid'])
def test_proxy_credentials_are_rejected_before_commands(tmp_path, proxy):
    with pytest.raises(ValueError, match='credential-free'): ImageBuilder({**config(tmp_path), 'buildkit_http_proxy': proxy})


@pytest.mark.parametrize('network', [None, 42, '', 'name,env.SECRET=bad'])
def test_network_driver_options_cannot_be_injected(tmp_path, network):
    with pytest.raises(ValueError, match='network'): ImageBuilder({**config(tmp_path), 'buildkit_network': network})


@pytest.mark.parametrize('result', ['100 4096 200000', '0 4096 0', 'garbage', '-1 4096 100', '1 0 100', '1 4096 -1', 'disconnected'])
def test_remote_capacity_uses_owned_volume_and_always_cleans_it(tmp_path, monkeypatch, result):
    value = ImageBuilder(config(tmp_path))
    monkeypatch.setattr('ml_exp_server.image_builder.os.statvfs', lambda *a: (_ for _ in ()).throw(AssertionError('local disk')))
    seen = []
    def docker(args, **kw):
        seen.append(args)
        if args[0] == 'ps': return ''
        assert args[0] == 'run' and kw['capture']
        assert '--pull=never' in args and 'type=volume,target=/probe' in args
        assert '--network=none' in args and '--read-only' in args and '--cap-drop=ALL' in args
        assert json.loads((tmp_path/'storage-probe.json').read_text())['docker_host'] == value._endpoint()
        if result == 'disconnected': raise ValueError('private transport error')
        return result
    value._docker = docker
    if result in {'100 4096 200000', '0 4096 0'}:
        blocks, size, inodes = map(int, result.split())
        assert value._storage_available(None) == (blocks*size, inodes)
    else:
        with pytest.raises(ValueError): value._storage_available(None)
    assert seen[-1][0] == 'ps' and not (tmp_path/'storage-probe.json').exists()


@pytest.mark.parametrize('case', ['exists', 'gone', 'bad-name', 'other-endpoint', 'cleanup-failed'])
def test_probe_recovery_is_exact_and_endpoint_bound(tmp_path, monkeypatch, case):
    name = 'ml-expd-storage-'+'a'*32
    path = tmp_path/'storage-probe.json'
    path.write_text(json.dumps({'name': 'default' if case == 'bad-name' else name,
                               'docker_host': 'unix:///different.sock' if case == 'other-endpoint' else config(tmp_path)['docker_host']}))
    seen = []
    def docker(self, args, **kw):
        seen.append(args)
        if args[0] == 'ps': return '' if case == 'gone' else name
        assert args == ['rm', '--force', '--volumes', name]
        if case == 'cleanup-failed': raise ValueError('offline')
        return ''
    monkeypatch.setattr(ImageBuilder, '_docker', docker)
    if case in {'bad-name', 'other-endpoint', 'cleanup-failed'}:
        with pytest.raises(ValueError): ImageBuilder(config(tmp_path))
        assert path.exists()
    else:
        ImageBuilder(config(tmp_path))
        assert not path.exists()
        assert len(seen) == (1 if case == 'gone' else 2)


def test_old_local_builder_cannot_be_cleaned_on_remote_engine(tmp_path, monkeypatch):
    path = tmp_path/'ephemeral-builder.json'
    path.write_text(json.dumps({'name': 'ml-expd-'+'b'*32}))
    monkeypatch.setattr(ImageBuilder, '_docker', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('wrong engine')))
    with pytest.raises(ValueError, match='endpoint changed'): ImageBuilder(config(tmp_path))
    assert path.exists()


@pytest.mark.parametrize('failure', [None, 'small', 'offline'])
def test_remote_preflight_precedes_large_pull_and_explicit_builder_creation(tmp_path, failure):
    value = ImageBuilder({**config(tmp_path), 'buildkit_http_proxy': 'http://http.docker.internal:3128', 'buildkit_network': 'private-builder'})
    (tmp_path/'Dockerfile').write_text('FROM approved/base@sha256:'+'b'*64+'\n')
    value._skopeo = lambda *a, **kw: json.dumps({'layers': [{'size': 100}]})
    def available(path):
        assert path is None
        if failure == 'offline': raise ValueError('private desktop transport')
        return (1 if failure == 'small' else 10**12, 1000000)
    value._storage_available = available
    seen = []
    def docker(args, **kw):
        seen.append(args)
        if args[1] == 'ls': return json.loads((tmp_path/'ephemeral-builder.json').read_text())['name']
        assert args[-1] == config(tmp_path)['docker_host']
        return ''
    value._docker = docker
    value._buildkit_image = lambda *a: 'verified'
    value._source_context = lambda context: nullcontext()
    if failure:
        with pytest.raises(BuildStorageError) as exc: value._publish_buildkit('registry/test', tmp_path)
        assert exc.value.code == ('BUILD_STORAGE_INSUFFICIENT' if failure == 'small' else 'BUILD_STORAGE_UNCHECKED')
        assert not seen and not (tmp_path/'ephemeral-builder.json').exists()
    else:
        # rm has a name rather than an endpoint as its last argument.
        value._remove_builder = lambda name: seen.append(['cleaned', name])
        assert value._publish_buildkit('registry/test', tmp_path) == 'verified'
        assert seen[0][:2] == ['buildx', 'create'] and seen[0][-1] == config(tmp_path)['docker_host']
        assert 'env.HTTP_PROXY=http://http.docker.internal:3128' in seen[0]
        assert 'network=private-builder' in seen[0]
        assert seen[-1][0] == 'cleaned' and not (tmp_path/'ephemeral-builder.json').exists()


def test_data_context_uses_private_network_and_counts_remote_payload(tmp_path):
    from ml_exp_server.image_builder import BUILD_REMOTE_CONTEXT, BUILD_CONTEXT_BYTES
    value=ImageBuilder({**config(tmp_path),'data_upload_container':'ml-expd-data-stage'})
    (tmp_path/'Dockerfile').write_text('FROM registry/base@sha256:'+'b'*64+'\n')
    value._skopeo=lambda *a,**kw:json.dumps({'layers':[{'size':100}]})
    value._storage_available=lambda path:(10**12,1000000)
    calls=[];value._docker=lambda args,**kw:calls.append(args) or ''
    value._remove_builder=lambda name:None
    value._buildkit_image=lambda *a:'verified'
    url=BUILD_REMOTE_CONTEXT.set('http://ml-expd-data-stage:8080/context.tar')
    size=BUILD_CONTEXT_BYTES.set(5*1024**3)
    try:
        assert value._publish_buildkit('registry/data',tmp_path)=='verified'
        assert 'env.NO_PROXY=ml-expd-data-stage' in calls[0]
    finally:
        BUILD_REMOTE_CONTEXT.reset(url);BUILD_CONTEXT_BYTES.reset(size)
