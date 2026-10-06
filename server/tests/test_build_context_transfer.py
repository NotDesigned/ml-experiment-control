"""Seal the actual archive, bypass WAN file sync, retain scoped recovery leases."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import threading
import urllib.request
from types import SimpleNamespace

import pytest

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.build_contexts import BuildContexts
from ml_exp_server.desktop_upload import DesktopUploads
from ml_exp_server.image_builder import ImageBuilder, BUILD_REMOTE_CONTEXT
from ml_exp_server import image_builder as module
from ml_exp_server import desktop_upload as desktop


@pytest.fixture
def transport(tmp_path):
    value = ImageBuilder({'state_root': str(tmp_path/'builder')})
    value.config.update(docker_host='unix:///run/private.sock', data_upload_container='ml-expd-data-stage')
    stage = DesktopUploads(tmp_path/'desktop', {})
    value.desktop_stage = lambda operation, data, body: {'ok': True, 'result': stage.call(operation, data, body)}
    def docker(arguments, **kwargs):
        assert arguments[0] == 'cp'
        suffix = arguments[2].removeprefix('ml-expd-data-stage:/stage/')
        shutil.copyfile(arguments[1], stage.root/suffix)
        return ''
    value._docker = docker
    source = tmp_path/'source'
    (source/'nested').mkdir(parents=True)
    (source/'Dockerfile').write_text('FROM approved/base\nCOPY . /workspace\n')
    (source/'nested/train.py').write_text('print("training")\n')
    return value, stage, source


def test_staging_seals_payload_and_cleans_only_owned_context(transport):
    value, stage, source = transport
    retained = stage.root/'assets/keep'
    retained.mkdir(parents=True); (retained/'data').write_text('keep')
    with value._source_context(source):
        url = BUILD_REMOTE_CONTEXT.get()
        assert url.startswith('http://ml-expd-data-stage:8080/build-contexts/context.')
        identity = json.loads((value.root/'source-context.json').read_text())['context_id']
        archive = BuildContexts(stage.root).archive(identity)
        with tarfile.open(archive, mode='r:gz') as tar:
            assert tar.extractfile('nested/train.py').read() == b'print("training")\n'
            assert tar.extractfile('Dockerfile').read() == (source/'Dockerfile').read_bytes()
    assert BUILD_REMOTE_CONTEXT.get() is None
    assert not (value.root/'source-context.json').exists()
    assert not list((stage.root/'build-contexts').iterdir())
    assert (retained/'data').read_text() == 'keep'
    assert not list(value.root.glob('source-context-*'))


def test_source_context_http_serves_only_sealed_archive(transport, monkeypatch):
    _, stage, _ = transport
    identity='context.'+'a'*32
    stage.call('context-create',{'context_id':identity,'bytes':3,'sha256':hashlib.sha256(b'abc').hexdigest()})
    monkeypatch.setattr(desktop,'store',lambda:stage)
    server=desktop.ThreadingHTTPServer(('127.0.0.1',0),desktop.ContextHandler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url='http://127.0.0.1:'+str(server.server_port)+'/build-contexts/'+identity+'.tar.gz'
    try:
        with pytest.raises(urllib.error.HTTPError):urllib.request.urlopen(url)
        path=BuildContexts(stage.root).path(identity)
        (path/'context.tar.gz').write_bytes(b'abc')
        stage.call('context-seal',{'context_id':identity})
        with urllib.request.urlopen(url) as response:
            assert response.read()==b'abc' and response.headers['Content-Length']=='3'
        stage.call('context-remove',{'context_id':identity})
        with pytest.raises(urllib.error.HTTPError):urllib.request.urlopen(url)
    finally:
        server.shutdown();server.server_close();thread.join()


@pytest.mark.parametrize('failure', ['copy', 'create', 'seal', 'sha', 'bytes', 'build'])
def test_staging_failure_cleans_exact_lease_and_resets_context(transport, failure):
    value, stage, source = transport
    original = value.desktop_stage
    if failure == 'copy':
        value._docker = lambda *a, **k: (_ for _ in ()).throw(ValueError('transport unavailable'))
    def stage_call(operation, data, body):
        if operation == 'context-'+failure:
            return {'ok': False}
        result = original(operation, data, body)
        if operation == 'context-seal' and failure in {'sha', 'bytes'}:
            result['result']['sha256' if failure=='sha' else 'bytes'] = 'wrong'
        return result
    value.desktop_stage = stage_call
    with pytest.raises(ValueError):
        with value._source_context(source):
            raise ValueError('build failed')
    assert BUILD_REMOTE_CONTEXT.get() is None
    assert not (value.root/'source-context.json').exists()
    assert not list(value.root.glob('source-context-*'))


def test_cleanup_failure_retains_recovery_record_then_retries_without_build(transport):
    value, stage, source = transport
    original = value.desktop_stage
    def call(operation, data, body):
        if operation == 'context-remove': return {'ok': False}
        return original(operation, data, body)
    value.desktop_stage = call
    with pytest.raises(ValueError, match='cleanup failed'):
        with value._source_context(source): pass
    assert (value.root/'source-context.json').exists()
    value.desktop_stage = original
    value._recover_source_context()
    assert not (value.root/'source-context.json').exists()


@pytest.mark.parametrize('change', [{'context_id':'../other'}, {'docker_host':'unix:///other.sock'}, {'container':'unrelated'}])
def test_recovery_rejects_changed_endpoint_or_identity(transport, change):
    value, _, _ = transport
    (value.root/'source-context.json').write_text(json.dumps({'context_id':'context.'+'a'*32,
        'docker_host':value._endpoint(), 'container':'ml-expd-data-stage', **change}))
    with pytest.raises(ValueError, match='recovery identity'):
        value._recover_source_context()
    assert (value.root/'source-context.json').exists()


@pytest.mark.parametrize('case', ['local', 'data', 'no-helper', 'symlink', 'special', 'large'])
def test_staging_preserves_data_context_and_rejects_unsafe_code(transport, monkeypatch, case):
    value, _, source = transport
    token = None
    if case=='local':value.config.pop('docker_host')
    if case=='data':token=BUILD_REMOTE_CONTEXT.set('http://data/context.tar')
    if case=='no-helper':value.config.pop('data_upload_container')
    if case=='symlink':(source/'link').symlink_to(source/'Dockerfile')
    if case=='special':
        import os
        os.mkfifo(source/'fifo')
    if case=='large':monkeypatch.setattr(module,'MAX_BYTES',1)
    try:
        if case in {'local','data'}:
            with value._source_context(source):assert BUILD_REMOTE_CONTEXT.get()==('http://data/context.tar' if case=='data' else None)
        else:
            with pytest.raises(ValueError):
                with value._source_context(source):pass
    finally:
        if token is not None:BUILD_REMOTE_CONTEXT.reset(token)


@pytest.mark.parametrize('case', ['metadata', 'timeout', 'socket', 'process', 'large', 'json', 'shape', 'ok-type'])
def test_rpc_failure_codes_are_transient_and_never_echo_process_output(tmp_path, monkeypatch, case):
    value = ImageBuilder({'state_root':str(tmp_path)})
    value.config.update(docker_host='unix:///run/private.sock', data_upload_container='ml-expd-data-stage')
    def run(command, **kwargs):
        assert '-i' not in command
        if case=='timeout':raise subprocess.TimeoutExpired('private-secret',1)
        if case=='socket':raise OSError('private-secret')
        stdout = b'{"ok":true,"result":{}}'
        if case=='large':stdout=b'x'*(8*1024**2+1)
        if case=='json':stdout=b'private-secret'
        if case=='shape':stdout=b'[]'
        if case=='ok-type':stdout=b'{"ok":"yes"}'
        return SimpleNamespace(returncode=1 if case=='process' else 0, stdout=stdout, stderr=b'private-secret')
    monkeypatch.setattr(module.subprocess,'run',run)
    if case=='metadata':
        assert value.desktop_stage('read',{},b'')=={'ok':True,'result':{}}
    else:
        with pytest.raises(ApplicationError) as error:value.desktop_stage('read',{},b'')
        assert error.value.status_code==503 and error.value.code.startswith('DESKTOP_RPC_')
        assert 'private-secret' not in str(error.value)


@pytest.mark.parametrize('change', [{'bytes':0}, {'bytes':True}, {'bytes':None}, {'bytes':301*1024**2}, {'sha256':None}, {'sha256':'bad'}])
def test_context_manifest_admission(transport, change):
    _, stage, _ = transport
    with pytest.raises(ValueError):stage.call('context-create',{'context_id':'context.'+'a'*32,'bytes':3,'sha256':'a'*64,**change})


@pytest.mark.parametrize('case', ['missing', 'unsealed', 'sha', 'symlink', 'size', 'identity', 'directory-link', 'root-link', 'operation', 'storage'])
def test_context_files_are_sealed_and_scoped(transport, monkeypatch, case):
    _, stage, _ = transport
    store=BuildContexts(stage.root); identity='context.'+'a'*32
    if case=='identity':
        with pytest.raises(ValueError):store.path('../escape')
        return
    if case=='storage':
        monkeypatch.setattr('ml_exp_server.build_contexts.shutil.disk_usage',lambda *a:SimpleNamespace(free=0))
        with pytest.raises(ValueError):store.call('context-create',{'context_id':identity,'bytes':3,'sha256':hashlib.sha256(b'abc').hexdigest()})
        return
    store.call('context-create',{'context_id':identity,'bytes':3,'sha256':hashlib.sha256(b'abc').hexdigest()})
    path=store.path(identity)
    if case=='root-link':
        moved=stage.root/'moved';store.root.rename(moved);store.root.symlink_to(moved,target_is_directory=True)
        with pytest.raises(ValueError):store.path(identity)
        return
    if case=='directory-link':
        path.rmdir() if not list(path.iterdir()) else shutil.rmtree(path)
        path.symlink_to(stage.root,target_is_directory=True)
        with pytest.raises(ValueError):store.path(identity)
        return
    if case=='operation':
        with pytest.raises(ValueError):store.call('unknown',{'context_id':identity})
        return
    if case!='missing':(path/'context.tar.gz').write_bytes(b'xyz' if case=='sha' else b'abc')
    if case=='symlink':(path/'context.tar.gz').unlink();(path/'context.tar.gz').symlink_to(path/'manifest.json')
    if case in {'missing','sha','symlink'}:
        with pytest.raises(ValueError):store.call('context-seal',{'context_id':identity})
    elif case=='unsealed':
        with pytest.raises(ValueError):store.archive(identity)
    else:
        store.call('context-seal',{'context_id':identity})
        (path/'context.tar.gz').chmod(0o600)  # Simulate an operator corrupting a sealed archive.
        (path/'context.tar.gz').write_bytes(b'longer')
        with pytest.raises(ValueError):store.archive(identity)
