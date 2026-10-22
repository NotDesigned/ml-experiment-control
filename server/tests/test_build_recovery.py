"""No retry proof from missing receipts alone, stale observations or live publishers."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import fcntl
import http.client
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_exp_server.builds import build_recovery as recovery
from ml_exp_server.builds.builder_client import BuildReceiptMissingError, BuildTransportError, builder_request
from ml_exp_server.builds.image_build_context import BUILD_LOG, BUILD_PROGRESS, BUILD_RETRY
from ml_exp_server import image_builder, container_execution
from ml_exp_server.container_execution import ContainerExecutionService
from ml_exp_server.storage import utc_now
from tests.test_container_api import client, import_source, prepare_runtime, wait_runtime
from tests.test_image_builder_boundary import builder

IDENTITY = 'a' * 64


def proof(identity=IDENTITY):
    return {'source': 'builder-reconcile.v1', 'bundle_id': identity, 'observed_at': utc_now(),
            'registry_manifest_status': 404, 'active_builders': 0, 'leases_clear': True}


@pytest.mark.parametrize('case', ['ok', 'status', 'large', 'socket', 'http'])
def test_docker_read_is_bounded_readonly_and_never_exposes_transport_errors(monkeypatch, case):
    calls=[]
    class Connection:
        def __init__(self, path, timeout):
            assert path == '/private/docker.sock' and timeout == 5
        def request(self, method, path, *, headers):
            assert headers == {'Host': 'localhost'}
            calls.append((method,path))
            if case=='socket':raise OSError('credential-must-not-escape')
            if case=='http':raise http.client.HTTPException('credential-must-not-escape')
        def getresponse(self):
            return SimpleNamespace(status=503 if case=='status' else 200,
                                   read=lambda n:b'x'*n if case=='large' else b'OK')
        def close(self):calls.append('close')
    monkeypatch.setattr(recovery,'UnixConnection',Connection)
    if case=='ok':assert recovery.docker_read({'docker_host':'unix:///private/docker.sock'},'/_ping')==b'OK'
    else:
        with pytest.raises(BuildTransportError) as error:
            recovery.docker_read({'docker_host':'unix:///private/docker.sock'},'/_ping')
        assert error.value.code=='BUILDER_DOCKER_UNAVAILABLE'
        assert error.value.details['publication_uncertain'] is False
        assert 'credential' not in str(error.value)
    assert calls[0]==('GET','/_ping') and calls[-1]=='close'


@pytest.mark.parametrize('host,body',[(None,b'bad'),('unix:///socket',b'OK'),('unix:///socket',b'bad')])
def test_desktop_ping_has_no_local_fallback(monkeypatch,host,body):
    monkeypatch.setattr(recovery,'docker_read',lambda *a:body)
    if host and body!=b'OK':
        with pytest.raises(BuildTransportError,match='BUILDER_DOCKER_UNAVAILABLE'):
            recovery.desktop_ready({'docker_host':host})
    else:recovery.desktop_ready({'docker_host':host})


@pytest.fixture
def observer(tmp_path,monkeypatch):
    value=SimpleNamespace(root=tmp_path,config={'docker_host':'unix:///socket','ephemeral_buildkit':True,
                                               'repository':'registry.example/runtime'})
    calls=[]
    monkeypatch.setattr(recovery,'docker_read',lambda config,path:calls.append(('docker',path)) or b'[]')
    class Registry:
        prefix='/v2/runtime'
        def __init__(self,repository,auth,timeout):
            assert repository=='registry.example/runtime' and timeout==15
        def request(self,method,path,**kw):
            assert method=='GET' and path.endswith('/manifests/bundle-'+IDENTITY)
            assert kw['expected']==(200,404)
            # Both locks must remain held throughout the registry observation.
            for name in [IDENTITY+'.lock','ephemeral-buildkit.lock']:
                with (tmp_path/name).open('a') as lock:
                    with pytest.raises(BlockingIOError):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            calls.append(('registry',path));return 404,{},b'{}'
    monkeypatch.setattr(recovery,'Registry',Registry)
    return value,calls


def test_authenticated_absence_requires_both_locks_and_only_reads(observer):
    value,calls=observer
    result=recovery.publication_absence(value,IDENTITY)
    assert recovery.valid_absence(result,IDENTITY)
    assert len(calls)==2 and calls[0][0]=='docker' and calls[1][0]=='registry'


@pytest.mark.parametrize('case',['local','not-ephemeral','bundle-busy','worker-busy','receipt','builder','context','probe','containers','invalid-json','transport','registry','published'])
def test_no_retry_on_busy_unknown_or_published_state(observer,monkeypatch,case):
    value,calls=observer
    marker={'receipt':IDENTITY+'.json','builder':'ephemeral-builder.json','context':'source-context.json','probe':'storage-probe.json'}.get(case)
    if marker:(value.root/marker).write_text('{}')
    if case=='local':value.config.pop('docker_host')
    if case=='not-ephemeral':value.config['ephemeral_buildkit']=False
    lock=None
    if case in ['bundle-busy','worker-busy']:
        lock=(value.root/(IDENTITY+'.lock' if case=='bundle-busy' else 'ephemeral-buildkit.lock')).open('a')
        fcntl.flock(lock,fcntl.LOCK_EX)
    if case=='containers':monkeypatch.setattr(recovery,'docker_read',lambda *a:b'[{"State":"running"}]')
    if case=='invalid-json':monkeypatch.setattr(recovery,'docker_read',lambda *a:b'not json')
    if case=='transport':monkeypatch.setattr(recovery,'docker_read',lambda *a:(_ for _ in ()).throw(BuildTransportError('BUILDER_DOCKER_UNAVAILABLE',{})))
    if case in ['registry','published']:
        class Registry:
            prefix='/v2/runtime'
            def __init__(self,*a,**kw):pass
            def request(self,*a,**kw):
                if case=='registry':raise ValueError('private-secret')
                return 200,{},b'{}'
        monkeypatch.setattr(recovery,'Registry',Registry)
    try:
        if case in ['invalid-json','transport','registry','published']:
            with pytest.raises(BuildTransportError) as error:recovery.publication_absence(value,IDENTITY)
            assert 'private-secret' not in str(error.value)
        else:assert recovery.publication_absence(value,IDENTITY) is None
    finally:
        if lock:lock.close()


@pytest.mark.parametrize('change',[None,{}, {'extra':1},{'source':'other'},{'bundle_id':'b'*64},
    {'registry_manifest_status':200},{'registry_manifest_status':404.0},{'active_builders':False},
    {'active_builders':1},{'leases_clear':False},{'observed_at':None},{'observed_at':'invalid'},
    {'observed_at':'2026-10-01T00:00:00'},{'observed_at':'2000-01-01T00:00:00Z'},
    {'observed_at':'2999-01-01T00:00:00Z'}])
def test_retry_proof_is_identity_bound_typed_and_fresh(change):
    data=None if change is None else {} if change=={} else {**proof(),**change}
    assert recovery.valid_absence(data,IDENTITY) is False


@pytest.mark.parametrize('case',['verified','invalid'])
def test_unknown_runtime_recovers_only_with_fresh_positive_proof(client,monkeypatch,case):
    source=import_source(client)
    value=prepare_runtime(client,{'source_id':source['source_id'],'entrypoint':['python3']}).json()
    endpoint='/api/projects/demo/runtimes/'+value['runtime_id']
    original=container_execution.builder_request
    def fail(*a):raise ValueError('not-preserved-private-secret')
    monkeypatch.setattr(container_execution,'builder_request',fail)
    client.post(endpoint+'/execute',json={'confirmation':value['confirmation']})
    failed=wait_runtime(client,endpoint)
    assert failed['build_error']['code']=='BUILD_UNCLASSIFIED_FAILURE'
    assert failed['build_error']['retry_safe'] is False
    data=proof(value['build_bundle_id']) if case=='verified' else {'untrusted':True}
    def missing(socket,payload):
        assert payload['operation']=='reconcile'
        raise BuildReceiptMissingError('missing',retry_proof=data)
    monkeypatch.setattr(container_execution,'builder_request',missing)
    result=client.post(endpoint+'/reconcile',json={'confirmation':value['confirmation']}).json()
    assert result['build_error']['retry_safe'] is (case=='verified')
    assert result['runtime_id']==value['runtime_id'] and result['spec']==value['spec']
    if case=='verified':
        monkeypatch.setattr(container_execution,'builder_request',original)
        assert client.post(endpoint+'/execute',json={'confirmation':value['confirmation']}).status_code==202
        final=wait_runtime(client,endpoint)
        assert final['status']=='READY' and final['build_error'] is None
        assert final['build_error_history'][-1]['code']=='BUILD_PUBLICATION_ABSENT'
    else:assert client.post(endpoint+'/execute',json={'confirmation':value['confirmation']}).status_code==409


@pytest.mark.parametrize('details',[{'publication_uncertain':False,'scheduler_submitted':False},
    {'publication_uncertain':True,'scheduler_submitted':False},{'publication_uncertain':False},[]])
def test_only_prebuild_disconnection_can_grant_retry(client,monkeypatch,details):
    source=import_source(client)
    value=prepare_runtime(client,{'source_id':source['source_id'],'entrypoint':['python3']}).json()
    service=ContainerExecutionService(client.app.state.runtime)
    monkeypatch.setattr(container_execution,'builder_request',lambda *a:(_ for _ in ()).throw(BuildTransportError('BUILDER_DOCKER_UNAVAILABLE',details)))
    result=service.execute('demo',value['runtime_id'],value['confirmation'])
    assert result['build_error']['retry_safe'] is (details=={'publication_uncertain':False,'scheduler_submitted':False})
    result=service.execute('demo',value['runtime_id'],value['confirmation'],reconcile=True)
    assert result['build_error']['retry_safe'] is False


def test_missing_receipt_from_worker_roundtrips_proof_and_never_builds(builder,tmp_path,monkeypatch):
    import os,threading
    value,request,_=builder
    identity=image_builder.bundle_id('demo',request['source_id'],request['base_image'])
    (value.root/(identity+'.definition.json')).write_text(json.dumps({'bundle_id':identity,'project':'demo','source_id':request['source_id'],'base_image':request['base_image']}))
    monkeypatch.setattr(image_builder,'publication_absence',lambda *a:proof(identity))
    value.config['client_uid']=os.getuid()
    server=image_builder.BuilderServer(str(tmp_path/'recovery.sock'),image_builder.Handler)
    server.builder=value;thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with pytest.raises(BuildReceiptMissingError) as error:
            builder_request(str(tmp_path/'recovery.sock'),{**request,'operation':'reconcile'})
        assert recovery.valid_absence(error.value.retry_proof,identity)
        assert not (value.root/(identity+'.json')).exists()
        assert BUILD_RETRY.get() is None
    finally:server.shutdown();thread.join();server.server_close()


@pytest.mark.parametrize('case',['none','absent','published'])
def test_retry_rechecks_current_publication_under_build_lock_before_creating_container(tmp_path,monkeypatch,case):
    from tests.test_ssh_builder import config
    value=image_builder.ImageBuilder(config(tmp_path))
    calls=[]
    monkeypatch.setattr(image_builder,'desktop_ready',lambda config:None)
    def observe(*a):
        calls.append('fresh-observation')
        if case=='published':raise BuildTransportError('BUILD_PUBLICATION_UNRESOLVED',{})
        return None if case=='none' else proof()
    monkeypatch.setattr(image_builder,'observe_absence',observe)
    value._recover_probe=lambda:None;value._recover_builder=lambda:None
    value._storage_preflight=lambda *a:(_ for _ in ()).throw(RuntimeError('after-positive-proof'))
    value._docker=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('must not create before proof'))
    token=BUILD_RETRY.set(IDENTITY)
    try:
        with pytest.raises(RuntimeError if case=='absent' else BuildTransportError):
            value._publish_buildkit('registry.example/runtime:bundle-'+IDENTITY,tmp_path)
    finally:BUILD_RETRY.reset(token)
    assert calls==['fresh-observation']


def test_stage_messages_are_in_logs_before_buildkit_starts(tmp_path):
    value=image_builder.ImageBuilder({'state_root':str(tmp_path)})
    log=BUILD_LOG.set(tmp_path/'bundle.log');progress=BUILD_PROGRESS.set(tmp_path/'bundle.progress.json')
    try:value._progress('WAITING_BUILDER','Waiting for connection')
    finally:BUILD_LOG.reset(log);BUILD_PROGRESS.reset(progress)
    assert (tmp_path/'bundle.log').read_text()=='WAITING_BUILDER: Waiting for connection\n'


def test_real_unix_http_docker_request_uses_valid_host(tmp_path):
    from http.server import BaseHTTPRequestHandler
    import socketserver,threading
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def do_GET(self):
            self.send_response(200 if self.headers['Host']=='localhost' else 400)
            self.end_headers();self.wfile.write(b'OK')
    server=socketserver.UnixStreamServer(str(tmp_path/'docker.sock'),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:recovery.desktop_ready({'docker_host':'unix://'+str(tmp_path/'docker.sock')})
    finally:server.shutdown();thread.join();server.server_close()


def test_retry_request_preserves_context_binding_and_clears_it_after_success(builder):
    value,request,_=builder
    identity=image_builder.bundle_id('demo',request['source_id'],request['base_image'])
    seen=[]
    def publish(*a):
        seen.append(BUILD_RETRY.get())
        if len(seen)==1:raise BuildTransportError('BUILDER_DOCKER_UNAVAILABLE',{})
        return 'registry.example/results@sha256:'+'b'*64
    value._publish_buildkit=publish
    with pytest.raises(BuildTransportError):value.request(request)
    assert BUILD_RETRY.get() is None
    value.request(request)
    assert seen==[None,identity] and BUILD_RETRY.get() is None


def test_builder_receipt_reappearing_is_read_without_new_build(builder):
    value,request,_=builder
    result=value.request(request)
    assert value.request({**request,'operation':'reconcile'})==result
