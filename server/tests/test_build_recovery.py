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
        again=client.post(endpoint+'/reconcile',json={'confirmation':value['confirmation']}).json()
        assert again['build_error_history']==result['build_error_history']
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




def test_stage_messages_are_in_logs_before_buildkit_starts(tmp_path):
    value=image_builder.ImageBuilder({'state_root':str(tmp_path)})
    log=BUILD_LOG.set(tmp_path/'bundle.log');progress=BUILD_PROGRESS.set(tmp_path/'bundle.progress.json')
    try:value._progress('WAITING_BUILDER','Waiting for connection')
    finally:BUILD_LOG.reset(log);BUILD_PROGRESS.reset(progress)
    assert (tmp_path/'bundle.log').read_text()=='WAITING_BUILDER: Waiting for connection\n'




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
