"""Transport failures stay diagnosable and cannot authorize replay."""
import json
import os
import subprocess
import threading
from types import SimpleNamespace

import pytest

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.image_builder import BuilderServer, Handler, ImageBuilder, BuildTransportError, builder_request, BUILD_LOG
from ml_exp_server.remote_data_uploads import stage_request
from ml_exp_server import image_builder as module
from tests.test_container_api import client, import_source, archive, wait_runtime
from tests.test_image_builder_boundary import builder


@pytest.mark.parametrize('case',['session','copy'])
def test_command_transport_classification_does_not_expose_private_output(tmp_path,monkeypatch,case):
    value=ImageBuilder({'state_root':str(tmp_path)})
    path=tmp_path/'build.log';token=BUILD_LOG.set(path)
    def failed(command,**kwargs):
        kwargs['stdout'].write(b'session healthcheck context deadline exceeded; private-secret\n' if case=='session' else b'private-secret\n')
        raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(module.subprocess,'run',failed)
    try:
        with pytest.raises(BuildTransportError) as error:
            value._command(['docker','--host','unix:///private.sock','buildx' if case=='session' else 'cp','private-secret'])
        assert error.value.code==('BUILD_SESSION_TIMEOUT' if case=='session' else 'BUILD_CONTEXT_TRANSPORT')
        assert error.value.details['publication_uncertain'] and not error.value.details['scheduler_submitted']
        assert 'private-secret' not in str(error.value)
    finally:BUILD_LOG.reset(token)


@pytest.mark.parametrize('case',['build-session','build-rpc','upload-rpc'])
def test_private_error_boundary_preserves_only_safe_codes(tmp_path,case):
    path=str(tmp_path/'builder.sock');server=BuilderServer(path,Handler)
    def fail(*args):
        if case=='build-session':raise BuildTransportError('BUILD_SESSION_TIMEOUT',{'publication_uncertain':True,'scheduler_submitted':False})
        raise ApplicationError('private-secret',status_code=503,code='DESKTOP_RPC_TIMEOUT')
    server.builder=SimpleNamespace(config={'client_uid':os.getuid()},request=fail,desktop_stage=fail)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with pytest.raises(ApplicationError if case=='upload-rpc' else BuildTransportError) as error:
            stage_request(path,'read',{}) if case=='upload-rpc' else builder_request(path,{'operation':'build'})
        assert 'private-secret' not in str(error.value)
        if case=='upload-rpc':assert error.value.status_code==503 and error.value.code=='DESKTOP_RPC_TIMEOUT'
        else:assert error.value.details['publication_uncertain']
    finally:server.shutdown();thread.join();server.server_close()


def test_failed_historical_build_logs_survive_transport_upgrade(builder,monkeypatch):
    value,request,_=builder
    legacy='c'*64
    (value.root/(legacy+'.definition.json')).write_text(json.dumps({'bundle_id':legacy,'project':request['project'],'source_id':request['source_id'],'base_image':request['base_image']}))
    (value.root/(legacy+'.log')).write_text('legacy session failed\n')
    assert value.request({**request,'operation':'logs','bundle_id':legacy})['lines']==['legacy session failed']
    old=value.request(request)
    (value.root/(old['bundle_id']+'.json')).unlink()
    (value.root/(old['bundle_id']+'.log')).write_text('new failed build\n')
    monkeypatch.setattr(module,'BUILD_REVISION','next-transfer')
    assert module.bundle_id(request['project'],request['source_id'],request['base_image'])!=old['bundle_id']
    assert value.request({**request,'operation':'logs','bundle_id':old['bundle_id']})['lines']==['new failed build']
    path=value.root/(old['bundle_id']+'.definition.json')
    path.write_text('{}')
    with pytest.raises(ValueError,match='identity mismatch'):value.request({**request,'operation':'logs','bundle_id':old['bundle_id']})


def test_runtime_transport_failure_freezes_evidence_and_rejects_automatic_replay(client,monkeypatch):
    source=import_source(client,archive({'Dockerfile':('FROM registry.example/python@sha256:'+'a'*64+'\n').encode(),'train.py':b'print("train")'}))
    prepared=client.post('/api/projects/demo/runtimes/prepare',json={'source_id':source['source_id'],'entrypoint':['python3','train.py']}).json()
    def failed(*a,**k):raise BuildTransportError('BUILD_SESSION_TIMEOUT',{'publication_uncertain':True,'scheduler_submitted':False})
    monkeypatch.setattr('ml_exp_server.container_execution.builder_request',failed)
    endpoint='/api/projects/demo/runtimes/'+prepared['runtime_id']
    assert client.post(endpoint+'/execute',json={'confirmation':prepared['confirmation']}).status_code==202
    value=wait_runtime(client,endpoint)
    assert value['status']=='RECONCILE_REQUIRED' and value['error']=='BUILD_SESSION_TIMEOUT'
    assert not value['build_error']['retry_safe'] and value['build_error']['details']['publication_uncertain']
    assert client.post(endpoint+'/execute',json={'confirmation':prepared['confirmation']}).status_code==409
