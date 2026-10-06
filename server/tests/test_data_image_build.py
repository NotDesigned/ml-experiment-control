import io
import json
import os
from pathlib import Path
import threading
import importlib.util
from types import SimpleNamespace

import pytest

from ml_exp_server import data_image_build as module
from ml_exp_server import image_builder as builder_module
from ml_exp_server.image_builder import ImageBuilder, BuilderServer, Handler, UnixConnection, BUILD_REMOTE_CONTEXT, BUILD_CONTEXT_BYTES
from ml_exp_server.remote_data_uploads import stage_request, remote_archive
from ml_exp_server.storage import atomic_json

BASE='registry.example/python@sha256:'+'a'*64
IMAGE='registry.example/data@sha256:'+'b'*64
ASSET='asset.'+'c'*64


@pytest.fixture
def data_builder(tmp_path):
    builder=ImageBuilder({'state_root':str(tmp_path/'builder'),'source_root':str(tmp_path/'sources'),
        'repository':'registry.example/data','publisher':'buildkit','data_base_image':BASE,
        'data_upload_container':'ml-expd-data-stage'})
    asset={'project':'demo','asset_id':ASSET,'archive_bytes':100,'files':[{'path':'x','bytes':1,'sha256':'a'*64}]}
    def stage(operation,data,body):
        return {'ok':True,'result':asset if operation=='asset' else {'data_worker_sha256':module.data_worker_digest()}}
    builder.desktop_stage=stage
    def publish(tag,control):
        assert BUILD_REMOTE_CONTEXT.get()=='http://ml-expd-data-stage:8080/contexts/demo/'+ASSET+'.tar'
        assert BUILD_CONTEXT_BYTES.get()==100
        assert (control/'Dockerfile').read_text()==module.recipe(BASE)
        assert [p.name for p in control.iterdir()]==['Dockerfile']
        return IMAGE
    builder._publish_buildkit=publish
    return builder,{'operation':'data-image','project':'demo','asset_id':ASSET},asset


def test_data_image_receipt_reuse_and_reconciliation(data_builder):
    builder,request,asset=data_builder
    with pytest.raises(ValueError):builder.request({**request,'action':'get'})
    assert builder.request({**request,'action':'progress'})['progress']['phase']=='EXECUTING'
    result=builder.request(request)
    assert result['image']==IMAGE and result['files_sha256']==module.hashlib.sha256(json.dumps(asset['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert builder.request({**request,'action':'get'})==result
    assert builder.request({**request,'action':'progress'})['progress']['phase']=='READY'
    assert BUILD_REMOTE_CONTEXT.get() is None and BUILD_CONTEXT_BYTES.get()==0
    receipt=next((builder.root/'data-images').glob('*.json'))
    # Use the actual receipt, not its separate progress file.
    receipt=builder.root/'data-images'/(result['data_image_id'].split('.')[1]+'.json')
    atomic_json(receipt,{**result,'image':'changed'})
    with pytest.raises(ValueError):builder.request(request)


def test_module_entry_builder_keeps_remote_context_and_progress(data_builder):
    """Loading the CLI module again must use the same request ContextVars."""
    original, request, _ = data_builder
    spec = importlib.util.spec_from_file_location('ml_exp_server._entry_probe', builder_module.__file__)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    builder = entry.ImageBuilder(original.config)
    builder.desktop_stage = original.desktop_stage
    manifest = {'mediaType': entry.MANIFEST_TYPE, 'config': {'digest': 'sha256:' + 'd'*64}}
    raw = json.dumps(manifest)
    digest = 'sha256:' + module.hashlib.sha256(raw.encode()).hexdigest()
    def docker(arguments, **kwargs):
        assert arguments[-1] == 'http://ml-expd-data-stage:8080/contexts/demo/' + ASSET + '.tar'
        assert entry.BUILD_CONTEXT_BYTES.get() == 100
        assert entry.BUILD_LOG.get().parent == builder.root / 'data-images'
        metadata = Path(arguments[arguments.index('--metadata-file') + 1])
        atomic_json(metadata, {'containerimage.digest': digest, 'containerimage.config.digest': manifest['config']['digest']})
        return ''
    builder._docker = docker
    builder._skopeo = lambda *a, **kw: raw
    result = builder.request(request)
    progress = json.loads((builder.root / 'data-images' / (result['data_image_id'].split('.')[1] + '.progress.json')).read_text())
    assert [event['phase'] for event in progress['events']] == ['PREPARING_DATA_IMAGE', 'BUILDING_AND_PUSHING', 'VERIFYING_IMAGE', 'IMAGE_READY']
    assert entry.BUILD_REMOTE_CONTEXT.get() is None and entry.BUILD_CONTEXT_BYTES.get() == 0


@pytest.mark.parametrize('case',['project','asset','base','publisher','policy','input','identity','worker','publish-error'])
def test_data_image_rejects_changed_inputs(data_builder,case):
    builder,request,asset=data_builder
    if case in {'project','asset'}:request['project' if case=='project' else 'asset_id']='invalid/escape'
    elif case=='base':builder.config['data_base_image']='latest'
    elif case=='publisher':builder.config['publisher']='archive'
    elif case=='policy':builder.config['base_image_prefixes']=['unapproved.example/']
    elif case=='input':builder.desktop_stage=lambda *a:{'ok':False}
    elif case=='identity':asset['project']='other'
    elif case=='worker':builder.desktop_stage=lambda op,*a:{'ok':True,'result':asset if op=='asset' else {'data_worker_sha256':'wrong'}}
    elif case=='publish-error':builder._publish_buildkit=lambda *a:(_ for _ in ()).throw(ValueError('publication uncertain'))
    with pytest.raises(ValueError):builder.request(request)
    assert BUILD_REMOTE_CONTEXT.get() is None


@pytest.mark.parametrize('case',['ok','bad-container','no-host','bad-operation','failed','large'])
def test_desktop_stage_exec_is_fixed_and_bounded(tmp_path,monkeypatch,case):
    builder=ImageBuilder({'state_root':str(tmp_path/'state'),'source_root':str(tmp_path/'sources'),
        'repository':'registry.example/data','publisher':'buildkit'})
    builder.config.update(docker_host='unix:///run/private.sock',data_upload_container='ml-expd-data-stage')
    operation='part'
    if case=='bad-container':builder.config['data_upload_container']='unrelated'
    if case=='no-host':builder.config.pop('docker_host')
    if case=='bad-operation':operation='shell'
    def run(command,**kw):
        assert command[1:3]==['--host','unix:///run/private.sock']
        assert command[3:7]==['exec','-i','ml-expd-data-stage','python'] and kw['input']==b'bytes'
        return SimpleNamespace(returncode=1 if case=='failed' else 0,
            stdout=b'x'*(8*1024**2+1) if case=='large' else b'{"ok":true,"result":{}}')
    monkeypatch.setattr(builder_module.subprocess,'run',run)
    if case!='ok':
        with pytest.raises(ValueError):builder.desktop_stage(operation,{},b'bytes')
    else:assert builder.desktop_stage(operation,{},b'bytes')=={'ok':True,'result':{}}


@pytest.mark.parametrize('case',['stage','stage-error','oversize','metadata','truncated','get','get-completed','get-wrong-uid','get-path','get-query','get-missing','get-spawn-failed'])
def test_private_builder_socket_stages_and_streams_owned_data(tmp_path,monkeypatch,case):
    path=str(tmp_path/'builder.sock');server=BuilderServer(path,Handler)
    config={'client_uid':os.getuid()+(1 if case=='get-wrong-uid' else 0),'docker_host':'unix:///run/private.sock','data_upload_container':'ml-expd-data-stage'}
    def stage(operation,data,body):
        return {'ok':case not in {'stage-error','get-missing'},'result':{'archive_bytes':3},'status_code':507,'error':'DATA_STORAGE'}
    server.builder=SimpleNamespace(config=config,desktop_stage=stage)
    class Process:
        stdout=io.BytesIO(b'abc')
        def poll(self):return 0 if case=='get-completed' else None
        def terminate(self):pass
        def wait(self,**kw):return 0
    def spawn(*a,**kw):
        if case=='get-spawn-failed':raise OSError()
        return Process()
    monkeypatch.setattr(builder_module.subprocess,'Popen',spawn)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        if case in {'stage','stage-error'}:
            if case=='stage':assert stage_request(path,'info',{})=={'archive_bytes':3}
            else:
                from ml_exp_server.application_errors import ApplicationError
                with pytest.raises(ApplicationError):stage_request(path,'info',{})
        elif case.startswith('get'):
            conn=UnixConnection(path,timeout=5)
            url='/data-stage/archive?project=demo&asset_id='+ASSET
            if case=='get-path':url='/other?project=demo&asset_id='+ASSET
            if case=='get-query':url='/data-stage/archive?project=demo&project=other&asset_id='+ASSET
            conn.request('GET',url);response=conn.getresponse()
            assert response.status==(200 if case in {'get','get-completed'} else 409)
            if case in {'get','get-completed'}:assert response.read()==b'abc'
            else:response.read()
            conn.close()
        else:
            conn=UnixConnection(path,timeout=5)
            headers={'X-ML-Expd-Data':'x'*8193 if case=='metadata' else '{}','Content-Length':str(64*1024**2+1) if case=='oversize' else '1'}
            # EOF reveals a truncated request without waiting for a timeout.
            # Rejected headers may close the socket before the body is sent.
            conn.request('POST','/data-stage/part',body=b'',headers=headers)
            if case=='truncated':conn.sock.shutdown(1)
            response=conn.getresponse();assert response.status==409;response.read();conn.close()
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2)
