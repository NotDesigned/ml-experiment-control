"""Real resumable desktop storage with only transport boundaries substituted."""
import hashlib
import io
import json
import os
import runpy
import sys
import threading
import urllib.request
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from ml_exp_server import desktop_upload as desktop
from ml_exp_server import remote_data_uploads as remote
from ml_exp_server.archive_limits import byte_limit, exceeds, minimum_limit, wire_limit
from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.data_assets import AssetStore
from ml_exp_server.storage import atomic_json
from tests.test_artifact_store import archive
from tests.test_sensecore_data_workflow import client, stored
from tests.test_multipart_upload import begin, send


@pytest.fixture
def stage(tmp_path):
    workers = tmp_path / 'workers'; workers.mkdir()
    for name in ('worker.py','legacy_worker.py','data_preparation.py','persistent_state.py','data_copy_worker.py'):
        (workers / name).write_text('# trusted ' + name)
    return desktop.DesktopUploads(tmp_path / 'desktop', {'upload_part_bytes':1024,'worker_directory':str(workers),
        'data_base_image':'registry.example/python@sha256:'+'a'*64})


@pytest.fixture
def remote_client(client, stored, stage, monkeypatch):
    config = json.loads(stored[0].read_text())
    config.update(data_upload_storage='desktop-builder',data_upload_socket='desktop.sock',max_archive_bytes=None)
    stored[0].write_text(json.dumps(config))
    def rpc(socket, operation, metadata, body=b''):
        assert socket == 'desktop.sock'
        return stage.call(operation,metadata,body)
    monkeypatch.setattr(remote, 'stage_request', rpc)
    def stream(socket, project, asset_id, size):
        body=stage.stream(project,asset_id)
        body.iter_chunks=lambda chunk_size:iter(lambda:body.read(chunk_size),b'')
        return body
    monkeypatch.setattr(remote, 'remote_archive', stream)
    return client


def test_desktop_parts_resume_publish_and_archive_without_server_bytes(remote_client, stage, stored):
    client=remote_client
    data=archive({'nested/tokens.bin':b'data'*700})
    value=begin(client,data)
    root=send(client,data,value,numbers=[0])
    assert list(begin(client,data)['parts'])==['0']
    assert client.get(root).json()['parts']['0']['bytes']==1024
    send(client,data,value)
    result=client.post(root+'/complete')
    assert result.status_code==200,result.text
    result=result.json()
    assert result['remote_storage']=='desktop-builder'
    assert client.post(root+'/complete').json()==result
    assert begin(client,data)['result']==result
    assert client.get('/api/projects/demo/assets/'+result['asset_id']+'/archive').content==data
    server=client.app.state.runtime.config.project_registry_root_path()
    assert not (server/'multipart-uploads').exists() and not stored[1]
    assert not (server/'data-assets/demo'/result['asset_id']/'tree').exists()
    assert not (stage.uploads.root/value['upload_id']/'parts').exists()
    assert not list(stage.root.glob('assets/demo/.validate-*'))
    limits=client.get('/api/storage-limits').json()
    assert limits['configured_byte_limits']['asset_archive_bytes'] is None
    assert limits['data_upload_storage']=='desktop-builder' and limits['data_upload_free_bytes']>0
    assert client.post('/api/assets/archive?project=demo&sha256='+hashlib.sha256(data).hexdigest(),content=data).status_code==409
    assert client.delete(root).status_code==409
    assert client.get('/api/executors').json()['executors'][0]['data_asset_transport']=='ccr-cpu-nas'
    assets=AssetStore(stored[0],server)
    with pytest.raises(ApplicationError,match='authenticated archive'):assets.download('demo',result['asset_id'])
    out=io.BytesIO();desktop.context(stage,'demo',result['asset_id'],out)
    out.seek(0)
    with tarfile.open(fileobj=out) as context:
        assert context.extractfile('dataset.tar').read()==data
        assert b'COPY dataset.tar' in context.extractfile('Dockerfile').read()
    # Restaging an existing historical asset changes only its sidecar locator.
    path=server/'data-assets/demo'/result['asset_id']/'asset.json'
    before=path.read_bytes()
    assets.publish_remote('demo',result)
    assert path.read_bytes()==before and assets.read('demo',result['asset_id'])['remote_storage']=='desktop-builder'
    previous=dict(result);previous['archive_bytes']+=1
    with pytest.raises(ValueError,match='sealed'):assets.publish_remote('demo',previous)
    sidecar=server/'desktop-data-assets/demo'/(result['asset_id']+'.json')
    atomic_json(sidecar,previous)
    with pytest.raises(ValueError,match='locator'):assets.read('demo',result['asset_id'])
    assert stage.publish({'project':'demo'},io.BytesIO(data),len(data),result['sha256'],value)==result


def test_remote_upload_abort_and_bad_parts_do_not_persist(remote_client, stage):
    client=remote_client;data=archive({'x':b'x'*3000});value=begin(client,data)
    endpoint='/api/projects/demo/asset-uploads/'+value['upload_id']+'/parts/0?sha256='+hashlib.sha256(data[:1024]).hexdigest()
    assert client.put(endpoint,content=data[:100]).status_code==400
    assert client.put(endpoint,content=b'x'*1024).status_code==400
    assert client.put(endpoint,content=data[:1025],headers={'Content-Length':'1024'}).status_code==413
    assert client.put(endpoint,content=b'x',headers={'Content-Length':'1024'}).status_code==400
    assert client.delete('/api/projects/demo/asset-uploads/'+value['upload_id']).json()['status']=='ABORTED'
    assert remote.RemoteDataUploads(SimpleNamespace(root=stage.root),'test').part_size(value,0)==1024
    with pytest.raises(ValueError):remote.RemoteDataUploads(SimpleNamespace(root=stage.root),'test').part(value['upload_id'],{},0,b'x','a'*64,1)


@pytest.mark.parametrize('case',['bad-operation','wrong-kind','wrong-size','wrong-hash','disk'])
def test_desktop_upload_fails_closed(stage,monkeypatch,case):
    binding={'kind':'asset','project':'demo'}
    value=stage.call('create',{'binding':binding,'sha256':'a'*64,'bytes':1024})
    data={'binding':binding,'upload_id':value['upload_id'],'number':0,'bytes':1024,'sha256':hashlib.sha256(b'x'*1024).hexdigest()}
    operation='part';body=b'x'*1024
    if case=='bad-operation':operation='unknown'
    if case=='wrong-kind':data['binding']={'kind':'artifacts'}
    if case=='wrong-size':body=b'x'
    if case=='wrong-hash':data['sha256']='b'*64
    if case=='disk':monkeypatch.setattr(desktop.shutil,'disk_usage',lambda p:SimpleNamespace(free=0))
    with pytest.raises((ValueError,ApplicationError)):stage.call(operation,data,body)
    assert not list(stage.uploads.root.glob('.part-*'))


@pytest.mark.parametrize('name',['../escape','/absolute','.secret','a\\b','keys/x','x.pem'])
def test_desktop_dataset_paths_are_not_extracted(tmp_path,name):
    with pytest.raises(ValueError):desktop.unpack(io.BytesIO(archive({name:b'x'})),tmp_path,None,20000)


@pytest.mark.parametrize('case',['empty','duplicate','symlink','count','quota','disk'])
def test_desktop_dataset_inventory_bounds(tmp_path,monkeypatch,case):
    out=io.BytesIO()
    with tarfile.open(fileobj=out,mode='w') as tar:
        root=tarfile.TarInfo('.');root.type=tarfile.DIRTYPE;tar.addfile(root)
        folder=tarfile.TarInfo('nested');folder.type=tarfile.DIRTYPE;tar.addfile(folder)
        if case!='empty':
            item=tarfile.TarInfo('nested/x');item.size=1
            if case=='symlink':item.type=tarfile.SYMTYPE;item.linkname='/etc/passwd';item.size=0
            tar.addfile(item,io.BytesIO(b'x'))
            if case=='duplicate':tar.addfile(item,io.BytesIO(b'x'))
    if case=='disk':monkeypatch.setattr(desktop.shutil,'disk_usage',lambda p:SimpleNamespace(free=0))
    with pytest.raises((ValueError,ApplicationError)):
        desktop.unpack(io.BytesIO(out.getvalue()),tmp_path,0 if case=='quota' else None,1 if case=='count' else 20000)


@pytest.mark.parametrize('value',[True,-1,0,'4GiB'])
def test_invalid_optional_quotas(value):
    with pytest.raises(ValueError):byte_limit(value)


def test_optional_limits_preserve_explicit_quotas_and_wire_integers():
    assert byte_limit(None) is None and byte_limit(123)==123
    assert not exceeds(2**40,None) and exceeds(11,10)
    assert wire_limit(None)==2**63-1 and wire_limit(10)==10
    assert minimum_limit(None,10,20)==10 and minimum_limit(None,None) is None


@pytest.mark.parametrize('change',[{'project':'other'},{'sha256':'bad'},{'files':[]},
    {'files':[{'path':'../x','bytes':1,'sha256':'a'*64}]}, {'files':[{'path':'x','bytes':-1,'sha256':'a'*64}]},
    {'files':[{'path':'x','bytes':1,'sha256':'bad'}]}])
def test_server_validates_private_desktop_receipts(stored,tmp_path,change):
    value={'project':'demo','asset_id':'asset.'+'a'*64,'sha256':'a'*64,'archive_bytes':100,'remote_storage':'desktop-builder',
           'files':[{'path':'x','bytes':1,'sha256':'a'*64}]}
    with pytest.raises(ValueError):AssetStore(stored[0],tmp_path/'registry').publish_remote('demo',{**value,**change})


@pytest.mark.parametrize('status,length,data',[(200,'3',b'abc'),(409,'3',b'abc'),(200,'4',b'abc'),(200,'3',b'a')])
def test_remote_archive_is_bounded_and_closes(monkeypatch,status,length,data):
    closed=[];body=io.BytesIO(data)
    class Connection:
        def __init__(self,*a,**kw):pass
        def request(self,*a,**kw):pass
        def getresponse(self):return SimpleNamespace(status=status,getheader=lambda name:length,read=body.read)
        def close(self):closed.append(True)
    monkeypatch.setattr(remote,'UnixConnection',Connection)
    if status!=200 or length!='3':
        with pytest.raises(ValueError):remote.remote_archive('sock','demo','asset.'+'a'*64,3)
    else:
        stream=remote.remote_archive('sock','demo','asset.'+'a'*64,3)
        if len(data)!=3:
            with pytest.raises(ValueError):list(stream.iter_chunks(2))
        else:assert b''.join(stream.iter_chunks(2))==data
    assert closed


@pytest.mark.parametrize('status,value',[(200,{'ok':True}),(507,{'error':'UPLOAD_STORAGE','status_code':507}),(200,None)])
def test_private_stage_response_bounds(monkeypatch,status,value):
    closed=[]
    class Connection:
        def __init__(self,*a,**kw):pass
        def request(self,*a,**kw):pass
        def getresponse(self):return SimpleNamespace(status=status,read=lambda size:b'x'*size if value is None else json.dumps(value).encode())
        def close(self):closed.append(True)
    monkeypatch.setattr(remote,'UnixConnection',Connection)
    if status!=200 or value is None:
        with pytest.raises((ApplicationError,ValueError)):remote.stage_request('sock','info',{})
    else:assert remote.stage_request('sock','info',{})==value
    assert closed==[True]


def test_desktop_context_http_and_cli_return_only_owned_payload(stage,monkeypatch):
    binding={'kind':'asset','project':'demo'};data=archive({'x':b'abc'});sha=hashlib.sha256(data).hexdigest()
    upload=stage.call('create',{'binding':binding,'sha256':sha,'bytes':len(data)})
    for number in range(upload['part_count']):
        part=data[number*1024:(number+1)*1024]
        stage.call('part',{'binding':binding,'upload_id':upload['upload_id'],'number':number,'bytes':len(part),
            'sha256':hashlib.sha256(part).hexdigest()},part)
    asset=stage.call('complete',{'binding':binding,'upload_id':upload['upload_id']})
    assert stage.call('complete',{'binding':binding,'upload_id':upload['upload_id']})==asset
    assert stage.call('asset',{'binding':binding,'asset_id':asset['asset_id']})==asset
    monkeypatch.setattr(desktop,'store',lambda:stage)
    server=desktop.ThreadingHTTPServer(('127.0.0.1',0),desktop.ContextHandler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        url='http://127.0.0.1:'+str(server.server_port)
        with urllib.request.urlopen(url+'/contexts/demo/'+asset['asset_id']+'.tar') as response:
            with tarfile.open(fileobj=io.BytesIO(response.read())) as tar:assert tar.extractfile('dataset.tar').read()==data
        for path in ('/bad','/contexts/demo/asset.'+'a'*64+'.tar'):
            with pytest.raises(urllib.error.HTTPError):urllib.request.urlopen(url+path)
    finally:
        server.shutdown();server.server_close();thread.join()
    output=io.BytesIO();stdout=SimpleNamespace(buffer=output,write=lambda s:output.write(s.encode()),flush=lambda:None)
    monkeypatch.setattr(desktop.sys,'stdout',stdout)
    monkeypatch.setattr(desktop.sys,'argv',['desktop','archive',json.dumps({'project':'demo','asset_id':asset['asset_id']})])
    desktop.main();assert output.getvalue()==data
    output.seek(0);output.truncate()
    monkeypatch.setattr(desktop.sys,'argv',['desktop','info','{}'])
    desktop.main();assert json.loads(output.getvalue())['ok']
    output.seek(0);output.truncate()
    monkeypatch.setattr(desktop.sys,'argv',['desktop','wrong','{}'])
    desktop.main();assert json.loads(output.getvalue())['status_code']==409
    output.seek(0);output.truncate()
    monkeypatch.setattr(stage,'call',lambda *a:(_ for _ in ()).throw(ApplicationError('capacity',status_code=507,code='UPLOAD_STORAGE')))
    desktop.main();assert json.loads(output.getvalue())['status_code']==507


def test_desktop_cli_stdin_serve_and_configuration(tmp_path,monkeypatch):
    stage=desktop.DesktopUploads(tmp_path/'stage',{'upload_part_bytes':1024})
    monkeypatch.setattr(desktop,'store',lambda:stage)
    monkeypatch.setattr(desktop.sys,'stdin',SimpleNamespace(buffer=io.BytesIO(b'data')))
    monkeypatch.setattr(desktop.sys,'argv',['desktop','part',json.dumps({'bytes':4,'binding':{'kind':'asset'},'upload_id':'invalid'})])
    output=io.BytesIO()
    monkeypatch.setattr(desktop.sys,'stdout',SimpleNamespace(write=lambda s:output.write(s.encode()),flush=lambda:None))
    desktop.main();assert json.loads(output.getvalue())['status_code']==409
    calls=[]
    class Server:
        def __init__(self,*a):calls.append(a)
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def serve_forever(self):calls.append('serve')
    monkeypatch.setattr(desktop,'ThreadingHTTPServer',Server)
    monkeypatch.setattr(desktop.sys,'argv',['desktop','serve'])
    desktop.main();assert calls[-1]=='serve'
    with pytest.raises(ValueError):stage.asset_path('../outside','asset.'+'a'*64)
    with pytest.raises(ValueError):stage.asset_path('demo','bad')


def test_desktop_entrypoint_reads_fixed_config(tmp_path,monkeypatch,capsys):
    configuration=tmp_path/'upload-config.json';workers=tmp_path/'workers';workers.mkdir()
    for file in ('worker.py','legacy_worker.py','data_preparation.py','persistent_state.py','data_copy_worker.py'):
        (workers/file).write_text('# trusted')
    configuration.write_text(json.dumps({'worker_directory':str(workers)}))
    real=Path
    monkeypatch.setattr(Path,'read_text',lambda self,*a,**kw:configuration.read_bytes().decode() if str(self)=='/app/upload-config.json' else self.read_bytes().decode())
    original=desktop.UploadStore.__init__
    monkeypatch.setattr(desktop.UploadStore,'__init__',lambda self,root,config:original(self,tmp_path/'stage',config))
    usage=desktop.shutil.disk_usage(tmp_path)
    monkeypatch.setattr(desktop.shutil,'disk_usage',lambda path:usage)
    monkeypatch.setattr(desktop.sys,'argv',['desktop','info','{}'])
    runpy.run_module('ml_exp_server.desktop_upload',run_name='__main__')
    assert json.loads(capsys.readouterr().out)['ok']


def test_desktop_extraction_rejects_truncated_stream_and_metadata_bomb(tmp_path,monkeypatch):
    class Archive:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def __iter__(self):
            item=tarfile.TarInfo('x');item.size=1
            return iter([item])
        def extractfile(self,item):return io.BytesIO(b'')
    with monkeypatch.context() as patch:
        patch.setattr(desktop.tarfile,'open',lambda **kw:Archive())
        with pytest.raises(ValueError,match='truncated'):desktop.unpack(io.BytesIO(b''),tmp_path,None,10)
    # A huge PAX metadata record is rejected before a giant allocation/read.
    header=tarfile.TarInfo('metadata');header.type=tarfile.XHDTYPE;header.size=16*1024**2
    with pytest.raises(ValueError,match='metadata'):desktop.unpack(io.BytesIO(header.tobuf()+b'x'*(2*1024**2)),tmp_path,0,0)
