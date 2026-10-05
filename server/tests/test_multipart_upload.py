"""Real archive publication, interrupted uploads, ownership and disk bounds."""
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.artifact_store import ArtifactStore
from ml_exp_server.multipart_upload import PartStream, UploadStore
from ml_exp_server.storage import atomic_json
from tests.test_sensecore_data_workflow import client, stored, custom_runtime, controller
from tests.test_artifact_store import archive, storage


def policy(stored):
    path, _ = stored
    config = json.loads(path.read_text())
    config['upload_part_bytes'] = 1024
    path.write_text(json.dumps(config))


def begin(client, data, endpoint='/api/projects/demo/asset-uploads', **kwargs):
    response = client.post(endpoint, json={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}, **kwargs)
    assert response.status_code == 200, response.text
    return response.json()


def send(client, data, value, endpoint='/api/projects/demo/asset-uploads', numbers=None, **kwargs):
    root = endpoint + '/' + value['upload_id']
    for number in range(value['part_count']) if numbers is None else numbers:
        part = data[number*value['part_bytes']:(number+1)*value['part_bytes']]
        response = client.put(root+'/parts/'+str(number), params={'sha256':hashlib.sha256(part).hexdigest()}, content=part, **kwargs)
        assert response.status_code == 200, response.text
    return root


def test_asset_upload_resumes_and_publishes_exactly_once(client, stored):
    policy(stored)
    data = archive({'tokens.bin':b'data'*700})
    value = begin(client,data)
    root = send(client,data,value,numbers=[0])
    send(client,data,value,numbers=[0])
    resumed = begin(client,data)
    assert resumed['upload_id'] == value['upload_id'] and list(resumed['parts']) == ['0']
    assert client.get(root).json() == resumed
    assert client.post(root+'/complete').status_code == 409
    send(client,data,resumed,numbers=range(1,value['part_count']))
    result = client.post(root+'/complete')
    assert result.status_code == 200, result.text
    result = result.json()
    assert result['sha256'] == hashlib.sha256(data).hexdigest()
    assert client.post(root+'/complete').json() == result
    assert begin(client,data)['result'] == result
    assert client.get('/api/projects/demo/assets/'+result['asset_id']+'/archive').content == data
    registry=client.app.state.runtime.config.project_registry_root_path()
    assert not (registry/'multipart-uploads'/value['upload_id']/'parts').exists()
    assert client.delete(root).status_code == 409
    assert client.put(root+'/parts/0',params={'sha256':hashlib.sha256(data[:1024]).hexdigest()},content=data[:1024]).status_code == 409


def test_bad_chunks_abort_and_expiry_never_publish(client, stored, monkeypatch):
    policy(stored)
    data=archive({'x.bin':b'x'*2000})
    value=begin(client,data)
    root='/api/projects/demo/asset-uploads/'+value['upload_id']
    endpoint=root+'/parts/0?sha256='+hashlib.sha256(data[:1024]).hexdigest()
    assert client.put(endpoint,content=b'x').status_code==400
    assert client.put(endpoint,content=b'x'*1024).status_code==400
    assert client.put(endpoint,content=b'x'*1025,headers={'Content-Length':'1024'}).status_code==413
    assert client.put(endpoint,content=b'x',headers={'Content-Length':'1024'}).status_code==400
    assert client.put(root+'/parts/-1?sha256='+'a'*64,content=b'x').status_code==409
    assert client.get(root).json()['parts']=={}
    send(client,data,value,numbers=[0])
    different=b'b'*1024
    assert client.put(root+'/parts/0?sha256='+hashlib.sha256(different).hexdigest(),content=different).status_code==409
    assert client.delete(root).json()['status']=='ABORTED'
    assert client.post(root+'/complete').status_code==409
    value=begin(client,data)
    send(client,data,value,numbers=[0])
    directory=client.app.state.runtime.config.project_registry_root_path()/'multipart-uploads'/value['upload_id']
    record=json.loads((directory/'upload.json').read_text())
    record['expires_at']='2000-01-01T00:00:00+00:00'
    atomic_json(directory/'upload.json',record)
    assert client.get(root).status_code==410
    assert client.delete(root).status_code==200
    assert not stored[1]
    value=begin(client,data)
    monkeypatch.setattr('ml_exp_server.api.upload_routes.shutil.disk_usage',lambda p:SimpleNamespace(free=0))
    assert client.put(endpoint,content=data[:1024]).status_code==507


@pytest.mark.parametrize('kind',['artifacts','checkpoint'])
def test_worker_multipart_is_bound_to_exact_attempt(client, stored, monkeypatch, kind):
    policy(stored)
    ctl=controller(client,custom_runtime(client,monkeypatch),checkpoint_upload={'interval_seconds':5})
    ctl.dispatch_command(ctl.store.load_attempt('attempt-001'))
    objects=ArtifactStore(stored[0],client.app.state.runtime.config.project_registry_root_path())
    with objects.record('demo','trial','attempt-001') as (_,record):token=record['token']
    headers={'Authorization':'Bearer '+token}
    endpoint='/api/attempt-uploads/demo/trial/attempt-001/'+kind
    data=archive({'checkpoint.pt':b'training state'})
    assert client.post(endpoint,json={'sha256':'a'*64,'bytes':100},headers={'Authorization':'Bearer wrong'}).status_code==401
    assert client.post(endpoint.replace('001','002'),json={'sha256':'a'*64,'bytes':100},headers=headers).status_code==401
    value=begin(client,data,endpoint,headers=headers)
    root=send(client,data,value,endpoint,headers=headers)
    assert client.get(root,headers=headers).json()['parts']
    assert client.post(root+'/complete',headers=headers).json()=={'status':'COMPLETED','sha256':hashlib.sha256(data).hexdigest()}
    assert begin(client,data,endpoint,headers=headers)['status']=='COMPLETED'
    assert 'result' not in client.get(root,headers=headers).json()
    assert client.delete(root,headers=headers).status_code==409
    if kind=='checkpoint':
        assert client.get('/api/runs/demo/trial/attempts/attempt-001/snapshots').json()['snapshots'][0]['sha256']==hashlib.sha256(data).hexdigest()
    else:
        with objects.record('demo','trial','attempt-001') as (_,record):assert record['receipt']['sha256']==hashlib.sha256(data).hexdigest()


def test_checkpoint_opt_in_limits_and_wrong_binding(client, stored, monkeypatch):
    policy(stored)
    ctl=controller(client,custom_runtime(client,monkeypatch))
    ctl.dispatch_command(ctl.store.load_attempt('attempt-001'))
    objects=ArtifactStore(stored[0],client.app.state.runtime.config.project_registry_root_path())
    with objects.record('demo','trial','attempt-001') as (_,record):token=record['token']
    assert client.post('/api/attempt-uploads/demo/trial/attempt-001/checkpoint',json={'sha256':'a'*64,'bytes':1},headers={'Authorization':'Bearer '+token}).status_code==401
    data=archive({'x':b'x'})
    value=begin(client,data)
    upload=UploadStore(client.app.state.runtime.config.project_registry_root_path(),json.loads(stored[0].read_text()))
    with pytest.raises(ApplicationError):upload.read(value['upload_id'],{'project':'other','kind':'asset'})
    assert client.post('/api/projects/demo/asset-uploads',json={'sha256':'a'*64,'bytes':4*1024**3+1}).status_code==413
    assert client.post('/api/projects/demo/asset-uploads',json={'sha256':'invalid','bytes':100}).status_code==422


def test_stream_supports_large_offsets_and_bounded_cross_part_reads(tmp_path):
    first,second=tmp_path/'0',tmp_path/'1'
    with first.open('wb') as stream:stream.truncate(2**32)
    second.write_bytes(b'0123456789abcdef')
    with io.BufferedReader(PartStream([first,second],2**32,2**32+16)) as stream:
        assert stream.readable() and stream.seekable()
        assert stream.seek(2**32-4)==2**32-4
        assert stream.read(8)==b'\0'*4+b'0123'
        assert stream.tell()==2**32+4
        assert stream.seek(-4,2)==2**32+12 and stream.read()==b'cdef'
        assert stream.seek(-4,1)==2**32+12
        assert stream.seek(2**32+32)==2**32+32 and stream.read()==b''
        with pytest.raises(ValueError):stream.seek(-1)
        with pytest.raises(ValueError):stream.seek(0,10)
    first.write_bytes(b'x')
    with pytest.raises(ValueError):PartStream([first],1024,10).read(10)


def test_upload_journal_rejects_corruption_and_cleans_owned_expired_staging(tmp_path,monkeypatch):
    store=UploadStore(tmp_path,{'upload_part_bytes':1024})
    binding={'kind':'asset','project':'demo'}
    data=b'a'*1024+b'b'*3
    value=store.create(binding,hashlib.sha256(data).hexdigest(),len(data),4096)
    directory=store.root/value['upload_id']
    for number,chunk in enumerate((data[:1024],data[1024:])):
        temporary=tmp_path/'part';temporary.write_bytes(chunk)
        store.part(value['upload_id'],binding,number,temporary,hashlib.sha256(chunk).hexdigest(),len(chunk))
    publish=lambda stream,size,digest:{'sha256':digest,'bytes':len(stream.read())}
    with pytest.raises(ApplicationError):store.complete(value['upload_id'],binding,publish,10**20)
    (directory/'parts/0').write_bytes(b'c'*1024)
    with pytest.raises(ValueError,match='checksum'):store.complete(value['upload_id'],binding,publish,0)
    (directory/'parts/0').write_bytes(data[:1024])
    record=json.loads((directory/'upload.json').read_text());record['sha256']='a'*64;atomic_json(directory/'upload.json',record)
    with pytest.raises(ValueError,match='archive SHA'):store.complete(value['upload_id'],binding,publish,0)
    record['expires_at']='2000-01-01T00:00:00+00:00';atomic_json(directory/'upload.json',record)
    (store.root/'upload.invalid').mkdir()
    (store.root/('upload.'+'c'*64)).mkdir()
    with (store.root/(value['upload_id']+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        store.prune()
        assert (directory/'parts').exists()
    store.prune()
    assert not (directory/'parts').exists()
    stale=store.root/'.part-stale';stale.write_bytes(b'stale')
    fresh=store.root/'.part-fresh';fresh.write_bytes(b'active')
    os.utime(stale,(1,1))
    store.prune()
    assert not stale.exists() and fresh.exists()
    assert store.create(binding,hashlib.sha256(data).hexdigest(),len(data),4096)['parts']=={}
    with pytest.raises(ValueError):store.read('bad',binding)
    with pytest.raises(ValueError):store.part(value['upload_id'],binding,0,tmp_path/'missing','a'*64,1)
    with pytest.raises(ApplicationError):store.create(binding,'bad',1,100)
    with pytest.raises(ApplicationError):store.create(binding,'a'*64,4097*1024,4097*1024)
    monkeypatch.setattr('ml_exp_server.multipart_upload.shutil.disk_usage',lambda p:SimpleNamespace(free=0))
    with pytest.raises(ApplicationError):store.create(binding,'a'*64,1,100)


@pytest.mark.parametrize('config',[{'upload_part_bytes':1000},{'upload_part_bytes':65*1024**2},{'upload_session_seconds':299},{'upload_session_seconds':604801}])
def test_invalid_multipart_policy_fails_closed(tmp_path,config):
    with pytest.raises(ValueError):UploadStore(tmp_path,config)


def test_attempt_capability_has_no_control_api_access(storage,tmp_path):
    from fastapi.testclient import TestClient
    from ml_exp_server.api.app import create_app
    from ml_exp_server.schemas import ServerConfig
    _,root,token,config,_=storage
    bearer=tmp_path/'bearer';bearer.write_text('a'*40);bearer.chmod(0o600)
    server=ServerConfig(index_db=str(tmp_path/'index.sqlite'),project_registry_root=str(tmp_path/'projects'),
        collector_enabled=False,telemetry={'enabled':False},http_auth={'bearer_token_file':str(bearer)},
        container_execution={'artifact_store_file':str(config)})
    with TestClient(create_app(server,poll=False)) as client:
        headers={'Authorization':'Bearer '+token}
        endpoint='/api/attempt-uploads/demo/run-a/attempt-001/artifacts'
        data=archive({'checkpoint.bin':b'result'})
        value=begin(client,data,endpoint,headers=headers)
        root=send(client,data,value,endpoint,headers=headers)
        assert client.post(root+'/complete',headers=headers).status_code==200
        for path in ['/api/projects','/api/projects/demo/asset-uploads','/api/attempt-uploads/demo/run-a/attempt-001/not-a-purpose']:
            assert client.get(path,headers=headers).status_code==401
        assert client.get(root,headers={}).status_code==401


def test_default_four_gib_boundary_uses_only_bounded_session_metadata(storage,tmp_path):
    from ml_exp_server.data_assets import AssetStore
    objects,_,_,config,_=storage
    assets=AssetStore(config,tmp_path/'registry')
    assert objects.limit==assets.limit==assets.expanded_limit==4*1024**3
    uploads=UploadStore(tmp_path/'registry',{})
    binding={'kind':'asset','project':'demo'}
    value=uploads.create(binding,'a'*64,4*1024**3,assets.limit)
    assert value['part_count']==256 and value['part_bytes']==16*1024**2
    assert not any((uploads.root/value['upload_id']/'parts').iterdir())
    assert uploads.abort(value['upload_id'],binding)['status']=='ABORTED'
    with pytest.raises(ApplicationError):uploads.create(binding,'b'*64,4*1024**3+1,assets.limit)


def test_disk_sync_failure_does_not_acknowledge_an_upload_part(tmp_path,monkeypatch):
    uploads=UploadStore(tmp_path,{})
    binding={'kind':'asset','project':'demo'}
    data=b'data'
    value=uploads.create(binding,hashlib.sha256(data).hexdigest(),len(data),100)
    temporary=tmp_path/'part';temporary.write_bytes(data)
    original=os.fsync
    monkeypatch.setattr(os,'fsync',lambda fd:(_ for _ in ()).throw(OSError('disk sync failed')))
    with pytest.raises(OSError):uploads.part(value['upload_id'],binding,0,temporary,hashlib.sha256(data).hexdigest(),len(data))
    monkeypatch.setattr(os,'fsync',original)
    assert uploads.read(value['upload_id'],binding)['parts']=={}
    temporary.write_bytes(data)
    uploads.part(value['upload_id'],binding,0,temporary,hashlib.sha256(data).hexdigest(),len(data))
    assert uploads.read(value['upload_id'],binding)['parts']['0']['sha256']==hashlib.sha256(data).hexdigest()
