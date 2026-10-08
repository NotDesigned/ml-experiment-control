"""New image workers resume exact-Attempt checkpoints and final archives."""
import hashlib
import io
import json
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from ml_exp_server.workers import worker_artifacts as worker
from ml_exp_server.results.artifact_store import ArtifactStore
from tests.test_multipart_upload import client, stored, policy, archive
from tests.test_sensecore_data_workflow import custom_runtime, controller


@pytest.mark.parametrize('kind',['artifacts','checkpoint'])
def test_real_routes_resume_worker_after_committed_part_response_is_lost(client,stored,monkeypatch,kind):
    policy(stored)
    ctl=controller(client,custom_runtime(client,monkeypatch),checkpoint_upload={'interval_seconds':5})
    ctl.dispatch_command(ctl.store.load_attempt('attempt-001'))
    objects=ArtifactStore(stored[0],client.app.state.runtime.config.project_registry_root_path())
    with objects.record('demo','trial','attempt-001') as (_,record):token=record['token']
    calls=[]
    lost=False
    def request(target,method,path,capability,data=b''):
        nonlocal lost
        calls.append((method,path))
        response=client.request(method,path,headers={'Authorization':'Bearer '+capability,'Content-Type':'application/json' if method=='POST' else 'application/octet-stream'},content=data)
        assert response.status_code==200,response.text
        if '/parts/0?' in path and not lost:
            lost=True
            raise OSError('response lost after commit')
        return response.json()
    monkeypatch.setattr(worker,'upload_request',request)
    monkeypatch.setattr(worker.time,'sleep',lambda seconds:None)
    data=archive({'weights.pt':b'weights'*500})
    marker='snapshot-transfers' if kind=='checkpoint' else 'artifact-transfers'
    url='https://api.example/api/'+marker+'/demo/trial/attempt-001'
    worker.upload_parts(url,token,io.BytesIO(data),len(data))
    assert len([p for _,p in calls if '/parts/0?' in p])==1
    before=len(calls)
    worker.upload_parts(url,token,io.BytesIO(data),len(data))
    assert len(calls)==before+1


def test_worker_http_transport_is_bounded_and_authenticates_each_request(monkeypatch):
    recorded=[]
    status=200
    class Connection:
        def __init__(self,*args,**kwargs):recorded.append(('connection',args,kwargs))
        def request(self,*args,**kwargs):recorded.append(('request',args,kwargs))
        def getresponse(self):return SimpleNamespace(status=status,read=lambda limit:b'{"status":"UPLOADING"}')
        def close(self):recorded.append(('closed',))
    monkeypatch.setattr(worker.http.client,'HTTPSConnection',Connection)
    target=urlsplit('https://api.example:444/api/path')
    assert worker.upload_request(target,'POST','/api/path','capability',b'{}')['status']=='UPLOADING'
    assert recorded[1][2]['headers']['Authorization']=='Bearer capability'
    status=503
    with pytest.raises(OSError):worker.upload_request(target,'POST','/api/path','capability')
    status=429
    with pytest.raises(OSError):worker.upload_request(target,'POST','/api/path','capability')
    status=401
    with pytest.raises(ValueError):worker.upload_request(target,'POST','/api/path','wrong')
    assert recorded[-1]==('closed',)


@pytest.mark.parametrize('url',['http://api.example/api/artifact-transfers/p/r/attempt-001','https://user:password@api.example/api/artifact-transfers/p/r/attempt-001','https://api.example/api/artifact-transfers/p/r/attempt-001?x=1','https://api.example/api/artifact-transfers/p/r/attempt-001#x','https://api.example/api/not-transfer'])
def test_worker_rejects_unbound_or_redirectable_transfer_urls(url):
    with pytest.raises(ValueError):worker.upload_parts(url,'capability',io.BytesIO(b'data'),4)


@pytest.mark.parametrize('change',[{'part_bytes':100},{'bytes':8},{'sha256':'a'*64},{'part_count':3},{'upload_id':'../bad'}, {'parts':{'0':{'sha256':'a'*64,'bytes':4}}}])
def test_worker_invalid_receipts_never_upload(change,monkeypatch):
    data=b'data'
    value={'upload_id':'upload.'+'a'*64,'status':'UPLOADING','part_bytes':1024,'part_count':1,
           'sha256':hashlib.sha256(data).hexdigest(),'bytes':4,'parts':{}}
    value.update(change)
    monkeypatch.setattr(worker,'upload_request',lambda *args:value)
    with pytest.raises(ValueError):worker.upload_parts('https://api.example/api/artifact-transfers/p/r/attempt-001','capability',io.BytesIO(data),4)


def test_worker_exhausts_only_transient_archive_retries(monkeypatch):
    calls=[]
    def offline(*args):
        calls.append(args)
        raise OSError('offline')
    monkeypatch.setattr(worker,'upload_request',offline)
    monkeypatch.setattr(worker.time,'sleep',lambda seconds:None)
    with pytest.raises(OSError):worker.upload_parts('https://api.example/api/artifact-transfers/p/r/attempt-001','capability',io.BytesIO(b'data'),4)
    assert len(calls)==3
