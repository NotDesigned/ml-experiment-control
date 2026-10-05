"""Standalone resumable uploads never replay scientific operations."""
import hashlib
import io
import json
from pathlib import Path

import pytest

from ml_exp_client.api import ClientError, upload_asset_parts, data_archive
from ml_exp_client.cli import main


class UploadClient:
    def __init__(self, data, fail=None, change=None):
        self.parts, self.requests = {}, []
        self.data, self.fail, self.change = data, fail, change

    def call(self, path, **kwargs):
        self.requests.append((path,kwargs))
        if path.endswith('/asset-uploads'):
            value={'upload_id':'upload.'+'a'*64,'status':'UPLOADING','part_bytes':1024,
                   'part_count':(len(self.data)+1023)//1024,'sha256':hashlib.sha256(self.data).hexdigest(),
                   'bytes':len(self.data),'parts':dict(self.parts)}
            if self.change:value.update(self.change)
            return value
        if '/parts/' in path:
            assert kwargs['method']=='PUT'
            number=path.split('/parts/')[1].split('?')[0]
            data=kwargs['raw']
            self.parts[number]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
            if self.fail:
                exc,self.fail=self.fail,None
                raise exc
            return self.parts[number]
        assert path.endswith('/complete')
        return {'project':'demo','asset_id':'asset.'+hashlib.sha256(self.data).hexdigest(),'sha256':hashlib.sha256(self.data).hexdigest(),'status':'READY'}


def run(client,tmp_path,data):
    return upload_asset_parts(client,'demo',io.BytesIO(data),hashlib.sha256(data).hexdigest(),len(data),tmp_path/'state.json')


@pytest.mark.parametrize('failure',[ClientError('lost response',retryable=True),ClientError('temporary outage',status=503),ClientError('rate limit',status=429)])
def test_lost_part_response_resumes_without_resending_committed_data(tmp_path,monkeypatch,failure):
    monkeypatch.setattr('ml_exp_client.api.time.sleep',lambda seconds:None)
    data=b'a'*1024+b'b'*900
    client=UploadClient(data,fail=failure)
    result=run(client,tmp_path,data)
    assert result['status']=='READY'
    assert len([p for p,_ in client.requests if '/parts/0?' in p])==1
    state=json.loads((tmp_path/'state.json').read_text())
    assert state['upload_id']=='upload.'+'a'*64 and 'token' not in state


@pytest.mark.parametrize('change',[{'part_bytes':1},{'part_bytes':65*1024**2},{'sha256':'a'*64},{'bytes':1000},{'part_count':10},{'upload_id':'../bad'}])
def test_invalid_receipt_is_not_retried(tmp_path,change):
    client=UploadClient(b'data',change=change)
    with pytest.raises(ClientError,match='receipt'):run(client,tmp_path,b'data')
    assert len(client.requests)==1


def test_completed_upload_conflicts_and_exhausted_retries(tmp_path,monkeypatch):
    monkeypatch.setattr('ml_exp_client.api.time.sleep',lambda seconds:None)
    client=UploadClient(b'data',change={'status':'COMPLETED','result':{'status':'READY'}})
    assert run(client,tmp_path,b'data')=={'status':'READY'}
    client=UploadClient(b'data',change={'parts':{'0':{'sha256':'a'*64,'bytes':4}}})
    with pytest.raises(ClientError,match='sealed'):run(client,tmp_path,b'data')
    client=UploadClient(b'data',fail=ClientError('rejected',status=401))
    with pytest.raises(ClientError):run(client,tmp_path,b'data')
    class Offline:
        def call(self,*args,**kwargs):raise ClientError('offline',retryable=True)
    with pytest.raises(ClientError):run(Offline(),tmp_path,b'data')


def test_cli_explicit_resume_checks_exact_archive(tmp_path,monkeypatch,capsys):
    directory=tmp_path/'data';directory.mkdir();(directory/'tokens.bin').write_bytes(b'data')
    stream=io.BytesIO();digest,length=data_archive(directory,stream)
    remote=UploadClient(stream.read())
    remote.negotiate=lambda:{'capabilities':['multipart-upload.v1']}
    call=remote.call
    remote.call=lambda path,**kw:{'asset_archive_bytes':4096} if path=='/api/storage-limits' else call(path,**kw)
    monkeypatch.setattr('ml_exp_client.cli.Client',lambda *args:remote)
    monkeypatch.setenv('ML_EXPD_API_TOKEN','local-test')
    state=tmp_path/'state.json'
    args=['asset-upload','--project','demo','--directory',str(directory),'--state',str(state)]
    assert main(args)==0
    assert json.loads(state.read_text())['status']=='READY'
    assert main(args)==2
    assert main(args+['--resume'])==0
    state.write_text(json.dumps({'project':'demo','asset_id':'asset.'+digest,'status':'UPLOADING'}))
    assert main(args+['--resume'])==0
    (directory/'tokens.bin').write_bytes(b'changed')
    assert main(args+['--resume'])==2
    assert 'differs' in capsys.readouterr().err
