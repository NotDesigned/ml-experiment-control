"""Dataset downloads verify every file and never send deletion receipts."""
import hashlib
import io
import json
import tarfile
from types import SimpleNamespace

import pytest

from ml_exp_client.api import ClientError
from ml_exp_client.assets import download_asset
from ml_exp_client.cli import main


def archive(files):
    raw=io.BytesIO()
    with tarfile.open(fileobj=raw,mode='w:gz') as package:
        for name,body in files.items():
            member=tarfile.TarInfo(name);member.size=len(body);package.addfile(member,io.BytesIO(body))
    return raw.getvalue()


def example():
    data=archive({'tokens/a.bin':b'dataset samples'})
    ticket={'transport':'oss-presigned-get','sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),
            'files':[{'path':'tokens/a.bin','bytes':15,'sha256':hashlib.sha256(b'dataset samples').hexdigest()}],
            'url':'https://private.example/data?signed=private'}
    ticket['asset_id']='asset.'+ticket['sha256']
    class API:
        def __init__(self):self.calls=[]
        def call(self,path,**kw):self.calls.append((path,kw));return ticket
        def open_object(self,url):assert url==ticket['url'];return io.BytesIO(data)
    return API(),ticket,data


def test_direct_dataset_download_hashes_archive_and_files_without_ack(tmp_path):
    api,ticket,data=example();out=tmp_path/'download'
    result=download_asset(api,'demo',ticket['asset_id'],out)
    assert result['verified'] and result['dataset_deleted'] is False
    assert (out/'data/tokens/a.bin').read_bytes()==b'dataset samples'
    assert (out/'dataset.tar').read_bytes()==data
    assert json.loads((out/'verification.json').read_text())==result
    assert len(api.calls)==1 and api.calls[0][1]=={}


@pytest.mark.parametrize('change',[{'transport':'other'},{'sha256':'bad'},{'asset_id':'asset.'+'b'*64},{'bytes':False},{'bytes':0},
                                  {'files':[]},{'files':[{'path':'../escape','bytes':0,'sha256':'a'*64}]},
                                  {'files':[{'path':'ok','bytes':True,'sha256':'a'*64}]},
                                  {'files':[{'path':'ok','bytes':0,'sha256':'bad'}]}])
def test_invalid_ticket_is_rejected_without_creating_output(tmp_path,change):
    api,ticket,_=example();identity=ticket['asset_id'];ticket.update(change)
    with pytest.raises(ClientError):download_asset(api,'demo',identity,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_corrupt_archive_file_hash_and_missing_files_never_report_success(tmp_path):
    for case in ['archive','file','missing','extra']:
        api,ticket,data=example()
        if case=='archive':ticket['sha256']='c'*64;ticket['asset_id']='asset.'+ticket['sha256']
        elif case=='file':ticket['files'][0]['sha256']='c'*64
        elif case=='missing':ticket['files'].append({'path':'missing','bytes':1,'sha256':'c'*64})
        else:ticket['files'][0]['path']='different'
        with pytest.raises(ClientError):download_asset(api,'demo',ticket['asset_id'],tmp_path/case)
        assert not (tmp_path/case/'verification.json').exists()


def test_cli_register_list_download_and_upload_capability(monkeypatch,tmp_path):
    calls=[]
    class API:
        def __init__(self,*args):pass
        def negotiate(self):return {'capabilities':[]}
        def call(self,path,**kw):calls.append((path,kw));return {}
    monkeypatch.setattr('ml_exp_client.cli.Client',API)
    assert main(['asset-download','--project','demo','--id','asset.'+'a'*64,'--out',str(tmp_path/'out')])==2 and calls==[]
    monkeypatch.setattr(API,'negotiate',lambda self:{'capabilities':['oss-datasets.v1']})
    assert main(['assets','--project','demo'])==0 and calls[-1][0]=='/api/projects/demo/assets'
    assert main(['asset-import-oss','--project','demo','--executor','pai','--object-key','users/owner/datasets/v1.tar','--sha256','a'*64,'--bytes','1024'])==0
    assert calls[-1][1]['data']['executor']=='pai' and calls[-1][1]['timeout']==1200
    from ml_exp_client import commands
    monkeypatch.setattr(commands,'download_asset',lambda *args:{'verified':True})
    assert main(['asset-download','--project','demo','--id','asset.'+'a'*64,'--out',str(tmp_path/'out')])==0
