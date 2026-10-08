import hashlib
import io
import json
import runpy
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_exp_server.workers import data_copy_worker as worker
from ml_exp_server.workers import managed_worker
from tests.test_artifact_store import archive


@pytest.mark.parametrize('url',['http://example/api/data-copy-transfers/x','https://user:pass@example/api/data-copy-transfers/x',
    'https://example/api/data-copy-transfers/x?q=x','https://example/other','https://example/api/data-copy-transfers/x#x'])
def test_data_copy_callback_requires_fixed_https(url):
    with pytest.raises(ValueError):worker.notify(url,'token',{})


@pytest.mark.parametrize('status',[200,401])
@pytest.mark.parametrize('prefix',['','/ml-expd'])
def test_data_copy_callback_closes_its_connection(monkeypatch,status,prefix):
    closed=[]
    class Connection:
        def __init__(self,*a,**kw):pass
        def request(self,method,path,**kw):
            assert method=='PUT' and kw['headers']['Authorization']=='Bearer exact'
        def getresponse(self):return SimpleNamespace(status=status,read=lambda n:b'{}')
        def close(self):closed.append(True)
    monkeypatch.setattr(worker.http.client,'HTTPSConnection',Connection)
    if status==200:worker.notify('https://example'+prefix+'/api/data-copy-transfers/x','exact',{})
    else:
        with pytest.raises(ValueError):worker.notify('https://example'+prefix+'/api/data-copy-transfers/x','exact',{})
    assert closed==[True]


@pytest.mark.parametrize('case',['success','corrupt','destination','notify-failed','timeout','budget','relative'])
def test_cpu_data_worker_uses_image_payload_and_verified_cache(tmp_path,monkeypatch,case):
    data=archive({'tokens.bin':b'expected'});payload=tmp_path/'payload';payload.mkdir()
    item={'asset_id':'asset.'+hashlib.sha256(data).hexdigest(),'sha256':hashlib.sha256(data).hexdigest(),'archive_bytes':len(data),
        'files':[{'path':'tokens.bin','bytes':8,'sha256':hashlib.sha256(b'expected').hexdigest()}]}
    (payload/'asset.json').write_text(json.dumps(item));(payload/'dataset.tar').write_bytes(data if case!='corrupt' else b'corrupt')
    real=worker.Path
    monkeypatch.setattr(worker,'Path',lambda value:payload/str(value).split('/')[-1] if str(value).startswith('/payload/') else real(value))
    for key,value in {'URL':'https://example/api/data-copy-transfers/x','TOKEN':'secret','ROOT':str(tmp_path/'cache') if case!='relative' else 'cache',
            'IMAGE':'registry.example/data@sha256:'+'a'*64,'SECONDS':'0' if case=='budget' else '30'}.items():
        monkeypatch.setenv('ML_EXPD_DATA_COPY_'+key,value)
    received=[];alarms=[]
    def notify(url,token,value):
        assert token=='secret';received.append(value)
        if case=='notify-failed':raise OSError('disconnected')
    monkeypatch.setattr(worker,'notify',notify)
    monkeypatch.setattr(worker.signal,'alarm',lambda n:alarms.append(n))
    if case=='destination':monkeypatch.setattr(worker,'deliver',lambda *a,**kw:tmp_path/'other')
    if case=='timeout':
        def expired(signum,handler):
            if callable(handler):handler(signum,None)
        # Deliver raises the same timeout as the bounded worker's alarm.
        monkeypatch.setattr(worker,'deliver',lambda *a,**kw:(_ for _ in ()).throw(TimeoutError()))
    if case in {'budget','relative'}:
        with pytest.raises(ValueError):worker.main()
        assert not received
    else:
        assert worker.main()==(0 if case=='success' else 65)
        assert received[-1]['status']==('READY' if case=='success' else 'FAILED')
        assert alarms==[30,0]
        assert 'ML_EXPD_DATA_COPY_TOKEN' not in worker.os.environ
        if case=='success':
            assert managed_worker.deliver({**item,'require_cached':True},'no-network',tmp_path/'cache').is_dir()
            with pytest.raises(ValueError,match='cache'):managed_worker.deliver({**item,'require_cached':True},'',tmp_path/'absent')


def test_fixed_copy_entrypoint_and_alarm_handler(tmp_path,monkeypatch):
    from ml_exp_server import worker_contract
    worker_contract.install_workers(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(sys.modules,'worker',managed_worker)
    monkeypatch.setenv('ML_EXPD_DATA_COPY_URL','https://example/api/data-copy-transfers/x')
    monkeypatch.setenv('ML_EXPD_DATA_COPY_TOKEN','secret')
    monkeypatch.setenv('ML_EXPD_DATA_COPY_ROOT',str(tmp_path/'cache'))
    monkeypatch.setenv('ML_EXPD_DATA_COPY_IMAGE','registry.example/data@sha256:'+'a'*64)
    monkeypatch.setenv('ML_EXPD_DATA_COPY_SECONDS','5')
    handlers=[]
    monkeypatch.setattr(worker.signal,'signal',lambda sig,handler:handlers.append(handler))
    monkeypatch.setattr(worker.signal,'alarm',lambda seconds:None)
    # Running the packaged script exercises its standalone imports/entrypoint;
    # an absent payload fails locally without contacting a scheduler.
    monkeypatch.setattr(worker.http.client,'HTTPSConnection',lambda *a,**kw:(_ for _ in ()).throw(OSError('offline')))
    with pytest.raises(SystemExit) as result:runpy.run_path(worker.__file__,run_name='__main__')
    assert result.value.code==65
    with pytest.raises(TimeoutError):handlers[0](signal.SIGALRM,None)
