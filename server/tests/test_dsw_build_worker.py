"""Docker API wire boundaries and sealed context publication in CPU DSW."""
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid
import pytest
from ml_exp_worker import dsw_build_worker as w

TAG='registry.example/repository:bundle-'+'a'*64
IMAGE='registry.example/repository@sha256:'+'b'*64

@pytest.fixture
def root():
    path=Path('/tmp/ml-expd-build-'+uuid.uuid4().hex);path.mkdir(mode=0o700)
    (path/'auth').mkdir();(path/'auth/config.json').write_text(json.dumps({'auths':{'registry.example':{'auth':base64.b64encode(b'u:password').decode()}}}))
    (path/'context.tar').write_bytes(b'sealed context')
    try:yield path
    finally:shutil.rmtree(path)


def spec(root):
    return {'tag':TAG,'context_sha256':hashlib.sha256((root/'context.tar').read_bytes()).hexdigest()}


def test_real_socket_connect_uses_only_private_unix_proxy(monkeypatch):
    calls=[]
    class Socket:
        def settimeout(self,value):calls.append(('timeout',value))
        def connect(self,path):calls.append(('connect',path))
    monkeypatch.setattr(w.socket,'socket',lambda *args:calls.append(args) or Socket())
    c=w.Engine('localhost',timeout=42);c.connect()
    assert calls==[(w.socket.AF_UNIX,w.socket.SOCK_STREAM),('timeout',42),('connect','/var/docker/proxy/docker-proxy.sock')]


@pytest.mark.parametrize('case',['metadata','stream','http','long_metadata','long_line','many_lines','error','detail','malformed'])
def test_engine_replies_are_bounded_and_stream_errors_fail(case,monkeypatch,capsys):
    closed=[];calls=[]
    raw={'metadata':b'{"Version":"28"}', 'stream':b'{"stream":"built"}\n', 'http':b'private',
         'long_metadata':b' '*65537,'long_line':b' '*65537,
         'many_lines':b'{"stream":"'+b'x'*60000+b'"}\n',
         'error':b'{"error":"failed"}\n','detail':b'{"errorDetail":{"message":"failed"}}\n','malformed':b'not-json'}[case]
    response=io.BytesIO(raw*(40 if case=='many_lines' else 1));response.status=503 if case=='http' else 200
    class Connection:
        def __init__(self,*args,**kwargs):pass
        def request(self,*args,**kwargs):calls.append((args,kwargs))
        def getresponse(self):return response
        def close(self):closed.append(True)
    monkeypatch.setattr(w,'Engine',Connection)
    if case in {'metadata','stream'}:
        v=w.request('GET','/version',stream=case=='stream')
        assert v==({'Version':'28'} if case=='metadata' else None)
    else:
        with pytest.raises(ValueError):w.request('GET','/version',stream=case not in {'long_metadata','http','malformed'})
    assert closed==[True] and calls[0][0]==('GET','/version')


@pytest.mark.parametrize('case',['success','hash','tag','digest_missing','digest_duplicate','invalid_root'])
def test_context_checksum_registry_auth_and_exact_digest(root,monkeypatch,capsys,case):
    p=spec(root);calls=[]
    if case=='hash':p['context_sha256']='0'*64
    if case=='tag':p['tag']='other:latest'
    def request(method,path,body=None,headers=None,**kwargs):
        calls.append((method,path,headers,body.read() if body else None,kwargs))
        if method=='GET':return {'RepoDigests':[] if case=='digest_missing' else [IMAGE,IMAGE] if case=='digest_duplicate' else ['other@sha256:'+'b'*64,IMAGE]}
    monkeypatch.setattr(w,'request',request)
    if case=='invalid_root':
        with pytest.raises(ValueError):w.run(Path('/tmp/unowned'),p)
    elif case!='success':
        with pytest.raises(ValueError):w.run(root,p)
    else:
        receipt=w.run(root,p)
        assert calls[0][0]=='POST' and 'version=1' in calls[0][1] and calls[0][3]==b'sealed context'
        credentials=json.loads(base64.urlsafe_b64decode(calls[0][2]['X-Registry-Config']))
        assert credentials['registry.example']['password']=='password'
        assert 'push?' in calls[1][1] and calls[2][0]=='GET'
        assert receipt=={'image':IMAGE,'context_sha256':p['context_sha256']}


def test_entrypoint_always_removes_only_owned_auth(root,monkeypatch):
    (root/'request.json').write_text(json.dumps(spec(root)))
    monkeypatch.setattr(w,'run',lambda *args:(_ for _ in ()).throw(ValueError('failed')))
    with pytest.raises(ValueError):w.main([str(root)])
    assert not (root/'auth').exists() and (root/'context.tar').exists()
    with pytest.raises(FileNotFoundError):w.main(['/tmp/unowned-dsw'])


def test_completed_receipt_is_atomic_and_entrypoint_cannot_keep_credentials(root,monkeypatch):
    (root/'request.json').write_text(json.dumps(spec(root)))
    receipt={'image':IMAGE,'context_sha256':spec(root)['context_sha256']}
    monkeypatch.setattr(w,'run',lambda *args:receipt)
    monkeypatch.setattr(w.sys,'argv',['worker.py',str(root)])
    w.main()
    assert json.loads((root/'status.json').read_text())=={'status':'SUCCEEDED','receipt':receipt}
    assert not (root/'status.tmp').exists() and not (root/'auth').exists()


def test_python_module_entrypoint_rejects_unowned_directory(monkeypatch):
    import runpy
    monkeypatch.setattr(w.sys,'argv',['worker.py','/tmp/unowned-dsw-entrypoint'])
    with pytest.raises(FileNotFoundError):runpy.run_path(w.__file__,run_name='__main__')


@pytest.mark.parametrize('images',[None,'bad',[None],['registry/base:tag'],['x']*65])
def test_prepull_rejects_unpinned_or_unbounded_image_list(images):
    with pytest.raises(ValueError):w.pull_bases(images,{})


def test_only_pull_is_retried_before_build_or_push(monkeypatch,capsys):
    calls=[];delays=[]
    def request(method,path,**kw):
        calls.append((method,path,kw))
        if len(calls)<3:raise OSError('connection reset by peer')
    monkeypatch.setattr(w,'request',request);monkeypatch.setattr(w.time,'sleep',delays.append)
    w.pull_bases([IMAGE],{})
    assert len(calls)==3 and delays==[1,2]
    assert all('/images/create?' in c[1] for c in calls)
    assert 'PULLING_BASE_IMAGE' in capsys.readouterr().out
    monkeypatch.setattr(w,'request',lambda *a,**k:(_ for _ in ()).throw(ValueError('reset')))
    with pytest.raises(ValueError):w.pull_bases([IMAGE],{})

@pytest.mark.parametrize('seconds',[None,True,0,-1,1.5,'1200'])
def test_invalid_worker_budget_is_rejected_before_docker(root,monkeypatch,seconds):
    value=spec(root);value['timeout_seconds']=seconds;calls=[]
    monkeypatch.setattr(w,'request',lambda *a,**kw:calls.append(a))
    with pytest.raises(ValueError,match='positive integer'):w.run(root,value)
    assert not calls


def test_worker_passes_twenty_minute_budget_to_pull_build_push_and_inspect(root,monkeypatch):
    value=spec(root);value.update(timeout_seconds=1200,base_images=[IMAGE]);calls=[]
    def request(method,path,**kwargs):
        calls.append((method,path,kwargs))
        if method=='GET':return {'RepoDigests':[IMAGE]}
    monkeypatch.setattr(w,'request',request)
    assert w.run(root,value)['image']==IMAGE
    assert len(calls)==4 and all(row[2]['timeout']==1200 for row in calls)


def test_docker_socket_uses_configured_read_timeout(monkeypatch):
    seen=[]
    class Connection:
        def __init__(self,*args,**kw):seen.append(kw['timeout'])
        def request(self,*a,**kw):pass
        def getresponse(self):
            response=io.BytesIO(b'{}');response.status=200;return response
        def close(self):pass
    monkeypatch.setattr(w,'Engine',Connection)
    assert w.request('GET','/version',timeout=1200)=={}
    assert seen==[1200]
