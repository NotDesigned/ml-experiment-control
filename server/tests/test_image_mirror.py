"""OCI delivery is exact-host, streamed, digest-preserving and idempotent."""
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from ml_exp_server.builds import image_mirror as p
from ml_exp_server.builds.builder_client import BuildTransportError
from ml_exp_server.data.registry_transport import digest
from tests.test_image_builder_boundary import builder

RAW=b'{"immutable":"manifest"}'
SOURCE='tokyo.example/ml-expd/build@'+digest(RAW)
TAG='ccr.example/team/runtime:bundle-'+'a'*64
TARGET=TAG.split(':')[0]+'@'+digest(RAW)
DELIVERY={'source_image':SOURCE,'target_tag':TAG,'builder_socket':'/run/builder.sock'}


@pytest.fixture
def setup(tmp_path,monkeypatch):
    b=SimpleNamespace(root=tmp_path,config={'registry_auth_file':'/private/registry.json','sensecore_mirror':{
        'source_repository':SOURCE.split('@')[0],'target_repository':TAG.split(':')[0]}})
    state={'status':404,'raw':RAW};calls=[]
    registry=SimpleNamespace(prefix='/v2/team/runtime',request=lambda *a,**kw:(state['status'],{},state['raw']))
    monkeypatch.setattr(p,'Registry',SimpleNamespace(from_config=lambda *a,**kw:registry))
    def run(argv,**kw):
        calls.append((argv,kw));state['status']=200
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(p.subprocess,'run',run)
    return b,state,calls


def test_streamed_mirror_verifies_and_reuses_without_another_copy(setup):
    b,state,calls=setup
    assert p.mirror(b,DELIVERY)=={'image':TARGET,'source_image':SOURCE,'verified':True}
    assert p.mirror(b,DELIVERY)['image']==TARGET and len(calls)==1
    argv,kw=calls[0]
    assert argv[:3]==['/usr/bin/skopeo','copy','--preserve-digests']
    assert argv[-2:]==['docker://'+SOURCE,'docker://'+TAG]
    assert kw['stdout']==subprocess.DEVNULL and kw['stderr']==subprocess.DEVNULL
    assert list(b.root.rglob('*')) and not list(b.root.rglob('*.tar'))
    assert json.loads(next(b.root.glob('mirrors/*.json')).read_text())['status']=='VERIFIED'


@pytest.mark.parametrize('kind',['nonzero','oserror','timeout','lost-response'])
def test_copy_failure_is_observed_and_only_exact_late_publication_can_succeed(setup,monkeypatch,kind):
    b,state,calls=setup
    def run(*a,**kw):
        if kind=='nonzero':return SimpleNamespace(returncode=1)
        if kind=='lost-response':state['status']=200
        if kind=='timeout':raise subprocess.TimeoutExpired('copy',600)
        raise OSError('private transport text')
    monkeypatch.setattr(p.subprocess,'run',run)
    if kind=='lost-response':assert p.mirror(b,DELIVERY)['verified'] is True
    else:
        with pytest.raises(BuildTransportError,match='IMAGE_MIRROR_UNAVAILABLE') as e:p.mirror(b,DELIVERY)
        assert 'private transport' not in str(e.value)
        assert json.loads(next(b.root.glob('mirrors/*.json')).read_text())['status']=='RECONCILE_REQUIRED'


@pytest.mark.parametrize('source,tag',[('mutable:tag',TAG),(SOURCE,'wrong'),(SOURCE,TAG+'x')])
def test_only_pinned_source_and_bundle_destination_are_accepted(setup,source,tag):
    b,_,calls=setup
    with pytest.raises(ValueError):p.mirror(b,{'source_image':source,'target_tag':tag})
    assert not calls


@pytest.mark.parametrize('case',['source','target','same','existing-tag','existing-intent'])
def test_wrong_policy_and_conflicting_target_are_never_overwritten(setup,case):
    b,state,calls=setup
    if case in ['source','target']:b.config['sensecore_mirror'][case+'_repository']='other.example/repo'
    elif case=='same':b.config['sensecore_mirror']['target_repository']=SOURCE.split('@')[0]
    elif case=='existing-tag':state.update(status=200,raw=b'different manifest')
    else:
        p.mirror(b,DELIVERY);calls.clear()
        path=next(b.root.glob('mirrors/*.json'));v=json.loads(path.read_text());v['source_image']='other';path.write_text(json.dumps(v))
    with pytest.raises(ValueError):p.mirror(b,DELIVERY)
    assert not calls


def test_delivery_rpc_verifies_response_and_private_builder_router(setup,monkeypatch,builder):
    calls=[]
    monkeypatch.setattr(p,'builder_request',lambda socket,body:calls.append((socket,body)) or {'image':TARGET})
    assert p.ensure_delivery(DELIVERY)==TARGET
    assert calls[0][0]=='/run/builder.sock' and calls[0][1]['operation']=='mirror'
    monkeypatch.setattr(p,'builder_request',lambda *a:{'image':'wrong'})
    with pytest.raises(ValueError):p.ensure_delivery(DELIVERY)
    b,_,_=builder;monkeypatch.setattr(p,'mirror',lambda *a:{'image':TARGET})
    assert b.request({'operation':'mirror'})=={'image':TARGET}


@pytest.mark.parametrize('case',['absent','valid','kind','source','target'])
def test_run_delivery_must_match_its_frozen_runtime_and_backend(case):
    run={'container':{'image':SOURCE},'backend':{'kind':'sensecore','image':TAG,'image_delivery':DELIVERY}}
    if case=='absent':run['backend'].pop('image_delivery')
    elif case=='kind':run['backend']['kind']='slurm'
    elif case=='source':run['container']['image']='other'
    elif case=='target':run['backend']['image']='other'
    if case in ['kind','source','target']:
        with pytest.raises(ValueError):p.delivery_for_run(run)
    else:assert p.delivery_for_run(run)==(None if case=='absent' else DELIVERY)
