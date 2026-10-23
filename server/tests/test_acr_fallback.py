"""Fallback preserves Git/context identity and never replays unknown publication."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_exp_server.builds import acr_fallback as p
from ml_exp_server.builds.builder_client import BuildTransportError
from ml_exp_server.data.registry_transport import MANIFEST, digest
from ml_exp_server.builds.image_build_context import BUILD_PROGRESS
from tests.test_image_builder_boundary import builder

ID='a'*64
REMOTE='https://github.com/example/private-build'
REPO='registry.example/lab/build'
RECEIPT={'bundle_id':ID,'source_id':'source.'+'b'*64,'project':'demo','base_image':'base@sha256:'+'c'*64}
RAW=json.dumps({'mediaType':MANIFEST,'config':{'digest':'sha256:'+'d'*64}}).encode()
IMAGE=REPO+'@'+digest(RAW)


@pytest.fixture
def setup(tmp_path,monkeypatch):
    b=SimpleNamespace(root=tmp_path,config={'repository':REPO,'acr_fallback':{'git_repository':REMOTE,'wait_seconds':0}})
    context=tmp_path/'context';context.mkdir();(context/'Dockerfile').write_text('FROM frozen\n')
    calls=[];state={'rows':b'','status':200,'config':{'os':'linux','architecture':'amd64','config':{'Labels':{'org.ml-expd.source':RECEIPT['source_id']}}}}
    def git(directory,args,**kw):
        calls.append(args)
        if args[0]=='ls-remote':return state['rows']
        return b''
    monkeypatch.setattr(p,'git',git)
    monkeypatch.setattr(p,'private_repository',lambda *args:None)
    monkeypatch.setattr(p,'commit_context',lambda *args:'e'*40)
    registry=SimpleNamespace(prefix='/v2/lab/build',json_blob=lambda sha:state['config'],
                             request=lambda *a,**kw:(state['status'],{},state.get('raw',RAW)))
    monkeypatch.setattr(p,'Registry',SimpleNamespace(from_config=lambda *a,**kw:registry))
    return b,context,calls,state


def test_git_plumbing_preserves_ignored_files_and_executable_mode_without_filters(tmp_path):
    context=tmp_path/'context';context.mkdir();(context/'sub').mkdir()
    (context/'.gitignore').write_text('ignored\n');(context/'ignored').write_bytes(b'payload\r\n')
    (context/'.gitattributes').write_text('* filter=untrusted\n');(context/'sub/run').write_text('echo ok\n');(context/'sub/run').chmod(0o755)
    directory=tmp_path/'git';directory.mkdir()
    first=p.commit_context(directory,context,ID)
    assert p.git(directory,['show',first+':ignored'])==b'payload\r\n'
    assert p.git(directory,['ls-tree',first,'sub/run']).startswith(b'100755 blob ')
    assert p.commit_context(directory,context,ID)==first


@pytest.mark.parametrize('case',['symlink','git','empty','special','large','total'])
def test_git_fallback_refuses_contexts_it_cannot_preserve(tmp_path,case,monkeypatch):
    context=tmp_path/'context';context.mkdir();directory=tmp_path/'git';directory.mkdir()
    if case=='symlink':(context/'link').symlink_to('missing')
    elif case=='git':(context/'.git').mkdir()
    elif case=='empty':(context/'empty').mkdir()
    elif case=='special':
        import os
        os.mkfifo(context/'fifo')
    else:
        for i in range(4 if case=='total' else 1):(context/str(i)).touch()
        original=Path.stat
        def stat(path,*a,**kw):
            value=original(path,*a,**kw)
            if path.parent==context:return SimpleNamespace(st_mode=value.st_mode,st_size=(96 if case=='large' else 90)*1024**2)
            return value
        monkeypatch.setattr(Path,'stat',stat)
    with pytest.raises(ValueError):p.commit_context(directory,context,ID)


def test_git_failure_is_bounded_and_never_exposes_stderr(tmp_path,monkeypatch):
    for result in [SimpleNamespace(returncode=1,stdout=b'',stderr=b'credential-secret'),
                   SimpleNamespace(returncode=0,stdout=b'x'*(1024**2+1),stderr=b'')]:
        monkeypatch.setattr(p.subprocess,'run',lambda *a,**kw:result)
        with pytest.raises(ValueError,match='Git operation failed') as e:p.git(tmp_path,['init'])
        assert 'credential-secret' not in str(e.value)


@pytest.mark.parametrize('status,body',[(0,b'true\n'),(0,b'false\n'),(1,b'true\n')])
def test_fallback_requires_verified_private_repository(monkeypatch,status,body):
    monkeypatch.setattr(p.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=status,stdout=body))
    if status==0 and body==b'true\n':p.private_repository(REMOTE)
    else:
        with pytest.raises(ValueError,match='verifiably private'):p.private_repository(REMOTE)


def test_published_fallback_pins_digest_and_same_tag_can_only_be_observed(setup):
    b,context,calls,state=setup
    progress=b.root/'progress.json';t=BUILD_PROGRESS.set(progress)
    try:assert p.publish(b,context,RECEIPT)==IMAGE
    finally:BUILD_PROGRESS.reset(t)
    assert sum(c[0]=='push' for c in calls)==1
    assert json.loads(p.journal(b,ID).read_text())['image']==IMAGE
    state['rows']=('e'*40+'\trefs/tags/release-vbundle-'+ID+'\n').encode()
    assert p.publish(b,context,RECEIPT)==IMAGE
    assert sum(c[0]=='push' for c in calls)==1
    definition={k:RECEIPT[k] for k in ['bundle_id','source_id','project','base_image']}
    assert p.reconcile(b,definition)=={**RECEIPT,'image':IMAGE}
    assert json.loads((b.root/(ID+'.json')).read_text())['image']==IMAGE


def test_timeout_and_lost_push_response_never_start_a_second_native_build(setup):
    b,context,calls,state=setup;state['status']=404
    with pytest.raises(BuildTransportError,match='PENDING'):p.publish(b,context,RECEIPT)
    assert p.reconcile(b,RECEIPT) is None
    with pytest.raises(BuildTransportError,match='UNCERTAIN'):p.publish(b,context,RECEIPT)
    assert sum(c[0]=='push' for c in calls)==1
    # A late publication is recovered without another Git call or DSW.
    state['status']=200
    assert p.reconcile(b,RECEIPT)['image']==IMAGE


def test_poll_waits_for_publication_without_retriggering(setup,monkeypatch):
    b,context,calls,state=setup;b.config['acr_fallback']['wait_seconds']=10;state['status']=404
    monkeypatch.setattr(p.time,'sleep',lambda seconds:state.update(status=200))
    assert p.publish(b,context,RECEIPT)==IMAGE
    assert sum(c[0]=='push' for c in calls)==1


@pytest.mark.parametrize('scope',[{'git_repository':'ssh://other'}, {'git_repository':None},
    {'git_repository':REMOTE,'wait_seconds':False},{'git_repository':REMOTE,'wait_seconds':2701}])
def test_invalid_fallback_config_performs_no_git_operation(setup,scope):
    b,context,calls,_=setup;b.config['acr_fallback']=scope
    with pytest.raises(BuildTransportError):p.publish(b,context,RECEIPT)
    assert calls==[]


@pytest.mark.parametrize('case',['changed','tag-collision','journal','repository','tag','receipt','media','source','platform','digest'])
def test_context_tag_and_image_conflicts_are_never_silently_adopted(setup,case,monkeypatch):
    b,context,calls,state=setup
    if case=='changed':
        values=iter(['sha256:before','sha256:after']);monkeypatch.setattr(p,'_tree_digest',lambda *a:next(values))
    elif case=='tag-collision':state['rows']=b'other\tref\n'
    elif case in ['media','source','platform']:
        if case=='media':state['raw']=b'{"mediaType":"unknown"}'
        elif case=='source':state['config']['config']['Labels']['org.ml-expd.source']='other'
        else:state['config']['os']='windows'
    else:
        assert p.publish(b,context,RECEIPT)==IMAGE
        v=json.loads(p.journal(b,ID).read_text())
        if case=='journal':v['context_digest']='other'
        elif case=='receipt':v['receipt']={**RECEIPT,'bundle_id':'wrong'}
        elif case=='digest':v['image']='other'
        else:v[case]='other'
        p.journal(b,ID).write_text(json.dumps(v))
        if case=='journal':
            with pytest.raises(BuildTransportError):p.publish(b,context,RECEIPT)
            return
        if case=='receipt':
            with pytest.raises(ValueError):p.reconcile(b,RECEIPT)
            return
        with pytest.raises(ValueError):p.publication(b,v)
        return
    with pytest.raises(BuildTransportError):p.publish(b,context,RECEIPT)


@pytest.mark.parametrize('case',['success','disabled','uncertain','fallback','existing'])
def test_fallback_requires_positive_primary_absence_and_existing_intent_never_recreates_dsw(setup,case,monkeypatch):
    from ml_exp_server.builds import pai_dsw,pai_build_recovery
    b,context,calls,state=setup;native=[];proof=[]
    def primary(*a):
        native.append('DSW')
        if case=='success':return IMAGE
        raise BuildTransportError('PRIMARY_FAILED',{})
    monkeypatch.setattr(pai_dsw,'DswPublisher',lambda *a:SimpleNamespace(publish=primary))
    monkeypatch.setattr(pai_build_recovery,'observe_dsw_absence',lambda *a:proof.append('checked') or (None if case=='uncertain' else {'leases_clear':True}))
    if case=='disabled':b.config.pop('acr_fallback')
    if case=='existing':
        p.publish(b,context,RECEIPT)
        state['rows']=('e'*40+'\trefs/tags/release-vbundle-'+ID+'\n').encode()
    if case in ['disabled','uncertain']:
        with pytest.raises(BuildTransportError,match='PRIMARY_FAILED'):p.publish_runtime(b,'tag',context,RECEIPT)
    else:assert p.publish_runtime(b,'tag',context,RECEIPT)==IMAGE
    assert native==([] if case=='existing' else ['DSW'])
    assert proof==(['checked'] if case in ['uncertain','fallback'] else [])


def test_pending_native_build_cannot_grant_primary_retry(setup):
    from ml_exp_server.builds.build_recovery import publication_absence
    b,context,calls,state=setup;p.publish(b,context,RECEIPT)
    assert publication_absence(b,ID) is None


@pytest.mark.parametrize('late',[False,True])
def test_deleted_dsw_can_resume_fallback_and_async_deletion_is_observed_before_switch(setup,monkeypatch,late):
    from ml_exp_server.builds import pai_dsw,pai_build_recovery
    b,context,calls,state=setup;lease=b.root/(ID+'.dsw.json');observations=[];sleeps=[]
    def primary(*a):
        assert late
        lease.write_text('{"phase":"DELETE_REQUESTED"}')
        raise BuildTransportError('PRIMARY_FAILED',{})
    if not late:lease.write_text('{"phase":"DELETE_REQUESTED"}')
    def observe(*a):
        observations.append(1)
        return None if late and len(observations)==1 else {'leases_clear':True}
    monkeypatch.setattr(pai_dsw,'DswPublisher',lambda *a:SimpleNamespace(publish=primary))
    monkeypatch.setattr(pai_build_recovery,'observe_dsw_absence',observe)
    monkeypatch.setattr(p.time,'sleep',lambda n:sleeps.append(n))
    assert p.publish_runtime(b,'tag',context,RECEIPT)==IMAGE
    assert sleeps==([3] if late else [])


@pytest.mark.parametrize('phase',['CREATED','DELETE_REQUESTED'])
def test_unknown_primary_or_expired_absence_wait_cannot_trigger_fallback(setup,monkeypatch,phase):
    from ml_exp_server.builds import pai_dsw,pai_build_recovery
    b,context,calls,state=setup
    def primary(*a):
        (b.root/(ID+'.dsw.json')).write_text(json.dumps({'phase':phase}))
        raise BuildTransportError('PRIMARY_FAILED',{})
    monkeypatch.setattr(pai_dsw,'DswPublisher',lambda *a:SimpleNamespace(publish=primary))
    monkeypatch.setattr(pai_build_recovery,'observe_dsw_absence',lambda *a:None)
    times=iter([0,61]);monkeypatch.setattr(p.time,'monotonic',lambda:next(times))
    with pytest.raises(BuildTransportError,match='PRIMARY_FAILED'):p.publish_runtime(b,'tag',context,RECEIPT)
    assert calls==[] and not p.journal(b,ID).exists()


def test_image_builder_recovers_late_fallback_receipt_via_reconcile(builder,monkeypatch):
    b,request,_=builder
    result=b.request(request)
    identity=result['bundle_id'];(b.root/(identity+'.json')).unlink()
    (b.root/(identity+'.acr.json')).write_text('{}')
    monkeypatch.setattr(p,'reconcile',lambda *args:result)
    assert b.request({**request,'operation':'reconcile'})==result
    monkeypatch.setattr(p,'reconcile',lambda *args:None)
    with pytest.raises(ValueError,match='not been published'):b.request({**request,'operation':'reconcile'})


def test_forty_five_minute_observation_survives_old_limit_without_retriggering(setup,monkeypatch,tmp_path):
    b,context,calls,state=setup;b.config['acr_fallback']['wait_seconds']=2700;state['status']=404
    moments=iter([0,301,2701]);monkeypatch.setattr(p.time,'monotonic',lambda:next(moments))
    sleeps=[];monkeypatch.setattr(p.time,'sleep',sleeps.append)
    token=BUILD_PROGRESS.set(tmp_path/'progress.json')
    try:
        with pytest.raises(BuildTransportError,match='PENDING'):p.publish(b,context,RECEIPT)
    finally:BUILD_PROGRESS.reset(token)
    assert sleeps==[5] and sum(c[0]=='push' for c in calls)==1
    assert json.loads((tmp_path/'progress.json').read_text())['timeout_seconds']==2700
