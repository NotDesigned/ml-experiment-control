"""Owned CPU builder lifecycle, bounded MCP transfers and independent OCI proof."""
import base64
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import pytest
from ml_exp_server.builds import pai_dsw as p
from ml_exp_server.builds.builder_client import BuildTransportError
from ml_exp_server.builds.image_build_context import BUILD_LOG, BUILD_PROGRESS
from ml_exp_server.data.registry_transport import digest
from tests.test_image_builder_boundary import builder

RAW=b'{"manifest":"proof"}'
IMAGE='registry.example/runtime@'+digest(RAW)
TAG='registry.example/runtime:bundle-'+'a'*64

class API:
    def __init__(self):
        self.config={'region':'us-southeast-1','access_key_id':'secret-ak','access_key_secret':'secret-sk','security_token':'secret-token'}
        self.calls=[];self.value=None;self.state='Running';self.initial=None
    def request(self,product,action,path,**kwargs):
        self.calls.append((action,path,kwargs))
        if action=='CreateInstance':
            body=kwargs['body']
            self.value={**body,'InstanceId':'dsw-owned'}
            return {'InstanceId':'dsw-owned'}
        if action=='GetInstance':
            if self.initial:
                state=self.initial.pop(0)
                return {**self.value,'Status':state}
            return {**self.value,'Status':self.state}
        if action=='GetInstanceShutdownTimer':return {'InstanceId':'dsw-owned','RemainingTimeInMs':1799999,'Success':True}
        if action=='StopInstance':self.state='Stopped'
        return {}

@pytest.fixture
def publisher(tmp_path,monkeypatch):
    auth=tmp_path/'auth.json';auth.write_text(json.dumps({'auths':{'registry.example':{'auth':base64.b64encode(b'u:registry-password').decode()}}}))
    config={'pai_dsw':{'workspace_id':'253','quota_id':'quota-fixed','image_id':'image-cpu'},
            'repository':'registry.example/runtime','registry_auth_file':str(auth)}
    api=API();monkeypatch.setattr(p.PaiAPI,'from_environment',lambda:api)
    pub=p.DswPublisher(SimpleNamespace(config=config,root=tmp_path))
    context=tmp_path/'context';context.mkdir();(context/'Dockerfile').write_text('FROM pinned\n')
    calls=[]
    def cli(args,private,**kwargs):
        calls.append((args,private,kwargs))
        if args[:3]==['dsw','instance','exec'] and 'status.json' in args[-1]:
            spec=json.loads(next(tmp_path.glob('dsw-context-*/request.json')).read_text())
            return 'ML_EXPD_DSW_STATUS_JSON='+json.dumps({'status':'SUCCEEDED','log_offset':1,'receipt':{'image':IMAGE,'context_sha256':spec['context_sha256']}})
        return ''
    pub.cli=cli
    class Registry:
        def __init__(self,*args):self.prefix='/v2/runtime'
        def request(self,*args,**kwargs):return 200,{},RAW
    monkeypatch.setattr(p,'Registry',Registry)
    monkeypatch.setattr(p.time,'sleep',lambda _:None)
    return pub,context,api,calls


def test_publisher_uses_only_fixed_cpu_quota_and_cleans_every_success(publisher,tmp_path):
    pub,context,api,calls=publisher
    progress=tmp_path/'progress.json';t=BUILD_PROGRESS.set(progress)
    try:assert pub.publish(TAG,context)==IMAGE
    finally:BUILD_PROGRESS.reset(t)
    body=api.calls[0][2]['body']
    assert body['ResourceId']=='quota-fixed' and body['RequestedResource']=={'CPU':'4','Memory':'8Gi','GPU':'0'}
    assert body['Accessibility']=='PRIVATE' and body['DockerConfig']['Enable']
    assert 'Datasets' not in body
    assert [c[0] for c in api.calls].count('CreateInstance')==1
    assert any(c[0]=='CreateInstanceShutdownTimer' and c[2]['body']['RemainingTimeInMs']==1800000 for c in api.calls)
    assert api.calls[-1][0]=='DeleteInstance'
    assert json.loads(pub.lease.read_text())['phase']=='DELETE_REQUESTED'
    assert not list(tmp_path.glob('dsw-context-*')) and len([c for c in calls if 'cp' in c[0]])==4
    assert json.loads(progress.read_text())['events'][-1]['phase']=='STOPPING_DSW_BUILDER'
    with pytest.raises(BuildTransportError,match='RECONCILE_REQUIRED'):pub.publish(TAG,context)
    assert [c[0] for c in api.calls].count('CreateInstance')==1


@pytest.mark.parametrize('internal,mount', [(True, None), (False, '/data/account-oss')])
def test_explicit_oss_prefix_is_mounted_and_recorded_in_owned_lease(publisher,internal,mount):
    pub,context,api,_=publisher
    uri='oss://example-bucket.oss-us-southeast-1'+('-internal' if internal else '')+'.aliyuncs.com/users/researcher/'
    pub.config['oss_uri']=uri
    if mount is not None:pub.config['oss_mount_path']=mount
    expected=[{'Uri':uri,'MountPath':mount or '/mnt/ml-expd-oss','MountAccess':'RW'}]
    assert pub.publish(TAG,context)==IMAGE
    assert api.calls[0][2]['body']['Datasets']==expected
    assert json.loads(pub.lease.read_text())['datasets']==expected
    assert api.calls[-1][0]=='DeleteInstance'


@pytest.mark.parametrize('settings', [
    {'oss_mount_path':'/mnt/account'}, {'oss_uri':None}, {'oss_uri':3},
    {'oss_uri':'oss://example-bucket.oss-us-southeast-1-internal.aliyuncs.com/users/researcher/','oss_mount_path':None},
    *[{'oss_uri':uri} for uri in [
        'oss://example-bucket.oss-us-southeast-1-internal.aliyuncs.com/',
        'oss://example-bucket.oss-us-southeast-1-internal.aliyuncs.com/users//researcher/',
        'oss://example-bucket.oss-us-southeast-1-internal.aliyuncs.com/users/../researcher/',
        'oss://example-bucket.oss-us-southeast-1-internal.aliyuncs.com/users/./researcher/',
        'oss://example-bucket.oss-us-southeast-1-internal.aliyuncs.com/users/researcher/?key=secret',
        'oss://example-bucket.oss-us-southeast-1-internal.aliyuncs.com/users/researcher/#fragment',
        'oss://example-bucket.oss-cn-hangzhou-internal.aliyuncs.com/users/researcher/',
        'oss://user:password@example-bucket.oss-us-southeast-1-internal.aliyuncs.com/users/researcher/',
        'oss://example-bucket.oss-us-southeast-1-internal.aliyuncs.com:443/users/researcher/',
        'oss://example-bucket.oss-us-southeast-1-internal.aliyuncs.com/users/%2e%2e/researcher/',
        'oss://example-bucket.oss-us-southeast-1-internal.aliyuncs.com/users/researcher/\n',
        'oss://example-bucket.oss-us-southeast-1-internal.aliyuncs.com/users/researcher',
    ]],
    *[{'oss_uri':'oss://example-bucket.oss-us-southeast-1-internal.aliyuncs.com/users/researcher/',
       'oss_mount_path':mount} for mount in ['/','/tmp/build','/mnt','/mnt/../root','/data//user','/mnt/user/']],
])
def test_invalid_oss_mount_fails_before_cloud_creation(publisher,settings):
    pub,context,api,_=publisher;pub.config.update(settings)
    with pytest.raises(ValueError,match='OSS mount'):pub.publish(TAG,context)
    assert not api.calls


@pytest.mark.parametrize('case',['tag','config','auth_missing','auth_invalid'])
def test_invalid_configuration_never_creates_a_cloud_instance(publisher,case):
    pub,context,api,_=publisher
    if case=='config':del pub.config['quota_id']
    if case=='auth_missing':Path(pub.builder.config['registry_auth_file']).write_text('{"auths":{}}')
    if case=='auth_invalid':Path(pub.builder.config['registry_auth_file']).write_text('{"auths":{"registry.example":{"auth":"eA=="}}}')
    with pytest.raises(ValueError):pub.publish('other:tag' if case=='tag' else TAG,context)
    assert not api.calls


@pytest.mark.parametrize('case',['waiting','failed','create','receipt_missing','receipt_double','receipt_bad','receipt_hash','manifest','cli'])
def test_failed_or_uncertain_build_never_replays_and_cleans_known_instance(publisher,monkeypatch,case):
    pub,context,api,_=publisher
    if case=='waiting':api.initial=['Starting','Starting']
    if case=='failed':api.state='Failed'
    if case=='create':
        monkeypatch.setattr(api,'request',lambda *args,**kw:(_ for _ in ()).throw(p.PaiError('CreateInstance',uncertain=True)))
    elif case.startswith('receipt') or case=='cli':
        original=pub.cli
        def cli(args,*a,**kw):
            if args[:3]==['dsw','instance','exec'] and 'status.json' in args[-1]:
                if case=='cli':raise ValueError('private provider failure')
                value=original(args,*a,**kw)
                if case=='receipt_missing':return ''
                if case=='receipt_double':return value+'\n'+value
                if case=='receipt_bad':return value.replace(IMAGE,'evil@sha256:'+'0'*64)
                return value.replace('context_sha256','wrong_hash')
            return original(args,*a,**kw)
        pub.cli=cli
    elif case=='manifest':monkeypatch.setattr(p,'digest',lambda raw:'sha256:'+'0'*64)
    if case=='waiting':assert pub.publish(TAG,context)==IMAGE
    else:
        with pytest.raises(BuildTransportError) as exc:pub.publish(TAG,context)
        assert exc.value.details['scheduler_submitted'] is False and exc.value.details['publication_uncertain']
    if case!='create':assert api.calls[-1][0]=='DeleteInstance'
    else:assert json.loads(pub.lease.read_text())['phase']=='CREATE_INTENT'


@pytest.mark.parametrize('key',['InstanceId','InstanceName','WorkspaceId','ResourceId','RequestedResource','Labels'])
def test_cleanup_never_targets_another_instances_identity(publisher,key):
    pub,_,api,_=publisher
    pub.instance='dsw-owned';pub.name='mine';pub.identity='a'*64
    api.value={'InstanceId':'dsw-owned','InstanceName':'mine','WorkspaceId':'253','ResourceId':'quota-fixed',
               'RequestedResource':{'CPU':'4','Memory':'8Gi','GPU':'0'},'Labels':[{'Key':'ml-expd.bundle','Value':'a'*64}]}
    api.value[key]={} if key=='RequestedResource' else [] if key=='Labels' else 'other'
    with pytest.raises(ValueError):pub.cleanup()
    assert [c[0] for c in api.calls]==['GetInstance']


@pytest.mark.parametrize('case',['bad_id','false_success','invalid_remaining','too_long'])
def test_cloud_shutdown_timer_is_read_back_and_fail_closed(publisher,case,monkeypatch):
    pub,_,api,_=publisher;pub.instance='dsw-owned'
    original=api.request
    def request(*args,**kw):
        value=original(*args,**kw)
        if args[1]=='GetInstanceShutdownTimer':
            if case=='bad_id':value['InstanceId']='other'
            elif case=='false_success':value['Success']=False
            elif case=='invalid_remaining':value['RemainingTimeInMs']=True
            else:value['RemainingTimeInMs']=1800001
        return value
    monkeypatch.setattr(api,'request',request)
    with pytest.raises(ValueError):pub.shutdown_timer()


def test_mcp_chunk_transfer_verifies_reassembly_then_discards_parts(publisher,tmp_path):
    pub,_,_,_=publisher;pub.instance='dsw-owned'
    source=tmp_path/'large';source.write_bytes(b'x'*(1024**2+7));target=tmp_path/'remote';sizes=[]
    def cli(args,private,**kwargs):
        if 'cp' in args:
            sizes.append(Path(args[3]).stat().st_size);shutil.copyfile(args[3],args[4].removeprefix('dsw:'))
        else:exec(args[-1])
    pub.cli=cli;pub.upload(source,str(target),tmp_path)
    assert target.read_bytes()==source.read_bytes() and sizes==[512*1024,512*1024,7]
    assert not list(tmp_path.glob('remote.part-*')) and not list(tmp_path.glob('dsw-parts-*'))


def test_cli_redacts_ak_registry_and_signed_query_and_bounds_failures(publisher,tmp_path,monkeypatch):
    pub,_,api,_=publisher;pub.secrets=['registry-password']
    def runner(command,**kw):
        assert kw['env']['PAI_CONFIG_DIR']==str(tmp_path) and command[:3]==['/usr/local/bin/pai','--profile','ml-expd']
        return SimpleNamespace(returncode=0,stdout='secret-ak secret-sk secret-token registry-password https://example.test/p?X-Goog-Signature=private',stderr='ok')
    monkeypatch.setattr(p,'run_process',runner)
    t=BUILD_LOG.set(tmp_path/'build.log')
    try:out=p.DswPublisher.cli(pub,['dsw','instance','ls'],tmp_path)
    finally:BUILD_LOG.reset(t)
    assert all(x not in out for x in ['secret-ak','secret-sk','secret-token','registry-password','Signature=private'])
    monkeypatch.setattr(p,'run_process',lambda *args,**kw:SimpleNamespace(returncode=1,stdout='',stderr='failed'))
    with pytest.raises(ValueError):p.DswPublisher.cli(pub,[],tmp_path)


def test_source_recipe_dispatch_can_use_dsw_without_changing_runtime_contract(builder,monkeypatch):
    value,request,_=builder;value.config.update(publisher='pai-dsw')
    monkeypatch.setattr(p.DswPublisher,'__init__',lambda self,builder:None)
    monkeypatch.setattr(p.DswPublisher,'publish',lambda self,tag,context:'registry.example/results@sha256:'+'b'*64)
    receipt=value.request(request)
    assert receipt['worker_contract']=='managed-worker.v1' and receipt['recipe']=='source-dockerfile-assets-v1'


@pytest.mark.parametrize('case',['running','negative_offset','boolean_offset','failed','timeout','empty','duplicate'])
def test_background_build_is_started_once_and_observed_without_replay(publisher,monkeypatch,case):
    pub,_,_,_=publisher;pub.instance='dsw-owned';calls=[];polls=[0]
    def cli(args,*a,**kw):
        calls.append(args)
        if 'status.json' not in args[-1]:return ''
        polls[0]+=1
        state={'status':'SUCCEEDED','receipt':{'image':IMAGE},'log_offset':1}
        if case=='running' and polls[0]==1:state['status']='RUNNING'
        elif case=='negative_offset':state['log_offset']=-1
        elif case=='boolean_offset':state['log_offset']=True
        elif case=='failed':state['status']='FAILED'
        elif case=='timeout':state['status']='RUNNING'
        if case=='empty':return 'some log\n'
        text='log line\nML_EXPD_DSW_STATUS_JSON='+json.dumps(state)
        return text+'\n'+text if case=='duplicate' else text
    pub.cli=cli
    if case=='timeout':
        moments=iter([0,601]);monkeypatch.setattr(p.time,'monotonic',lambda:next(moments))
    if case=='running':assert pub.build('/tmp/ml-expd-build-'+'a'*32,Path('/private'))=={'image':IMAGE}
    else:
        with pytest.raises(ValueError):pub.build('/tmp/ml-expd-build-'+'a'*32,Path('/private'))
    assert sum('Popen' in c[-1] for c in calls)==1


@pytest.mark.parametrize('case',['stopped','stopping','stop_wait','stop_timeout','startup_timeout','large_context'])
def test_instance_and_context_bounds(publisher,monkeypatch,case):
    pub,context,api,_=publisher
    if case=='startup_timeout':
        api.state='Starting';moments=iter([0,601,602,603]);monkeypatch.setattr(p.time,'monotonic',lambda:next(moments,9999))
        # Permit final cleanup after the bounded startup failure.
        original=api.request
        def request(*args,**kw):
            if args[1]=='StopInstance':api.state='Stopped'
            return original(*args,**kw)
        monkeypatch.setattr(api,'request',request)
        with pytest.raises(BuildTransportError):pub.publish(TAG,context)
        return
    if case=='large_context':
        original=p.Path.stat
        def stat(self,*a,**kw):
            value=original(self,*a,**kw)
            return SimpleNamespace(st_size=301*1024**2) if self.name=='context.tar' else value
        monkeypatch.setattr(p.Path,'stat',stat)
        with pytest.raises(BuildTransportError):pub.publish(TAG,context)
        return
    pub.identity='a'*64;pub.name='mine';pub.instance='dsw-owned';pub.lease=pub.builder.root/'lease.json'
    api.value={'InstanceId':'dsw-owned','InstanceName':'mine','WorkspaceId':'253','ResourceId':'quota-fixed',
               'RequestedResource':{'CPU':'4','Memory':'8Gi','GPU':'0'},'Labels':[{'Key':'ml-expd.bundle','Value':'a'*64}]}
    if case=='stopped':api.state='Stopped'
    elif case=='stopping':api.state='Stopped';api.initial=['Stopping']
    elif case=='stop_wait':api.state='Stopped';api.initial=['Running','Stopping']
    else:
        api.state='Stopping';moments=iter([0,181]);monkeypatch.setattr(p.time,'monotonic',lambda:next(moments))
        with pytest.raises(ValueError,match='stop remains'):pub.cleanup()
        return
    pub.cleanup();assert api.calls[-1][0]=='DeleteInstance'


def test_optional_security_token_and_empty_registry_secret_are_not_replacements(publisher,tmp_path,monkeypatch):
    pub,_,api,_=publisher;api.config.pop('security_token')
    monkeypatch.setattr(p,'run_process',lambda *a,**kw:SimpleNamespace(returncode=0,stdout='ok',stderr=''))
    assert 'ok' in p.DswPublisher.cli(pub,[],tmp_path)


def test_cleanup_failure_keeps_verified_publication_and_requires_operator_followup(publisher,monkeypatch):
    pub,context,api,_=publisher
    original=api.request
    def request(*args,**kw):
        if args[1]=='StopInstance':raise p.PaiError('StopInstance',uncertain=True)
        return original(*args,**kw)
    monkeypatch.setattr(api,'request',request)
    with pytest.raises(BuildTransportError,match='PAI_DSW_CLEANUP_REQUIRED') as exc:pub.publish(TAG,context)
    assert exc.value.details['cleanup_required'] is True
    value=json.loads(pub.lease.read_text())
    assert value['phase']=='CLEANUP_REQUIRED' and value['image']==IMAGE and value['workspace_id']=='253'
    assert value['quota_id']=='quota-fixed' and value['gpu']==0
