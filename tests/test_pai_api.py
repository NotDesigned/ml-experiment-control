"""Official SDK wire options, scope verification and uncertain-create boundaries."""
import builtins
from copy import deepcopy
import json
from types import SimpleNamespace
import pytest
from experiment_control.backends import pai_api as p

CONFIG={'region':'us-southeast-1','workspace_id':'253','access_key_id':'test-ak','access_key_secret':'private-secret'}
B={'workspace_id':'253','quota_id':'quota-test'}

@pytest.fixture
def wire(monkeypatch):
    from alibabacloud_tea_openapi.client import Client
    calls=[];reply={'statusCode':200,'body':{'ok':True}}
    def call(self,params,request,runtime):
        calls.append((params,request,runtime,self._endpoint))
        if isinstance(reply,Exception):raise reply
        return deepcopy(reply)
    monkeypatch.setattr(Client,'call_api',call)
    def set_reply(value):
        nonlocal reply
        reply=value
    return calls,set_reply


def test_transport_uses_sdk_https_scope_and_disables_write_retries(wire):
    calls,_=wire
    api=p.PaiAPI(CONFIG)
    assert api.request('pai-dlc','CreateJob','/api/v1/jobs',method='POST',query={'PageSize':5},body={'ResourceId':'quota-test'})=={'ok':True}
    params,request,runtime,host=calls[0]
    assert host=='pai-dlc.us-southeast-1.aliyuncs.com'
    assert params.protocol=='HTTPS' and params.style=='ROA' and request.query=={'PageSize':'5'}
    assert request.body=={'ResourceId':'quota-test'} and not runtime.autoretry and runtime.max_attempts==1 and not runtime.ignore_ssl
    assert runtime.read_timeout==30000 and runtime.connect_timeout==5000
    api.request('paistudio','GetQuota','/api/v1/quotas/quota-test')
    assert calls[-1][3]=='pai.us-southeast-1.aliyuncs.com'
    api.request('cr','ListInstance','/',style='RPC')
    assert calls[-1][0].req_body_type=='formData'


@pytest.mark.parametrize('config',[None,{},dict(CONFIG,region='../escape'),dict(CONFIG,access_key_secret=0),dict(CONFIG,access_key_id='')])
def test_invalid_credentials_never_reach_the_sdk(config):
    with pytest.raises(p.PaiError,match='CONFIGURATION_INVALID'):p.PaiAPI(config)


def test_private_configuration_is_explicit_and_errors_are_sanitized(tmp_path,monkeypatch):
    config=tmp_path/'pai.json';monkeypatch.setenv('EXPERIMENTCTL_PAI_CONFIG',str(config))
    with pytest.raises(p.PaiError,match='UNAVAILABLE'):p.PaiAPI.from_environment()
    config.write_text('{bad')
    with pytest.raises(p.PaiError):p.PaiAPI.from_environment()
    config.write_text(json.dumps(CONFIG));assert p.PaiAPI.from_environment().config==CONFIG
    monkeypatch.delenv('EXPERIMENTCTL_PAI_CONFIG')
    monkeypatch.setattr(p.Path,'read_text',lambda self:json.dumps(CONFIG))
    assert p.PaiAPI.from_environment().config==CONFIG


@pytest.mark.parametrize('kwargs',[{'product':'unknown'},{'action':'bad\n'},{'path':'relative'}, {'path':'/x?secret'}, {'method':'PATCH'},{'style':'unknown'}])
def test_operation_validation_precedes_network(kwargs):
    args={'product':'pai-dlc','action':'ListJobs','path':'/api/v1/jobs',**kwargs}
    with pytest.raises(ValueError):p.PaiAPI(CONFIG).request(**args)


def test_optional_sdk_and_untrusted_endpoint_fail_closed(wire,monkeypatch):
    api=p.PaiAPI({**CONFIG,'endpoints':{'pai-dlc':'evil.test'}})
    with pytest.raises(ValueError,match='HTTPS'):api.request('pai-dlc','ListJobs','/api/v1/jobs')
    original=builtins.__import__
    def missing(name,*args,**kwargs):
        if name=='alibabacloud_tea_openapi.client':raise ImportError('private path')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',missing)
    with pytest.raises(p.PaiError,match='SDK_UNAVAILABLE'):p.PaiAPI(CONFIG).request('pai-dlc','ListJobs','/api/v1/jobs')


@pytest.mark.parametrize('method',['GET','POST'])
@pytest.mark.parametrize('data',[None,{'Code':'Denied','statusCode':403},{'Code':'InternalError','statusCode':503},{'Code':'private key\n','statusCode':'500'}])
def test_provider_errors_do_not_echo_or_replay_credentials(wire,method,data):
    calls,set_reply=wire
    err=RuntimeError('private-secret');err.data=data;set_reply(err)
    with pytest.raises(p.PaiError) as e:p.PaiAPI(CONFIG).request('pai-dlc','CreateJob','/api/v1/jobs',method=method)
    assert len(calls)==1 and 'private-secret' not in str(e.value) and 'private key' not in str(e.value)
    assert e.value.uncertain==(method!='GET' and (not data or data.get('statusCode')!=403))


@pytest.mark.parametrize('body',[None,[],{'Code':'AUTHENTICATION_FAILED','Message':'private-secret'},{'Code':'bad\nsecret'},{'IsSuccess':False}])
def test_http_200_error_envelopes_are_not_accepted(wire,body):
    calls,set_reply=wire;set_reply({'statusCode':200,'body':body})
    with pytest.raises(p.PaiError) as e:p.PaiAPI(CONFIG).request('cr','ListInstance','/')
    assert len(calls)==1 and 'private-secret' not in str(e.value)


@pytest.mark.parametrize('identity',['../x','bad/token','-option',None])
def test_ids_cannot_escape_paths(identity):
    with pytest.raises(ValueError):p.component(identity)


def test_exact_job_workspace_quota_and_native_quota_envelopes(wire):
    _,set_reply=wire;api=p.PaiAPI(CONFIG)
    job={'JobId':'dlc-test','WorkspaceId':'253','ResourceId':'quota-test'}
    set_reply({'body':job});assert api.job(B,'dlc-test')==job
    for key in job:
        set_reply({'body':{**job,key:'wrong'}})
        with pytest.raises(ValueError):api.job(B,'dlc-test')
    set_reply({'body':{'WorkspaceId':253}});assert api.workspace('253')['WorkspaceId']==253
    set_reply({'body':{'WorkspaceId':'wrong'}})
    with pytest.raises(ValueError):api.workspace('253')
    for q in [{'QuotaId':'quota-test'},{'Quota':{'QuotaId':'quota-test'}}]:
        set_reply({'body':q});assert api.quota('quota-test')['QuotaId']=='quota-test'
    set_reply({'body':{'Quota':None}})
    with pytest.raises(ValueError):api.quota('quota-test')


def test_job_lookup_is_paginated_exact_and_not_limited_to_last_seven_days(wire):
    calls,set_reply=wire;api=p.PaiAPI(CONFIG)
    row={'JobId':'dlc-test','WorkspaceId':'253','ResourceId':'quota-test'}
    set_reply({'body':{'Jobs':[row],'TotalCount':1}})
    assert api.jobs(B,tags={'identity':'x'},name='name')==[row]
    assert calls[-1][1].query['StartTime']=='2020-01-01T00:00:00Z'
    assert json.loads(calls[-1][1].query['Tags'])=={'identity':'x'}
    for body in [{'Jobs':{},'TotalCount':1},{'Jobs':[],'TotalCount':-1},{'Jobs':[],'TotalCount':1},
                 {'Jobs':[row,row],'TotalCount':2},{'Jobs':[row],'TotalCount':0},
                 {'Jobs':[{**row,'WorkspaceId':'other'}],'TotalCount':1},
                 {'Jobs':[{**row,'ResourceId':'other'}],'TotalCount':1}]:
        set_reply({'body':body})
        with pytest.raises(ValueError):api.jobs(B)
    set_reply({'body':{'Jobs':[],'TotalCount':0}});assert api.jobs(B)==[]
    counter=[0]
    def request(*args,**kwargs):
        counter[0]+=1
        return {'Jobs':[{**row,'JobId':'dlc-'+str(counter[0])}],'TotalCount':1000}
    monkey=pytest.MonkeyPatch();monkey.setattr(api,'request',request)
    try:
        with pytest.raises(ValueError,match='limit'):api.jobs(B)
    finally:monkey.undo()


def test_dsw_timer_literal_null_is_only_accepted_for_verified_success(wire):
    _,set_reply=wire;api=p.PaiAPI(CONFIG)
    set_reply({'statusCode':200,'body':{'Code':'null','Success':True}})
    assert api.request('pai-dsw','CreateInstanceShutdownTimer','/timer')['Success']
    assert api.request('pai-dsw','GetInstanceShutdownTimer','/timer')['Success']
    for product,action in [('pai-dsw','Other'),('pai-dlc','GetInstanceShutdownTimer')]:
        with pytest.raises(p.PaiError):api.request(product,action,'/timer')
    set_reply({'body':{'Code':None,'Success':False}})
    with pytest.raises(p.PaiError):api.request('pai-dsw','CreateInstanceShutdownTimer','/timer')
    set_reply({'body':{'Code':'null','Success':False}})
    with pytest.raises(p.PaiError):api.request('pai-dsw','CreateInstanceShutdownTimer','/timer')


def test_image_pull_credentials_are_injected_only_into_private_write(wire,tmp_path):
    import base64
    calls,_=wire;path=tmp_path/'auth.json'
    document={'JobSpecs':[{'Image':'registry.test/repo@sha256:'+'a'*64}]}
    p.PaiAPI(CONFIG).create_job(document)
    assert 'ImageConfig' not in calls[-1][1].body['JobSpecs'][0]
    path.write_text(json.dumps({'auths':{'registry.test':{'auth':base64.b64encode(b'user:secret').decode()}}}))
    api=p.PaiAPI(dict(CONFIG,registry_auth_file=str(path)));api.create_job(document)
    assert calls[-1][1].body['JobSpecs'][0]['ImageConfig']=={'DockerRegistry':'registry.test','Username':'user','Password':'secret'}
    assert 'ImageConfig' not in document['JobSpecs'][0]
    for raw in [b'user:',b':secret',b'user:secret\n',b'nocolon']:
        path.write_text(json.dumps({'auths':{'https://registry.test':{'auth':base64.b64encode(raw).decode()}}}))
        with pytest.raises(p.PaiError,match='IMAGE_AUTH'):api.create_job(document)
    path.write_text('{}')
    with pytest.raises(p.PaiError):api.create_job(document)
    path.unlink()
    with pytest.raises(p.PaiError):api.create_job(document)
