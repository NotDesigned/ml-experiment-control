"""DLC identity, limits, cancellation and native logging without cloud effects."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
import pytest
from backend_harness import services,QueueRunner,submission_intent
from experiment_control.backends.pai import PaiBackend,normalized,job_tags
from experiment_control.backends.pai_api import PaiError


def run_spec():
    return {'run_id':'pai-run','image_id':'sha256:'+'a'*64,'resources':{'gpus':1,'cpus':8,'memory_gb':32},
            'backend':{'kind':'pai-dlc','region':'us-southeast-1','workspace_id':'253','quota_id':'quota-test',
                       'gpu_type':'H200','job_name':'pai-run','image':'registry.test/ns/image:bundle-abc','time':'00:01:01'}}


@pytest.fixture
def adapter(tmp_path):
    run=run_spec();calls=[];data={'jobs':[],'job':{'JobId':'dlc-owned','Status':'Running','GmtCreateTime':'2026-10-23T00:00:00Z',
        'ReasonCode':'','ReasonMessage':'waiting','Settings':{'Tags':job_tags(run,'attempt-001')}}}
    def request(product,action,path,**kwargs):
        calls.append((action,path,kwargs))
        if action=='CreateJob':return {'JobId':'dlc-owned'}
        if action=='GetPodLogs':return {'Logs':['loss=1','last']}
        return {}
    api=SimpleNamespace(config={'region':'us-southeast-1','workspace_id':'253'},
        workspace=lambda x:{'WorkspaceId':x,'Status':'ENABLED'},quota=lambda x:{'QuotaId':x,'Status':'Ready'},
        jobs=lambda *a,**kw:data['jobs'],job=lambda *a:data['job'],request=request,
        create_job=lambda document:request('pai-dlc','CreateJob','/api/v1/jobs',method='POST',body=document))
    s=services(tmp_path,QueueRunner([]),record={'attempt_id':'attempt-001','backend_job_id':'dlc-owned'})
    backend=PaiBackend(replace(s,pai_api=lambda:api))
    return backend,run,api,calls,data


@pytest.mark.parametrize('raw,expected',[('Creating','QUEUED'),('Queuing','QUEUED'),('Bidding','QUEUED'),('Dequeued','STARTING'),('WaitingConfig','STARTING'),('EnvPreparing','STARTING'),('SanityChecking','STARTING'),('Running','RUNNING'),('Restarting','RUNNING'),('Stopping','UNKNOWN'),('SucceededReserving','SUCCEEDED'),('Succeeded','SUCCEEDED'),('FailedReserving','FAILED'),('Failed','FAILED'),('Stopped','CANCELLED'),('Deleted','UNKNOWN')])
def test_states_do_not_turn_unknown_or_stopping_into_success(raw,expected):
    assert normalized(raw)==expected


def test_readiness_and_immutable_native_source_staging(adapter):
    b,r,api,_,_=adapter
    assert b.api is api and b.api is api
    assert b.availability().ready and b.preflight(r,scope='submit').ready
    assert b.environment({},r,'source','attempt-001')=={}
    assert b.stage({},r,'source',None)
    assert b.verify_assets(r,[])['verification']=='image-manifest'
    with pytest.raises(ValueError):b.preflight(r,scope='invalid')
    api.config['region']='another';assert not b.preflight(r,scope='submit').ready
    api.config['region']='us-southeast-1';api.quota=lambda _:{'Status':'Creating'}
    assert not b.preflight(r,scope='submit').ready and b.preflight(r,scope='observe').ready
    api.workspace=lambda _:(_ for _ in ()).throw(PaiError('GetWorkspace'))
    assert not b.availability().ready and not b.preflight(r,scope='observe').ready
    api.config={};assert not b.availability().ready


@pytest.mark.parametrize('key,value',[('gpus',0),('gpus',9),('cpus',True),('cpus',0),('memory_gb',4097)])
def test_single_node_resource_limits(adapter,key,value):
    b,r,*_=adapter;r['resources'][key]=value
    with pytest.raises(ValueError):b.validate(r)


@pytest.mark.parametrize('change',[{'worker_nodes':2},{'time':'bad'},{'time':'00:00:00'}, {'data_sources':None},
    {'data_sources':[{}]}, {'data_sources':[{'DataSourceId':'d','DataSourceVersion':'v1','MountAccess':'RW','MountPath':'/x/../y'}]},
    {'data_sources':[{'DataSourceId':'d','DataSourceVersion':'v1','MountAccess':'RW','MountPath':'/x','Secret':'no'}]}])
def test_native_mounts_and_walltime_are_explicit(adapter,change):
    b,r,*_=adapter;r['backend'].update(change)
    with pytest.raises(ValueError):b.validate(r)


def manifest(r):return {**deepcopy(r),'attempt_id':'attempt-001','command':['python','train.py','a b']}


def test_submission_render_pins_quota_digest_and_disables_restarts(adapter):
    b,r,_,calls,_=adapter
    r['backend']['data_sources']=[{'DataSourceId':'d-native','DataSourceVersion':'v1','MountAccess':'RO','MountPath':'/inputs/native'}]
    m=manifest(r);d=json.loads(b.render(m))
    assert d['ResourceId']=='quota-test' and d['WorkspaceId']=='253' and d['JobMaxRunningTimeMinutes']==2
    spec=d['JobSpecs'][0]
    assert spec['PodCount']==1 and spec['ResourceConfig']['GPU']=='1' and spec['RestartPolicy']=='Never' and not spec['UseSpotInstance']
    assert spec['Image']=='registry.test/ns/image@'+r['image_id']
    assert d['Settings']['JobReservedMinutes']==0 and not d['Settings']['EnableErrorMonitoringInAIMaster']
    assert d['DataSources']==r['backend']['data_sources']
    assert b.submit({},r,m,dry_run=True)=='DRY_RUN' and calls==[]
    intent=submission_intent(b,r)
    assert b.submit({},r,m,dry_run=False,intent=intent)=='dlc-owned'
    assert calls[-1][0]=='CreateJob' and calls[-1][2]['method']=='POST'
    tags=calls[-1][2]['body']['Settings']['Tags'];assert tags['ml-expd.submission']==intent['submission_token']
    assert 'private-secret' not in b.render(m)


def test_submission_and_recovery_refuse_identity_changes_or_ambiguity(adapter):
    b,r,api,calls,data=adapter;m=manifest(r);intent=submission_intent(b,r)
    assert b.recover_submission(r,intent,'attempt-001') is None
    assert b.identity({},r,'attempt-001').available
    own={'JobId':'dlc-owned','Settings':{'Tags':job_tags(r,'attempt-001',intent['submission_token'])}}
    data['jobs']=[own,{'JobId':'unrelated','Settings':{'Tags':{**job_tags(r,'attempt-001'),'ml-expd.run':'other'}}}]
    assert b.recover_submission(r,intent,'attempt-001')=='dlc-owned'
    assert not b.identity({},r,'attempt-001').available
    with pytest.raises(FileExistsError):b.submit({},r,m,dry_run=False,intent=intent)
    assert calls==[]
    data['jobs']=[own,own]
    assert b.identity({},r,'attempt-001').ambiguous
    with pytest.raises(ValueError,match='ambiguous'):b.recover_submission(r,intent,'attempt-001')
    wrong=deepcopy(intent);wrong['request']['image_digest']='sha256:'+'b'*64
    with pytest.raises(ValueError):b.recover_submission(r,wrong,'attempt-001')
    with pytest.raises(ValueError):b.submit({},r,m,dry_run=False,intent=wrong)
    m['image_id']='sha256:'+'b'*64
    with pytest.raises(ValueError):b.submit({},r,m,dry_run=True)


def test_status_logs_historical_uid_and_cancel_bind_the_exact_attempt(adapter):
    b,r,api,calls,data=adapter
    status=b.status({},r);assert status['state']=='RUNNING' and status['exit_code'] is None and status['reason']=='waiting'
    empty=b.logs({},r,tail=2);assert not empty['available'] and empty['lines']==[]
    pod={'PodId':'pod-1','PodUid':'uid-1'}
    data['job'].update(Status='Succeeded',HistoryPods=[[pod]])
    logs=b.logs({},r,tail=1);assert logs['lines']==['last'] and logs['historical']
    assert calls[-1][2]['query']['PodUid']=='uid-1'
    assert b.collect({},r)['model_observed'] is False
    b.cancel({},r);assert calls[-1][0]=='StopJob' and calls[-1][2]['method']=='PUT'
    data['job']['Settings']['Tags']['ml-expd.attempt']='attempt-002'
    with pytest.raises(ValueError,match='another'):b.cancel({},r)
    api.request=lambda *a,**kw:{'Logs':[None]};data['job']['Settings']['Tags']=job_tags(r,'attempt-001')
    with pytest.raises(ValueError,match='logs'):b.logs({},r,tail=1)


def test_incomplete_provider_tags_cannot_prove_absence_or_authorize_replay(adapter):
    b,r,api,calls,data=adapter
    for settings in [None,{}, {'Tags':None},{'Tags':{'ml-expd.run':r['run_id']}}]:
        data['jobs']=[{'JobId':'dlc-unknown','Settings':settings}]
        with pytest.raises(ValueError,match='identity tags'):b.identity({},r,'attempt-001')
        data['job']['Settings']=settings
        with pytest.raises(ValueError,match='identity tags'):b.status({},r)
    assert not calls


def test_duplicate_native_mount_or_invalid_dataset_version_is_rejected(adapter):
    b,r,*_=adapter
    source={'DataSourceId':'d-data','DataSourceVersion':'v1','MountPath':'/data','MountAccess':'RW'}
    r['backend']['data_sources']=[source,dict(source,MountPath='/data/')]
    with pytest.raises(ValueError,match='unique'):b.validate(r)
    for k in ['DataSourceId','DataSourceVersion']:
        r['backend']['data_sources']=[dict(source,**{k:True})]
        with pytest.raises(ValueError):b.validate(r)
