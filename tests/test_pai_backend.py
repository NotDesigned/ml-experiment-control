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
    {'time':'24:00:01'},{'time':'25:00:00'},
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
    b.cancel({},r);assert calls[-1][0]=='StopJob' and calls[-1][2]['method']=='POST'
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


def personal(r):
    r['backend'].update(cpfs_uri='bmcpfs://cpfs-approved.us-southeast-1.cpfs.aliyuncs.com/',
        workspace_user='owner',data_role_arn='acs:ram::123456:role/pai-data-owner')
    return r


def test_dlc_personal_cpfs_and_runtime_role_use_native_dlc_fields(adapter):
    b,r,_,calls,_=adapter;personal(r)
    d=b.create_document(manifest(r))
    assert [(m['MountPath'],m['MountAccess']) for m in d['DataSources']]==[
        ('/mnt/workspace','RW'),('/mnt/models','RO'),('/mnt/datasets','RO')]
    assert d['DataSources'][0]['Uri'].endswith('/users/owner/')
    assert d['DataSources'][1]['Uri'].endswith('/public/models/')
    assert d['DataSources'][2]['Uri'].endswith('/public/datasets/')
    credentials=d['CredentialConfig']
    assert credentials['EnableCredentialInject'] and credentials['AliyunEnvRoleKey']=='0'
    assert credentials['CredentialConfigItems']==[{'Key':'0','Type':'Role','Roles':[
        {'RoleArn':'acs:ram::123456:role/pai-data-owner','RoleType':'service','AssumeRoleFor':'123456'}]}]
    assert 'Enable' not in credentials and 'Configs' not in credentials
    assert not calls


@pytest.mark.parametrize('change',[
    {'cpfs_uri':None},{'user_vpc':None},{'user_vpc':{}},{'user_vpc':{'VpcId':'other'}},
    {'user_vpc':{'VpcId':None,'VSwitchId':'vsw-ok','SecurityGroupId':'sg-ok'}},
    {'user_vpc':{'VpcId':'wrong','VSwitchId':'vsw-ok','SecurityGroupId':'sg-ok'}},{'cpfs_uri':'bmcpfs://cpfs-approved.other-region.cpfs.aliyuncs.com/'},
    {'cpfs_uri':'bmcpfs://cpfs-approved.us-southeast-1.cpfs.aliyuncs.com/users/other/'},
    {'workspace_user':None},{'workspace_user':'../other'},
    {'cpfs_uri':'bmcpfs://cpfs-approved-vpc-target.us-southeast-1.cpfs.aliyuncs.com/'},
    {'data_role_arn':None},{'data_role_arn':'acs:ram::123456:role/pai-data-other'},
    {'data_sources':[{'DataSourceId':'d','DataSourceVersion':'v1','MountPath':'/mnt','MountAccess':'RW'}]},
    {'data_sources':[{'DataSourceId':'d','DataSourceVersion':'v1','MountPath':'/mnt/workspace/other','MountAccess':'RO'}]},
])
def test_personal_mounts_reject_other_users_roles_regions_and_shadowing(adapter,change):
    b,r,*_=adapter;personal(r);r['backend'].update(change)
    with pytest.raises(ValueError):b.validate(r)


def test_role_without_cpfs_and_explicit_vpc_mount(adapter):
    b,r,*_=adapter;personal(r)
    del r['backend']['cpfs_uri']
    assert b.create_document(manifest(r))['CredentialConfig']['EnableCredentialInject']
    r['backend']['workspace_user']=None
    with pytest.raises(ValueError,match='workspace user'):b.validate(r)
    personal(r)
    r['backend']['cpfs_uri']='bmcpfs://cpfs-approved-vpc-target.us-southeast-1.cpfs.aliyuncs.com/'
    r['backend']['user_vpc']={'VpcId':'vpc-approved','VSwitchId':'vsw-approved','SecurityGroupId':'sg-approved'}
    assert len(b.create_document(manifest(r))['DataSources'])==3


def test_quota_gpu_selection_verifies_frozen_hardware_without_native_gpu_enum(adapter):
    b,r,api,_,_=adapter
    r['backend']['gpu_type_from_quota']='yes'
    with pytest.raises(ValueError,match='explicit'):b.validate(r)
    r['backend']['gpu_type_from_quota']=True
    assert not b.preflight(r,scope='submit').ready
    api.quota=lambda _: {'Status':'Ready','GPUType':'H200'}
    assert b.preflight(r,scope='submit').ready
    assert 'GPUType' not in b.create_document(manifest(r))['JobSpecs'][0]['ResourceConfig']
    assert r['backend']['gpu_type']=='H200'

@pytest.mark.parametrize('mode',['guaranteed','allow_idle','idle_only'])
def test_idle_selection_and_priority_are_native_and_not_paid_spot(adapter,mode):
    from experiment_control.backends.pai import ALLOCATION_POLICIES
    backend,run,_,_,_=adapter
    manifest={**run,'attempt_id':'attempt-001','command':['python3','train.py']}
    manifest['backend'].update(oversold_type=ALLOCATION_POLICIES[mode],priority=5)
    value=backend.create_document(manifest)
    assert value['Priority']==5 and value['JobSpecs'][0]['OversoldType']==ALLOCATION_POLICIES[mode]
    assert value['JobSpecs'][0]['UseSpotInstance'] is False


@pytest.mark.parametrize('key,value',[('oversold_type','spot'),('oversold_type',None),('priority',True),('priority',0),('priority',10),('priority','5')])
def test_invalid_idle_policy_and_priority_never_submit(adapter,key,value):
    backend,run,_,calls,_=adapter;run['backend'][key]=value
    with pytest.raises(ValueError):backend.create_document({**run,'attempt_id':'attempt-001','command':['true']})
    assert not calls


def test_queue_blocker_is_exposed_without_changing_scheduler_state_or_replaying(adapter):
    backend,run,api,calls,data=adapter;lookups=[]
    blocker={'CodeType':'ConfigRule','Code':'roleMaximumResource','Reason':
             'Current Limit is CPU: 200 limited by Role PAI.AlgoDeveloper.Memory: 1600 limited by Role PAI.AlgoDeveloper.GPU: 8 limited by Role PAI.AlgoDeveloper.'}
    row={'Code':'CheckSelfQuotaFailed','CodeType':'QuotaCheckFail','Reason':json.dumps(blocker),
         'Position':2,'Priority':5,'Status':'Enqueued','GmtEnqueuedTime':'2026-10-24T01:00:00Z','Secret':'not exported'}
    api.queue_workload=lambda scope,job:(lookups.append((scope,job)) or row)
    data['job']['Status']='Queuing'
    result=backend.status({},run);queue=result['detail']['queue']
    assert result['state']=='QUEUED' and result['reason']==blocker['Reason']
    assert queue['blocker']==blocker and queue['code']=='roleMaximumResource'
    assert queue['code_type']=='ConfigRule' and queue['provider_code']=='CheckSelfQuotaFailed'
    assert queue['position']==2 and queue['priority']==5 and queue['status']=='Enqueued'
    assert lookups==[(run['backend'],'dlc-owned')] and not calls and 'Secret' not in queue
    data['job']['Status']='Running'
    result=backend.status({},run)
    assert result['state']=='RUNNING' and result['detail']['queue'] is None and len(lookups)==1


@pytest.mark.parametrize('failure',[None,PaiError('ListQuotaWorkloads',code='Forbidden',status=403),ValueError('private raw response')])
def test_unavailable_queue_diagnostics_keep_the_known_job_state(adapter,failure):
    backend,run,api,calls,data=adapter;data['job']['Status']='Queuing'
    def lookup(*args):
        if failure is not None:raise failure
        return None
    api.queue_workload=lookup
    result=backend.status({},run);queue=result['detail']['queue']
    assert result['state']=='QUEUED' and result['reason']=='waiting' and not calls
    assert queue['availability']==('NOT_OBSERVED' if failure is None else 'UNAVAILABLE')
    if failure is not None:
        assert queue['query_error']==('Forbidden' if isinstance(failure,PaiError) else 'PAI_QUEUE_RESPONSE_INVALID')
    assert 'private raw response' not in json.dumps(result)


@pytest.mark.parametrize('reason',['plain queue reason','[]','{"Code":null,"Reason":"","CodeType":123,"Password":"hidden"}'])
def test_queue_diagnostics_allow_plain_or_malformed_rules_without_leaking_extra_fields(reason):
    from experiment_control.backends.pai import queue_details
    result=queue_details({'Reason':reason,'Position':True,'Priority':-1,'Code':None})
    assert 'position' not in result and 'priority' not in result and 'provider_code' not in result
    if 'blocker' in result:assert result['blocker']=={'Reason':''}
    assert 'hidden' not in json.dumps(result)


def test_queue_reason_redaction_and_size_bound():
    from experiment_control.backends.pai import queue_details
    result=queue_details({'Reason':json.dumps({'CodeType':'ConfigRule','Code':'blocked',
        'Reason':'password=private-value https://host.test/x?X-Goog-Signature=private-signature'})})
    assert 'private-value' not in json.dumps(result) and 'private-signature' not in json.dumps(result)
    assert len(queue_details({'Reason':'x'*10000})['reason'])==2048
