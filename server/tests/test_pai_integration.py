"""Third scheduler contracts: explicit capabilities and no invented data fallback."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import json
from tests.test_container_api import client
from experiment_control.backends.pai_api import PaiError
from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.backends.operations import PaiOperations, backend_operations
from ml_exp_server.results.checkpoint_registry import storage_scope
from ml_exp_server.runs import allocation_status as a
from ml_exp_server.runs.executor_capabilities import declaration, match, ExecutorSelector, ExecutionRequirements
from tests.test_executor_capabilities import request

B={'kind':'pai-dlc','region':'us-southeast-1','workspace_id':'253','quota_id':'quota-test','gpu_type':'H200'}
Q={'QuotaId':'quota-test','Status':'Ready','GPUType':'H200','Workspaces':[{'WorkspaceId':253}],
   'QuotaDetails':{'ActualMinQuota':{'GPU':'64','CPU':'1728','Memory':'14400Gi'},
                   'AllocatedQuota':{'GPU':'2','CPU':'8','Memory':'16Gi'}}}

@pytest.fixture
def api(monkeypatch):
    result=deepcopy(Q)
    value=SimpleNamespace(config={'region':'us-southeast-1'},quota=lambda identity:deepcopy(result),
                          job=lambda *args:{'Status':'Running'})
    monkeypatch.setattr(a.PaiAPI,'from_environment',lambda:value)
    return value,result


def test_quota_resources_have_explicit_units_and_cache_identity(api):
    value,result=api
    s=a.AllocationStatus();v=s.get('pai',{'backend':B})
    assert v['source']=='pai-quota' and v['scope']=='quota' and v['status']=='OBSERVED'
    assert v['resources']['gpus']=={'total':64,'allocated':2,'unallocated':62}
    assert v['resources']['memory_bytes']['allocated']==16*1024**3 and not v['admission_guaranteed']
    assert v['pending_queue_entries'] is None and v['nodes']==[]
    result['QuotaDetails']['AllocatedQuota']['GPU']='3'
    assert s.get('other',{'backend':B})['resources']['gpus']['allocated']==2
    for unit,factor in [('Ki',1024),('Mi',1024**2),('Ti',1024**4)]:
        result['QuotaDetails']['ActualMinQuota']['Memory']='100'+unit
        result['QuotaDetails']['AllocatedQuota']['Memory']='0'+unit
        assert a.pai_quota(B)['resources']['memory_bytes']['total']==100*factor


@pytest.mark.parametrize('case',['region','workspace','memory','over','gpu','missing','transport'])
def test_quota_bad_or_missing_evidence_never_becomes_free_capacity(api,case):
    value,q=api
    if case=='region':value.config['region']='cn-hangzhou'
    elif case=='workspace':q['Workspaces']=[]
    elif case=='memory':q['QuotaDetails']['AllocatedQuota']['Memory']='16GB'
    elif case=='over':q['QuotaDetails']['AllocatedQuota']['GPU']='65'
    elif case=='gpu':q['QuotaDetails']['AllocatedQuota']['GPU']='NaN'
    elif case=='missing':del q['QuotaDetails']
    else:value.quota=lambda *_:(_ for _ in ()).throw(PaiError('GetQuota'))
    v=a.AllocationStatus().get('pai',{'backend':B})
    assert v['status']=='UNKNOWN' and v['resources'] is None


def test_capabilities_require_explicit_native_persistent_dataset():
    profile={'backend':B,'storage_root':'/data/ml-expd'}
    v=declaration(profile,artifact_store=True,desktop=False)
    assert not v['storage']['persistent'] and not v['data']['assets']
    assert v['data']['download_script'] is None and v['resources']['inventory_query']
    assert v['jobs']['exact_submission_lookup'] and v['observability']['queue_reason']
    mount={'DataSourceId':'d-data','DataSourceVersion':'v1','MountPath':'/data','MountAccess':'RW'}
    profile['backend']={**B,'storage_id':'nas-immutable','filesystem_type':'nas','data_sources':[mount]}
    v=declaration(profile,artifact_store=True,desktop=False)
    assert v['storage']['persistent'] and v['storage']['checkpoint_registration']
    assert v['data']['download_script']=='inside_job' and not v['data']['assets']
    assert storage_scope(profile['backend'],'/data/ml-expd')==v['storage']['storage_id']
    assert storage_scope({**profile['backend'],'quota_id':'another'},'/data/ml-expd')==v['storage']['storage_id']
    assert storage_scope({**profile['backend'],'storage_id':'another'},'/data/ml-expd')!=v['storage']['storage_id']
    mounts=[{},dict(mount,MountAccess='RO'),dict(mount,DataSourceVersion=''),dict(mount,DataSourceId=''),
            dict(mount,MountPath='relative'),dict(mount,MountPath='/other')]
    for bad in mounts:
        p=deepcopy(profile);p['backend']['data_sources']=[bad]
        assert not declaration(p,artifact_store=True,desktop=False)['storage']['persistent']
    profile['backend']['filesystem_type']='oss'
    assert not declaration(profile,artifact_store=True,desktop=False)['storage']['persistent']


def test_resume_matching_rejects_unmounted_pai_instead_of_key_error():
    profile={'backend':B,'storage_root':'/data/ml-expd'}
    v=declaration(profile,artifact_store=True,desktop=False)
    selected=match({'pai':profile},{'pai':v},request(),ExecutionRequirements(),ExecutorSelector(backend='pai-dlc'))
    assert selected['executor']=='pai'
    with pytest.raises(ApplicationError,match='checkpoint.storage_scope'):
        match({'pai':profile},{'pai':v},request(),ExecutionRequirements(),ExecutorSelector(backend='pai-dlc'),resume_scope='storage.any',project='test')


def test_server_operations_have_no_silent_sensecore_or_cpu_fallback(api,monkeypatch):
    value,_=api;p=backend_operations('pai-dlc');assert isinstance(p,PaiOperations)
    with pytest.raises(ApplicationError) as exc:p.prepare_asset(None,None,None,None,None,None,None)
    assert exc.value.code=='PAI_DATA_DELIVERY_UNAVAILABLE'
    assert p.prepare_script(None,None,None,None,None) is None
    assert p.script_progress(None,None,{'x':1})=={'x':1}
    with pytest.raises(ApplicationError):p.confirm_terminal(None,B,'dlc-test')
    value.job=lambda *args:{'Status':'Succeeded'}
    p.confirm_terminal(None,B,'dlc-test')
    with pytest.raises(ApplicationError) as exc:p.collect(None,None,None,None,None,None,None)
    assert exc.value.code=='PAI_RECOVERY_UNAVAILABLE'
    calls=[]
    monkeypatch.setattr('ml_exp_server.backends.operations.publish_existing',lambda *args:calls.append(args))
    assert p.collect('owner','identity','root','definition','transfer','operation','local')
    assert calls==[('owner','identity','local','transfer')]


@pytest.mark.parametrize('case',['workspaces_type','workspace_item','total_type','used_type','memory_type','details_type'])
def test_malformed_quota_shapes_fail_closed(api,case):
    value,q=api
    if case=='workspaces_type':q['Workspaces']=None
    elif case=='workspace_item':q['Workspaces']=[None]
    elif case=='details_type':q['QuotaDetails']=None
    elif case=='total_type':q['QuotaDetails']['ActualMinQuota']=None
    elif case=='used_type':q['QuotaDetails']['AllocatedQuota']=None
    else:q['QuotaDetails']['AllocatedQuota']['Memory']=None
    assert a.AllocationStatus().get('pai',{'backend':B})['status']=='UNKNOWN'


@pytest.mark.parametrize("policy,priority,max_time", [(None,None,None),("guaranteed",3,"02:00:00"),("allow_idle",5,"24:00:00"),("idle_only",9,None)])
def test_http_source_runtime_freeze_and_status_need_no_cloud_cli(client,api,policy,priority,max_time):
    """Exercise the real HTTP/storage/controller path; provider and OCI are fakes."""
    import yaml
    from pathlib import Path
    from tests.test_container_api import runtime
    from ml_exp_server.container_controller import Controller
    path=Path(client.app.state.runtime.config.container_execution.profiles_file)
    profiles=yaml.safe_load(path.read_text())
    profiles['executors']['pai']={'backend':B,'storage_root':'/data/lab',
                                  'capacity':{'gpus':8,'cpus':216,'memory_gb':1800}}
    path.write_text(yaml.safe_dump(profiles))
    listing=client.get('/api/executors').json()['executors']
    published=next(v for v in listing if v['id']=='pai')
    assert published['kind']=='pai-dlc' and published['dockerfile_execution']
    assert not published['capabilities']['data']['assets']
    assert client.get('/api/executors/pai/status').json()['resources']['gpus']['unallocated']==62
    image=runtime(client)
    definition={'run_id':'pai-http','runtime_id':image['runtime_id'],'executor':'pai','resources':{}}
    if max_time is not None:definition['resources']['max_time']=max_time
    if policy is not None:definition['resources']['allocation_policy']=policy
    if priority is not None:definition['resources']['priority']=priority
    response=client.post('/api/projects/demo/runs',json=definition)
    assert response.status_code==200,response.text
    assert response.json()['image']==image['image']
    root=Path(client.app.state.runtime.project('demo').base_dir)
    campaign=yaml.safe_load((root/'experiments/campaigns/run-pai-http.yaml').read_text())
    campaign['local_root']=str(client.app.state.runtime.config.project_run_root_path('demo'))
    controller=Controller(campaign,'pai-http','attempt-001')
    controller.prepare('campaign.'+'c'*64)
    frozen=controller.store.load_manifest()
    assert frozen['backend']['kind']=='pai-dlc' and frozen['backend']['quota_id']=='quota-test'
    assert frozen['image_id']==image['image'].split('@')[1]
    expected=max_time or '24:00:00'
    assert frozen['resources']['max_time']==frozen['backend']['time']==expected
    assert controller.duration_seconds()==(7200 if expected=='02:00:00' else 86400)
    from experiment_control.backends.pai import ALLOCATION_POLICIES
    assert frozen['backend']['oversold_type']==ALLOCATION_POLICIES[policy or 'guaranteed']
    assert frozen['backend']['priority']==(priority if priority is not None else 5)
    assert client.post('/api/projects/demo/runs',json=definition).status_code==200
    value,_=api
    value.workspace=lambda identity:{'WorkspaceId':identity}
    value.jobs=lambda *args,**kw:[]
    assert controller.backend.preflight(controller.run,scope='submit').ready
    assert controller.backend.identity(campaign,controller.run,'attempt-001').available
    document=json.loads(controller.backend.render(controller.store.load_attempt('attempt-001')))
    assert document['Priority']==(priority if priority is not None else 5)
    assert document['JobSpecs'][0]['OversoldType']==ALLOCATION_POLICIES[policy or 'guaranteed']
    assert document['JobSpecs'][0]['UseSpotInstance'] is False
    assert document['ResourceId']=='quota-test' and document['JobSpecs'][0]['PodCount']==1
    assert document['JobMaxRunningTimeMinutes']==(120 if expected=='02:00:00' else 1440)
    assert 'access_key_secret' not in json.dumps(document) and 'ImageConfig' not in document['JobSpecs'][0]


def test_personal_cpfs_enables_same_storage_recovery_across_quota():
    backend={**B,'storage_id':'cpfs-approved','filesystem_type':'cpfs',
        'cpfs_uri':'bmcpfs://cpfs-approved.us-southeast-1.cpfs.aliyuncs.com/',
        'workspace_user':'owner','data_role_arn':'acs:ram::123:role/pai-data-owner'}
    profile={'backend':backend,'storage_root':'/mnt/workspace/ml-expd'}
    value=declaration(profile,artifact_store=True,desktop=False)
    assert value['storage']['persistent'] and value['storage']['checkpoint_registration']
    assert value['data']['download_script']=='inside_job' and not value['data']['assets']
    assert storage_scope({**backend,'quota_id':'another'},profile['storage_root'])==value['storage']['storage_id']
    assert storage_scope({**backend,'workspace_user':'another'},profile['storage_root'])!=value['storage']['storage_id']
    assert storage_scope({**backend,'cpfs_uri':backend['cpfs_uri'].replace('cpfs-approved','cpfs-other')},profile['storage_root'])!=value['storage']['storage_id']
    for path in ['/mnt/models/weights','/mnt/datasets/data','/other']:
        assert not declaration({**profile,'storage_root':path},artifact_store=True,desktop=False)['storage']['persistent']


def test_resource_defaults_preserve_historical_intents_and_freeze_new_choices():
    from ml_exp_server.container_execution import Resources,pai_scheduling
    old={'gpus':1,'cpus':8,'memory_gb':32,'max_time':'00:10:00'}
    assert Resources().model_dump()==old
    assert pai_scheduling(Resources(),None)=={'oversold_type':'ForbiddenQuotaOverSold','priority':5}
    assert pai_scheduling(Resources(),{'backend':{'kind':'pai-dlc'}})=={}
    frozen={'backend':{'oversold_type':'ForbiddenQuotaOverSold','priority':5}}
    assert pai_scheduling(Resources(),frozen)==frozen['backend']
    explicit=Resources(allocation_policy='idle_only',priority=9)
    assert pai_scheduling(explicit,{'backend':{}})=={'oversold_type':'ForceQuotaOverSold','priority':9}
    assert explicit.model_dump()['allocation_policy']=='idle_only'
    for value in [True,0,10,'5']:
        with pytest.raises(ValueError):Resources(priority=value)


def test_dlc_controls_are_rejected_for_other_backends_before_build(client):
    from tests.test_container_api import runtime
    image=runtime(client)
    for field,value in [('allocation_policy','allow_idle'),('priority',5)]:
        response=client.post('/api/projects/demo/runs',json={'run_id':'not-pai-'+field,'executor':'gpu','runtime_id':image['runtime_id'],'resources':{field:value}})
        assert response.status_code==409 and 'resources.'+field in response.text


def test_legacy_pai_resource_fields_remain_absent_on_read_recreation(client,monkeypatch):
    import yaml
    from pathlib import Path
    from tests.test_container_api import runtime
    from ml_exp_server import container_execution as module
    path=Path(client.app.state.runtime.config.container_execution.profiles_file)
    profiles=yaml.safe_load(path.read_text());profiles['executors']['pai']={'backend':B,'storage_root':'/data/lab','capacity':{'gpus':8,'cpus':216,'memory_gb':1800}};path.write_text(yaml.safe_dump(profiles))
    image=runtime(client);definition={'run_id':'legacy-pai','runtime_id':image['runtime_id'],'executor':'pai'}
    scheduling=module.pai_scheduling
    walltime=module.run_walltime
    # Freeze a legacy manifest first, then restore the upgraded writer.
    monkeypatch.setattr(module,'pai_scheduling',lambda resources,previous:{})
    monkeypatch.setattr(module,'run_walltime',lambda resources,*args:resources.max_time)
    assert client.post('/api/projects/demo/runs',json=definition).status_code==200
    monkeypatch.setattr(module,'pai_scheduling',scheduling)
    monkeypatch.setattr(module,'run_walltime',walltime)
    response=client.post('/api/projects/demo/runs',json=definition)
    assert response.status_code==200,response.text
    assert 'priority' not in response.json()['resources'] and 'allocation_policy' not in response.json()['resources']
    assert response.json()['resources']['max_time']=='00:10:00'


def test_dlc_platform_walltime_cap_and_explicit_overrides():
    from ml_exp_server.container_execution import Resources,run_walltime
    assert run_walltime(Resources(),'pai-dlc')=='24:00:00'
    assert run_walltime(Resources(max_time='01:00:00'),'pai-dlc')=='01:00:00'
    assert run_walltime(Resources(),'pai-dlc',{'resources':{'max_time':'08:00:00'}})=='08:00:00'
    assert run_walltime(Resources(),'slurm')=='00:10:00'
    assert run_walltime(Resources(max_time='72:00:00'),'sensecore')=='72:00:00'
    with pytest.raises(ApplicationError) as error:run_walltime(Resources(max_time='24:00:01'),'pai-dlc')
    assert error.value.code=='PAI_DURATION_LIMIT'
