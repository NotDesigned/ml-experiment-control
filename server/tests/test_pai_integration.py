"""Third scheduler contracts: explicit capabilities and no invented data fallback."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
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
