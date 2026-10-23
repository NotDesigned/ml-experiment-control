"""Neither a missing receipt nor a stopped/uncertain DSW permits replay."""
from copy import deepcopy
import fcntl
import json
from types import SimpleNamespace
import pytest
from experiment_control.backends.pai_api import PaiError
from ml_exp_server.builds import pai_build_recovery as p
from ml_exp_server.builds.build_recovery import publication_absence,valid_absence
from ml_exp_server.builds.builder_client import BuildTransportError

ID='a'*64
LEASE={'phase':'DELETE_REQUESTED','bundle_id':ID,'workspace_id':'w','quota_id':'q','region':'cn-test','gpu':0,'instance_id':'dsw-old','name':'ml_exp_build_old'}


@pytest.fixture
def setup(tmp_path,monkeypatch):
    cfg={'publisher':'pai-dsw','pai_dsw':{'workspace_id':'w','quota_id':'q'},'repository':'registry.example/lab','registry_auth_file':'auth.json'}
    b=SimpleNamespace(config=cfg,root=tmp_path);(tmp_path/(ID+'.dsw.json')).write_text(json.dumps(LEASE))
    api=SimpleNamespace(config={'region':'cn-test'},calls=[])
    def request(product,action,path,**kw):
        api.calls.append((action,kw))
        if action=='GetInstance':raise PaiError(action,status=400,code='InstanceNotExistError')
        return {'Instances':[],'TotalCount':0}
    api.request=request;monkeypatch.setattr(p.PaiAPI,'from_environment',lambda:api)
    registry=SimpleNamespace(prefix='/v2/lab',request=lambda *a,**kw:(404,{},b''))
    monkeypatch.setattr(p,'Registry',lambda *a,**kw:registry)
    return b,api,registry


def test_two_observations_and_authenticated_absence_preserve_lease_before_retry(setup):
    b,api,r=setup;original=(b.root/(ID+'.dsw.json')).read_bytes()
    proof=publication_absence(b,ID);assert valid_absence(proof,ID)
    assert proof['source']=='pai-dsw-reconcile.v1' and [x[0] for x in api.calls]==['GetInstance','ListInstances']
    assert p.preserve_deleted_lease(b,ID)['leases_clear'] is True
    history=list((b.root/(ID+'.dsw-history')).glob('*.json'))
    assert len(history)==1 and json.loads(history[0].read_text())==LEASE and history[0].stat().st_mode&0o777==0o600
    assert (b.root/(ID+'.dsw.json')).read_bytes()==original
    with (b.root/(ID+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert publication_absence(b,ID) is None


@pytest.mark.parametrize('change',[{'phase':'CREATE_INTENT'},{'phase':'CREATED'},{'phase':'CLEANUP_REQUIRED'},
    {'workspace_id':'other'},{'quota_id':'other'},{'region':'elsewhere'},{'gpu':1},{'gpu':False},{'bundle_id':'b'*64}])
def test_unknown_or_wrong_scope_never_grants_retry(setup,change):
    b,api,r=setup;(b.root/(ID+'.dsw.json')).write_text(json.dumps({**LEASE,**change}))
    assert publication_absence(b,ID) is None and not api.calls
    with pytest.raises(BuildTransportError,match='RECONCILE_REQUIRED'):p.preserve_deleted_lease(b,ID)


def test_present_receipt_or_absent_lease_grants_nothing(setup):
    b,api,r=setup;(b.root/(ID+'.json')).write_text('{}');assert publication_absence(b,ID) is None
    (b.root/(ID+'.json')).unlink();(b.root/(ID+'.dsw.json')).unlink();assert publication_absence(b,ID) is None


@pytest.mark.parametrize('case',['active','forbidden','named_row','id_row','bad_row','bad_total','bad_list','malformed','not_mapping','registry_present','registry_error','duplicate','incomplete','missing_id','limit'])
def test_unchecked_or_present_native_state_cannot_become_safe(setup,case,monkeypatch):
    b,api,r=setup
    old=api.request
    def request(product,action,path,**kw):
        if action=='GetInstance':
            if case=='active':return {'Status':'Stopped'}
            if case=='forbidden':raise PaiError(action,status=403)
            return old(product,action,path,**kw)
        rows={'named_row':[{'InstanceName':LEASE['name']}], 'id_row':[{'InstanceId':'dsw-old'}],
              'bad_row':[None], 'missing_id':[{}], 'duplicate':[{'InstanceId':'same'}]}.get(case,[])
        total=2 if case in ['duplicate','incomplete'] else 99999 if case=='limit' else 0
        if case=='limit':rows=[{'InstanceId':str(kw['query']['PageNumber'])}]
        return {'Instances':None if case=='bad_list' else rows,'TotalCount':-1 if case=='bad_total' else total}
    monkeypatch.setattr(api,'request',request)
    if case in ['malformed','not_mapping']:(b.root/(ID+'.dsw.json')).write_text('invalid' if case=='malformed' else '[]')
    if case=='registry_present':r.request=lambda *a,**kw:(200,{},b'present')
    if case=='registry_error':r.request=lambda *a,**kw:(_ for _ in ()).throw(OSError('private'))
    if case in ['active','named_row','id_row','bad_row']:assert publication_absence(b,ID) is None
    else:
        with pytest.raises(BuildTransportError):publication_absence(b,ID)


def test_native_404_is_also_exact_absence_and_complete_other_rows_are_not_ours(setup,monkeypatch):
    b,api,r=setup
    def request(product,action,path,**kw):
        if action=='GetInstance':raise PaiError(action,status=404)
        return {'Instances':[{'InstanceId':'other','InstanceName':'different'}],'TotalCount':1}
    monkeypatch.setattr(api,'request',request)
    assert valid_absence(publication_absence(b,ID),ID)
