"""DSW native SSH, scoped mounts, lost responses, and timer enforcement."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from experiment_control.backends.pai_api import PaiError
from ml_exp_server.debug_dsw import DswDebug
from tests.test_debug_sessions import KEY


C={'workspace_id':'253','quota_id':'quota-test','image_id':'image-test','workspace_user':'alice',
   'cpfs_uri':'bmcpfs://cpfs-test.cn-test.cpfs.aliyuncs.com/','data_role_arn':'acs:ram::123:role/pai-data-alice',
   'user_vpc':{'VpcId':'vpc-abc','VSwitchId':'vsw-abc','SecurityGroupId':'sg-abc'},
   'ssh_access_type':'internet','ssh_nat_gateway_id':'ngw-abc','ssh_eip_id':'eip-abc'}
P={'provider':'pai-dsw','cpus':4,'memory_gib':8,'gpus':0,'native':C}


def record():return {'name':'ml-expd_test','identity':'hash','request':{'ssh_public_key':KEY},'provider_id':None,'expires_at':10**12}


class API:
    config={'region':'cn-test'}
    def __init__(self):self.value=None;self.calls=[];self.timer=500;self.bad_timer=False;self.missing=False
    def request(self,product,action,path,**kw):
        self.calls.append((action,path,kw))
        if action=='CreateInstance':
            self.value={**deepcopy(kw['body']),'InstanceId':'dsw-one','Status':'Running'}
            return {'InstanceId':'dsw-one'}
        if action=='GetInstance':
            if self.missing:raise PaiError(action,status=404)
            return deepcopy(self.value)
        if action=='ListInstances':return {'Instances':[] if self.value is None else [self.value],'TotalCount':int(self.value is not None)}
        if action=='CreateInstanceShutdownTimer':self.timer=kw['body']['RemainingTimeInMs'];return {}
        if action=='GetInstanceShutdownTimer':return {'InstanceId':'wrong' if self.bad_timer else 'dsw-one','Success':True,'RemainingTimeInMs':self.timer}
        if action=='StopInstance':self.value['Status']='Stopped'
        elif action=='DeleteInstance':self.missing=True
        return {}


def test_native_create_shutdown_connection_stop_delete(monkeypatch):
    api=API();p=DswDebug(deepcopy(P),api);r=record();saved=[];claims=[]
    def claim(k):
        if k in claims:return False
        claims.append(k);return True
    value=p.provision(r,lambda:saved.append(deepcopy(r)),claim)
    body=api.calls[1][2]['body']
    assert body['Accessibility']=='PRIVATE' and body['RequestedResource']['GPU']=='0'
    assert body['WorkspaceSource']=='/mnt/ml-expd-personal'
    assert [d['MountAccess'] for d in body['Datasets']]==['RW','RO','RO']
    assert body['UserVpc']['ForwardInfos'][0]['SSHPublicKey']==KEY
    assert 'ExternalPort' not in body['UserVpc']['ForwardInfos'][0]
    assert body['CredentialConfig']['Configs'][0]['Roles'][0]['RoleArn'].endswith('pai-data-alice')
    assert r['expiry_enforcement']=='provider_shutdown_timer'
    assert p.state(value)=='RUNNING' and p.connection(value) is None
    value['UserVpc']['ForwardInfos'][0]['ConnectInfo']={'Phase':'Ready','Internet':{'Endpoint':'example.test','Port':'2222'}}
    assert p.connection(value)['port']==2222
    assert p.lookup(r)['InstanceId']=='dsw-one'
    p.stop(r,value);assert api.value['Status']=='Stopped'
    p.stop(r,api.value,delete=True);assert p.lookup(r) is None
    api.missing=False;api.bad_timer=True
    with pytest.raises(ValueError,match='timer'):p.timer(r)


def test_creation_uncertainty_list_recovery_and_identity(monkeypatch):
    api=API();p=DswDebug(deepcopy(P),api);r=record()
    assert p.provision(r,lambda:None,lambda k:False) is None
    api.value={**p.document(r),'InstanceId':'dsw-one'}
    assert p.lookup(r)['InstanceId']=='dsw-one'
    api.value['Labels']=[]
    with pytest.raises(ValueError,match='identity'):p.lookup(r)
    for change in [{'InstanceId':None},{'InstanceName':'other'},{'WorkspaceId':'other'},{'ResourceId':'other'}]:
        v={**p.document(r),'InstanceId':'dsw-one',**change}
        with pytest.raises(ValueError,match='identity'):p.checked(r,v)
    monkeypatch.setattr(api,'request',lambda *a,**k:{'Instances':[{'InstanceName':r['name']}]*2,'TotalCount':2})
    with pytest.raises(ValueError,match='ambiguous'):p.lookup(r)
    monkeypatch.setattr(api,'request',lambda *a,**k:{'Instances':None,'TotalCount':-1})
    with pytest.raises(ValueError,match='pagination'):p.lookup(r)
    monkeypatch.setattr(api,'request',lambda *a,**k:{'Instances':[],'TotalCount':999999})
    with pytest.raises(ValueError,match='limit'):p.lookup(r)
    r['provider_id']='dsw-one'
    monkeypatch.setattr(api,'request',lambda *a,**k:(_ for _ in ()).throw(PaiError('GetInstance',status=403)))
    with pytest.raises(PaiError):p.lookup(r)


@pytest.mark.parametrize('change',[{'extra':'value'},{'workspace_user':''},{'ssh_access_type':'any'},
    {'ssh_nat_gateway_id':'wrong'},{'ssh_eip_id':None},{'user_vpc':{}},
    {'cpfs_uri':'bmcpfs://cpfs-test.cn-test.cpfs.aliyuncs.com/users/bob/'},
    {'data_role_arn':'acs:ram::123:role/admin'}])
def test_invalid_static_profile_is_rejected_before_create(change):
    p=DswDebug({**P,'native':{**C,**change}},API())
    with pytest.raises(ValueError):p.document(record())


def test_private_ssh_and_incomplete_routes_and_stop_states():
    api=API();p=DswDebug({**P,'native':{**C,'ssh_access_type':'intranet'}},api);r=record();body=p.document(r)
    assert 'NatGatewayId' not in body['UserVpc']['ForwardInfos'][0]
    v={**body,'InstanceId':'dsw-one','Status':'Stopped'};r['provider_id']='dsw-one'
    forward=v['UserVpc']['ForwardInfos'][0];forward['ConnectInfo']={'Phase':'Ready','Intranet':{'Endpoint':'internal.test','Port':'22'}}
    assert p.connection(v)['access']=='intranet'
    forward['Name']='other';assert p.connection(v) is None
    forward['Name']='ssh';forward['ForwardPort']='23';assert p.connection(v) is None
    forward['ForwardPort']='22';forward['ConnectInfo']['Phase']='Starting';assert p.connection(v) is None
    assert p.state(v)=='STOPPED' and p.state({'Status':'Failed'})=='FAILED' and p.state({})=='STARTING'
    p.stop(r,v);assert not api.calls
    p.stop(r,{**v,'Status':'Stopping'},delete=True);assert not api.calls
    api.value=v;p.stop(r,{**v,'Status':'Running'},delete=True);assert api.calls[-1][0]=='StopInstance'


def test_visibility_lag_and_native_missing_error_do_not_trigger_another_create(monkeypatch):
    api=API();api.missing=True;p=DswDebug(deepcopy(P),api);r=record()
    assert p.provision(r,lambda:None,lambda k:True) is None
    assert r['provider_id']=='dsw-one' and r['create_accepted'] is True
    api.missing=False
    p.provision(r,lambda:None,lambda k:False)
    assert len([c for c in api.calls if c[0]=='CreateInstance'])==1
    monkeypatch.setattr(api,'request',lambda *a,**k:(_ for _ in ()).throw(PaiError('GetInstance',status=400,code='InstanceNotExistError')))
    assert p.lookup(r) is None
    p.config.pop('user_vpc')
    with pytest.raises(ValueError,match='VPC'):p.document(record())
