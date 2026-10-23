"""CCI identity pinning and own DNAT lifecycle; never migrate an existing port."""
from copy import deepcopy
import pytest
from experiment_control.backends.sensecore_rest import RESTError
from ml_exp_server.debug_cci import CciDebug, subset, startup
from tests.test_debug_sessions import KEY


W={'region':'cn-test-01','subscription_name':'sub','resource_group_name':'group','zone':'cn-test-01e','name':'ws'}
POOL={**W,'name':'debug','vpc_id':'vpc'}
E={**W,'name':'eip','id':'/scope/eip','properties':{'vpc_id':'vpc','association_id':'nat'}}
SPEC={'name':'cpu','cpu':{'vcpu_allocatable':2},'memory':{'allocatable':4},'device':{'number':0}}
P={'provider':'sensecore-cci','cpus':2,'memory_gib':4,'gpus':0,'native':{'workspace':'ws','aec2':'debug','worker_spec':'cpu',
    'image':'registry/image:tag','mounts':[{'type':'PV_AFS','mount_path':'/data','id':'nas','subdir':'/alice'}],
    'eip_name':'eip','external_ip':'192.0.2.1'}}


class API:
    def __init__(self):self.app=None;self.service=None;self.rows=[];self.calls=[];self.uid='owner'
    def workspace(self,name):return W
    def identity(self):return self.uid
    def pool(self,c):return POOL
    def specs(self,c):return [SPEC]
    def resources(self,kind):return [deepcopy(E)]
    def pages(self,url,field):return deepcopy(self.rows)
    def request(self,url,**kw):
        self.calls.append((url,deepcopy(kw)))
        method=kw.get('method','GET');body=kw.get('body')
        if url.endswith('/me'):return {'id':self.uid,'tenant_id':'tenant'}
        if '/dnatRules/' in url:
            if url.endswith('/bind'):
                self.rows[0]=deepcopy(body);self.rows[0]['state']='ACTIVE';return {}
            if url.endswith('/unbind'):
                self.rows[0]['properties'].update(internal_instance_name='',internal_instance_type='UNSPECIFIED',internal_port='0');self.rows[0]['state']='CREATED';return {}
            if method=='POST':self.rows=[{**deepcopy(body),'state':'CREATED'}];return {}
            if method=='DELETE':self.rows=[];return {}
        elif '/services' in url:
            if method=='POST':self.service={**deepcopy(body),'uid':'service-id'};return {}
            if method=='DELETE':self.service=None;return {}
            return deepcopy(self.service) or {'ports':[]}
        elif '/apps' in url:
            if url.endswith(':stop'):self.app['state']='SUSPENDED';self.app['replicas']=0;return {}
            if method=='POST':self.app={**deepcopy(body),'uid':'app-id','state':'RUNNING','ownership':{'user_id':self.uid}};return {}
            if method=='DELETE':self.app=None;return {}
            if self.app is not None:return deepcopy(self.app)
        raise RESTError('missing',status=404)


def record():return {'name':'ml-expd-test','identity':'hash','provider_id':None,'request':{'ssh_public_key':KEY},
    'expires_at':2000000000,'owner':'owner','create_request_id':'request','service_request_id':'service'}


def harness():
    api=API();p=CciDebug(deepcopy(P),api);r=record();r['document']=p.document(r);claims=[]
    def claim(k):
        if k in claims:return False
        claims.append(k);return True
    return api,p,r,claim


def test_complete_cci_and_owned_network_lifecycle():
    api,p,r,claim=harness();saved=[]
    for _ in range(4):value=p.provision(r,lambda:saved.append(deepcopy(r)),claim)
    assert value['uid']=='app-id' and r['provider_id']=='app-id'
    assert r['connection']['host']=='192.0.2.1' and 20000<=r['connection']['port']<=65535
    assert r['expiry_enforcement']=='server_cleanup_and_absolute_sshd_deadline'
    assert len([kw for _,kw in api.calls if kw.get('body',{}).get('name')==r['name'] and kw.get('method')=='POST'])==4
    assert p.connection(value) is None and p.state(value)=='RUNNING'
    script=startup(r);assert 'PasswordAuthentication no' in script and 'exec timeout' in script and 'apt-get' not in script
    p.stop(r,value);assert api.app['state']=='SUSPENDED'
    value=p.lookup(r);assert p.state(value)=='STOPPED'
    for _ in range(6):
        done=p.stop(r,p.lookup(r))
        if done:break
    assert done is True and api.app['state']=='SUSPENDED' and api.service is not None and api.rows==[]
    assert not any('/apps/' in url and kw.get('method')=='DELETE' for url,kw in api.calls)
    assert p.state({'state':'FAILED'})=='FAILED' and p.state({})=='STARTING'


def test_uncertain_create_or_service_is_not_replayed():
    api,p,r,claim=harness()
    assert p.provision(r,lambda:None,lambda k:False) is None
    p.provision(r,lambda:None,claim)
    api.service=None
    before=len(api.calls);p.provision(r,lambda:None,lambda k:False)
    assert not any(kw.get('method')=='POST' for _,kw in api.calls[before:])
    api.app['ownership']['user_id']='someone-else'
    with pytest.raises(ValueError,match='identity'):p.lookup(r)


def test_service_replacement_and_wrong_ports_are_never_mutated():
    for change in [{'uid':'other'},{'selector':{'app':'other'}},{'ports':[{'port':80,'target_port':80}]}]:
        api,p,r,claim=harness();p.provision(r,lambda:None,claim)
        api.service.update(change)
        with pytest.raises(ValueError,match='service'):p.provision(r,lambda:None,claim)
    api,p,r,claim=harness();p.provision(r,lambda:None,claim);api.app=None;api.rows=[]
    api.service['uid']='other'
    with pytest.raises(ValueError,match='stopping only'):p.stop(r,None,delete=True)


def test_unrelated_port_rule_is_never_bound_unbound_or_deleted():
    api,p,r,claim=harness()
    p.provision(r,lambda:None,claim)
    original=deepcopy(api.rows[0])
    for key,value in [('uid','someone-elses-rule'),('creator_id','other')]:
        api.rows[0]=deepcopy(original);api.rows[0][key]=value
        with pytest.raises(ValueError,match='ownership'):p.network(r,lambda:None,claim)
    for key,value in [('external_port','123'),('internal_instance_name','someone-elses-cci'),('internal_instance_type','VM')]:
        api.rows[0]=deepcopy(original);api.rows[0]['properties'][key]=value
        with pytest.raises(ValueError,match='ownership'):p.rule(r)


def test_mount_scope_resource_and_network_preflights(monkeypatch):
    for change in [{'extra':'forbidden'},{'worker_spec':'unknown'},{'image':'bad image'},
                   {'mounts':[]},{'mounts':[{'type':'PV_AFS','mount_path':'/','id':'nas','subdir':'/alice'}]},
                   {'mounts':[{'type':'PV_AFS','mount_path':'/data','id':'nas','subdir':'../other'}]}]:
        api,p,r,_=harness();p.config.update(change)
        with pytest.raises(ValueError):p.document(r)
        assert not api.calls
    api,p,r,_=harness();p.profile['cpus']=3
    with pytest.raises(ValueError,match='specification'):p.document(r)
    monkeypatch.setattr(api,'resources',lambda kind:[])
    with pytest.raises(ValueError,match='EIP'):p.eip()
    monkeypatch.setattr(api,'resources',lambda kind:[{**E,'zone':'elsewhere'}])
    with pytest.raises(ValueError,match='network'):p.eip()


def test_gpu_explicit_resource_key(monkeypatch):
    api,p,r,_=harness();p.profile['gpus']=1
    spec=deepcopy(SPEC);spec['device'].update(number=1,resource_key='nvidia.com/gpu')
    monkeypatch.setattr(api,'specs',lambda c:[spec]);body=p.document(r)
    assert body['template']['containers'][0]['resource_request']['nvidia.com/gpu']=='1'
    spec['device']['resource_key']='bad'
    with pytest.raises(ValueError,match='resource key'):p.document(r)


def test_dnat_invalid_port_data_and_owner_block_mutation(monkeypatch):
    for port in ['bad','0','65536','4-1']:
        api,p,r,claim=harness();api.rows=[{'name':'other','properties':{'external_port':port}}]
        with pytest.raises(ValueError,match='port'):p.network(r,lambda:None,claim)
    api,p,r,claim=harness();api.rows=[{'name':'other','properties':{'external_port':'1-65535'}}]
    with pytest.raises(ValueError,match='exhausted'):p.network(r,lambda:None,claim)
    api,p,r,claim=harness();api.uid='changed'
    with pytest.raises(ValueError,match='account'):p.network(r,lambda:None,claim)
    monkeypatch.setattr(api,'request',lambda *a,**k:(_ for _ in ()).throw(RESTError('forbidden',status=403)))
    with pytest.raises(RESTError):p.optional('https://example.test')
    assert subset({'a':[1,{'b':2}]},{'a':[1,{'b':2,'extra':True}]})
    assert not subset([1],[1,2]) and not subset({'a':1},{})


def test_pending_visibility_and_inflight_port_writes_are_never_replayed(monkeypatch):
    api,p,r,claim=harness();monkeypatch.setattr(p,'lookup',lambda r:None)
    assert p.provision(r,lambda:None,claim) is None
    api,p,r,claim=harness();p.provision(r,lambda:None,claim)
    api.rows=[]
    before=len(api.calls);p.network(r,lambda:None,lambda k:False)
    assert len(api.calls)==before
    p.network(r,lambda:None,lambda k:True)
    before=len(api.calls);p.network(r,lambda:None,lambda k:False)
    assert len(api.calls)==before
    api.rows[0]['state']='CREATING';p.network(r,lambda:None,claim)
    assert p.stop(r,None) is False
    api.app['state']='SUSPENDED';assert p.stop(r,api.app) is False
    api.rows[0]['state']='DELETING';assert p.stop(r,api.app) is False
    r.pop('network_plan');assert p.stop(r,api.app) is True


def test_delete_stops_compute_before_attempting_port_cleanup():
    api,p,r,claim=harness();p.provision(r,lambda:None,claim)
    api.rows[0]['properties']['internal_port']='22';api.rows[0]['properties']['internal_instance_name']='app-id'
    assert p.stop(r,api.app) is False and api.app['state']=='SUSPENDED'
    assert api.calls[-1][0].endswith(':stop')
    api.app['state']='SUSPENDING';before=len(api.calls)
    assert p.stop(r,api.app) is False and len(api.calls)==before
    api.rows[0]['properties']['internal_port']='23'
    with pytest.raises(ValueError,match='ownership'):p.rule(r)
