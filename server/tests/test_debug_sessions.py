"""Debug sessions never borrow Runs or replay an uncertain cloud creation."""
import base64
from copy import deepcopy
import json
import struct
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from ml_exp_server.api.app import create_app
from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.debug_models import DebugRequest, public_key
from ml_exp_server import debug_sessions as d
from ml_exp_server.schemas import ServerConfig
ORIGINAL_ADAPTER=d.adapter


def string(v):
    return struct.pack('>I', len(v)) + v

KEY = 'ssh-ed25519 ' + base64.b64encode(string(b'ssh-ed25519') + string(bytes(range(32)))).decode()
PROFILE = {'provider':'pai-dsw','description':'CPU debug','cpus':4,'memory_gib':8,'gpus':0,'native':{}}


class Fake:
    def __init__(self, profile):
        self.profile = profile
        self.exists = False
        self.creates = 0
        self.deleted = 0
        self.stopped = False
        self.uncertain = False
        self.fail = False
    def document(self, record):return {'safe':'scope'}
    def lookup(self, record):return {'id':'native-id','stopped':self.stopped} if self.exists else None
    def provision(self, record, save, claim):
        if not self.exists and claim('create'):
            self.creates += 1
            if self.uncertain:raise OSError('private token: NEVER_EXPOSE')
            self.exists = True
            record['provider_id'] = 'native-id'
            save()
        if self.fail:raise ValueError('private detail')
        if self.exists:record['expiry_enforcement']='provider_shutdown_timer'
        return self.lookup(record)
    def state(self, value):return 'STOPPED' if value['stopped'] else 'RUNNING'
    def connection(self, value):return {'host':'example.test','port':2222,'user':'root','access':'internet'}
    def stop(self, record, value, delete=False):
        if delete:
            self.deleted += 1
            self.exists = False
        else:self.stopped = True
        return True


@pytest.fixture
def service(tmp_path, monkeypatch):
    profiles=tmp_path/'profiles.json';profiles.write_text(json.dumps({'profiles':{'cpu':PROFILE}}))
    config=ServerConfig(index_db=str(tmp_path/'index.sqlite'),telemetry={'enabled':False},collector_enabled=False,
                        debug_sessions={'profiles_file':str(profiles)},action_runtime={'allow_scheduler_mutations':True})
    fake=Fake(PROFILE);monkeypatch.setattr(d,'adapter',lambda p:fake)
    return d.DebugSessions(config),fake


def req(**kwargs):return DebugRequest(session_id='test-one', profile='cpu', ssh_public_key=KEY, **kwargs)


def test_complete_lifecycle_and_frozen_identity(service):
    s,f=service
    assert s.catalog()['profiles'][0]['cpus']==4
    v=s.create(req()); assert v['status']=='RUNNING' and v['connection']['ssh_command']=='ssh -p 2222 root@example.test'
    assert 'native' not in v and v['connection']['host_key_verified'] is False
    assert s.create(req())==v and f.creates==1
    assert s.read('test-one')==v and len(s.listing()['sessions'])==1
    assert s.refresh('test-one',intent='stop')['status']=='STOPPING'
    assert s.refresh('test-one')['status']=='STOPPED'
    assert s.refresh('test-one',intent='delete')['status']=='DELETING'
    assert s.refresh('test-one')['status']=='DELETED'
    assert s.refresh('test-one',intent='stop')['status']=='DELETED'
    assert f.deleted==1 and not f.exists


def test_lost_create_is_never_replayed_and_remains_capacity_owned(service):
    s,f=service;f.uncertain=True
    value=s.create(req());assert value['status']=='RECONCILE_REQUIRED'
    assert 'NEVER_EXPOSE' not in json.dumps(value)
    assert s.refresh('test-one')['error']['code']=='DEBUG_PROVIDER_UNCERTAIN'
    assert s.create(req())['status']=='RECONCILE_REQUIRED' and f.creates==1
    assert s.refresh('test-one',intent='delete')['status']=='RECONCILE_REQUIRED'
    # Later provider visibility recovers the original resource, without a second create.
    f.exists=True
    assert s.refresh('test-one')['status']=='DELETING'
    assert f.creates==1


def test_restart_expiry_cleanup_does_not_require_profile_to_remain_enabled(service, monkeypatch):
    s,f=service;s.create(req());clock=s.load('test-one')['expires_at']+1
    monkeypatch.setattr(d.time,'time',lambda:clock)
    s.config.debug_sessions.profiles_file=None
    other=d.DebugSessions(s.config)
    assert other.read('test-one')['connection'] is None
    other.maintain();other.maintain()
    assert other.read('test-one')['status']=='DELETED' and not f.exists


def test_request_scope_capacity_and_policy(service):
    s,f=service;s.config.debug_sessions.max_active=1;s.create(req())
    with pytest.raises(ApplicationError,match='another request'):s.create(req(duration_seconds=600))
    with pytest.raises(ApplicationError,match='capacity'):s.create(req().model_copy(update={'session_id':'another'}))
    with pytest.raises(ApplicationError,match='duration'):s.create(req(duration_seconds=8000).model_copy(update={'session_id':'other'}))
    with pytest.raises(ApplicationError,match='unknown'):s.create(req().model_copy(update={'profile':'missing','session_id':'other'}))
    with pytest.raises(ApplicationError,match='not found'):s.read('missing')
    s.config.action_runtime.allow_scheduler_mutations=False
    assert s.catalog()['enabled'] is False
    s.maintain()
    with pytest.raises(ApplicationError,match='disabled'):s.refresh('test-one')


@pytest.mark.parametrize('key',['PRIVATE KEY','ssh-dss AAAA',KEY+'\n'+KEY,'command="curl bad" '+KEY,
                                'ssh-ed25519 !!!!','ssh-ed25519 AAAA','ssh-rsa AAAA',None,'x'*16385])
def test_invalid_public_keys(key):
    with pytest.raises(ValueError):public_key(key)


def test_public_key_validation_strips_comment_and_rejects_truncated_fields():
    assert public_key(KEY+' optional comment')==KEY
    for blob in [string(b'ssh-ed25519')+string(b'x'),string(b'ssh-ed25519')+b'\0\0\0\xffx',string(b'ssh-rsa')+string(b'x'*32)]:
        with pytest.raises(ValueError):public_key('ssh-ed25519 '+base64.b64encode(blob).decode())
    rsa=string(b'ssh-rsa')+string(b'\x01\x00\x01')+string(b'\0\x80'+b'0'*255)
    assert public_key('ssh-rsa '+base64.b64encode(rsa).decode()).startswith('ssh-rsa ')


def test_durable_identity_tamper_preflight_and_errors(service,monkeypatch):
    s,f=service
    monkeypatch.setattr(f,'document',lambda r:(_ for _ in ()).throw(ValueError('secret')))
    with pytest.raises(ApplicationError,match='preflight'):s.create(req())
    assert not s.store('test-one').path.exists() and f.creates==0
    monkeypatch.setattr(f,'document',lambda r:{})
    s.create(req());v=s.load('test-one');v['request']['profile']='changed';s.save(v)
    with pytest.raises(ValueError,match='frozen'):s.read('test-one')
    assert d.safe_error(SimpleNamespace(details={'provider_code':'secret://value','operation':'CreateInstance','http_status':403,'uncertain':False}))=={
        'code':'DEBUG_PROVIDER_UNCERTAIN','error_class':'SimpleNamespace','operation':'CreateInstance','http_status':403,'uncertain':False}


def test_http_routes_admission_and_redacted_invalid_inputs(service):
    s,f=service
    with TestClient(create_app(s.config,poll=False)) as client:
        assert client.get('/api/debug-session-profiles').json()['enabled'] is True
        assert client.get('/api/debug-sessions').json()=={'sessions':[]}
        bad=client.post('/api/debug-sessions',json={**req().model_dump(),'ssh_public_key':'PRIVATE KEY super-secret'})
        assert bad.status_code==422 and 'super-secret' not in bad.text
        v=client.post('/api/debug-sessions',json=req().model_dump())
        assert v.status_code==202 and v.json()['status']=='RUNNING'
        assert client.get('/api/debug-sessions/test-one').status_code==200
        assert client.post('/api/debug-sessions/test-one/refresh').status_code==200
        assert client.post('/api/debug-sessions/test-one/stop').status_code==200
        assert client.delete('/api/debug-sessions/test-one').status_code==200
        assert client.post('/api/debug-sessions/test-one/refresh').json()['status']=='DELETED'
        reservations=[client.app.state.job_executors['action'].reserve() for _ in range(16)]
        try:assert client.post('/api/debug-sessions/test-one/refresh').status_code==503
        finally:
            for r in reservations:r.release()


def test_disabled_catalog_invalid_profiles_ids_and_ssh_route(service):
    s,f=service;s.config.debug_sessions.profiles_file=None
    assert s.catalog()['profiles']==[]
    for key in ['../elsewhere','.','a/b','X']:
        with pytest.raises(ApplicationError):s.read(key)
    s.config.debug_sessions.profiles_file=str(s.root/'bad.json');s.root.mkdir(parents=True,exist_ok=True)
    (s.root/'bad.json').write_text('{}')
    with pytest.raises(ValueError,match='profile'):s.profiles()
    s.config.debug_sessions.profiles_file=None
    record={'status':'RUNNING','expires_at':float('inf'),'profile':PROFILE,'request':{'profile':'cpu'},'connection':{'host':'-oProxyCommand=bad','port':22}}
    with pytest.raises(ValueError,match='SSH'):s.view(record)


def test_rejected_create_is_terminal_but_post_create_get_failure_is_uncertain(service,monkeypatch):
    from experiment_control.backends.pai_api import PaiError
    s,f=service
    def rejected(record,save,claim):
        claim('create');raise PaiError('CreateInstance',status=400,code='ValidationError',uncertain=False)
    monkeypatch.setattr(f,'provision',rejected)
    value=s.create(req());assert value['status']=='REJECTED' and value['error']['code']=='DEBUG_PROVIDER_REJECTED'
    assert s.refresh('test-one')==value
    assert s.refresh('test-one',intent='delete')['status']=='DELETED'
    def accepted(record,save,claim):
        claim('create');record['create_accepted']=True;save();raise PaiError('GetInstance',status=403,uncertain=False)
    monkeypatch.setattr(f,'provision',accepted)
    value=s.create(req().model_copy(update={'session_id':'two'}));assert value['status']=='RECONCILE_REQUIRED'


def test_factory_and_cci_stop_delete_states(service,monkeypatch):
    from ml_exp_server import debug_dsw,debug_cci
    s,f=service
    monkeypatch.setattr(debug_dsw,'DswDebug',lambda p:('dsw',p));monkeypatch.setattr(debug_cci,'CciDebug',lambda p:('cci',p))
    assert ORIGINAL_ADAPTER({'provider':'pai-dsw'})[0]=='dsw'
    assert ORIGINAL_ADAPTER({'provider':'sensecore-cci'})[0]=='cci'
    s.create(req());r=s.load('test-one');r['profile']['provider']='sensecore-cci';r['identity']=d.identity({'request':r['request'],'profile':r['profile']});s.save(r)
    assert s.refresh('test-one',intent='stop')['status']=='STOPPING'
    assert s.refresh('test-one')['status']=='STOPPED'
    monkeypatch.setattr(f,'stop',lambda *a,**k:False)
    with pytest.raises(ApplicationError,match='stopping only'):s.refresh('test-one',intent='delete')
    r=s.load('test-one');r['intent']='delete';s.save(r)
    monkeypatch.setattr(f,'stop',lambda *a,**k:True)
    assert s.refresh('test-one')['intent']=='stop'
    s.maintain()


def test_claimed_but_not_visible_creation_retains_generic_diagnostic(service,monkeypatch):
    s,f=service
    def invisible(r,save,claim):
        claim('create');claim('create');return None
    monkeypatch.setattr(f,'provision',invisible)
    assert s.create(req())['error']['code']=='DEBUG_CREATION_UNCERTAIN'
    assert s.refresh('test-one',intent='delete')['error']['code']=='DEBUG_CREATION_UNCERTAIN'


def test_expiry_loop_survives_fault_and_observes_stop():
    from ml_exp_server.api.lifecycle import _debug_loop
    stop=SimpleNamespace(is_set=lambda:False,wait=lambda seconds:True)
    for fail in [False,True]:
        def maintain():
            if fail:raise ValueError('private input')
        app=SimpleNamespace(state=SimpleNamespace(_stop=stop,debug_sessions=SimpleNamespace(maintain=maintain)))
        _debug_loop(app)
        assert app.state.debug_session_error==('ValueError' if fail else None)


def test_creation_keeps_provider_owner_and_maintenance_skips_unexpired(service):
    s,f=service;f.owner='owner';s.create(req());assert s.load('test-one')['owner']=='owner'
    s.maintain();assert s.read('test-one')['status']=='RUNNING'


def test_running_does_not_imply_ssh_route(service,monkeypatch):
    s,f=service;monkeypatch.setattr(f,'connection',lambda v:None)
    v=s.create(req());assert v['status']=='RUNNING' and v['connection'] is None


def test_disabled_profile_is_visible_but_cannot_allocate(service):
    s,f=service;path=s.config.debug_sessions.profiles_file
    from pathlib import Path
    Path(path).write_text(json.dumps({'profiles':{'cpu':{**PROFILE,'enabled':False,'unavailable_reason':'cleanup permission missing'}}}))
    assert s.catalog()['enabled'] is False and s.catalog()['profiles'][0]['enabled'] is False
    with pytest.raises(ApplicationError,match='disabled'):s.create(req())
    assert f.creates==0


def test_permission_failure_keeps_machine_evidence_and_requires_explicit_retry(service,monkeypatch):
    from experiment_control.backends.pai_api import PaiError
    s,f=service;s.create(req())
    def forbidden(*a,**k):raise PaiError('DeleteInstance',status=403,uncertain=False)
    monkeypatch.setattr(f,'stop',forbidden)
    r=s.refresh('test-one',intent='delete')
    assert r['machine_status']=='RUNNING' and r['error']['http_status']==403
    monkeypatch.setattr(f,'lookup',lambda r:(_ for _ in ()).throw(AssertionError('automatic retry forbidden')))
    s.maintain()
    assert s.read('test-one')['error']['http_status']==403


def test_expiry_loop_can_exit_before_first_cloud_query():
    from ml_exp_server.api.lifecycle import _debug_loop
    _debug_loop(SimpleNamespace(state=SimpleNamespace(_stop=SimpleNamespace(is_set=lambda:True))))


def test_expiry_poller_observes_stop_between_cycles():
    from ml_exp_server.api.lifecycle import _debug_loop
    calls=iter([False,True])
    app=SimpleNamespace(state=SimpleNamespace(_stop=SimpleNamespace(is_set=lambda:next(calls),wait=lambda t:False),
                                             debug_sessions=SimpleNamespace(maintain=lambda:None)))
    _debug_loop(app)


def test_cci_expiry_stops_without_deleting_instance(service,monkeypatch):
    s,f=service;s.create(req());r=s.load('test-one');r['profile']['provider']='sensecore-cci'
    r['expires_at']=0;r['identity']=d.identity({'request':r['request'],'profile':r['profile']});s.save(r)
    s.maintain();s.maintain()
    v=s.read('test-one');assert v['intent']=='stop' and v['status']=='STOPPED' and f.deleted==0


def test_dsw_delete_cannot_be_downgraded_to_stop(service):
    s,f=service;s.create(req());assert s.refresh('test-one',intent='delete')['status']=='DELETING'
    assert s.refresh('test-one',intent='stop')['intent']=='delete'
