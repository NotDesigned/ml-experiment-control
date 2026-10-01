"""Native tracking contract, isolation, and explicit upload boundary."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest
from fastapi.testclient import TestClient

from ml_exp_server.api.app import create_app
from ml_exp_server.api_contract import CLIENT_PROTOCOL_HEADER
from ml_exp_server.credentials import CredentialStore
from ml_exp_server.identity import workspace_identity
from ml_exp_server.ingest.indexer import RunIndex
from ml_exp_server.schemas import ServerConfig, TrackingConfig, RunIndexRow, AttemptSummary
from ml_exp_server.tracking import attempt_run_id, read_tracking, safe_url
from ml_exp_server.tracking_sync import sync_attempt


def evidence(root, workspace='w', project='p', run='r', attempt='a'):
    root.mkdir(parents=True,exist_ok=True)
    rid=attempt_run_id(workspace,project,run,attempt)
    payload={'schema_version':1,'identity':{'workspace_id':workspace,'project_name':project,'run_id':run,'attempt_id':attempt},
             'wandb_run_id':rid,'state':'FINISHED','mode':'offline','url':None}
    (root/'tracking.json').write_text(json.dumps(payload))
    file=root/'wandb'/('offline-run-test-'+rid)/f'run-{rid}.wandb'
    file.parent.mkdir(parents=True); file.write_bytes(b'native SDK fixture')
    return payload,file


def test_identity_and_untrusted_evidence(tmp_path):
    config=TrackingConfig()
    data,file=evidence(tmp_path/'collected_run')
    read=lambda:read_tracking(tmp_path,'w','p','r','a',config)
    assert read()['state']=='OFFLINE_READY'
    assert attempt_run_id('w','p','r','a')!=attempt_run_id('w','p','r','a2')
    assert attempt_run_id('w','p','r','a')!=attempt_run_id('w2','p','r','a')
    data['identity']['attempt_id']='wrong'
    (tmp_path/'collected_run/tracking.json').write_text(json.dumps(data))
    assert read()['state']=='INVALID_EVIDENCE'
    data['identity']['attempt_id']='a'
    (tmp_path/'collected_run/tracking.json').write_text(json.dumps(data))
    file.unlink(); file.symlink_to('/etc/passwd')
    assert read()['state']=='INVALID_EVIDENCE'


@pytest.mark.parametrize('url',[None,'https://user:secret@host/run','https://host/run?token=secret',
    'http://public.example/run','https://host:bad','https://host/#secret','https://host/\nrun', 'https://host\\evil'])
def test_unsafe_urls_are_rejected(url):
    assert safe_url(url) is None


def test_online_url_requires_exact_target(tmp_path):
    data,_=evidence(tmp_path)
    config=TrackingConfig(entity='team',project='experiments')
    data.update(mode='online',url='https://evil.example/run')
    (tmp_path/'tracking.json').write_text(json.dumps(data))
    assert read_tracking(tmp_path,'w','p','r','a',config)['dashboard_url'] is None
    data['url']=f'https://wandb.ai/team/experiments/runs/{data["wandb_run_id"]}'
    (tmp_path/'tracking.json').write_text(json.dumps(data))
    assert read_tracking(tmp_path,'w','p','r','a',config)['dashboard_url']==data['url']


def test_no_archive_publisher_and_api_retirement(tmp_path):
    config=ServerConfig(index_db=str(tmp_path/'index.sqlite'),action_root=str(tmp_path/'actions'),
                        tracking={'credential_root':str(tmp_path/'credentials')},telemetry={'enabled':False})
    with TestClient(create_app(config,poll=False,projects=[])) as client:
        client.headers[CLIENT_PROTOCOL_HEADER]='1'
        health=client.get('/api/health').json()
        assert 'tracking.v1' in health['capabilities'] and 'observability.v1' not in health['capabilities']
        assert not health['observability_mutations']
        assert client.get('/api/observability').json()['state']=='RETIRED'
        assert client.get('/api/tracking/attempts/p/r/a').status_code==404
        assert not hasattr(client.app.state.runtime,'observability')
        assert not hasattr(client.app.state,'_publisher_thread')
    assert not (tmp_path/'index.observability.sqlite').exists()
    with pytest.raises(ValueError):
        ServerConfig(observability={'local_wandb':{'enabled':True}})


def setup_sync(tmp_path):
    cfg=ServerConfig(index_db=str(tmp_path/'index.sqlite'),tracking={'credential_root':str(tmp_path/'credentials'),
                      'credential_ref':'cloud','entity':'team','project':'experiments'})
    root=tmp_path/'r'/'attempts'/'a'
    data,file=evidence(root/'collected_run',workspace_identity(cfg))
    index=RunIndex(cfg.index_db_path())
    index.upsert_run(RunIndexRow(project='p',run_id='r',run_dir=str(tmp_path/'r'),
                                 attempts=[AttemptSummary(attempt_id='a',state='SUCCEEDED')]))
    index.close()
    CredentialStore(Path(cfg.tracking.credential_root)).set_wandb_api_key('cloud','SECRET-NOT-IN-ARGV')
    return cfg,root,file


def test_sync_uses_official_cli_and_retains_markers_across_collection(tmp_path,monkeypatch):
    cfg,root,file=setup_sync(tmp_path)
    calls=[]
    def process(command,**kwargs):
        assert 'SECRET-NOT-IN-ARGV' not in repr(command)
        assert kwargs['env']['WANDB_API_KEY']=='SECRET-NOT-IN-ARGV'
        assert kwargs['env']['WANDB_BASE_URL']=='https://api.wandb.ai'
        assert kwargs['stdout']==subprocess.DEVNULL and '--skip-console' in command
        calls.append(command)
        path=Path(command[-1]); path.with_suffix(path.suffix+'.synced').touch()
        return SimpleNamespace(wait=lambda **kw:0)
    monkeypatch.setattr('ml_exp_server.tracking_sync.subprocess.Popen',process)
    assert sync_attempt(cfg,'p','r','a')['state']=='CLI_COMPLETED'
    assert sync_attempt(cfg,'p','r','a')['state']=='CLI_COMPLETED'
    assert len(calls)==1
    assert read_tracking(root,workspace_identity(cfg),'p','r','a',cfg.tracking)['sync_state']=='CLI_COMPLETED'
    file.write_bytes(b'changed')
    assert read_tracking(root,workspace_identity(cfg),'p','r','a',cfg.tracking)['sync_state']=='SOURCE_CHANGED'
    cfg.tracking.entity='another-team'
    assert 'sync_state' not in read_tracking(root,workspace_identity(cfg),'p','r','a',cfg.tracking)
    cfg.tracking.entity='team'
    with pytest.raises(ValueError,match='changed'):
        sync_attempt(cfg,'p','r','a')


def test_failed_sync_is_visible_without_touching_job_state(tmp_path,monkeypatch):
    cfg,root,_=setup_sync(tmp_path)
    monkeypatch.setattr('ml_exp_server.tracking_sync.subprocess.Popen',lambda *a,**kw:SimpleNamespace(wait=lambda **kw:1))
    before=cfg.index_db_path().read_bytes()
    assert sync_attempt(cfg,'p','r','a')['state']=='SYNC_FAILED'
    assert cfg.index_db_path().read_bytes()==before
    assert 'SECRET' not in (root/'tracking-sync.json').read_text()
    assert read_tracking(root,workspace_identity(cfg),'p','r','a',cfg.tracking)['sync_state']=='SYNC_FAILED'

@pytest.mark.parametrize('mutation',[
    'bad_mode','bad_state','error','recording','empty_file','empty_json','too_large','not_object','json_symlink',
])
def test_tracking_partial_and_invalid_evidence(tmp_path,mutation):
    data,file=evidence(tmp_path)
    path=tmp_path/'tracking.json'
    expected='INVALID_EVIDENCE'
    if mutation=='bad_mode':data['mode']='disabled'
    if mutation=='bad_state':data['state']='SUCCEEDED'
    if mutation=='error':data['state']='ERROR';expected='ERROR'
    if mutation=='recording':data['state']='RECORDING';expected='RECORDING'
    path.write_text(json.dumps(data))
    if mutation=='empty_file':file.write_bytes(b'')
    if mutation=='empty_json':path.write_text('')
    if mutation=='too_large':path.write_text(' '*8193)
    if mutation=='not_object':path.write_text('[]')
    if mutation=='json_symlink':path.unlink();path.symlink_to('/etc/passwd')
    assert read_tracking(tmp_path,'w','p','r','a',TrackingConfig())['state']==expected


def test_native_file_validation_and_loopback_urls(tmp_path):
    from ml_exp_server.tracking import native_files,read_json
    assert safe_url('http://127.0.0.1:8080')=='http://127.0.0.1:8080'
    assert safe_url('ftp://host') is None
    with pytest.raises(ValueError):native_files(tmp_path,'../bad')
    with pytest.raises(ValueError):native_files(tmp_path,'a'*32)
    (tmp_path/'folder').mkdir()
    with pytest.raises(ValueError):read_json(tmp_path,tmp_path/'folder')


@pytest.mark.parametrize('failure',['target_missing','unknown','not_finished','state_symlink','lock_symlink','destination_symlink',
    'changed_destination','segment_symlink','marker_missing','timeout','spawn','temporary_symlink'])
def test_sync_failure_boundaries(tmp_path,monkeypatch,failure):
    cfg,root,file=setup_sync(tmp_path)
    from ml_exp_server import tracking_sync
    if failure=='target_missing':cfg.tracking.entity=None
    if failure=='unknown':
        with pytest.raises(ValueError,match='terminal'):sync_attempt(cfg,'p','r','missing')
        return
    if failure=='not_finished':
        data=json.loads((file.parents[2]/'tracking.json').read_text());data['state']='RECORDING'
        (file.parents[2]/'tracking.json').write_text(json.dumps(data))
    staged=root/'tracking-upload';staged.mkdir()
    if failure=='lock_symlink':(root/'.tracking-sync.lock').symlink_to('/etc/passwd')
    if failure=='state_symlink':(root/'tracking-sync.json').symlink_to('/etc/passwd')
    if failure=='destination_symlink':(staged/'destination.json').symlink_to('/etc/passwd')
    if failure=='changed_destination':(staged/'destination.json').write_text('{}')
    if failure=='segment_symlink':(staged/(file.parent.name+'.wandb')).symlink_to('/etc/passwd')
    if failure=='temporary_symlink':(root/'.tracking-sync.json.tmp').symlink_to('/etc/passwd')
    if failure in {'target_missing','not_finished','state_symlink','lock_symlink','destination_symlink','changed_destination','segment_symlink'}:
        with pytest.raises(ValueError):sync_attempt(cfg,'p','r','a')
        return
    killed=[]
    monkeypatch.setattr(tracking_sync.os,'killpg',lambda *args:killed.append(args))
    class Child:
        pid=123
        def wait(self,**kwargs):
            if failure=='timeout' and kwargs:raise subprocess.TimeoutExpired('SDK',1)
            return 0
    def spawn(*args,**kwargs):
        if failure=='spawn':raise OSError('SECRET')
        return Child()
    monkeypatch.setattr(tracking_sync.subprocess,'Popen',spawn)
    if failure=='temporary_symlink':
        with pytest.raises(ValueError):sync_attempt(cfg,'p','r','a')
    else:
        assert sync_attempt(cfg,'p','r','a')['state']=='SYNC_FAILED'
        assert bool(killed)==(failure=='timeout')


def test_tracking_api_known_attempt_and_credentials_not_exposed(tmp_path):
    cfg,root,_=setup_sync(tmp_path)
    cfg.telemetry.enabled=False
    cfg.action_root=str(tmp_path/'actions')
    with TestClient(create_app(cfg,poll=False,projects=[])) as client:
        client.headers[CLIENT_PROTOCOL_HEADER]='1'
        payload=client.get('/api/tracking').json()
        assert payload['sync_configured'] and not payload['automatic_upload']
        assert 'credential_ref' not in repr(payload) and 'SECRET' not in repr(payload)
        response=client.get('/api/tracking/attempts/p/r/a')
        assert response.status_code==200 and response.json()['state']=='OFFLINE_READY'
        assert client.get('/api/observability/attempts/p/r/a').json()['targets']==[]


def test_tracking_endpoint_configuration_and_sync_cli(tmp_path,monkeypatch,capsys):
    from ml_exp_server import cli,project_config,tracking_sync
    cfg,_,_=setup_sync(tmp_path)
    for field,value in [('api_url','https://user:secret@host'),('entity','../bad')]:
        with pytest.raises(ValueError):TrackingConfig(**{field:value})
    assert TrackingConfig(api_url='https://host/').api_url=='https://host'
    assert TrackingConfig(entity=None).entity is None
    monkeypatch.setattr(project_config,'load_server_config',lambda path:cfg)
    command=['--config','unused.yaml','tracking-sync','p','r','a']
    monkeypatch.setattr(tracking_sync,'sync_attempt',lambda *args:{'state':'CLI_COMPLETED'})
    assert cli.main(command)==0
    monkeypatch.setattr(tracking_sync,'sync_attempt',lambda *args:{'state':'SYNC_FAILED'})
    assert cli.main(command)==1
    monkeypatch.setattr(tracking_sync,'sync_attempt',lambda *args:(_ for _ in ()).throw(ValueError('SECRET')))
    assert cli.main(command)==1
    assert 'SECRET' not in capsys.readouterr().out
