"""Real stdlib worker with SSH replaced by local CPU transport, no GPU jobs."""
import io
import json
from pathlib import Path
import shlex
import subprocess

import pytest

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.container_execution import ContainerExecutionService
from ml_exp_server.experiment_preparation import ExperimentPreparationRequest
from ml_exp_server.wyd_data import WydDataStager
from tests.test_desktop_data_upload import remote_client, stage, stored, client
from tests.test_multipart_upload import begin, send
from tests.test_artifact_store import archive
from tests.test_experiment_preparation import service, spec


@pytest.fixture
def backend(remote_client, tmp_path, monkeypatch):
    data=archive({'nested/tokens.bin':b'training data'*1024})
    upload=begin(remote_client,data);root=send(remote_client,data,upload)
    asset=remote_client.post(root+'/complete').json()['asset_id']
    profile=ContainerExecutionService(remote_client.app.state.runtime).profiles()['gpu']
    profile['storage_root']=str(tmp_path/'backend')
    original=subprocess.Popen
    commands=[]
    def local(command,**kwargs):
        commands.append(command)
        assert command[0]=='ssh' and 'BatchMode=yes' in command
        return original(shlex.split(command[-1]),**kwargs)
    monkeypatch.setattr('ml_exp_server.wyd_data.subprocess.Popen',local)
    return remote_client,profile,asset,commands


def test_real_backend_stream_verified_reused_and_corruption_rejected(backend):
    client,profile,asset,commands=backend
    service=WydDataStager(client.app.state.runtime)
    assert service.check('demo',asset,profile)=={'status':'NOT_FOUND'}
    receipt=service.publish('demo',asset,profile)
    assert receipt['status']=='READY' and receipt['transport']=='ssh_stream_shared_storage'
    assert service.check('demo',asset,profile)=={'status':'READY'}
    assert 'private-capability' not in json.dumps(commands) and not list(Path(profile['storage_root']).rglob('.input-*'))
    file=Path(profile['storage_root'])/'demo/data-assets'/asset/'nested/tokens.bin'
    file.chmod(0o644);file.write_bytes(b'corruption')
    with pytest.raises(ApplicationError):service.check('demo',asset,profile)
    assert file.read_bytes()==b'corruption'


def test_server_preparation_seals_cache_without_gpu_or_compute_https(backend, service, monkeypatch):
    client,profile,asset,_=backend
    config=Path(client.app.state.runtime.config.container_execution.profiles_file)
    import yaml
    doc=yaml.safe_load(config.read_text());doc['executors']['gpu']=profile;config.write_text(yaml.safe_dump(doc))
    body=spec(client,run={'run_id':'cached-wyd','executor':'gpu','inputs':[{'asset_id':asset,'mount_path':'/inputs/data'}]})
    accepted,pending=service.accept('demo',ExperimentPreparationRequest.model_validate(body))
    result=service.finish(pending)
    assert result['status']=='READY',result
    assert result['data_receipts'][asset]['status']=='READY'
    request=yaml.safe_load((Path(client.app.state.runtime.project('demo').base_dir)/'experiments/campaigns/run-cached-wyd.yaml').read_text())['runs'][0]
    assert request['inputs'][0]['require_cached']
    # Another Run verifies the same cache without transferring it again.
    monkeypatch.setattr(WydDataStager,'publish',lambda *a:pytest.fail('must reuse'))
    body['run']['run_id']='reused-wyd'
    _,pending=service.accept('demo',ExperimentPreparationRequest.model_validate(body))
    assert service.finish(pending)['status']=='READY'


def test_uncertain_wyd_transfer_never_replayed_then_verified_continue(backend, service, monkeypatch):
    client,profile,asset,_=backend
    import yaml
    p=Path(client.app.state.runtime.config.container_execution.profiles_file)
    v=yaml.safe_load(p.read_text());v['executors']['gpu']=profile;p.write_text(yaml.safe_dump(v))
    body=spec(client,run={'run_id':'interrupted-wyd','executor':'gpu','inputs':[{'asset_id':asset,'mount_path':'/inputs/data'}]})
    _,pending=service.accept('demo',ExperimentPreparationRequest.model_validate(body))
    service.update(*pending,delivery_requested=['wyd:'+asset])
    stopped=service.finish(pending)
    assert stopped['error_code']=='WYD_DATA_NOT_READY'
    WydDataStager(client.app.state.runtime).publish('demo',asset,profile)
    _,pending=service.continue_preparation(*pending)
    monkeypatch.setattr(WydDataStager,'publish',lambda *a:pytest.fail('uncertain transfer cannot replay'))
    assert service.finish(pending)['status']=='READY'


def test_metadata_bound_is_checked_before_ssh(backend):
    client,profile,_,_=backend
    with pytest.raises(ApplicationError) as error:
        with WydDataStager(client.app.state.runtime).process(profile,{'data':'x'*(8*1024*1024)}):pass
    assert error.value.code=='WYD_DATA_METADATA_TOO_LARGE'


@pytest.mark.parametrize('mode',['timeout','nonzero','oversized','invalid','failed','brokenpipe','live_cleanup'])
def test_transport_errors_are_safe_and_process_is_cleaned(backend,monkeypatch,mode):
    client,profile,asset,_=backend
    class Process:
        stdin=io.BytesIO()
        stdout=io.BytesIO(b'bad json' if mode=='invalid' else b'x'*8193 if mode=='oversized' else json.dumps({'status':'FAILED' if mode=='failed' else 'READY'}).encode())
        kills=0
        def kill(self):self.kills+=1
        def poll(self):return None if mode=='live_cleanup' and not self.kills else 0
        def wait(self,**kwargs):return 1 if mode=='nonzero' else 0
    process=Process()
    monkeypatch.setattr('ml_exp_server.wyd_data.subprocess.Popen',lambda *a,**kw:process)
    class Timer:
        def __init__(self,seconds,fn):self.fn=fn
        def start(self):
            if mode=='timeout':self.fn()
        def cancel(self):pass
    monkeypatch.setattr('ml_exp_server.wyd_data.threading.Timer',Timer)
    if mode=='live_cleanup':
        with WydDataStager(client.app.state.runtime).process(profile,{'x':1}):pass
    else:
        with pytest.raises(ApplicationError):
            with WydDataStager(client.app.state.runtime).process(profile,{'x':1}) as proc:
                if mode=='brokenpipe':raise BrokenPipeError('Bearer private')
    assert process.stdin.closed and process.stdout.closed


def test_archive_closed_on_failure_and_ready_receipt_required(backend,monkeypatch):
    client,profile,asset,_=backend
    service=WydDataStager(client.app.state.runtime)
    from contextlib import contextmanager
    class Stream:
        closed=False
        def iter_chunks(self,**kwargs):yield b'archive'
        def close(self):self.closed=True
    body=Stream()
    monkeypatch.setattr(service.assets,'archive',lambda *a:({},body))
    @contextmanager
    def process(*args):
        from types import SimpleNamespace
        p=SimpleNamespace(stdin=io.BytesIO(),result={'status':'NOT_FOUND'})
        yield p
    monkeypatch.setattr(service,'process',process)
    with pytest.raises(ApplicationError) as error:service.publish('demo',asset,profile)
    assert body.closed and error.value.code=='WYD_DATA_NOT_READY'
