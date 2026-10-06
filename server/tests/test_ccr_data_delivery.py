"""Zero-GPU scheduler scope, durable submission uncertainty and sealed NAS receipts."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_exp_server import data_delivery as module
from ml_exp_server.data_delivery import DataDeliveryService, recover_data_deliveries
from ml_exp_server.storage import atomic_json
from ml_exp_server.application_errors import ApplicationError
from tests.test_sensecore_data_workflow import client, stored, custom_runtime
from tests.test_artifact_store import archive
from tests.test_desktop_data_upload import stage, remote_client
from tests.test_multipart_upload import begin, send

IMAGE='registry.example/data@sha256:'+'b'*64


@pytest.fixture
def delivery(remote_client,stored,monkeypatch):
    config=json.loads(stored[0].read_text())
    config['data_delivery']={'sco_bin':'sco','aec2':'debug','worker_spec':'N6lS.Iu.I10.2c4g',
        'gpus':0,'cpus':2,'memory_gb':4,'copy_timeout_seconds':5,'queue_timeout_seconds':5}
    stored[0].write_text(json.dumps(config))
    data=archive({'tokens.bin':b'training tokens'})
    upload=begin(remote_client,data);root=send(remote_client,data,upload)
    asset=remote_client.post(root+'/complete').json()
    service=DataDeliveryService(remote_client.app.state.runtime)
    value=service.prepare('demo',asset['asset_id'],'cloud')
    image={'project':'demo','asset_id':asset['asset_id'],'files_sha256':value['files_sha256'],
        'data_worker_sha256':value['worker_sha256'],'data_image_id':'data-image.'+'c'*64,'image':IMAGE}
    monkeypatch.setattr(module,'builder_request',lambda *a:image)
    return remote_client,service,value,asset


def receipt(value,status='READY'):
    return {'status':status,'asset_id':value['asset_id'],'archive_sha256':value['archive_sha256'],
        'files_sha256':value['files_sha256'],'image':IMAGE,
        'data_path':value['copy_profile']['data_root']+'/'+value['asset_id']}


def secret(service,value):
    with service.state('demo',value['delivery_id']) as (_,snapshot):return snapshot.value['copy_token']


@pytest.mark.parametrize('result,invalid', [('No jobs found\n',False),('[]',False),('{}',True),('[null]',True),('unknown output',True)])
def test_exact_name_query_handles_real_cli_empty_results(delivery,monkeypatch,result,invalid):
    _,service,value,_=delivery
    def sco(copy,args):
        assert args == ['acp','jobs','list','--workspace-name',copy['workspace'],
                        '--name',value['scheduler_name'],'--all','--format','json']
        return result
    monkeypatch.setattr(service,'sco',sco)
    if invalid:
        with pytest.raises(ValueError):service.find(value)
    else:assert service.find(value) is None


def test_control_revision_creates_new_delivery_without_replaying_history(delivery,monkeypatch):
    _,service,value,asset=delivery
    assert value['control_revision'] == module.CONTROL_REVISION
    monkeypatch.setattr(module,'CONTROL_REVISION','acp-data-copy.next')
    revised=service.prepare('demo',asset['asset_id'],'cloud')
    assert revised['delivery_id'] != value['delivery_id'] and revised['status']=='PREPARED'
    assert service.read('demo',value['delivery_id']) == value


def test_reserved_default_and_explicit_spot_have_distinct_frozen_scope(delivery):
    _,service,value,asset=delivery
    assert value['copy_profile']['quota_type']=='reserved'
    with service.state('demo',value['delivery_id']) as (_,snapshot):
        command=service.create_command({**snapshot.value,'image':IMAGE})
    assert command[command.index('--quota-type')+1]=='reserved'
    service.config['quota_type']='spot'
    revised=service.prepare('demo',asset['asset_id'],'cloud')
    assert revised['delivery_id']!=value['delivery_id']
    with service.state('demo',revised['delivery_id']) as (_,snapshot):
        command=service.create_command({**snapshot.value,'image':IMAGE})
    assert command[command.index('--quota-type')+1]=='spot'
    assert service.read('demo',value['delivery_id'])==value


@pytest.mark.parametrize('text,status,reason', [
    ('400 Bad Request\\nBackend: {\\"reason\\":\\"tjInvalidArgument\\",\\"command\\":\\"secret\\"}',400,'tjInvalidArgument'),
    ('403 Forbidden Authorization: Bearer secret',403,None),
    ('unknown CLI failure contains secret',None,None),
])
def test_control_error_reports_only_provider_codes(delivery,monkeypatch,text,status,reason):
    _,service,value,_=delivery
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1,stdout='',stderr=text))
    with pytest.raises(module.ACPControlError) as error:service.sco(value['copy_profile'],['acp','jobs','create','--command','secret'])
    assert error.value.details=={'operation':'acp jobs create','exit_code':1,'http_status':status,'provider_reason':reason}
    assert 'secret' not in json.dumps(error.value.details)
    pending=service.begin('demo',value['delivery_id'],value['confirmation'])
    result=service.finish(pending)
    assert result['status']=='RECONCILE_REQUIRED' and result['control_error']['operation']=='acp jobs list'


def test_prepare_execute_requires_exact_cpu_scope_and_seals_ready(delivery,monkeypatch):
    client,service,value,asset=delivery
    assert 'copy_token' not in value
    assert service.prepare('demo',asset['asset_id'],'cloud')==value
    with pytest.raises(ApplicationError):service.ready_for('demo',asset['asset_id'],'cloud')
    assert service.read('demo',value['delivery_id'])==value
    pending=service.begin('demo',value['delivery_id'],value['confirmation'])
    with pytest.raises(ApplicationError):service.begin('demo',value['delivery_id'],value['confirmation'])
    calls=[]
    def sco(copy,args):
        calls.append(args)
        if args[:3]==['acp','jobs','list']:return '[]'
        if args[:3]==['aec2','clusters','list-workerspec']:
            return '| N6lS.Iu.I10.2c4g | CPU | 0 | spot | 2 | 4 | ready |'
        assert args[:3]==['acp','jobs','create']
        assert args[args.index('--worker-spec')+1]=='N6lS.Iu.I10.2c4g'
        assert args[args.index('--aec2-name')+1]=='debug'
        assert '--wait' not in args
        # READY can arrive while create's response is still in flight. A late
        # QUEUED update must never overwrite the sealed callback receipt.
        service.callback('demo',value['delivery_id'],secret(service,value),receipt(value,'COPYING'))
        service.callback('demo',value['delivery_id'],secret(service,value),receipt(value))
        return '{}'
    monkeypatch.setattr(service,'sco',sco)
    result=service.finish(pending)
    assert result['status']=='READY' and result['receipt']==receipt(value)
    assert service.begin('demo',value['delivery_id'],value['confirmation']) is None
    assert service.callback('demo',value['delivery_id'],secret(service,value),receipt(value))==result
    assert service.ready_for('demo',asset['asset_id'],'cloud')['image']==IMAGE
    assert len(calls)==3
    for changed in (receipt(value,'COPYING'),{**receipt(value),'files_sha256':'a'*64}):
        with pytest.raises(ValueError):service.callback('demo',value['delivery_id'],secret(service,value),changed)
    assert service.update('demo',value['delivery_id'],status='FAILED')['status']=='READY'
    # The Run records verified NAS provenance and the worker refuses fallback.
    bundle=custom_runtime(client,monkeypatch)
    response=client.post('/api/projects/demo/runs',json={'run_id':'ready-data','runtime_id':bundle['runtime_id'],
        'executor':'cloud','inputs':[{'asset_id':asset['asset_id'],'mount_path':'/inputs/data'}]})
    assert response.status_code==200,response.text
    assert client.get('/api/projects/demo/data-deliveries/'+value['delivery_id']).json()['status']=='READY'
    from ml_exp_server.container_execution import ContainerExecutionService
    bundle_path=service.runtime.config.project_registry_root_path()/'runtime-bundles/demo'/(bundle['runtime_id']+'.json')
    # Historical images remain readable but cannot silently gain cache gating.
    raw=json.loads(bundle_path.read_text());raw['capabilities'].remove('data-cache-required.v1')
    atomic_json(bundle_path,raw)
    assert client.post('/api/projects/demo/runs',json={'run_id':'old-worker','runtime_id':bundle['runtime_id'],
        'executor':'cloud','inputs':[{'asset_id':asset['asset_id'],'mount_path':'/inputs/data'}]}).status_code==409


@pytest.mark.parametrize('case',['confirmation','wrong-project','missing','bad-executor','bad-asset','cpu-config','quota-config','worker-changed','control-changed','reconcile-prepared'])
def test_invalid_preparation_never_submits(delivery,stored,monkeypatch,case):
    _,service,value,asset=delivery
    with pytest.raises((ValueError,ApplicationError)):
        if case=='confirmation':service.begin('demo',value['delivery_id'],'wrong')
        elif case=='wrong-project':service.read('../escape',value['delivery_id'])
        elif case=='missing':service.read('demo','delivery.'+'a'*64)
        elif case=='bad-executor':service.prepare('demo',asset['asset_id'],'gpu')
        elif case=='bad-asset':service.prepare('demo','bad','cloud')
        elif case=='cpu-config':
            service.config['gpus']=1
            service.prepare('demo',asset['asset_id'],'cloud')
        elif case=='quota-config':
            service.config['quota_type']='unknown'
            service.prepare('demo',asset['asset_id'],'cloud')
        elif case=='worker-changed':
            monkeypatch.setattr(module,'data_worker_digest',lambda:'changed')
            service.begin('demo',value['delivery_id'],value['confirmation'])
        elif case=='control-changed':
            monkeypatch.setattr(module,'CONTROL_REVISION','changed')
            service.begin('demo',value['delivery_id'],value['confirmation'])
        else:service.begin('demo',value['delivery_id'],value['confirmation'],reconcile=True)


@pytest.mark.parametrize('case',['missing-job','ambiguous-job','wrong-image','wrong-cpu','cpu-absent','terminal','expired','error-race','found'])
def test_reconciliation_never_replays_unknown_submissions(delivery,monkeypatch,case):
    _,service,value,_=delivery
    pending=service.begin('demo',value['delivery_id'],value['confirmation'])
    created=[];clock=iter([0,0,100])
    def sco(copy,args):
        if args[:3]==['acp','jobs','list']:
            job={'name':value['scheduler_name'],'state':'SUCCEEDED' if case=='terminal' else 'RUNNING'}
            if case=='ambiguous-job':return json.dumps([job,job])
            if case in {'terminal','expired','found'}:return json.dumps([job])
            return '[]'
        if args[:3]==['aec2','clusters','list-workerspec']:
            return '' if case=='cpu-absent' else '| N6lS.Iu.I10.2c4g | CPU | 1 | spot | 2 | 4 | ready |' if case=='wrong-cpu' else '| N6lS.Iu.I10.2c4g | CPU | 0 | spot | 2 | 4 | ready |'
        if args[:3]==['acp','jobs','stop']:
            created.append('stop');return '{}'
        created.append('create')
        if case=='error-race':service.callback('demo',value['delivery_id'],secret(service,value),{'status':'FAILED','error_class':'OSError'})
        raise ValueError('uncertain ACP create')
    monkeypatch.setattr(service,'sco',sco)
    monkeypatch.setattr(module,'time',SimpleNamespace(sleep=lambda x:None,monotonic=lambda:next(clock)))
    if case=='wrong-image':monkeypatch.setattr(module,'builder_request',lambda *a:{'image':'bad'})
    if case=='found':
        def read(project,id):return {'status':'READY'}
        monkeypatch.setattr(service,'read',read)
    result=service.finish(pending)
    expected='FAILED' if case in {'expired','error-race'} else 'READY' if case=='found' else 'RECONCILE_REQUIRED'
    assert result['status']==expected
    if case=='expired':assert created==['stop']
    if result['status']=='RECONCILE_REQUIRED':
        pending=service.begin('demo',value['delivery_id'],value['confirmation'],reconcile=True)
        before=created.count('create')
        service.finish(pending)
        assert created.count('create')==before


def test_startup_recovers_only_owned_active_deliveries(delivery):
    _,service,value,_=delivery
    service.begin('demo',value['delivery_id'],value['confirmation'])
    (service.root/'demo/invalid.json').write_text('{}')
    assert recover_data_deliveries(service.runtime)==1
    assert recover_data_deliveries(service.runtime)==0
    assert service.read('demo',value['delivery_id'])['status']=='RECONCILE_REQUIRED'


@pytest.mark.parametrize('case',['missing','wrong','expired','phase','identity','invalid-status'])
def test_data_callback_capability_and_identity_fail_closed(delivery,case):
    _,service,value,_=delivery
    token=secret(service,value);received=receipt(value)
    if case=='missing':project='other'
    else:project='demo'
    if case=='wrong':token='wrong'
    if case=='expired':
        with service.state('demo',value['delivery_id']) as (store,snapshot):
            store.commit({**snapshot.value,'created_at':'2000-01-01T00:00:00Z'},expected_revision=snapshot.revision,event={'event':'test-expire'})
    if case in {'identity','invalid-status'}:
        service.update('demo',value['delivery_id'],status='QUEUED',image=IMAGE)
        received['asset_id']='wrong' if case=='identity' else received['asset_id']
        if case=='invalid-status':received['status']='RUNNING'
    with pytest.raises(ValueError):service.callback(project,value['delivery_id'],token,received)


def test_data_delivery_routes_authenticate_before_reading_receipt(delivery,monkeypatch):
    client,service,value,asset=delivery
    endpoint='/api/projects/demo/data-deliveries/'+value['delivery_id']
    transfer='/api/data-copy-transfers/demo/'+value['delivery_id']
    token=secret(service,value);headers={'Authorization':'Bearer '+token}
    assert client.put(transfer,json=receipt(value),headers={'Authorization':'Bearer wrong'}).status_code==401
    assert client.put(transfer,json=receipt(value)).status_code==401
    assert client.put(transfer,content='null',headers=headers).status_code==422
    assert client.put(transfer,content=b'x'*16385,headers=headers).status_code==413
    assert client.post('/api/projects/demo/assets/'+asset['asset_id']+'/deliveries/prepare',json={'executor':'cloud'}).json()==value
    queued=[];client.app.state.submit_job=lambda function,pending:queued.append(pending)
    response=client.post(endpoint+'/execute',json={'confirmation':value['confirmation']})
    assert response.status_code==202 and queued
    service.update('demo',value['delivery_id'],status='QUEUED',image=IMAGE)
    assert client.put(transfer,json=receipt(value),headers=headers).status_code==200
    assert client.post(endpoint+'/execute',json={'confirmation':value['confirmation']}).json()['status']=='READY'
    assert client.put(transfer,json=receipt(value,'COPYING'),headers=headers).status_code==409


def test_data_delivery_enqueue_failure_is_reconcilable(delivery):
    client,service,value,_=delivery
    client.app.state.submit_job=lambda *a:(_ for _ in ()).throw(RuntimeError('unavailable'))
    endpoint='/api/projects/demo/data-deliveries/'+value['delivery_id']
    assert client.post(endpoint+'/execute',json={'confirmation':value['confirmation']}).status_code==503
    assert client.post(endpoint+'/reconcile',json={'confirmation':value['confirmation']}).status_code==503


def test_disabled_delivery_and_sco_failures_fail_closed(client,stored,monkeypatch):
    runtime=client.app.state.runtime
    runtime.config.container_execution.artifact_store_file=None
    with pytest.raises(ApplicationError):DataDeliveryService(runtime)
    runtime.config.container_execution.artifact_store_file=str(stored[0])
    with pytest.raises(ApplicationError):DataDeliveryService(runtime)
    config=json.loads(stored[0].read_text());config['data_delivery']={'sco_bin':'sco'};stored[0].write_text(json.dumps(config))
    service=DataDeliveryService(runtime)
    def execute(command,**kw):
        assert all(key.lower() not in {'http_proxy','https_proxy','all_proxy'} for key in kw['env'])
        return SimpleNamespace(returncode=1,stdout='private',stderr='private')
    monkeypatch.setattr(module.subprocess,'run',execute)
    with pytest.raises(ValueError):service.sco({'sco_bin':'sco'},['acp','jobs','list'])
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0,stdout='[]'))
    assert service.sco({'sco_bin':'sco'},['acp','jobs','list'])=='[]'


def test_cpu_spec_skips_unrelated_rows_and_callback_rejects_http(delivery,monkeypatch):
    _,service,value,_=delivery
    monkeypatch.setattr(service,'sco',lambda *a:'header\n| other.2c4g | CPU | 0 | spot | 2 | 4 | ready |\n| N6lS.Iu.I10.2c4g | CPU | 0 | spot | 2 | 4 | ready |')
    service.verify_cpu(value['copy_profile'])
    service.assets.objects.config['public_transfer_base']='http://example'
    with pytest.raises(ValueError):service.create_command(value)


def test_copy_callback_preserves_reverse_proxy_prefix(delivery):
    _,service,value,_=delivery
    service.assets.objects.config['public_transfer_base']='https://example/ml-expd/api/artifact-transfers'
    with service.state('demo',value['delivery_id']) as (_,snapshot):raw=snapshot.value
    raw['image']=IMAGE
    command=service.create_command(raw)
    assert 'https://example/ml-expd/api/data-copy-transfers/demo/' in command[-1]
    service.assets.objects.config['public_transfer_base']='https://example/other'
    with pytest.raises(ValueError):service.create_command(raw)
