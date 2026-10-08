"""Zero-GPU scheduler scope, durable submission uncertainty and sealed NAS receipts."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_exp_server.data import data_delivery as module
from ml_exp_server.data.data_delivery import DataDeliveryService, recover_data_deliveries
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
    config['data_delivery']={'aec2':'cpu-pool','worker_spec':'N6lS.Iu.I10.2c4g',
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


def test_exact_name_rest_query_is_scoped(delivery, monkeypatch):
    _, service, value, _ = delivery
    calls = []
    rest = SimpleNamespace(find=lambda copy, name: calls.append((copy["workspace"], name)) or [])
    service._rest = rest
    assert service.find(value) is None
    assert calls == [(value["copy_profile"]["workspace"], value["scheduler_name"])]


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
        command=service.create_document({**snapshot.value,'image':IMAGE})
    assert command['scheduling']['quota_type']=='RESERVED'
    service.config['quota_type']='spot'
    revised=service.prepare('demo',asset['asset_id'],'cloud')
    assert revised['delivery_id']!=value['delivery_id']
    with service.state('demo',revised['delivery_id']) as (_,snapshot):
        command=service.create_document({**snapshot.value,'image':IMAGE})
    assert command['scheduling']['quota_type']=='SPOT'
    assert service.read('demo',value['delivery_id'])==value


def test_control_error_persists_only_safe_rest_codes(delivery):
    _, service, value, _ = delivery
    error = module.RESTError("GET jobs", status=403, reason="Denied")
    service._rest = SimpleNamespace(find=lambda *a: (_ for _ in ()).throw(error))
    result = service.finish(service.begin("demo", value["delivery_id"], value["confirmation"]))
    assert result["status"] == "RECONCILE_REQUIRED"
    assert result["control_error"] == error.details


def test_prepare_execute_requires_exact_cpu_scope_and_seals_ready(delivery,monkeypatch):
    client,service,value,asset=delivery
    assert 'copy_token' not in value
    assert service.prepare('demo',asset['asset_id'],'cloud')==value
    with pytest.raises(ApplicationError):service.ready_for('demo',asset['asset_id'],'cloud')
    assert service.read('demo',value['delivery_id'])==value
    pending=service.begin('demo',value['delivery_id'],value['confirmation'])
    with pytest.raises(ApplicationError):service.begin('demo',value['delivery_id'],value['confirmation'])
    calls=[]
    def create(copy, document, **kwargs):
        calls.append("create")
        assert document["resource_pool"]["name"] == "cpu-pool"
        spec = document["roles"][0]["resource_spec"][0]
        assert spec["name"] == "N6lS.Iu.I10.2c4g"
        assert spec["requests"] == {"cpu": "2", "memory": "3Gi"}
        assert spec["limits"] == {"cpu": "2", "memory": "4Gi"}
        service.callback('demo',value['delivery_id'],secret(service,value),receipt(value,'COPYING'))
        service.callback('demo',value['delivery_id'],secret(service,value),receipt(value))
        return {}
    service._rest = SimpleNamespace(
        find=lambda *a: calls.append("find") or [],
        specs=lambda *a: calls.append("specs") or [{"name":"N6lS.Iu.I10.2c4g", "device":{"number":0},
            "cpu":{"vcpu_allocatable":2}, "memory":{"allocatable":4}}], create=create)
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
    def find(copy, name):
        job={'name':value['scheduler_name'],'state':'SUCCEEDED' if case=='terminal' else 'RUNNING'}
        if case=='ambiguous-job': return [job,job]
        return [job] if case in {'terminal','expired','found'} else []
    def create(*args, **kwargs):
        created.append('create')
        if case=='error-race':service.callback('demo',value['delivery_id'],secret(service,value),{'status':'FAILED','error_class':'OSError'})
        raise ValueError('uncertain ACP create')
    service._rest = SimpleNamespace(find=find, create=create,
        stop=lambda *a: created.append('stop'),
        specs=lambda *a: [] if case=='cpu-absent' else [{"name":"N6lS.Iu.I10.2c4g",
            "device":{"number":1 if case=='wrong-cpu' else 0},
            "cpu":{"vcpu_allocatable":2}, "memory":{"allocatable":4}}])
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


def test_disabled_delivery_and_rest_configuration_fail_closed(client,stored,monkeypatch):
    runtime=client.app.state.runtime
    runtime.config.container_execution.artifact_store_file=None
    with pytest.raises(ApplicationError):DataDeliveryService(runtime)
    runtime.config.container_execution.artifact_store_file=str(stored[0])
    with pytest.raises(ApplicationError):DataDeliveryService(runtime)
    config=json.loads(stored[0].read_text());config['data_delivery']={'configured':True};stored[0].write_text(json.dumps(config))
    service=DataDeliveryService(runtime)
    monkeypatch.setattr(module.SenseCoreREST, "from_environment", lambda: SimpleNamespace())
    assert service.rest is service.rest


def test_cpu_spec_skips_unrelated_rows_and_callback_rejects_http(delivery,monkeypatch):
    _,service,value,_=delivery
    service._rest = SimpleNamespace(specs=lambda *a: [
        {"name": "other.2c4g"}, {"name":"N6lS.Iu.I10.2c4g", "device":{"number":0},
         "cpu":{"vcpu_allocatable":2}, "memory":{"allocatable":4}}])
    service.verify_cpu(value['copy_profile'])
    service.assets.objects.config['public_transfer_base']='http://example'
    with pytest.raises(ValueError):service.create_document(value)


def test_copy_callback_preserves_reverse_proxy_prefix(delivery):
    _,service,value,_=delivery
    service.assets.objects.config['public_transfer_base']='https://example/ml-expd/api/artifact-transfers'
    with service.state('demo',value['delivery_id']) as (_,snapshot):raw=snapshot.value
    raw['image']=IMAGE
    command=service.create_document(raw)
    assert 'https://example/ml-expd/api/data-copy-transfers/demo/' in command['roles'][0]['startup_script']
    service.assets.objects.config['public_transfer_base']='https://example/other'
    with pytest.raises(ValueError):service.create_document(raw)


def test_debug_cpu_copy_remains_allowed_and_historical_delivery_remains_readable(delivery):
    _, service, value, asset = delivery
    service.config['aec2'] = 'DEBUG-cluster'
    revised = service.prepare('demo', asset['asset_id'], 'cloud')
    assert revised['copy_profile']['aec2'] == 'DEBUG-cluster'
    with service.state('demo', revised['delivery_id']) as (_, snapshot):
        document = service.create_document({**snapshot.value, 'image': IMAGE})
    assert document['resource_pool']['name'] == 'DEBUG-cluster'
    assert document['roles'][0]['resource_spec'][0]['requests']['cpu'] == '2'
    assert service.read('demo', value['delivery_id'])['status'] == 'PREPARED'
