"""Private archival uses sealed receipts, bounded streams and verified cloud bytes."""
from concurrent.futures import Future
from datetime import timedelta
import hashlib
import io
import json
from types import SimpleNamespace

import alibabacloud_oss_v2 as oss
import pytest
from fastapi.testclient import TestClient

from ml_exp_server.results import oss_archive as a
from ml_exp_server.results.artifact_lifecycle import ArtifactLifecycle
from ml_exp_server.results.artifact_store import ArtifactStore
from ml_exp_server.storage_service import StorageService
from ml_exp_server.storage import atomic_json
from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.execution_queue import QueueUnavailable, job_lane
from tests.test_artifact_store import storage, archive

IDENTITY = ('demo', 'run-a', 'attempt-001')
SETTINGS = {'bucket': 'ai-oss-h200-1020', 'region': 'us-southeast-1',
            'user': 'researcher', 'credentials_file': '/etc/private/archive.json'}


def service_error(code):
    return oss.exceptions.OperationError(name='GetObject', error=oss.exceptions.ServiceError(
        status_code=404, code=code, request_id='test', message='private provider data',
        ec='', timestamp='', request_target=''))


@pytest.fixture
def published(storage, monkeypatch):
    store, root, token, config, garage = storage
    store.config['oss_archive'] = dict(SETTINGS)
    data = archive({'weights.bin': b'owned weights', 'training.txt': b'step=8'})
    receipt = store.receive(*IDENTITY, token, io.BytesIO(data), len(data))
    original = store.client()
    original.get_object = lambda *, Bucket, Key: {'Body': io.BytesIO(garage[(Bucket, Key)])}
    original.delete_object = lambda *, Bucket, Key: garage.pop((Bucket, Key), None)
    monkeypatch.setattr(ArtifactStore, 'client', lambda self, **kwargs: original)
    cloud = {}
    class Destination:
        def get_object(self, request):
            if request.key not in cloud:
                raise service_error('NoSuchKey')
            # Use the real SDK body adapter: read() is unbounded/no size argument.
            from alibabacloud_oss_v2.io_utils import StreamBodyReader
            stream=io.BytesIO(cloud[request.key])
            def chunks(*,block_size):
                while data:=stream.read(block_size):yield data
            return SimpleNamespace(body=StreamBodyReader(SimpleNamespace(iter_bytes=chunks,close=stream.close)))
        def presign(self, request, *, expires):
            assert request.bucket == SETTINGS['bucket'] and expires == timedelta(minutes=5)
            return SimpleNamespace(url='https://private.example/object?signed=secret')
    destination = Destination()
    monkeypatch.setattr(a, 'client', lambda settings: destination)
    class Uploader:
        def __init__(self, target, **options):
            assert target is destination
            assert options == dict(part_size=a.PART_BYTES, parallel_num=1, leave_parts_on_error=False)
        def upload_from(self, request, reader):
            assert request.forbid_overwrite is True
            content = b''
            while chunk := reader.read(a.PART_BYTES):
                content += chunk
            assert request.key not in cloud
            cloud[request.key] = content
    monkeypatch.setattr(oss, 'Uploader', Uploader)
    return a.OSSArchiveService(store), receipt, cloud, data, garage


def state(service):
    with service.store.record(*IDENTITY) as (_, value):
        return value


def edit(service, function):
    with service.store.record(*IDENTITY) as (path, value):
        function(value)
        atomic_json(path, value)


def proof(receipt):
    return {'archive_sha256': receipt['sha256'], 'archive_bytes': receipt['bytes'],
            'files': {'outputs/' + item['path']: {k: item[k] for k in ('bytes', 'sha256')}
                      for item in receipt['files']}}


def test_new_receipt_streams_private_copy_verifies_bytes_and_ack_releases_only_garage(published):
    service, receipt, cloud, data, garage = published
    first = state(service)
    assert first['oss_archive']['status'] == 'PENDING' and a.pending(first)
    assert first['oss_archive']['object_key'].startswith('users/researcher/results/ml-expd/demo/run-a/attempt-001/')
    lifecycle = ArtifactLifecycle(service.store)
    lifecycle.acknowledge(*IDENTITY, proof(receipt))
    assert lifecycle.run_due() == 0 and garage
    service.finish(IDENTITY)
    result = state(service)
    assert result['oss_archive']['status'] == 'VERIFIED' and not a.pending(result)
    assert result['oss_archive']['bytes_copied'] == len(data)
    assert cloud == {result['oss_archive']['object_key']: data}
    ticket = service.ticket(*IDENTITY)
    assert ticket['sha256'] == receipt['sha256'] and ticket['files'] == receipt['files']
    assert ticket['transport'] == 'oss-presigned-get' and not ticket['acknowledgement_required']
    assert lifecycle.run_due() == 1 and not garage and cloud
    assert service.ticket(*IDENTITY)['bytes'] == len(data)
    service.finish(IDENTITY)
    assert state(service)['oss_archive']['attempts'] == 1


@pytest.mark.parametrize('settings', [None, {}, {'bucket':'UPPER'}, {'region':'../x'}, {'user':'../x'},
                                      {'user':None}, {'credentials_file':'relative'}, {'credentials_file':None},
                                      {'bucket':None}, {'region':None}])
def test_invalid_configuration_fails_before_publishing_new_result(storage, settings):
    store, _, token, _, garage = storage
    if settings is None:
        assert a.target(store.config, IDENTITY, {}) is None
        return
    store.config['oss_archive'] = settings if not settings else {**SETTINGS, **settings}
    data = archive({'weights.bin': b'x'})
    with pytest.raises(ValueError, match='archive configuration'):
        store.receive(*IDENTITY, token, io.BytesIO(data), len(data))
    assert not garage


def test_old_sealed_receipt_is_not_retroactively_enqueued(storage):
    store, _, token, _, _ = storage
    data = archive({'weights.bin': b'x'})
    receipt = store.receive(*IDENTITY, token, io.BytesIO(data), len(data))
    store.config['oss_archive'] = dict(SETTINGS)
    assert store.receive(*IDENTITY, token, io.BytesIO(data), len(data)) == receipt
    assert 'oss_archive' not in state(a.OSSArchiveService(store))


def test_exact_existing_cloud_copy_reconciles_after_lost_publish_response(published, monkeypatch):
    service, _, cloud, data, _ = published
    cloud[state(service)['oss_archive']['object_key']] = data
    monkeypatch.setattr(service, 'copy', lambda *args: pytest.fail('existing publication was replayed'))
    service.finish(IDENTITY)
    assert state(service)['oss_archive']['status'] == 'VERIFIED'


@pytest.mark.parametrize('failure', ['conflict', 'oversize', 'permission', 'publish_missing', 'source_key', 'source_bytes'])
def test_archive_failure_keeps_primary_receipt_and_sanitizes_diagnostics(published, monkeypatch, failure):
    service, receipt, cloud, data, garage = published
    key = state(service)['oss_archive']['object_key']
    if failure == 'conflict': cloud[key] = b'corrupt'
    elif failure == 'oversize': cloud[key] = data + b'x'
    elif failure == 'permission': monkeypatch.setattr(a, 'verify', lambda *args: (_ for _ in ()).throw(service_error('AccessDenied')))
    elif failure == 'publish_missing': monkeypatch.setattr(service, 'copy', lambda *args: None)
    elif failure == 'source_key': edit(service, lambda v: v['receipt'].update(object_key='wrong/source'))
    else: garage[(service.store.config['bucket'], receipt['object_key'])] = b'corrupt'
    service.finish(IDENTITY)
    result = state(service)
    assert result['oss_archive']['status'] == 'RETRY_REQUIRED'
    assert 'private provider data' not in json.dumps(result)
    assert result['oss_archive']['retry_after'] > a.time.time()
    assert result['receipt']['sha256'] == receipt['sha256'] and garage


@pytest.mark.parametrize('cloud_exists', [False, True])
def test_expired_primary_source_is_never_claimed_archived_without_cloud_bytes(published, cloud_exists):
    service, _, cloud, data, _ = published
    edit(service, lambda v: v.update(retention={'status':'EXPIRED'}))
    if cloud_exists: cloud[state(service)['oss_archive']['object_key']] = data
    service.finish(IDENTITY)
    result = state(service)['oss_archive']
    assert result['status'] == ('VERIFIED' if cloud_exists else 'SOURCE_EXPIRED')


def test_bound_progress_does_not_buffer_entire_archive_or_accept_a_bad_digest():
    content = b'x' * (a.PART_BYTES + 3)
    progress = []
    reader = a.DigestReader(io.BytesIO(content), progress.append, len(content), hashlib.sha256(content).hexdigest())
    assert reader.read(0) == b'' and not progress
    assert len(reader.read()) == a.PART_BYTES
    assert reader.read(99999999) == b'xxx' and reader.read() == b''
    for value, size, digest in [(b'x', 0, 'wrong'), (b'x', 1, 'wrong'), (b'x', 2, 'wrong')]:
        reader = a.DigestReader(io.BytesIO(value), lambda _: None, size, digest)
        with pytest.raises(ValueError):
            while reader.read(): pass


def test_large_copy_records_transfer_progress(published):
    service, receipt, _, _, garage = published
    data = b'x' * (a.PART_BYTES + 1)
    digest = hashlib.sha256(data).hexdigest()
    new_receipt = {**receipt, 'sha256':digest, 'bytes':len(data), 'object_key':'/'.join([*IDENTITY, digest+'.tar'])}
    garage[(service.store.config['bucket'],new_receipt['object_key'])] = data
    edit(service, lambda v: v.update(receipt=new_receipt, oss_archive=a.target(service.store.config,IDENTITY,new_receipt)))
    service.finish(IDENTITY)
    assert state(service)['oss_archive']['bytes_copied'] == len(data)


def test_automatic_admits_only_one_copy_and_respects_retry_backpressure(published, monkeypatch):
    service, _, _, _, _ = published
    calls=[]
    def submit(fn, identity):
        calls.append(identity)
        f=Future()
        return f
    service.automatic(submit); service.automatic(submit)
    assert calls == [IDENTITY]
    service.future.set_result(None)
    edit(service, lambda v: v['oss_archive'].update(retry_after=a.time.time()+60))
    service.automatic(submit)
    assert calls == [IDENTITY]
    edit(service, lambda v: v['oss_archive'].update(retry_after=0,status='SOURCE_EXPIRED'))
    service.automatic(submit)
    assert calls == [IDENTITY]
    edit(service, lambda v: v['oss_archive'].update(status='PENDING'))
    def busy(*args): raise QueueUnavailable('collection')
    service.automatic(busy)
    assert job_lane(service.finish)=='collection'
    service.finish(IDENTITY)
    service.automatic(submit)
    assert calls == [IDENTITY]


def test_missing_disabled_and_changed_identity_are_refused(published):
    service, receipt, _, _, _ = published
    with pytest.raises(ApplicationError) as error: service.ticket(*IDENTITY)
    assert error.value.code=='OSS_ARCHIVE_UNAVAILABLE'
    service.finish(IDENTITY)
    service.store.config['oss_archive']['user']='other'
    with pytest.raises(ValueError,match='identity'):service.ticket(*IDENTITY)
    edit(service,lambda v:v['oss_archive'].update(status='PENDING'))
    service.finish(IDENTITY)
    assert state(service)['oss_archive']['status']=='RETRY_REQUIRED'
    with pytest.raises(ValueError,match='advanced'): service.update(IDENTITY, {'sha256':'wrong'})
    edit(service,lambda v:v.pop('oss_archive'))
    service.finish(IDENTITY)
    service.automatic(lambda *args:pytest.fail('unconfigured history must not archive'))
    assert not a.pending(state(service))


def test_archive_lock_blocks_duplicate_worker(published):
    import fcntl
    service, _, _, _, _ = published
    with service.store.root.joinpath(*IDENTITY[:2],IDENTITY[2]+'.oss.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        service.finish(IDENTITY)
    assert state(service)['oss_archive']['attempts']==0


def test_deleted_run_guard_and_other_application_errors(published, monkeypatch):
    service, _, _, _, _ = published
    def deleted(*args): raise ApplicationError('deleted',code='RUN_DELETED')
    monkeypatch.setattr(service.store,'record',deleted)
    service.finish(IDENTITY)
    def other(*args): raise ApplicationError('other',code='OTHER')
    monkeypatch.setattr(service.store,'record',other)
    with pytest.raises(ApplicationError):service.finish(IDENTITY)


def test_personal_credential_scope_tls_signer_and_no_secret_in_metadata(tmp_path, monkeypatch):
    path=tmp_path/'creds';path.write_text(json.dumps(dict(workspace_user='researcher',region=SETTINGS['region'],
                                                       access_key_id='ak',access_key_secret='sk')))
    captured=[]
    monkeypatch.setattr(oss,'Client',lambda config:captured.append(config) or object())
    settings={**SETTINGS,'credentials_file':str(path)}
    assert a.client(settings)
    assert captured[0].insecure_skip_verify is False and captured[0].signature_version=='v1'
    settings['user']='other'
    with pytest.raises(ValueError,match='user/region'):a.client(settings)


def test_sdk_permission_denial_is_propagated_and_reader_is_nonseekable():
    class Denied:
        def get_object(self, request): raise service_error('AccessDenied')
    with pytest.raises(oss.exceptions.OperationError):
        a.verify(Denied(),{'bucket':'bucket','object_key':'key'})
    assert a.DigestReader(io.BytesIO(),lambda _:None,0,hashlib.sha256(b'').hexdigest()).readable()


def test_post_upload_source_verification_defends_against_a_broken_uploader(published, monkeypatch):
    service, _, _, _, _ = published
    class Broken:
        def __init__(self,*args,**kw): pass
        def upload_from(self,*args): pass
    monkeypatch.setattr(oss,'Uploader',Broken)
    service.finish(IDENTITY)
    assert state(service)['oss_archive']['status']=='RETRY_REQUIRED'


def test_pre_read_record_failure_does_not_fabricate_an_archive_failure(published, monkeypatch):
    service, _, _, _, _ = published
    def failed(*args):raise OSError('record unavailable')
    monkeypatch.setattr(service.store,'record',failed)
    service.finish(IDENTITY)


def test_repair_freezes_new_oss_identity_and_preserves_previous_archive_metadata(published):
    service, receipt, _, _, _ = published
    service.finish(IDENTITY)
    old=state(service)['oss_archive']
    content=archive({'weights.bin':b'owned weights','training.txt':b'step=8','additional.txt':b'new'})
    token=state(service)['token']
    fixed=service.store.receive(*IDENTITY,token,io.BytesIO(content),len(content),repair_sha256=receipt['sha256'])
    result=state(service)
    assert result['oss_archive_history']==[old]
    assert result['oss_archive']['sha256']==fixed['sha256'] and fixed['sha256']!=old['sha256']


def test_storage_maintenance_uses_existing_bounded_collection_lane(published):
    service, _, _, _, _ = published
    facade=StorageService(service.store)
    calls=[]
    def submit(fn, identity):
        calls.append(fn.__self__)
        future=Future();future.set_result(None)
        return future
    facade.maintain(submit=submit)
    assert calls==[facade.archives]
    assert facade.archives is facade.archives


def test_authenticated_oss_ticket_route_requires_verified_cloud_copy(published, tmp_path):
    from ml_exp_server.api.app import create_app
    from ml_exp_server.schemas import ServerConfig
    service, _, _, _, _ = published
    config=service.store.config_file
    config.write_text(json.dumps(service.store.config))
    bearer=tmp_path/'bearer';bearer.write_text('a'*40);bearer.chmod(0o600)
    settings=ServerConfig(index_db=str(tmp_path/'index.sqlite'),project_registry_root=str(service.store.root.parent),
        collector_enabled=False,telemetry={'enabled':False},http_auth={'bearer_token_file':str(bearer)},
        container_execution={'artifact_store_file':str(config)})
    endpoint='/api/runs/demo/run-a/attempts/attempt-001/oss-archive/download'
    headers={'Authorization':'Bearer '+'a'*40,'X-ML-Expd-Client-Protocol':'2'}
    with TestClient(create_app(settings,poll=False)) as client:
        assert client.get(endpoint).status_code==401
        assert client.get(endpoint,headers=headers).status_code==404
        service.finish(IDENTITY)
        response=client.get(endpoint,headers=headers)
        assert response.status_code==200 and response.json()['transport']=='oss-presigned-get'


@pytest.mark.parametrize('code',['AccessDenied','secret://not-a-code'])
def test_error_details_report_only_status_and_safe_provider_code(code):
    error=service_error(code)
    details=a.failure(error)
    assert details=={'error_class':'OperationError','provider_code':code if code=='AccessDenied' else 'OSS_REQUEST_FAILED','http_status':404}
    assert 'private provider data' not in json.dumps(details)


def test_error_chain_has_a_hard_inspection_bound():
    class Wrapper(Exception):
        def unwrap(self):return self
    assert a.failure(Wrapper())=={'error_class':'Wrapper'}


def test_source_sha_verified_stream_uses_required_transport_checks_only(tmp_path,monkeypatch):
    import boto3
    path=tmp_path/'store.json'
    path.write_text(json.dumps({'endpoint':'https://garage.example','bucket':'results','access_key':'test','secret_key':'test'}))
    captured=[]
    monkeypatch.setattr(boto3,'client',lambda *args,**kw:captured.append(kw['config']) or object())
    store=ArtifactStore(path,tmp_path/'registry')
    store.client();store.client(verified_stream=True)
    assert captured[0].response_checksum_validation is None
    assert captured[1].response_checksum_validation=='when_required'
