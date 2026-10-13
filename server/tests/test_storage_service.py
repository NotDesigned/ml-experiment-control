"""The shared policy preserves resumability and canonical research records."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.data import multipart_upload
from ml_exp_server.records.store import RecordStore
from ml_exp_server.results.artifact_store import ArtifactStore
from ml_exp_server.runtime import ExperimentServerRuntime
from ml_exp_server.schemas import ServerConfig
from ml_exp_server.storage import atomic_json
from ml_exp_server.storage_capacity import CapacityError
from ml_exp_server.storage_service import StorageService
from tests.test_artifact_store import archive, storage

IDENTITY = ('demo', 'run-a', 'attempt-001')


def stage(service, binding, body, tmp_path):
    value = service.uploads.create(binding, hashlib.sha256(body).hexdigest(), len(body), None)
    for number in range(value['part_count']):
        part = body[number * value['part_bytes']:(number + 1) * value['part_bytes']]
        temporary = tmp_path / 'incoming'
        temporary.write_bytes(part)
        service.uploads.part(value['upload_id'], binding, number, temporary,
                             hashlib.sha256(part).hexdigest(), len(part))
    return value['upload_id']


def test_lazy_runtime_storage_preserves_current_configuration(storage, tmp_path):
    _, _, _, config, _ = storage
    settings = ServerConfig(index_db=str(tmp_path / 'index.sqlite'), collector_enabled=False,
                            telemetry={'enabled': False}, project_registry_root=str(tmp_path / 'registry'))
    with ExperimentServerRuntime.create(settings, projects=[]) as runtime:
        with pytest.raises(ApplicationError) as error:
            runtime.storage_service
        assert error.value.code == 'RESULT_COLLECTION_DISABLED'
        runtime.config.container_execution.artifact_store_file = str(config)
        first = runtime.storage_service
        assert first.objects.config['bucket'] == 'results'
        assert not first.objects.root.parent.joinpath('storage-reservations').exists()
        assert first.capacity is first.objects.capacity is first.uploads.capacity
        value = json.loads(config.read_text())
        value['local_storage_headroom_bytes'] = 123
        config.write_text(json.dumps(value))
        assert runtime.storage_service.capacity.headroom == 123


def test_separate_facades_share_disk_admission_and_durable_promises(storage, monkeypatch):
    objects, _, _, config, _ = storage
    objects.config['local_storage_headroom_bytes'] = 0
    value = json.loads(config.read_text())
    value['local_storage_headroom_bytes'] = 0
    config.write_text(json.dumps(value))
    monkeypatch.setattr('shutil.disk_usage', lambda _: SimpleNamespace(free=1000))
    first = StorageService(objects)
    second = StorageService(ArtifactStore(config, objects.root.parent))
    binding = {'project': 'demo', 'kind': 'asset'}
    with first.capacity.reserve(500):
        with pytest.raises(CapacityError):
            second.uploads.create(binding, 'a' * 64, 600, None)
    upload = second.uploads.create(binding, 'a' * 64, 600, None)
    with pytest.raises(CapacityError):
        with first.capacity.reserve(401):
            pytest.fail('concurrent transfer promises must remain reserved')
    second.uploads.abort(upload['upload_id'], binding)
    with first.capacity.reserve(1000):
        pass


def test_interrupted_publication_ack_resume_and_regenerated_cache_preserve_records(storage, tmp_path, monkeypatch):
    objects, root, token, config, remote = storage
    objects.config.update(upload_part_bytes=1024, local_storage_headroom_bytes=0)
    service = StorageService(objects)
    journal = RecordStore(objects.root.parent)
    journal.bind('demo', 'run-a', {'enabled': False}, {})
    metric = {'name': 'loss', 'unit': 'nats/token', 'value': 1.5, 'step': 2}
    files = {'weights.pt': b'final weights', 'metrics.jsonl': (json.dumps(metric) + '\n').encode()}
    body = archive(files)
    binding = dict(zip(('project', 'run_id', 'attempt_id'), IDENTITY), kind='artifacts')
    upload_id = stage(service, binding, body, tmp_path)
    directory = service.uploads.root / upload_id
    publish = lambda stream, size, digest: objects.receive(*IDENTITY, token, stream, size)
    original = multipart_upload.atomic_json

    def interrupt(path, value):
        if value['status'] == 'COMPLETED':
            raise OSError('interrupted after sealed receipt')
        return original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(multipart_upload, 'atomic_json', interrupt)
        with pytest.raises(OSError):
            service.uploads.complete(upload_id, binding, publish, len(body) * 2)
    assert service.uploads.read(upload_id, binding)['status'] == 'UPLOADING'
    assert len(list((directory / 'parts').iterdir())) > 1
    with objects.record(*IDENTITY) as (_, transfer):
        receipt = transfer['receipt']
    assert receipt['metrics_ingestion']['records'] == 1 and len(remote) == 1
    assert service.maintain() == 0  # No ACK; the sealed object and parts survive.
    assert (directory / 'parts').is_dir() and len(remote) == 1

    # Recreate the facade as after a daemon restart. The original receipt wins;
    # completion must not republish the archive or duplicate imported history.
    restarted = StorageService(ArtifactStore(config, objects.root.parent))
    with monkeypatch.context() as patch:
        client = objects.client()
        patch.setattr(client, 'upload_fileobj', lambda *a, **kw: pytest.fail('sealed archive was republished'))
        patch.setattr(ArtifactStore, 'client', lambda self: client)
        assert restarted.uploads.complete(upload_id, binding, publish, len(body) * 2) == receipt
    assert not (directory / 'parts').exists()
    assert len(journal.metric_records(*IDENTITY)) == 1
    objects.restore_cache(*IDENTITY)
    cache = root / 'attempts/attempt-001/uploaded_outputs'
    assert (cache / 'weights.pt').read_bytes() == files['weights.pt']

    # Canonical records, original outputs and separately governed datasets and
    # checkpoints are deliberately outside the delivery-cache allowlist.
    protected = [root / 'manifest.yaml', root / 'attempts/attempt-001/attempt.yaml']
    for relative in ['attempts/attempt-001/outputs/weights.pt', 'checkpoints/ready.json', 'datasets/tokens.bin']:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'canonical research state')
        protected.append(path)
    originals = {path: path.read_bytes() for path in protected}
    proof = {'archive_sha256': receipt['sha256'], 'archive_bytes': receipt['bytes'],
             'files': {'outputs/' + item['path']: {k: item[k] for k in ('bytes', 'sha256')}
                       for item in receipt['files']}}
    client = objects.client()
    deletes = []

    def delete_object(*, Bucket, Key):
        deletes.append((Bucket, Key))
        remote.pop((Bucket, Key), None)

    client.delete_object = delete_object
    monkeypatch.setattr(ArtifactStore, 'client', lambda self: client)
    service.lifecycle.acknowledge(*IDENTITY, proof, release=False)
    assert service.maintain() == 0 and cache.exists() and remote
    service.lifecycle.acknowledge(*IDENTITY, proof)
    assert service.maintain() == 1 and not cache.exists() and not remote
    assert service.lifecycle.acknowledge(*IDENTITY, proof)['status'] == 'RELEASED'
    assert service.maintain() == 0 and len(deletes) == 1
    cache.mkdir()
    for name, content in files.items():
        (cache / name).write_bytes(content)
    assert service.maintain() == 0 and not cache.exists() and len(deletes) == 1
    with objects.record(*IDENTITY) as (_, final):
        assert final['receipt'] == receipt and final['download_acknowledgement']['archive_sha256'] == receipt['sha256']
    assert {path: path.read_bytes() for path in protected} == originals
    assert [item['value'] for item in journal.metric_records(*IDENTITY)] == [1.5]


def test_maintenance_removes_only_expired_parts_before_release(storage, tmp_path, monkeypatch):
    objects, root, _, _, _ = storage
    objects.config.update(upload_part_bytes=1024, local_storage_headroom_bytes=0)
    service = StorageService(objects)
    binding = {'project': 'demo', 'kind': 'asset'}
    upload_id = stage(service, binding, b'dataset', tmp_path)
    directory = service.uploads.root / upload_id
    with service.uploads.record(upload_id) as (_, path, value):
        value['expires_at'] = '2000-01-01T00:00:00+00:00'
        atomic_json(path, value)
    protected = root / 'checkpoints/ready.json'
    protected.parent.mkdir()
    protected.write_text('persistent')

    def release():
        assert not (directory / 'parts').exists()
        assert json.loads((directory / 'upload.json').read_text())['status'] == 'EXPIRED'
        return 3

    monkeypatch.setattr(service.lifecycle, 'run_due', release)
    assert service.maintain() == 3 and protected.read_text() == 'persistent'
