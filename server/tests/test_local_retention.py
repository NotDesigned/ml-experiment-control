"""Local expiry is explicit, crash-safe and independent of download ACK/backend deletion."""
import io
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.data.data_assets import AssetStore
from ml_exp_server.data.locations import for_backend
from ml_exp_server.local_retention import LocalRetention, retention_seconds, remove_payloads
from ml_exp_server.results.artifact_lifecycle import ArtifactLifecycle
from ml_exp_server.results import artifact_cache
from ml_exp_server.results.artifact_store import ArtifactStore
from ml_exp_server.storage import atomic_json
from tests.test_artifact_store import storage, archive

IDENTITY = ('demo', 'run-a', 'attempt-001')
NOW = datetime.now(timezone.utc) + timedelta(days=3)


def policy(storage, monkeypatch):
    objects, root, token, config, remote = storage
    objects.config['local_payload_retention_seconds'] = 86400
    data = json.loads(config.read_text())
    data['local_payload_retention_seconds'] = 86400
    config.write_text(json.dumps(data))
    client = objects.client()
    deleted = []
    def delete_object(*, Bucket, Key):
        deleted.append(Key)
        remote.pop((Bucket, Key), None)
    client.delete_object = delete_object
    monkeypatch.setattr(ArtifactStore, 'client', lambda self: client)
    row = SimpleNamespace(project='demo', run_id='run-a', run_dir=str(root),
                          attempts=[SimpleNamespace(attempt_id='attempt-001', state='SUCCEEDED')])
    service = LocalRetention(objects, SimpleNamespace(list_runs=lambda: [row]))
    return service, row, deleted


@pytest.mark.parametrize('value', [0, -1, 59, True, 1.5, '86400'])
def test_invalid_policy_is_rejected(value):
    with pytest.raises(ValueError):
        retention_seconds({'local_payload_retention_seconds': value})


def test_default_policy_preserves_existing_keep_and_unack_contract(storage):
    objects, _, _, _, _ = storage
    assert retention_seconds(objects.config) is None
    assert LocalRetention(objects).maintain() is None


@pytest.mark.parametrize('keep', [False, True])
def test_archive_expiry_is_not_ack_and_preserves_receipt_metric_and_backend_records(storage, monkeypatch, keep):
    objects, root, token, _, remote = storage
    body = archive({'weights.pt': b'weights', 'metrics.jsonl': b'{"name":"loss","unit":"nats/token","value":1.5}\n'})
    receipt = objects.receive(*IDENTITY, token, io.BytesIO(body), len(body))
    if keep:
        with objects.record(*IDENTITY) as (path, value):
            value['retention'] = {'status': 'KEEP', 'sha256': receipt['sha256'], 'release_after': None}
            atomic_json(path, value)
    service, row, deleted = policy(storage, monkeypatch)
    protected = root / 'manifest.yaml'
    manifest = protected.read_bytes()
    objects.restore_cache(*IDENTITY)
    service.expire_attempt(row, 'attempt-001', NOW)
    with objects.record(*IDENTITY) as (_, value):
        assert value['receipt'] == receipt and 'download_acknowledgement' not in value
        assert value['retention']['status'] == 'EXPIRED'
        if keep:
            assert value['retention_history'][0]['status'] == 'KEEP'
    assert not remote and deleted == [receipt['object_key']]
    assert not (root / 'attempts/attempt-001/uploaded_outputs/weights.pt').exists()
    assert (root / 'attempts/attempt-001/uploaded_outputs/metrics.jsonl').exists()
    assert protected.read_bytes() == manifest
    for call in [lambda: objects.artifact_download(*IDENTITY), lambda: objects.restore_cache(*IDENTITY),
                 lambda: objects.receive(*IDENTITY, token, io.BytesIO(body), len(body)),
                 lambda: ArtifactLifecycle(objects).acknowledge(*IDENTITY, {})]:
        with pytest.raises(ApplicationError) as error:
            call()
        assert error.value.status_code == 410 and error.value.code == 'ARTIFACT_EXPIRED'
    service.expire_attempt(row, 'attempt-001', NOW)
    assert len(deleted) == 1


def test_archive_expiring_retries_only_exact_delete_and_preserves_previous_keep(storage, monkeypatch):
    objects, _, token, _, remote = storage
    body = archive({'weights.pt': b'weights'})
    receipt = objects.receive(*IDENTITY, token, io.BytesIO(body), len(body))
    service, row, deleted = policy(storage, monkeypatch)
    client = objects.client()
    original = client.delete_object
    def lost_reply(**kwargs):
        original(**kwargs)
        raise OSError('lost delete reply')
    client.delete_object = lost_reply
    with pytest.raises(OSError):
        service.expire_attempt(row, 'attempt-001', NOW)
    assert not remote
    with objects.record(*IDENTITY) as (_, value):
        assert value['retention']['status'] == 'EXPIRING'
    client.delete_object = original
    service.expire_attempt(row, 'attempt-001', NOW)
    assert deleted == [receipt['object_key']] * 2


def test_new_archives_active_collection_and_external_run_roots_are_retained(storage, monkeypatch):
    objects, root, token, _, remote = storage
    body = archive({'weights.pt': b'weights'})
    objects.receive(*IDENTITY, token, io.BytesIO(body), len(body))
    service, row, deleted = policy(storage, monkeypatch)
    service.expire_attempt(row, 'attempt-001', datetime.now(timezone.utc))
    assert remote and not deleted
    collection = objects.root.parent / 'result-collections/demo/run-a/attempt-001.json'
    atomic_json(collection, {'status': 'RECONCILE_REQUIRED'})
    service.expire_attempt(row, 'attempt-001', NOW)
    assert remote and not deleted
    collection.unlink()
    row.run_dir = str(root.parent.parent.parent.parent / 'external')
    service.expire_attempt(row, 'attempt-001', NOW)
    assert remote and not deleted


def test_unknown_transfer_wrong_root_and_invalid_object_identity_cannot_delete(storage, monkeypatch):
    objects, root, token, _, _ = storage
    service, row, deleted = policy(storage, monkeypatch)
    service.expire_attempt(row, 'attempt-002', NOW)
    body = archive({'weights.pt': b'w'})
    objects.receive(*IDENTITY, token, io.BytesIO(body), len(body))
    with objects.record(*IDENTITY) as (path, value):
        value['receipt']['object_key'] = 'another/project/object'
        atomic_json(path, value)
    with pytest.raises(ValueError, match='identity'):
        service.expire_attempt(row, 'attempt-001', NOW)
    assert not deleted
    row.run_dir = str(root / 'other')
    service.expire_attempt(row, 'attempt-001', NOW)


def test_legacy_dataset_expiry_retains_manifest_and_returns_typed_410(storage, monkeypatch):
    objects, _, _, config, remote = storage
    service, _, deleted = policy(storage, monkeypatch)
    assets = AssetStore(config, objects.root.parent)
    body = archive({'tokens.bin': b'data'})
    import hashlib
    value = assets.receive('demo', io.BytesIO(body), hashlib.sha256(body).hexdigest(), len(body))
    path = assets.root / 'demo' / value['asset_id']
    original = (path / 'asset.json').read_bytes()
    service.expire_asset(assets, 'demo', value['asset_id'], datetime.now(timezone.utc))
    assert not deleted
    service.expire_asset(assets, 'demo', value['asset_id'], NOW)
    assert not remote and not (path / 'tree').exists() and (path / 'asset.json').read_bytes() == original
    assert assets.read('demo', value['asset_id'])['local_retention']['status'] == 'EXPIRED'
    for call in [lambda: assets.archive('demo', value['asset_id']), lambda: assets.download('demo', value['asset_id']),
                 lambda: assets.receive('demo', io.BytesIO(body), value['sha256'], len(body))]:
        with pytest.raises(ApplicationError) as error:
            call()
        assert error.value.status_code == 410 and error.value.code == 'ASSET_EXPIRED'
    service.expire_asset(assets, 'demo', value['asset_id'], NOW)
    assert len(deleted) == 1


def test_remote_dataset_and_backend_copy_survive_local_expiry(storage, monkeypatch):
    objects, _, _, config, _ = storage
    service, _, deleted = policy(storage, monkeypatch)
    assets = AssetStore(config, objects.root.parent)
    asset = 'asset.' + 'a' * 64
    with assets.record('demo', asset) as path:
        path.mkdir()
        atomic_json(path / 'asset.json', {'project': 'demo', 'asset_id': asset, 'sha256': 'a' * 64,
                                         'remote_storage': 'wyd', 'files': [], 'archive_bytes': 123})
    service.expire_asset(assets, 'demo', asset, NOW)
    assert not deleted
    AssetStore.require_available({'remote_storage': 'wyd', 'local_retention': {'status': 'EXPIRED'}})


def test_payload_eviction_leaves_metadata_links_and_young_files(tmp_path):
    root = tmp_path / 'cache'
    root.mkdir()
    (root / 'weights.pt').write_bytes(b'weights')
    (root / 'metrics.jsonl').write_text('science')
    outside = tmp_path / 'outside.pt'
    outside.write_text('outside')
    (root / 'link.pt').symlink_to(outside)
    assert remove_payloads(root, seconds=86400, now=datetime.now(timezone.utc)) == 0
    assert remove_payloads(root, seconds=86400, now=NOW) == 7
    assert (root / 'metrics.jsonl').read_text() == 'science' and outside.read_text() == 'outside'
    assert remove_payloads(tmp_path / 'absent', seconds=86400, now=NOW) == 0


def test_maintain_skips_running_attempt_and_runs_expiry_only_when_enabled(storage, monkeypatch):
    service, row, _ = policy(storage, monkeypatch)
    calls = []
    monkeypatch.setattr(service, 'expire_attempt', lambda row, attempt, now: calls.append(attempt))
    row.attempts[0].state = 'RUNNING'
    service.maintain()
    assert not calls
    row.attempts[0].state = 'FAILED'
    service.maintain()
    assert calls == ['attempt-001']


def test_policy_limits_are_public_but_no_backend_or_secret_is_exposed(storage, monkeypatch):
    objects, _, _, config, _ = storage
    policy(storage, monkeypatch)
    limits = AssetStore(config, objects.root.parent).limits()['local_payload_retention']
    assert limits['max_age_seconds'] == 86400 and limits['keep_overrides_max_age'] is False
    assert limits['backend_storage_deleted'] is False


def test_asset_scan_validates_names_and_works_without_a_run_index(storage, monkeypatch):
    objects, _, _, config, _ = storage
    service, _, _ = policy(storage, monkeypatch)
    service.index = None
    assets = AssetStore(config, objects.root.parent)
    good = 'asset.' + 'a' * 64
    for project, asset in [('demo', good), ('bad project', good), ('demo', 'bad-asset')]:
        atomic_json(assets.root / project / asset / 'asset.json', {})
    calls = []
    monkeypatch.setattr(service, 'expire_asset', lambda assets, project, asset, now: calls.append((project, asset)))
    service.maintain()
    assert calls == [('demo', good)]


def test_asset_identity_and_metadata_marker_links_fail_closed(storage, monkeypatch, tmp_path):
    objects, _, _, config, _ = storage
    service, _, deleted = policy(storage, monkeypatch)
    assets = AssetStore(config, objects.root.parent)
    asset = 'asset.' + 'a' * 64
    service.expire_asset(assets, 'demo', asset, NOW)
    with assets.record('demo', asset) as path:
        path.mkdir()
    service.expire_asset(assets, 'demo', asset, NOW)
    outside = tmp_path / 'outside.json'
    outside.write_text('{}')
    metadata = path / 'asset.json'
    metadata.symlink_to(outside)
    with pytest.raises(ValueError, match='symlink'):
        service.expire_asset(assets, 'demo', asset, NOW)
    metadata.unlink()
    value = {'project': 'wrong', 'asset_id': asset, 'sha256': 'a' * 64}
    atomic_json(metadata, value)
    with pytest.raises(ValueError, match='identity'):
        service.expire_asset(assets, 'demo', asset, NOW)
    atomic_json(metadata, {**value, 'project': 'demo'})
    marker = path / 'local-retention.json'
    marker.symlink_to(outside)
    with pytest.raises(ValueError, match='symlink'):
        service.expire_asset(assets, 'demo', asset, NOW)
    with pytest.raises(ValueError, match='symlink'):
        assets.read('demo', asset)
    assert not deleted and outside.read_text() == '{}'


def test_replace_between_stat_open_and_second_open_never_authorizes_delete(tmp_path, monkeypatch):
    root = tmp_path / 'cache'
    root.mkdir()
    weight = root / 'weights.pt'
    weight.write_bytes(b'weights')
    original = os.fstat
    def changed(fd):
        value = original(fd)
        import stat
        if stat.S_ISREG(value.st_mode):
            return SimpleNamespace(**{key: getattr(value, key) for key in ['st_dev', 'st_ino', 'st_mtime_ns', 'st_ctime_ns']},
                                   st_size=value.st_size + 1)
        return value
    with monkeypatch.context() as patch:
        patch.setattr(os, 'fstat', changed)
        assert remove_payloads(root, seconds=86400, now=NOW) == 0
    assert weight.read_bytes() == b'weights'
    before = artifact_cache.signature(weight.stat())
    weight.write_bytes(b'changed')
    assert artifact_cache.evict_file(weight, {'bytes': 7, 'sha256': 'unused'}, {}, '', expected_signature=before) is None
    assert weight.read_bytes() == b'changed'


def test_expired_legacy_input_fails_preflight_but_remote_reference_still_works():
    profile = {'backend': {'kind': 'slurm', 'ssh_alias': 'wyd'}, 'storage_root': '/datapool/ml-expd'}
    asset = {'local_retention': {'status': 'EXPIRED'}}
    with pytest.raises(ApplicationError) as error:
        for_backend(asset, profile, 'demo')
    assert error.value.code == 'ASSET_EXPIRED'
