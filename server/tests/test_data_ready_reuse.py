"""Readiness survives worker upgrades without relaunching uncertain copy jobs."""
import json

import pytest

from ml_exp_server import data_delivery as module
from ml_exp_server.image_builder import BuildStorageError, BuildTransportError
from tests.test_ccr_data_delivery import delivery, receipt, secret, IMAGE
from tests.test_sensecore_data_workflow import client, stored, custom_runtime
from tests.test_desktop_data_upload import stage, remote_client


def seal(service, value):
    service.update('demo', value['delivery_id'], status='QUEUED', image=IMAGE)
    return service.callback('demo', value['delivery_id'], secret(service, value), receipt(value))


def test_worker_upgrade_reuses_sealed_data_and_preserves_pending_operation(delivery, monkeypatch):
    api, service, old, asset = delivery
    old_path = service.root / 'demo' / (old['delivery_id'] + '.json')
    monkeypatch.setattr(module, 'data_worker_digest', lambda: 'd' * 64)
    pending = service.prepare('demo', asset['asset_id'], 'cloud')
    service.update('demo', pending['delivery_id'], status='RECONCILE_REQUIRED', error='unknown publication')
    pending_path = service.root / 'demo' / (pending['delivery_id'] + '.json')
    seal(service, old)
    old_bytes, pending_bytes = old_path.read_bytes(), pending_path.read_bytes()
    old_journal = old_path.with_suffix('.jsonl').read_bytes()
    pending_journal = pending_path.with_suffix('.jsonl').read_bytes()
    monkeypatch.setattr(module, 'builder_request', lambda *a: pytest.fail('must not build'))
    service._rest = object()  # Any scheduler call would fail.
    chosen = service.prepare('demo', asset['asset_id'], 'cloud')
    assert chosen['delivery_id'] == old['delivery_id'] and chosen['status'] == 'READY'
    assert service.ready_for('demo', asset['asset_id'], 'cloud')['delivery_id'] == old['delivery_id']
    endpoint = '/api/projects/demo/data-deliveries/' + pending['delivery_id']
    view = api.get(endpoint).json()
    assert view['status'] == 'READY' and view['operation_status'] == 'RECONCILE_REQUIRED'
    assert view['delivery_id'] == pending['delivery_id']
    assert view['reused_from'] == old['delivery_id']
    assert view['ready_delivery']['receipt'] == receipt(old)
    assert 'copy_token' not in json.dumps(view)
    for action in ('execute', 'reconcile'):
        assert api.post(endpoint + '/' + action, json={'confirmation': pending['confirmation']}).json()['status'] == 'READY'
    assert old_path.read_bytes() == old_bytes and pending_path.read_bytes() == pending_bytes
    assert old_path.with_suffix('.jsonl').read_bytes() == old_journal
    assert pending_path.with_suffix('.jsonl').read_bytes() == pending_journal
    assert service.read('demo', old['delivery_id']) == chosen
    # Run creation pins the canonical old receipt, then requires worker SHA checks.
    bundle = custom_runtime(api, monkeypatch)
    result = api.post('/api/projects/demo/runs', json={
        'run_id': 'reuse-ready', 'runtime_id': bundle['runtime_id'], 'executor': 'cloud',
        'inputs': [{'asset_id': asset['asset_id'], 'mount_path': '/inputs/data'}]})
    assert result.status_code == 200, result.text


@pytest.mark.parametrize('key', ['workspace', 'storage_mount', 'data_root'])
def test_different_nas_scope_never_reuses_ready(delivery, key):
    _, service, old, asset = delivery
    seal(service, old)
    definition = {**old, 'copy_profile': {**old['copy_profile'], key: 'other'}}
    assert service.ready_candidate(definition) is None


@pytest.mark.parametrize('key', ['project', 'asset_id', 'archive_sha256', 'files_sha256'])
def test_different_data_never_reuses_ready(delivery, key):
    _, service, old, asset = delivery
    seal(service, old)
    with service.state('demo', old['delivery_id']) as (store, snap):
        store.commit({**snap.value, key: 'other'}, expected_revision=snap.revision, event={'event': 'test-other-data'})
    with pytest.raises(ValueError): service.prepare('demo', asset['asset_id'], 'cloud')
    definition = {**old, 'delivery_id': 'delivery.' + 'e' * 64}
    assert service.ready_candidate(definition) is None


@pytest.mark.parametrize('case', ['id', 'image-type', 'image-digest', 'receipt-type', 'status', 'hash', 'path', 'symlink'])
def test_forged_ready_proof_fails_closed(delivery, case):
    _, service, old, asset = delivery
    seal(service, old)
    path = service.root / 'demo' / (old['delivery_id'] + '.json')
    if case == 'symlink':
        target = path.with_suffix('.saved'); path.rename(target); path.symlink_to(target)
    else:
        with service.state('demo', old['delivery_id']) as (store, snap):
            value = dict(snap.value)
            if case == 'id': value['delivery_id'] = 'other'
            elif case.startswith('image'): value['image'] = None if case == 'image-type' else 'unpinned'
            elif case == 'receipt-type': value['receipt'] = None
            else:
                value['receipt'] = {**value['receipt'], {'status': 'status', 'hash': 'files_sha256', 'path': 'data_path'}[case]: 'wrong'}
            store.commit(value, expected_revision=snap.revision, event={'event': 'test-corrupt-proof'})
    with pytest.raises(ValueError): service.prepare('demo', asset['asset_id'], 'cloud')


def test_invalid_named_record_is_not_a_reuse_candidate(delivery):
    _, service, old, _ = delivery
    (service.root / 'demo/delivery.invalid.json').write_text('{}')
    assert service.ready_candidate(old) is None


@pytest.mark.parametrize('key', ['project', 'delivery_id'])
def test_stored_identity_mismatch_and_interrupted_reused_operation(delivery, monkeypatch, key):
    _, service, old, asset = delivery
    monkeypatch.setattr(module, 'data_worker_digest', lambda: 'd' * 64)
    new = service.prepare('demo', asset['asset_id'], 'cloud')
    service.begin('demo', new['delivery_id'], new['confirmation'])
    seal(service, old)
    assert module.recover_data_deliveries(service.runtime) == 1
    assert service.read('demo', new['delivery_id'])['operation_status'] == 'RECONCILE_REQUIRED'
    with service.state('demo', new['delivery_id']) as (store, snap):
        store.commit({**snap.value, key: 'wrong'}, expected_revision=snap.revision, event={'event': 'corrupt-id'})
    with pytest.raises(ValueError): service.read('demo', new['delivery_id'])


@pytest.mark.parametrize('error', [BuildStorageError('BUILD_STORAGE_UNCHECKED', {'error_class': 'TimeoutExpired'}),
    BuildTransportError('DESKTOP_RPC_UNAVAILABLE', {'publication_uncertain': True}), ValueError('SECRET must never be returned')])
def test_failure_reports_phase_without_raw_exception_or_replay(delivery, monkeypatch, error):
    _, service, old, _ = delivery
    monkeypatch.setattr(module, 'builder_request', lambda *a: (_ for _ in ()).throw(error))
    value = service.finish(service.begin('demo', old['delivery_id'], old['confirmation']))
    assert value['status'] == 'RECONCILE_REQUIRED'
    assert value['diagnostic']['phase'] == 'IMAGE_PUBLICATION'
    assert value['diagnostic']['safe_to_retry'] is False
    assert value['diagnostic']['next_action'] == 'inspect_publication'
    assert 'SECRET' not in json.dumps(value)
    if hasattr(error, 'details'): assert value['diagnostic']['details'] == error.details
