"""Durable dataset transport: lost replies, partial writes and immutable publication."""
import base64
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.data import registry_transport
from ml_exp_server.data.ccr_uploads import CCRDataUploads, payload_prefix, payload_suffix
from ml_exp_server.data.data_assets import AssetStore
from ml_exp_server.data.registry_transport import digest
from tests.fake_registry import MemoryRegistry


class Assets:
    validate_inventory = AssetStore.validate_inventory
    file_limit, expanded_limit = 20, None
    def __init__(self, root, config):
        self.root, self.objects, self.published = root, SimpleNamespace(config=config), []
        root.mkdir()
    def publish_remote(self, project, value):
        assert value['project'] == project
        self.validate_inventory(value['files'])
        self.published.append(deepcopy(value))
        return deepcopy(value)


@pytest.fixture
def upload(tmp_path, monkeypatch):
    memory = MemoryRegistry()
    auth = tmp_path / 'auth.json'
    auth.write_text(json.dumps({'auths': {memory.host: {'auth': base64.b64encode(
        ('fake-user:' + memory.secret).encode()).decode()}}}))
    monkeypatch.setattr(registry_transport.http.client, 'HTTPSConnection', memory.connection)
    assets = Assets(tmp_path / 'data-assets', {'upload_part_bytes': 1024, 'upload_max_parts': 16,
        'data_registry': {'repository': memory.repository, 'base_image': memory.repository + '@' + memory.base_digest,
                          'registry_auth_file': str(auth)}})
    target = {'executor': 'cloud', 'backend': 'sensecore', 'storage_scope': 'nas:fake', 'profile_sha256': 'a' * 64}
    binding = {'project': 'trial', 'kind': 'asset', 'target': target}
    content = bytes(range(256)) * 10
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as archive:
        item = tarfile.TarInfo('tokens.bin'); item.size = len(content)
        archive.addfile(item, io.BytesIO(content))
    body = stream.getvalue()
    files = [{'path': 'tokens.bin', 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}]
    layer = payload_prefix(len(body)) + body + payload_suffix(len(body))
    driver = CCRDataUploads(assets, target, files, hashlib.sha256(layer).hexdigest())
    value = driver.create(binding, hashlib.sha256(body).hexdigest(), len(body), None)
    return SimpleNamespace(driver=driver, value=value, binding=binding, body=body,
        layer=layer, files=files, assets=assets, memory=memory, root=tmp_path, target=target)


def send(upload, number, *, body=None, sha=None, size=None):
    body = (upload.body[number*1024:(number+1)*1024] if body is None else body)
    path = upload.root / 'part.tmp'
    path.write_bytes(body)
    return upload.driver.part(upload.value['upload_id'], upload.binding, number, path,
        hashlib.sha256(body).hexdigest() if sha is None else sha, len(body) if size is None else size)


def all_parts(upload):
    for number in range(upload.value['part_count']):
        send(upload, number)


def reload(upload):
    upload.driver = CCRDataUploads(upload.assets, upload.target, upload.files,
                                   hashlib.sha256(upload.layer).hexdigest())


def test_dataset_stream_is_byte_exact_and_completed_publication_survives_restart(upload):
    all_parts(upload)
    result = upload.driver.complete(upload.value['upload_id'], upload.binding)
    assert upload.memory.blobs[digest(upload.layer)] == upload.layer
    with tarfile.open(fileobj=io.BytesIO(upload.layer)) as archive:
        assert archive.extractfile('payload/dataset.tar').read() == upload.body
    assert result['asset_id'] == 'asset.' + hashlib.sha256(upload.body).hexdigest()
    assert result['remote_storage'] == 'sensecore-ccr' and result['verification'] == 'PENDING_BACKEND'
    image = result['remote_locator']['image']
    manifest = json.loads(upload.memory.manifests[image.rsplit('@', 1)[1]])
    assert manifest['layers'][-2]['digest'] == digest(upload.layer)
    assert manifest['layers'][-2]['size'] == len(upload.layer)
    config = json.loads(upload.memory.blobs[manifest['config']['digest']])
    assert config['rootfs']['diff_ids'][-2] == digest(upload.layer)
    assert config['config']['Cmd'] == ['/bin/true']
    before = len(upload.memory.calls)
    reload(upload)
    assert upload.driver.complete(upload.value['upload_id'], upload.binding) == result
    assert len(upload.memory.calls) == before
    public = json.dumps(upload.driver.read(upload.value['upload_id'], upload.binding))
    for private in [upload.memory.secret, 'private-location-secret', 'auth.json', 'fake-bearer-secret']:
        assert private not in public
    assert list(upload.driver.root.glob('*.json')) and not list(upload.driver.root.glob('*.part'))


@pytest.mark.parametrize('failure', ['lost-ack', 700])
def test_lost_part_ack_and_partial_write_resume_from_registry_offset(upload, failure):
    upload.memory.patch_failure = failure
    with pytest.raises(ApplicationError) as failed:
        send(upload, 0)
    assert failed.value.code == 'CCR_TRANSPORT'
    reload(upload)
    observed = upload.driver.read(upload.value['upload_id'], upload.binding)
    assert len(observed['parts']) == (1 if failure == 'lost-ack' else 0)
    receipt = send(upload, 0)
    assert receipt['durable'] and receipt['number'] == 0
    for number in range(1, upload.value['part_count']):
        send(upload, number)
    upload.driver.complete(upload.value['upload_id'], upload.binding)
    assert upload.memory.blobs[digest(upload.layer)] == upload.layer


@pytest.mark.parametrize('stage', ['tail', 'finish', 'manifest'])
def test_completion_reconciles_lost_response_without_duplicate_bytes(upload, stage):
    all_parts(upload)
    if stage == 'tail':
        upload.memory.patch_failure = 'lost-ack'
    elif stage == 'finish':
        upload.memory.finish_failure = True
    else:
        upload.memory.publish_failure = True
    with pytest.raises(ApplicationError):
        upload.driver.complete(upload.value['upload_id'], upload.binding)
    reload(upload)
    result = upload.driver.complete(upload.value['upload_id'], upload.binding)
    assert upload.memory.blobs[digest(upload.layer)] == upload.layer
    assert result['remote_locator']['image'].rsplit('@', 1)[1] in upload.memory.manifests
    data_tags = [key for key in upload.memory.manifests if key.startswith('data-')]
    assert len(data_tags) == 1


def test_wrong_whole_layer_hash_never_publishes_an_asset(upload):
    upload.driver.transport_sha256 = 'f' * 64
    with upload.driver.state(upload.value['upload_id'], upload.binding) as (path, value):
        value['transport_sha256'] = 'f' * 64
        path.write_text(json.dumps(value))
    all_parts(upload)
    with pytest.raises(ApplicationError) as failure:
        upload.driver.complete(upload.value['upload_id'], upload.binding)
    assert failure.value.code == 'CCR_HTTP_400'
    assert upload.assets.published == []
    assert not any(key.startswith('data-') for key in upload.memory.manifests)


def test_bound_target_and_inventory_cannot_change_under_existing_upload_identity(upload):
    for target in [{**upload.target, 'executor': 'other'}, {**upload.target, 'storage_scope': 'nas:other'},
                   {**upload.target, 'profile_sha256': 'b' * 64}]:
        other = CCRDataUploads(upload.assets, target, upload.files, hashlib.sha256(upload.layer).hexdigest())
        with pytest.raises(ValueError, match='another target'):
            other.read(upload.value['upload_id'], upload.binding)
    different = CCRDataUploads(upload.assets, upload.target,
        [{**upload.files[0], 'path': 'different.bin'}], hashlib.sha256(upload.layer).hexdigest())
    with pytest.raises(ValueError, match='inventory'):
        different.create(upload.binding, upload.value['sha256'], upload.value['bytes'], None)
    upload.driver.transport_sha256 = 'b' * 64
    with pytest.raises(ValueError, match='inventory'):
        upload.driver.create(upload.binding, upload.value['sha256'], upload.value['bytes'], None)


def test_upload_retries_are_idempotent_but_different_part_bytes_and_order_are_rejected(upload):
    with pytest.raises(ApplicationError) as failure:
        send(upload, 1)
    assert failure.value.code == 'UPLOAD_PART_ORDER'
    receipt = send(upload, 0)
    calls = len(upload.memory.calls)
    assert send(upload, 0) == receipt and len(upload.memory.calls) == calls
    with pytest.raises(ValueError, match='already sealed'):
        send(upload, 0, body=b'different'.ljust(1024, b'.'))
    with pytest.raises(ValueError, match='checksum'):
        send(upload, 1, sha='f' * 64)
    with pytest.raises(ValueError, match='size'):
        send(upload, 1, size=1023)
    with pytest.raises(ValueError, match='missing parts'):
        upload.driver.complete(upload.value['upload_id'], upload.binding)


def test_upload_abort_and_expiry_have_explicit_terminal_states(upload):
    unknown = 'upload.' + 'f' * 64
    for operation in [upload.driver.read, upload.driver.abort]:
        with pytest.raises(ApplicationError) as failure:
            operation(unknown, upload.binding)
        assert failure.value.code == 'UNKNOWN_UPLOAD'
    with upload.driver.state(upload.value['upload_id'], upload.binding) as (path, value):
        value['expires_at'] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        path.write_text(json.dumps(value))
    with pytest.raises(ApplicationError) as failure:
        send(upload, 0)
    assert failure.value.code == 'UPLOAD_EXPIRED'
    assert upload.driver.abort(upload.value['upload_id'], upload.binding)['status'] == 'ABORTED'
    with pytest.raises(ApplicationError) as failure:
        send(upload, 0)
    assert failure.value.code == 'UPLOAD_NOT_ACTIVE'


def test_pending_part_with_impossible_registry_offset_requires_inspection(upload):
    upload.memory.patch_failure = 700
    with pytest.raises(ApplicationError):
        send(upload, 0)
    identity = next(iter(upload.memory.uploads))
    upload.memory.uploads[identity].extend(b'x' * 5000)
    with pytest.raises(ApplicationError) as failure:
        upload.driver.read(upload.value['upload_id'], upload.binding)
    assert failure.value.code == 'CCR_UPLOAD_UNCERTAIN'
    assert upload.assets.published == []


def test_upload_identity_and_state_paths_refuse_wrong_owner_or_symlink(upload):
    with pytest.raises(ValueError, match='identity'):
        upload.driver.read('../outside', upload.binding)
    other = {**upload.binding, 'project': 'another-project'}
    with pytest.raises(ValueError, match='another target'):
        upload.driver.read(upload.value['upload_id'], other)
    path = upload.driver.root / (upload.value['upload_id'] + '.json')
    original = path.read_bytes()
    path.unlink()
    target = upload.root / 'outside.json'; target.write_bytes(original)
    path.symlink_to(target)
    with pytest.raises(ValueError, match='link'):
        upload.driver.read(upload.value['upload_id'], upload.binding)


@pytest.mark.parametrize('change', ['sha', 'size-type', 'zero-size', 'limit', 'part-size', 'part-count', 'transport', 'inventory'])
def test_create_validates_limits_and_required_frozen_identities_before_remote_request(upload, change):
    sha, size, limit = upload.value['sha256'], upload.value['bytes'], None
    if change == 'sha': sha = 'invalid'
    elif change == 'size-type': size = True
    elif change == 'zero-size': size = 0
    elif change == 'limit': limit = size - 1
    elif change == 'part-size': upload.assets.objects.config['upload_part_bytes'] = 1
    elif change == 'part-count': upload.assets.objects.config['upload_max_parts'] = 1
    elif change == 'transport': upload.driver.transport_sha256 = None
    else: upload.driver.files = []
    before = len(upload.memory.calls)
    with pytest.raises(ApplicationError) as failure:
        upload.driver.create(upload.binding, sha, size, limit)
    assert failure.value.code == 'UPLOAD_LIMIT' and len(upload.memory.calls) == before


@pytest.mark.parametrize('settings', [None, {}, {'repository': 'registry.example/test/data',
    'base_image': 'other.example/data@sha256:' + 'a' * 64, 'registry_auth_file': '/private/fake'}])
def test_create_requires_complete_matching_registry_configuration(upload, settings):
    upload.assets.objects.config['data_registry'] = settings
    with pytest.raises(ApplicationError) as failure:
        upload.driver.create(upload.binding, 'b' * 64, upload.value['bytes'], None)
    assert failure.value.code == 'CCR_DATA_DISABLED'


def test_recreate_same_session_is_read_only_and_unknown_completion_is_not_active(upload):
    before = len(upload.memory.calls)
    assert upload.driver.create(upload.binding, upload.value['sha256'], upload.value['bytes'], None) == upload.value
    assert len(upload.memory.calls) == before
    with pytest.raises(ApplicationError) as failure:
        upload.driver.complete('upload.' + 'f' * 64, upload.binding)
    assert failure.value.code == 'UNKNOWN_UPLOAD'


def test_partial_pending_part_rejects_a_different_retry_checksum(upload):
    upload.memory.patch_failure = 700
    with pytest.raises(ApplicationError):
        send(upload, 0)
    with pytest.raises(ValueError, match='pending part differs'):
        send(upload, 0, body=b'different'.ljust(1024, b'.'))


def test_partial_tail_restarts_from_confirmed_offset(upload):
    all_parts(upload)
    upload.memory.patch_failure = 700
    with pytest.raises(ApplicationError):
        upload.driver.complete(upload.value['upload_id'], upload.binding)
    reload(upload)
    upload.driver.complete(upload.value['upload_id'], upload.binding)
    assert upload.memory.blobs[digest(upload.layer)] == upload.layer


@pytest.mark.parametrize('field,value', [('architecture', 'arm64'), ('os', 'windows'), ('rootfs', {'type': 'invalid'})])
def test_data_base_image_architecture_and_rootfs_are_checked_before_publication(upload, field, value):
    from ml_exp_server.data.registry_transport import encoded
    config = {**upload.memory.base_config, field: value}
    raw = encoded(config)
    upload.memory.blobs[digest(raw)] = raw
    base = {**upload.memory.base, 'config': {**upload.memory.base['config'], 'digest': digest(raw), 'size': len(raw)}}
    base_sha = digest(encoded(base)); upload.memory.manifests[base_sha] = encoded(base)
    with upload.driver.state(upload.value['upload_id'], upload.binding) as (path, state):
        state['registry']['base_image'] = upload.memory.repository + '@' + base_sha
        path.write_text(json.dumps(state))
    all_parts(upload)
    with pytest.raises(ValueError, match='linux/amd64'):
        upload.driver.complete(upload.value['upload_id'], upload.binding)
    assert upload.assets.published == []


def test_abort_is_idempotent_but_never_removes_published_images(upload):
    all_parts(upload)
    upload.driver.complete(upload.value['upload_id'], upload.binding)
    with pytest.raises(ValueError, match='published'):
        upload.driver.abort(upload.value['upload_id'], upload.binding)
    # A second explicitly cancelled unfinished session has no locator to delete.
    other = upload.driver.create(upload.binding, 'b' * 64, upload.value['bytes'], None)
    upload.driver.abort(other['upload_id'], upload.binding)
    before = len(upload.memory.calls)
    assert upload.driver.abort(other['upload_id'], upload.binding)['status'] == 'ABORTED'
    assert len(upload.memory.calls) == before


def test_finalization_disconnection_before_commit_retries_only_the_original_complete_upload(upload):
    all_parts(upload)
    upload.memory.finish_failure = 'before-commit'
    with pytest.raises(ApplicationError):
        upload.driver.complete(upload.value['upload_id'], upload.binding)
    assert digest(upload.layer) not in upload.memory.blobs
    reload(upload)
    result = upload.driver.complete(upload.value['upload_id'], upload.binding)
    assert upload.memory.blobs[digest(upload.layer)] == upload.layer
    assert result['remote_locator']['image'].rsplit('@', 1)[1] in upload.memory.manifests
    # Dataset finalization reuses its exact upload session; metadata/config may
    # legitimately create their own content-addressed blob sessions afterward.
    finalized = [call for call in upload.memory.calls if call[1] == 'PUT' and '/uploads/session-1' in call[2]]
    assert len(finalized) == 2


@pytest.mark.parametrize('failure', ['head-503', 'offset-missing', 'offset-transport', 'offset-mismatch'])
def test_uncertain_finalization_never_sends_put_without_missing_blob_and_exact_offset(upload, failure):
    from tests.fake_registry import Response
    all_parts(upload)
    upload.memory.finish_failure = 'before-commit'
    with pytest.raises(ApplicationError):
        upload.driver.complete(upload.value['upload_id'], upload.binding)
    state = json.loads((upload.driver.root / (upload.value['upload_id'] + '.json')).read_text())
    if failure == 'head-503':
        upload.memory.overrides['HEAD', upload.memory.prefix + '/blobs/' + digest(upload.layer)] = Response(503)
    elif failure == 'offset-missing':
        upload.memory.overrides['GET', state['location']] = Response(204)
    elif failure == 'offset-transport':
        upload.memory.overrides['GET', state['location']] = OSError('fake offset disconnected')
    else:
        upload.memory.uploads['session-1'].pop()
    before = len([call for call in upload.memory.calls if call[1] == 'PUT'])
    with pytest.raises((ApplicationError, ValueError)) as stopped:
        upload.driver.complete(upload.value['upload_id'], upload.binding)
    if failure == 'offset-mismatch':
        assert stopped.value.code == 'CCR_UPLOAD_UNCERTAIN'
    assert len([call for call in upload.memory.calls if call[1] == 'PUT']) == before
    assert upload.assets.published == []


def test_explicit_abort_then_create_starts_new_generation_without_reusing_old_parts(upload):
    send(upload, 0)
    path = upload.driver.root / (upload.value['upload_id'] + '.json')
    prior = json.loads(path.read_text())
    assert prior['parts']
    upload.driver.abort(upload.value['upload_id'], upload.binding)
    again = upload.driver.create(upload.binding, upload.value['sha256'], upload.value['bytes'], None)
    current = json.loads(path.read_text())
    assert again['upload_id'] == upload.value['upload_id'] and again['status'] == 'UPLOADING'
    assert again['parts'] == {} and current['offset'] == 0
    assert current['generation'] == prior['generation'] + 1
    assert current['location'] != prior['location']
    all_parts(upload)
    upload.driver.complete(upload.value['upload_id'], upload.binding)
    assert upload.memory.blobs[digest(upload.layer)] == upload.layer


def test_uncertain_begin_is_not_automatically_replayed_and_requires_explicit_abort(upload):
    post_path = upload.memory.prefix + '/blobs/uploads/'
    upload.memory.overrides['POST', post_path] = OSError('fake lost create response')
    with pytest.raises(ApplicationError):
        upload.driver.create(upload.binding, 'b' * 64, upload.value['bytes'], None)
    identity = 'upload.' + hashlib.sha256(json.dumps([upload.binding, 'b' * 64,
        upload.value['bytes']], sort_keys=True).encode()).hexdigest()
    before = len(upload.memory.calls)
    assert upload.driver.read(identity, upload.binding)['status'] == 'CREATING'
    assert upload.driver.create(upload.binding, 'b' * 64, upload.value['bytes'], None)['status'] == 'CREATING'
    assert len(upload.memory.calls) == before
    upload.driver.abort(identity, upload.binding)
    del upload.memory.overrides['POST', post_path]
    assert upload.driver.create(upload.binding, 'b' * 64, upload.value['bytes'], None)['status'] == 'UPLOADING'
    state = json.loads((upload.driver.root / (identity + '.json')).read_text())
    assert state['generation'] == 2


@pytest.mark.parametrize('size', [8*1024**3-1, 8*1024**3, 64*1024**3, 1024**4])
def test_payload_header_represents_large_archives_without_allocating_dataset_bytes(size):
    header = payload_prefix(size)
    assert len(header) == 512
    member = tarfile.TarInfo.frombuf(header, encoding='utf-8', errors='strict')
    assert member.name == 'payload/dataset.tar' and member.size == size
    assert member.isfile() and member.mode == 0o444
    assert len(payload_suffix(size)) == 1024 + (-size) % 512


@pytest.mark.parametrize('size', [1, 511, 512, 513, 1024, 1025])
def test_small_payload_transport_matches_an_independent_gnu_tar_writer(size):
    body = bytes(index % 251 for index in range(size))
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w', format=tarfile.GNU_FORMAT) as archive:
        member = tarfile.TarInfo('payload/dataset.tar')
        member.mode, member.size = 0o444, size
        archive.addfile(member, io.BytesIO(body))
    # tarfile pads whole archives to its 10 KiB record boundary; OCI needs only
    # the member's block padding followed by the two standard end blocks.
    layer = payload_prefix(size) + body + payload_suffix(size)
    assert stream.getvalue()[:len(layer)] == layer
    assert stream.getvalue()[len(layer):].strip(b'\0') == b''
    with tarfile.open(fileobj=io.BytesIO(layer), mode='r:') as archive:
        assert archive.extractfile('payload/dataset.tar').read() == body


def test_completed_part_confirmation_is_local_idempotent_even_after_upload_expiry(upload):
    all_parts(upload)
    upload.driver.complete(upload.value['upload_id'], upload.binding)
    path = upload.driver.root / (upload.value['upload_id'] + '.json')
    state = json.loads(path.read_text())
    state['expires_at'] = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    path.write_text(json.dumps(state))
    original = path.read_bytes()
    # Completed receipts remain self-contained even if the old credential file
    # no longer exists; recovery must not make a new registry request.
    Path(state['registry']['registry_auth_file']).unlink()
    before = len(upload.memory.calls)
    first = send(upload, 0)
    second = send(upload, 0)
    assert first == second == {'contract': 'upload-part.v1', 'upload_id': upload.value['upload_id'],
        'number': 0, 'sha256': hashlib.sha256(upload.body[:1024]).hexdigest(), 'bytes': 1024, 'durable': True}
    assert path.read_bytes() == original
    assert len(upload.memory.calls) == before


@pytest.mark.parametrize('failure', ['size', 'checksum', 'changed-sealed-bytes', 'truncated', 'oversized', 'missing-sealed-part'])
def test_completed_part_confirmation_still_requires_exact_original_bytes_and_sealed_metadata(upload, failure):
    all_parts(upload)
    upload.driver.complete(upload.value['upload_id'], upload.binding)
    before = len(upload.memory.calls)
    number, body, sha, size = 0, upload.body[:1024], None, None
    path = upload.driver.root / (upload.value['upload_id'] + '.json')
    if failure == 'size': size = 1023
    elif failure == 'checksum': sha = 'f' * 64
    elif failure == 'changed-sealed-bytes': body = b'different'.ljust(1024, b'.')
    elif failure == 'truncated': body, size = body[:-1], 1024
    elif failure == 'oversized': body, size = body + b'extra', 1024
    else:
        state = json.loads(path.read_text()); state['parts'].pop('0')
        path.write_text(json.dumps(state))
    original = path.read_bytes()
    with pytest.raises(ValueError):
        send(upload, number, body=body, sha=sha, size=size)
    assert path.read_bytes() == original and len(upload.memory.calls) == before
    assert json.loads(path.read_text())['status'] == 'COMPLETED'


@pytest.mark.parametrize('scope', ['nas:fake', 'nas:another'])
def test_content_image_is_deterministic_across_upload_times_and_executor_aliases(upload, scope):
    all_parts(upload)
    first = upload.driver.complete(upload.value['upload_id'], upload.binding)
    alias = {**upload.target, 'executor':'cloud-alias', 'storage_scope':scope, 'profile_sha256':'b'*64}
    binding = {**upload.binding, 'target':alias}
    driver = CCRDataUploads(upload.assets, alias, upload.files, hashlib.sha256(upload.layer).hexdigest())
    value = driver.create(binding, upload.value['sha256'], upload.value['bytes'], None)
    path = driver.root/(value['upload_id']+'.json')
    state = json.loads(path.read_text()); state['created_at']='2050-01-01T00:00:00+00:00'
    path.write_text(json.dumps(state))
    copied = SimpleNamespace(**{**vars(upload), 'driver':driver, 'binding':binding, 'value':value, 'target':alias})
    all_parts(copied)
    second = driver.complete(value['upload_id'], binding)
    assert second['created_at'] != first['created_at']
    assert second['executor'] != first['executor']
    assert second['remote_locator']['image'] == first['remote_locator']['image']
    assert second['remote_locator']['data_image_id'] == first['remote_locator']['data_image_id']
    manifest = json.loads(upload.memory.manifests[first['remote_locator']['image'].rsplit('@',1)[1]])
    control = upload.memory.blobs[manifest['layers'][-1]['digest']]
    with tarfile.open(fileobj=io.BytesIO(control)) as archive:
        metadata = json.loads(archive.extractfile('payload/asset.json').read())
    assert set(metadata) == {'project','asset_id','sha256','archive_bytes','files'}
    assert metadata['sha256'] == upload.value['sha256'] and metadata['files'] == upload.files
