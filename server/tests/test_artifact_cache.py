"""Client ACK authorizes matching delivery copies, not unrelated research bytes."""
import errno
import os
import stat

import pytest

from ml_exp_server.results import artifact_cache as cache
from ml_exp_server.results.artifact_lifecycle import ArtifactLifecycle
from ml_exp_server.storage import atomic_json
from tests.test_artifact_lifecycle import sealed, IDENTITY
from tests.test_artifact_store import storage


def copy(root, name):
    path = root / 'attempts/attempt-001' / name
    path.mkdir(parents=True, exist_ok=True)
    (path/'weights.pt').write_bytes(b'weights')
    (path/'sub').mkdir(exist_ok=True)
    (path/'sub/metrics.jsonl').write_bytes(b'{}\n')
    return path


@pytest.mark.parametrize('state', ['unacknowledged', 'KEEP', 'RELEASED'])
def test_all_delivery_caches_obey_original_ack_and_do_not_redelete_object(sealed, state):
    store, root, _, _, objects, proof, deletes = sealed
    lifecycle = ArtifactLifecycle(store)
    if state != 'unacknowledged':
        lifecycle.acknowledge(*IDENTITY, proof, release=state != 'KEEP')
    if state == 'RELEASED':
        assert lifecycle.run_due() == 1
    names = ['uploaded_outputs', 'recovered_outputs', 'collected_run/attempts/attempt-001/outputs']
    paths = [copy(root, name) for name in names]
    protected = copy(root, 'outputs')
    assert lifecycle.run_due() == 0
    assert all(p.exists() for p in paths) == (state != 'RELEASED')
    assert (protected/'weights.pt').read_bytes() == b'weights'
    assert len(deletes) == (state == 'RELEASED')
    if state == 'RELEASED':
        with store.record(*IDENTITY) as (_, value):
            assert value['cache_cleanup'] == {'files_removed': 6, 'bytes_removed': 30, 'retained': {}}
        record = next(store.root.glob('*/*/attempt-*.json'))
        original = record.read_bytes()
        assert lifecycle.run_due() == 0 and record.read_bytes() == original
        # A delayed mirror is still checked; no permanent "already scanned" exemption.
        late = copy(root, names[-1])
        assert lifecycle.run_due() == 0 and not late.exists() and len(deletes) == 1


def test_partial_cleanup_preserves_changes_hidden_files_modes_and_skips_unchanged_mismatches(sealed, monkeypatch):
    store, root, _, _, _, proof, _ = sealed
    path = copy(root, 'uploaded_outputs')
    (path/'weights.pt').write_bytes(b'changed')
    (path/'.unique').write_bytes(b'unlisted')
    for directory in (path, path/'sub'): directory.chmod(0o500)
    lifecycle = ArtifactLifecycle(store); lifecycle.acknowledge(*IDENTITY, proof)
    assert lifecycle.run_due() == 1
    assert (path/'weights.pt').read_bytes() == b'changed' and (path/'.unique').read_bytes() == b'unlisted'
    assert not (path/'sub').exists() and stat.S_IMODE(path.stat().st_mode) == 0o500
    with monkeypatch.context() as patch:
        patch.setattr(cache, 'stream_digest', lambda stream: pytest.fail('unchanged mismatch was rehashed'))
        assert lifecycle.run_due() == 0
    path.chmod(0o700); (path/'weights.pt').write_bytes(b'weights'); path.chmod(0o500)
    assert lifecycle.run_due() == 0 and not (path/'weights.pt').exists()
    assert (path/'.unique').read_bytes() == b'unlisted' and stat.S_IMODE(path.stat().st_mode) == 0o500
    with store.record(*IDENTITY) as (_, value): assert not value['cache_cleanup']['retained']


@pytest.mark.parametrize('kind', ['ancestor-link', 'file-link', 'fifo', 'size', 'missing', 'legacy'])
def test_unsafe_or_unproved_copies_are_retained(sealed, tmp_path, kind):
    store, root, _, _, _, proof, _ = sealed
    path = copy(root, 'uploaded_outputs'); weight = path/'weights.pt'
    external = tmp_path/'outside'; external.mkdir(); (external/'metrics.jsonl').write_bytes(b'{}\n')
    if kind == 'ancestor-link':
        (path/'sub/metrics.jsonl').unlink(); (path/'sub').rmdir(); (path/'sub').symlink_to(external)
    elif kind == 'file-link':
        weight.unlink(); weight.symlink_to(external/'metrics.jsonl')
    elif kind == 'fifo': weight.unlink(); os.mkfifo(weight)
    elif kind == 'size': weight.write_bytes(b'different size')
    elif kind == 'missing': weight.unlink()
    else:
        with store.record(*IDENTITY) as (record, value):
            for item in value['receipt']['files']: item.pop('sha256')
            atomic_json(record, value)
    lifecycle = ArtifactLifecycle(store); lifecycle.acknowledge(*IDENTITY, proof)
    assert lifecycle.run_due() == 1
    assert (external/'metrics.jsonl').read_bytes() == b'{}\n'
    if kind == 'legacy': assert weight.read_bytes() == b'weights'
    elif kind == 'size': assert weight.read_bytes() == b'different size'
    elif kind == 'fifo': assert stat.S_ISFIFO(weight.lstat().st_mode)
    elif kind == 'file-link': assert weight.is_symlink()
    elif kind == 'ancestor-link': assert (path/'sub').is_symlink()
    else: assert not path.exists()


@pytest.mark.parametrize('changed', ['contents', 'replacement'])
def test_file_changed_during_hash_is_not_unlinked(tmp_path, monkeypatch, changed):
    path = tmp_path/'weights.pt'; path.write_bytes(b'weights')
    original = cache.stream_digest
    def race(stream):
        digest = original(stream)
        if changed == 'replacement': path.unlink()
        path.write_bytes(b'updated')
        return digest
    monkeypatch.setattr(cache, 'stream_digest', race)
    assert cache.evict_file(path, {'bytes': 7, 'sha256': 'irrelevant'}, {}, 'weights.pt') is None
    assert path.read_bytes() == b'updated'


def test_zero_byte_file_is_removed_and_directory_permissions_are_restored(tmp_path):
    from hashlib import sha256
    path = tmp_path/'empty'; path.touch(); tmp_path.chmod(0o500)
    assert cache.evict_file(path, {'bytes': 0, 'sha256': sha256(b'').hexdigest()}, {}, 'empty') == 0
    assert not path.exists() and stat.S_IMODE(tmp_path.stat().st_mode) == 0o500


@pytest.mark.parametrize('released', [False, True])
def test_cache_failure_retries_independently_of_object_delete(sealed, monkeypatch, released):
    from ml_exp_server.results import artifact_lifecycle as module
    store, root, _, _, _, proof, deletes = sealed
    lifecycle = ArtifactLifecycle(store); lifecycle.acknowledge(*IDENTITY, proof)
    if released: assert lifecycle.run_due() == 1
    path = copy(root, 'uploaded_outputs')
    original = module.evict_caches
    monkeypatch.setattr(module, 'evict_caches', lambda value: (_ for _ in ()).throw(PermissionError('private')))
    assert lifecycle.run_due() == (0 if released else 1) and len(deletes) == 1
    with store.record(*IDENTITY) as (_, value):
        assert value['retention']['status'] == 'RELEASED'
        assert value['cache_cleanup']['diagnostic'] == 'LOCAL_CACHE_CLEANUP_RETRY_REQUIRED'
    monkeypatch.setattr(module, 'evict_caches', original)
    assert lifecycle.run_due() == 0 and len(deletes) == 1 and not path.exists()
    with store.record(*IDENTITY) as (_, value): assert 'diagnostic' not in value['cache_cleanup']


@pytest.mark.parametrize('released', [False, True])
def test_forged_release_proof_cannot_delete_local_files(sealed, released):
    store, root, _, _, _, proof, _ = sealed
    lifecycle = ArtifactLifecycle(store); lifecycle.acknowledge(*IDENTITY, proof)
    if released: assert lifecycle.run_due() == 1
    path = copy(root, 'uploaded_outputs')
    with store.record(*IDENTITY) as (record, value):
        value['download_acknowledgement']['archive_bytes'] += 1; atomic_json(record, value)
    with pytest.raises(ValueError, match='identity'): lifecycle.run_due()
    assert (path/'weights.pt').read_bytes() == b'weights'


@pytest.mark.parametrize('operation', ['directory', 'file', 'rmdir'])
def test_unexpected_io_errors_are_not_silenced(tmp_path, monkeypatch, operation):
    if operation == 'directory':
        monkeypatch.setattr(cache, 'open_directory', lambda path: (_ for _ in ()).throw(PermissionError(errno.EACCES, 'private')))
        with pytest.raises(PermissionError):
            with cache.directory(tmp_path): pass
    elif operation == 'file':
        original = os.open
        def open_file(name, *args, **kwargs):
            if name == 'weights.pt': raise PermissionError(errno.EACCES, 'private')
            return original(name, *args, **kwargs)
        monkeypatch.setattr(cache.os, 'open', open_file)
        with pytest.raises(PermissionError): cache.evict_file(tmp_path/'weights.pt', {'bytes': 7}, {}, 'weights.pt')
    else:
        monkeypatch.setattr(cache.os, 'rmdir', lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError(errno.EACCES, 'private')))
        with pytest.raises(PermissionError): cache.prune_empty(tmp_path/'absent')


def test_prune_ignores_missing_or_linked_parent(tmp_path):
    cache.prune_empty(tmp_path/'absent/child')


def test_hidden_archive_files_do_not_block_other_cache_cleanup(storage, monkeypatch):
    import io
    from tests.test_artifact_store import archive
    from ml_exp_server.results.artifact_store import ArtifactStore
    from types import SimpleNamespace
    store, root, token, _, _ = storage
    body = archive({'.hidden': b'private', 'visible': b'public'})
    receipt = store.receive(*IDENTITY, token, io.BytesIO(body), len(body))
    proof = {'archive_sha256': receipt['sha256'], 'archive_bytes': receipt['bytes'],
             'files': {'outputs/' + f['path']: {k: f[k] for k in ('sha256', 'bytes')} for f in receipt['files']}}
    directory = root/'attempts/attempt-001/uploaded_outputs'; directory.mkdir()
    (directory/'.hidden').write_bytes(b'private'); (directory/'visible').write_bytes(b'public')
    deletes=[]
    monkeypatch.setattr(ArtifactStore, 'client', lambda self: SimpleNamespace(delete_object=lambda **kwargs: deletes.append(kwargs)))
    lifecycle = ArtifactLifecycle(store); lifecycle.acknowledge(*IDENTITY, proof)
    assert lifecycle.run_due() == 1 and len(deletes) == 1
    assert (directory/'.hidden').read_bytes() == b'private' and not (directory/'visible').exists()
    assert lifecycle.run_due() == 0 and len(deletes) == 1


@pytest.mark.parametrize('unsafe', ['../escape', '/absolute', 'bad\\path'])
def test_unsupported_receipt_path_is_conservatively_retained(tmp_path, unsafe):
    report=cache.evict_caches({'run_dir': str(tmp_path), 'attempt_id': 'attempt-001',
                             'receipt': {'files': [{'path': unsafe, 'bytes': 1, 'sha256': 'a'*64}]}})
    assert report['files_removed'] == report['bytes_removed'] == 0
