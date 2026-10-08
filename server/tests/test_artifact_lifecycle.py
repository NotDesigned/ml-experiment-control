"""A download acknowledgement is the only authorization to release an archive."""
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.api.app import create_app
from ml_exp_server.results.artifact_store import ArtifactStore, require_artifact_available, artifact_released
from ml_exp_server.results.artifact_lifecycle import ArtifactLifecycle
from ml_exp_server.results.artifacts import ArtifactService
from ml_exp_server.results.result_collection import ResultCollectionService
from ml_exp_server.schemas import ServerConfig, RunIndexRow, AttemptSummary
from ml_exp_server.storage import atomic_json
from tests.test_artifact_store import storage, archive

IDENTITY = ('demo', 'run-a', 'attempt-001')


@pytest.fixture
def sealed(storage, monkeypatch):
    store, root, token, config, objects = storage
    body = archive({'weights.pt': b'weights', 'sub/metrics.jsonl': b'{}\n'})
    receipt = store.receive(*IDENTITY, token, io.BytesIO(body), len(body))
    proof = {'archive_sha256': receipt['sha256'], 'archive_bytes': receipt['bytes'],
             'files': {'outputs/' + item['path']: {k: item[k] for k in ('sha256', 'bytes')} for item in receipt['files']}}
    deletes = []
    def delete_object(*, Bucket, Key):
        deletes.append((Bucket, Key)); objects.pop((Bucket, Key), None)
    monkeypatch.setattr(ArtifactStore, 'client', lambda self: SimpleNamespace(delete_object=delete_object))
    return store, root, token, config, objects, proof, deletes


def due(store):
    with store.record(*IDENTITY) as (path, value):
        value['retention']['release_after'] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        atomic_json(path, value)


def test_acknowledgement_is_idempotent_and_does_not_extend_grace(sealed):
    store, root, _, _, objects, proof, deletes = sealed
    lifecycle = ArtifactLifecycle(store)
    assert not (root/'attempts/attempt-001/uploaded_outputs').exists()
    assert lifecycle.run_due() == 0 and objects
    first = lifecycle.acknowledge(*IDENTITY, proof)
    assert first['status'] == 'SCHEDULED'
    wait = datetime.fromisoformat(first['release_after']) - datetime.now(timezone.utc)
    assert timedelta(hours=23, minutes=59) < wait <= timedelta(hours=24)
    assert lifecycle.acknowledge(*IDENTITY, proof) == first
    assert lifecycle.run_due() == 0 and objects and not deletes
    kept = lifecycle.acknowledge(*IDENTITY, proof, release=False)
    assert kept['status'] == 'KEEP' and kept['release_after'] is None
    assert lifecycle.run_due() == 0
    assert lifecycle.acknowledge(*IDENTITY, proof)['status'] == 'SCHEDULED'


@pytest.mark.parametrize('change', ['sha', 'size', 'missing', 'extra', 'file-sha', 'file-bytes'])
def test_conflicting_client_proof_never_schedules_deletion(sealed, change):
    store, _, _, _, objects, proof, deletes = sealed
    if change == 'sha': proof['archive_sha256'] = 'a'*64
    if change == 'size': proof['archive_bytes'] += 1
    if change == 'missing': proof['files'].pop('outputs/weights.pt')
    if change == 'extra': proof['files']['outputs/other'] = {'bytes': 0, 'sha256': 'b'*64}
    if change == 'file-sha': proof['files']['outputs/weights.pt']['sha256'] = 'c'*64
    if change == 'file-bytes': proof['files']['outputs/weights.pt']['bytes'] += 1
    with pytest.raises(ApplicationError, match='differs'):
        ArtifactLifecycle(store).acknowledge(*IDENTITY, proof)
    assert ArtifactLifecycle(store).run_due() == 0 and objects and not deletes


def test_unknown_unpublished_attempt_and_invalid_identity(sealed):
    store, _, _, _, _, proof, _ = sealed
    with pytest.raises(ApplicationError): ArtifactLifecycle(store).acknowledge('demo','run-a','attempt-002',proof)
    with pytest.raises(ValueError): ArtifactLifecycle(store).acknowledge('demo','../outside','attempt-001',proof)
    with store.record(*IDENTITY) as (path, value):
        value['receipt'] = None; atomic_json(path, value)
    with pytest.raises(ApplicationError): ArtifactLifecycle(store).acknowledge(*IDENTITY,proof)
    assert not artifact_released(None)


@pytest.mark.parametrize('cache', ['missing', 'identical', 'extra', 'changed', 'legacy', 'file-link', 'dir-link', 'root-link'])
def test_release_retains_receipts_checkpoints_and_unrelated_local_files(sealed, cache, tmp_path):
    store, root, _, _, objects, proof, deletes = sealed
    checkpoint = root/'persistent.pt'; checkpoint.write_bytes(b'recovery state')
    outputs = root/'attempts/attempt-001/outputs'; outputs.mkdir(); (outputs/'unique').write_bytes(b'original')
    directory = root/'attempts/attempt-001/uploaded_outputs'
    if cache != 'missing':
        directory.mkdir(); (directory/'weights.pt').write_bytes(b'weights')
        (directory/'sub').mkdir(); (directory/'sub/metrics.jsonl').write_bytes(b'{}\n')
    if cache == 'extra': (directory/'extra').write_bytes(b'unique')
    if cache == 'changed': (directory/'weights.pt').write_bytes(b'changed')
    if cache == 'legacy':
        with store.record(*IDENTITY) as (path,value):
            for item in value['receipt']['files']: item.pop('sha256')
            atomic_json(path,value)
    if cache == 'file-link':
        (directory/'weights.pt').unlink(); (directory/'weights.pt').symlink_to(checkpoint)
    if cache == 'dir-link': (directory/'link').symlink_to(tmp_path, target_is_directory=True)
    if cache == 'root-link':
        from ml_exp_server.projects.source_imports import remove_staging
        remove_staging(directory); directory.symlink_to(tmp_path, target_is_directory=True)
    with store.record(*IDENTITY) as (_,value): original = dict(value['receipt'])
    lifecycle=ArtifactLifecycle(store); lifecycle.acknowledge(*IDENTITY, proof); due(store)
    assert lifecycle.run_due() == 1 and not objects and len(deletes) == 1
    with store.record(*IDENTITY) as (_,value):
        assert value['receipt'] == original
        assert value['retention']['status'] == 'RELEASED'
    assert checkpoint.read_bytes() == b'recovery state' and (outputs/'unique').read_bytes() == b'original'
    assert directory.exists() == (cache not in {'identical','missing'})
    assert lifecycle.run_due() == 0
    assert lifecycle.acknowledge(*IDENTITY,proof)['status'] == 'RELEASED'
    with pytest.raises(ApplicationError) as exc: store.artifact_download(*IDENTITY)
    assert exc.value.status_code == 410
    with pytest.raises(ApplicationError): store.restore_cache(*IDENTITY)


@pytest.mark.parametrize('after_effect', [False, True])
def test_failed_delete_retries_same_key_without_compute_or_losing_receipt(sealed, monkeypatch, after_effect):
    store, _, _, _, objects, proof, deletes = sealed
    lifecycle=ArtifactLifecycle(store); lifecycle.acknowledge(*IDENTITY,proof);due(store)
    client=store.client()
    def interrupted(**kwargs):
        if after_effect: client.delete_object(**kwargs)
        raise OSError('private external error')
    monkeypatch.setattr(ArtifactStore,'client',lambda self:SimpleNamespace(delete_object=interrupted))
    assert lifecycle.run_due() == 0
    with store.record(*IDENTITY) as (_,value):
        assert value['retention']['status'] == 'RELEASING'
        assert value['retention']['diagnostic'] == 'ARCHIVE_RELEASE_RETRY_REQUIRED'
    assert lifecycle.acknowledge(*IDENTITY,proof,release=False)['status'] == 'RELEASING'
    monkeypatch.setattr(ArtifactStore,'client',lambda self:client)
    assert lifecycle.run_due() == 1 and not objects
    assert len(set(deletes)) == 1


@pytest.mark.parametrize('change', ['key','sha','proof'])
def test_release_binding_cannot_delete_another_object(sealed, change):
    store, _, _, _, objects, proof, deletes = sealed
    lifecycle=ArtifactLifecycle(store);lifecycle.acknowledge(*IDENTITY,proof);due(store)
    with store.record(*IDENTITY) as (path,value):
        if change == 'key':value['receipt']['object_key']='other/run/secret.tar'
        if change == 'sha':value['retention']['sha256']='a'*64
        if change == 'proof':value['download_acknowledgement']['archive_bytes']+=1
        atomic_json(path,value)
    with pytest.raises(ValueError,match='identity'): lifecycle.run_due()
    assert objects and not deletes


def test_api_ack_requires_main_bearer_and_reports_released_metadata(sealed,tmp_path):
    store, root, worker_token, config, objects, proof, deletes = sealed
    settings=json.loads(config.read_text());settings['public_endpoint']='https://objects.example';config.write_text(json.dumps(settings))
    bearer=tmp_path/'bearer';bearer.write_text('a'*40);bearer.chmod(0o600)
    cfg=ServerConfig(index_db=str(tmp_path/'index.sqlite'),project_registry_root=str(store.root.parent),
                     collector_enabled=False,telemetry={'enabled':False},http_auth={'bearer_token_file':str(bearer)},
                     action_runtime={'allow_project_writes':True},
                     container_execution={'artifact_store_file':str(config)})
    with TestClient(create_app(cfg,poll=False)) as client:
        runtime=client.app.state.runtime
        runtime.index.upsert_run(RunIndexRow(project='demo',run_id='run-a',run_dir=str(root),
                                  attempts=[AttemptSummary(attempt_id='attempt-001',state='FAILED')]))
        endpoint='/api/runs/demo/run-a/attempts/attempt-001/artifacts/ack'
        assert client.post(endpoint,json=proof,headers={'Authorization':'Bearer '+worker_token}).status_code == 401
        headers={'Authorization':'Bearer '+'a'*40}
        assert client.post(endpoint.replace('001','002'),json=proof,headers=headers).status_code == 404
        assert client.post(endpoint,json={**proof,'archive_bytes':True},headers=headers).status_code == 422
        assert client.post(endpoint,json={**proof,'archive_sha256':'b'*64},headers=headers).status_code == 409
        runtime.config.action_runtime.allow_project_writes=False
        assert client.post(endpoint,json=proof,headers=headers).status_code == 409
        runtime.config.action_runtime.allow_project_writes=True
        assert client.post(endpoint,json=proof,headers=headers).json()['status'] == 'SCHEDULED'
        assert objects and not deletes
        due(store);assert ArtifactLifecycle(store).run_due() == 1
        for suffix in ['/artifacts/download','/artifacts/archive','/files/outputs/weights.pt']:
            assert client.get(endpoint.removesuffix('/artifacts/ack')+suffix,headers=headers).status_code == 410
        files=client.get(endpoint.removesuffix('/artifacts/ack')+'/files',headers=headers).json()
        assert not files['available'] and not files['collection_required'] and files['storage_status']=='RELEASED'
        assert len(files['files'])==2
        # Released receipts remain readable and must not trigger CPU recovery.
        service=ResultCollectionService(runtime)
        with store.record(*IDENTITY) as (path,value):
            definition={'project':'demo','run_id':'run-a','attempt_id':'attempt-001'}
            (root/'attempts/attempt-001/attempt.yaml').write_text(json.dumps(definition))
        state=service.read(*IDENTITY)
        assert state['status']=='RELEASED' and not state['download_available']
        runtime.config.container_execution.artifact_store_file=None
        assert client.post(endpoint,json=proof,headers=headers).status_code == 404


def test_explicit_legacy_cache_request_is_supported(storage):
    store,root,token,_,_=storage
    with store.record(*IDENTITY) as (path,value):
        value['cache_outputs']=True;atomic_json(path,value)
    data=archive({'weights.pt':b'weights'})
    store.receive(*IDENTITY,token,io.BytesIO(data),len(data))
    assert (root/'attempts/attempt-001/uploaded_outputs/weights.pt').read_bytes()==b'weights'


@pytest.mark.parametrize('enabled', [True,False])
def test_collector_respects_project_write_pause(sealed,monkeypatch,enabled):
    from ml_exp_server.api import app as module
    calls=[]
    class Stop:
        def is_set(self):return False
        def wait(self,seconds):return True
    state=SimpleNamespace(_stop=Stop(),index=SimpleNamespace(set_meta=lambda *args:None),
                          runtime=SimpleNamespace(config=SimpleNamespace(
                              container_execution=SimpleNamespace(artifact_store_file='configured'),
                              action_runtime=SimpleNamespace(allow_project_writes=enabled))),submit_job=lambda *args:None)
    service=SimpleNamespace(objects=sealed[0],automatic=lambda submit:calls.append('observe'))
    monkeypatch.setattr(module,'ResultCollectionService',lambda runtime:service)
    monkeypatch.setattr(ArtifactLifecycle,'run_due',lambda self:calls.append('release'))
    collector=SimpleNamespace(run_cycle=lambda:None,config=SimpleNamespace(poll_interval_seconds=1))
    module._poll_loop(SimpleNamespace(state=state),collector)
    assert calls==(['release','observe'] if enabled else ['observe'])


def test_crash_after_delete_before_final_commit_reconciles_only_owned_object(sealed,monkeypatch):
    from ml_exp_server.results import artifact_lifecycle as module
    store,_,_,_,objects,proof,deletes=sealed
    lifecycle=ArtifactLifecycle(store);lifecycle.acknowledge(*IDENTITY,proof);due(store)
    original=module.atomic_json
    def interrupted(path,value):
        if value['retention']['status']=='RELEASED':raise OSError('commit interrupted')
        original(path,value)
    monkeypatch.setattr(module,'atomic_json',interrupted)
    with pytest.raises(OSError):lifecycle.run_due()
    assert not objects
    with store.record(*IDENTITY) as (_,value):assert value['retention']['status']=='RELEASING'
    monkeypatch.setattr(module,'atomic_json',original)
    assert lifecycle.run_due()==1 and len(set(deletes))==1
