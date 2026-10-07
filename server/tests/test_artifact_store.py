"""Exact-Attempt S3 transfer and authorization, including interrupt recovery."""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest
import yaml

from ml_exp_server.api.app import create_app
from ml_exp_server.artifact_store import ArtifactStore
from ml_exp_server.worker_artifacts import archive_outputs
from ml_exp_server.schemas import ServerConfig, RunIndexRow, AttemptSummary
from ml_exp_server.source_imports import remove_staging
import tarfile


def archive(files):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as output:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            output.addfile(member, io.BytesIO(data))
    return stream.getvalue()


@pytest.fixture
def storage(tmp_path, monkeypatch):
    config = tmp_path / 's3.json'
    config.write_text(json.dumps({'endpoint':'http://localhost:3900','bucket':'results','access_key':'test',
                                'secret_key':'test','public_transfer_base':'https://example/api/artifact-transfers'}))
    store = ArtifactStore(config, tmp_path / 'projects')
    # Real tar parsing and private metadata; only the external S3 transport is substituted.
    objects = {}
    class S3:
        def upload_fileobj(self, stream, bucket, key, **kw): objects[(bucket, key)] = stream.read()
        def download_fileobj(self, bucket, key, stream, **kw): stream.write(objects[(bucket, key)])
    monkeypatch.setattr(ArtifactStore, 'client', lambda self:S3())
    monkeypatch.setattr(ArtifactStore, '_transfer_config', staticmethod(lambda:None))
    root = tmp_path / 'runs/demo/study/run-a'
    attempt = root / 'attempts/attempt-001'
    attempt.mkdir(parents=True)
    (root/'manifest.yaml').write_text('project: demo\nrun_id: run-a\n')
    (attempt/'attempt.yaml').write_text('project: demo\nrun_id: run-a\nattempt_id: attempt-001\n')
    url, token, _ = store.issue('demo','run-a','attempt-001',root,['**/*'])
    return store, root, token, config, objects


def test_s3_receipt_idempotence_conflict_and_cache_restore(storage):
    store, root, token, _, objects = storage
    data = archive({'checkpoint.bin':b'weights', 'metrics.jsonl':b'{"loss":1.0}\n'})
    first = store.receive('demo','run-a','attempt-001',token,io.BytesIO(data),len(data))
    assert first['sha256']==hashlib.sha256(data).hexdigest() and len(objects)==1
    assert store.receive('demo','run-a','attempt-001',token,io.BytesIO(data),len(data))==first
    with pytest.raises(ValueError, match='already sealed'):
        changed=archive({'checkpoint.bin':b'changed'})
        store.receive('demo','run-a','attempt-001',token,io.BytesIO(changed),len(changed))
    cache=root/'attempts/attempt-001/uploaded_outputs'
    remove_staging(cache)
    store.restore_cache('demo','run-a','attempt-001')
    assert (cache/'checkpoint.bin').read_bytes()==b'weights'
    with pytest.raises(ValueError):store.authorize('demo','run-a','attempt-002',token)
    with pytest.raises(ValueError):store.authorize('demo','run-a','attempt-001','different')


def test_worker_capability_cannot_read_or_control_the_daemon(storage,tmp_path):
    store, root, token, config, _ = storage
    bearer=tmp_path/'bearer';bearer.write_text('a'*40);bearer.chmod(0o600)
    server=ServerConfig(index_db=str(tmp_path/'index.sqlite'),project_registry_root=str(tmp_path/'projects'),
                        collector_enabled=False,telemetry={'enabled':False},http_auth={'bearer_token_file':str(bearer)},
                        container_execution={'artifact_store_file':str(config)})
    with TestClient(create_app(server,poll=False)) as client:
        endpoint='/api/artifact-transfers/demo/run-a/attempt-001'
        data=archive({'checkpoint.bin':b'result'})
        headers={'Authorization':'Bearer '+token}
        assert client.put(endpoint,content=data,headers=headers).status_code==200
        assert client.get('/api/projects',headers=headers).status_code==401
        assert client.get(endpoint,headers=headers).status_code==401
        assert client.put(endpoint.replace('001','002'),content=data,headers=headers).status_code==401
        assert client.put(endpoint,content=data,headers={'Authorization':'Bearer wrong'}).status_code==401
        client.app.state.runtime.index.upsert_run(RunIndexRow(project='demo',run_id='run-a',run_dir=str(root),
                                       attempts=[AttemptSummary(attempt_id='attempt-001',state='SUCCEEDED')]))
        download='/api/runs/demo/run-a/attempts/attempt-001/files/outputs/checkpoint.bin'
        response=client.get(download,headers={'Authorization':'Bearer '+'a'*40,'X-ML-Expd-Client-Protocol':'2'})
        assert response.status_code==200 and response.content==b'result'


def test_archive_declared_paths_and_escape_fail_before_s3(storage):
    store,root,token,_,objects=storage
    for name in ['../outside','.env','secrets/key']:
        data=archive({name:b'x'})
        with pytest.raises(ValueError):store.receive('demo','run-a','attempt-001',token,io.BytesIO(data),len(data))
    assert objects=={} and not (root/'attempts/attempt-001/uploaded_outputs').exists()


def test_worker_archive_bounds_and_output_filters(tmp_path):
    (tmp_path/'checkpoint.bin').write_bytes(b'checkpoint')
    (tmp_path/'private.log').write_bytes(b'log')
    (tmp_path/'.env').write_bytes(b'private')
    stream=io.BytesIO()
    assert archive_outputs(tmp_path,stream,100,['*.bin'])==1
    with pytest.raises(ValueError,match='limit'):
        archive_outputs(tmp_path,io.BytesIO(),1,['**/*'])
