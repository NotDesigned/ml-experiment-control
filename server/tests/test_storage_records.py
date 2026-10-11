"""Accepted metrics survive archive release and optional cloud publication."""
import io
import json
import pytest
from fastapi.testclient import TestClient
from ml_exp_server.api.app import create_app
from ml_exp_server.records.ingest import import_metrics
from ml_exp_server.records.store import RecordStore
from ml_exp_server.results.artifact_lifecycle import ArtifactLifecycle
from ml_exp_server.runs.context import RunContext
from ml_exp_server.schemas import ServerConfig
from ml_exp_server.tracking.tracking_service import normalized_metric
from ml_exp_server.workers import worker_launcher as launcher
from tests.test_artifact_store import storage, archive
from tests.test_worker_launcher import document
IDENTITY = ('demo', 'run-a', 'attempt-001')


def test_archive_tail_durable_before_release_without_wandb(storage, monkeypatch):
    objects, root, token, _, remote = storage
    record = {'name': 'loss', 'unit': 'nats/token', 'value': 2.0, 'step': 1, 'variant_id': 'small'}
    store = RecordStore(objects.root.parent)
    store.bind('demo', 'run-a', {'enabled': False}, {})
    scope = store.scope(*IDENTITY)
    store.append(scope, [('live', normalized_metric(record, {}))])
    tail = {**record, 'value': 1.5, 'step': 2}
    body = archive({'metrics.jsonl': (json.dumps(record)+'\n'+json.dumps(tail)+'\n').encode()})
    receipt = objects.receive(*IDENTITY, token, io.BytesIO(body), len(body))
    assert receipt['metrics_ingestion'] == {'status': 'COMPLETE', 'records': 2, 'invalid_lines': 0}
    assert len(store.metric_records(*IDENTITY)) == 2
    proof = {'archive_sha256': receipt['sha256'], 'archive_bytes': receipt['bytes'],
             'files': {'outputs/'+f['path']: {'sha256': f['sha256'], 'bytes': f['bytes']} for f in receipt['files']}}
    client = objects.client()
    client.delete_object = lambda **kw: remote.pop((kw['Bucket'], kw['Key']))
    monkeypatch.setattr(objects, 'client', lambda: client)
    lifecycle = ArtifactLifecycle(objects)
    lifecycle.acknowledge(*IDENTITY, proof)
    assert lifecycle.run_due() == 1
    assert not remote and not (root/'attempts/attempt-001/uploaded_outputs/metrics.jsonl').exists()
    assert [r['value'] for r in RecordStore(objects.root.parent).metric_records(*IDENTITY)] == [2.0, 1.5]


def test_metrics_database_failure_prevents_publication(storage, monkeypatch):
    objects, _, token, _, remote = storage
    body = archive({'metrics.jsonl': b'{"name":"loss","unit":"nats/token","value":1}\n'})
    original = RecordStore.append
    def fail(*args): raise OSError(28, 'private-path')
    monkeypatch.setattr(RecordStore, 'append', fail)
    with pytest.raises(OSError): objects.receive(*IDENTITY, token, io.BytesIO(body), len(body))
    with objects.record(*IDENTITY) as (_, value): assert value['receipt'] is None
    assert not remote
    monkeypatch.setattr(RecordStore, 'append', original)
    assert objects.receive(*IDENTITY, token, io.BytesIO(body), len(body))['metrics_ingestion']['records'] == 1


@pytest.mark.parametrize('content', [b'bad\n[]\n\xff\n', b'x'*140000+b'\n{}\n', b'x'*140000])
def test_malformed_metrics_retained_without_raw_content(tmp_path, content):
    tree = tmp_path/'tree'; tree.mkdir()
    (tree/'train_metrics.jsonl').write_bytes(content)
    result = import_metrics(tmp_path, *IDENTITY, tree, {'resolved_config': {'evaluation': {}}})
    assert result['invalid_lines'] > 0
    store = RecordStore(tmp_path)
    scope = store.scope(*IDENTITY)
    events = store.events(scope)
    assert events[0]['payload']['kind'] == 'diagnostic'
    assert store.latest_lifecycle(scope) == {}
    assert len(json.dumps(events)) < 3000
    assert import_metrics(tmp_path, *IDENTITY, tree, {}) == result
    assert store.total(scope) == len(events)


def test_plot_sampling_preserves_complete_validation():
    rows = [{'name': 'loss', 'unit': 'nats/token', 'step': 1, 'value': 4},
            {'name': 'loss', 'unit': 'nats/token', 'step': 2, 'value': float('nan')},
            {'name': 'loss', 'unit': 'nats/token', 'step': 3, 'value': 2},
            {'name': 'loss', 'unit': 'nats/token', 'step': 3, 'value': 3},
            {'name': 'loss', 'unit': 'nats/token', 'step': 4, 'value': 1}]
    result = RunContext._metric_payload(rows, keys=None, max_points=1, source=None, source_attempt_id='attempt-001')
    assert result['points'][0]['value'] == 1
    assert result['validation']['state'] == 'FAILED'
    assert result['validation']['failed_records'] == result['validation']['conflicting_records'] == 1
    assert RunContext._metric_payload([], keys=None, max_points=1, source=None, source_attempt_id=None)['validation']['state'] == 'NOT_OBSERVED'
    partial = RunContext._metric_payload(rows[-1:], keys=None, max_points=1, source=None, source_attempt_id=None,
              contract={'metrics_schema': {'definitions': {'accuracy': {'required': True}}}})
    assert partial['validation']['state'] == 'PARTIAL'


def test_normalized_nan_evidence_survives_query():
    rows = normalized_metric({'name': 'loss', 'unit': 'nats/token', 'value': float('nan')}, {})['observations']
    result = RunContext._metric_payload(rows, normalized=True, keys=None, max_points=1, source=None, source_attempt_id=None)
    assert result['metrics']['records'][0]['invalid_value'] == 'nan'


def test_launch_diagnostic_scope_bounds_and_privacy(storage, document, tmp_path):
    objects, _, token, config, _ = storage
    path = objects.seal_launch(*IDENTITY, document).removeprefix('https://example')
    server = ServerConfig(index_db=str(tmp_path/'idx.sqlite'), project_registry_root=str(objects.root.parent),
            collector_enabled=False, telemetry={'enabled': False}, container_execution={'artifact_store_file': str(config)})
    headers = {'Authorization': 'Bearer '+token}
    data = {'phase': 'CREATE_OUTPUT', 'status': 'FAILED', 'error_code': 'LAUNCH_IO_FAILED', 'errno': 28, 'exit_code': 78}
    with TestClient(create_app(server, poll=False)) as client:
        assert client.put(path, json=data, headers=headers).status_code == 200
        assert client.put(path, json=data, headers=headers).status_code == 200
        assert client.put(path, json=data, headers={'Authorization': 'Basic '+token}).status_code == 401
        assert client.put(path.replace('001', '002'), json=data, headers=headers).status_code == 401
        assert client.put(path, content=b'x'*4097, headers=headers).status_code == 413
        for invalid in [[], None, {'phase': 'private-path', 'status': 'FAILED'}, {**data, 'errno': 'private'},
                        {**data, 'exit_code': True}, {**data, 'errno': 9999}, {**data, 'error_code': 'private'},
                        {**data, 'status': 'private'}, {**data, 'extra': 'private'}]:
            assert client.put(path, json=invalid, headers=headers).status_code == 422
        assert client.put(path, content=b'bad', headers=headers).status_code == 422
        for i in range(20): assert client.put(path, json={**data, 'errno': i}, headers=headers).status_code == 200
    with objects.record(*IDENTITY) as (_, value):
        assert len(value['launch_diagnostics']) == 16
        assert value['launch_diagnostics'][-1]['errno'] == 19
        assert token not in json.dumps(value['launch_diagnostics'])
    server.container_execution.artifact_store_file = None
    with TestClient(create_app(server, poll=False)) as client: assert client.put(path, headers=headers).status_code == 404


def test_launch_reports_nas_errno_without_writable_nas(document, monkeypatch, capsys):
    reports = []
    monkeypatch.setattr(launcher, 'report', lambda url, token, data: reports.append(data))
    monkeypatch.setattr(launcher, 'fetch', lambda *a: document)
    def no_space(*args, **kwargs): raise OSError(28, 'private-path private-token')
    monkeypatch.setattr(launcher.Path, 'mkdir', no_space)
    assert launcher.main(['--manifest-url', 'https://example']) == 78
    assert reports[-1] == {'phase': 'CREATE_OUTPUT', 'status': 'FAILED', 'exit_code': 78, 'error_code': 'LAUNCH_IO_FAILED', 'errno': 28}
    assert 'private' not in capsys.readouterr().err


def test_launch_transport_bounded_and_best_effort(monkeypatch):
    calls = []
    class Connection:
        def request(self, *args, **kwargs): calls.append((args, kwargs))
        def getresponse(self): raise OSError('private')
        def close(self): calls.append('closed')
    monkeypatch.setattr(launcher, 'https_connection', lambda *a, **kw: Connection())
    url = 'https://example/api/launch-transfers/demo/run-a/attempt-001/'+'a'*64
    launcher.report(url, 'token', {'phase': 'FETCH_MANIFEST', 'status': 'STARTED'})
    assert calls[-1] == 'closed' and calls[0][0][0] == 'PUT'
    launcher.report('http://unsafe', 'token', {})


def test_local_journal_can_be_explicitly_backfilled_after_archive_release(client, stored):
    from tests.test_tracking import create_trial
    from ml_exp_server.tracking.tracking_contract import WandbOptions
    from ml_exp_server.tracking.tracking_service import store_for, backfill, observe
    _, _, controller = create_trial(client, options={'enabled': False})
    controller.dispatch_command({'command': controller.command('attempt-001'), 'attempt_id': 'attempt-001'})
    objects = __import__('ml_exp_server.results.artifact_store', fromlist=['ArtifactStore']).ArtifactStore(
        stored[0], client.app.state.runtime.config.project_registry_root_path())
    with objects.record('demo', 'trial', 'attempt-001') as (_, value):
        assert value['record_stream'] and 'ML_EXPD_RECORD_URL' in value['launch_manifest']['environment']
        token = value['token']
    endpoint = '/api/record-transfers/demo/trial/attempt-001'
    data = {'records': [{'event_id': 'live', 'kind': 'metrics', 'data': {'name': 'loss', 'value': 2, 'unit': 'nats/token', 'step': 1}}]}
    assert client.post(endpoint, json=data, headers={'Authorization': 'Bearer '+token}).status_code == 200
    assert client.get('/api/runs/demo/trial/metrics').json()['metrics']['records'][0]['value'] == 2
    assert client.get('/api/runs/demo/trial/tracking').json()['scopes'] == []
    store = store_for(client.app.state.runtime)
    from ml_exp_server.tracking.tracking_contract import WandbSettings
    store.configure(WandbSettings(api_key='private-test', entity='team'), lambda key: None)
    backfill(client.app.state.runtime, 'demo', 'trial', WandbOptions())
    assert store.metric_records('demo', 'trial--backfill', 'attempt-001')[0]['value'] == 2
    observe(client.app.state.runtime, store)
    assert len(store.metric_records('demo', 'trial--backfill', 'attempt-001')) == 1


def test_storage_maintenance_failure_does_not_hide_scheduler_progress(storage, monkeypatch):
    from types import SimpleNamespace
    from ml_exp_server.api import app as module
    from ml_exp_server.data.multipart_upload import UploadStore
    calls = []
    class Stop:
        def is_set(self): return False
        def wait(self, seconds): return True
    state = SimpleNamespace(_stop=Stop(), index=SimpleNamespace(set_meta=lambda *a: calls.append(a)),
            runtime=SimpleNamespace(config=SimpleNamespace(container_execution=SimpleNamespace(artifact_store_file='file'),
                    action_runtime=SimpleNamespace(allow_project_writes=True))), submit_job=lambda *a: None)
    monkeypatch.setattr(module, 'ResultCollectionService', lambda runtime: SimpleNamespace(objects=storage[0], automatic=lambda submit: None))
    def fail(self): raise OSError(28, 'private')
    monkeypatch.setattr(UploadStore, 'prune', fail)
    module._poll_loop(SimpleNamespace(state=state), SimpleNamespace(run_cycle=lambda: calls.append('scheduler'),
            config=SimpleNamespace(poll_interval_seconds=1)))
    assert 'scheduler' in calls and ('storage_maintenance_last_error', 'OSError') in calls

# Register the actual API fixture chain, rather than substitute its runtime.
from tests.test_tracking import client, stored, runtime


def test_archive_import_batches_long_histories(tmp_path):
    tree = tmp_path/'tree'; tree.mkdir()
    (tree/'metrics.jsonl').write_text(''.join(json.dumps({'name': 'loss', 'unit': 'nats/token', 'value': 1, 'step': n})+'\n' for n in range(65)))
    assert import_metrics(tmp_path, *IDENTITY, tree, {})['records'] == 65
    assert len(RecordStore(tmp_path).metric_records(*IDENTITY)) == 65
