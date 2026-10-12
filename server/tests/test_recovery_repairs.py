"""Production regressions: partial archives, expired review and disappearing logs."""
import io
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.results.artifacts import ArtifactService
from ml_exp_server.runs.submissions import ExperimentSubmissionService
from ml_exp_server.container_execution import RunRequest
from experiment_control.backends.sensecore_rest import RESTError
from tests.test_artifact_store import storage, archive
from tests.test_result_collection import recovery, IDENTITY, cpu, output
from tests.test_result_job_diagnostics import provider, read
from tests.test_container_api import client as container_client, runtime as ready_runtime


@pytest.mark.parametrize('status', ['PREPARED', 'AUTHORIZED'])
@pytest.mark.parametrize('expires', [None, 'bad', '2020-01-01T00:00:00Z', '2099-01-01T00:00:00Z', '2099-01-01T00:00:00'])
def test_expired_or_invalid_submission_needs_new_review(status, expires):
    view = ExperimentSubmissionService._view({'execution': {'status': status}, 'gate_expires_at': expires})
    expected = ('AUTHORIZE' if status == 'PREPARED' else 'EXECUTE') if expires and expires.endswith('Z') and expires.startswith('2099') else 'REPREPARE'
    assert view['next_action'] == expected


def test_local_file_read_does_not_restore_large_s3_archive(recovery, monkeypatch):
    service, root, *_ = recovery
    output(root)
    restore = Mock()
    monkeypatch.setattr(type(service.objects), 'restore_cache', restore)
    opened = ArtifactService(service.runtime).open(*IDENTITY, 'outputs/weights.bin')
    assert b''.join(opened.chunks()) == b'weights'
    restore.assert_not_called()
    with pytest.raises(ApplicationError):
        ArtifactService(service.runtime).open(*IDENTITY, 'outputs/missing.bin')
    assert restore.call_count == 1


@pytest.mark.parametrize('uid,message,expected,conflict', [
    ('a'*32, "pod <pt-"+'a'*32+"-worker-0>'s container <pytorch> exited with code 78 | time: <2026-10-10T00:00:00Z>", 78, False),
    ('b'*32, "pod <pt-"+'a'*32+"-worker-0>'s container <pytorch> exited with code 78 | time: <2026-10-10T00:00:00Z>", None, False),
    ('a'*32, "pod <pt-"+'a'*32+"-worker-0>'s container <pytorch> exited with code 999 | time: <2026-10-10T00:00:00Z>", None, False),
    ('a'*32, 'exited with code 74', None, False),
])
def test_provider_exit_event_is_bound_to_exact_job_uid(uid, message, expected, conflict):
    rest = provider(); rest.describe.return_value.update(uid=uid)
    rest.request.return_value = {'events': [None, {'reason': 'ExitedWithCode', 'message': message}]}
    value = read(rest)
    assert (value['job_exit_code'], value['exit_code_conflict']) == (expected, conflict)
    assert len(value['exit_evidence']) == int(expected is not None)
    if expected is not None:
        rest.describe.return_value['exit_code'] = 137
        assert read(rest)['exit_code_conflict']


def test_training_observation_is_durable_and_expired_logs_do_not_erase_evidence(recovery, monkeypatch):
    service, root, _, _, _, definition, _ = recovery
    assert service.training_diagnostics(*IDENTITY)['logs']['status'] == 'NOT_APPLICABLE'
    definition['backend']['kind'] = 'sensecore'
    (root/'attempts/attempt-001/attempt.yaml').write_text(yaml.safe_dump(definition))
    rest = provider(); rest.describe.return_value.update(uid='a'*32, exit_code=1)
    rest.logs.return_value = {'text': 'safe failure evidence', 'available': True}
    monkeypatch.setattr('ml_exp_server.results.result_collection.SenseCoreREST.from_environment', lambda: rest)
    assert service.training_diagnostics(*IDENTITY)['job_exit_code'] is None
    first = service.training_diagnostics(*IDENTITY, refresh=True)
    assert first['job_exit_code'] == 1
    assert service.training_diagnostics(*IDENTITY) == first
    rest.describe.return_value.pop('exit_code'); rest.logs.return_value = {'text': ''}
    second = service.training_diagnostics(*IDENTITY, refresh=True)
    assert second['job_exit_code'] == 1 and second['logs']['lines'] == ['safe failure evidence']
    assert second['logs_observed_at'] == first['observed_at']
    rest.logs.return_value = {'text': 'new diagnostic evidence', 'available': True}
    rest.describe.return_value['exit_code'] = 137
    rest.workers.return_value = [{'containers': [{'exit_code': 1}]}]
    conflicted = service.training_diagnostics(*IDENTITY, refresh=True)
    assert conflicted['exit_code_conflict'] and conflicted['job_exit_code'] is None
    assert conflicted['logs']['lines'] == ['new diagnostic evidence']
    rest.describe.return_value.pop('exit_code'); rest.workers.return_value = []
    rest.describe.return_value['uid'] = 'b'*32
    assert service.training_diagnostics(*IDENTITY, refresh=True)['job_exit_code'] is None
    with service.objects.record(*IDENTITY) as (path, value):
        value['training_diagnostics']['scheduler_name'] = 'different'
        path.write_text(json.dumps(value))
    assert service.training_diagnostics(*IDENTITY)['job_exit_code'] is None
    def absent(): raise RESTError('configuration')
    monkeypatch.setattr('ml_exp_server.results.result_collection.SenseCoreREST.from_environment', absent)
    assert service.training_diagnostics(*IDENTITY, refresh=True)['logs']['unavailable_reason'] == 'PROVIDER_CONFIGURATION_UNAVAILABLE'
    assert not rest.create.called and not rest.stop.called
    assert not (service.root/'demo/run-a/attempt-001.json').exists()


def test_required_outputs_are_frozen_and_missing_files_rejected_before_publication(storage):
    store, root, token, _, remote = storage
    other = root/'attempts/attempt-002'; other.mkdir()
    (other/'attempt.yaml').write_text('project: demo\nrun_id: run-a\nattempt_id: attempt-002\n')
    _, cap, _ = store.issue('demo', 'run-a', 'attempt-002', root, ['**/*'], required_outputs=['**/weights.bin'])
    with pytest.raises(ValueError, match='bound'):
        store.issue('demo', 'run-a', 'attempt-002', root, ['**/*'])
    data = archive({'summary.json': b'{}'})
    with pytest.raises(ApplicationError) as error:
        store.receive('demo', 'run-a', 'attempt-002', cap, io.BytesIO(data), len(data))
    assert error.value.code == 'RESULT_OUTPUTS_INCOMPLETE' and not remote
    data = archive({'weights.bin': b'weights'})
    assert len(store.receive('demo', 'run-a', 'attempt-002', cap, io.BytesIO(data), len(data))['files']) == 1
    with pytest.raises(ValueError):
        RunRequest(run_id='a', runtime_id='runtime.'+'a'*64, executor='gpu', required_outputs=['../escape'])


@pytest.mark.parametrize('backend', ['slurm', 'sensecore'])
def test_archive_repair_preserves_original_receipt_and_object_and_is_idempotent(recovery, monkeypatch, backend):
    service, root, token, _, objects, *_ = recovery
    if backend == 'sensecore': cpu(recovery, monkeypatch)
    with service.objects.record(*IDENTITY) as (p, v):
        v['outputs'] = ['metrics.jsonl', 'weights.bin', 'logs/nested.json']
        p.write_text(json.dumps(v))
    old = archive({'metrics.jsonl': b'old', 'logs/nested.json': b'unchanged'})
    receipt = service.objects.receive(*IDENTITY, token, io.BytesIO(old), len(old))
    path = output(root); (path/'metrics.jsonl').write_bytes(b'old')
    (path/'logs').mkdir(); (path/'logs/nested.json').write_bytes(b'unchanged')
    (path/'.private').write_bytes(b'private'); (path/'excluded.txt').write_bytes(b'excluded')
    (path/'link').symlink_to(path/'weights.bin')
    # Simulate the old expanded cache, which must not shadow the corrected archive.
    service.objects.restore_cache(*IDENTITY)
    fixed = service.repair(*IDENTITY, receipt['sha256'])
    assert fixed['archive']['file_count'] == 3 and len(objects) == 2
    assert next(value for (_, key), value in objects.items() if key == receipt['object_key']) == old
    with service.objects.record(*IDENTITY) as (_, value):
        assert value['receipt_history'] == [receipt]
    assert not (root/'attempts/attempt-001/uploaded_outputs').exists()
    assert service.repair(*IDENTITY, receipt['sha256']) == fixed
    with service.objects.record(*IDENTITY) as (_, value):
        new = objects[('results', value['receipt']['object_key'])]
    assert service.objects.receive(*IDENTITY, token, io.BytesIO(new), len(new), repair_sha256=receipt['sha256'])['files']


@pytest.mark.parametrize('case', ['wrong-base', 'ack', 'changed', 'removed', 'no-receipt', 'cache-link', 'cache-changed', 'cache-root-link'])
def test_archive_repair_rejects_changes_and_unproven_cache_deletion(storage, tmp_path, case):
    store, root, token, _, objects = storage
    old = archive({'a': b'old'}); receipt = store.receive(*IDENTITY, token, io.BytesIO(old), len(old))
    expected = receipt['sha256']
    if case in {'ack', 'no-receipt'}:
        with store.record(*IDENTITY) as (p, v):
            if case == 'ack': v['retention'] = {'status': 'KEEP'}
            else: v['receipt'] = None
            p.write_text(json.dumps(v))
    if case == 'wrong-base': expected = '0'*64
    files = {'a': b'changed' if case == 'changed' else b'old', 'b': b'new'}
    if case == 'removed': files.pop('a')
    if case.startswith('cache-'):
        store.restore_cache(*IDENTITY); cache = root/'attempts/attempt-001/uploaded_outputs'
        cache.chmod(0o700)
        if case == 'cache-link': (cache/'extra').symlink_to(tmp_path)
        if case == 'cache-changed':
            (cache/'a').chmod(0o600)
            (cache/'a').write_bytes(b'changed')
        if case == 'cache-root-link':
            cache.rename(root/'cache-retained'); cache.symlink_to(root/'cache-retained')
    data = archive(files)
    with pytest.raises(ValueError): store.receive(*IDENTITY, token, io.BytesIO(data), len(data), repair_sha256=expected)
    assert len(objects) == 1


@pytest.mark.parametrize('case', ['active', 'no-receipt', 'wrong-base', 'no-files', 'capacity', 'cloud-active', 'unsupported', 'late-capacity'])
def test_repair_service_guards(recovery, monkeypatch, case):
    service, root, token, _, _, _, row = recovery
    data = archive({'metrics.jsonl': b'old'})
    receipt = service.objects.receive(*IDENTITY, token, io.BytesIO(data), len(data))
    if case != 'no-files': output(root)
    if case == 'active': row.attempts[0].state = 'RUNNING'
    if case == 'cloud-active': cpu(recovery, monkeypatch).describe.return_value = {'state': 'RUNNING'}
    if case == 'unsupported':
        definition = yaml.safe_load((root/'attempts/attempt-001/attempt.yaml').read_text())
        definition['backend']['kind'] = 'unsupported'
        (root/'attempts/attempt-001/attempt.yaml').write_text(yaml.safe_dump(definition))
    if case == 'no-receipt':
        with service.objects.record(*IDENTITY) as (p, v):
            v['receipt'] = None; p.write_text(json.dumps(v))
    if case == 'capacity': monkeypatch.setattr('ml_exp_server.results.result_collection.shutil.disk_usage', lambda _: Mock(free=0))
    if case == 'late-capacity':
        values = iter([10**20, 0])
        monkeypatch.setattr('ml_exp_server.results.result_collection.shutil.disk_usage', lambda _: Mock(free=next(values)))
    with pytest.raises(ApplicationError): service.repair(*IDENTITY, '0'*64 if case == 'wrong-base' else receipt['sha256'])


def test_wyd_recovery_does_not_trust_filtered_generic_mirror(recovery, monkeypatch):
    service, root, _, _, objects, *_ = recovery
    mirror = root/'attempts/attempt-001/collected_run/attempts/attempt-001/outputs'
    mirror.mkdir(parents=True); (mirror/'summary.json').write_text('{}')
    def complete_copy(backend, definition, attempt, local):
        local.mkdir(); (local/'weights.bin').write_bytes(b'complete')
    copy = Mock(side_effect=complete_copy)
    monkeypatch.setattr(service, 'read_wyd_outputs', copy)
    service.begin(*IDENTITY); service.finish(IDENTITY)
    assert service.read(*IDENTITY)['status'] == 'AVAILABLE' and copy.call_count == 1
    assert len(objects) == 1 and not (mirror/'weights.bin').exists()


@pytest.mark.parametrize('fresh_code,expected,conflict', [(None, 74, False), (137, None, True)])
def test_cpu_exit_event_saved_before_upgrade_survives_provider_event_expiry(recovery, monkeypatch, fresh_code, expected, conflict):
    service, *_ = recovery
    cpu(recovery, monkeypatch); service.begin(*IDENTITY)
    name = service.read(*IDENTITY)['scheduler_name']
    event = {'reason': 'ExitedWithCode', 'message': "pod <pt-"+'a'*32+"-worker-0>'s container <pytorch> exited with code 74 | time: <2026-10-10T00:00:00Z>"}
    service.update(IDENTITY, cpu_profile={'workspace': 'ws'}, cpu_diagnostics={'scheduler_name': name, 'events': [event]})
    rest = provider(); rest.describe.return_value['uid'] = 'a'*32
    if fresh_code is not None: rest.describe.return_value['exit_code'] = fresh_code
    rest.request.return_value = {'events': []}
    monkeypatch.setattr('ml_exp_server.backends.sensecore_results.SenseCoreREST.from_environment', lambda: rest)
    value = service.result_job_diagnostics(IDENTITY, refresh=True)
    assert (value['job_exit_code'], value['exit_code_conflict']) == (expected, conflict)
    value['logs'] = {'status': 'AVAILABLE', 'lines': ['retained diagnostic']}
    service.update(IDENTITY, cpu_diagnostics=value)
    value = service.result_job_diagnostics(IDENTITY, refresh=True)
    assert (value['job_exit_code'], value['exit_code_conflict']) == (expected, conflict)
    assert value['logs']['lines'] == ['retained diagnostic']
    rest.logs.return_value = {'text': 'fresh log', 'available': True}
    assert service.result_job_diagnostics(IDENTITY, refresh=True)['logs']['lines'] == ['fresh log']
    service.update(IDENTITY, cpu_diagnostics={'scheduler_name': name, 'events': []})
    assert not service.result_job_diagnostics(IDENTITY, refresh=True)['exit_evidence']


def test_diagnostics_and_repair_http_authentication_never_launch_compute(recovery, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from ml_exp_server.api.app import create_app
    service, root, token, _, _, _, row = recovery
    data = archive({'metrics.jsonl': b'old'})
    receipt = service.objects.receive(*IDENTITY, token, io.BytesIO(data), len(data))
    files = output(root); (files/'metrics.jsonl').write_bytes(b'old')
    bearer = tmp_path/'bearer'; bearer.write_text('a'*40); bearer.chmod(0o600)
    service.runtime.config.http_auth.bearer_token_file = str(bearer)
    with TestClient(create_app(service.runtime.config, poll=False)) as api:
        api.app.state.runtime.index.upsert_run(row)
        submit = Mock(wraps=api.app.state.submit_job); api.app.state.submit_job = submit
        base = '/api/runs/demo/run-a/attempts/attempt-001'
        headers = {'Authorization': 'Bearer '+'a'*40, 'X-ML-Expd-Client-Protocol': '2'}
        assert api.get(base+'/diagnostics').status_code == 401
        assert api.get(base+'/diagnostics', headers={'Authorization': 'Bearer '+token}).status_code == 401
        assert api.get(base+'/diagnostics?refresh=true', headers=headers).status_code == 200
        assert api.post(base+'/collection/repair', json={'expected_sha256': receipt['sha256']}).status_code == 401
        assert api.post(base+'/collection/repair', json={'expected_sha256': 'wrong'}, headers=headers).status_code == 422
        assert api.post(base+'/collection/repair', json={'expected_sha256': receipt['sha256']}, headers=headers).json()['archive']['file_count'] == 2
        # Repair now owns a bounded local collection worker; this does not
        # submit a scheduler/CPU/GPU job. Preserve authentication and results.
        submit.assert_called_once()
        assert submit.call_args.kwargs['executor'] is api.app.state.job_executors['collection']
        assert submit.call_args.args[0].func.__name__ == 'repair'


def test_required_outputs_freeze_with_run_and_attempt_transfer(container_client, tmp_path):
    from ml_exp_server.container_controller import Controller
    from ml_exp_server.results.artifact_store import ArtifactStore
    api = container_client
    ready = ready_runtime(api)
    cfg = api.app.state.runtime.config
    config_file = tmp_path/'objects.json'
    config_file.write_text(json.dumps({'endpoint': 'http://localhost:3900', 'bucket': 'results',
        'access_key': 'test', 'secret_key': 'test', 'public_transfer_base': 'https://example/api/artifact-transfers'}))
    cfg.container_execution.artifact_store_file = str(config_file)
    required = ['weights/*.pt', 'summary.json']
    response = api.post('/api/projects/demo/runs', json={'run_id': 'required', 'runtime_id': ready['runtime_id'],
                        'executor': 'gpu', 'outputs': ['metrics.jsonl'], 'required_outputs': required})
    assert response.status_code == 200, response.text
    project = Path(api.app.state.runtime.project('demo').base_dir)
    value = yaml.safe_load((project/'experiments/campaigns/run-required.yaml').read_text())
    assert value['runs'][0]['required_outputs'] == required
    assert value['runs'][0]['outputs'] == ['metrics.jsonl', *required]
    value['local_root'] = str(cfg.project_run_root_path('demo'))
    controller = Controller(value, 'required', 'attempt-001')
    controller.prepare()
    assert controller.store.load_manifest()['resolved_config']['required_outputs'] == required
    controller.dispatch_command(controller.store.load_attempt('attempt-001'))
    store = ArtifactStore(config_file, cfg.project_registry_root_path())
    with store.record('demo', 'required', 'attempt-001') as (_, transfer):
        assert transfer['required_outputs'] == required
