"""Exact CPU recovery evidence, delayed logs and safe missing exit codes."""
import json
from unittest.mock import Mock

import pytest

from experiment_control.backends.sensecore_rest import RESTError
from ml_exp_server.backends.sensecore_results import SenseCoreResultOperations
from tests.test_artifact_store import storage
from tests.test_result_collection import IDENTITY, recovery, cpu


def provider():
    rest = Mock()
    rest.config = {'access_key_id': 'private-access', 'access_key_secret': 'private-secret'}
    rest.describe.return_value = {'state': 'FAILED'}
    rest.workers.return_value = [{'containers': [{'name': 'worker'}]}]
    rest.jobs_url.return_value = 'https://api.sensecoreapi.cn/jobs'
    rest.request.return_value = {'events': [{'reason': 'SuccessfulDelete', 'type': 'Normal',
                                          'message': 'private-secret Bearer unknown https://host/path?cap=abc ' + 'A' * 256}]}
    rest.logs.return_value = {'text': '', 'available': False, 'source': 'offline', 'historical': True,
                             'exit_code': 0, 'unavailable_reason': 'OFFLINE_LOGS_EMPTY'}
    return rest


def result():
    return {'scheduler_name': 'cpu-job', 'job_state': None, 'job_exit_code': None,
            'exit_code_conflict': False, 'events': [],
            'logs': {'status': 'PENDING', 'source': None, 'historical': None,
                     'truncated': False, 'lines': [], 'unavailable_reason': None}, 'errors': []}


def read(rest):
    return SenseCoreResultOperations.collect_result_job_diagnostics(
        rest, {'workspace': 'ws'}, 'cpu-job', ['private-secret', 'private-access', None, ''], result())


def test_missing_container_exit_code_and_empty_delayed_logs_remain_unknown():
    rest = provider()
    value = read(rest)
    assert value['job_state'] == 'FAILED' and value['job_exit_code'] is None
    assert value['logs']['status'] == 'PENDING' and not value['logs']['lines']
    # exit_code on the log query reports request success, never container exit.
    assert rest.logs.call_args.args[-1] == 200
    public = json.dumps(value)
    assert 'private-secret' not in public and 'Bearer unknown' not in public and 'cap=abc' not in public
    assert 'A' * 256 not in public and '[ENCODED REDACTED]' in public
    assert not rest.create.called and not rest.stop.called


@pytest.mark.parametrize('job,workers,expected,conflict', [
    ({'state': 'FAILED', 'exit_code': 74}, [], 74, False),
    ({'state': 'FAILED'}, [{'containers': [{'state': {'terminated': {'exitCode': 137}}}]}], 137, False),
    ({'state': 'FAILED', 'exit_code': 74}, [{'status': {'exit_code': 137}, 'containers': []}], None, True),
    ({'state': 'FAILED', 'exit_code': True}, [{'containers': [{'exit_code': '74'}]}], None, False),
    ({'state': 'unknown', 'exitCode': 999}, [{'last_state': 'not-a-mapping', 'containers': None}], None, False),
    ({'state': 'SUCCEEDED'}, [{'containers': [None, {'exit_code': 0, 'state': None}]}], 0, False),
])
def test_only_explicit_matching_numeric_container_codes_are_reported(job, workers, expected, conflict):
    rest = provider(); rest.describe.return_value = job; rest.workers.return_value = workers
    value = read(rest)
    assert value['job_exit_code'] == expected and value['exit_code_conflict'] is conflict
    if job['state'] == 'unknown': assert value['job_state'] is None


@pytest.mark.parametrize('operation', ['describe', 'workers', 'events', 'logs'])
@pytest.mark.parametrize('error', [RESTError('GET opaque', status=403), ValueError('private-secret')])
def test_query_errors_are_safe_and_do_not_submit_or_stop_jobs(operation, error):
    rest = provider()
    getattr(rest, {'describe': 'describe', 'workers': 'workers', 'events': 'request', 'logs': 'logs'}[operation]).side_effect = error
    value = read(rest)
    assert value['errors'][0]['operation'] == operation
    assert 'private-secret' not in json.dumps(value)
    assert not rest.create.called and not rest.stop.called
    if operation in {'describe', 'logs'}: assert value['logs']['status'] == 'UNAVAILABLE'


@pytest.mark.parametrize('logs,status', [
    ({'text': 'private-secret\nBearer capability\nunit loss: 1.2', 'historical': True, 'source': 'offline', 'truncated': True}, 'AVAILABLE'),
    ({'text': '', 'available': False, 'error': {'operation': 'private-secret'}, 'unavailable_reason': 'OFFLINE_LOGS_UNAVAILABLE'}, 'UNAVAILABLE'),
    ({'text': [], 'unavailable_reason': None}, 'PENDING'),
    ({'text': 'data', 'available': False}, 'PENDING'),
])
def test_live_or_offline_log_evidence_is_bounded_and_redacted(logs, status):
    rest = provider(); rest.logs.return_value = logs
    rest.request.return_value = {'events': [None, {'reason': None, 'type': 17, 'message': 100},
                                          {'reason': 'Failed', 'type': 'Warning', 'message': 'B ' * 3000}]}
    value = read(rest)
    assert value['logs']['status'] == status
    assert 'private-secret' not in json.dumps(value) and 'Bearer capability' not in json.dumps(value)
    assert len(value['events'][-1]['message']) == 2048
    assert value['events'][0] == {'reason': '', 'type': '', 'message': ''}


def test_diagnostic_query_rejects_invalid_job_identity():
    with pytest.raises(ValueError):
        SenseCoreResultOperations.collect_result_job_diagnostics(provider(), {}, '../job', [], result())


@pytest.mark.parametrize('operation,value', [
    ('describe', None), ('describe', {'state': []}),
    ('workers', None), ('workers', [None]), ('events', []), ('events', {'events': None}), ('logs', None),
])
def test_malformed_provider_observations_fail_as_safe_missing_evidence(operation, value):
    rest = provider()
    getattr(rest, {'describe': 'describe', 'workers': 'workers', 'events': 'request', 'logs': 'logs'}[operation]).return_value = value
    observed = read(rest)
    assert observed['job_exit_code'] is None
    if operation == 'describe' and isinstance(value, dict):
        assert observed['job_state'] is None
    else:
        assert observed['errors'][0]['operation'] == operation
    assert not rest.create.called and not rest.stop.called


def test_service_keeps_cache_bound_to_current_cpu_job_and_refreshes_readonly(recovery, monkeypatch):
    service, _, _, _, _, _, _ = recovery
    assert service.result_job_diagnostics(IDENTITY)['logs']['status'] == 'NOT_APPLICABLE'
    cpu(recovery, monkeypatch)
    service.begin(*IDENTITY)
    assert service.result_job_diagnostics(IDENTITY)['logs']['status'] == 'NOT_APPLICABLE'
    service.update(IDENTITY, cpu_profile={'workspace': 'ws'}, job_state='FAILED')
    baseline = service.result_job_diagnostics(IDENTITY)
    assert baseline['logs']['status'] == 'PENDING'
    service.update(IDENTITY, cpu_diagnostics={'scheduler_name': 'old-job', 'job_exit_code': 1})
    assert service.result_job_diagnostics(IDENTITY)['job_exit_code'] is None
    cached = {**baseline, 'job_exit_code': 74}
    service.update(IDENTITY, cpu_diagnostics=cached)
    assert service.result_job_diagnostics(IDENTITY) == cached
    rest = provider()
    monkeypatch.setattr('ml_exp_server.backends.sensecore_results.SenseCoreREST.from_environment', lambda: rest)
    value = service.result_job_diagnostics(IDENTITY, refresh=True)
    assert value['job_exit_code'] is None and value['logs']['status'] == 'PENDING'
    assert not rest.create.called and not rest.stop.called
    assert service.result_job_diagnostics(IDENTITY) == cached


def test_service_missing_private_configuration_returns_safe_unavailable(recovery, monkeypatch):
    service, _, _, _, _, _, _ = recovery
    cpu(recovery, monkeypatch); service.begin(*IDENTITY); service.update(IDENTITY, cpu_profile={'workspace': 'ws'})
    def missing(): raise RESTError('configuration')
    monkeypatch.setattr('ml_exp_server.backends.sensecore_results.SenseCoreREST.from_environment', missing)
    assert service.result_job_diagnostics(IDENTITY, refresh=True)['logs']['unavailable_reason'] == 'CPU_CONFIGURATION_UNAVAILABLE'
