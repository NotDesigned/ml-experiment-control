"""Source-free observations and explicit reviewed retries share one backend contract."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from ml_exp_client.cli import main, experiment_summary
from ml_exp_client.commands import query_status, status_actions
from ml_exp_client.api import ClientError
from ml_exp_client.state import save


ORIGINAL = 'action-' + 'a' * 16
RETRY = 'action-' + 'b' * 16


class Portal:
    def __init__(self, backend='slurm'):
        self.base = 'https://api.example/ml-expd'
        self.calls = []
        self.backend = backend
        self.action = {'action_id': RETRY, 'operation': 'RETRY_ATTEMPT',
            'scope': {'project': 'demo', 'scope_type': 'attempt', 'object_id': 'trial::attempt-001'},
            'run_id': 'trial', 'attempt_id': 'attempt-002', 'ready': True,
            'intent_digest': 'sha256:' + 'c' * 64, 'gate_bundle_digest': 'sha256:' + 'd' * 64,
            'preflight_summary': {'requested_gpu_hours': 1, 'max_gpu_hours': 2,
                                  'resource_approval': 'budget_cap', 'backend': {'kind': backend}},
            'gates': [{'name': 'budget', 'status': 'PASS'}], 'execution': {'status': 'PREPARED'}}

    def negotiate(self):
        self.calls.append(('health', {}))
        return {'capabilities': []}

    def call(self, path, **kwargs):
        self.calls.append((path, deepcopy(kwargs)))
        if path == '/api/submissions/' + ORIGINAL:
            return {'project': 'demo', 'run_id': 'trial', 'attempt_id': 'attempt-001',
                    'submission_id': ORIGINAL, 'status': 'VERIFIED'}
        if '/logs?' in path or '/metrics?' in path:
            return {'project': 'demo', 'run_id': 'trial', 'attempt_id': 'attempt-001', 'records': []}
        if path.endswith('/retry'):
            return {'action': deepcopy(self.action)}
        if path == '/api/actions/authorize':
            self.action['execution']['status'] = 'AUTHORIZED'
        elif path == '/api/actions/execute':
            self.action['execution']['status'] = 'EXECUTING'
        elif path != '/api/actions/' + RETRY:
            raise AssertionError(path)
        return deepcopy(self.action)


def source_state(path):
    save(path, {'server': 'https://api.example/ml-expd', 'project': 'demo', 'run_id': 'trial',
                'config_sha256': 'e' * 64, 'source': '/already-deleted-source',
                'submission': {'submission_id': ORIGINAL, 'attempt_id': 'attempt-001'}})
    return path


@pytest.mark.parametrize('command', ['logs', 'metrics', 'download'])
def test_saved_state_reads_need_no_source_or_repeated_ids(tmp_path, monkeypatch, capsys, command):
    api = Portal()
    original = source_state(tmp_path / 'experiment.json')
    before = original.read_bytes()
    urls = []
    monkeypatch.delenv('ML_EXPD_API_URL', raising=False)
    monkeypatch.setattr('ml_exp_client.cli.Client', lambda url, token: (urls.append(url) or api))
    downloads = []
    monkeypatch.setattr('ml_exp_client.commands.download', lambda client, project, run, attempt, out, **kwargs:
                        downloads.append((project, run, attempt, out)) or {'downloaded': True})
    argv = [command, '--state', str(original)]
    if command == 'download':
        argv += ['--out', str(tmp_path / 'results')]
    assert main(argv) == 0
    json.loads(capsys.readouterr().out)
    assert urls == [api.base] and original.read_bytes() == before
    assert not any(kwargs for _, kwargs in api.calls)
    assert downloads == ([('demo', 'trial', 'attempt-001', tmp_path / 'results')] if command == 'download' else [])


def test_conflicting_server_is_rejected_before_authentication(tmp_path, monkeypatch, capsys):
    original = source_state(tmp_path / 'experiment.json')
    monkeypatch.setattr('ml_exp_client.cli.Client', lambda *args: pytest.fail('must not send credentials'))
    assert main(['--url', 'https://other.example', 'logs', '--state', str(original)]) == 2
    assert 'no request was sent' in capsys.readouterr().err


@pytest.mark.parametrize('backend', ['slurm', 'sensecore', 'pai-dlc'])
def test_retry_from_experiment_has_same_review_and_no_replay_for_all_backends(tmp_path, monkeypatch, capsys, backend):
    api = Portal(backend)
    monkeypatch.delenv('ML_EXPD_API_URL', raising=False)
    monkeypatch.setattr('ml_exp_client.cli.Client', lambda *args: api)
    original = source_state(tmp_path / 'experiment.json')
    before = original.read_bytes()
    plan = tmp_path / 'retry.json'
    assert main(['retry', '--from-state', str(original), '--state', str(plan), '--max-gpu-hours', '2']) == 0
    output = capsys.readouterr().out
    assert 'attempt-001 → attempt-002' in output and 'no checkpoint is selected or changed' in output
    assert [p for p, kw in api.calls if kw] == ['/api/attempts/demo/trial%3A%3Aattempt-001/retry']
    assert original.read_bytes() == before
    assert main(['retry', '--state', str(plan), '--execute', '--json']) == 0
    assert json.loads(capsys.readouterr().out)['action']['execution']['status'] == 'EXECUTING'
    api.calls.clear()
    assert main(['retry', '--state', str(plan), '--execute']) == 0
    capsys.readouterr()
    assert api.calls == [('health', {}), ('/api/actions/' + RETRY, {})]


def test_retry_cannot_overwrite_source_state(tmp_path, monkeypatch, capsys):
    api = Portal()
    monkeypatch.delenv('ML_EXPD_API_URL', raising=False)
    monkeypatch.setattr('ml_exp_client.cli.Client', lambda *args: api)
    original = source_state(tmp_path / 'experiment.json')
    before = original.read_bytes()
    assert main(['retry', '--from-state', str(original), '--state', str(original), '--max-gpu-hours', '2']) == 2
    assert 'separate' in capsys.readouterr().err
    assert original.read_bytes() == before and api.calls == [('health', {})]


@pytest.mark.parametrize('command', ['logs', 'metrics', 'download'])
def test_preparation_without_attempt_never_resolves_an_unrelated_current_attempt(tmp_path, monkeypatch, capsys, command):
    api = Portal()
    monkeypatch.delenv('ML_EXPD_API_URL', raising=False)
    monkeypatch.setattr('ml_exp_client.cli.Client', lambda *args: api)
    path = source_state(tmp_path / 'experiment.json')
    value = json.loads(path.read_text()); value.pop('submission'); save(path, value)
    argv = [command, '--state', str(path)]
    if command == 'download': argv += ['--out', str(tmp_path / 'results')]
    assert main(argv) == 2
    assert 'no bound Attempt' in capsys.readouterr().err
    assert api.calls == [('health', {})]


def compact(state='RUNNING', publication='NOT_COLLECTED'):
    return {'project': 'demo', 'run_id': 'trial', 'attempt_id': 'attempt-001',
            'scheduler': {'state': state}, 'training': {'status': 'UNKNOWN'},
            'publication': {'status': publication, 'download_available': False},
            'verification': {'status': 'PENDING'}}


@pytest.mark.parametrize('json_output', [False, True])
def test_watch_pins_original_attempt_and_has_bounded_client_wait(monkeypatch, capsys, json_output):
    calls, ticks = [], [0.0]
    def read(client, health, project, run, attempt, **kwargs):
        calls.append((project, run, attempt, kwargs))
        return compact()
    monkeypatch.setattr('ml_exp_client.commands.run_status', read)
    monkeypatch.setattr('ml_exp_client.commands.time.monotonic', lambda: ticks[0])
    monkeypatch.setattr('ml_exp_client.commands.time.sleep', lambda n: ticks.__setitem__(0, ticks[0] + n))
    args = SimpleNamespace(watch=True, seconds=6, project='demo', run='trial', attempt=None,
                           state=None, details=False, json=json_output)
    result = query_status(args, Portal(), {})
    assert [c[2] for c in calls] == [None, 'attempt-001', 'attempt-001']
    assert ticks[0] == 6 and result['scheduler']['state'] == 'RUNNING'
    output = capsys.readouterr().err
    assert ('"scheduler"' in output) if json_output else ('Scheduler: RUNNING' in output)


@pytest.mark.parametrize('seconds', [-1, float('nan'), float('inf')])
def test_invalid_watch_duration_does_not_query(seconds):
    with pytest.raises(ClientError, match='finite and nonnegative'):
        query_status(SimpleNamespace(watch=True, seconds=seconds), Portal(), {})


def test_status_actions_distinguish_released_missing_and_verified_results():
    value = compact('SUCCEEDED', 'RELEASED')
    assert status_actions(value, 'https://api.example', None)[0]['kind'] == 'USE_LOCAL_RESULTS'
    value['publication'].update(status='AVAILABLE', download_available=True)
    assert status_actions(value, 'https://api.example', None)[0]['argv'][-2:] == ['--out', 'results/trial']
    value = compact('FAILED')
    assert [a['kind'] for a in status_actions(value, 'https://api.example', None)] == ['READ_LOGS', 'REVIEW_RETRY']
    value['attempt_id'] = None
    assert status_actions(value, 'https://api.example', 'old.state')[0]['argv'][-2:] == ['--state', 'old.state']


def test_experiment_summary_does_not_dump_workflow_or_claim_training():
    args = SimpleNamespace(state='trial.state', execute=False)
    result = {'project': 'demo', 'run_id': 'trial', 'submission': {'status': 'PREPARED', 'attempt_id': 'attempt-001'}}
    text = experiment_summary(result, state_file=args.state, executed=args.execute)
    assert 'not been authorized by this command' in text and 'status --state trial.state' in text
    args.execute = True
    result.update(result={'scheduler_state': 'SUCCEEDED'}, download={'status': 'VERIFIED'})
    assert 'Scheduler: SUCCEEDED' in experiment_summary(result, state_file=args.state, executed=args.execute)


def test_future_retry_target_without_scheduler_job_does_not_suggest_nonexistent_logs():
    value = compact('NOT_SUBMITTED')
    value['attempt_id'] = 'attempt-002'
    actions = status_actions(value, 'https://api.example', 'reviewed-retry.state')
    assert [item['kind'] for item in actions] == ['OBSERVE_PREPARATION']
    assert actions[0]['argv'][-3:] == ['status', '--state', 'reviewed-retry.state']
