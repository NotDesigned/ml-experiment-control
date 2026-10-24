"""Saved observation stays exact even after source removal or newer Attempts."""
from copy import deepcopy
import json

import pytest

from ml_exp_client.api import ClientError
from ml_exp_client.observation import observation_reference, observation_server, resolve_observation


BASE = 'https://api.example/prefix'
AID = 'action-' + 'a' * 16
PID = 'preparation-' + 'b' * 32


def experiment(**changes):
    return {'server': BASE, 'project': 'demo', 'run_id': 'train', 'config_sha256': 'b' * 64,
            'submission': {'submission_id': AID, 'attempt_id': 'attempt-001'}, **changes}


def operation(kind='retry', **changes):
    plan = {'action_id': AID, 'run_id': 'train', 'attempt_id': 'attempt-002' if kind == 'retry' else 'attempt-001',
            'operation': 'RETRY_ATTEMPT' if kind == 'retry' else 'CANCEL_RUN',
            'scope': {'project': 'demo', 'scope_type': 'attempt', 'object_id': 'train::attempt-001'},
            'intent_digest': 'sha256:' + 'c' * 64, 'gate_bundle_digest': 'sha256:' + 'd' * 64,
            'execution': {'status': 'PREPARED'}}
    return {'contract': 'training-operation.v1', 'server': BASE, 'project': 'demo', 'run_id': 'train',
            'operation': kind, 'source_attempt_id': 'attempt-001', 'request': {}, 'plan': plan, 'action': deepcopy(plan), **changes}


def write(path, value):
    path.write_text(json.dumps(value))
    return path


class API:
    base = BASE

    def __init__(self, value=None):
        self.value = value
        self.calls = []

    def call(self, endpoint, **kwargs):
        assert not kwargs, 'identity resolution must only GET original records'
        self.calls.append(endpoint)
        if endpoint == '/api/submissions/' + AID:
            return self.value or {'submission_id': AID, 'project': 'demo', 'run_id': 'train', 'attempt_id': 'attempt-001'}
        assert endpoint == '/api/actions/' + AID
        return deepcopy(self.value)


class PreparationAPI(API):
    def __init__(self):
        super().__init__()
        self.preparation = {'project': 'demo', 'run_id': 'train', 'preparation_id': PID, 'status': 'READY',
            'submission': {'project': 'demo', 'run_id': 'train', 'submission_id': AID, 'attempt_id': 'attempt-001'}}

    def call(self, endpoint, **kwargs):
        if endpoint == '/api/projects/demo/experiment-preparations/' + PID:
            assert not kwargs
            self.calls.append(endpoint)
            return deepcopy(self.preparation)
        return super().call(endpoint, **kwargs)


def test_experiment_resolves_without_source_or_config_and_does_not_change_state(tmp_path):
    path = write(tmp_path / 'state.json', experiment())
    before = path.read_bytes()
    api = API()
    result = resolve_observation(api, state_path=path)
    assert (result['server'], result['project'], result['run_id'], result['attempt_id']) == (BASE, 'demo', 'train', 'attempt-001')
    assert api.calls == ['/api/submissions/' + AID]
    assert path.read_bytes() == before and result['state_kind'] == 'experiment'
    assert observation_reference(path)['attempt_id'] == 'attempt-001'


@pytest.mark.parametrize('kind', ['retry', 'cancel'])
def test_operation_reads_its_reviewed_target_not_source_or_current(tmp_path, kind):
    saved = operation(kind)
    path = write(tmp_path / 'plan.json', saved)
    api = API(saved['plan'])
    result = resolve_observation(api, state_path=path)
    assert result['attempt_id'] == saved['plan']['attempt_id']
    assert api.calls == ['/api/actions/' + AID] and result['action'] == saved['plan']


def test_unknown_preparation_only_exposes_server_for_action_attachment(tmp_path):
    path = write(tmp_path / 'unknown.json', operation(plan=None, action=None))
    assert observation_server(path) == BASE
    api = API()
    with pytest.raises(ClientError, match='unknown'):
        resolve_observation(api, state_path=path)
    assert not api.calls


@pytest.mark.parametrize('field,value', [('server', 'https://foreign.example'), ('project', 'foreign'),
    ('run_id', 'foreign'), ('server', None), ('project', ''), ('run_id', 123)])
def test_conflicting_or_invalid_saved_identity_never_makes_http(tmp_path, field, value):
    path = write(tmp_path / 'state.json', experiment(**{field: value}))
    api = API()
    with pytest.raises(ClientError):
        resolve_observation(api, project='demo', run='train', state_path=path)
    assert not api.calls


@pytest.mark.parametrize('kwargs', [{'project': 'other'}, {'run': 'other'}, {'attempt': 'attempt-009'}])
def test_explicit_arguments_cannot_rebind_saved_reference(tmp_path, kwargs):
    api = API()
    with pytest.raises(ClientError, match='differ'):
        resolve_observation(api, state_path=write(tmp_path / 'state.json', experiment()), **kwargs)
    assert not api.calls


def test_legacy_missing_identity_requires_explicit_context(tmp_path):
    saved = experiment()
    for key in ('server', 'project', 'run_id'):
        saved.pop(key)
    path = write(tmp_path / 'legacy.json', saved)
    api = API()
    with pytest.raises(ClientError, match='provide --project'):
        resolve_observation(api, state_path=path)
    assert not api.calls and observation_server(path) is None
    assert resolve_observation(api, 'demo', 'train', state_path=path)['attempt_id'] == 'attempt-001'


def test_preparation_only_state_has_no_guessed_attempt_and_url_slashes_normalize(tmp_path):
    api = API()
    path = write(tmp_path / 'state.json', experiment(server=BASE + '/', submission=None))
    result = resolve_observation(api, state_path=path)
    assert result['attempt_id'] is None and not api.calls
    assert resolve_observation(api, 'demo', 'train', 'attempt-004')['attempt_id'] == 'attempt-004'
    assert resolve_observation(api, 'demo', 'train')['attempt_id'] is None


def test_original_preparation_can_recover_its_explicit_submission_without_source_or_state_changes(tmp_path):
    path = write(tmp_path / 'state.json', experiment(submission=None, preparation={'preparation_id': PID}))
    before = path.read_bytes()
    api = PreparationAPI()
    result = resolve_observation(api, state_path=path)
    assert result['attempt_id'] == 'attempt-001' and result['submission']['submission_id'] == AID
    assert result['preparation'] == api.preparation
    assert api.calls == ['/api/projects/demo/experiment-preparations/' + PID, '/api/submissions/' + AID]
    assert path.read_bytes() == before


@pytest.mark.parametrize('changes', [{'project': 'other'}, {'run_id': 'other'}, {'preparation_id': 'preparation-' + 'c' * 32}])
def test_recovered_preparation_is_bound_to_original_identity(tmp_path, changes):
    api = PreparationAPI(); api.preparation.update(changes)
    with pytest.raises(ClientError, match='Preparation identity'):
        resolve_observation(api, state_path=write(tmp_path / 'state.json',
            experiment(submission=None, preparation={'preparation_id': PID})))
    assert len(api.calls) == 1


@pytest.mark.parametrize('changes', [{'project': 'other'}, {'run_id': 'other'}, {'attempt_id': 'current'},
                                     {'submission_id': 'wrong'}])
def test_recovered_nested_submission_cannot_change_exact_scope(tmp_path, changes):
    api = PreparationAPI(); api.preparation['submission'].update(changes)
    with pytest.raises(ClientError):
        resolve_observation(api, state_path=write(tmp_path / 'state.json',
            experiment(submission=None, preparation={'preparation_id': PID})))
    assert len(api.calls) == 1


def test_recovered_submission_rejects_an_explicit_different_attempt(tmp_path):
    api = PreparationAPI()
    with pytest.raises(ClientError, match='Run/Attempt'):
        resolve_observation(api, attempt='attempt-002', state_path=write(tmp_path / 'state.json',
            experiment(submission=None, preparation={'preparation_id': PID})))
    assert len(api.calls) == 1


@pytest.mark.parametrize('changed', ['submission_id', 'attempt_id'])
def test_saved_preparation_submission_also_remains_frozen(tmp_path, changed):
    api = PreparationAPI()
    original = deepcopy(api.preparation)
    api.preparation['submission'][changed] = 'action-' + 'd' * 16 if changed == 'submission_id' else 'attempt-002'
    with pytest.raises(ClientError, match='saved exact'):
        resolve_observation(api, state_path=write(tmp_path / 'state.json', experiment(submission=None, preparation=original)))
    assert len(api.calls) == 1


@pytest.mark.parametrize('status', ['EXECUTING', 'READY'])
def test_preparation_without_submission_never_resolves_run_current(status, tmp_path):
    api = PreparationAPI(); api.preparation.pop('submission'); api.preparation['status'] = status
    result = resolve_observation(api, state_path=write(tmp_path / 'state.json',
        experiment(submission=None, preparation={'preparation_id': PID})))
    assert result['attempt_id'] is None and result['submission'] is None
    assert api.calls == ['/api/projects/demo/experiment-preparations/' + PID]


@pytest.mark.parametrize('where', ['saved_submission', 'response'])
def test_submission_context_is_checked_independently(tmp_path, where):
    saved = experiment()
    if where == 'saved_submission':
        saved['submission']['project'] = 'other'
    api = API({'project': 'other', 'run_id': 'train', 'submission_id': AID, 'attempt_id': 'attempt-001'})
    with pytest.raises(ClientError, match='Submission'):
        resolve_observation(api, state_path=write(tmp_path / 'state.json', saved))
    assert len(api.calls) == (0 if where == 'saved_submission' else 1)


@pytest.mark.parametrize('field,value', [('action_id', 'action-' + 'e' * 16), ('run_id', 'other'),
    ('attempt_id', 'attempt-003'), ('scope', {}), ('operation', 'CANCEL_RUN'),
    ('intent_digest', 'sha256:' + 'f' * 64), ('gate_bundle_digest', None), ('execution', None)])
def test_live_action_cannot_replace_saved_identity(tmp_path, field, value):
    saved = operation()
    action = {**saved['plan'], field: value}
    api = API(action)
    with pytest.raises(ClientError, match='Action identity'):
        resolve_observation(api, state_path=write(tmp_path / 'plan.json', saved))
    assert api.calls == ['/api/actions/' + AID]


@pytest.mark.parametrize('case', ['wrong_target', 'cancel_target', 'request_target', 'request_type', 'changed_snapshot',
                                 'bad_action_id', 'missing_source', 'bad_operation', 'missing_digest'])
def test_invalid_operation_does_not_make_http(tmp_path, case):
    saved = operation()
    if case == 'wrong_target': saved['plan']['attempt_id'] = 'attempt-001'
    if case == 'cancel_target': saved['operation'] = 'cancel'; saved['plan']['operation'] = 'CANCEL_RUN'
    if case == 'request_target': saved['request'] = {'new_attempt_id': 'attempt-004'}
    if case == 'request_type': saved['request'] = None
    if case == 'changed_snapshot': saved['action']['run_id'] = 'foreign'
    if case == 'bad_action_id': saved['plan']['action_id'] = 'not-an-action'
    if case == 'missing_source': saved.pop('source_attempt_id')
    if case == 'bad_operation': saved['operation'] = 'delete'
    if case == 'missing_digest': saved['plan']['intent_digest'] = None
    api = API()
    with pytest.raises(ClientError):
        resolve_observation(api, state_path=write(tmp_path / 'plan.json', saved))
    assert not api.calls


@pytest.mark.parametrize('value', [[], {'runtime_id': 'runtime.' + 'a' * 64},
    experiment(submission=[]), experiment(preparation=True), experiment(submission={'submission_id': AID}),
    experiment(submission={'submission_id': 'wrong', 'attempt_id': 'attempt-001'})])
def test_malformed_or_unrelated_state_is_not_inferred(tmp_path, value):
    api = API()
    with pytest.raises(ClientError):
        resolve_observation(api, state_path=write(tmp_path / 'bad.json', value))
    assert not api.calls


@pytest.mark.parametrize('exists', [True, False])
def test_unreadable_state_has_a_safe_client_error(tmp_path, exists):
    path = tmp_path / 'bad.json'
    if exists: path.write_text('secret invalid JSON')
    with pytest.raises(ClientError, match='cannot read observation') as failure:
        observation_reference(path)
    assert 'secret' not in str(failure.value)
