"""Gateway startup, pre-GPU checks and data preparation preserve frozen Runs."""
import copy
import json
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
import yaml

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.container_controller import Controller
from ml_exp_server.container_execution import ContainerExecutionService
from ml_exp_server.executor_capabilities import declaration
from ml_exp_server import data_preparation as data, managed_worker as worker
from tests.test_container_api import archive, client, import_source, runtime
from tests.test_script_data_preparation import definition
from tests.test_sensecore_data_workflow import stored


@pytest.fixture
def gateway(client, stored):
    config = Path(client.app.state.runtime.config.container_execution.profiles_file)
    doc = yaml.safe_load(config.read_text())
    origin = urlsplit(json.loads(stored[0].read_text())['public_transfer_base'])
    relay = {'endpoint': 'tcp://127.0.0.1:18443', 'origin': f'https://{origin.netloc}'}
    doc['executors']['gpu']['backend']['api_relay'] = relay
    doc['executors']['gpu']['compute_internet'] = False
    config.write_text(yaml.safe_dump(doc))
    source = import_source(client, archive({'train.py': b'pass', 'download.py': b'pass',
        'Dockerfile': ('FROM registry.example/python@sha256:'+'a'*64+'\n').encode()}))
    bundle = runtime(client, source)
    return client, stored, bundle, config


def controller(gateway, **extra):
    api, _, bundle, _ = gateway
    result = api.post('/api/projects/demo/runs', json={'run_id': 'relay-run', 'runtime_id': bundle['runtime_id'], 'executor': 'gpu', **extra})
    assert result.status_code == 200, result.text
    root = Path(api.app.state.runtime.project('demo').base_dir)
    campaign = yaml.safe_load((root/'experiments/campaigns/run-relay-run.yaml').read_text())
    campaign['local_root'] = str(api.app.state.runtime.config.project_run_root_path('demo'))
    ctl = Controller(campaign, 'relay-run', 'attempt-001'); ctl.prepare()
    return ctl


@pytest.mark.parametrize("unsquash", [False, True])
def test_relay_frozen_before_launch_and_capabilities_do_not_leak_address(gateway, unsquash):
    config = gateway[3]
    doc = yaml.safe_load(config.read_text()); doc["executors"]["gpu"]["backend"]["apptainer_unsquash"] = unsquash
    config.write_text(yaml.safe_dump(doc))
    ctl = controller(gateway, data_preparation={'script': 'download.py'})
    dispatch = ctl.dispatch_command(ctl.store.load_attempt('attempt-001'))
    assert 'ML_EXPD_API_RELAY=tcp://127.0.0.1:18443' in dispatch[:4]
    from ml_exp_server.artifact_store import ArtifactStore
    store = ArtifactStore(gateway[1][0], Path(ctl.campaign['source_store']))
    with store.record('demo', 'relay-run', 'attempt-001') as (_, record):
        env = record['launch_manifest']['environment']
        assert env['ML_EXPD_DATA_PREPARATION_CACHED'] == '1'
        assert env['ML_EXPD_API_RELAY'] == ctl.environment('attempt-001')['ML_EXPD_API_RELAY']
    caps = gateway[0].get('/api/executors').json()['executors'][0]['capabilities']
    service = ContainerExecutionService(gateway[0].app.state.runtime)
    cap = service.profile_capabilities()['gpu']
    assert cap['network'] == {'compute_internet': False, 'api_transport': 'tcp_relay'}
    assert cap['data']['download_script'] == 'before_job'
    assert '127.0.0.1' not in json.dumps(cap)
    calls = []
    ctl.runner = SimpleNamespace(run=lambda args, **kwargs: calls.append((args, kwargs)))
    ctl.check_api_transport()
    assert calls[0][0][:6] == ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'cluster']
    assert calls[0][1]['timeout_seconds'] == 30
    assert json.loads(calls[0][1]['input_text'])['environment']['ML_EXPD_API_RELAY']
    ctl.prepare_gateway_data(require_cached=False)
    ctl.prepare_gateway_data(require_cached=True)
    for args, kwargs in calls[1:]:
        command = shlex.split(args[-1])
        assert 'apptainer' in command and '--nv' not in command and 'sbatch' not in command
        assert json.loads(kwargs['input_text']) == ctl.run['data_preparation']
    assert shlex.split(calls[-1][0][-1])[-1] == '1'
    assert ('--unsquash' in shlex.split(calls[-1][0][-1])) == unsquash
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(ctl.backend, 'stage', lambda *args: True)
        assert ctl.stage()
        def fail(*args, **kwargs): raise subprocess.CalledProcessError(1, ['ssh'])
        ctl.runner = SimpleNamespace(run=fail)
        with pytest.raises(subprocess.CalledProcessError): ctl.submit(None)
    finally: monkey.undo()


def test_relay_without_script_and_old_worker_are_checked(gateway, monkeypatch):
    ctl = controller(gateway)
    ctl.prepare_gateway_data(require_cached=True)  # No user script to execute.
    from ml_exp_server.executor_capabilities import validate_worker, ExecutionRequirements
    service = ContainerExecutionService(gateway[0].app.state.runtime)
    request = __import__('ml_exp_server.container_execution', fromlist=['RunRequest']).RunRequest(
        run_id='old', executor='gpu', runtime_id=gateway[2]['runtime_id'])
    with pytest.raises(ApplicationError, match='api-tcp-relay.v1'):
        validate_worker(request, {'capabilities': []}, service.profile_capabilities()['gpu'])
    cloud = copy.deepcopy(service.profiles()['cloud']); cloud['backend']['api_relay'] = ctl.run['backend']['api_relay']
    with pytest.raises(ApplicationError, match='Slurm gateway'):
        declaration(cloud, artifact_store=True, desktop=True)


@pytest.mark.parametrize('missing', [False, True])
def test_relay_must_match_the_configured_api(gateway, missing):
    api, _, bundle, config = gateway
    doc = yaml.safe_load(config.read_text()); doc['executors']['gpu']['backend']['api_relay']['origin'] = 'https://wrong.test'
    config.write_text(yaml.safe_dump(doc))
    if missing: api.app.state.runtime.config.container_execution.artifact_store_file = None
    result = api.post('/api/projects/demo/runs', json={'run_id': 'wrong', 'runtime_id': bundle['runtime_id'], 'executor': 'gpu'})
    assert result.status_code == 409 and result.headers['X-ML-Expd-Error-Code'] == 'EXECUTOR_TRANSPORT_INVALID'


def test_prepared_data_missing_never_runs_download_or_training(tmp_path, monkeypatch):
    workspace = tmp_path/'source'; spec = definition(workspace)
    with pytest.raises(ValueError, match='DATA_PREPARED_CACHE_MISSING'):
        data.prepare(spec, tmp_path/'cache', workspace=workspace, require_cached=True)
    tree, receipt = data.prepare(spec, tmp_path/'cache', workspace=workspace)
    assert data.prepare(spec, tmp_path/'cache', workspace=workspace, require_cached=True)[1]['cache_reused']
    root = tmp_path/'project/runs/trial/attempts/attempt-001/outputs'; root.mkdir(parents=True)
    for key, value in {'OUTPUT_DIR': str(root), 'ML_EXPD_UPLOAD_URL': 'https://example/api', 'ML_EXPD_UPLOAD_TOKEN': 'private',
        'ML_EXPD_UPLOAD_LIMIT': '2000000', 'ML_EXPD_DATA_PREPARATION': json.dumps(spec), 'ML_EXPD_DATA_PREPARATION_CACHED': '1'}.items(): monkeypatch.setenv(key, value)
    monkeypatch.setattr(worker, 'link_path', lambda *args: None)
    monkeypatch.setattr(worker, 'prepare_data', lambda spec, cache, **kwargs: data.prepare(spec, cache, workspace=workspace, **kwargs))
    monkeypatch.setattr(worker, 'upload', lambda *args: None)
    assert worker.main(['python3', '-c', 'raise SystemExit("training must not run")']) == 65
    assert json.loads((root/'data-preparation.json').read_text())['error'] == 'DATA_PREPARED_CACHE_MISSING'
