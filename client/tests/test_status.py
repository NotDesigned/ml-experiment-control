"""A compact read must not collapse scheduler, training and publication evidence."""
from copy import deepcopy
import json

import pytest

from ml_exp_client.api import ClientError
from ml_exp_client.cli import main
from ml_exp_client.status import run_status


HEALTH = {"capabilities": ["result-collection.v1", "execution-progress.v1"]}
PREPARATION = "preparation-" + "a" * 32
SUBMISSION = "action-" + "b" * 16


class Reader:
    base = "https://api.example"

    def __init__(self):
        self.calls = []
        self.run = {"project": "demo", "run_id": "trial", "scheduler_state": "SUCCEEDED",
                    "evidence": {"scheduler": {"attempt_id": "attempt-002", "state": "SUCCEEDED", "stale": False,
                         "as_of": "2026-10-13T00:00:00Z", "detail": {"reason": "resource allocation"}},
                         "process": {"attempt_id": "attempt-002", "state": "EXITED", "stale": False}}}
        self.attempts = {"project": "demo", "run_id": "trial", "current_attempt_id": "attempt-002",
            "attempts": [{"identity": "trial::" + item, "attempt_id": item, "state": "FAILED" if item.endswith("001") else "SUCCEEDED"}
                         for item in ("attempt-001", "attempt-002")]}
        self.collection = {"project": "demo", "run_id": "trial", "attempt_id": "attempt-002", "status": "NOT_COLLECTED",
                           "training": {"status": "UNKNOWN", "exit_code": None}, "verification": "PENDING", "download_available": False}
        self.preparation = {"project": "demo", "run_id": "trial", "preparation_id": PREPARATION,
                            "status": "READY", "phase": "READY"}
        self.submission = {"project": "demo", "run_id": "trial", "submission_id": SUBMISSION,
                           "attempt_id": "attempt-002", "status": "VERIFIED"}
        self.progress = {"submission_id": SUBMISSION, "run_id": "trial", "attempt_id": "attempt-002", "phase": "SUCCEEDED", "workload_evidence": {"process": {"state": "EXITED"}}}
        self.submissions = []

    def negotiate(self):
        return HEALTH

    def call(self, path, **kwargs):
        self.calls.append((path, kwargs))
        assert not kwargs, "status must only read"
        if "experiment-preparations/" in path:
            return deepcopy(self.preparation)
        if path == "/api/submissions/" + SUBMISSION:
            return deepcopy(self.submission)
        if path == "/api/experiments/demo/trial/submissions":
            return {"project": "demo", "run_id": "trial", "submissions": deepcopy(self.submissions)}
        if path.endswith("/progress"):
            return deepcopy(self.progress)
        if path.endswith("/collection"):
            if isinstance(self.collection, Exception):
                raise self.collection
            return deepcopy(self.collection)
        if path.endswith("/show"):
            return {"run_id": "trial", "attempt": {"attempt_id": "attempt-002"}, "failure_assessment": {},
                    "scope": {"project": "demo", "scope_type": "attempt", "object_id": "trial::attempt-002"}}
        if path.endswith("/attempts"):
            return deepcopy(self.attempts)
        if isinstance(self.run, Exception):
            raise self.run
        return deepcopy(self.run)


def state(path, **changes):
    value = {"server": Reader.base, "project": "demo", "run_id": "trial", "config_sha256": "c" * 64,
             "preparation": {"preparation_id": PREPARATION},
             "submission": {"submission_id": SUBMISSION, "attempt_id": "attempt-002"}}
    value.update(changes)
    path.write_text(json.dumps(value))
    return path


def test_scheduler_success_never_proves_training_or_download():
    api = Reader()
    result = run_status(api, HEALTH, "demo", "trial")
    assert result["scheduler"]["state"] == "SUCCEEDED"
    assert result["training"]["status"] == "UNKNOWN" and result["training"]["process_state"] == "EXITED"
    assert result["publication"]["status"] == "NOT_COLLECTED" and not result["publication"]["download_available"]
    assert result["verification"]["status"] == "PENDING" and "evidence" not in result
    assert len(api.calls) == 4


@pytest.mark.parametrize("status,verification,advertised,available", [
    ("AVAILABLE", "SHA256_VERIFIED", True, True), ("AVAILABLE", "PENDING", True, False),
    ("COLLECTING", "SHA256_VERIFIED", True, False), ("RELEASED", "SHA256_VERIFIED", False, False),
    ("FAILED", "PENDING", False, False)])
def test_collection_and_verification_must_both_allow_download(status, verification, advertised, available):
    api = Reader()
    api.collection.update(status=status, verification=verification, download_available=advertised,
                          training={"status": "COMPLETED", "exit_code": 0}, diagnostic="CPU_JOB_EXITED_WITHOUT_RECEIPT")
    result = run_status(api, HEALTH, "demo", "trial")
    assert result["training"]["status"] == "COMPLETED"
    assert result["publication"] == {"status": status, "phase": None, "download_available": available,
                                    "diagnostic": "CPU_JOB_EXITED_WITHOUT_RECEIPT", "error": None}


def test_explicit_old_attempt_does_not_borrow_current_run_evidence():
    api = Reader()
    api.collection["attempt_id"] = "attempt-001"
    result = run_status(api, HEALTH, "demo", "trial", "attempt-001")
    assert result["scheduler"] == {"state": "FAILED", "observed_at": None, "stale": True, "reason": None,
                                   "backend": None, "job_id": None, "native_state": None}
    assert result["training"]["process_state"] is None and result["training"]["stale"]
    assert api.calls[-1][0].endswith("/attempt-001/collection")


def test_details_expand_evidence_without_refresh_or_effects(tmp_path):
    api = Reader()
    saved = state(tmp_path / "state.json")
    original = saved.read_bytes()
    result = run_status(api, HEALTH, "demo", "trial", state_path=saved, details=True)
    assert result["preparation"] == {"status": "READY", "phase": "READY", "error_code": None}
    assert set(result["evidence"]) == {"preparation", "submission", "progress", "run", "attempts", "collection", "attempt"}
    assert saved.read_bytes() == original
    assert len(api.calls) == 7 and not any("refresh" in path for path, _ in api.calls)


def test_preparation_before_run_exists_is_visible_without_guessing_attempt(tmp_path):
    api = Reader()
    api.run = ClientError("not found", status=404)
    api.preparation.update(status="EXECUTING", phase="BUILDING_IMAGE")
    result = run_status(api, HEALTH, "demo", "trial", state_path=state(tmp_path / "state", submission=None))
    assert result["attempt_id"] is None and result["scheduler"]["state"] == "NOT_SUBMITTED"
    assert result["preparation"]["phase"] == "BUILDING_IMAGE" and not result["publication"]["download_available"]
    assert len(api.calls) == 2


@pytest.mark.parametrize("change", [{"server": "https://other"}, {"project": "other"}, {"run_id": "other"}, {"config_sha256": None}])
def test_wrong_state_rejected_before_http(tmp_path, change):
    api = Reader()
    saved = state(tmp_path / "state", **change)
    if change.get("config_sha256", "present") is None:
        value = json.loads(saved.read_text()); value.pop("config_sha256"); saved.write_text(json.dumps(value))
    with pytest.raises(ClientError):
        run_status(api, HEALTH, "demo", "trial", state_path=saved)
    assert not api.calls


def test_legacy_experiment_state_can_be_observed(tmp_path):
    api = Reader()
    saved = state(tmp_path / "state")
    value = json.loads(saved.read_text())
    for key in ("project", "run_id", "server"):
        value.pop(key)
    saved.write_text(json.dumps(value))
    assert run_status(api, HEALTH, "demo", "trial", state_path=saved)["attempt_id"] == "attempt-002"


@pytest.mark.parametrize("change", [{"submission_id": "action-" + "d" * 16}, {"attempt_id": "attempt-009"}, {"run_id": "other"}])
def test_saved_submission_identity_is_not_rebound(tmp_path, change):
    api = Reader(); api.submission.update(change)
    with pytest.raises(ClientError):
        run_status(api, HEALTH, "demo", "trial", state_path=state(tmp_path / "state"))
    assert len(api.calls) == 1


def test_saved_submission_rejects_other_requested_attempt(tmp_path):
    api = Reader()
    with pytest.raises(ClientError, match="differs from the saved"):
        run_status(api, HEALTH, "demo", "trial", "attempt-001", state_path=state(tmp_path / "state"))


@pytest.mark.parametrize("where", ["run", "listing", "collection", "entry"])
def test_mismatched_server_identity_rejected(where):
    api = Reader()
    if where == "run": api.run["project"] = "other"
    if where == "listing": api.attempts["run_id"] = "other"
    if where == "collection": api.collection["attempt_id"] = "attempt-001"
    if where == "entry": api.attempts["attempts"][-1]["identity"] = "other::attempt-002"
    with pytest.raises(ClientError):
        run_status(api, HEALTH, "demo", "trial")


@pytest.mark.parametrize("status", [404, 409])
def test_missing_collection_binding_is_explicit_unknown_not_downloadable(status):
    api = Reader(); api.collection = ClientError("unavailable", status=status, error_code="FROZEN_IDENTITY_INVALID")
    result = run_status(api, HEALTH, "demo", "trial", details=True)
    assert result["scheduler"]["state"] == "SUCCEEDED"
    assert result["publication"]["status"] == "NOT_OBSERVED" and not result["publication"]["download_available"]
    assert result["publication"]["error"] == {"status": status, "error_code": "FROZEN_IDENTITY_INVALID"}


def test_server_without_collection_capability_does_not_guess_archive_ready():
    api = Reader()
    result = run_status(api, {}, "demo", "trial")
    assert result["verification"]["status"] == "NOT_OBSERVED" and not result["publication"]["download_available"]
    assert len(api.calls) == 2


def test_empty_unsubmitted_run_has_no_invented_attempt():
    api = Reader(); api.attempts.update(current_attempt_id=None, attempts=[])
    result = run_status(api, HEALTH, "demo", "trial")
    assert result["attempt_id"] is None and result["scheduler"]["state"] == "NOT_SUBMITTED"
    assert len(api.calls) == 2


@pytest.mark.parametrize("change", [{"attempts": None}, {"current_attempt_id": None}, {"current_attempt_id": "latest"}])
def test_malformed_attempt_listing_is_not_guessed(change):
    api = Reader(); api.attempts.update(change)
    with pytest.raises(ClientError):
        run_status(api, HEALTH, "demo", "trial")


def test_unexpected_run_or_collection_error_is_propagated():
    api = Reader(); api.run = ClientError("not found", status=404)
    with pytest.raises(ClientError): run_status(api, HEALTH, "demo", "trial")
    api = Reader(); api.collection = ClientError("offline", status=503)
    with pytest.raises(ClientError): run_status(api, HEALTH, "demo", "trial")


def test_cli_status_is_read_only_and_defaults_to_compact(monkeypatch, capsys):
    api = Reader(); monkeypatch.setattr("ml_exp_client.cli.Client", lambda *args: api)
    assert main(["status", "--project", "demo", "--run", "trial", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert "evidence" not in result and result["attempt_id"] == "attempt-002"
    assert main(["status", "--project", "demo", "--run", "trial", "--details", "--json"]) == 0
    assert "evidence" in json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("key", ["submission_id", "run_id", "attempt_id"])
def test_progress_is_bound_to_exact_submission(tmp_path, key):
    api = Reader(); api.progress[key] = "other"
    with pytest.raises(ClientError, match="progress identity"):
        run_status(api, HEALTH, "demo", "trial", state_path=state(tmp_path / "state"))


def test_preparation_id_is_bound_to_original_state(tmp_path):
    api = Reader(); api.preparation["preparation_id"] = "preparation-" + "d" * 32
    with pytest.raises(ClientError, match="Preparation identity"):
        run_status(api, HEALTH, "demo", "trial", state_path=state(tmp_path / "state"))


def test_details_do_not_accept_other_attempt_scope():
    class Changed(Reader):
        def call(self, path, **kwargs):
            value = super().call(path, **kwargs)
            if path.endswith("/show"):
                value["scope"]["object_id"] = "other::attempt-002"
            return value
    with pytest.raises(ClientError, match="expanded evidence"):
        run_status(Changed(), HEALTH, "demo", "trial", details=True)


@pytest.mark.parametrize("key,value", [("preparation", "other"), ("submission", []), ("submission", True)])
def test_malformed_saved_resource_is_rejected_before_http(tmp_path, key, value):
    api = Reader()
    with pytest.raises(ClientError, match="invalid saved"):
        run_status(api, HEALTH, "demo", "trial", state_path=state(tmp_path / "state", **{key: value}))
    assert not api.calls


@pytest.mark.parametrize("missing_run", [False, True])
@pytest.mark.parametrize("phase", ["VALIDATING_SOURCE", "STAGING", "FAILED_BEFORE_SUBMISSION"])
def test_exact_saved_submission_can_precede_attempt_materialization(tmp_path, missing_run, phase):
    api = Reader()
    api.submission["status"] = "FAILED" if phase == "FAILED_BEFORE_SUBMISSION" else "EXECUTING"
    if phase == "FAILED_BEFORE_SUBMISSION":
        api.submission["execution"] = {"resolution": phase}
    api.progress["phase"] = phase
    # Existing history must not be mistaken for the new planned Attempt.
    api.attempts.update(current_attempt_id="attempt-001", attempts=api.attempts["attempts"][:1])
    if missing_run:
        api.run = ClientError("not found", status=404)
    saved = state(tmp_path / "state")
    original = saved.read_bytes()
    result = run_status(api, HEALTH, "demo", "trial", state_path=saved, details=True)
    assert result["attempt_id"] == "attempt-002"
    assert result["scheduler"]["state"] == "NOT_SUBMITTED" and result["scheduler"]["stale"]
    assert result["training"]["status"] == "UNKNOWN" and result["training"]["process_state"] is None
    assert not result["publication"]["download_available"]
    assert result["evidence"]["progress"]["phase"] == phase
    assert "attempt" not in result["evidence"]
    assert not any(path.endswith(("/collection", "/show")) or kwargs for path, kwargs in api.calls)
    assert saved.read_bytes() == original


@pytest.mark.parametrize("status,phase,resolution", [
    ("EXECUTING", "SCHEDULER_SUBMITTING", None),
    ("EXECUTING", "VERIFYING_SUBMISSION", None),
    ("EXECUTING", "RUNNING", None),
    ("VERIFIED", "STAGING", None),
    ("RECONCILE_REQUIRED", "STAGING", None),
    ("FAILED", "FAILED", "SUBMISSION_UNCERTAIN"),
    ("FAILED", "FAILED", None),
])
def test_missing_attempt_is_not_guessed_from_uncertain_or_later_submission(tmp_path, status, phase, resolution):
    api = Reader()
    api.attempts.update(current_attempt_id=None, attempts=[])
    api.submission.update(status=status, execution={"resolution": resolution})
    api.progress["phase"] = phase
    with pytest.raises(ClientError, match="absent from the Run listing"):
        run_status(api, HEALTH, "demo", "trial", state_path=state(tmp_path / "state"))
    assert not any(path.endswith(("/collection", "/show")) or kwargs for path, kwargs in api.calls)


@pytest.mark.parametrize("contradiction", ["current", "identity"])
def test_before_submission_does_not_hide_contradictory_attempt_listing(tmp_path, contradiction):
    api = Reader()
    api.submission["status"] = "EXECUTING"
    api.progress["phase"] = "STAGING"
    if contradiction == "current":
        api.attempts["attempts"] = []
    else:
        api.attempts["attempts"][-1]["identity"] = "other::attempt-002"
    with pytest.raises(ClientError, match="absent from the Run listing"):
        run_status(api, HEALTH, "demo", "trial", state_path=state(tmp_path / "state"))


def test_staging_without_bound_progress_does_not_excuse_missing_attempt(tmp_path):
    api = Reader()
    api.submission["status"] = "EXECUTING"
    api.attempts.update(current_attempt_id=None, attempts=[])
    with pytest.raises(ClientError, match="absent from the Run listing"):
        run_status(api, {}, "demo", "trial", state_path=state(tmp_path / "state"))


@pytest.mark.parametrize('backend', ['slurm', 'sensecore', 'pai-dlc'])
def test_unique_exact_submission_exposes_queue_data_and_model_progress(backend):
    api = Reader()
    api.run['backend'] = backend
    api.run['evidence']['scheduler']['detail'].update(backend=backend, backend_job_id='native-job', raw_state='Pending')
    api.submissions = [api.submission]
    api.run['evidence']['model'] = {'attempt_id': 'attempt-002', 'state': 'step 25', 'as_of': 123, 'stale': False,
                                   'detail': {'step': 25, 'name': 'train_loss', 'value': 1.25, 'unit': 'nats/token'}}
    api.run['evidence']['worker'] = {'attempt_id': 'attempt-002', 'state': 'TRAINING', 'as_of': 122, 'stale': False}
    blocker = {'CodeType': 'ConfigRule', 'Code': 'roleMaximumResource', 'Reason': 'CPU limit'}
    api.progress.update(phase='RUNNING', message='Scheduler state is not a training heartbeat',
        queue={'reason': 'CPU limit', 'blocker': blocker, 'stale': False, 'as_of': 123},
        input_delivery={'phase': 'VERIFYING', 'received_bytes': 1234, 'expected_bytes': 2000, 'bytes_per_second': 100},
        worker_phase={'phase': 'TRAINING', 'observed_at': 123, 'source': 'worker_record', 'sequence': 5})
    result = run_status(api, HEALTH, 'demo', 'trial')
    assert result['progress']['phase'] == 'RUNNING' and result['scheduler']['reason'] == 'CPU limit'
    assert result['queue']['blocker'] == blocker
    assert {key: result['scheduler'][key] for key in ('backend', 'job_id', 'native_state')} == {
        'backend': backend, 'job_id': 'native-job', 'native_state': 'Pending'}
    assert result['model']['step'] == 25 and result['model']['last_metric']['value'] == 1.25
    assert result['input_delivery']['received_bytes'] == 1234 and result['worker_phase']['phase'] == 'TRAINING'
    assert all(not kwargs and 'refresh' not in path for path, kwargs in api.calls)


def test_running_scheduler_without_worker_records_never_proves_training():
    api = Reader(); api.submissions = [api.submission]
    api.run['evidence']['scheduler']['state'] = 'RUNNING'
    api.run['evidence']['process'] = {'attempt_id': 'attempt-002', 'state': None, 'stale': True}
    api.progress.update(phase='RUNNING', worker_phase=None, workload_evidence={
        key: {'state': None, 'availability': 'NOT_OBSERVED', 'stale': True} for key in ('worker', 'process', 'model')})
    result = run_status(api, HEALTH, 'demo', 'trial')
    assert result['scheduler']['state'] == 'RUNNING' and result['training']['status'] == 'UNKNOWN'
    assert result['model']['state'] is None and result['model']['step'] is None and result['worker_phase'] is None


@pytest.mark.parametrize('matching', [False, True])
def test_missing_or_ambiguous_submission_never_uses_another_attempt(matching):
    api = Reader()
    api.submissions = [deepcopy(api.submission), deepcopy(api.submission)]
    if not matching:
        for item in api.submissions: item['attempt_id'] = 'attempt-001'
    result = run_status(api, HEALTH, 'demo', 'trial')
    assert result['progress']['availability'] == 'UNAVAILABLE'
    assert result['progress']['diagnostic'] == ('AMBIGUOUS_SUBMISSIONS' if matching else 'NO_EXACT_SUBMISSION')
    assert not any(path.endswith('/progress') for path, _ in api.calls)


def test_preparation_state_does_not_fall_back_to_an_unrelated_current_attempt(tmp_path):
    api = Reader()
    saved = state(tmp_path / 'experiment.json', submission=None)
    result = run_status(api, HEALTH, state_path=saved)
    assert result['attempt_id'] is None and result['scheduler']['state'] == 'NOT_SUBMITTED'
    assert not any(path.endswith(('/collection', '/progress')) for path, _ in api.calls)


def test_preparation_only_state_observes_server_completed_submission_once(tmp_path):
    api = Reader()
    api.preparation['submission'] = deepcopy(api.submission)
    api.attempts['current_attempt_id'] = 'attempt-001'
    saved = state(tmp_path / 'experiment.json', submission=None)
    original = saved.read_bytes()
    result = run_status(api, HEALTH, state_path=saved, details=True)
    assert result['attempt_id'] == 'attempt-002' and result['scheduler']['state'] == 'SUCCEEDED'
    assert result['evidence']['submission']['submission_id'] == SUBMISSION
    assert result['progress']['phase'] == 'SUCCEEDED'
    assert sum('experiment-preparations/' in path for path, _ in api.calls) == 1
    assert not any(path.endswith('/submissions') for path, _ in api.calls)
    assert saved.read_bytes() == original


@pytest.mark.parametrize('status', [404, 503])
def test_stateless_progress_outage_keeps_known_scheduler_evidence(status):
    class Outage(Reader):
        def call(self, path, **kwargs):
            if path.endswith('/submissions'):
                raise ClientError('unavailable', status=status, error_code='PROGRESS_UNAVAILABLE')
            return super().call(path, **kwargs)
    result = run_status(Outage(), HEALTH, 'demo', 'trial')
    assert result['scheduler']['state'] == 'SUCCEEDED'
    assert result['progress']['availability'] == 'UNAVAILABLE' and result['progress']['error']['status'] == status


@pytest.mark.parametrize('bad', [None, [None], [{'project': 'other', 'run_id': 'trial', 'attempt_id': 'attempt-002'}]])
def test_invalid_submission_listing_cannot_supply_progress(bad):
    api = Reader(); api.submissions = bad
    with pytest.raises(ClientError): run_status(api, HEALTH, 'demo', 'trial')
    assert not any(path.endswith('/progress') for path, _ in api.calls)


@pytest.mark.parametrize('status,phase', [('PREPARED', 'PREPARED'), ('EXECUTING', 'STAGING')])
def test_retry_plan_observes_its_new_attempt_before_materialization(tmp_path, status, phase):
    action = {'action_id': SUBMISSION, 'operation': 'RETRY_ATTEMPT', 'run_id': 'trial', 'attempt_id': 'attempt-003',
        'scope': {'project': 'demo', 'scope_type': 'attempt', 'object_id': 'trial::attempt-001'},
        'intent_digest': 'sha256:' + 'd' * 64, 'gate_bundle_digest': 'sha256:' + 'e' * 64,
        'execution': {'status': status}}
    class Planned(Reader):
        def call(self, path, **kwargs):
            if path == '/api/actions/' + SUBMISSION:
                self.calls.append((path, kwargs)); assert not kwargs
                return deepcopy(action)
            return super().call(path, **kwargs)
    api = Planned(); api.progress.update(attempt_id='attempt-003', phase=phase)
    path = tmp_path / 'retry.json'
    path.write_text(json.dumps({'contract': 'training-operation.v1', 'server': api.base,
        'project': 'demo', 'run_id': 'trial', 'operation': 'retry', 'source_attempt_id': 'attempt-001',
        'request': {}, 'plan': action, 'action': action}))
    before = path.read_bytes()
    result = run_status(api, HEALTH, state_path=path, details=True)
    assert result['attempt_id'] == 'attempt-003' and result['scheduler']['state'] == 'NOT_SUBMITTED'
    assert result['progress']['phase'] == phase and 'attempt' not in result['evidence']
    assert not any(path.endswith('/collection') for path, _ in api.calls)
    assert path.read_bytes() == before


@pytest.mark.parametrize('timestamp,step', [(None, 4), ('2026-10-24T12:34:56Z', None)])
def test_exact_live_journal_metric_retains_context_without_inventing_freshness(timestamp, step):
    api = Reader(); api.submissions = [api.submission]
    observation = {'name': 'validation_loss', 'value': 2.5, 'unit': 'nats/token', 'step': step,
                   'variant_id': 'heldout', 'protocol_id': 'sha256:' + 'a' * 64}
    if timestamp is not None: observation['timestamp'] = timestamp
    metric = {'observation': observation, 'event_id': 'metric.gen.123', 'sequence': 7, 'source': 'record-journal'}
    api.progress['last_metric'] = metric
    result = run_status(api, HEALTH, 'demo', 'trial')
    assert result['model']['last_metric'] == observation and result['model']['step'] == step
    assert result['model']['metric_record'] == {'event_id': 'metric.gen.123', 'sequence': 7, 'source': 'record-journal'}
    assert result['model']['observed_at'] == timestamp
    assert result['model']['stale'] is (True if timestamp is None else None)
    assert result['training']['status'] == 'UNKNOWN'


def test_bound_index_model_is_not_replaced_by_a_different_live_metric():
    api = Reader(); api.submissions = [api.submission]
    api.run['evidence']['model'] = {'attempt_id': 'attempt-002', 'state': 'step 50',
                                   'detail': {'step': 50, 'loss': 1.2}, 'as_of': 100, 'stale': True}
    api.progress['last_metric'] = {'observation': {'name': 'other_protocol', 'step': 999, 'value': 0}, 'sequence': 8}
    result = run_status(api, HEALTH, 'demo', 'trial')
    assert result['model']['step'] == 50 and result['model']['last_metric'] == {'step': 50, 'loss': 1.2}
    assert result['model']['metric_record'] == {'source': 'indexed-model', 'sequence': None, 'event_id': None}
