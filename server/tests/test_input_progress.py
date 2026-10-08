"""Data preparation diagnoses do not leak credentials or mix Attempts."""
from datetime import datetime, timezone
import errno
import io
import json
from pathlib import Path
import ssl

import pytest
from fastapi.testclient import TestClient

from ml_exp_server.workers.data_input import DeliveryTrace, InputDeliveryError, fetch
from ml_exp_server.runs.input_progress import read_input_progress, MARKER
from ml_exp_server.schemas import RunIndexRow
from tests.test_submissions import _app, _action


def event(**changes):
    return {'asset_id': 'asset.' + 'a' * 64, 'phase': 'FAILED', 'at': datetime.now(timezone.utc).isoformat(),
            'code': 'INPUT_TIMEOUT', 'error_class': 'TimeoutError', 'failed_phase': 'DOWNLOADING', **changes}


@pytest.mark.parametrize('error,code', [(TimeoutError('Bearer private'), 'INPUT_TIMEOUT'),
    (ssl.SSLError('private URL'), 'INPUT_TLS_FAILED'), (OSError(errno.ENOSPC, 'private path'), 'INPUT_NETWORK_OR_STORAGE_FAILED'),
    (ValueError('private response'), 'INPUT_VALIDATION_FAILED'), (InputDeliveryError('INPUT_ARCHIVE_TRUNCATED', 'private'), 'INPUT_ARCHIVE_TRUNCATED')])
def test_worker_safe_failure_and_throttled_bytes(error, code, capsys, monkeypatch):
    times = iter([0, 1, 2, 10, 11])
    monkeypatch.setattr('ml_exp_server.workers.data_input.time.monotonic', lambda: next(times))
    trace = DeliveryTrace('asset.' + 'a'*64, 100)
    trace.report('DOWNLOADING', received_bytes=1)
    trace.report('DOWNLOADING', received_bytes=2, force=False)
    trace.report('DOWNLOADING', received_bytes=3, force=False)
    trace.failed(error)
    text = capsys.readouterr().out
    assert 'private' not in text and 'Bearer' not in text
    values = [json.loads(line.split('=',1)[1]) for line in text.splitlines()]
    assert len(values) == 3 and values[-1]['code'] == code
    assert values[-1]['received_bytes'] == 3 and values[-1]['failed_phase'] == 'DOWNLOADING'


def test_reader_handles_historical_and_unsafe_or_missing_logs(tmp_path):
    assert read_input_progress(tmp_path/'absent') is None
    (tmp_path/'stdout.log').symlink_to(tmp_path/'missing')
    (tmp_path/'stderr.log').write_text('irrelevant\nML_EXPD_INPUT_DELIVERY=FAILED\n')
    assert read_input_progress(tmp_path)['code'] == 'INPUT_DELIVERY_FAILED_DETAILS_UNAVAILABLE'
    (tmp_path/'stderr.log').write_text('unrelated\n')
    assert read_input_progress(tmp_path) is None


@pytest.mark.parametrize('value', ['bad json', json.dumps([]), json.dumps({'at':None}), json.dumps(event(at='yesterday')),
    json.dumps(event(at='2026-01-01T00:00:00')), json.dumps(event(phase='evil')), json.dumps(event(asset_id='private-secret'))])
def test_reader_ignores_malformed_events(tmp_path, value):
    (tmp_path/'stdout.log').write_text(MARKER+value+'\n')
    assert read_input_progress(tmp_path) is None


def test_reader_bounds_values_and_timestamps_and_preserves_latest(tmp_path):
    earlier=event(at='2026-01-01T00:00:00+00:00',phase='DOWNLOADING')
    final=event(received_bytes=3, expected_bytes=100, http_status=200, errno=28,
                bytes_per_second=float('nan'), elapsed_seconds='private', code='Bearer private',
                error_class='TimeoutError', path='/private', url='https://private')
    (tmp_path/'stdout.log').write_text('x'*150000+'\n'+MARKER+json.dumps(final)+'\n')
    (tmp_path/'stderr.log').write_text(MARKER+json.dumps(earlier)+'\n')
    result=read_input_progress(tmp_path)
    assert result['phase']=='FAILED' and result['received_bytes']==3 and result['seconds_since_progress']>=0
    assert result['code']=='INPUT_DELIVERY_FAILED_DETAILS_UNAVAILABLE'
    assert not {'path','url','bytes_per_second','elapsed_seconds'} & result.keys()


def test_submission_progress_exposes_exact_legacy_failure(tmp_path):
    app,_=_app(tmp_path)
    with TestClient(app) as api:
        p=api.post('/api/experiments/demo/run-a/submissions/prepare',json={'max_gpu_hours':2}).json()
        endpoint='/api/submissions/'+p['submission_id']
        api.post(endpoint+'/authorize',json={'note':'test'})
        api.post(endpoint+'/execute',json={'confirmation':p['confirmation']});_action(api,p['submission_id'])
        app.state.index.upsert_run(RunIndexRow(project='demo',run_id='run-a',run_dir=str(tmp_path)))
        attempt=tmp_path/'attempts/attempt-001';attempt.mkdir(parents=True)
        (attempt/'stderr.log').write_text('ML_EXPD_INPUT_DELIVERY=FAILED\n')
        v=api.get(endpoint+'/progress').json()
        assert v['diagnostic']=='INPUT_DELIVERY_FAILED_DETAILS_UNAVAILABLE'
        (attempt/'stdout.log').write_text(MARKER+json.dumps(event(received_bytes=12))+'\n')
        v=api.get(endpoint+'/progress').json()
        assert v['diagnostic']=='INPUT_TIMEOUT' and v['input_delivery']['received_bytes']==12
        assert v['last_progress_unix'] is None


def test_delivery_trace_initial_failure_and_success_progress(tmp_path, monkeypatch, capsys):
    trace=DeliveryTrace()
    trace.failed(ValueError("private"))
    assert 'INITIALIZING' in capsys.readouterr().out
    from ml_exp_server.workers import data_input
    data=b'abc'; body=io.BytesIO(data)
    class Connection:
        def __init__(self,*args,**kwargs):pass
        def request(self,*args,**kwargs):pass
        def getresponse(self):
            from types import SimpleNamespace
            return SimpleNamespace(status=200,getheader=lambda *args:'3',read=body.read)
        def close(self):pass
    monkeypatch.setattr(data_input.http.client,'HTTPSConnection',Connection)
    import hashlib
    fetch('https://api.example/input','private',io.BytesIO(),hashlib.sha256(data).hexdigest(),3,trace)
    assert trace.value['received_bytes']==3
    from tests.test_managed_data_worker import input_item,tar_bytes
    files={'x':b'abc'};archive=tar_bytes(files)
    result=data_input.deliver(input_item(archive,files),None,tmp_path/'cache',archive_stream=io.BytesIO(archive),trace=trace)
    assert result.is_dir() and trace.value['phase']=='VERIFYING'


def test_reader_reports_ready_without_failure_code(tmp_path):
    (tmp_path/'stdout.log').write_text(MARKER+json.dumps(event(phase='READY'))+'\n')
    result=read_input_progress(tmp_path)
    assert result['phase']=='READY'


def test_prepared_cache_missing_has_specific_code(tmp_path):
    from ml_exp_server.workers.data_input import deliver
    with pytest.raises(InputDeliveryError) as error:
        deliver({'asset_id':'asset.'+'a'*64,'sha256':'a'*64,'require_cached':True},None,tmp_path)
    assert error.value.code=='INPUT_PREPARED_CACHE_MISSING'
