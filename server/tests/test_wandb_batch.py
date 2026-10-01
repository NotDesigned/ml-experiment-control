"""One SDK session per ordered Attempt batch, with independent target recovery."""
import json
import sys
from types import SimpleNamespace

import pytest

from ml_exp_server.observability_store import ObservabilityStore, AttemptRef, SourceRef, OutboxRecord
from ml_exp_server.wandb_publisher import (
    AttemptIdentity, PublicationItem, TargetKind, WandbPublisher, _run_worker,
)
from tests.test_wandb_publisher import FakeAdapter, target
from tests.test_observability_runtime import _coordinator
from tests.test_observability_runtime_edges import Store, outbox, coordinator, config

ATTEMPT = AttemptRef('workspace', 'demo', 'run-a', 'attempt-001')
IDENTITY = AttemptIdentity(*ATTEMPT.values())


def populate(store, count=5, attempt=ATTEMPT):
    store.enqueue_and_advance(
        SourceRef(attempt, 'metrics'), expected=None, generation='g', byte_offset=count,
        records=[OutboxRecord(str(n), 'metric', {'loss': n}) for n in range(count)],
        targets=['local', 'cloud'], now=1,
    )


def test_batch_claim_is_ordered_target_isolated_and_recovered_after_lease(tmp_path):
    store = ObservabilityStore(tmp_path / 'outbox.sqlite')
    populate(store)
    for invalid in (0, 51):
        with pytest.raises(ValueError, match='batch_size'):
            store.claim('local', 'a', batch_size=invalid)
    first = store.claim('local', 'a', limit=1, batch_size=3, now=2)
    assert [row.record_key for row in first] == ['0', '1', '2']
    assert store.claim('local', 'b', batch_size=50, now=3) == []
    cloud = store.claim('cloud', 'c', limit=1, batch_size=50, now=3)
    assert len(cloud) == 5
    resumed = store.claim('local', 'b', limit=1, batch_size=50, now=63)
    assert [row.record_key for row in resumed] == ['0', '1', '2', '3', '4']
    for row in resumed:
        store.acknowledge(row.id, 'b', now=64)
    assert store.claim('local', 'c', batch_size=50, now=65) == []
    store.close()


@pytest.mark.parametrize('barrier', ['backoff', 'terminal', 'lease'])
def test_batch_never_skips_unavailable_record(tmp_path, barrier):
    store = ObservabilityStore(tmp_path / 'outbox.sqlite')
    populate(store)
    second = store._conn.execute("SELECT id FROM publication_outbox WHERE target='local' ORDER BY id LIMIT 1 OFFSET 1").fetchone()[0]
    column, value = {'backoff': ('available_at', 100), 'terminal': ('terminal_at', 1), 'lease': ('lease_until', 100)}[barrier]
    store._conn.execute(f'UPDATE publication_outbox SET {column}=? WHERE id=?', (value, second))
    store._conn.commit()
    batch = store.claim('local', 'worker', limit=1, batch_size=50, now=3)
    assert len(batch) == 1 and batch[0].record_key == '0'
    store.acknowledge(batch[0].id, 'worker', now=4)
    assert store.claim('local', 'worker', batch_size=50, now=5) == []
    store.close()


def test_sdk_init_and_finish_once_for_whole_batch(monkeypatch, tmp_path):
    adapter = FakeAdapter()
    publisher = WandbPublisher(adapter)
    items = tuple(PublicationItem(TargetKind.LOCAL, str(n), n, 'metric', {'loss': n}) for n in range(50))
    result = publisher.publish_batch(target(tmp_path, TargetKind.LOCAL), IDENTITY, items)
    assert len(adapter.calls) == 1 and all(item.acknowledged for item in result)
    calls = []
    class Run:
        def log(self, payload, **kwargs):
            calls.append(('log', kwargs['step'], payload['loss']))
        def finish(self, **kwargs):
            calls.append(('finish', kwargs))
    def init(**kwargs):
        calls.append(('init', kwargs['id'], kwargs['resume']))
        return Run()
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=init, Settings=lambda **kwargs: kwargs))
    request = adapter.calls[0][0]
    assert _run_worker(json.dumps(request.worker_payload())) == 0
    assert calls[0] == ('init', IDENTITY.wandb_run_id, 'allow')
    assert calls[1:51] == [('log', n, n) for n in range(50)]
    assert calls[-1] == ('finish', {'exit_code': 0})
    for invalid in ((), items + items[:1], tuple(reversed(items)), (items[0], items[0])):
        with pytest.raises(ValueError):
            publisher.publish_batch(target(tmp_path, TargetKind.LOCAL), IDENTITY, invalid)


def test_partial_sdk_failure_never_acknowledges_batch(monkeypatch, tmp_path):
    store = ObservabilityStore(tmp_path / 'outbox.sqlite')
    populate(store)
    coordinator = _coordinator(tmp_path, store, local_enabled=True)
    class Adapter:
        def publish(self, request, *, environment):
            calls = []
            class Run:
                def log(self, payload, **kwargs):
                    calls.append(kwargs['step'])
                    if len(calls) == 2:
                        raise OSError('simulated network failure')
                def finish(self, **kwargs):
                    raise AssertionError('must not finish a failed batch')
            monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=lambda **kwargs: Run(), Settings=lambda **kwargs: kwargs))
            assert _run_worker(json.dumps(request.worker_payload())) == 1
            assert len(calls) == 2
            raise RuntimeError('worker failed')
    coordinator.publisher = WandbPublisher(Adapter(), credential_provider=lambda _: 'test')
    coordinator.publish_once()
    statuses = {status.target: status for status in store.statuses()}
    assert statuses['local'].delivered == 0 and statuses['local'].pending == 5
    assert statuses['cloud'].delivered == 0 and statuses['cloud'].pending == 5
    store.close()


def test_coordinator_batch_limit_and_acknowledgement_identity(tmp_path):
    store = ObservabilityStore(tmp_path / 'outbox.sqlite')
    populate(store, 55)
    value = _coordinator(tmp_path, store, local_enabled=True)
    adapter = FakeAdapter()
    value.publisher = WandbPublisher(adapter, credential_provider=lambda _: 'test')
    value.publish_once(limit_per_target=50)
    assert len(adapter.calls) == 1 and len(adapter.calls[0][0].items) == 50
    value.publish_once(limit_per_target=50)
    assert len(adapter.calls) == 2 and len(adapter.calls[1][0].items) == 5
    statuses = {status.target: status for status in store.statuses()}
    assert statuses['local'].delivered == 55 and statuses['cloud'].delivered == 0
    store.close()
    fake = Store([outbox()])
    value = coordinator(tmp_path, fake)
    value._target_config = lambda *_: config(tmp_path)
    value.publisher = SimpleNamespace(publish_batch=lambda *_: [])
    value.publish_once()
    assert ('retry', 1, 'ValueError') in fake.calls


def test_wrong_target_acknowledgement_is_not_committed(tmp_path):
    from ml_exp_server.wandb_publisher import PublishResult
    for target_kind, key in ((TargetKind.CLOUD, 'record'), (TargetKind.LOCAL, 'wrong')):
        fake = Store([outbox()])
        value = coordinator(tmp_path, fake)
        value._target_config = lambda *_: config(tmp_path)
        value.publisher = SimpleNamespace(publish_batch=lambda *_: [PublishResult(
            acknowledged=True, target=target_kind, record_key=key, run_id='run', dashboard_url=None,
        )])
        value.publish_once()
        assert ('retry', 1, 'ValueError') in fake.calls
        assert ('ack', 1) not in fake.calls
