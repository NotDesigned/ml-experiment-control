"""Crash, migration, rollback, and competing-writer Action contracts."""
import json
import multiprocessing
import os
import signal
import sqlite3

import pytest

from ml_exp_server.actions.store import ActionStore
from ml_exp_server.actions.migrate import migrate, main
from ml_exp_server.storage import DurableJsonState, StorageError, TransitionConflict, atomic_json

ACTION = 'action-0123456789abcdef'


def prepared(root):
    store = ActionStore(root)
    store.save_plan({'action_id': ACTION, 'ready': True, 'operation': 'TEST'})
    return store


def legacy(root, status='RECONCILE_REQUIRED'):
    path = root / ACTION
    path.mkdir(parents=True)
    atomic_json(path / 'plan.json', {'action_id': ACTION, 'ready': True, 'operation': 'TEST'})
    state = DurableJsonState(path / 'execution.json', path / 'journal.jsonl')
    state.commit({'status': status, 'revision': 1}, event={'event': 'prepared'})
    state.commit({'status': status, 'revision': 2, 'result': {'job_id': 'immutable-job'}},
                 expected_revision=1, event={'event': 'observed'})
    state.append_event({'event': 'audit'}, event_id='once')
    return path


def test_migration_and_rollback_preserve_all_states_and_original_bytes(tmp_path):
    for status in ('PREPARED', 'AUTHORIZED', 'EXECUTING', 'RECONCILE_REQUIRED', 'VERIFIED', 'FAILED', 'BLOCKED'):
        root = tmp_path / status
        directory = legacy(root, status)
        original = {p.name: p.read_bytes() for p in directory.iterdir()}
        assert migrate(root)['actions'] == 1
        store = ActionStore(root)
        snapshot = store.snapshot(ACTION)
        assert snapshot['execution']['status'] == status
        assert snapshot['execution']['revision'] == 2
        assert snapshot['execution']['result']['job_id'] == 'immutable-job'
        assert {p.name: p.read_bytes() for p in directory.iterdir()} == original
        store.append_journal(ACTION, 'audit', {}, event_id='once')
        assert len(store.snapshot(ACTION)['journal']) == 3
        export = tmp_path / (status + '-export')
        assert migrate(root, export)['exported']
        old = DurableJsonState(export / ACTION / 'execution.json', export / ACTION / 'journal.jsonl')
        assert old.snapshot({}).value == snapshot['execution']
        old.repair_journal(old.snapshot({}))
        migrate(export)
        assert ActionStore(export).snapshot(ACTION) == snapshot
        with pytest.raises(FileExistsError):
            migrate(root, export)
        with pytest.raises(ValueError, match='outside'):
            migrate(root, root / 'nested')


def test_import_repairs_only_private_copy_and_never_reimports_stale_json(tmp_path):
    directory = legacy(tmp_path)
    journal = directory / 'journal.jsonl'
    first = journal.read_bytes().splitlines(keepends=True)[0]
    journal.write_bytes(first + b'{"truncated":')
    migrate(tmp_path)
    store = ActionStore(tmp_path)
    snapshot = store.snapshot(ACTION)
    assert len(snapshot['journal']) == 2
    assert journal.read_bytes() == first + b'{"truncated":'
    (directory / 'execution.json').write_text('stale and broken')
    journal.write_text('stale and broken')
    assert store.snapshot(ACTION) == snapshot
    store.database.path.unlink()
    with pytest.raises(StorageError, match='missing'):
        store.snapshot(ACTION)


def test_corrupt_legacy_fails_closed_and_migration_can_resume(tmp_path):
    directory = legacy(tmp_path)
    path = directory / 'execution.json'
    original = path.read_text()
    path.write_text('{broken')
    store = ActionStore(tmp_path)
    with pytest.raises(StorageError, match='explicit migration'):
        store.snapshot(ACTION)
    with pytest.raises(StorageError, match='unreadable'):
        migrate(tmp_path)
    path.write_text(original)
    migrate(tmp_path)
    assert store.snapshot(ACTION)['execution']['revision'] == 2
    bad = tmp_path / 'action-bad'
    bad.mkdir()
    atomic_json(bad / 'plan.json', {'action_id': 'invalid/path'})
    with pytest.raises(StorageError, match='unreadable Action plan'):
        migrate(tmp_path)


def test_legacy_without_metadata_and_missing_execution_are_distinct(tmp_path):
    root = tmp_path / 'old'
    directory = legacy(root)
    atomic_json(directory / 'execution.json', {'status': 'FAILED'})
    (directory / 'journal.jsonl').write_text('{"event":"old audit"}\n')
    migrate(root)
    store = ActionStore(root)
    assert store.execution(ACTION) == {'status': 'FAILED', 'revision': 0}
    export = tmp_path / 'export'
    migrate(root, export)
    assert '_durability' not in json.loads((export / ACTION / 'execution.json').read_text())
    root2 = tmp_path / 'missing'
    directory2 = legacy(root2)
    (directory2 / 'execution.json').unlink()
    with pytest.raises(StorageError, match='without execution'):
        migrate(root2)
    root3 = tmp_path / 'drift'
    directory3 = legacy(root3)
    raw = json.loads((directory3 / 'execution.json').read_text())
    raw['revision'] = 99
    atomic_json(directory3 / 'execution.json', raw)
    with pytest.raises(StorageError, match='revision does not match'):
        migrate(root3)


def test_event_write_failure_rolls_back_state_and_revision(tmp_path):
    store = prepared(tmp_path)
    before = store.snapshot(ACTION)
    with store.database.transaction() as conn:
        conn.execute("CREATE TRIGGER fail_event BEFORE INSERT ON events BEGIN SELECT RAISE(ABORT,'disk fault'); END")
    with pytest.raises(StorageError, match='transaction failed'):
        store.set_execution(ACTION, {**before['execution'], 'status': 'AUTHORIZED'}, event='authorized')
    assert store.snapshot(ACTION) == before
    with store.database.transaction() as conn:
        conn.execute('DROP TRIGGER fail_event')
    with pytest.raises(TransitionConflict, match='expected revision'):
        store.database.commit(ACTION, {'revision': 0}, event={})
    with pytest.raises(TransitionConflict, match='expected revision'):
        store.database.commit(ACTION, {'revision': 99}, event={})
    assert store.snapshot(ACTION) == before


def _crash_transaction(root, ready):
    store = ActionStore(root)
    with store.database.transaction() as conn:
        conn.execute("UPDATE executions SET payload='{}',revision=999")
        ready.send(True)
        signal.pause()


def _claim(root, ready, results):
    store = ActionStore(root)
    ready.wait()
    try:
        current = store.execution(ACTION)
        store.begin_execution(ACTION, {**current, 'status': 'EXECUTING'}, intent_digest='same-intent')
        results.put('won')
    except RuntimeError:
        results.put('lost')


def test_sigkill_rolls_back_and_competing_processes_claim_once(tmp_path):
    store = prepared(tmp_path)
    before = store.snapshot(ACTION)
    context = multiprocessing.get_context('fork')
    receive, send = context.Pipe(False)
    process = context.Process(target=_crash_transaction, args=(tmp_path, send))
    process.start()
    assert receive.poll(5) and receive.recv()
    process.kill()
    process.join(5)
    assert process.exitcode == -signal.SIGKILL
    assert store.snapshot(ACTION) == before
    store.set_execution(ACTION, {**before['execution'], 'status': 'AUTHORIZED'}, event='authorized')
    ready, results = context.Event(), context.Queue()
    workers = [context.Process(target=_claim, args=(tmp_path, ready, results)) for _ in range(4)]
    for worker in workers:
        worker.start()
    ready.set()
    outcomes = [results.get(timeout=10) for _ in workers]
    for worker in workers:
        worker.join(5)
        assert worker.exitcode == 0
    assert outcomes.count('won') == 1
    snapshot = store.snapshot(ACTION)
    assert snapshot['execution']['revision'] == 3
    assert sum(item['event'] == 'execution_started' for item in snapshot['journal']) == 1


def test_schema_guard_private_mode_empty_migration_and_cli(tmp_path, capsys):
    assert main(['--root', str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)['actions'] == 0
    assert (tmp_path / 'actions.sqlite3').stat().st_mode & 0o777 == 0o600
    with sqlite3.connect(tmp_path / 'actions.sqlite3') as conn:
        conn.execute('PRAGMA user_version=99')
    with pytest.raises(StorageError, match='unsupported'):
        migrate(tmp_path)


def test_integrity_failure_prevents_rollback_export(tmp_path):
    root = tmp_path / 'actions'
    migrate(root)
    with sqlite3.connect(root / 'actions.sqlite3') as conn:
        conn.execute('PRAGMA ignore_check_constraints=ON')
        conn.execute("INSERT INTO executions VALUES ('orphan',-1,'{}')")
    destination = tmp_path / 'export'
    with pytest.raises(StorageError, match='integrity check'):
        migrate(root, destination)
    assert not destination.exists()


def test_legacy_journal_cannot_advance_without_authoritative_metadata(tmp_path):
    directory = legacy(tmp_path)
    atomic_json(directory / 'execution.json', {'status': 'AUTHORIZED'})
    with pytest.raises(StorageError, match='without authoritative metadata'):
        migrate(tmp_path)


def test_runtime_refuses_legacy_state_and_reads_do_not_create_actions(tmp_path, monkeypatch):
    directory = legacy(tmp_path)
    originals = {p.name: p.read_bytes() for p in directory.iterdir()}
    store = ActionStore(tmp_path)
    with pytest.raises(StorageError, match='explicit migration'):
        store.snapshot(ACTION)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == originals
    migrate(tmp_path)
    # The runtime cannot decode or repair legacy JSON after explicit import.
    monkeypatch.setattr(DurableJsonState, 'snapshot', lambda *_: pytest.fail('legacy code used'))
    assert store.execution(ACTION)['status'] == 'RECONCILE_REQUIRED'
    assert store.execution('action-unknown') == {}
    assert store.database.journal('action-unknown') == []
    with store.database.transaction() as connection:
        assert connection.execute('SELECT count(*) FROM executions').fetchone()[0] == 1


def test_explicit_import_is_all_or_nothing_and_preserves_committed_sqlite(tmp_path):
    first = legacy(tmp_path)
    broken = tmp_path / 'action-ffffffffffffffff'
    broken.mkdir()
    atomic_json(broken / 'plan.json', {'action_id': broken.name, 'ready': True})
    (broken / 'execution.json').write_text('{broken')
    with pytest.raises(StorageError, match='unreadable'):
        migrate(tmp_path)
    with sqlite3.connect(tmp_path / 'actions.sqlite3') as connection:
        assert connection.execute('SELECT count(*) FROM executions').fetchone()[0] == 0
        assert connection.execute('SELECT count(*) FROM events').fetchone()[0] == 0
    (broken / 'execution.json').unlink()
    assert migrate(tmp_path)['actions'] == 2
    store = ActionStore(tmp_path)
    before = store.snapshot(ACTION)
    (first / 'execution.json').write_text('stale and broken')
    assert migrate(tmp_path)['actions'] == 2
    assert store.snapshot(ACTION) == before


def test_single_transaction_checks_status_revision_and_keeps_caller_immutable(tmp_path):
    store = prepared(tmp_path)
    prepared_state = store.execution(ACTION)
    proposed = {**prepared_state, 'status': 'AUTHORIZED'}
    result = store.database.commit(ACTION, proposed, event={'event': 'authorized'}, expected_status='PREPARED')
    assert result['revision'] == 2 and proposed['revision'] == 1
    with pytest.raises(TransitionConflict, match='expected PREPARED, found AUTHORIZED'):
        store.database.commit(ACTION, proposed, event={}, expected_status='PREPARED')
    with pytest.raises(TransitionConflict, match='expected revision 1, found 2'):
        store.database.commit(ACTION, proposed, event={})
    assert store.execution(ACTION) == result
