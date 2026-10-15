"""Late result collectors cannot cross a deletion claim or recreate a Run."""
from contextlib import contextmanager
import json
import shutil

import pytest

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.deletion_state import RunFence
from ml_exp_server.storage import atomic_json
from tests.test_artifact_store import storage
from tests.test_result_collection import recovery

IDENTITY = ("demo", "run-a", "attempt-001")


def unavailable_claim(service):
    with pytest.raises(ApplicationError) as error:
        with RunFence(service.root.parent).exclusive():
            pytest.fail("deletion crossed an active collector")
    assert error.value.code == "RUN_DELETION_BUSY"


def test_state_holds_fence_before_state_lock_and_nested_update(recovery):
    service, _, _, _, _, _, _ = recovery
    with service.state(*IDENTITY) as (_, snapshot):
        assert snapshot.value == {}
        unavailable_claim(service)
    assert service.update(IDENTITY, status="QUEUED")["status"] == "QUEUED"
    with service.fence.guard(*IDENTITY[:2]):
        service.update(IDENTITY, status="WAITING")
        unavailable_claim(service)


@pytest.mark.parametrize("method,inner,args,kwargs", [
    ("begin", "_begin", IDENTITY, {"retry": True, "reconcile": True}),
    ("finish", "_finish", (IDENTITY,), {}),
    ("finish_one", "_finish_one", (IDENTITY,), {}),
    ("repair", "_repair", (*IDENTITY, "a" * 64), {}),
    ("publish_local", "_publish_local", (IDENTITY, "outputs", {"token": "scoped"}), {"repair_sha256": "b" * 64}),
])
def test_whole_collection_effect_and_claim_are_shared_guarded(recovery, monkeypatch, method, inner, args, kwargs):
    service, _, _, _, _, _, _ = recovery
    calls = []
    def effect(*actual, **options):
        unavailable_claim(service)
        calls.append((actual, options))
        # State acquisition uses the same Run-before-state lock order.
        service.update(IDENTITY, status="COLLECTING")
        return "effect-result"
    monkeypatch.setattr(service, inner, effect)
    assert getattr(service, method)(*args, **kwargs) == "effect-result"
    assert len(calls) == 1


def test_late_queued_finish_and_direct_writers_do_not_recreate_deleted_state(recovery, monkeypatch):
    service, root, token, _, remote, _, _ = recovery
    _, pending = service.begin(*IDENTITY)
    target = service.root / IDENTITY[0] / IDENTITY[1]
    shutil.rmtree(target)
    RunFence(service.root.parent).mark(*IDENTITY[:2])
    monkeypatch.setattr(service, "_finish", lambda identity: pytest.fail("late collector started effects"))
    assert service.finish(pending) is None and not target.exists()
    methods = [lambda: service.begin(*IDENTITY), lambda: service.update(IDENTITY, status="AVAILABLE"),
               lambda: service.finish_one(IDENTITY), lambda: service.repair(*IDENTITY, "a" * 64),
               lambda: service.publish_local(IDENTITY, root, {"token": token})]
    for call in methods:
        with pytest.raises(ApplicationError) as error: call()
        assert error.value.code == "RUN_DELETED"
    with pytest.raises(ApplicationError):
        with service.state(*IDENTITY): pytest.fail("deleted state reopened")
    assert not target.exists() and not remote


def test_interruption_scan_skips_deleted_state_and_preserves_other_runs(recovery):
    service, _, _, _, _, _, _ = recovery
    service.update(IDENTITY, status="COLLECTING")
    deleted = service.root / "demo/run-a/attempt-001.json"
    original = deleted.read_bytes()
    other = service.root / "demo/other/attempt-001.json"
    atomic_json(other, {"status": "WAITING"})
    RunFence(service.root.parent).mark(*IDENTITY[:2])
    assert service.recover_interrupted() == 1
    assert deleted.read_bytes() == original
    assert json.loads(other.read_text())["status"] == "RECONCILE_REQUIRED"
    assert service.recover_interrupted() == 0


def test_unrelated_collection_errors_are_not_hidden(recovery, monkeypatch):
    service, _, _, _, _, _, _ = recovery
    failure = ApplicationError("unrelated failure", code="OTHER_ERROR")
    monkeypatch.setattr(service, "_finish", lambda identity: (_ for _ in ()).throw(failure))
    with pytest.raises(ApplicationError) as error: service.finish(IDENTITY)
    assert error.value is failure
    service.update(IDENTITY, status="QUEUED")
    @contextmanager
    def broken_state(*identity):
        raise failure
        yield
    monkeypatch.setattr(service, "state", broken_state)
    with pytest.raises(ApplicationError) as error: service.recover_interrupted()
    assert error.value is failure


def test_automatic_late_cycle_and_training_receipt_cannot_revive_run(recovery):
    service, _, token, _, remote, _, _ = recovery
    training = dict(zip(("project", "run_id", "attempt_id"), IDENTITY), contract="training-result.v1", exit_code=0)
    service.register_training(*IDENTITY, token, training)
    RunFence(service.root.parent).mark(*IDENTITY[:2])
    service.automatic(lambda *args: pytest.fail("deleted Run submitted recovery"))
    with pytest.raises(ApplicationError) as error: service.register_training(*IDENTITY, token, training)
    assert error.value.code == "RUN_DELETED" and not remote
    assert not (service.root / "demo/run-a").exists()
