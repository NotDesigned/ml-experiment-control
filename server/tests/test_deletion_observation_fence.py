"""Legacy observation identities stay readable without widening managed writes."""
from types import SimpleNamespace

import pytest
import yaml

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.deletion_state import RunFence
from ml_exp_server.runs.deletion_guards import action_guard


@pytest.mark.parametrize("run", ["legacy+run", "旧实验", "r" * 129, "../outside", None, 7])
def test_observation_queries_do_not_construct_legacy_paths_or_relax_managed_validation(tmp_path, run):
    fence = RunFence(tmp_path / "state")
    assert fence.is_deleted("demo", run) is False
    fence.require_observed_live("demo", run)
    assert not fence.root.exists()
    for check in [fence.path, fence.mark, fence.require_live]:
        with pytest.raises(ValueError, match="invalid Run deletion identity"):
            check("demo", run)
    with pytest.raises(ValueError):
        with fence.guard("demo", run):
            pytest.fail("managed mutation accepted an unsupported identity")
    assert not fence.root.exists()


def test_observation_guard_uses_shared_fence_and_rejects_managed_tombstones(tmp_path):
    fence = RunFence(tmp_path)
    with fence.observation_guard("demo", "legacy+run"):
        with pytest.raises(ApplicationError) as busy:
            RunFence(tmp_path).mark("demo", "managed-run")
        assert busy.value.code == "RUN_DELETION_BUSY"
    fence.mark("demo", "managed-run")
    for check in [fence.require_live, fence.require_observed_live]:
        with pytest.raises(ApplicationError) as deleted:
            check("demo", "managed-run")
        assert (deleted.value.status_code, deleted.value.code) == (410, "RUN_DELETED")
    for guard in [fence.guard, fence.observation_guard]:
        with pytest.raises(ApplicationError) as deleted:
            with guard("demo", "managed-run"):
                pytest.fail("managed tombstone was ignored")
        assert deleted.value.code == "RUN_DELETED"


@pytest.mark.parametrize("scope_type", ["run", "attempt"])
def test_ordinary_actions_keep_complete_legacy_identity_and_shared_fence(tmp_path, scope_type):
    fence = RunFence(tmp_path)
    fence.mark("demo", "managed-run")
    service = SimpleNamespace(deletion_service=SimpleNamespace(fence=fence))
    legacy = "managed-run::legacy+observation"
    object_id = legacy + "::attempt-001" if scope_type == "attempt" else legacy
    snapshot = {"scope": {"project": "demo", "scope_type": scope_type, "object_id": object_id},
                "operation": "CANCEL_RUN", "run_id": legacy}
    with action_guard(service, snapshot):
        with pytest.raises(ApplicationError) as busy:
            RunFence(tmp_path).mark("demo", "another-managed-run")
        assert busy.value.code == "RUN_DELETION_BUSY"
    snapshot["scope"]["object_id"] = "managed-run::attempt-001" if scope_type == "attempt" else "managed-run"
    snapshot["run_id"] = "managed-run"
    with pytest.raises(ApplicationError) as deleted:
        with action_guard(service, snapshot):
            pytest.fail("ordinary Action crossed a managed tombstone")
    assert deleted.value.code == "RUN_DELETED"


@pytest.mark.parametrize("definition", [
    {"run_id": "legacy+run"}, {"run_id": None}, {"run_id": 7}, {"run_id": []}, {}, "invalid entry",
    {"run_id": "new-run", "resume_from": {"run_id": "legacy+source"}},
    {"run_id": "new-run", "resume_from": {"run_id": None}},
])
def test_campaign_actions_still_validate_proposed_managed_and_source_identities(tmp_path, definition):
    service = SimpleNamespace(deletion_service=SimpleNamespace(fence=RunFence(tmp_path)))
    snapshot = {"scope": {"project": "demo", "scope_type": "project", "object_id": "demo"},
                "operation": "WRITE_CAMPAIGN", "proposed_content": yaml.safe_dump({"runs": [definition]})}
    with pytest.raises(ApplicationError) as invalid:
        with action_guard(service, snapshot):
            pytest.fail("campaign introduced an invalid managed identity")
    assert (invalid.value.status_code, invalid.value.code) == (422, "INVALID_RUN_IDENTITY")
    assert not (tmp_path / "deleted-runs").exists()
