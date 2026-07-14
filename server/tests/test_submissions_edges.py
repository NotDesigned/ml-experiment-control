"""Fail-closed and idempotency edges for first-class submissions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_exp_server.application import ApplicationError
from ml_exp_server.schemas import (
    CampaignRef,
    CampaignRevision,
    CampaignRunMembership,
    ControllerConfig,
    ResearchProject,
)
from ml_exp_server.submissions import ExperimentSubmissionService, _reusable


def _expiry(delta: int) -> str:
    return (
        datetime.now(timezone.utc) + timedelta(seconds=delta)
    ).isoformat().replace("+00:00", "Z")


@pytest.mark.parametrize(("view", "expected"), [
    ({"status": "FAILED"}, False),
    ({"status": "VERIFIED"}, True),
    ({"status": "PREPARED"}, False),
    ({"status": "AUTHORIZED", "gate_expires_at": "invalid"}, False),
    ({"status": "PREPARED", "gate_expires_at": _expiry(-10)}, False),
    ({"status": "AUTHORIZED", "gate_expires_at": _expiry(60)}, True),
])
def test_reusable_submission_state_matrix(view, expected):
    assert _reusable(view) is expected


@pytest.mark.parametrize(("status", "next_action"), [
    ("PREPARED", "AUTHORIZE"),
    ("AUTHORIZED", "EXECUTE"),
    ("EXECUTING", "RECONCILE"),
    ("RECONCILE_REQUIRED", "RECONCILE"),
    ("BLOCKED", "REPREPARE"),
    ("VERIFIED", "NONE"),
])
def test_submission_view_maps_every_lifecycle_status(status, next_action):
    view = ExperimentSubmissionService._view({
        "action_id": "action-a", "execution": {"status": status},
    })
    assert view["status"] == status
    assert view["next_action"] == next_action


class Store:
    def __init__(self, snapshot=None, listed=None):
        self.value = snapshot
        self.listed = listed or []

    def snapshot(self, _submission_id):
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value

    def list_for_scope(self, _scope):
        return self.listed


def _runtime(*, store=None, project=None, row=None, cloud=None, configured=True):
    cloud = cloud or SimpleNamespace(
        enabled=False, default_credential_ref=None, entity=None,
    )
    credentials = SimpleNamespace(
        status=lambda _ref: SimpleNamespace(configured=configured),
    )

    def load(name):
        if project is None:
            raise KeyError(name)
        return project

    return SimpleNamespace(
        action_store=store or Store(listed=[]),
        config=SimpleNamespace(
            observability=SimpleNamespace(wandb_cloud=cloud),
        ),
        credential_store=credentials,
        project=load,
        index=SimpleNamespace(get_run=lambda *_args: row),
        action_service=SimpleNamespace(prepare=lambda *_args: {}),
    )


def _service(runtime):
    application = SimpleNamespace(
        authorize_action=lambda *_args: {},
        execute_action=lambda *_args: {},
        reconcile_action=lambda *_args: {},
    )
    return ExperimentSubmissionService(application, runtime)


@pytest.mark.parametrize("error", [FileNotFoundError(), ValueError("bad id")])
def test_submission_snapshot_maps_missing_or_invalid_id(error):
    service = _service(_runtime(store=Store(snapshot=error)))
    with pytest.raises(ApplicationError) as caught:
        service.get("missing")
    assert caught.value.code == "UNKNOWN_SUBMISSION"


def test_submission_snapshot_rejects_non_submission_action():
    service = _service(_runtime(store=Store(snapshot={"operation": "WRITE_CAMPAIGN"})))
    with pytest.raises(ApplicationError) as caught:
        service.get("action-a")
    assert caught.value.code == "UNKNOWN_SUBMISSION"


def test_prepare_rejects_unavailable_cloud_publication():
    cloud = SimpleNamespace(
        enabled=True, default_credential_ref="cloud", entity="team",
    )
    service = _service(_runtime(cloud=cloud, configured=False))
    with pytest.raises(ApplicationError) as caught:
        service.prepare_first_attempt(
            "demo", "run-a", max_gpu_hours=1, reason="", wandb_cloud_sync=True,
        )
    assert caught.value.code == "PUBLISHER_UNAVAILABLE"


def test_prepare_skips_expired_submission_then_reports_unknown_project():
    expired = {
        "action_id": "action-old", "operation": "SUBMIT_RUN",
        "execution": {"status": "PREPARED"},
        "gate_expires_at": _expiry(-10),
    }
    service = _service(_runtime(store=Store(listed=[expired])))
    with pytest.raises(ApplicationError) as caught:
        service.prepare_first_attempt("missing", "run-a", max_gpu_hours=1, reason="")
    assert caught.value.code == "UNKNOWN_PROJECT"


def _project(tmp_path: Path, *, controller=True, revision=True, file=None):
    campaign_file = file or tmp_path / "study.yml"
    current = None
    if revision:
        current = CampaignRevision(
            campaign="study", project="demo", revision_id="campaign.revision",
            file=str(campaign_file),
            memberships=[CampaignRunMembership(run_id="run-a", kind="materialize")],
        )
    return ResearchProject(
        project="demo", title="Demo", run_roots=[], base_dir=tmp_path,
        controller=ControllerConfig(
            python="python", experimentctl="controller.py", workdir=".",
        ) if controller else None,
        campaigns=[CampaignRef(name="study", current_revision=current)],
    )


def _prepare_error(runtime, expected):
    with pytest.raises(ApplicationError) as caught:
        _service(runtime).prepare_first_attempt(
            "demo", "run-a", max_gpu_hours=1, reason="",
        )
    assert caught.value.code == expected


def test_prepare_requires_controller_and_one_materializer(tmp_path):
    _prepare_error(_runtime(project=_project(tmp_path, controller=False)),
                   "CONTROLLER_UNAVAILABLE")
    _prepare_error(_runtime(project=_project(tmp_path, revision=False)),
                   "RUN_NOT_MATERIALIZABLE")


def test_prepare_searches_all_campaign_memberships(tmp_path):
    campaign = tmp_path / "study.yml"
    campaign.write_text("campaign: study\n")
    revision = CampaignRevision(
        campaign="study", project="demo", revision_id="campaign.revision",
        file=str(campaign), memberships=[
            CampaignRunMembership(run_id="other", kind="materialize"),
            CampaignRunMembership(run_id="run-a", kind="materialize"),
        ],
    )
    unrelated = CampaignRevision(
        campaign="unrelated", project="demo", revision_id="campaign.other",
        file=str(campaign), memberships=[
            CampaignRunMembership(run_id="other", kind="materialize"),
        ],
    )
    project = _project(tmp_path)
    project.campaigns = [
        CampaignRef(name="unrelated", current_revision=unrelated),
        CampaignRef(name="study", current_revision=revision),
    ]
    runtime = _runtime(project=project)
    runtime.action_service.prepare = lambda *_args, **_kwargs: {
        "action_id": "action-a", "operation": "SUBMIT_RUN",
        "execution": {"status": "PREPARED"},
    }
    result = _service(runtime).prepare_first_attempt(
        "demo", "run-a", max_gpu_hours=1, reason="",
    )
    assert result["submission_id"] == "action-a"


def test_prepare_resolves_relative_campaign_and_requires_existing_file(tmp_path):
    project = _project(tmp_path, file=Path("study.yml"))
    _prepare_error(_runtime(project=project), "CAMPAIGN_FILE_MISSING")


@pytest.mark.parametrize(("state", "submitted", "expected"), [
    ("RUNNING", False, "RUN_NOT_SUBMITTABLE"),
    ("NOT_SUBMITTED", True, "RUN_ALREADY_SUBMITTED"),
])
def test_prepare_rejects_observed_run_that_already_started(
    tmp_path, state, submitted, expected,
):
    campaign = tmp_path / "study.yml"
    campaign.write_text("campaign: study\n")
    row = SimpleNamespace(
        scheduler_state=state,
        attempts=[SimpleNamespace(attempt_id="attempt-001", has_submission=submitted)],
    )
    _prepare_error(_runtime(project=_project(tmp_path), row=row), expected)


def test_prepare_uses_last_unsubmitted_attempt_identity(tmp_path):
    campaign = tmp_path / "study.yml"
    campaign.write_text("campaign: study\n")
    row = SimpleNamespace(
        scheduler_state="NOT_SUBMITTED",
        attempts=[
            SimpleNamespace(attempt_id="attempt-001", has_submission=False),
            SimpleNamespace(attempt_id="attempt-002", has_submission=False),
        ],
        model_dump=lambda **_kwargs: {"run_id": "run-a"},
    )
    captured = {}
    runtime = _runtime(project=_project(tmp_path), row=row)
    runtime.action_service.prepare = (
        lambda _scope, _project, intent: captured.setdefault("intent", intent) or {}
    )
    # The test double returns the prepared intent itself as the action view.
    _service(runtime).prepare_first_attempt(
        "demo", "run-a", max_gpu_hours=1, reason="",
    )
    import yaml

    assert yaml.safe_load(captured["intent"]["draft"])["attempt_id"] == "attempt-002"
