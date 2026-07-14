"""First-class experiment submission lifecycle and reconciliation."""

from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from ml_exp_server.authored_runs import authored_run_placeholder
from ml_exp_server.actions.service import ActionService
from ml_exp_server.api.app import create_app
from ml_exp_server.campaign_lifecycle import campaign_record_path
from ml_exp_server.project_config import load_research_project
from ml_exp_server.schemas import (
    ActionRuntimeConfig,
    CampaignRef,
    CampaignRevision,
    CampaignRunMembership,
    ResearchProject,
    RunIndexRow,
    ServerConfig,
)


class SubmissionController:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.status_visible = True

    def __call__(self, command, *, cwd, timeout):
        self.calls.append(list(command))
        verb = command[3]
        if verb == "submit" and "--dry-run" in command:
            campaign = yaml.safe_load(Path(command[2]).read_text())
            manifest_path = (
                Path(campaign["local_root"]) / "study" / "run-a" / "manifest.yaml"
            )
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(yaml.safe_dump({
                "identity_version": 2,
                "run_id": "run-a",
                "source_id": "git:abc",
                "image_id": "sha256:image",
                "backend": {"kind": "slurm", "time": "01:00:00"},
                "resources": {"gpus": 2, "cpus": 8},
                "storage": {
                    "run_dir": "/data/demo/runs/study/run-a",
                    "checkpoint_dir": "/data/demo/runs/study/run-a/checkpoints",
                },
                "command": ["python", "train.py"],
                "assets": [{"kind": "dataset", "identity": "dataset-v1"}],
                "checkpoint": {
                    "expected_first_minutes": 5,
                    "max_uncheckpointed_minutes": 10,
                },
            }))
            return self._result([{
                "run_id": "run-a", "manifest_path": str(manifest_path),
                "scheduler_mutated": False,
            }])
        if verb == "submit":
            return self._result([{
                "run_id": "run-a", "attempt_id": "attempt-001",
                "backend_job_id": "job-42",
            }])
        if verb == "status":
            if not self.status_visible:
                return self._result([])
            return self._result([{
                "run_id": "run-a", "attempt_id": "attempt-001",
                "backend_job_id": "job-42", "state": "QUEUED",
            }])
        return self._result([{"ready": True}])

    @staticmethod
    def _result(payload):
        return {
            "returncode": 0, "timeout": False, "payload": payload,
            "stdout": "", "stderr": "",
        }


def _app(tmp_path):
    experiments = tmp_path / "science" / "experiments"
    experiments.mkdir(parents=True)
    campaign = experiments / "study.yml"
    campaign.write_text(yaml.safe_dump({
        "schema_version": 1,
        "project": "demo",
        "campaign": "study",
        "local_root": "outputs/runs",
        "runs": [{"run_id": "run-a", "research_role": "candidate"}],
    }, sort_keys=False))
    project_file = experiments / "research_project.yaml"
    project_file.write_text(yaml.safe_dump({
        "schema_version": 1,
        "project": "demo",
        "title": "Demo",
        "run_roots": ["outputs/runs"],
        "campaigns": [{"name": "study", "file": str(campaign)}],
        "controller": {
            "python": "python",
            "experimentctl": "tools/experimentctl.py",
            "workdir": ".",
            "capabilities": {"submit_outbox": True, "run_identity_v2": True},
        },
    }, sort_keys=False))
    project = load_research_project(project_file)
    config = ServerConfig(
        index_db=str(tmp_path / "index.sqlite"),
        action_root=str(tmp_path / "actions"),
        collector_enabled=False,
        action_runtime=ActionRuntimeConfig(allow_scheduler_mutations=True),
    )
    app = create_app(config, projects=[project])
    runner = SubmissionController()
    app.state.runtime.action_service = ActionService(
        app.state.runtime.action_store,
        config.action_runtime,
        runner,
        actor_provider=lambda: "trusted:operator",
    )
    return app, runner


def test_authored_unmaterialized_run_is_selectable_from_tui_read_model(tmp_path):
    app, _ = _app(tmp_path)
    with TestClient(app) as client:
        assert client.app.state.index.get_run("demo", "run-a") is None

        snapshot = client.get("/api/terminal/snapshot").json()
        row = next(item for item in snapshot["runs"]["demo"]
                   if item["run_id"] == "run-a")
        assert row["scheduler_state"] == "NOT_SUBMITTED"
        assert row["run_dir"] == ""
        assert row["provenance"]["authored_only"] is True
        assert row["campaign_memberships"][0]["membership"]["kind"] == "materialize"

        operations = client.get("/api/operations", params={
            "project": "demo", "scope_type": "run", "object_id": "run-a",
        })
        assert operations.status_code == 200
        submit = next(item for item in operations.json()
                      if item["operation"]["operation_id"] == "run.submit")
        assert submit["status"] == "AVAILABLE"

        unknown = client.get("/api/operations", params={
            "project": "demo", "scope_type": "run", "object_id": "missing-run",
        })
        assert unknown.status_code == 404
        assert unknown.headers["X-ML-Expd-Error-Code"] == "UNKNOWN_RUN"
        missing_attempt = client.get("/api/operations", params={
            "project": "demo", "scope_type": "attempt",
            "object_id": "run-a::attempt-001",
        })
        assert missing_attempt.status_code == 404
        assert missing_attempt.headers["X-ML-Expd-Error-Code"] == "UNKNOWN_RUN"

        # Durable observed evidence always wins over the authored placeholder.
        client.app.state.index.upsert_run(RunIndexRow(
            project="demo", campaign="study", run_id="run-a",
            run_dir=str(tmp_path / "materialized"), scheduler_state="CREATED",
            provenance={"observed": True},
        ))
        refreshed = client.get("/api/terminal/snapshot").json()
        rows = [item for item in refreshed["runs"]["demo"]
                if item["run_id"] == "run-a"]
        assert len(rows) == 1
        assert rows[0]["scheduler_state"] == "CREATED"
        assert rows[0]["provenance"] == {"observed": True}


def test_reuse_only_membership_does_not_create_authored_run_placeholder():
    project = ResearchProject(
        project="demo", title="Demo", run_roots=[], campaigns=[CampaignRef(
            name="reuse-study", file="reuse.yml",
            current_revision=CampaignRevision(
                campaign="reuse-study", project="demo",
                revision_id="campaign." + "a" * 64, file="reuse.yml",
                memberships=[CampaignRunMembership(
                    run_id="historical-run", kind="reuse",
                )],
            ),
        )],
    )
    assert authored_run_placeholder(project, "historical-run") is None


def test_archived_authored_run_is_visible_but_submit_is_blocked(tmp_path):
    app, _ = _app(tmp_path)
    project = app.state.runtime.project("demo")
    revision = project.campaigns[0].current_revision
    assert revision is not None
    record = campaign_record_path(project, "study", revision.revision_id, "archive")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(yaml.safe_dump({
        "project": "demo", "campaign": "study",
        "revision_id": revision.revision_id, "reason": "retired",
    }))

    with TestClient(app) as client:
        snapshot = client.get("/api/terminal/snapshot").json()
        assert any(item["run_id"] == "run-a" for item in snapshot["runs"]["demo"])
        operations = client.get("/api/operations", params={
            "project": "demo", "scope_type": "run", "object_id": "run-a",
        }).json()
        submit = next(item for item in operations
                      if item["operation"]["operation_id"] == "run.submit")
        assert submit["status"] == "BLOCKED"
        assert any("lifecycle=ARCHIVED" in reason for reason in submit["reasons"])


def test_unmaterialized_experiment_has_first_class_submission_lifecycle(tmp_path):
    app, runner = _app(tmp_path)
    with TestClient(app) as client:
        assert client.app.state.index.get_run("demo", "run-a") is None
        prepared = client.post(
            "/api/experiments/demo/run-a/submissions/prepare",
            json={
                "max_gpu_hours": 2,
                "reason": "run the authored candidate",
            },
        )
        assert prepared.status_code == 200
        submission = prepared.json()
        assert submission["status"] == "PREPARED"
        assert submission["attempt_id"] == "attempt-001"
        assert submission["next_action"] == "AUTHORIZE"
        assert submission["preflight_summary"]["source_id"] == "git:abc"

        repeated = client.post(
            "/api/experiments/demo/run-a/submissions/prepare",
            json={
                "max_gpu_hours": 2,
            },
        ).json()
        assert repeated["submission_id"] == submission["submission_id"]
        assert repeated["reused"] is True
        conflict = client.post(
            "/api/experiments/demo/run-a/submissions/prepare",
            json={"max_gpu_hours": 3},
        )
        assert conflict.status_code == 409
        assert conflict.headers["X-ML-Expd-Error-Code"] == "SUBMISSION_INTENT_EXISTS"

        listed = client.get("/api/experiments/demo/run-a/submissions").json()
        assert [item["submission_id"] for item in listed["submissions"]] == [
            submission["submission_id"],
        ]
        authorized = client.post(
            f"/api/submissions/{submission['submission_id']}/authorize",
            json={"note": "scheduler mutation approved"},
        ).json()
        assert authorized["status"] == "AUTHORIZED"
        executed = client.post(
            f"/api/submissions/{submission['submission_id']}/execute",
            json={"confirmation": submission["confirmation"]},
        ).json()
        assert executed["status"] == "VERIFIED"
        assert executed["execution"]["result"]["observation"]["state"] == "QUEUED"
        live_submits = [
            call for call in runner.calls if call[3] == "submit" and "--dry-run" not in call
        ]
        assert len(live_submits) == 1


def test_uncertain_submission_reconciles_by_status_without_resubmitting(tmp_path):
    app, runner = _app(tmp_path)
    runner.status_visible = False
    with TestClient(app) as client:
        prepared = client.post(
            "/api/experiments/demo/run-a/submissions/prepare",
            json={"max_gpu_hours": 2},
        ).json()
        client.post(
            f"/api/submissions/{prepared['submission_id']}/authorize",
            json={"note": "approved"},
        )
        uncertain = client.post(
            f"/api/submissions/{prepared['submission_id']}/execute",
            json={"confirmation": prepared["confirmation"]},
        ).json()
        assert uncertain["status"] == "RECONCILE_REQUIRED"

        runner.status_visible = True
        reconciled = client.post(
            f"/api/submissions/{prepared['submission_id']}/reconcile",
        ).json()
        assert reconciled["status"] == "VERIFIED"
        assert reconciled["execution"]["result"]["observation"]["backend_job_id"] == "job-42"
        live_submits = [
            call for call in runner.calls if call[3] == "submit" and "--dry-run" not in call
        ]
        assert len(live_submits) == 1
