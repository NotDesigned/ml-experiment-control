"""First-class experiment submission lifecycle and reconciliation."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from ml_exp_server.authored_runs import authored_run_placeholder
from ml_exp_server.actions.service import ActionService
from ml_exp_server.api.app import create_app
from ml_exp_server.application import ExperimentServerApplication
from ml_exp_server.observability_store import (
    AttemptRef, OutboxRecord, SourceRef,
)
from ml_exp_server.campaign_lifecycle import campaign_record_path
from ml_exp_server.project_config import load_research_project
from ml_exp_server.schemas import (
    ActionRuntimeConfig,
    AttemptSummary,
    CampaignRef,
    CampaignRevision,
    CampaignRunMembership,
    ObservabilityConfig,
    ResearchProject,
    RunIndexRow,
    ServerConfig,
    WandbCloudConfig,
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


def _app(tmp_path, *, cloud=False, observability_mutations=False):
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
            "capabilities": {
                "submit_outbox": True,
                "run_identity_v2": True,
                "cancel_outbox": True,
            },
        },
    }, sort_keys=False))
    project = load_research_project(project_file)
    config = ServerConfig(
        index_db=str(tmp_path / "index.sqlite"),
        action_root=str(tmp_path / "actions"),
        collector_enabled=False,
        action_runtime=ActionRuntimeConfig(
            allow_scheduler_mutations=True,
            allow_observability_mutations=observability_mutations,
        ),
        observability=ObservabilityConfig(
            credential_root=str(tmp_path / "credentials"),
            log_archive_root=str(tmp_path / "archive"),
            wandb_cloud=WandbCloudConfig(
                enabled=cloud,
                default_credential_ref="cloud-primary" if cloud else None,
                entity="research-team" if cloud else None,
            ),
        ),
    )
    app = create_app(config, projects=[project])
    runner = SubmissionController()

    def configure_runtime(runtime):
        runtime.action_service = ActionService(
            runtime.action_store,
            config.action_runtime,
            runner,
            actor_provider=lambda: "trusted:operator",
            internal_executor=runtime.action_service.internal_executor,
        )

    app.state.runtime_initializers.append(configure_runtime)
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
        # The cloud control remains absent until daemon-host policy and its
        # credential are both ready.
        parameter_keys = {
            item["key"] for item in submit["operation"]["parameters"]
        }
        assert parameter_keys == {"max_gpu_hours"}

        unsupported_cloud = client.post("/api/operations/direct", json={
            "project": "demo",
            "scope_type": "run",
            "object_id": "run-a",
            "operation_id": "run.submit",
            "parameters": {
                "max_gpu_hours": 2,
                "wandb_cloud_sync": "yes",
            },
        })
        assert unsupported_cloud.status_code == 409
        assert unsupported_cloud.headers["X-ML-Expd-Error-Code"] == "INVALID_OPERATION"
        assert "wandb_cloud_sync" in unsupported_cloud.json()["detail"]
        assert not list((tmp_path / "actions").glob("action-*"))

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


def test_cloud_ready_daemon_exposes_submission_option_without_secret_metadata(tmp_path):
    app, _ = _app(tmp_path, cloud=True)
    with TestClient(app) as client:
        client.app.state.runtime.credential_store.set_wandb_api_key(
            "cloud-primary", "secret-cloud-key",
        )
        operations = client.get("/api/operations", params={
            "project": "demo", "scope_type": "run", "object_id": "run-a",
        }).json()
        submit = next(item for item in operations
                      if item["operation"]["operation_id"] == "run.submit")
        parameters = {item["key"]: item for item in submit["operation"]["parameters"]}
        assert set(parameters) == {"max_gpu_hours", "wandb_cloud_sync"}
        assert parameters["wandb_cloud_sync"]["default"] == "no"

        prepared = client.post("/api/operations/direct", json={
            "project": "demo",
            "scope_type": "run",
            "object_id": "run-a",
            "operation_id": "run.submit",
            "parameters": {"max_gpu_hours": 2, "wandb_cloud_sync": "yes"},
        })
        assert prepared.status_code == 200
        encoded = repr(prepared.json())
        assert "secret-cloud-key" not in encoded
        assert "cloud-primary" not in encoded
        action_id = prepared.json()["action"]["action_id"]
        # Preparing/authorizing the scheduler Action does not start a mirror.
        assert client.get(
            "/api/observability/attempts/demo/run-a/attempt-001",
        ).json()["targets"] == []
        authorized = client.post("/api/actions/authorize", json={
            "action_id": action_id, "note": "reviewed",
        })
        assert authorized.status_code == 200
        executed = client.post("/api/actions/execute", json={
            "action_id": action_id, "confirmation": f"EXECUTE {action_id}",
        })
        assert executed.status_code == 200, executed.text
        target = client.get(
            "/api/observability/attempts/demo/run-a/attempt-001",
        ).json()["targets"]
        assert target[0]["target"] == "cloud"
        assert target[0]["state"] == "PENDING"

        # Restart reconciliation closes the crash window between persisting a
        # VERIFIED Action and activating its target.
        store = client.app.state.runtime.observability_store
        with store._lock:
            store._conn.execute("DELETE FROM publication_targets")
            store._conn.commit()
        ExperimentServerApplication(
            client.app.state.runtime,
        ).recover_observability_policies()
        recovered = client.get(
            "/api/observability/attempts/demo/run-a/attempt-001",
        ).json()["targets"]
        assert recovered[0]["target"] == "cloud"


def test_active_submission_cannot_change_cloud_policy(tmp_path):
    app, _ = _app(tmp_path, cloud=True)
    with TestClient(app) as client:
        client.app.state.runtime.credential_store.set_wandb_api_key(
            "cloud-primary", "secret-cloud-key",
        )
        prepared = client.post(
            "/api/experiments/demo/run-a/submissions/prepare",
            json={"max_gpu_hours": 2, "wandb_cloud_sync": True},
        )
        assert prepared.status_code == 200
        conflict = client.post(
            "/api/experiments/demo/run-a/submissions/prepare",
            json={"max_gpu_hours": 2, "wandb_cloud_sync": False},
        )
        assert conflict.status_code == 409
        assert conflict.headers["X-ML-Expd-Error-Code"] == "SUBMISSION_INTENT_EXISTS"


def test_reconcile_required_does_not_activate_cloud_target(tmp_path):
    app, runner = _app(tmp_path, cloud=True)
    with TestClient(app) as client:
        client.app.state.runtime.credential_store.set_wandb_api_key(
            "cloud-primary", "secret-cloud-key",
        )
        prepared = client.post(
            "/api/experiments/demo/run-a/submissions/prepare",
            json={"max_gpu_hours": 2, "wandb_cloud_sync": True},
        ).json()
        action_id = prepared["submission_id"]
        client.post(
            f"/api/submissions/{action_id}/authorize",
            json={"note": "reviewed"},
        )
        runner.status_visible = False
        result = client.post(
            f"/api/submissions/{action_id}/execute",
            json={"confirmation": f"EXECUTE {action_id}"},
        )
        assert result.status_code == 200
        assert result.json()["status"] == "RECONCILE_REQUIRED"
        targets = client.get(
            "/api/observability/attempts/demo/run-a/attempt-001",
        ).json()["targets"]
        assert targets == []


def test_audited_observability_backfill_operation_rewinds_exact_attempt(tmp_path):
    app, _ = _app(
        tmp_path, cloud=True, observability_mutations=True,
    )
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        runtime.credential_store.set_wandb_api_key(
            "cloud-primary", "secret-cloud-key",
        )
        attempt = AttemptRef(
            runtime.workspace_id, "demo", "run-a", "attempt-001",
        )
        source = SourceRef(attempt, "metrics")
        runtime.observability_store.enqueue_and_advance(
            source, expected=None, generation="g", byte_offset=4,
            records=[OutboxRecord("record-1", "metrics", {"step": 1})],
            targets=[], now=1,
        )
        client.app.state.index.upsert_run(RunIndexRow(
            project="demo", campaign="study", run_id="run-a",
            run_dir=str(tmp_path / "run-a"), scheduler_state="SUCCEEDED",
            attempts=[AttemptSummary(attempt_id="attempt-001", state="SUCCEEDED")],
        ))
        operations = client.get("/api/operations", params={
            "project": "demo", "scope_type": "run", "object_id": "run-a",
        }).json()
        backfill = next(
            item for item in operations
            if item["operation"]["operation_id"] == "observability.backfill"
        )
        assert backfill["status"] == "AVAILABLE"
        assert backfill["operation"]["parameters"][0]["choices"] == [
            ["W&B Cloud", "cloud"],
        ]
        prepared = client.post("/api/operations/direct", json={
            "project": "demo", "scope_type": "run", "object_id": "run-a",
            "operation_id": "observability.backfill",
            "parameters": {"target": "cloud", "reason": "publish historical evidence"},
        })
        assert prepared.status_code == 200
        action_id = prepared.json()["action"]["action_id"]
        assert prepared.json()["preflight"] == {
            "target": "cloud", "attempt_count": 1,
        }
        assert client.post("/api/actions/authorize", json={
            "action_id": action_id, "note": "approved historical publication",
        }).status_code == 200
        executed = client.post("/api/actions/execute", json={
            "action_id": action_id, "confirmation": f"EXECUTE {action_id}",
        })
        assert executed.status_code == 200, executed.text
        assert executed.json()["execution"]["status"] == "VERIFIED"
        assert executed.json()["execution"]["result"] == {
            "target": "cloud", "attempt_count": 1, "rewound_attempts": 1,
        }
        assert app.state.runtime.observability_store.get_cursor(source) is None
        targets = client.get(
            "/api/observability/attempts/demo/run-a/attempt-001",
        ).json()["targets"]
        assert targets[0]["target"] == "cloud"
        assert targets[0]["state"] == "PENDING"


def test_observability_backfill_defaults_closed_by_daemon_policy(tmp_path):
    app, _ = _app(tmp_path, cloud=True)
    with TestClient(app) as client:
        client.app.state.runtime.credential_store.set_wandb_api_key(
            "cloud-primary", "secret-cloud-key",
        )
        client.app.state.index.upsert_run(RunIndexRow(
            project="demo", campaign="study", run_id="run-a",
            run_dir=str(tmp_path / "run-a"), scheduler_state="SUCCEEDED",
            attempts=[AttemptSummary(attempt_id="attempt-001", state="SUCCEEDED")],
        ))
        operations = client.get("/api/operations", params={
            "project": "demo", "scope_type": "run", "object_id": "run-a",
        }).json()
    backfill = next(
        item for item in operations
        if item["operation"]["operation_id"] == "observability.backfill"
    )
    assert backfill["status"] == "BLOCKED"
    assert "Observability mutations are disabled by daemon policy" in backfill["reasons"]


def test_nested_materialized_run_replaces_placeholder_and_exposes_exact_cancel(
    tmp_path,
):
    app, _ = _app(tmp_path)
    with TestClient(app) as client:
        project = client.app.state.runtime.project("demo")
        revision = project.campaigns[0].current_revision
        assert revision is not None
        run_dir = (
            project.resolved_run_roots()[0]
            / "state" / "instance-a" / "study" / "run-a"
        )
        attempt_dir = run_dir / "attempts" / "attempt-001"
        attempt_dir.mkdir(parents=True)
        (run_dir / "manifest.yaml").write_text(yaml.safe_dump({
            "project": "demo",
            "campaign": "study",
            "campaign_id": revision.revision_id,
            "run_id": "run-a",
            "research_role": "candidate",
            "source_id": "git:abc",
            "image_id": "sha256:image",
        }))
        status = {
            "project": "demo",
            "run_id": "run-a",
            "attempt_id": "attempt-001",
            "backend": "sensecore",
            "backend_job_id": "exact-job-42",
            "state": "RUNNING",
        }
        (run_dir / "status.json").write_text(json.dumps(status))
        (attempt_dir / "status.json").write_text(json.dumps(status))
        (attempt_dir / "backend.json").write_text(json.dumps({
            "attempt_id": "attempt-001",
            "backend": "sensecore",
            "backend_job_id": "exact-job-42",
        }))
        (attempt_dir / "submission.json").write_text(json.dumps({
            "attempt_id": "attempt-001",
            "backend": "sensecore",
            "backend_job_id": "exact-job-42",
            "state": "SUBMITTED",
            "submission_token": "a" * 32,
        }))
        refreshed = client.post(
            "/api/terminal/refresh", json={"project": "demo"},
        ).json()
        rows = [
            item for item in refreshed["runs"]["demo"]
            if item["run_id"] == "run-a"
        ]
        assert len(rows) == 1
        assert rows[0]["scheduler_state"] == "RUNNING"
        assert rows[0]["provenance"].get("authored_only") is None
        assert rows[0]["attempts"] == [{
            "attempt_id": "attempt-001",
            "state": "RUNNING",
            "backend": "sensecore",
            "backend_job_id": "exact-job-42",
            "decision": {},
            "has_submission": True,
        }]

        run_operations = client.get("/api/operations", params={
            "project": "demo", "scope_type": "run", "object_id": "run-a",
        }).json()
        submit = next(
            item for item in run_operations
            if item["operation"]["operation_id"] == "run.submit"
        )
        assert submit["status"] == "BLOCKED"

        attempt_operations = client.get("/api/operations", params={
            "project": "demo", "scope_type": "attempt",
            "object_id": "run-a::attempt-001",
        }).json()
        cancel = next(
            item for item in attempt_operations
            if item["operation"]["operation_id"] == "attempt.cancel"
        )
        assert cancel["status"] == "AVAILABLE"


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
    with TestClient(app) as client:
        project = client.app.state.runtime.project("demo")
        revision = project.campaigns[0].current_revision
        assert revision is not None
        record = campaign_record_path(project, "study", revision.revision_id, "archive")
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(yaml.safe_dump({
            "project": "demo", "campaign": "study",
            "revision_id": revision.revision_id, "reason": "retired",
        }))
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
