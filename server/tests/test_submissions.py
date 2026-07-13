"""First-class experiment submission lifecycle and reconciliation."""

from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from ml_exp_server.actions.service import ActionService
from ml_exp_server.api.app import create_app
from ml_exp_server.project_config import load_research_project
from ml_exp_server.schemas import ActionRuntimeConfig, ServerConfig


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
        agent_root=str(tmp_path / "agents"),
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


def test_unmaterialized_experiment_has_first_class_submission_lifecycle(tmp_path):
    app, runner = _app(tmp_path)
    with TestClient(app) as client:
        assert client.app.state.index.get_run("demo", "run-a") is None
        prepared = client.post(
            "/api/experiments/demo/run-a/submissions/prepare",
            json={
                "max_gpu_hours": 2,
                "reason": "run the authored candidate",
                "approval_note": "reviewed campaign and budget",
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
                "approval_note": "same reviewed intent",
            },
        ).json()
        assert repeated["submission_id"] == submission["submission_id"]
        assert repeated["reused"] is True
        conflict = client.post(
            "/api/experiments/demo/run-a/submissions/prepare",
            json={"max_gpu_hours": 3, "approval_note": "changed budget"},
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
            json={"max_gpu_hours": 2, "approval_note": "reviewed"},
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
