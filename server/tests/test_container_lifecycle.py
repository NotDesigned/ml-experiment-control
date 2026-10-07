"""Managed controller lifecycle against an injected scheduler, including recovery."""

import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import yaml

from experiment_control.identity import IdentityReport
from ml_exp_server import container_controller as module
from ml_exp_server.container_controller import Controller, parse_metric
from tests.test_container_api import client, runtime


class Scheduler:
    kind = "slurm"

    def __init__(self, controller):
        self.controller = controller
        self.submissions = []
        self.recovered = None
        self.state = "RUNNING"
        self.identity_report = IdentityReport(available=True, ambiguous=False)

    def validate(self, run):
        assert run["run_id"] == self.controller.run["run_id"]

    def preflight(self, run, *, scope):
        return SimpleNamespace(require_ready=lambda: None)

    def stage(self, campaign, run, source_id, source):
        assert source.root.is_dir() and source_id == run["source_id"]
        return True

    def submission_request(self, campaign, run, attempt):
        return {"scheduler_name": run["run_id"] + "--" + attempt}

    def recover_submission(self, run, intent, attempt):
        return self.recovered or (self.controller.store.load_backend(attempt) or {}).get("backend_job_id")

    def submit(self, campaign, run, manifest, *, dry_run, intent):
        assert not dry_run and intent["submission_token"]
        self.submissions.append(manifest["attempt_id"])
        return "job-1"

    def identity(self, *args):
        return self.identity_report

    def status(self, campaign, run):
        return {"state": self.state, "backend_job_id": (self.controller.store.load_backend(self.controller.attempt_id) or {}).get("backend_job_id")}

    def collect(self, campaign, run):
        return {"backend": self.kind}

    def logs(self, campaign, run, *, tail):
        return {"stdout": ["training output"], "stderr": ["training diagnostic"], "lines": None}

    def cancel(self, campaign, run):
        self.state = "CANCELLED"
        return self.status(campaign, run)


@pytest.fixture
def controller(client):
    bundle = runtime(client)
    assert client.post("/api/projects/demo/runs", json={"run_id": "gpu", "runtime_id": bundle["runtime_id"], "executor": "gpu"}).status_code == 200
    project = client.app.state.runtime.project("demo")
    campaign = yaml.safe_load((Path(project.base_dir) / "experiments/campaigns/run-gpu.yaml").read_text())
    campaign["local_root"] = str(client.app.state.runtime.config.project_run_root_path("demo"))
    value = Controller(campaign, "gpu", "attempt-001")
    value.backend = Scheduler(value)
    return value


@pytest.mark.parametrize("value,expected", [
    ("not-json", None), (None, None), ("[]", None), ("{}", None),
    ('{"global_step":8,"loss":1.25,"flag":true,"value":NaN}', {"step": 8, "loss": 1.25}),
    ('{"step":9,"global_step":8}', {"step": 9, "global_step": 8}),
])
def test_metric_parser_preserves_only_finite_numeric_observations(value, expected):
    assert parse_metric(None, value) == expected


def test_source_binding_and_initial_status(controller):
    assert controller.status()["state"] == "NOT_SUBMITTED"
    source = controller.source()
    assert controller.source(str(source)) == source
    with pytest.raises(ValueError, match="frozen source"):
        controller.source("/wrong/source")
    assert controller.stage(str(source)) is True
    controller.check_identity()
    controller.prepare()
    assert controller.status()["state"] == "NOT_SUBMITTED"


@pytest.mark.parametrize("existing", [None, "job-recovered"])
def test_submission_recovery_never_duplicates_scheduler_call(controller, existing):
    controller.backend.recovered = existing
    first = controller.submit(None)
    second = controller.submit(None)
    assert first == second
    assert first["backend_job_id"] == (existing or "job-1")
    assert len(controller.backend.submissions) == (0 if existing else 1)


def test_dispatch_credentials_remain_outside_frozen_command(controller, monkeypatch, tmp_path):
    controller.prepare()
    original = controller.store.load_attempt("attempt-001")["command"]
    assert controller.dispatch_command({"command": original}) == original
    controller.campaign["artifact_store"] = str(tmp_path / "s3.json")
    monkeypatch.setattr("ml_exp_server.artifact_store.ArtifactStore.__init__", lambda self, *args: None)
    monkeypatch.setattr("ml_exp_server.artifact_store.ArtifactStore.issue", lambda self, *args: ("https://transfer.example/attempt", "test-write-capability", 10000))
    sealed = []
    monkeypatch.setattr("ml_exp_server.artifact_store.ArtifactStore.seal_launch", lambda self, *args: sealed.append(args[-1]) or "https://transfer.example/manifest")
    launched = controller.dispatch_command({"command": original, "attempt_id": "attempt-001"})
    assert "ML_EXPD_BOOTSTRAP_TOKEN=test-write-capability" in launched
    assert "ML_EXPD_UPLOAD_TOKEN" not in sealed[0]["environment"]
    assert "test-write-capability" not in " ".join(controller.store.load_attempt("attempt-001")["command"])
    assert controller.command("attempt-001") == ["ml-exp-worker"]
    assert controller.oci_pull_environment() == {}
    credentials = tmp_path / "pull.json"
    credentials.write_text(json.dumps({"registry": "registry.example", "username": "test-user", "password": "test-password"}))
    controller.campaign["registry_pull"] = str(credentials)
    controller.run["backend"]["oci_image"] = "other.example/image@sha256:" + "a" * 64
    assert controller.oci_pull_environment() == {}
    controller.run["backend"]["oci_image"] = "registry.example/image@sha256:" + "a" * 64
    assert controller.oci_pull_environment()["APPTAINER_DOCKER_PASSWORD"] == "test-password"


def test_unavailable_retry_without_prior_evidence_is_blocked(controller):
    controller.backend.identity_report = IdentityReport(available=False, ambiguous=False)
    with pytest.raises(ValueError, match="frozen retry"):
        controller.check_identity()
    controller.prepare()
    (controller.root / "attempts" / "unrelated").mkdir()
    with pytest.raises(ValueError, match="frozen retry"):
        controller.check_identity()


def test_collect_preserves_complete_uploaded_metrics_and_attempt_logs(controller):
    controller.submit(None)
    controller.status()
    outputs = controller.attempt / "uploaded_outputs"
    outputs.mkdir()
    (outputs / "metrics.jsonl").write_text('invalid\n{"global_step":1,"loss":2}\n{"global_step":2,"loss":1}\n')
    (outputs / "link-metrics.jsonl").symlink_to(outputs / "metrics.jsonl")
    oversized = outputs / "large" / "metrics.jsonl"
    oversized.parent.mkdir()
    with oversized.open("wb") as stream:
        stream.truncate(32 * 1024 * 1024 + 1)
    result = controller.collect()
    assert result["latest_metric"] == {"step": 2, "loss": 1}
    assert result["scheduler_state"] == "RUNNING" and result["artifacts"]["download_available"]
    assert (controller.attempt / "stdout.log").read_text() == "training output\n"
    assert (controller.root / "collection.json").is_file()


def test_cloud_collect_exact_remote_outputs_without_overwriting_latest_attempt(controller):
    controller.prepare()
    other = Controller(controller.campaign, "gpu", "attempt-002")
    other.prepare()
    other.store.begin_submission(project="demo", run_id="gpu", attempt_id="attempt-002", backend="slurm", request={"scheduler_name": "gpu--attempt-002"})
    other.store.reconcile_submission(project="demo", run_id="gpu", attempt_id="attempt-002", backend_job_id="job-latest")
    (controller.root / "collection.json").write_text('{"attempt_id":"attempt-002"}')
    controller.backend.kind = "sensecore"
    controller.run["artifact_ssh"] = "test-artifacts"
    calls = []
    controller.runner = SimpleNamespace(run=lambda command, **kwargs: calls.append(command))
    controller.backend.logs = lambda *args, **kwargs: {"lines": ["cloud output"]}
    result = controller.collect()
    assert "artifacts" not in result
    assert calls[0][-2].endswith("/attempts/attempt-001/outputs/")
    assert (controller.attempt / "stdout.log").read_text() == "cloud output\n"
    assert json.loads((controller.root / "collection.json").read_text())["attempt_id"] == "attempt-002"
    assert controller.summarize({}, controller.attempt)["latest_metric"] is None


@pytest.mark.parametrize("state,expected", [("FAILED", "REVIEW_RETRY"), ("RUNNING", "WAIT")])
def test_decision_reports_failure_and_retry_budget(controller, state, expected):
    controller.prepare()
    controller.store.write_status_payload("attempt-001", {"state": state})
    result = controller.decide()
    assert result["action"] == expected and result["retries_used"] == 0
    assert (controller.attempt / "decision.json").is_file()


@pytest.mark.parametrize("verb", ["submit-dry", "submit", "stage", "preflight", "preflight-observe", "assets-verify", "check-identity", "status", "observe", "collect", "decide", "cancel"])
def test_controller_cli_dispatches_to_injected_backend(controller, tmp_path, monkeypatch, capsys, verb):
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(controller.campaign))
    monkeypatch.setattr(module, "Controller", lambda *args: controller)
    if verb in {"observe", "cancel"}:
        controller.submit(None)
    elif verb not in {"submit", "submit-dry"}:
        controller.prepare()
    args = [str(path), "submit" if verb == "submit-dry" else "preflight" if verb == "preflight-observe" else verb, "--run", "gpu", "--local-root", str(tmp_path / "override"), "--source-id", controller.run["source_id"]]
    if verb == "preflight-observe":
        args.extend(["--scope", "observe"])
        monkeypatch.setattr(controller, "check_api_transport", lambda: pytest.fail("observation must not require gateway health"))
    if verb == "submit-dry":
        args.append("--dry-run")
    module.cli(args)
    result = json.loads(capsys.readouterr().out)[0]
    assert isinstance(result, dict)
    if verb == "cancel":
        assert result["state"] == "CANCELLED"


@pytest.mark.parametrize("failure", ["wrong-source", "cancel-unsubmitted"])
def test_controller_cli_failures_are_explicit_without_scheduler_calls(controller, tmp_path, monkeypatch, capsys, failure):
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(controller.campaign))
    monkeypatch.setattr(module, "Controller", lambda *args: controller)
    args = [str(path), "cancel" if failure == "cancel-unsubmitted" else "status", "--run", "gpu"]
    if failure == "wrong-source":
        args.extend(["--source-id", "source." + "0" * 64])
    with pytest.raises(SystemExit) as error:
        module.cli(args)
    assert error.value.code == 1 and "ValueError" in capsys.readouterr().err
    assert not controller.backend.submissions


def test_controller_module_entrypoint_reports_invalid_definition(tmp_path, monkeypatch, capsys):
    path = tmp_path / "campaign.yaml"
    path.write_text("runs: []\n")
    monkeypatch.setattr("sys.argv", ["controller", str(path), "status", "--run", "missing"])
    with pytest.warns(RuntimeWarning, match="found in sys.modules"), pytest.raises(SystemExit) as error:
        runpy.run_module(module.__name__, run_name="__main__")
    assert error.value.code == 1 and "StopIteration" in capsys.readouterr().err


def test_historical_preemption_terminal_lifecycle_survives_stop_classification_fix(controller):
    controller.submit(None)
    controller.store.write_status_payload('attempt-001', {'state': 'PREEMPTED'})
    controller.backend.kind = 'sensecore'
    controller.backend.status = lambda *args: {'run_id': 'gpu', 'backend': 'sensecore',
        'backend_job_id': 'job-1', 'state': 'CANCELLED', 'raw_state': 'SUSPENDED',
        'failure_class': None, 'reason': 'provider_stopped_cause_unknown'}
    result = controller.status()
    assert result['state'] == 'PREEMPTED' and result['failure_class'] is None
    assert result['detail']['observed_normalized_state'] == 'CANCELLED'
    assert controller.store.load_status_payload('attempt-001')['state'] == 'PREEMPTED'
