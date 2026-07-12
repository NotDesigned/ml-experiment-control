from __future__ import annotations

import json
import hashlib

import pytest

from backend_harness import QueueRunner, sensecore_run, services, slurm_run
from experiment_control.backends.sensecore import (
    SenseCoreBackend,
    digest_pinned_image,
    scheduler_job_name as sensecore_scheduler_job_name,
)
from experiment_control.backends.wyd import WydSlurmBackend
from experiment_control.project import AssetProbe, AssetRequirement
from experiment_control.runner import CommandResult


def test_sensecore_preflight_checks_cli_and_sanitized_workspace_access(tmp_path):
    fake = QueueRunner([
        CommandResult(("sco-version",), 0, "v1.2.0\n"),
        CommandResult(("safe-list",), 0, "[]\n"),
    ])
    report = SenseCoreBackend(services(tmp_path, fake)).preflight(
        sensecore_run(), scope="submit"
    )
    assert report.ready is True
    assert [check.name for check in report.checks] == [
        "sco-cli", "workspace-access",
    ]


def test_sensecore_preflight_fails_closed_on_malformed_sanitized_response(tmp_path):
    fake = QueueRunner([
        CommandResult(("sco-version",), 0, "v1.2.0\n"),
        CommandResult(
            ("safe-list",), 1, "",
            "safe_sco: input was not valid JSON; raw response suppressed",
        ),
        CommandResult(
            ("redact",), 0,
            "safe_sco: input was not valid JSON; raw response suppressed",
        ),
    ])
    report = SenseCoreBackend(services(tmp_path, fake)).preflight(
        sensecore_run(), scope="submit"
    )
    assert report.ready is False


def test_sensecore_identity_reports_consumed_exact_attempt_name(tmp_path):
    fake = QueueRunner([
        CommandResult(
            ("safe-list",), 0,
            '[{"name":"sensecore-run--attempt-001","state":"RUNNING"}]\n',
        )
    ])
    report = SenseCoreBackend(services(tmp_path, fake)).identity(
        {"campaign": "identity-test"}, sensecore_run(), "attempt-001"
    )
    assert report.available is False
    assert report.ambiguous is False
    assert report.scheduler_job_ids == ("sensecore-run--attempt-001",)


def test_sensecore_render_and_submission_request_pin_image_digest(tmp_path):
    run = sensecore_run()
    manifest = {**run, "attempt_id": "attempt-002", "command": ["python", "train.py"]}
    fake = QueueRunner([])
    backend = SenseCoreBackend(services(tmp_path, fake))

    request = backend.submission_request({}, run, "attempt-002")
    assert request["scheduler_name"] == "sensecore-run--attempt-002"
    assert request["image_reference"] == (
        f"registry.example/project/image@{run['image_id']}"
    )
    rendered = backend.render(manifest)
    assert "--name sensecore-run--attempt-002" in rendered
    assert request["image_reference"] in rendered
    assert "image:source-fixed" not in rendered
    assert backend.submit({}, run, manifest, dry_run=True) == "DRY_RUN"
    assert fake.commands == []


def test_sensecore_attempt_names_and_digest_references_are_deterministic():
    first = sensecore_scheduler_job_name("run", "attempt-001")
    assert first == "run--attempt-001"
    assert first != sensecore_scheduler_job_name("run", "attempt-002")
    assert len(sensecore_scheduler_job_name("r" * 80, "attempt-001")) <= 63
    digest = "sha256:" + "c" * 64
    assert digest_pinned_image("registry.example/ns/image:tag", digest) == (
        f"registry.example/ns/image@{digest}"
    )


@pytest.mark.parametrize(
    ("base_name", "attempt_id"),
    [
        ("Run", "attempt-001"),
        ("run_name", "attempt-001"),
        ("1run", "attempt-001"),
        ("run", "Attempt-001"),
        ("run", "attempt-001-"),
    ],
)
def test_sensecore_resource_name_rejects_invalid_values(base_name, attempt_id):
    with pytest.raises(ValueError, match="SenseCore"):
        sensecore_scheduler_job_name(base_name, attempt_id)


@pytest.mark.parametrize("method_name", ["find", "describe"])
def test_sensecore_query_errors_are_redacted(tmp_path, method_name):
    secret = "credential-value-that-must-not-escape"
    fake = QueueRunner([
        CommandResult(("safe-query",), 1, stderr=f"access_key_secret={secret}\n"),
        CommandResult(("redact",), 0, stdout="access_key_secret=<redacted>\n"),
    ])
    backend = SenseCoreBackend(services(tmp_path, fake))
    with pytest.raises(RuntimeError) as captured:
        getattr(backend, method_name)(sensecore_run(), "job--attempt-001")
    assert secret not in str(captured.value)
    assert "<redacted>" in str(captured.value)
    assert "redact-lines" in fake.commands[0][-1]


def test_slurm_preflight_checks_tools_resources_and_storage(tmp_path):
    fake = QueueRunner([
        CommandResult(("ssh-version",), 0),
        CommandResult(("rsync-version",), 0),
        CommandResult(
            ("slurm-live",), 0,
            "accelerator|up|3-00:00:00|gpu:accelerator:8\n"
            "user|lab||normal|normal\n",
        ),
        CommandResult(("runtime-storage",), 0),
    ])
    report = WydSlurmBackend(services(tmp_path, fake)).preflight(
        slurm_run(), scope="stage"
    )
    assert report.ready is True
    assert [check.name for check in report.checks] == [
        "ssh-cli", "rsync-cli", "slurm-access", "runtime-storage",
    ]


def test_slurm_observe_preflight_only_requires_control_access(tmp_path):
    fake = QueueRunner([
        CommandResult(("ssh-version",), 0),
        CommandResult(("squeue",), 0, ""),
    ])
    report = WydSlurmBackend(services(tmp_path, fake)).preflight(
        slurm_run(), scope="observe"
    )
    assert report.ready is True
    assert [check.name for check in report.checks] == ["ssh-cli", "slurm-access"]


def test_slurm_preflight_fails_before_remote_access_when_ssh_is_missing(tmp_path):
    fake = QueueRunner([
        CommandResult(("ssh-version",), 127),
        CommandResult(("rsync-version",), 0),
    ])
    report = WydSlurmBackend(services(tmp_path, fake)).preflight(
        slurm_run(), scope="submit"
    )
    assert report.ready is False
    assert len(fake.commands) == 1


def test_slurm_status_uses_injected_backend_record(tmp_path):
    fake = QueueRunner([
        CommandResult(
            ("sacct",), 0,
            "1234|backend-run|accelerator|COMPLETED|00:01:00|0:0\n",
        )
    ])
    status = WydSlurmBackend(services(tmp_path, fake)).status({}, slurm_run())
    assert status["state"] == "SUCCEEDED"
    assert status["backend_job_id"] == "1234"


def slurm_manifest(run: dict, attempt_id: str = "attempt-001") -> dict:
    return {
        **run,
        "campaign": "backend-test",
        "attempt_id": attempt_id,
        "command": ["python", "train.py"],
        "execution": {"source_mount": "/app", "workdir": "/app"},
    }


def test_slurm_submit_claims_identity_and_stages_manifest_before_script(tmp_path):
    run = slurm_run()
    manifest = slurm_manifest(run)
    (tmp_path / "manifest.yaml").write_text("run_id: backend-run\n")
    fake = QueueRunner([
        CommandResult(
            ("validate-live",), 0,
            "accelerator|up|3-00:00:00|gpu:accelerator:8\n"
            "user|lab||normal|normal\n",
        ),
        CommandResult(("claim",), 0),
        CommandResult(("manifest-rsync",), 0),
        CommandResult(("script-rsync",), 0),
        CommandResult(("sbatch",), 0, "4321\n"),
    ])
    job_id = WydSlurmBackend(services(tmp_path, fake)).submit(
        {"campaign": "backend-test"}, run, manifest, dry_run=False
    )
    assert job_id == "4321"
    assert ".submission-attempt-001" in " ".join(fake.commands[1])
    assert fake.commands[2][-1].endswith("/manifest.yaml")
    assert "controller-attempt-001.sbatch" in fake.commands[3][-1]


def test_slurm_submit_claim_blocks_duplicate_scheduler_mutation(tmp_path):
    run = slurm_run()
    manifest = slurm_manifest(run)
    fake = QueueRunner([
        CommandResult(
            ("validate-live",), 0,
            "accelerator|up|3-00:00:00|gpu:accelerator:8\n"
            "user|lab||normal|normal\n",
        ),
        CommandResult(("claim",), 1, "", "already exists"),
    ])
    with pytest.raises(FileExistsError, match="submission claim"):
        WydSlurmBackend(services(tmp_path, fake)).submit(
            {"campaign": "backend-test"}, run, manifest, dry_run=False
        )
    assert len(fake.commands) == 2


def test_slurm_recovery_rejects_multiple_matching_jobs(tmp_path):
    fake = QueueRunner([
        CommandResult(("squeue",), 0, "1732|backend-run--attempt-001|token\n"),
        CommandResult(
            ("sacct",), 0,
            "1731|backend-run--attempt-001\n1732|backend-run--attempt-001\n",
        ),
    ])
    with pytest.raises(RuntimeError, match="2 jobs match"):
        WydSlurmBackend(services(tmp_path, fake)).recover_submission(
            slurm_run(), {"submission_token": "token"}, "attempt-001"
        )


def test_slurm_identity_reports_remote_manifest_digest_match(tmp_path):
    local_manifest = tmp_path / "manifest.yaml"
    local_manifest.write_text("run_id: backend-run\n", encoding="utf-8")
    digest = hashlib.sha256(local_manifest.read_bytes()).hexdigest()
    fake = QueueRunner([
        CommandResult(("squeue",), 0, ""),
        CommandResult(("sacct",), 0, ""),
        CommandResult(("manifest",), 0, ""),
        CommandResult(("sha256sum",), 0, f"{digest}\n"),
    ])
    report = WydSlurmBackend(services(tmp_path, fake)).identity(
        {"campaign": "backend-test"}, slurm_run(), "attempt-002"
    )
    assert report.available is False
    assert report.remote_manifest_exists is True
    assert report.remote_manifest_matches is True


@pytest.mark.parametrize(
    "results",
    [
        [CommandResult(("squeue",), 255, stderr="ssh unavailable")],
        [
            CommandResult(("squeue",), 0, ""),
            CommandResult(("sacct",), 255, stderr="ssh unavailable"),
        ],
        [
            CommandResult(("squeue",), 0, ""),
            CommandResult(("sacct",), 0, ""),
            CommandResult(("manifest",), 255, stderr="ssh unavailable"),
        ],
    ],
)
def test_slurm_identity_fails_closed_when_remote_evidence_is_unavailable(
    tmp_path, results,
):
    with pytest.raises(RuntimeError, match="evidence is unavailable"):
        WydSlurmBackend(services(tmp_path, QueueRunner(results))).identity(
            {"campaign": "backend-test"}, slurm_run(), "attempt-001"
        )


def test_slurm_asset_probe_distinguishes_missing_from_transport_failure(tmp_path):
    probe = AssetProbe(
        AssetRequirement("dataset", "dataset-id", "training"),
        "/shared/dataset",
    )
    missing = WydSlurmBackend(services(
        tmp_path, QueueRunner([CommandResult(("test",), 1)])
    )).verify_assets(slurm_run(), [probe])
    assert missing["missing"][0]["identity"] == "dataset-id"

    failed = WydSlurmBackend(services(
        tmp_path,
        QueueRunner([CommandResult(("test",), 255, stderr="ssh failed")]),
    ))
    with pytest.raises(RuntimeError, match="evidence is unavailable"):
        failed.verify_assets(slurm_run(), [probe])


@pytest.mark.parametrize(
    ("resources", "gres", "error"),
    [
        ({"gpus": 2}, "gpu:accelerator:1", "does not match"),
        ({"gpus": 1, "nodes": 2}, "gpu:accelerator:1", "resources.nodes=1"),
    ],
)
def test_slurm_validation_rejects_resource_request_drift(
    tmp_path, resources, gres, error,
):
    run = slurm_run()
    run["resources"] = resources
    run["backend"]["gres"] = gres
    with pytest.raises(ValueError, match=error):
        WydSlurmBackend(services(tmp_path, QueueRunner([]))).validate(run)


def test_slurm_logs_are_bounded_and_redacted(tmp_path):
    run = slurm_run()
    run_dir = run["storage"]["run_dir"]
    fake = QueueRunner([
        CommandResult(
            ("stdout",), 0,
            f"{run_dir}/slurm-1234.out\none\rtwo\rthree\n",
        ),
        CommandResult(
            ("stderr",), 0,
            f"{run_dir}/slurm-1234.err\ntoken=top-secret\nfailure\n",
        ),
    ])
    logs = WydSlurmBackend(services(tmp_path, fake)).logs({}, run, tail=2)
    assert logs["stdout"] == ["two", "three"]
    assert logs["stderr"] == ["token=<redacted>", "failure"]
    assert "top-secret" not in json.dumps(logs)


@pytest.mark.parametrize("tail", [0, 10001])
def test_slurm_logs_reject_unbounded_tail_before_remote_access(tmp_path, tail):
    fake = QueueRunner([])
    with pytest.raises(ValueError, match="tail must be between"):
        WydSlurmBackend(services(tmp_path, fake)).logs({}, slurm_run(), tail=tail)
    assert fake.commands == []


def test_slurm_collection_includes_sanitized_process_evidence(tmp_path):
    run = slurm_run()
    run_dir = run["storage"]["run_dir"]
    fake = QueueRunner([
        CommandResult(("collect-rsync",), 0),
        CommandResult(("checkpoint-probe",), 0),
        CommandResult(("stdout",), 1),
        CommandResult(
            ("stderr",), 0,
            f"{run_dir}/slurm-1234.err\n"
            "access_key_secret=do-not-persist\n"
            "ModuleNotFoundError: No module named 'dependency'\n",
        ),
    ])
    summary = WydSlurmBackend(services(tmp_path, fake)).collect({}, run)
    evidence = summary["process_evidence"]
    assert evidence["observed"] is True
    assert evidence["stdout_tail"] == []
    assert evidence["stderr_tail"] == [
        "access_key_secret=<redacted>",
        "ModuleNotFoundError: No module named 'dependency'",
    ]


def test_slurm_collection_reports_latest_completed_checkpoint(tmp_path):
    fake = QueueRunner([
        CommandResult(("collect-rsync",), 0),
        CommandResult(("checkpoint-probe",), 0, "checkpoint_8\ncheckpoint_21\n"),
        CommandResult(("stdout",), 1),
        CommandResult(("stderr",), 1),
    ])
    summary = WydSlurmBackend(services(tmp_path, fake)).collect({}, slurm_run())
    assert summary["latest_completed_checkpoint"].endswith("/checkpoint_21")
    assert summary["latest_completed_checkpoint_step"] == 21


def test_sensecore_logs_classify_expired_stream(tmp_path):
    resource_name = "sensecore-run--attempt-002"
    fake = QueueRunner([
        CommandResult(
            ("stream",), 1,
            stderr="real-time job logs have expired (403); token=secret\n",
        ),
        CommandResult(
            ("redact",), 0,
            stdout="real-time job logs have expired (403); token=<redacted>\n",
        ),
    ])
    backend = SenseCoreBackend(services(
        tmp_path, fake,
        record={"attempt_id": "attempt-002", "backend_job_id": resource_name},
    ))
    logs = backend.logs({}, sensecore_run(), tail=5)
    assert logs["expired"] is True
    assert "secret" not in "\n".join(logs["lines"])
    assert logs["backend_job_id"] == resource_name


def test_sensecore_submit_checks_exact_created_job(tmp_path):
    run = sensecore_run()
    fake = QueueRunner([
        CommandResult(("safe-list",), 0, "[]\n"),
        CommandResult(("sco-create",), 0, ""),
        CommandResult(("safe-describe",), 0, json.dumps({
            "name": "sensecore-run--attempt-001",
            "state": "WAITING",
            "normalized_state": "QUEUED",
        })),
    ])
    job_id = SenseCoreBackend(services(tmp_path, fake)).submit(
        {}, run,
        {**run, "attempt_id": "attempt-001", "command": ["python", "train.py"]},
        dry_run=False,
    )
    assert job_id == "sensecore-run--attempt-001"
    assert any(run["image_id"] in argument for argument in fake.commands[1])


def test_sensecore_cancel_preserves_terminal_preemption(tmp_path, monkeypatch):
    writes = []
    service_bundle = services(tmp_path, QueueRunner([]))
    backend = SenseCoreBackend(type(service_bundle)(
        service_bundle.run_command,
        service_bundle.local_run_dir,
        service_bundle.backend_record,
        service_bundle.summarize_run,
        service_bundle.parse_metric,
        service_bundle.parse_checkpoint,
        lambda *args, **kwargs: writes.append((args, kwargs)),
        service_bundle.utc_now,
    ))
    monkeypatch.setattr(backend, "status", lambda _campaign, _run: {
        "state": "PREEMPTED", "raw_state": "SUSPENDED", "backend_job_id": "job",
    })
    result = backend.cancel({}, {"run_id": "run", "backend": {}})
    assert result["state"] == "PREEMPTED"
    assert writes == []


@pytest.mark.parametrize(
    ("phase", "expected"),
    [("Pending", "PENDING"), ("Running", "ALLOCATED"), ("Deleted", "RELEASED")],
)
def test_sensecore_worker_query_is_sanitized_and_normalized(
    tmp_path, phase, expected,
):
    resource_name = "sensecore-run--attempt-001"
    fake = QueueRunner([
        CommandResult(("workers",), 0, json.dumps([{
            "worker_name": "worker-0", "resource": "4 accelerators",
            "phase": phase,
        }]))
    ])
    backend = SenseCoreBackend(services(
        tmp_path, fake,
        record={"attempt_id": "attempt-001", "backend_job_id": resource_name},
    ))
    result = backend.workers({}, sensecore_run())
    assert result["worker_state"] == expected
    assert "worker-list" in fake.commands[0][-1]
