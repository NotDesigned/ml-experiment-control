from __future__ import annotations

import json
import hashlib
import subprocess

import pytest

from backend_harness import (
    SUBMISSION_TOKEN,
    QueueRunner,
    sensecore_run,
    services,
    slurm_run,
    submission_intent,
)
from experiment_control.backends.sensecore import (
    SenseCoreBackend,
    digest_pinned_image,
    scheduler_job_name as sensecore_scheduler_job_name,
    submission_resource_name,
)
from experiment_control.backends.wyd import (
    WydSlurmBackend,
)
from experiment_control.project import AssetProbe, AssetRequirement
from experiment_control.runner import CommandResult


def test_sensecore_availability_uses_rest_without_cli(tmp_path):
    fake = QueueRunner([CommandResult(("workspaces",), 0, "{}")])
    report = SenseCoreBackend(services(tmp_path, fake)).availability()
    assert report.ready
    assert [check.name for check in report.checks] == ["sensecore-rest"]
    assert fake.commands[0][0] == "REST"


def test_sensecore_availability_reports_safe_authentication_failure(tmp_path):
    report = SenseCoreBackend(services(tmp_path, QueueRunner([
        CommandResult(("workspaces",), 1, stderr="raw secret"),
    ]))).availability()
    assert not report.ready
    assert "secret" not in report.checks[0].message


def test_slurm_availability_checks_all_local_transport_tools(tmp_path):
    ready = WydSlurmBackend(services(tmp_path, QueueRunner([
        CommandResult(("ssh",), 0), CommandResult(("rsync",), 0),
    ]))).availability()
    assert ready.ready is True

    missing = WydSlurmBackend(services(tmp_path, QueueRunner([
        CommandResult(("ssh",), 127), CommandResult(("rsync",), 127),
    ]))).availability()
    assert missing.ready is False
    assert [check.name for check in missing.checks] == ["ssh-cli", "rsync-cli"]


def test_sensecore_preflight_uses_rest_exact_query(tmp_path):
    fake = QueueRunner([CommandResult(("find",), 0, "[]")])
    report = SenseCoreBackend(services(tmp_path, fake)).preflight(sensecore_run(), scope="submit")
    assert report.ready
    assert fake.commands == [("REST", "find", "workspace", "sensecore-run")]


def test_sensecore_preflight_fails_closed_on_malformed_or_failed_response(tmp_path):
    for result in [CommandResult(("find",), 0, "{}"), CommandResult(("find",), 1)]:
        report = SenseCoreBackend(services(tmp_path, QueueRunner([result]))).preflight(sensecore_run(), scope="observe")
        assert not report.ready


def test_sensecore_preflight_observation_does_not_require_allocatable_spec(tmp_path):
    backend = SenseCoreBackend(services(tmp_path, QueueRunner([CommandResult(("find",),0,"[]")])))
    assert backend.preflight(sensecore_run(), scope="observe").ready
    backend = SenseCoreBackend(services(tmp_path, QueueRunner([CommandResult(("find",),0,"[]")])))
    backend.rest.specs = lambda _backend: []
    assert not backend.preflight(sensecore_run(), scope="submit").ready


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
    assert json.loads(rendered)["training_job"]["name"] == "sensecore-run--attempt-002"
    assert request["image_reference"] in rendered
    assert "image:source-fixed" not in rendered
    assert backend.submit({}, run, manifest, dry_run=True) == "DRY_RUN"
    assert fake.commands == []


def test_sensecore_attempt_names_and_digest_references_are_deterministic():
    first = sensecore_scheduler_job_name("run", "attempt-001")
    assert first == "run--attempt-001"
    assert first != sensecore_scheduler_job_name("run", "attempt-002")
    assert len(sensecore_scheduler_job_name("r" * 80, "attempt-001")) <= 63
    bound = submission_resource_name("run", "attempt-001", SUBMISSION_TOKEN)
    assert bound == f"run--attempt-001--{SUBMISSION_TOKEN}"
    assert bound != submission_resource_name("run", "attempt-001", "f" * 32)
    assert len(submission_resource_name("r" * 80, "attempt-001", SUBMISSION_TOKEN)) <= 63
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
    assert "REST request failed" in str(captured.value)
    assert fake.commands[0][0] == "REST"


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
    assert status["observation_source"] == "sacct"
    assert status["observed_at"]


def test_slurm_status_queries_queue_reason_when_accounting_is_empty(tmp_path):
    fake = QueueRunner([
        CommandResult(("sacct",), 0, ""),
        CommandResult(
            ("squeue",), 0,
            "1234|backend-run|accelerator|PENDING|00:00|0:0|Priority\n",
        ),
    ])
    status = WydSlurmBackend(services(tmp_path, fake)).status({}, slurm_run())
    assert status["state"] == "QUEUED"
    assert status["reason"] == "Priority"
    assert status["detail"] == {"pending_reason": "Priority"}
    assert status["observation_source"] == "squeue"
    assert "%R" in fake.commands[1][-1]


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
    backend = WydSlurmBackend(services(tmp_path, fake))
    job_id = backend.submit(
        {"campaign": "backend-test"}, run, manifest, dry_run=False,
        intent=submission_intent(backend, run),
    )
    assert job_id == "4321"
    assert ".submission-attempt-001" in " ".join(fake.commands[1])
    assert fake.commands[2][-1].endswith("/manifest.yaml")
    assert "controller-attempt-001.sbatch" in fake.commands[3][-1]
    script = (tmp_path / "attempts" / "attempt-001" / "job.sbatch").read_text()
    assert f"#SBATCH --comment=ml-exp-{SUBMISSION_TOKEN}" in script


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
        backend = WydSlurmBackend(services(tmp_path, fake))
        backend.submit(
            {"campaign": "backend-test"}, run, manifest, dry_run=False,
            intent=submission_intent(backend, run),
        )
    assert len(fake.commands) == 2


def test_slurm_recovery_rejects_multiple_matching_jobs(tmp_path):
    fake = QueueRunner([
        CommandResult(
            ("squeue",), 0,
            f"1732|backend-run--attempt-001|ml-exp-{SUBMISSION_TOKEN}\n",
        ),
        CommandResult(
            ("sacct",), 0,
            f"1731|backend-run--attempt-001|ml-exp-{SUBMISSION_TOKEN}\n"
            f"1732|backend-run--attempt-001|ml-exp-{SUBMISSION_TOKEN}\n",
        ),
    ])
    backend = WydSlurmBackend(services(tmp_path, fake))
    with pytest.raises(RuntimeError, match="2 jobs match"):
        backend.recover_submission(
            slurm_run(), submission_intent(backend, slurm_run()), "attempt-001"
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
    ("manifest_returncode", "expected_available", "expected_exists"),
    [(1, True, False), (0, False, True)],
)
def test_slurm_identity_handles_absent_remote_or_local_manifest(
    tmp_path, manifest_returncode, expected_available, expected_exists,
):
    fake = QueueRunner([
        CommandResult(("squeue",), 0, ""),
        CommandResult(("sacct",), 0, ""),
        CommandResult(("manifest",), manifest_returncode),
    ])
    report = WydSlurmBackend(services(tmp_path, fake)).identity(
        {"campaign": "backend-test"}, slurm_run(), "attempt-002"
    )
    assert report.available is expected_available
    assert report.remote_manifest_exists is expected_exists
    assert report.remote_manifest_matches is None


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

    checkpoint_probe = AssetProbe(
        AssetRequirement("checkpoint", "checkpoint-id", "resume"),
        "/shared/checkpoint",
    )
    present = WydSlurmBackend(services(
        tmp_path,
        QueueRunner([
            CommandResult(("dataset",), 0),
            CommandResult(("checkpoint",), 0),
        ]),
    )).verify_assets(slurm_run(), [probe, checkpoint_probe])
    assert present["missing"] == []


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
    collect_command = fake.commands[0]
    assert "--include=/summary.json" in collect_command
    assert "--include=/backend-run.json" in collect_command
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


def test_sensecore_logs_do_not_treat_progress_step_403_as_expired(tmp_path):
    resource_name = "sensecore-run--attempt-002"
    progress = "Epoch 1: 403/19017 [04:24<3:23:01, 1.53it/s]"
    fake = QueueRunner([
        CommandResult(("stream",), 124, stdout=progress + "\n"),
        CommandResult(("redact",), 0, stdout=progress + "\n"),
    ])
    backend = SenseCoreBackend(services(
        tmp_path, fake,
        record={"attempt_id": "attempt-002", "backend_job_id": resource_name},
    ))

    logs = backend.logs({}, sensecore_run(), tail=5)

    assert logs["expired"] is False
    assert logs["stream_exit_code"] == 124
    assert logs["lines"] == [progress]


def test_sensecore_collection_merges_identity_bound_structured_evidence(tmp_path):
    run = sensecore_run()
    payload = {
        "run_id": run["run_id"],
        "attempt_id": "attempt-001",
        "image_id": run["image_id"],
        "state": "SUCCEEDED",
        "train_loss": 1.25,
        "g_ppl": 42.0,
        "artifacts": {"train_metrics": {"records": 3}},
        "backend": "spoofed",
        "worker_state": "spoofed",
    }
    sentinel = "EXPERIMENT_EVIDENCE_JSON=" + json.dumps(payload, separators=(",", ":"))
    fake = QueueRunner([
        CommandResult(("stream",), 0, f"training output\n{sentinel}\n"),
        CommandResult(("redact",), 0, f"training output\n{sentinel}\n"),
        CommandResult(("workers",), 0, '[{"phase":"Succeeded"}]\n'),
    ])

    summary = SenseCoreBackend(services(tmp_path, fake)).collect({}, run)

    assert summary["state"] == "SUCCEEDED"
    assert summary["train_loss"] == 1.25
    assert summary["g_ppl"] == 42.0
    assert summary["artifacts"] == {"train_metrics": {"records": 3}}
    assert summary["backend"] == "sensecore"
    assert summary["worker_state"] == "RELEASED"
    assert summary["structured_evidence"] == {
        "identity_verified": True,
        "source": "sensecore_stream_logs",
    }
    assert summary["process_evidence"]["stdout_tail"] == ["training output"]


def test_sensecore_collection_accepts_bounded_evidence_above_legacy_limit(tmp_path):
    run = sensecore_run()
    payload = {
        "run_id": run["run_id"],
        "attempt_id": "attempt-001",
        "image_id": run["image_id"],
        "state": "SUCCEEDED",
        "large_summary": "x" * 131073,
    }
    sentinel = "EXPERIMENT_EVIDENCE_JSON=" + json.dumps(payload, separators=(",", ":"))
    fake = QueueRunner([
        CommandResult(("stream",), 0, sentinel + "\n"),
        CommandResult(("redact",), 0, sentinel + "\n"),
        CommandResult(("workers",), 0, '[{"phase":"Succeeded"}]\n'),
    ])

    summary = SenseCoreBackend(services(tmp_path, fake)).collect({}, run)

    assert len(summary["large_summary"]) == 131073
    assert summary["structured_evidence"]["identity_verified"] is True


def test_sensecore_collection_rejects_structured_evidence_identity_conflict(tmp_path):
    run = sensecore_run()
    payload = {
        "run_id": "different-run",
        "attempt_id": "attempt-001",
        "image_id": run["image_id"],
        "state": "SUCCEEDED",
    }
    sentinel = "EXPERIMENT_EVIDENCE_JSON=" + json.dumps(payload, separators=(",", ":"))
    fake = QueueRunner([
        CommandResult(("stream",), 0, sentinel + "\n"),
        CommandResult(("redact",), 0, sentinel + "\n"),
    ])

    with pytest.raises(RuntimeError, match="run_id conflicts"):
        SenseCoreBackend(services(tmp_path, fake)).collect({}, run)


def test_sensecore_submit_checks_exact_created_job(tmp_path):
    run = sensecore_run()
    resource_name = submission_resource_name(
        "sensecore-run", "attempt-001", SUBMISSION_TOKEN
    )
    fake = QueueRunner([
        CommandResult(("safe-list",), 0, "[]\n"),
        CommandResult(("rest-create",), 0, ""),
        CommandResult(("safe-describe",), 0, json.dumps({
            "name": resource_name,
            "display_name": "render test",
            "state": "WAITING",
            "normalized_state": "QUEUED",
        })),
    ])
    backend = SenseCoreBackend(services(tmp_path, fake))
    job_id = backend.submit(
        {}, run,
        {**run, "attempt_id": "attempt-001", "command": ["python", "train.py"]},
        dry_run=False,
        intent=submission_intent(backend, run),
    )
    assert job_id == resource_name
    assert run["image_id"] in fake.commands[1][-1]
    assert resource_name == json.loads(fake.commands[1][-1])["name"]
    assert any(
        f"BACKEND_JOB_ID={resource_name}" in argument
        for argument in fake.commands[1]
    )


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
    assert fake.commands[0][:2] == ("REST", "workers")


def test_packaged_oci_source_is_not_staged_or_masked_by_host_files(tmp_path):
    from experiment_control.backends.wyd import render_job
    from experiment_control.project import SourceBundle
    run = slurm_run()
    run['backend']['oci_image'] = 'registry.example/runtime@sha256:'+'a'*64
    backend = WydSlurmBackend(services(tmp_path, QueueRunner([])))
    converted = []
    backend._stage_oci_image = lambda selected: converted.append(selected['backend']['oci_image'])
    source_id = run['backend']['source_dir'].rsplit('/',1)[-1]
    assert backend.stage({}, run, source_id, SourceBundle(root=tmp_path/'missing-source'))
    assert converted == [run['backend']['oci_image']]
    manifest = {**run, 'attempt_id':'attempt-001', 'execution':{'source_mount':'/workspace','workdir':'/workspace'}, 'command':['python','train.py']}
    packaged = render_job(manifest)
    assert run['backend']['source_dir'] not in packaged
    assert '--pwd /workspace' in packaged
    assert '#SBATCH --output=/dev/null' not in packaged
    assert '/attempts/attempt-001/slurm-%j.err' in packaged
    run['backend']['apptainer_unsquash'] = True
    assert 'apptainer exec --nv --unsquash' in render_job(manifest)
    manifest['execution']['managed_io'] = True
    managed = render_job(manifest)
    assert run['storage']['run_dir']+'/attempts/attempt-001/inputs:/inputs' in managed
    assert run['storage']['run_dir']+'/attempts/attempt-001/outputs:/outputs' in managed
    assert 'mkdir -p '+run['storage']['run_dir']+'/attempts/attempt-001/inputs' in managed
    manifest['execution'].pop('managed_io')
    run['backend'].pop('oci_image')
    assert run['backend']['source_dir']+':/workspace' in render_job(manifest)


def test_slurm_bootstrap_creates_apptainer_cache_and_sandbox_directories(tmp_path):
    import os
    from experiment_control.backends.wyd import render_job
    run = slurm_run()
    root = tmp_path/'data'
    root.mkdir()
    sif = root/'image.sif'
    sif.write_text('fixture')
    run['storage']['run_dir'] = str(root/'run')
    run['storage']['project_data_root'] = str(root)
    run['backend'].update(oci_image='registry.example/runtime@sha256:'+'a'*64,sif_path=str(sif),mount_root=str(root))
    manifest = {**run,'attempt_id':'attempt-002','execution':{'source_mount':'/workspace','workdir':'/workspace'},'command':['true']}
    binpath = tmp_path/'bin'
    binpath.mkdir()
    srun = binpath/'srun'
    srun.write_text('#!/bin/sh\ntest -d "$APPTAINER_CACHEDIR" && test -d "$APPTAINER_TMPDIR"\n')
    srun.chmod(0o755)
    script = tmp_path/'job.sh'
    script.write_text(render_job(manifest))
    subprocess.run(['bash',str(script)],check=True,env={**os.environ,'SLURM_JOB_ID':'1234','PATH':str(binpath)+':/usr/bin:/bin'})
    assert (root/'apptainer/tmp').is_dir() and (root/'apptainer/cache').is_dir()


def test_offline_metrics_are_retained_without_fresh_liveness(tmp_path):
    from unittest.mock import Mock
    backend = SenseCoreBackend(services(tmp_path, QueueRunner([
        CommandResult(('redact',), 0, 'Step 7 loss 1.25\n')
    ])))
    backend._rest = Mock()
    backend._rest.logs.return_value = {
        'text': 'Step 7 loss 1.25\n', 'expired': False, 'exit_code': 0, 'available': True,
        'source': 'offline', 'historical': True, 'last_log_at': '2026-01-02T00:00:00Z', 'truncated': True}
    backend._rest.workers.return_value = [{'phase': 'Deleted'}]
    result = backend.collect({}, sensecore_run())
    assert result['log_source'] == 'sensecore_offline_logs'
    assert result['last_log_at'] == '2026-01-02T00:00:00Z'
    assert not result['process_evidence']['observed'] and not result['live_logs_available']
    assert result['process_evidence']['stdout_tail'] == ['Step 7 loss 1.25']
    assert result['worker_state'] == 'RELEASED'


def test_provider_suspended_is_stopped_without_proven_preemption(tmp_path):
    from unittest.mock import Mock
    backend = SenseCoreBackend(services(tmp_path, QueueRunner([])))
    backend._rest = Mock()
    backend._rest.describe.return_value = {'name': '1234', 'state': 'SUSPENDED'}
    status = backend.status({}, sensecore_run())
    assert status['state'] == 'CANCELLED' and status['failure_class'] is None
    assert status['reason'] == 'provider_stopped_cause_unknown'
    assert status['raw_state'] == 'SUSPENDED'
