"""Fail-closed backend branches using only injected command runners."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from backend_harness import QueueRunner, sensecore_run, services, slurm_run
from experiment_control.backends import build_registry
from experiment_control.backends.sensecore import (
    SenseCoreBackend,
    digest_pinned_image,
    normalize_state as normalize_sensecore_state,
)
from experiment_control.backends.wyd import (
    WydSlurmBackend,
    log_probe_command,
    normalize_state as normalize_slurm_state,
    parse_accounting,
    scheduler_job_name as slurm_job_name,
)
from experiment_control.project import SourceBundle
from experiment_control.runner import CommandResult


@pytest.mark.parametrize(
    ("image", "digest"),
    [
        ("registry/image:tag", "not-a-digest"),
        ("registry/image:tag@sha256:old", "sha256:" + "a" * 64),
        ("registry/image", "sha256:" + "a" * 64),
        ("registry/image:latest", "sha256:" + "a" * 64),
    ],
)
def test_sensecore_digest_pin_rejects_mutable_or_malformed_identity(image, digest):
    with pytest.raises(ValueError):
        digest_pinned_image(image, digest)


def test_scheduler_normalizers_cover_every_semantic_state():
    sensecore_cases = {
        "WAITING": "QUEUED", "STARTING": "STARTING", "RUNNING": "RUNNING",
        "COMPLETED": "SUCCEEDED", "SUSPENDED": "PREEMPTED", "ERROR": "FAILED",
        "DELETED": "CANCELLED", "future": "UNKNOWN",
    }
    for raw, expected in sensecore_cases.items():
        assert normalize_sensecore_state(raw) == expected
    assert normalize_sensecore_state("SUSPENDED", cancellation_requested=True) == "CANCELLED"
    assert normalize_slurm_state("COMPLETED", "1:0") == "FAILED"
    assert normalize_slurm_state("RUNNING+") == "RUNNING"
    assert normalize_slurm_state("future") == "UNKNOWN"


def test_backend_registry_builds_all_injected_adapters(tmp_path):
    registry = build_registry(services(tmp_path, QueueRunner([])))
    assert registry.get("local").kind == "local"
    assert registry.get("sensecore").kind == "sensecore"
    assert registry.get("slurm").kind == "slurm"


def test_sensecore_validate_and_environment_boundaries(tmp_path, monkeypatch):
    backend = SenseCoreBackend(services(tmp_path, QueueRunner([])))
    valid = sensecore_run()
    valid["storage"] = {
        "run_dir": "/shared/run", "data_root": "/shared", "description": "test",
    }
    valid["backend"]["storage_mount"] = "volume:/shared"
    backend.validate(valid)
    assert backend.environment({}, valid, "source", "attempt-002")["BACKEND_JOB_ID"].endswith(
        "attempt-002"
    )

    cases = []
    missing = sensecore_run(); missing["backend"].pop("workspace"); cases.append(missing)
    quota = sensecore_run(); quota["backend"]["quota_type"] = "normal"; cases.append(quota)
    image = sensecore_run(); image["image_id"] = "tag"; cases.append(image)
    mount = sensecore_run(); mount["backend"]["storage_mount"] = "relative"; cases.append(mount)
    outside = sensecore_run(); outside["storage"] = {"run_dir": "/outside"}; cases.append(outside)
    for run in cases:
        with pytest.raises(ValueError):
            backend.validate(run)

    monkeypatch.setenv("EXPERIMENTCTL_SCO_CREATE_TIMEOUT_SECONDS", "9")
    with pytest.raises(ValueError, match="10 to 600"):
        backend.create_timeout_seconds()
    with pytest.raises(ValueError, match="unsupported preflight"):
        backend.preflight(valid, scope="delete")


def test_sensecore_observation_shape_and_recovery_errors(tmp_path):
    backend = SenseCoreBackend(services(tmp_path, QueueRunner([])))
    assert backend.verify_assets({}, [object()])["missing"] is None
    assert backend.stage({}, {}, "source", object()) is True
    assert "job-list" in backend.safe_command(["sco", "list"], "job-list")[-1]

    with pytest.raises(RuntimeError, match="conflicting scheduler name"):
        backend.recover_submission(
            sensecore_run(), {"scheduler_name": "wrong"}, "attempt-001",
        )

    ambiguous = SenseCoreBackend(services(tmp_path, QueueRunner([
        CommandResult(("find",), 0, json.dumps([
            {"name": "sensecore-run--attempt-001"},
            {"name": "sensecore-run--attempt-001"},
        ])),
    ])))
    with pytest.raises(RuntimeError, match="ambiguous"):
        ambiguous.recover_submission(sensecore_run(), {}, "attempt-001")


@pytest.mark.parametrize("method", ["describe", "find", "workers"])
def test_sensecore_sanitized_queries_reject_wrong_json_shapes(tmp_path, method):
    payload = "[]" if method == "describe" else "{}"
    backend = SenseCoreBackend(services(
        tmp_path,
        QueueRunner([CommandResult((method,), 0, payload)]),
        record={"attempt_id": "attempt-001", "backend_job_id": "sensecore-run--attempt-001"},
    ))
    with pytest.raises(ValueError):
        if method == "workers":
            backend.workers({}, sensecore_run())
        else:
            getattr(backend, method)(sensecore_run(), "sensecore-run--attempt-001")


def test_sensecore_submit_fails_closed_for_drift_duplicates_and_create_errors(tmp_path):
    run = sensecore_run()
    manifest = {**run, "attempt_id": "attempt-001", "command": ["python", "train.py"]}
    backend = SenseCoreBackend(services(tmp_path, QueueRunner([])))
    with pytest.raises(ValueError, match="image_id conflicts"):
        backend.submit({}, run, {**manifest, "image_id": "sha256:" + "c" * 64}, dry_run=False)

    duplicate = SenseCoreBackend(services(tmp_path, QueueRunner([
        CommandResult(("find",), 0, '[{"name":"sensecore-run--attempt-001"}]'),
    ])))
    with pytest.raises(FileExistsError):
        duplicate.submit({}, run, manifest, dry_run=False)

    failed = SenseCoreBackend(services(tmp_path, QueueRunner([
        CommandResult(("find",), 0, "[]"),
        CommandResult(("create",), 1, stderr="failed"),
        CommandResult(("redact",), 0, "sanitized"),
    ])))
    with pytest.raises(RuntimeError, match="sanitized"):
        failed.submit({}, run, manifest, dry_run=False)

    conflict = SenseCoreBackend(services(tmp_path, QueueRunner([
        CommandResult(("find",), 0, "[]"),
        CommandResult(("create",), 0),
        CommandResult(("describe",), 0, '{"name":"other"}'),
    ])))
    with pytest.raises(RuntimeError, match="not observable"):
        conflict.submit({}, run, manifest, dry_run=False)


def test_sensecore_status_cancel_markers_and_active_cancel(tmp_path, monkeypatch):
    run = sensecore_run()
    resource = "sensecore-run--attempt-001"
    backend = SenseCoreBackend(services(
        tmp_path, QueueRunner([CommandResult(("describe",), 0, json.dumps({
            "name": resource, "state": "RUNNING", "pool": "p", "spec": "s",
        }))]), record={"attempt_id": "attempt-001", "backend_job_id": resource},
    ))
    assert backend.status({}, run)["state"] == "RUNNING"

    marker = tmp_path / "cancel_requested.json"
    marker.write_text("not-json")
    with pytest.raises(RuntimeError, match="unreadable"):
        backend._cancellation_requested({}, run, resource)
    marker.write_text("{}")
    with pytest.raises(RuntimeError, match="malformed"):
        backend._cancellation_requested({}, run, resource)
    marker.write_text(json.dumps({"backend_job_id": "other"}))
    assert backend._cancellation_requested({}, run, resource) is False
    marker.write_text(json.dumps({"backend_job_id": resource}))
    assert backend._cancellation_requested({}, run, resource) is True
    marker.unlink()

    calls = iter([
        {"state": "RUNNING", "backend_job_id": resource},
        {"state": "CANCELLED", "backend_job_id": resource},
    ])
    monkeypatch.setattr(backend, "status", lambda *args: next(calls))
    backend.s = replace(backend.s, run_command=QueueRunner([CommandResult(("stop",), 0)]).run)
    assert backend.cancel({}, run)["state"] == "CANCELLED"
    assert marker.is_file()


def test_sensecore_collects_metrics_checkpoints_and_worker_fallback(tmp_path, monkeypatch):
    service = services(tmp_path, QueueRunner([]))
    service = replace(
        service,
        parse_metric=lambda campaign, line: {"loss": 1.0} if "Step " in line else None,
        parse_checkpoint=lambda campaign, line: (
            {"step": 2, "path": "/checkpoint_2"} if "checkpoint" in line else None
        ),
    )
    backend = SenseCoreBackend(service)
    monkeypatch.setattr(backend, "logs", lambda *args, **kwargs: {
        "lines": ["Step 2 loss", "checkpoint", "plan ppl"], "expired": True,
    })
    monkeypatch.setattr(backend, "workers", lambda *args: (_ for _ in ()).throw(RuntimeError()))
    result = backend.collect({}, {"run_id": "run"})
    assert result["latest_metric"] == {"loss": 1.0}
    assert result["latest_completed_checkpoint_step"] == 2
    assert result["worker_state"] == "UNKNOWN"
    assert result["evidence_unavailable_reason"] == "live_logs_expired"

    monkeypatch.setattr(backend, "logs", lambda *args, **kwargs: {
        "lines": ["quiet worker"], "expired": False,
    })
    monkeypatch.setattr(backend, "workers", lambda *args: {
        "worker_state": "RUNNING", "worker_evidence_available": True,
    })
    quiet = backend.collect({}, {"run_id": "run"})
    assert quiet["latest_metric"] is None
    assert quiet["worker_state"] == "RUNNING"
    assert "evidence_unavailable_reason" not in quiet
    assert "latest_completed_checkpoint" not in quiet


def test_sensecore_worker_unknown_and_log_tail_bounds(tmp_path):
    backend = SenseCoreBackend(services(
        tmp_path, QueueRunner([CommandResult(("workers",), 0, '[{"phase":"mystery"}]')]),
        record={"attempt_id": "attempt-001", "backend_job_id": "sensecore-run--attempt-001"},
    ))
    assert backend.workers({}, sensecore_run())["worker_state"] == "UNKNOWN"
    with pytest.raises(ValueError, match="tail"):
        backend.logs({}, sensecore_run(), tail=0)


def test_slurm_helpers_and_validation_edges(tmp_path):
    assert len(slurm_job_name("r" * 130, "attempt-001")) <= 128
    assert parse_accounting("", job_id="1", run_id="r", partition="p")["state"] == "UNKNOWN"
    with pytest.raises(ValueError, match="tail"):
        log_probe_command(["log"], tail=0)
    with pytest.raises(ValueError, match="at least one"):
        log_probe_command([], tail=1)

    backend = WydSlurmBackend(services(tmp_path, QueueRunner([])))
    valid = slurm_run()
    backend.validate(valid)
    assert backend.environment({}, valid, "source", "attempt")["QUOTA_TYPE"] == "normal"
    assert backend.submission_request({}, valid, "attempt")["scheduler_name"]
    with pytest.raises(ValueError, match="unsupported preflight"):
        backend.preflight(valid, scope="delete")

    mutations = []
    missing = slurm_run(); missing["backend"].pop("qos"); mutations.append(missing)
    gres = slurm_run(); gres["backend"]["gres"] = "gpu"; mutations.append(gres)
    bool_gpu = slurm_run(); bool_gpu["resources"]["gpus"] = True; mutations.append(bool_gpu)
    digest = slurm_run(); digest["image_id"] = "tag"; mutations.append(digest)
    unsafe = slurm_run(); unsafe["backend"]["account"] = "bad/account"; mutations.append(unsafe)
    mount = slurm_run(); mount["backend"]["mount_root"] = "relative"; mutations.append(mount)
    cache = slurm_run(); cache["backend"]["apptainer_cache_dir"] = "relative"; mutations.append(cache)
    outside = slurm_run(); outside["storage"]["run_dir"] = "/outside"; mutations.append(outside)
    for run in mutations:
        with pytest.raises(ValueError):
            backend.validate(run)


def test_slurm_stage_transfers_source_verifies_image_and_required_paths(tmp_path):
    run = slurm_run()
    digest = run["image_id"].removeprefix("sha256:")
    source = SourceBundle(
        root=tmp_path / "source", excludes=(".git",), required_paths=("train.py",),
    )
    fake = QueueRunner([
        CommandResult(("mkdir",), 0),
        CommandResult(("source-marker",), 1),
        CommandResult(("rsync",), 0),
        CommandResult(("touch-source",), 0),
        CommandResult(("sif-marker",), 1),
        CommandResult(("sha256",), 0, f"{digest}  image.sif\n"),
        CommandResult(("touch-sif",), 0),
        CommandResult(("required-path",), 0),
    ])
    backend = WydSlurmBackend(services(tmp_path, fake))

    assert backend.stage({}, run, "source-fixed", source) is True
    assert any("--exclude" in command for command in fake.commands)
    assert any("train.py" in " ".join(command) for command in fake.commands)


def test_slurm_stage_reuses_verified_source_and_image(tmp_path):
    run = slurm_run()
    fake = QueueRunner([
        CommandResult(("mkdir",), 0),
        CommandResult(("source-marker",), 0),
        CommandResult(("sif-marker",), 0),
    ])
    backend = WydSlurmBackend(services(tmp_path, fake))

    assert backend.stage(
        {}, run, "source-fixed", SourceBundle(root=Path("/local/source")),
    ) is True
    assert len(fake.commands) == 3


@pytest.mark.parametrize(
    ("scope", "error", "category"),
    [
        ("observe", subprocess.CalledProcessError(1, ["ssh"]), "transport"),
        ("submit", RuntimeError("partition unavailable"), "resource"),
        ("submit", RuntimeError("account unavailable"), "authorization"),
    ],
)
def test_slurm_preflight_classifies_remote_failures(tmp_path, monkeypatch, scope, error, category):
    fake = QueueRunner([CommandResult(("ssh-version",), 0)])
    backend = WydSlurmBackend(services(tmp_path, fake))
    target = "validate_live" if scope == "submit" else "validate_control"
    monkeypatch.setattr(backend, target, lambda run: (_ for _ in ()).throw(error))
    report = backend.preflight(slurm_run(), scope=scope)
    assert report.ready is False
    assert report.checks[-1].category == category


def test_sensecore_validation_wraps_scheduler_and_image_identity_errors(tmp_path):
    backend = SenseCoreBackend(services(tmp_path, QueueRunner([])))
    invalid_name = sensecore_run()
    invalid_name["backend"]["job_name"] = "Invalid/Name"
    with pytest.raises(ValueError, match="SenseCore base job name"):
        backend.validate(invalid_name)

    invalid_image = sensecore_run()
    invalid_image["backend"]["image"] = "registry.example/project/image:latest"
    with pytest.raises(ValueError, match="immutable source-qualified tag"):
        backend.validate(invalid_image)


@pytest.mark.parametrize(("payload", "expected"), [("[]", None), (
    '[{"name":"sensecore-run--attempt-001"}]',
    "sensecore-run--attempt-001",
)])
def test_sensecore_recovery_distinguishes_absent_and_exact_job(tmp_path, payload, expected):
    backend = SenseCoreBackend(services(
        tmp_path, QueueRunner([CommandResult(("find",), 0, payload)]),
    ))
    assert backend.recover_submission(sensecore_run(), {}, "attempt-001") == expected


def test_sensecore_status_rejects_conflicting_exact_resource(tmp_path):
    resource = "sensecore-run--attempt-001"
    backend = SenseCoreBackend(services(
        tmp_path,
        QueueRunner([CommandResult(("describe",), 0, '{"name":"other"}')]),
        record={"attempt_id": "attempt-001", "backend_job_id": resource},
    ))
    with pytest.raises(RuntimeError, match="conflicting resource"):
        backend.status({}, sensecore_run())


def test_sensecore_cancellation_reads_legacy_root_marker_for_attempt_dir(tmp_path):
    attempt_dir = tmp_path / "attempts" / "attempt-001"
    attempt_dir.mkdir(parents=True)
    resource = "sensecore-run--attempt-001"
    (tmp_path / "cancel_requested.json").write_text(
        json.dumps({"backend_job_id": resource}), encoding="utf-8",
    )
    backend = SenseCoreBackend(services(attempt_dir, QueueRunner([])))
    assert backend._cancellation_requested({}, sensecore_run(), resource) is True


def test_sensecore_cancel_and_worker_failures_remain_sanitized(tmp_path, monkeypatch):
    resource = "sensecore-run--attempt-001"
    runner = QueueRunner([
        CommandResult(("stop",), 1, stderr="raw secret"),
        CommandResult(("redact",), 0, stdout="sanitized stop failure"),
    ])
    backend = SenseCoreBackend(services(tmp_path, runner))
    monkeypatch.setattr(backend, "status", lambda *args: {
        "state": "RUNNING", "backend_job_id": resource,
    })
    with pytest.raises(RuntimeError, match="sanitized stop failure"):
        backend.cancel({}, sensecore_run())

    worker = SenseCoreBackend(services(
        tmp_path,
        QueueRunner([
            CommandResult(("workers",), 1, stderr="raw secret"),
            CommandResult(("redact",), 0, stdout="sanitized worker failure"),
        ]),
        record={"attempt_id": "attempt-001", "backend_job_id": resource},
    ))
    with pytest.raises(RuntimeError, match="sanitized worker failure"):
        worker.workers({}, sensecore_run())


def test_slurm_recovery_handles_blank_evidence_and_one_exact_match(tmp_path):
    expected_name = "backend-run--attempt-001"
    backend = WydSlurmBackend(services(tmp_path, QueueRunner([
        CommandResult(
            ("squeue",), 0,
            f"\n999|unrelated-job|other-token\n123|{expected_name}|token\n\n",
        ),
        CommandResult(("sacct",), 0, "\n998|unrelated-job\n\n"),
    ])))
    assert backend.recover_submission(
        slurm_run(), {"submission_token": "token"}, "attempt-001",
    ) == "123"


@pytest.mark.parametrize(
    ("queue", "accounting", "error"),
    [
        ("malformed\n", None, "queue identity query returned malformed"),
        ("job|backend-run--attempt-001|token\n", None, "invalid job ID"),
        ("", "malformed\n", "accounting identity query returned malformed"),
        ("", "job|backend-run--attempt-001\n", "invalid job ID"),
    ],
)
def test_slurm_recovery_rejects_malformed_scheduler_evidence(
    tmp_path, queue, accounting, error,
):
    results = [CommandResult(("squeue",), 0, queue)]
    if accounting is not None:
        results.append(CommandResult(("sacct",), 0, accounting))
    backend = WydSlurmBackend(services(tmp_path, QueueRunner(results)))
    with pytest.raises(RuntimeError, match=error):
        backend.recover_submission(
            slurm_run(), {"submission_token": "token"}, "attempt-001",
        )


def test_slurm_identity_rejects_malformed_remote_manifest_digest(tmp_path):
    (tmp_path / "manifest.yaml").write_text("run_id: backend-run\n", encoding="utf-8")
    backend = WydSlurmBackend(services(tmp_path, QueueRunner([
        CommandResult(("squeue",), 0, ""),
        CommandResult(("sacct",), 0, ""),
        CommandResult(("manifest",), 0, ""),
        CommandResult(("digest",), 0, "not-a-digest\n"),
    ])))
    with pytest.raises(RuntimeError, match="digest probe returned malformed"):
        backend.identity({"campaign": "test"}, slurm_run(), "attempt-001")


def test_slurm_stage_rejects_source_image_and_required_path_drift(tmp_path):
    run = slurm_run()
    backend = WydSlurmBackend(services(tmp_path, QueueRunner([])))
    with pytest.raises(ValueError, match="source_dir must end"):
        backend.stage({}, run, "different-source", SourceBundle(tmp_path))

    mismatch = WydSlurmBackend(services(tmp_path, QueueRunner([
        CommandResult(("mkdir",), 0),
        CommandResult(("source-marker",), 0),
        CommandResult(("sif-marker",), 1),
        CommandResult(("sha256",), 0, f"{'0' * 64}  image.sif\n"),
    ])))
    with pytest.raises(ValueError, match="SIF checksum mismatch"):
        mismatch.stage({}, run, "source-fixed", SourceBundle(tmp_path))

    unsafe = WydSlurmBackend(services(tmp_path, QueueRunner([
        CommandResult(("mkdir",), 0),
        CommandResult(("source-marker",), 0),
        CommandResult(("sif-marker",), 0),
    ])))
    with pytest.raises(ValueError, match="required source path must be relative"):
        unsafe.stage(
            {}, run, "source-fixed",
            SourceBundle(tmp_path, required_paths=("../train.py",)),
        )

    missing = WydSlurmBackend(services(tmp_path, QueueRunner([
        CommandResult(("mkdir",), 0),
        CommandResult(("source-marker",), 0),
        CommandResult(("sif-marker",), 0),
        CommandResult(("required-path",), 1),
    ])))
    with pytest.raises(RuntimeError, match="missing required project path"):
        missing.stage(
            {}, run, "source-fixed",
            SourceBundle(tmp_path, required_paths=("train.py",)),
        )


@pytest.mark.parametrize(
    ("output", "error"),
    [
        ("other|up|3-00:00:00|gpu:accelerator:8\n", "not currently visible"),
        ("accelerator|down|3-00:00:00|gpu:accelerator:8\n", "not currently usable"),
        (
            "accelerator|up|3-00:00:00|gpu:accelerator:8\n"
            "user|other||other|other\n",
            "association does not expose",
        ),
    ],
)
def test_slurm_live_validation_reports_specific_resource_drift(tmp_path, output, error):
    backend = WydSlurmBackend(services(
        tmp_path, QueueRunner([CommandResult(("live",), 0, output)]),
    ))
    with pytest.raises(RuntimeError, match=error):
        backend.validate_live(slurm_run())


def slurm_attempt_manifest(run):
    return {
        **run,
        "campaign": "backend-test",
        "attempt_id": "attempt-001",
        "command": ["python", "train.py"],
        "execution": {"source_mount": "/app", "workdir": "/app"},
    }


def test_slurm_submit_dry_run_claim_failure_and_bad_scheduler_response(tmp_path):
    run = slurm_run()
    manifest = slurm_attempt_manifest(run)
    dry_run = WydSlurmBackend(services(tmp_path, QueueRunner([])))
    assert dry_run.submit({}, run, manifest, dry_run=True) == "DRY_RUN"

    live = (
        "accelerator|up|3-00:00:00|gpu:accelerator:8\n"
        "user|lab||normal|normal\n"
    )
    claim_failure = WydSlurmBackend(services(tmp_path / "claim", QueueRunner([
        CommandResult(("live",), 0, live),
        CommandResult(("claim",), 2, stderr="transport failure"),
    ])))
    with pytest.raises(RuntimeError, match="scheduler/storage state is unknown"):
        claim_failure.submit({}, run, manifest, dry_run=False)

    bad_response = WydSlurmBackend(services(tmp_path / "response", QueueRunner([
        CommandResult(("live",), 0, live),
        CommandResult(("claim",), 0),
        CommandResult(("manifest-rsync",), 0),
        CommandResult(("script-rsync",), 0),
        CommandResult(("sbatch",), 0, "not-a-job-id\n"),
    ])))
    with pytest.raises(ValueError, match="unexpected sbatch response"):
        bad_response.submit({}, run, manifest, dry_run=False)


def test_slurm_status_falls_back_to_queue_when_accounting_is_delayed(tmp_path):
    backend = WydSlurmBackend(services(tmp_path, QueueRunner([
        CommandResult(("sacct",), 0, ""),
        CommandResult(
            ("squeue",), 0,
            "1234|backend-run|accelerator|RUNNING|00:01:00|0:0\n",
        ),
    ])))
    assert backend.status({}, slurm_run())["state"] == "RUNNING"


def test_slurm_cancel_short_circuits_terminal_and_rechecks_active_job(tmp_path, monkeypatch):
    backend = WydSlurmBackend(services(tmp_path, QueueRunner([])))
    terminal = {"state": "SUCCEEDED", "backend_job_id": "1234"}
    monkeypatch.setattr(backend, "status", lambda *args: terminal)
    assert backend.cancel({}, slurm_run()) == terminal

    statuses = iter([
        {"state": "RUNNING", "backend_job_id": "1234"},
        {"state": "CANCELLED", "backend_job_id": "1234"},
    ])
    commands = []
    monkeypatch.setattr(backend, "status", lambda *args: next(statuses))
    monkeypatch.setattr(
        backend, "remote_exec",
        lambda alias, command, **kwargs: commands.append((alias, command)),
    )
    assert backend.cancel({}, slurm_run())["state"] == "CANCELLED"
    assert commands == [("test-login", "scancel 1234")]


def test_slurm_logs_reject_bad_job_identity_and_remote_probe_failure(tmp_path):
    bad_identity = WydSlurmBackend(services(
        tmp_path, QueueRunner([]),
        record={"attempt_id": "attempt-001", "backend_job_id": "not-numeric"},
    ))
    with pytest.raises(ValueError, match="job ID must be numeric"):
        bad_identity.logs({}, slurm_run(), tail=10)

    failed_probe = WydSlurmBackend(services(
        tmp_path,
        QueueRunner([CommandResult(("stdout",), 2, stderr="storage unavailable")]),
    ))
    with pytest.raises(RuntimeError, match="log probe failed"):
        failed_probe.logs({}, slurm_run(), tail=10)
