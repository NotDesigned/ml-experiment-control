from __future__ import annotations

import json
import runpy
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiment_control import _local_worker
from experiment_control.backends import local as local_module
from experiment_control.backends.local import LocalBackend
from experiment_control.backends.services import BackendServices
from experiment_control.manifest import atomic_write
from experiment_control.project import AssetProbe, AssetRequirement, SourceBundle


SUBMISSION_TOKEN = "b" * 32


def local_run(tmp_path: Path) -> dict:
    workdir = tmp_path / "work"
    workdir.mkdir()
    (tmp_path / "runs").mkdir()
    return {
        "run_id": "local-run",
        "storage": {"run_dir": str(tmp_path / "runs" / "local-run")},
        "backend": {
            "kind": "local",
            "workdir": str(workdir),
            "cancel_grace_seconds": 0.1,
        },
    }


def local_backend(tmp_path: Path, run: dict):
    record = {"attempt_id": "attempt-001", "backend_job_id": None}
    services = BackendServices(
        run_command=lambda *args, **kwargs: None,
        local_run_dir=lambda _campaign, _run: Path(run["storage"]["run_dir"]),
        backend_record=lambda _campaign, _run: dict(record),
        summarize_run=lambda _campaign, path: {"summary_root": str(path)},
        parse_metric=lambda _campaign, line: (
            {"step": int(line.split("=")[1])} if line.startswith("step=") else None
        ),
        parse_checkpoint=lambda _campaign, line: (
            {"step": int(line.rsplit("=", 1)[1]), "path": line.split()[0].split("=")[1]}
            if line.startswith("checkpoint=") else None
        ),
        atomic_write=atomic_write,
        utc_now=lambda: "2026-07-13T00:00:00Z",
    )
    return LocalBackend(services), record


def local_intent(backend: LocalBackend, run: dict, attempt_id: str = "attempt-001") -> dict:
    return {
        "submission_token": SUBMISSION_TOKEN,
        "request": backend.submission_request({}, run, attempt_id),
    }


def wait_for_terminal(backend: LocalBackend, run: dict, timeout: float = 5) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = backend.status({}, run)
        if status["state"] in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            return status
        time.sleep(0.02)
    raise AssertionError("local process did not become terminal")


def test_local_backend_runs_collects_and_recovers_a_real_process(tmp_path):
    run = local_run(tmp_path)
    backend, record = local_backend(tmp_path, run)
    manifest = {
        "attempt_id": "attempt-001",
        "source_id": "source-local",
        "command": [
            sys.executable,
            "-c",
            "print('step=3'); print('checkpoint=/tmp/checkpoint step=4'); "
            "import sys; print('token=secret', file=sys.stderr)",
        ],
    }

    assert backend.identity({}, run, "attempt-001").available is True
    intent = local_intent(backend, run)
    assert backend.recover_submission(run, intent, "attempt-001") is None
    job_id = backend.submit({}, run, manifest, dry_run=False, intent=intent)
    record["backend_job_id"] = job_id
    assert backend.identity({}, run, "attempt-001").scheduler_job_ids == (job_id,)
    assert backend.recover_submission(
        run, intent, "attempt-001"
    ) == job_id
    with pytest.raises(RuntimeError, match="conflicting submission token"):
        backend.recover_submission(
            run, {**intent, "submission_token": "c" * 32}, "attempt-001"
        )
    status = wait_for_terminal(backend, run)
    assert status["state"] == "SUCCEEDED"
    assert status["exit_code"] == 0

    logs = backend.logs({}, run, tail=10)
    assert logs["stdout"] == ["step=3", "checkpoint=/tmp/checkpoint step=4"]
    assert logs["stderr"] == ["token=<redacted>"]
    summary = backend.collect({}, run)
    assert summary["latest_metric"] == {"step": 3}
    assert summary["latest_completed_checkpoint_step"] == 4
    assert summary["process_evidence"]["observed"] is True
    assert backend.cancel({}, run) == status


def test_local_backend_cancels_an_exact_process_group(tmp_path):
    run = local_run(tmp_path)
    backend, record = local_backend(tmp_path, run)
    manifest = {
        "attempt_id": "attempt-001",
        "command": [sys.executable, "-c", "import time; time.sleep(30)"],
    }
    job_id = backend.submit(
        {}, run, manifest, dry_run=False, intent=local_intent(backend, run)
    )
    record["backend_job_id"] = job_id
    assert backend.status({}, run)["state"] == "RUNNING"
    assert backend.cancel({}, run)["state"] == "CANCELLED"


def test_local_backend_validation_preflight_assets_stage_and_dry_run(tmp_path):
    run = local_run(tmp_path)
    backend, _ = local_backend(tmp_path, run)
    backend.validate(run)
    report = backend.preflight(run, scope="submit")
    assert report.ready is True
    assert backend.environment({}, run, "source", "attempt-002") == {
        "LOCAL_ATTEMPT_ID": "attempt-002"
    }
    assert backend.render({"command": ["python", "train.py", "--seed", "1"]}) == (
        "python train.py --seed 1"
    )
    assert backend.submit(
        {}, run, {"attempt_id": "attempt-001", "command": ["true"]}, dry_run=True
    ) == "DRY_RUN"
    bad_intent = local_intent(backend, run)
    bad_intent["request"]["scheduler_name"] = "wrong"
    with pytest.raises(RuntimeError, match="conflicting scheduler_name"):
        backend.submit(
            {}, run, {"attempt_id": "attempt-001", "command": ["true"]},
            dry_run=False, intent=bad_intent,
        )

    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_text("print('ok')", encoding="utf-8")
    assert backend.stage({}, run, "source", SourceBundle(
        source, required_paths=("train.py",),
    )) is True
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    empty_file = tmp_path / "empty"
    empty_file.touch()
    probes = [
        AssetProbe(AssetRequirement("dataset", "data", "train"), str(dataset)),
        AssetProbe(AssetRequirement("checkpoint", "ckpt", "resume"), str(empty_file), True),
    ]
    result = backend.verify_assets(run, probes)
    assert [item["identity"] for item in result["missing"]] == ["ckpt"]

    with pytest.raises(ValueError, match="unsupported preflight"):
        backend.preflight(run, scope="delete")
    with pytest.raises(ValueError, match="command must not be empty"):
        backend.submit({}, run, {"attempt_id": "attempt-001", "command": []}, dry_run=False)
    with pytest.raises(ValueError, match="tail"):
        backend.logs({}, run, tail=0)
    assert backend.logs({}, run, tail=1)["stdout"] == []

    with pytest.raises(ValueError, match="root"):
        backend.stage({}, run, "source", SourceBundle(tmp_path / "missing"))
    with pytest.raises(ValueError, match="must be relative"):
        backend.stage({}, run, "source", SourceBundle(source, required_paths=("../x",)))
    with pytest.raises(RuntimeError, match="missing required"):
        backend.stage({}, run, "source", SourceBundle(source, required_paths=("missing.py",)))

    shutil.rmtree(run["backend"]["workdir"])
    shutil.rmtree(Path(run["storage"]["run_dir"]).parent)
    original_identity = local_module._process_identity
    local_module._process_identity = lambda _pid: None
    try:
        report = backend.preflight(run, scope="observe")
        assert [check.status for check in report.checks] == [
            "FAIL", "FAIL", "FAIL",
        ]
    finally:
        local_module._process_identity = original_identity


@pytest.mark.parametrize("mutation", ["backend", "workdir", "run_dir"])
def test_local_backend_rejects_invalid_configuration(tmp_path, mutation):
    run = local_run(tmp_path)
    backend, _ = local_backend(tmp_path, run)
    if mutation == "backend":
        run["backend"] = None
    elif mutation == "workdir":
        run["backend"]["workdir"] = "relative"
    else:
        run["storage"]["run_dir"] = "relative"
    with pytest.raises(ValueError):
        backend.validate(run)


def test_local_backend_fails_closed_for_claim_and_record_drift(tmp_path):
    run = local_run(tmp_path)
    backend, record = local_backend(tmp_path, run)
    attempt_dir = Path(run["storage"]["run_dir"]) / "attempts" / "attempt-001"
    attempt_dir.mkdir(parents=True)
    control = attempt_dir / "local-process.json"

    control.write_text("{", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unreadable"):
        backend.identity({}, run, "attempt-001")
    control.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="malformed"):
        backend.identity({}, run, "attempt-001")
    control.write_text(json.dumps({"attempt_id": "other"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="conflicting attempt"):
        backend.identity({}, run, "attempt-001")
    control.write_text(json.dumps({
        "attempt_id": "attempt-001", "state": "LAUNCHING",
        "submission_token": SUBMISSION_TOKEN,
    }), encoding="utf-8")
    assert backend.identity({}, run, "attempt-001").available is False
    with pytest.raises(RuntimeError, match="without a process identity"):
        backend.recover_submission(run, local_intent(backend, run), "attempt-001")
    with pytest.raises(RuntimeError, match="conflicting workdir"):
        backend.recover_submission(run, {
            "submission_token": SUBMISSION_TOKEN,
            "request": {
                **backend.submission_request({}, run, "attempt-001"),
                "workdir": "/other",
            },
        }, "attempt-001")

    control.write_text(json.dumps({
        "attempt_id": "attempt-001", "state": "RUNNING", "backend_job_id": "1:1",
        "pid": 1, "start_ticks": "1",
    }), encoding="utf-8")
    record["backend_job_id"] = "other"
    with pytest.raises(RuntimeError, match="conflicting process identity"):
        backend.status({}, run)

    record["backend_job_id"] = "1:1"
    control.unlink()
    with pytest.raises(RuntimeError, match="not recorded"):
        backend.status({}, run)


def test_local_backend_rejects_result_drift_and_reports_lost_process(tmp_path):
    run = local_run(tmp_path)
    backend, record = local_backend(tmp_path, run)
    attempt_dir = Path(run["storage"]["run_dir"]) / "attempts" / "attempt-001"
    attempt_dir.mkdir(parents=True)
    control = {
        "attempt_id": "attempt-001", "state": "RUNNING", "backend_job_id": "999999:1",
        "pid": 999999, "start_ticks": "1",
    }
    (attempt_dir / "local-process.json").write_text(json.dumps(control))
    record["backend_job_id"] = "999999:1"
    assert backend.status({}, run)["raw_state"] == "LOST"

    (attempt_dir / "local-result.json").write_text(json.dumps({
        "worker_pid": 1, "exit_code": "bad",
    }))
    with pytest.raises(RuntimeError, match="result record"):
        backend.status({}, run)


def test_local_process_identity_and_launch_fail_closed(tmp_path, monkeypatch):
    run = local_run(tmp_path)
    backend, _ = local_backend(tmp_path, run)
    original_read = local_module.Path.read_text

    def fail_read(self, *args, **kwargs):
        if str(self).startswith("/proc/"):
            raise OSError("gone")
        return original_read(self, *args, **kwargs)

    monkeypatch.setattr(local_module.Path, "read_text", fail_read)
    assert local_module._process_identity(999999) is None
    monkeypatch.setattr(
        local_module.Path, "read_text",
        lambda self, **kwargs: "malformed" if str(self).startswith("/proc/")
        else original_read(self, **kwargs),
    )
    assert local_module._process_identity(1) is None
    monkeypatch.setattr(
        local_module.Path, "read_text",
        lambda self, **kwargs: "1 (x) S short" if str(self).startswith("/proc/")
        else original_read(self, **kwargs),
    )
    assert local_module._process_identity(1) is None

    class FakeProcess:
        pid = 123

    monkeypatch.setattr(local_module.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(local_module, "_process_identity", lambda _pid: None)
    killed = []
    monkeypatch.setattr(local_module.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    attempt = Path(run["storage"]["run_dir"]) / "attempts" / "attempt-001"
    attempt.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="verifiable"):
        backend._launch(
            ["true"], workdir=Path(run["backend"]["workdir"]),
            result_path=attempt / "result", stdout_path=attempt / "out",
            stderr_path=attempt / "err", environment={},
        )
    assert killed

    monkeypatch.setattr(
        local_module.os, "killpg",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError()),
    )
    with pytest.raises(RuntimeError, match="verifiable"):
        backend._launch(
            ["true"], workdir=Path(run["backend"]["workdir"]),
            result_path=attempt / "result", stdout_path=attempt / "out",
            stderr_path=attempt / "err", environment={},
        )


def test_local_submit_terminates_worker_if_identity_record_cannot_persist(
    tmp_path, monkeypatch,
):
    run = local_run(tmp_path)
    backend, _ = local_backend(tmp_path, run)
    backend.s = replace(
        backend.s,
        atomic_write=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk")),
    )
    monkeypatch.setattr(backend, "_launch", lambda *args, **kwargs: (123, "456"))
    terminated = []
    monkeypatch.setattr(
        backend, "_terminate", lambda control, grace: terminated.append((control, grace)),
    )
    with pytest.raises(OSError, match="disk"):
        backend.submit({}, run, {
            "attempt_id": "attempt-001", "command": ["true"],
        }, dry_run=False, intent=local_intent(backend, run))
    assert terminated[0][0]["backend_job_id"] == "123:456"
    assert terminated[0][1] == 0


def test_local_termination_boundaries(monkeypatch):
    assert LocalBackend._alive({}) is False
    with pytest.raises(ValueError, match="must not be negative"):
        LocalBackend._terminate({}, -1)
    LocalBackend._terminate({}, 0)

    states = iter([True, True, True])
    monkeypatch.setattr(LocalBackend, "_alive", staticmethod(lambda _control: next(states)))
    monkeypatch.setattr(local_module.time, "monotonic", lambda: 1.0)
    killed = []
    monkeypatch.setattr(local_module.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    LocalBackend._terminate({"pid": 123}, 0)
    assert killed == [(123, local_module.signal.SIGTERM), (123, local_module.signal.SIGKILL)]


def test_local_worker_records_command_start_failure(tmp_path):
    run = local_run(tmp_path)
    backend, record = local_backend(tmp_path, run)
    job_id = backend.submit({}, run, {
        "attempt_id": "attempt-001", "command": ["/definitely/missing/command"],
    }, dry_run=False, intent=local_intent(backend, run))
    record["backend_job_id"] = job_id
    status = wait_for_terminal(backend, run)
    assert status["state"] == "FAILED"
    assert status["exit_code"] == 127
    assert backend.collect({}, run)["process_evidence"]["observed"] is True


def test_local_worker_main_records_results_and_validates_command(tmp_path, monkeypatch):
    records = []
    monkeypatch.setattr(_local_worker, "atomic_write", lambda path, value: records.append((path, value)))
    monkeypatch.setattr(
        _local_worker.subprocess, "run", lambda command, check: SimpleNamespace(returncode=3),
    )
    result_path = tmp_path / "result.json"
    assert _local_worker.main(["--result", str(result_path), "--", "command"]) == 0
    assert records[-1][1]["exit_code"] == 3
    assert _local_worker.main(["--result", str(result_path), "command"]) == 0

    monkeypatch.setattr(
        _local_worker.subprocess, "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("missing")),
    )
    assert _local_worker.main(["--result", str(result_path), "missing"]) == 0
    assert records[-1][1]["exit_code"] == 127
    with pytest.raises(SystemExit):
        _local_worker.main(["--result", str(result_path)])


def test_local_worker_module_entrypoint(tmp_path, monkeypatch):
    result = tmp_path / "entrypoint.json"
    monkeypatch.setattr(sys, "argv", [
        "experiment_control._local_worker", "--result", str(result), "--", "/bin/true",
    ])
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        with pytest.raises(SystemExit) as captured:
            runpy.run_module("experiment_control._local_worker", run_name="__main__")
    assert captured.value.code == 0
    assert json.loads(result.read_text())["exit_code"] == 0
