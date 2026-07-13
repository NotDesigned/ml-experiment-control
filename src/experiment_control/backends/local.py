"""Durable local-process backend for development and smoke tests."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .services import BackendServices
from ..contracts import (
    AssetVerification,
    AttemptManifest,
    BackendStatus,
    CollectionResult,
    PreflightScope,
    RunSpec,
    StreamBackendLogs,
    SubmissionRequest,
)
from ..identity import IdentityReport
from ..manifest import atomic_create
from ..preflight import PreflightCheck, PreflightReport
from ..project import SourceBundle
from ..redaction import redact_line


TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "CANCELLED"})


def _process_identity(pid: int) -> tuple[str, str] | None:
    """Return Linux process state and start ticks without trusting PID alone."""
    try:
        value = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    _, separator, fields = value.rpartition(") ")
    if not separator:
        return None
    parts = fields.split()
    return (parts[0], parts[19]) if len(parts) > 19 else None


class LocalBackend:
    """Launch attempt-qualified local process groups with durable result files."""

    kind = "local"

    def __init__(self, services: BackendServices):
        self.s = services

    @staticmethod
    def _attempt_dir(run: dict[str, Any], attempt_id: str) -> Path:
        return Path(str(run["storage"]["run_dir"])) / "attempts" / attempt_id

    @classmethod
    def _control_path(cls, run: dict[str, Any], attempt_id: str) -> Path:
        return cls._attempt_dir(run, attempt_id) / "local-process.json"

    @classmethod
    def _result_path(cls, run: dict[str, Any], attempt_id: str) -> Path:
        return cls._attempt_dir(run, attempt_id) / "local-result.json"

    @staticmethod
    def _load_object(path: Path, label: str) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"local {label} record is unreadable") from error
        if not isinstance(payload, dict):
            raise RuntimeError(f"local {label} record is malformed")
        return payload

    def validate(self, run: RunSpec) -> None:
        backend = run.get("backend")
        if not isinstance(backend, dict) or not backend.get("workdir"):
            raise ValueError(f"run {run.get('run_id')} local backend requires workdir")
        workdir = Path(str(backend["workdir"]))
        if not workdir.is_absolute():
            raise ValueError("local backend workdir must be absolute")
        run_dir = Path(str(run.get("storage", {}).get("run_dir", "")))
        if not run_dir.is_absolute():
            raise ValueError("local backend storage.run_dir must be absolute")

    def preflight(self, run: RunSpec, *, scope: PreflightScope) -> PreflightReport:
        if scope not in {"stage", "submit", "observe"}:
            raise ValueError(f"unsupported preflight scope: {scope}")
        self.validate(run)
        workdir = Path(str(run["backend"]["workdir"]))
        run_parent = Path(str(run["storage"]["run_dir"])).parent
        workdir_ready = workdir.is_dir()
        storage_ready = run_parent.is_dir() and os.access(run_parent, os.W_OK)
        identity_ready = _process_identity(os.getpid()) is not None
        checks = (
            PreflightCheck(
                "local-workdir", "storage", "PASS" if workdir_ready else "FAIL",
                "local workdir is available" if workdir_ready else "local workdir is unavailable",
            ),
            PreflightCheck(
                "local-run-storage", "storage",
                "PASS" if storage_ready else "FAIL",
                "local run storage is writable" if storage_ready
                else "local run storage is unavailable",
            ),
            PreflightCheck(
                "local-process-identity", "tool",
                "PASS" if identity_ready else "FAIL",
                "Linux process identity is available" if identity_ready
                else "Linux process identity is unavailable",
            ),
        )
        return PreflightReport(self.kind, scope, checks)

    def environment(self, campaign, run, source_id, attempt_id) -> dict[str, str]:
        return {"LOCAL_ATTEMPT_ID": str(attempt_id)}

    def submission_request(self, campaign, run, attempt_id) -> SubmissionRequest:
        return {
            "scheduler_name": f"local/{run['run_id']}/{attempt_id}",
            "workdir": str(run["backend"]["workdir"]),
        }

    def _control(self, run: dict[str, Any], attempt_id: str) -> dict[str, Any] | None:
        payload = self._load_object(self._control_path(run, attempt_id), "process")
        if payload is not None and payload.get("attempt_id") != attempt_id:
            raise RuntimeError("local process record has a conflicting attempt identity")
        return payload

    def recover_submission(self, run, intent, attempt_id) -> str | None:
        expected = self.submission_request({}, run, attempt_id)
        for key in ("scheduler_name", "workdir"):
            if intent.get(key, expected[key]) != expected[key]:
                raise RuntimeError(f"local submission intent has a conflicting {key}")
        control = self._control(run, attempt_id)
        if control is None:
            return None
        job_id = control.get("backend_job_id")
        if not job_id:
            raise RuntimeError("local submission claim exists without a process identity")
        return str(job_id)

    def identity(self, campaign, run, attempt_id) -> IdentityReport:
        control = self._control(run, attempt_id)
        if control is None:
            return IdentityReport(available=True, ambiguous=False)
        job_id = control.get("backend_job_id")
        return IdentityReport(
            available=False,
            ambiguous=False,
            scheduler_job_ids=(str(job_id),) if job_id else (),
        )

    def verify_assets(self, run, probes) -> AssetVerification:
        missing = []
        for probe in probes:
            path = Path(probe.path)
            exists = path.is_file() and path.stat().st_size > 0 if probe.file else path.is_dir()
            if not exists:
                missing.append({**probe.requirement.__dict__, "path": probe.path})
        return {"missing": missing, "verification": "local-filesystem", "verified_on": "localhost"}

    def stage(self, campaign, run, source_id, source_bundle: SourceBundle) -> bool:
        if not source_bundle.root.is_dir():
            raise ValueError("local source bundle root must be a directory")
        for required_path in source_bundle.required_paths:
            relative = Path(required_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"required source path must be relative: {required_path}")
            if not (source_bundle.root / relative).is_file():
                raise RuntimeError(f"local source is missing required project path: {required_path}")
        return True

    def render(self, manifest: AttemptManifest) -> str:
        return shlex.join([str(value) for value in manifest["command"]])

    def _launch(
        self, command: list[str], *, workdir: Path, result_path: Path,
        stdout_path: Path, stderr_path: Path, environment: dict[str, str],
    ) -> tuple[int, str]:
        worker = [
            sys.executable, "-m", "experiment_control._local_worker",
            "--result", str(result_path), "--", *command,
        ]
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            process = subprocess.Popen(
                worker, cwd=workdir, env=environment, stdout=stdout, stderr=stderr,
                start_new_session=True, close_fds=True,
            )
        identity = _process_identity(process.pid)
        if identity is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise RuntimeError("local worker started without a verifiable process identity")
        return process.pid, identity[1]

    def submit(self, campaign, run, manifest, *, dry_run: bool) -> str:
        self.validate(run)
        command = [str(value) for value in manifest.get("command", [])]
        if not command:
            raise ValueError("local manifest command must not be empty")
        if dry_run:
            return "DRY_RUN"
        attempt_id = str(manifest["attempt_id"])
        attempt_dir = self._attempt_dir(run, attempt_id)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        control_path = self._control_path(run, attempt_id)
        request = self.submission_request(campaign, run, attempt_id)
        claim = {
            "attempt_id": attempt_id,
            "scheduler_name": request["scheduler_name"],
            "state": "LAUNCHING",
            "created_at": self.s.utc_now(),
        }
        atomic_create(control_path, claim)
        environment = os.environ.copy()
        environment.update(self.environment(campaign, run, manifest.get("source_id"), attempt_id))
        pid, start_ticks = self._launch(
            command,
            workdir=Path(str(run["backend"]["workdir"])),
            result_path=self._result_path(run, attempt_id),
            stdout_path=attempt_dir / "stdout.log",
            stderr_path=attempt_dir / "stderr.log",
            environment=environment,
        )
        job_id = f"{pid}:{start_ticks}"
        control = {
            **claim, "state": "RUNNING", "pid": pid,
            "start_ticks": start_ticks, "backend_job_id": job_id,
        }
        try:
            self.s.atomic_write(control_path, control)
        except Exception:
            self._terminate(control, 0)
            raise
        return job_id

    @staticmethod
    def _alive(control: dict[str, Any]) -> bool:
        pid = control.get("pid")
        start_ticks = control.get("start_ticks")
        if not isinstance(pid, int) or not isinstance(start_ticks, str):
            return False
        identity = _process_identity(pid)
        return bool(identity and identity[0] != "Z" and identity[1] == start_ticks)

    def status(self, campaign, run) -> BackendStatus:
        record = self.s.backend_record(campaign, run)
        attempt_id = str(record["attempt_id"])
        control = self._control(run, attempt_id)
        if control is None or not control.get("backend_job_id"):
            raise RuntimeError("local process identity is not recorded")
        job_id = str(control["backend_job_id"])
        if str(record.get("backend_job_id")) != job_id:
            raise RuntimeError("local backend record has a conflicting process identity")
        result = self._load_object(self._result_path(run, attempt_id), "result")
        exit_code = result.get("exit_code") if result else None
        if result is not None and (
            result.get("worker_pid") != control.get("pid") or not isinstance(exit_code, int)
        ):
            raise RuntimeError("local result record has a conflicting process identity")
        if result is not None and exit_code == 0:
            state, raw_state, failure_class = "SUCCEEDED", "EXITED", None
        elif control.get("state") == "CANCEL_REQUESTED" and not self._alive(control):
            state, raw_state, failure_class = "CANCELLED", "CANCELLED", None
        elif result is not None:
            state, raw_state, failure_class = "FAILED", "EXITED", "unknown"
        elif self._alive(control):
            state, raw_state, failure_class = "RUNNING", "RUNNING", None
        else:
            state, raw_state, failure_class = "FAILED", "LOST", "scheduler"
        return {
            "run_id": run["run_id"], "backend": self.kind,
            "backend_job_id": job_id, "attempt_id": attempt_id,
            "state": state, "raw_state": raw_state,
            "exit_code": exit_code, "failure_class": failure_class,
        }

    @staticmethod
    def _terminate(control: dict[str, Any], grace_seconds: float) -> None:
        if grace_seconds < 0:
            raise ValueError("local cancel_grace_seconds must not be negative")
        if not LocalBackend._alive(control):
            return
        pid = int(control["pid"])
        os.killpg(pid, signal.SIGTERM)
        deadline = time.monotonic() + grace_seconds
        while LocalBackend._alive(control) and time.monotonic() < deadline:
            time.sleep(0.02)
        if LocalBackend._alive(control):
            os.killpg(pid, signal.SIGKILL)

    def cancel(self, campaign, run) -> BackendStatus:
        current = self.status(campaign, run)
        if current["state"] in TERMINAL_STATES:
            return current
        attempt_id = str(current["attempt_id"])
        control_path = self._control_path(run, attempt_id)
        control = self._control(run, attempt_id) or {}
        updated = {**control, "state": "CANCEL_REQUESTED", "cancel_requested_at": self.s.utc_now()}
        self.s.atomic_write(control_path, updated)
        self._terminate(updated, float(run["backend"].get("cancel_grace_seconds", 5)))
        return self.status(campaign, run)

    @staticmethod
    def _tail(path: Path, tail: int) -> list[str]:
        if not path.is_file():
            return []
        lines = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n").splitlines()
        return [redact_line(line) for line in lines if line.strip()][-tail:]

    def logs(self, campaign, run, *, tail: int) -> StreamBackendLogs:
        if not 1 <= tail <= 10000:
            raise ValueError("tail must be between 1 and 10000")
        record = self.s.backend_record(campaign, run)
        attempt_id = str(record["attempt_id"])
        attempt_dir = self._attempt_dir(run, attempt_id)
        return {
            "run_id": run["run_id"], "backend": self.kind,
            "backend_job_id": record["backend_job_id"], "attempt_id": attempt_id,
            "tail": tail,
            "sources": {
                "stdout": str(attempt_dir / "stdout.log"),
                "stderr": str(attempt_dir / "stderr.log"),
            },
            "stdout": self._tail(attempt_dir / "stdout.log", tail),
            "stderr": self._tail(attempt_dir / "stderr.log", tail),
        }

    def collect(self, campaign, run) -> CollectionResult:
        run_dir = Path(str(run["storage"]["run_dir"]))
        summary = self.s.summarize_run(campaign, run_dir)
        diagnostics = self.logs(campaign, run, tail=200)
        lines = [*diagnostics["stdout"], *diagnostics["stderr"]]
        metrics = [value for line in lines if (value := self.s.parse_metric(campaign, line))]
        checkpoints = [
            value for line in lines if (value := self.s.parse_checkpoint(campaign, line))
        ]
        if metrics:
            summary["latest_metric"] = metrics[-1]
        if checkpoints:
            latest = max(checkpoints, key=lambda item: int(item["step"]))
            summary["latest_completed_checkpoint"] = latest["path"]
            summary["latest_completed_checkpoint_step"] = latest["step"]
        summary.update({
            "run_dir": str(run_dir),
            "collected_from": str(run_dir),
            "process_evidence": {
                "observed": bool(lines),
                "sources": diagnostics["sources"],
                "stdout_tail": diagnostics["stdout"],
                "stderr_tail": diagnostics["stderr"],
            },
        })
        return summary
