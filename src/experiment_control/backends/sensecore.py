"""SenseCore SCO side-effect adapter with immediate output sanitization."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .services import BackendServices
from ..contracts import (
    AssetVerification,
    AttemptManifest,
    BackendStatus,
    CollectionResult,
    LiveBackendLogs,
    PreflightScope,
    RunSpec,
    SubmissionIntent,
    SubmissionRequest,
)
from ..preflight import PreflightCheck, PreflightReport
from ..identity import IdentityReport
from ..submission import require_submission_intent, validate_submission_token


SENSECORE_BASE_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*$")
SENSECORE_ATTEMPT_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
_DOCTOR_TOOL_PROBE_TIMEOUT_SECONDS = 5.0
_DOCTOR_WORKSPACE_PROBE_TIMEOUT_SECONDS = 25.0
_STRUCTURED_EVIDENCE_PREFIX = "EXPERIMENT_EVIDENCE_JSON="
_STRUCTURED_EVIDENCE_MAX_CHARS = 1048576
_STRUCTURED_EVIDENCE_RESERVED_KEYS = frozenset({
    "backend",
    "backend_job_id",
    "evidence_unavailable_reason",
    "live_logs_expired",
    "process_evidence",
    "worker_evidence_available",
    "worker_phases",
    "worker_state",
})


def _structured_evidence(
    lines: list[str], *, run: RunSpec, attempt_id: str,
) -> dict[str, Any] | None:
    """Return the last identity-bound project summary emitted by the worker."""
    for line in reversed(lines):
        marker = line.find(_STRUCTURED_EVIDENCE_PREFIX)
        if marker < 0:
            continue
        encoded = line[marker + len(_STRUCTURED_EVIDENCE_PREFIX):].strip()
        if not encoded or len(encoded) > _STRUCTURED_EVIDENCE_MAX_CHARS:
            raise RuntimeError("SenseCore structured evidence is empty or oversized")
        try:
            payload = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise RuntimeError("SenseCore structured evidence is malformed") from error
        if not isinstance(payload, dict):
            raise RuntimeError("SenseCore structured evidence must be an object")
        expected = {
            "run_id": str(run["run_id"]),
            "attempt_id": str(attempt_id),
            "image_id": str(run["image_id"]),
        }
        for key, value in expected.items():
            if str(payload.get(key) or "") != value:
                raise RuntimeError(
                    f"SenseCore structured evidence {key} conflicts with the exact Attempt"
                )
        return {
            key: value for key, value in payload.items()
            if key not in _STRUCTURED_EVIDENCE_RESERVED_KEYS
        }
    return None


def scheduler_job_name(base_name: str, attempt_id: str) -> str:
    """Return a deterministic attempt-qualified SenseCore resource name."""
    if not SENSECORE_BASE_NAME_RE.fullmatch(base_name):
        raise ValueError(
            "SenseCore base job name must start with a lowercase letter and use only "
            "lowercase letters, digits, and hyphens"
        )
    if not SENSECORE_ATTEMPT_NAME_RE.fullmatch(attempt_id):
        raise ValueError(
            "SenseCore attempt ID must use lowercase letters, digits, and internal hyphens"
        )
    raw = f"{base_name}--{attempt_id}"
    if len(raw) <= 63:
        return raw
    digest = hashlib.sha256(raw.encode()).hexdigest()[:10]
    return f"{raw[:51]}--{digest}"


def submission_resource_name(base_name: str, attempt_id: str, token: str) -> str:
    """Bind one durable submission token into a unique SCO resource name."""
    scheduler_job_name(base_name, attempt_id)  # validate both authored parts
    token = validate_submission_token(token)
    raw = f"{base_name}--{attempt_id}"
    if len(raw) + len(token) + 2 <= 63:
        return f"{raw}--{token}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:10]
    prefix_budget = 63 - len(token) - len(digest) - 4
    prefix = raw[:prefix_budget].rstrip("-")
    return f"{prefix}--{digest}--{token}"


def digest_pinned_image(image_tag: str, image_id: str) -> str:
    """Resolve an authored registry tag to the manifest's immutable digest."""
    if not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", image_id):
        raise ValueError("SenseCore image_id must be a registry sha256 digest")
    if "@" in image_tag:
        raise ValueError("SenseCore backend.image must retain the authored immutable tag")
    leaf = image_tag.rsplit("/", 1)[-1]
    if ":" not in leaf:
        raise ValueError("SenseCore backend.image must include an immutable source-qualified tag")
    repository, tag = image_tag.rsplit(":", 1)
    if not repository or not tag or tag in {"latest", "runtime", "seed"}:
        raise ValueError("SenseCore backend.image must include an immutable source-qualified tag")
    return f"{repository}@{image_id}"


def normalize_state(raw_state: str, *, cancellation_requested: bool = False) -> str:
    raw = str(raw_state or "").upper()
    if cancellation_requested and raw in {"SUSPENDING", "SUSPENDED", "DELETING", "DELETED"}:
        return "CANCELLED"
    if raw in {"WAITING", "INIT", "QUEUEING", "PENDING", "CREATING"}:
        return "QUEUED"
    if raw in {"STARTING", "RECOVERING"}:
        return "STARTING"
    if raw in {"RUNNING", "RESTARTING"}:
        return "RUNNING"
    if raw in {"SUCCEEDED", "COMPLETED"}:
        return "SUCCEEDED"
    if raw in {"SUSPENDING", "SUSPENDED"}:
        return "PREEMPTED"
    if raw in {"FAILED", "ERROR"}:
        return "FAILED"
    if raw in {"DELETING", "DELETED", "CANCELLED", "CANCELED"}:
        return "CANCELLED"
    return "UNKNOWN"


class SenseCoreBackend:
    kind = "sensecore"

    def __init__(self, services: BackendServices):
        self.s = services

    def availability(self) -> PreflightReport:
        """Check SCO, its sanitizer, and credential-backed API access.

        Command output is captured and deliberately discarded.  In particular,
        Doctor never renders a workspace table or a SCO authentication error.
        """
        sco = self.sco_bin({})
        tools = (
            ("sco-cli", [sco, "version"]),
            ("safe-sco", [self.safe_sco_bin(), "normalize-state", "RUNNING"]),
            ("bash-cli", ["bash", "--version"]),
            ("timeout-cli", ["timeout", "--version"]),
        )
        checks = []
        for name, command in tools:
            try:
                result = self.s.run_command(
                    command, check=False,
                    timeout_seconds=_DOCTOR_TOOL_PROBE_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired:
                checks.append(PreflightCheck(
                    name, "tool", "FAIL", f"{name} probe timed out",
                ))
                continue
            checks.append(PreflightCheck(
                name, "tool", "PASS" if result.returncode == 0 else "FAIL",
                f"{name} is executable" if result.returncode == 0
                else f"{name} is unavailable",
            ))
        if all(check.status == "PASS" for check in checks):
            try:
                access = self.s.run_command([
                    "timeout", "20s", "env",
                    "-u", "http_proxy", "-u", "https_proxy", "-u", "all_proxy",
                    "-u", "HTTP_PROXY", "-u", "HTTPS_PROXY", "-u", "ALL_PROXY",
                    sco, "ws", "instances", "list",
                ], check=False,
                    timeout_seconds=_DOCTOR_WORKSPACE_PROBE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                checks.append(PreflightCheck(
                    "workspace-access", "authentication", "FAIL",
                    "SCO workspace access probe timed out",
                ))
            else:
                checks.append(PreflightCheck(
                    "workspace-access", "authentication",
                    "PASS" if access.returncode == 0 else "FAIL",
                    "SCO credentials permit a workspace query" if access.returncode == 0
                    else "SCO authentication, components, or API connectivity is unavailable",
                ))
        return PreflightReport(self.kind, "doctor", tuple(checks))

    @staticmethod
    def sco_bin(run: dict[str, Any]) -> str:
        """Resolve a non-secret CLI override without exposing SCO credentials."""
        return str(
            run.get("backend", {}).get("sco_bin")
            or os.environ.get("EXPERIMENTCTL_SCO_BIN", "sco")
        )

    @staticmethod
    def safe_sco_bin() -> str:
        """Resolve the packaged Rust sanitizer executable."""
        return os.environ.get("EXPERIMENTCTL_SAFE_SCO_BIN", "experiment-safe-sco")

    @staticmethod
    def create_timeout_seconds() -> int:
        """Bound ambiguous create waits so the durable outbox can reconcile."""
        raw = os.environ.get("EXPERIMENTCTL_SCO_CREATE_TIMEOUT_SECONDS", "120")
        if not raw.isdigit() or not 10 <= int(raw) <= 600:
            raise ValueError(
                "EXPERIMENTCTL_SCO_CREATE_TIMEOUT_SECONDS must be an integer from 10 to 600"
            )
        return int(raw)

    def validate(self, run: RunSpec) -> None:
        backend = run["backend"]
        required = {"workspace", "aec2", "worker_spec", "image", "storage_mount", "quota_type", "job_name"}
        missing = sorted(key for key in required if not backend.get(key))
        if missing:
            raise ValueError(f"run {run['run_id']} backend is missing: {missing}")
        try:
            scheduler_job_name(str(backend["job_name"]), "attempt-001")
        except ValueError as error:
            raise ValueError(f"run {run['run_id']} {error}") from error
        if backend["quota_type"] != "spot":
            raise ValueError("SenseCore runs for this account must use spot quota")
        if not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", str(run["image_id"])):
            raise ValueError(f"run {run['run_id']} SenseCore image_id must be a registry digest")
        try:
            digest_pinned_image(str(backend["image"]), str(run["image_id"]))
        except ValueError as error:
            raise ValueError(f"run {run['run_id']} {error}") from error
        mount_path = Path(str(backend["storage_mount"]).rsplit(":", 1)[-1])
        if not mount_path.is_absolute():
            raise ValueError(f"run {run['run_id']} SenseCore storage mount path must be absolute")
        for field, value in run["storage"].items():
            if field == "run_dir" or field.endswith(("_root", "_home", "_cache")):
                path = Path(str(value))
                if not path.is_relative_to(mount_path):
                    raise ValueError(
                        f"run {run['run_id']} storage.{field} must be under mounted path {mount_path}"
                    )

    def environment(self, campaign, run, source_id, attempt_id) -> dict[str, str]:
        backend = run["backend"]
        return {
            "QUOTA_TYPE": str(backend["quota_type"]),
            "RESOURCE_SPEC": str(backend["worker_spec"]),
        }

    def preflight(self, run: RunSpec, *, scope: PreflightScope) -> PreflightReport:
        """Check the SCO executable and sanitized workspace access."""
        if scope not in {"stage", "submit", "observe"}:
            raise ValueError(f"unsupported preflight scope: {scope}")
        sco = self.sco_bin(run)
        version = self.s.run_command(
            ["env", "-u", "http_proxy", "-u", "https_proxy", "-u", "all_proxy",
             "-u", "HTTP_PROXY", "-u", "HTTPS_PROXY", "-u", "ALL_PROXY",
             sco, "version"],
            check=False,
        )
        checks = [
            PreflightCheck(
                "sco-cli", "tool", "PASS" if version.returncode == 0 else "FAIL",
                "SCO CLI is executable" if version.returncode == 0 else "SCO CLI is unavailable",
            )
        ]
        if version.returncode == 0:
            try:
                self.find(run)
            except (RuntimeError, ValueError):
                checks.append(PreflightCheck(
                    "workspace-access", "authentication", "FAIL",
                    "sanitized exact-name query failed; refresh SCO login or connectivity",
                ))
            else:
                checks.append(PreflightCheck(
                    "workspace-access", "authorization", "PASS",
                    "exact-name query is permitted",
                ))
        return PreflightReport(self.kind, scope, tuple(checks))

    def submission_request(self, campaign, run, attempt_id) -> SubmissionRequest:
        backend = run["backend"]
        return {
            "scheduler_name": scheduler_job_name(str(backend["job_name"]), attempt_id),
            "image_tag": str(backend["image"]),
            "image_digest": str(run["image_id"]),
            "image_reference": digest_pinned_image(
                str(backend["image"]), str(run["image_id"])
            ),
        }

    def recover_submission(self, run, intent, attempt_id) -> str | None:
        token, request = require_submission_intent(intent)
        base_name = scheduler_job_name(str(run["backend"]["job_name"]), attempt_id)
        requested = str(request.get("scheduler_name") or "")
        if requested != base_name:
            raise RuntimeError("SenseCore submission intent has a conflicting scheduler name")
        expected = submission_resource_name(
            str(run["backend"]["job_name"]), attempt_id, token
        )
        matches = self.find(run, expected)
        if len(matches) > 1:
            raise RuntimeError(
                f"ambiguous scheduler identity: {len(matches)} jobs match this attempt"
            )
        return expected if matches else None

    def identity(self, campaign, run, attempt_id) -> IdentityReport:
        resource_name = scheduler_job_name(str(run["backend"]["job_name"]), attempt_id)
        matches = self.find(run, resource_name)
        return IdentityReport(
            available=not matches,
            ambiguous=len(matches) > 1,
            scheduler_job_ids=tuple(str(item["name"]) for item in matches),
        )

    def verify_assets(self, run, probes) -> AssetVerification:
        return {
            "missing": None,
            "verification": "requires-running-sensecore-worker",
            "verified_on": None,
        }

    def safe_command(self, arguments: list[str], mode: str) -> list[str]:
        sco = shlex.join(["env", "-u", "http_proxy", "-u", "https_proxy", "-u", "all_proxy",
                          "-u", "HTTP_PROXY", "-u", "HTTPS_PROXY", "-u", "ALL_PROXY", *arguments])
        sanitizer = shlex.join([self.safe_sco_bin(), mode])
        redactor = shlex.join([self.safe_sco_bin(), "redact-lines"])
        return [
            "bash", "-o", "pipefail", "-c",
            f"{sco} 2> >({redactor} >&2) | {sanitizer}",
        ]

    def describe(
        self, run: dict[str, Any], resource_name: str | None = None
    ) -> dict[str, Any]:
        backend = run["backend"]
        exact_name = str(resource_name or backend["job_name"])
        result = self.s.run_command(
            self.safe_command([self.sco_bin(run), "acp", "jobs", "describe", exact_name,
                               "--workspace-name", backend["workspace"], "-o", "json"], "job-summary"),
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                self._redact_error(result.stderr)
                or "sanitized SenseCore describe failed"
            )
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict):
            raise ValueError("SenseCore describe sanitizer returned a non-object")
        return payload

    def find(
        self, run: dict[str, Any], resource_name: str | None = None
    ) -> list[dict[str, Any]]:
        backend = run["backend"]
        exact_name = str(resource_name or backend["job_name"])
        result = self.s.run_command(
            self.safe_command([self.sco_bin(run), "acp", "jobs", "list", "--workspace-name", backend["workspace"],
                               "--name", exact_name, "--page-size", "5", "-o", "json"], "job-list"),
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                self._redact_error(result.stderr)
                or "sanitized SenseCore list failed"
            )
        payload = json.loads(result.stdout)
        if not isinstance(payload, list):
            raise ValueError("SenseCore list sanitizer returned a non-list")
        return [item for item in payload if item.get("name") == exact_name]

    def stage(self, campaign, run, source_id, source_bundle) -> bool:
        # SenseCore consumes an immutable registry image; no controller-side
        # source upload is required for this backend.
        return True

    def _create_command(
        self, manifest: dict[str, Any], *, submission_token: str | None = None,
    ) -> list[str]:
        backend = manifest["backend"]
        resource_name = (
            submission_resource_name(
                str(backend["job_name"]), str(manifest["attempt_id"]),
                submission_token,
            )
            if submission_token else scheduler_job_name(
                str(backend["job_name"]), str(manifest["attempt_id"])
            )
        )
        command = [
            "env", f"BACKEND_JOB_ID={resource_name}",
            *[str(value) for value in manifest["command"]],
        ]
        return [
            "timeout", f"{self.create_timeout_seconds()}s",
            "env", "-u", "http_proxy", "-u", "https_proxy", "-u", "all_proxy",
            "-u", "HTTP_PROXY", "-u", "HTTPS_PROXY", "-u", "ALL_PROXY",
            self.sco_bin(manifest), "acp", "jobs", "create",
            "--workspace-name", backend["workspace"],
            "--aec2-name", backend["aec2"], "--name", resource_name,
            "--job-name", backend["display_name"],
            "--container-image-url", digest_pinned_image(
                str(backend["image"]), str(manifest["image_id"])
            ),
            "--training-framework", "pytorch", "--worker-spec", backend["worker_spec"],
            "--worker-nodes", str(backend.get("worker_nodes", 1)),
            "--priority", str(backend.get("priority", "NORMAL")),
            "--quota-type", backend["quota_type"], "--storage-mount", backend["storage_mount"],
            "--wait", "--command", shlex.join(command),
        ]

    def render(self, manifest: AttemptManifest) -> str:
        return shlex.join(self._create_command(manifest))

    def _redact_error(self, text: str) -> str:
        result = self.s.run_command(
            [self.safe_sco_bin(), "redact-lines"],
            input_text=text, check=False,
        )
        return result.stdout.strip()

    def submit(
        self, campaign, run, manifest, *, dry_run: bool,
        intent: SubmissionIntent | None = None,
    ) -> str:
        backend = run["backend"]
        if str(manifest.get("image_id")) != str(run.get("image_id")):
            raise ValueError("SenseCore frozen manifest image_id conflicts with the run")
        if dry_run:
            return "DRY_RUN"
        token, request = require_submission_intent(intent)
        expected_request = self.submission_request(
            campaign, run, str(manifest["attempt_id"])
        )
        if request.get("scheduler_name") != expected_request["scheduler_name"]:
            raise RuntimeError("SenseCore submission intent has a conflicting scheduler name")
        resource_name = submission_resource_name(
            str(backend["job_name"]), str(manifest["attempt_id"]), token
        )
        create = self._create_command(manifest, submission_token=token)
        if self.find(run, resource_name):
            raise FileExistsError(f"SenseCore job already exists: {resource_name}")
        result = self.s.run_command(create, check=False)
        if result.returncode != 0:
            raise RuntimeError(self._redact_error(result.stderr) or "SenseCore create failed")
        summary = self.describe(run, resource_name)
        if summary.get("name") != resource_name:
            raise RuntimeError("SenseCore accepted create but exact job was not observable")
        return resource_name

    def status(self, campaign, run) -> BackendStatus:
        record = self.s.backend_record(campaign, run)
        resource_name = str(record["backend_job_id"])
        summary = self.describe(run, resource_name)
        if summary.get("name") != resource_name:
            raise RuntimeError("SenseCore exact job describe returned a conflicting resource")
        state = normalize_state(
            str(summary.get("state", "")),
            cancellation_requested=self._cancellation_requested(
                campaign, run, resource_name
            ),
        )
        return {
            "run_id": run["run_id"], "backend": "sensecore",
            "backend_job_id": record["backend_job_id"],
            "state": state,
            "raw_state": summary.get("state"), "pool": summary.get("pool"), "spec": summary.get("spec"),
            "failure_class": "preemption" if state == "PREEMPTED" else None,
        }

    def _cancellation_requested(
        self, campaign: dict[str, Any], run: dict[str, Any], resource_name: str
    ) -> bool:
        """Match attempt-local or legacy run-level cancellation evidence exactly."""
        local_dir = self.s.local_run_dir(campaign, run)
        markers = [local_dir / "cancel_requested.json"]
        if local_dir.parent.name == "attempts":
            markers.append(local_dir.parent.parent / "cancel_requested.json")
        for marker in markers:
            if not marker.is_file():
                continue
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    "SenseCore cancellation evidence is unreadable"
                ) from error
            if not isinstance(payload, dict) or not payload.get("backend_job_id"):
                raise RuntimeError("SenseCore cancellation evidence is malformed")
            if str(payload["backend_job_id"]) == resource_name:
                return True
        return False

    def cancel(self, campaign, run) -> BackendStatus:
        current = self.status(campaign, run)
        if current["state"] in {"SUCCEEDED", "FAILED", "PREEMPTED", "CANCELLED"}:
            return current
        marker = self.s.local_run_dir(campaign, run) / "cancel_requested.json"
        self.s.atomic_write(marker, {
            "run_id": run["run_id"], "backend_job_id": current["backend_job_id"],
            "requested_at": self.s.utc_now(),
        })
        backend = run["backend"]
        resource_name = str(current["backend_job_id"])
        result = self.s.run_command(
            ["env", "-u", "http_proxy", "-u", "https_proxy", "-u", "all_proxy",
             "-u", "HTTP_PROXY", "-u", "HTTPS_PROXY", "-u", "ALL_PROXY",
             self.sco_bin(run), "acp", "jobs", "stop", resource_name, "--workspace-name", backend["workspace"]],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(self._redact_error(result.stderr) or "SenseCore stop failed")
        return self.status(campaign, run)

    def collect(self, campaign, run) -> CollectionResult:
        snapshot = self.logs(campaign, run, tail=200)
        lines = snapshot["lines"]
        record = self.s.backend_record(campaign, run)
        structured = _structured_evidence(
            lines, run=run, attempt_id=str(record["attempt_id"]),
        )
        metrics = [metric for line in lines if (metric := self.s.parse_metric(campaign, line))]
        checkpoints = [
            checkpoint for line in lines
            if (checkpoint := self.s.parse_checkpoint(campaign, line))
        ]
        metric_lines = [line for line in lines if "Step " in line or "gPPL:" in line or ("plan" in line.lower() and "ppl" in line.lower())]
        process_lines = [
            line for line in lines if _STRUCTURED_EVIDENCE_PREFIX not in line
        ]
        result = {"run_id": run["run_id"], "backend": "sensecore", "model_observed": bool(metrics),
                  "latest_metric": metrics[-1] if metrics else None, "metric_log_lines": metric_lines[-20:],
                  "live_logs_expired": snapshot["expired"],
                  "process_evidence": {
                      "observed": bool(process_lines) and not snapshot["expired"],
                      "sources": {"combined": "sensecore_stream_logs"},
                      # SCO exposes one sanitized combined stream rather than
                      # distinct process stdout/stderr channels.
                      "stdout_tail": process_lines,
                      "stderr_tail": [],
                  }}
        if structured is not None:
            result.update(structured)
            result["run_id"] = run["run_id"]
            result["backend"] = "sensecore"
            result["model_observed"] = True
            result["structured_evidence"] = {
                "identity_verified": True,
                "source": "sensecore_stream_logs",
            }
        if snapshot["expired"]:
            result["evidence_unavailable_reason"] = "live_logs_expired"
        try:
            worker = self.workers(campaign, run)
        except (RuntimeError, ValueError):
            worker = {
                "worker_state": "UNKNOWN", "worker_phases": [],
                "worker_evidence_available": False,
            }
        result.update(worker)
        if checkpoints:
            latest = max(checkpoints, key=lambda item: int(item["step"]))
            result["latest_completed_checkpoint"] = latest["path"]
            result["latest_completed_checkpoint_step"] = latest["step"]
        return result

    def workers(self, campaign, run) -> dict[str, Any]:
        """Return sanitized worker-allocation evidence for the exact job."""
        backend = run["backend"]
        record = self.s.backend_record(campaign, run)
        resource_name = str(record["backend_job_id"])
        result = self.s.run_command(
            self.safe_command([
                self.sco_bin(run), "acp", "jobs", "get-workers", resource_name,
                "--workspace-name", backend["workspace"],
            ], "worker-list"),
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                self._redact_error(result.stderr)
                or "sanitized SenseCore worker query failed"
            )
        payload = json.loads(result.stdout)
        if not isinstance(payload, list):
            raise ValueError("SenseCore worker sanitizer returned a non-list")
        phases = [str(item.get("phase", "")) for item in payload if isinstance(item, dict)]
        normalized = {phase.casefold() for phase in phases}
        if normalized & {"running", "ready"}:
            state = "ALLOCATED"
        elif normalized & {"pending", "creating", "starting"}:
            state = "PENDING"
        elif normalized and normalized <= {
            "deleted", "succeeded", "failed", "stopped", "completed"
        }:
            state = "RELEASED"
        else:
            state = "UNKNOWN"
        return {
            "worker_state": state,
            "worker_phases": phases,
            "worker_evidence_available": True,
        }

    def logs(self, campaign, run, *, tail: int) -> LiveBackendLogs:
        if not 1 <= tail <= 10000:
            raise ValueError("tail must be between 1 and 10000")
        backend = run["backend"]
        record = self.s.backend_record(campaign, run)
        resource_name = str(record["backend_job_id"])
        result = self.s.run_command(
            ["timeout", "20s", "env", "-u", "http_proxy", "-u", "https_proxy", "-u", "all_proxy",
             "-u", "HTTP_PROXY", "-u", "HTTPS_PROXY", "-u", "ALL_PROXY", self.sco_bin(run), "acp", "jobs",
             "stream-logs", resource_name, "--workspace-name", backend["workspace"]],
            check=False,
        )
        redacted = self._redact_error(result.stdout + "\n" + result.stderr)
        lines = redacted.splitlines()[-tail:]
        expired = result.returncode != 0 and any(
            token in redacted.lower() for token in ("expired", "403", "offline log")
        )
        return {
            "run_id": run["run_id"], "backend": "sensecore",
            "backend_job_id": resource_name, "tail": tail,
            "lines": lines, "expired": expired, "stream_exit_code": result.returncode,
        }
