"""SenseCore REST scheduler adapter with exact identity and bounded evidence."""

from __future__ import annotations

from .capabilities import SENSECORE

import hashlib
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any

from .services import BackendServices
from .sensecore_rest import create_document
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
_COLLECTION_LOG_TAIL = 10000
_PROCESS_EVIDENCE_LOG_TAIL = 200
_STRUCTURED_EVIDENCE_PREFIX = "EXPERIMENT_EVIDENCE_JSON="
_STRUCTURED_EVIDENCE_MAX_CHARS = 1048576
_STRUCTURED_EVIDENCE_RESERVED_KEYS = frozenset({
    "backend",
    "backend_job_id",
    "evidence_unavailable_reason",
    "live_logs_expired",
    "live_logs_available",
    "log_error",
    "log_source",
    "logs_historical",
    "last_log_at",
    "process_observed_at",
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
    """Bind one durable submission token into a unique resource name."""
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
    if raw == "SUSPENDING":
        return "UNKNOWN"
    if raw in {"SUSPENDED", "STOPPED"}:
        return "CANCELLED"
    if raw in {"FAILED", "ERROR"}:
        return "FAILED"
    if raw in {"DELETING", "DELETED", "CANCELLED", "CANCELED"}:
        return "CANCELLED"
    return "UNKNOWN"


class SenseCoreBackend:
    kind = "sensecore"
    capabilities = SENSECORE

    def __init__(self, services: BackendServices):
        self.s = services
        self._rest = None

    @property
    def rest(self):
        if self._rest is None:
            self._rest = self.s.sensecore_rest()
        return self._rest

    def availability(self) -> PreflightReport:
        """Use signed, read-only identity and workspace queries; no SCO probe."""
        try:
            self.rest.identity()
            self.rest.resources("compute.workspace.v1.instance")
        except (RuntimeError, ValueError):
            check = PreflightCheck("sensecore-rest", "authentication", "FAIL",
                                   "SenseCore REST credentials, scope or connectivity is unavailable")
        else:
            check = PreflightCheck("sensecore-rest", "authentication", "PASS",
                                   "SenseCore REST identity and resource queries are permitted")
        return PreflightReport(self.kind, "doctor", (check,))

    @staticmethod
    def redactor_bin() -> str:
        """Resolve the packaged Rust sanitizer executable."""
        return os.environ.get("EXPERIMENTCTL_REDACTOR_BIN", "experiment-redact")

    @staticmethod
    def create_timeout_seconds() -> int:
        """Bound ambiguous create waits so the durable outbox can reconcile."""
        raw = os.environ.get("EXPERIMENTCTL_SENSECORE_CREATE_TIMEOUT_SECONDS", "120")
        if not raw.isdigit() or not 10 <= int(raw) <= 600:
            raise ValueError(
                "EXPERIMENTCTL_SENSECORE_CREATE_TIMEOUT_SECONDS must be an integer from 10 to 600"
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
        if scope not in {"stage", "submit", "observe"}:
            raise ValueError(f"unsupported preflight scope: {scope}")
        try:
            self.rest.identity()
            self.find(run)
            if scope in {"stage", "submit"}:
                specs = self.rest.specs(run["backend"])
                if not any(s["name"] == run["backend"]["worker_spec"] for s in specs):
                    raise ValueError("requested worker spec is unavailable")
        except (RuntimeError, ValueError):
            check = PreflightCheck("sensecore-rest", "authorization", "FAIL",
                                   "SenseCore REST identity, exact job query or requested resources are unavailable")
        else:
            check = PreflightCheck("sensecore-rest", "authorization", "PASS",
                                   "REST exact-name query and requested resources are permitted")
        return PreflightReport(self.kind, scope, (check,))

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

    def describe(self, run: dict[str, Any], resource_name: str | None = None) -> dict[str, Any]:
        backend = run["backend"]
        raw = self.rest.describe(backend, str(resource_name or backend["job_name"]))
        role = (raw.get("roles") or [{}])[0]
        spec = (role.get("resource_spec") or [{}])[0]
        return {"name": raw["name"], "state": raw.get("state"),
                "pool": raw.get("resource_pool", {}).get("name"), "spec": spec.get("name")}

    def find(self, run: dict[str, Any], resource_name: str | None = None) -> list[dict[str, Any]]:
        backend = run["backend"]
        rows = self.rest.find(backend, str(resource_name or backend["job_name"]))
        return [{"name": row["name"], "state": row.get("state")} for row in rows]

    def stage(self, campaign, run, source_id, source_bundle) -> bool:
        # SenseCore consumes an immutable registry image; no controller-side
        # source upload is required for this backend.
        return True

    def create_document(self, manifest: dict[str, Any], *, submission_token: str | None = None) -> dict[str, Any]:
        backend = manifest["backend"]
        resource_name = (
            submission_resource_name(str(backend["job_name"]), str(manifest["attempt_id"]), submission_token)
            if submission_token else scheduler_job_name(str(backend["job_name"]), str(manifest["attempt_id"]))
        )
        command = ["env", f"BACKEND_JOB_ID={resource_name}",
                   *[str(v) for v in (self.s.dispatch_command(manifest) if submission_token else manifest["command"])]]
        return create_document(backend, resource_name, digest_pinned_image(str(backend["image"]), str(manifest["image_id"])), shlex.join(command))

    def render(self, manifest: AttemptManifest) -> str:
        return json.dumps({"method": "POST", "workspace": manifest["backend"]["workspace"],
                           "training_job": self.create_document(manifest)}, sort_keys=True)

    def _redact_error(self, text: str) -> str:
        result = self.s.run_command(
            [self.redactor_bin()],
            input_text=text, check=False,
        )
        if result.returncode:
            raise RuntimeError("credential redactor failed")
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
        document = self.create_document(manifest, submission_token=token)
        if self.find(run, resource_name):
            raise FileExistsError(f"SenseCore job already exists: {resource_name}")
        self.rest.create(backend, document, timeout=self.create_timeout_seconds())
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
        cancellation_requested = self._cancellation_requested(campaign, run, resource_name)
        state = normalize_state(
            str(summary.get("state", "")),
            cancellation_requested=cancellation_requested,
        )
        return {
            "run_id": run["run_id"], "backend": "sensecore",
            "backend_job_id": record["backend_job_id"],
            "state": state,
            "raw_state": summary.get("state"), "pool": summary.get("pool"), "spec": summary.get("spec"),
            "failure_class": None,
            "reason": "cancellation_requested" if cancellation_requested else (
                "provider_stopped_cause_unknown" if summary.get("state") in {"SUSPENDED", "STOPPED"} else None),
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
        self.rest.stop(backend, resource_name)
        return self.status(campaign, run)

    def collect(self, campaign, run) -> CollectionResult:
        # Checkpoints and structured evidence are sparse compared with tqdm output
        # and generation-evaluation logs.  Keep the public process tail bounded,
        # but inspect the largest supported live-log window for durable evidence.
        snapshot = self.logs(campaign, run, tail=_COLLECTION_LOG_TAIL)
        historical = snapshot.get("historical", False)
        log_source = "sensecore_offline_logs" if historical else "sensecore_stream_logs"
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
        ][-_PROCESS_EVIDENCE_LOG_TAIL:]
        result = {"run_id": run["run_id"], "backend": "sensecore", "model_observed": bool(metrics),
                  "latest_metric": metrics[-1] if metrics else None, "metric_log_lines": metric_lines[-20:],
                  "live_logs_expired": snapshot["expired"],
                  "live_logs_available": snapshot.get("available", True) and not historical,
                  "log_source": log_source, "logs_historical": historical,
                  "last_log_at": snapshot.get("last_log_at"),
                  **({"process_observed_at": snapshot.get("last_log_at")} if historical else {}),
                  "log_error": snapshot.get("error") or snapshot.get("live_error"),
                  "process_evidence": {
                      "observed": bool(process_lines) and not historical and not snapshot["expired"] and snapshot.get("available", True),
                      "sources": {"combined": log_source},
                      **({"observed_at": snapshot.get("last_log_at")} if historical else {}),
                      # REST exposes one sanitized combined stream rather than
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
                "source": log_source,
            }
        if snapshot["expired"]:
            result["evidence_unavailable_reason"] = "live_logs_expired"
        elif not snapshot.get("available", True):
            result["evidence_unavailable_reason"] = snapshot.get("unavailable_reason") or "live_logs_unavailable"
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
        payload = self.rest.workers(backend, resource_name)
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
        result = self.rest.logs(backend, resource_name, tail)
        redacted = self._redact_error(result["text"])
        return {
            "run_id": run["run_id"], "backend": "sensecore",
            "backend_job_id": resource_name, "tail": tail,
            "lines": redacted.splitlines()[-tail:], "expired": result["expired"],
            "stream_exit_code": result["exit_code"],
            "available": result.get("available", True), "error": result.get("error"),
            "live_error": result.get("live_error"),
            "source": result.get("source", "live"), "historical": result.get("historical", False),
            "last_log_at": result.get("last_log_at"), "truncated": result.get("truncated", False),
            "unavailable_reason": result.get("unavailable_reason"),
        }
