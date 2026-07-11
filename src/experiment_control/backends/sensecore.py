"""SenseCore SCO side-effect adapter with immediate output sanitization."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any

from .services import BackendServices
from ..preflight import PreflightCheck, PreflightReport
from ..identity import IdentityReport


def scheduler_job_name(base_name: str, attempt_id: str) -> str:
    """Return a deterministic attempt-qualified SenseCore resource name."""
    raw = f"{base_name}--{attempt_id}"
    if len(raw) <= 63:
        return raw
    digest = hashlib.sha256(raw.encode()).hexdigest()[:10]
    return f"{raw[:51]}--{digest}"


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

    @staticmethod
    def sco_bin(run: dict[str, Any]) -> str:
        """Resolve a non-secret CLI override without exposing SCO credentials."""
        return str(
            run.get("backend", {}).get("sco_bin")
            or os.environ.get("EXPERIMENTCTL_SCO_BIN", "sco")
        )

    def validate(self, run: dict[str, Any]) -> None:
        backend = run["backend"]
        required = {"workspace", "aec2", "worker_spec", "image", "storage_mount", "quota_type", "job_name"}
        missing = sorted(key for key in required if not backend.get(key))
        if missing:
            raise ValueError(f"run {run['run_id']} backend is missing: {missing}")
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
            "BACKEND_JOB_ID": scheduler_job_name(str(backend["job_name"]), attempt_id),
            "QUOTA_TYPE": str(backend["quota_type"]),
            "RESOURCE_SPEC": str(backend["worker_spec"]),
        }

    def preflight(self, run: dict[str, Any], *, scope: str) -> PreflightReport:
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

    def submission_request(self, campaign, run, attempt_id) -> dict[str, Any]:
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
        expected = scheduler_job_name(str(run["backend"]["job_name"]), attempt_id)
        requested = str(intent.get("scheduler_name") or expected)
        if requested != expected:
            raise RuntimeError("SenseCore submission intent has a conflicting scheduler name")
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

    def verify_assets(self, run, probes) -> dict[str, Any]:
        return {
            "missing": None,
            "verification": "requires-running-sensecore-worker",
            "verified_on": None,
        }

    def safe_command(self, arguments: list[str], mode: str) -> list[str]:
        sco = shlex.join(["env", "-u", "http_proxy", "-u", "https_proxy", "-u", "all_proxy",
                          "-u", "HTTP_PROXY", "-u", "HTTPS_PROXY", "-u", "ALL_PROXY", *arguments])
        sanitizer = shlex.join([sys.executable, "-m", "experiment_control.safe_sco", mode])
        return ["bash", "-o", "pipefail", "-c", f"{sco} 2>&1 | {sanitizer}"]

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

    def _create_command(self, manifest: dict[str, Any]) -> list[str]:
        backend = manifest["backend"]
        resource_name = scheduler_job_name(
            str(backend["job_name"]), str(manifest["attempt_id"])
        )
        return [
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
            "--wait", "--command", shlex.join(manifest["command"]),
        ]

    def render(self, manifest) -> str:
        return shlex.join(self._create_command(manifest))

    def _redact_error(self, text: str) -> str:
        result = self.s.run_command(
            [sys.executable, "-m", "experiment_control.safe_sco", "redact-lines"],
            input_text=text, check=False,
        )
        return result.stdout.strip()

    def submit(self, campaign, run, manifest, *, dry_run: bool) -> str:
        backend = run["backend"]
        if str(manifest.get("image_id")) != str(run.get("image_id")):
            raise ValueError("SenseCore frozen manifest image_id conflicts with the run")
        resource_name = scheduler_job_name(
            str(backend["job_name"]), str(manifest["attempt_id"])
        )
        create = self._create_command(manifest)
        if dry_run:
            return "DRY_RUN"
        if self.find(run, resource_name):
            raise FileExistsError(f"SenseCore job already exists: {resource_name}")
        result = self.s.run_command(create, check=False)
        if result.returncode != 0:
            raise RuntimeError(self._redact_error(result.stderr) or "SenseCore create failed")
        summary = self.describe(run, resource_name)
        if summary.get("name") != resource_name:
            raise RuntimeError("SenseCore accepted create but exact job was not observable")
        return resource_name

    def status(self, campaign, run) -> dict[str, Any]:
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

    def cancel(self, campaign, run) -> dict[str, Any]:
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

    def collect(self, campaign, run) -> dict[str, Any]:
        snapshot = self.logs(campaign, run, tail=200)
        lines = snapshot["lines"]
        metrics = [metric for line in lines if (metric := self.s.parse_metric(campaign, line))]
        checkpoints = [
            checkpoint for line in lines
            if (checkpoint := self.s.parse_checkpoint(campaign, line))
        ]
        metric_lines = [line for line in lines if "Step " in line or "gPPL:" in line or ("plan" in line.lower() and "ppl" in line.lower())]
        result = {"run_id": run["run_id"], "backend": "sensecore", "model_observed": bool(metrics),
                  "latest_metric": metrics[-1] if metrics else None, "metric_log_lines": metric_lines[-20:],
                  "live_logs_expired": snapshot["expired"]}
        if checkpoints:
            latest = max(checkpoints, key=lambda item: int(item["step"]))
            result["latest_completed_checkpoint"] = latest["path"]
            result["latest_completed_checkpoint_step"] = latest["step"]
        return result

    def logs(self, campaign, run, *, tail: int) -> dict[str, Any]:
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
