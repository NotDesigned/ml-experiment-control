"""Execution for ActionService."""
from __future__ import annotations
import json
import hashlib
import os
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from ..controller_gateway import (
    redact as _redact, _safe_log_tail, _open_regular_nofollow,
    _STDOUT_LOG_BYTES, _STDERR_LOG_BYTES,
)
from ..storage import utc_now
from .errors import ActionError
from .files import file_sha as _file_sha
from .project_writes import ProjectWriteConflict, ProjectWriteError
from .helpers import PendingActionExecution, _SHA256_DIGEST

_STAGE_PHASES = {
    "IMAGE_STAGE", "LOCK_WAIT", "CACHE_VERIFY", "IMAGE_PULL_CONVERT",
    "SIF_VERIFY", "SIF_PUBLISH",
}
_STAGE_ERRORS = {"LOCK_TIMEOUT", "STAGE_COMMAND_FAILED", "STAGE_TIMEOUT"}
_STAGE_EVENT_PREFIX = "ML_EXPD_STAGE_EVENT="
_STAGE_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def _stage_event(value: Any, *, started_at: str | None) -> dict[str, Any] | None:
    """Accept only the fixed, credential-free image preparation event schema."""
    if not isinstance(value, dict):
        return None
    phase, status, timestamp = (value.get(key) for key in ("phase", "status", "timestamp"))
    if (
        not isinstance(phase, str) or phase not in _STAGE_PHASES
        or not isinstance(status, str) or status not in {"START", "READY", "FAILED"}
        or not isinstance(timestamp, str) or not _STAGE_TIMESTAMP.fullmatch(timestamp)
    ):
        return None
    try:
        observed_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if started_at and observed_at < datetime.fromisoformat(started_at.replace("Z", "+00:00")).replace(microsecond=0):
            return None
    except (TypeError, ValueError):
        return None
    event: dict[str, Any] = {"phase": phase, "status": status, "timestamp": timestamp}
    exit_code = value.get("exit_code")
    if type(exit_code) is int and 0 <= exit_code <= 255:
        event["exit_code"] = exit_code
    error_code = value.get("error_code")
    if isinstance(error_code, str) and error_code in _STAGE_ERRORS:
        event["error_code"] = error_code
    return event


def _stage_diagnostic(result: dict[str, Any], *, started_at: str | None) -> dict[str, Any]:
    events = []
    for stream in ("stdout", "stderr"):
        for line in str(result.get(stream) or "").splitlines():
            if not line.startswith(_STAGE_EVENT_PREFIX):
                continue
            try:
                value = json.loads(line[len(_STAGE_EVENT_PREFIX):])
            except json.JSONDecodeError:
                continue
            event = _stage_event(value, started_at=started_at)
            if event is not None:
                events.append(event)
    events.sort(key=lambda item: item["timestamp"])
    last_event = events[-1] if events else None
    return {
        "phase": last_event["phase"] if last_event else "UNKNOWN",
        "last_event": last_event,
        "events": events[-64:],
        "controller_exit_code": result.get("returncode"),
        "controller_timeout": bool(result.get("timeout")),
        "scheduler_submitted": False,
        "remote_preparation_may_continue": bool(result.get("timeout")),
        "retry_scope": "NEW_ACTION_BEFORE_SCHEDULER_SUBMISSION",
        "next_check": "Inspect backend image preparation state and logs before preparing a new Action",
    }


class ActionExecution:
    """Execution operations; state belongs to the application facade."""

    def diagnostics(self, action_id: str, *, refresh: bool = False, runtime=None) -> dict[str, Any]:
        """Read saved staging evidence; optionally observe a managed WYD receipt."""
        snapshot = self.store.snapshot(action_id)
        execution = snapshot["execution"]
        result = execution.get("result") or {}
        saved = result.get("stage_command") if isinstance(result, dict) else None
        saved = saved if isinstance(saved, dict) else {}
        command = [str(item) for item in snapshot.get("stage_command_preview") or []]
        stage = {
            "returncode": saved.get("returncode"), "timeout": bool(saved.get("timeout")),
            "stdout": _safe_log_tail(saved.get("stdout"), limit=_STDOUT_LOG_BYTES, command=command),
            "stderr": _safe_log_tail(saved.get("stderr"), limit=_STDERR_LOG_BYTES, command=command),
        }
        diagnostic = _stage_diagnostic(stage, started_at=execution.get("started_at"))
        submitted = {"SUBMITTED": True, "FAILED_BEFORE_SUBMISSION": False}.get(execution.get("resolution"))
        retry = submitted is False and execution.get("safe_to_retry") is True and execution.get("next_action") == "PREPARE_NEW_ACTION"
        diagnostic.update({
            "scheduler_submitted": submitted,
            "retry_scope": "NEW_ACTION_BEFORE_SCHEDULER_SUBMISSION" if retry else None,
            "next_action": "PREPARE_NEW_ACTION" if retry else "MONITOR_RUN" if submitted is True else "OBSERVE_ACTION",
            "next_check": "Inspect backend image preparation state and logs before preparing a new Action" if retry else
                          "Monitor the exact Run/Attempt; do not resubmit this Action" if submitted is True else
                          "Consult the saved Action state and reconcile any uncertain scheduler submission before retrying",
        })
        return {
            "contract": "action-stage-diagnostics.v1", "action_id": action_id,
            "status": execution.get("status"), "started_at": execution.get("started_at"),
            "finished_at": execution.get("finished_at"), "stage_command": stage,
            "stage_diagnostic": diagnostic,
            "remote": self._remote_stage_diagnostic(snapshot, runtime) if refresh else {"status": "NOT_REQUESTED"},
            "historical_execution_unchanged": True,
        }

    def _remote_stage_diagnostic(self, snapshot: dict[str, Any], runtime) -> dict[str, Any]:
        from ..container_execution import ContainerExecutionService
        from ..container_controller import Controller
        import yaml

        try:
            if not snapshot["execution"].get("started_at"):
                return {"status": "NOT_STARTED", "reason": "The Action has no preparation start time"}
            service = ContainerExecutionService(runtime)
            project_name = snapshot["scope"]["project"]
            project = runtime.project(project_name)
            root = service.runtime.config.project_registry_root_path().resolve()
            managed = root / "managed" / project_name
            if (
                project.base_dir is None or project.base_dir.resolve() != managed
                or project.controller is None
                or not project.controller.capabilities.get("container_execution")
                or project.controller.experimentctl != "tools/experimentctl.py"
            ):
                return {"status": "UNSUPPORTED", "reason": "Only the built-in managed WYD controller is queried"}
            directory = self.store.directory(snapshot["action_id"])
            path = directory / "campaign.execution.yml"
            if (directory.is_symlink() or path.is_symlink()
                    or path.resolve().parent != directory.resolve()
                    or Path(snapshot.get("execution_campaign_file") or "") != path):
                raise ValueError("invalid frozen campaign location")
            descriptor, _ = _open_regular_nofollow(path)
            with os.fdopen(descriptor, "rb") as stream:
                raw = stream.read()
            if "sha256:" + hashlib.sha256(raw).hexdigest() != snapshot.get("execution_campaign_sha256"):
                raise ValueError("frozen campaign digest differs")
            campaign = yaml.safe_load(raw)
            run_id, expected_source_id = snapshot.get("run_id"), snapshot.get("expected_source_id")
            if (not isinstance(campaign, dict) or campaign.get("project") != project_name
                    or campaign.get("source_store") != str(root)):
                raise ValueError("frozen campaign project binding differs")
            runs = [run for run in campaign.get("runs", []) if isinstance(run, dict) and run.get("run_id") == run_id]
            if len(runs) != 1:
                raise ValueError("frozen campaign Run binding differs")
            run = runs[0]
            source_id = run.get("source_id")
            if (not isinstance(source_id, str) or not re.fullmatch(r"source\.[0-9a-f]{64}", source_id)
                    or expected_source_id not in (None, "") and expected_source_id != source_id):
                raise ValueError("frozen campaign source binding differs")
            if run.get("backend", {}).get("kind") != "slurm" or not run["backend"].get("oci_image"):
                return {"status": "UNSUPPORTED", "reason": "The Action does not prepare a managed WYD OCI image"}
            # Embedded OCI code has no separate source staging arguments. Its
            # provenance still comes from the exact frozen Run and Runtime.
            extra = (["--source-root", str(root / "source-revisions" / "sources" / project_name / source_id / "tree"), "--source-id", source_id]
                     if expected_source_id else [])
            if snapshot.get("campaign_revision"):
                extra.extend(["--campaign-id", snapshot["campaign_revision"]])
            call = self.controller.build(project, path, "stage", run_id, attempt_id=snapshot.get("attempt_id"), extra=extra)
            if call.argv != snapshot.get("stage_command_preview") or str(call.cwd) != snapshot.get("stage_cwd"):
                raise ValueError("frozen stage command binding differs")
            controller = Controller(campaign, run_id, snapshot.get("attempt_id") or "attempt-001")
            observed = controller.backend.stage_progress(controller.run)
            event = _stage_event(observed, started_at=snapshot["execution"].get("started_at"))
            if event is None:
                return {"status": "UNAVAILABLE", "reason": "No matching current preparation receipt was observed"}
            finished = snapshot["execution"].get("finished_at")
            after_action = bool(finished and datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00")) > datetime.fromisoformat(finished.replace("Z", "+00:00")))
            ready = event["phase"] in {"CACHE_VERIFY", "SIF_PUBLISH"} and event["status"] == "READY"
            new_action = ready and snapshot["execution"].get("next_action") == "PREPARE_NEW_ACTION"
            return {
                "status": "OBSERVED", "event": event, "observed_at": utc_now(),
                "observed_after_action": after_action,
                "image_ready_reported": ready,
                "cache_reverified": False,
                "historical_failure_cause": False,
                "next_action": "PREPARE_NEW_ACTION" if new_action else "OBSERVE_PREPARATION",
            }
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
            return {"status": "UNAVAILABLE", "reason": "Managed preparation binding or observation could not be verified", "error_class": type(error).__name__}

    def authorize(self, action_id: str, note: str) -> dict[str, Any]:
        snapshot = self.store.snapshot(action_id)
        execution = snapshot["execution"]
        if not snapshot.get("ready") or execution.get("status") != "PREPARED":
            raise ActionError("only a ready PREPARED action can be execution-authorized")
        expires_at = datetime.fromisoformat(
            str(snapshot["gate_expires_at"]).replace("Z", "+00:00")
        )
        if datetime.now(timezone.utc) >= expires_at:
            raise ActionError("gate bundle has expired; prepare a fresh intent")
        actor = self.actor_provider().strip()
        if not actor:
            raise ActionError("server could not establish a trusted actor identity")
        execution.update({
            "status": "AUTHORIZED", "authorized_at": utc_now(),
            "authorization_note": note, "authorization_actor": actor,
            "authorized_intent_digest": snapshot["intent_digest"],
            "authorized_gate_bundle_digest": snapshot["gate_bundle_digest"],
            "authorization_expires_at": snapshot["gate_expires_at"],
        })
        try:
            return self.store.set_execution(
                action_id, execution, event="execution_authorized",
                expected_status="PREPARED",
            )
        except RuntimeError as exc:
            raise ActionError(str(exc)) from exc

    def begin_execute(
        self, action_id: str, confirmation: str,
    ) -> tuple[dict[str, Any], PendingActionExecution | None]:
        """Claim an authorized Action without waiting for its slow executor.

        The claim is committed before this method returns.  A caller may safely
        return the EXECUTING snapshot to an HTTP client and run ``finish_execute``
        in a daemon-owned background task.  VERIFIED Actions remain idempotent.
        """
        snapshot = self.store.snapshot(action_id)
        execution = snapshot["execution"]
        if execution.get("status") == "VERIFIED":
            return snapshot, None
        dispatch = self.execution_policy.validate(snapshot, confirmation)
        if dispatch.local_evidence_rebuild:
            collection_path = Path(str(snapshot.get("collection_path") or ""))
            if _file_sha(collection_path) != snapshot.get("expected_collection_sha256"):
                raise ActionError(
                    "collection changed after local evidence Action preparation; "
                    "prepare a fresh action"
                )
            try:
                self.controller.verify_execution_bundle(
                    snapshot.get("controller_snapshot") or {},
                )
            except (OSError, ValueError) as exc:
                raise ActionError(
                    f"approved private controller snapshot is invalid: {exc}"
                ) from exc
        execution.update({
            "status": "EXECUTING",
            "started_at": utc_now(),
            "error": None,
            "resolution": "STILL_IN_PROGRESS",
            "safe_to_retry": False,
            "next_action": "WAIT",
        })
        try:
            started = self.store.begin_execution(
                action_id, execution, intent_digest=str(snapshot["intent_digest"]),
            )
        except RuntimeError as exc:
            raise ActionError(str(exc)) from exc
        pending = PendingActionExecution(
            plan=snapshot,
            # The HTTP response and worker must not share a mutable execution
            # mapping; a fast worker could otherwise rewrite EXECUTING to
            # VERIFIED before response serialization.
            execution=deepcopy(started["execution"]),
            dispatch=dispatch,
        )
        return started, pending

    def finish_execute(self, pending: PendingActionExecution) -> dict[str, Any]:
        """Complete one previously claimed Action."""
        snapshot = pending.plan
        execution = pending.execution
        dispatch = pending.dispatch
        if dispatch.project_write:
            return self._execute_write(snapshot, execution)
        if dispatch.local_evidence_rebuild:
            return self._execute_local_evidence_rebuild(snapshot, execution)
        return self._execute_controller(snapshot, execution)

    def execute(self, action_id: str, confirmation: str) -> dict[str, Any]:
        """Synchronous compatibility wrapper used outside the HTTP adapter."""
        started, pending = self.begin_execute(action_id, confirmation)
        return started if pending is None else self.finish_execute(pending)

    def _execute_local_evidence_rebuild(
        self, plan: dict[str, Any], execution: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute and verify the one allowlisted pure-local controller verb."""
        arguments = [str(item) for item in plan.get("snapshot_arguments") or []]
        if "refresh-evidence-local" not in arguments:
            raise ActionError("local evidence Action has no allowlisted controller verb")
        try:
            result = self.controller.execute_snapshot(
                plan.get("controller_snapshot") or {}, arguments,
                timeout=self.config.timeout_seconds,
            )
        except (OSError, ValueError) as exc:
            result = {
                "returncode": 1, "timeout": False, "payload": None,
                "stdout": "", "stderr": str(exc),
            }
        failure = None
        if result.get("timeout"):
            failure = "local evidence rebuild timed out"
        elif result.get("returncode") != 0:
            failure = str(
                result.get("stderr") or result.get("stdout")
                or "local evidence rebuild controller failed"
            )[:1000]
        payload = result.get("payload")
        record = (
            payload[0] if isinstance(payload, list) and len(payload) == 1
            and isinstance(payload[0], dict) else None
        )
        collection_path = Path(str(plan.get("collection_path") or ""))
        actual_digest = _file_sha(collection_path)
        if failure is None and record is None:
            failure = "local evidence rebuild did not return exactly one result"
        if failure is None and record is not None:
            exact_identity = (
                record.get("project") == plan["scope"]["project"]
                and record.get("run_id") == plan.get("run_id")
                and record.get("attempt_id") == plan.get("attempt_id")
            )
            exact_result = (
                exact_identity
                and record.get("input_digest") == plan.get("input_digest")
                and record.get("old_digest") == plan.get("expected_collection_sha256")
                and record.get("new_digest")
                == plan.get("expected_new_collection_sha256")
                and record.get("expected_new_collection_digest")
                == plan.get("expected_new_collection_sha256")
                and actual_digest == plan.get("expected_new_collection_sha256")
                and str(Path(str(record.get("collection_path") or "")).resolve())
                == str(collection_path.resolve())
                and record.get("local_only") is True
                and record.get("backend_accessed") is False
                and record.get("scheduler_accessed") is False
                and record.get("atomic_collection_replace") is True
                and record.get("write_protocol") == "dirfd-fsync-rename-v1"
                and record.get("controller_snapshot_sha256")
                == (plan.get("controller_snapshot") or {}).get("manifest_sha256")
                and result.get("controller_snapshot_sha256")
                == (plan.get("controller_snapshot") or {}).get("manifest_sha256")
                and _SHA256_DIGEST.fullmatch(str(record.get("new_digest") or ""))
                is not None
            )
            if not exact_result:
                failure = "local evidence rebuild result failed exact identity verification"
        if failure is not None:
            execution.update({
                "status": "RECONCILE_REQUIRED", "finished_at": utc_now(),
                "error": failure,
                "result": {
                    "controller": _redact(result),
                    "observed_collection_sha256": actual_digest,
                    "expected_new_collection_sha256": plan.get(
                        "expected_new_collection_sha256"
                    ),
                },
                "resolution": "UNKNOWN_DO_NOT_RETRY",
                "safe_to_retry": False,
                "next_action": "RECONCILE",
            })
            return self.store.set_execution(
                plan["action_id"], execution,
                event="local_evidence_rebuild_reconcile_required",
            )
        execution.update({
            "status": "VERIFIED", "finished_at": utc_now(), "error": None,
            "result": _redact(record),
            "resolution": "APPLIED",
            "safe_to_retry": False,
            "next_action": "OBSERVE_RESULT",
        })
        return self.store.set_execution(
            plan["action_id"], execution,
            event="local_evidence_rebuild_verified",
        )

    def _execute_write(self, plan: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
        try:
            result = self.project_write_transaction.apply(plan)
            if not plan.get("files"):
                only_file = result["files"][0]
                result = {
                    "target_path": only_file["path"],
                    "sha256": only_file["sha256"],
                }
        except ProjectWriteError as exc:
            failed_cleanly = isinstance(exc, ProjectWriteConflict) and not exc.partial
            execution.update({
                "status": "FAILED" if failed_cleanly else "RECONCILE_REQUIRED",
                "finished_at": utc_now(),
                "last_reconciled_at": utc_now(),
                "error": str(exc),
                "resolution": (
                    "FAILED_BEFORE_EFFECT"
                    if failed_cleanly else "UNKNOWN_DO_NOT_RETRY"
                ),
                "safe_to_retry": failed_cleanly,
                "next_action": (
                    "PREPARE_NEW_ACTION" if failed_cleanly else "RECONCILE"
                ),
            })
            return self.store.set_execution(
                plan["action_id"], execution,
                event=(
                    "project_write_failed"
                    if execution["status"] == "FAILED"
                    else "project_write_reconcile_required"
                ),
            )
        except Exception as exc:
            # The transaction may already be APPLIED when operation-specific
            # validation or durable transaction metadata fails. Preserve the
            # exact intent for restart reconciliation instead of reporting a
            # clean failure after a potentially visible effect.
            execution.update({
                "status": "RECONCILE_REQUIRED",
                "finished_at": utc_now(),
                "last_reconciled_at": utc_now(),
                "error": f"{type(exc).__name__}: {exc}"[:1000],
                "resolution": "UNKNOWN_DO_NOT_RETRY",
                "safe_to_retry": False,
                "next_action": "RECONCILE",
            })
            return self.store.set_execution(
                plan["action_id"], execution,
                event="project_write_reconcile_required",
            )
        execution.update({
            "status": "VERIFIED", "finished_at": utc_now(),
            "last_reconciled_at": execution.get("last_reconciled_at"),
            "error": None,
            "result": result,
            "resolution": "APPLIED",
            "safe_to_retry": False,
            "next_action": "OBSERVE_RESULT",
        })
        return self.store.set_execution(
            plan["action_id"], execution, event="project_write_verified",
        )

    @staticmethod
    def _single_status_record(payload: Any) -> dict[str, Any] | None:
        if isinstance(payload, dict):
            return payload
        if (
            isinstance(payload, list)
            and len(payload) == 1
            and isinstance(payload[0], dict)
        ):
            return payload[0]
        return None

    def _verify_submission(
        self, plan: dict[str, Any], *, expected_job_id: str | None,
    ) -> tuple[bool, dict[str, Any] | None, dict[str, Any], str]:
        command = plan.get("verification_command_preview")
        cwd = plan.get("verification_cwd") or plan.get("cwd")
        if not isinstance(command, list) or not command or not cwd:
            return False, None, {}, "submission plan has no immutable status verification command"
        result = self.controller.execute_command(
            [str(item) for item in command],
            cwd=Path(str(cwd)), timeout=self.config.timeout_seconds,
        )
        if result.get("timeout"):
            return False, None, result, "exact Attempt status verification timed out"
        if result.get("returncode") != 0:
            detail = str(result.get("stderr") or "status controller failed")[:500]
            return False, None, result, f"exact Attempt status verification failed: {detail}"
        observed = self._single_status_record(result.get("payload"))
        if observed is None:
            return False, None, result, "status did not return exactly one Attempt record"
        job_id = str(observed.get("backend_job_id") or "")
        if not job_id:
            return False, observed, result, "status did not expose a backend_job_id"
        if expected_job_id and job_id != expected_job_id:
            return (
                False, observed, result,
                f"status backend_job_id {job_id!r} does not match submit result {expected_job_id!r}",
            )
        for key in ("run_id", "attempt_id"):
            expected = str(plan.get(key) or "")
            actual = str(observed.get(key) or "")
            if actual and expected and actual != expected:
                return (
                    False, observed, result,
                    f"status {key} {actual!r} does not match submission {expected!r}",
                )
        return True, observed, result, "exact Attempt is visible in backend status"

    def _submission_result(
        self, plan: dict[str, Any], execution: dict[str, Any],
        submit_result: dict[str, Any], *, submit_error: str | None = None,
        stage_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._execution_progress(plan, "VERIFYING_SUBMISSION", "Verifying the exact Attempt and scheduler job identity")
        submitted = self._single_status_record(submit_result.get("payload"))
        expected_job_id = str((submitted or {}).get("backend_job_id") or "") or None
        verified, observed, verification_result, detail = self._verify_submission(
            plan, expected_job_id=expected_job_id,
        )
        result = {
            "stage_command": _redact(stage_result),
            "submission": _redact(submitted),
            "submission_command": _redact(submit_result),
            "observation": _redact(observed),
            "verification_command": _redact(verification_result),
        }
        if verified:
            execution.update({
                "status": "VERIFIED", "finished_at": utc_now(),
                "error": None, "result": result,
                "resolution": "SUBMITTED",
                "safe_to_retry": False,
                "next_action": "MONITOR_RUN",
            })
            return self.store.set_execution(
                plan["action_id"], execution, event="submission_verified",
            )
        error = submit_error or (
            "submit returned without a unique backend_job_id"
            if expected_job_id is None else detail
        )
        if submit_error:
            error = f"{submit_error}; {detail}"
        execution.update({
            "status": "RECONCILE_REQUIRED", "finished_at": utc_now(),
            "error": error, "result": result,
            "resolution": "UNKNOWN_DO_NOT_RETRY",
            "safe_to_retry": False,
            "next_action": "RECONCILE",
        })
        return self.store.set_execution(
            plan["action_id"], execution, event="execution_reconcile_required",
        )

    def _execute_controller(self, plan: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
        self._execution_progress(plan, "VALIDATING_SOURCE", "Checking the approved immutable source and execution command")
        command = [str(item) for item in plan["command_preview"]]
        is_submission = plan["operation"] in {
            "SUBMIT_RUN", "RETRY_ATTEMPT", "RUN_EVALUATION",
        }
        stage_command = [
            str(item) for item in plan.get("stage_command_preview") or []
        ]
        expected_source_id = str(plan.get("expected_source_id") or "")
        if expected_source_id.startswith("source."):
            try:
                if self.source_resolver is None:
                    raise ValueError("source resolver is unavailable")
                source_root = self.source_resolver(
                    str(plan["scope"]["project"]), expected_source_id,
                )
                for approved_command in (command, stage_command):
                    root_index = approved_command.index("--source-root") + 1
                    source_index = approved_command.index("--source-id") + 1
                    if (
                        approved_command[root_index] != str(source_root)
                        or approved_command[source_index] != expected_source_id
                    ):
                        raise ValueError("approved command source binding changed")
            except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
                execution.update({
                    "status": "FAILED", "finished_at": utc_now(),
                    "error": f"imported source failed execution-time validation: {exc}",
                    "result": None,
                    "resolution": "FAILED_BEFORE_SUBMISSION",
                    "safe_to_retry": True,
                    "next_action": "PREPARE_NEW_ACTION",
                })
                return self.store.set_execution(
                    plan["action_id"], execution,
                    event="execution_source_validation_failed",
                )
        stage_result: dict[str, Any] | None = None
        if is_submission:
            stage_cwd = plan.get("stage_cwd")
            if not stage_command or not stage_cwd:
                execution.update({
                    "status": "FAILED", "finished_at": utc_now(),
                    "error": "submission plan has no immutable source staging command",
                    "result": None,
                    "resolution": "FAILED_BEFORE_SUBMISSION",
                    "safe_to_retry": True,
                    "next_action": "PREPARE_NEW_ACTION",
                })
                return self.store.set_execution(
                    plan["action_id"], execution, event="execution_stage_failed",
                )
            self._execution_progress(plan, "STAGING", "Preparing the backend image and source code: checking its cache, then converting or reusing the pinned image", timeout_seconds=self.config.stage_timeout_seconds or self.config.timeout_seconds)
            stage_result = self.controller.execute_command(
                stage_command,
                cwd=Path(str(stage_cwd)),
                timeout=self.config.stage_timeout_seconds or self.config.timeout_seconds,
            )
            if stage_result.get("timeout") or stage_result.get("returncode") != 0:
                diagnostic = _stage_diagnostic(stage_result, started_at=execution.get("started_at"))
                detail = str(
                    _redact(stage_result.get("stderr"))
                    or _redact(stage_result.get("stdout"))
                    or "backend image and source staging controller failed"
                )[-1000:]
                error = (
                    "backend image and source staging timed out before scheduler submission; "
                    "remote preparation may continue, inspect its state before preparing a new Action"
                    if stage_result.get("timeout")
                    else f"backend image and source staging failed before scheduler submission: {detail}"
                )
                execution.update({
                    "status": "FAILED", "finished_at": utc_now(),
                    "error": error,
                    "phase": diagnostic["phase"],
                    "result": {"stage_command": _redact(stage_result), "stage_diagnostic": diagnostic},
                    "resolution": "FAILED_BEFORE_SUBMISSION",
                    "safe_to_retry": True,
                    "next_action": "PREPARE_NEW_ACTION",
                })
                return self.store.set_execution(
                    plan["action_id"], execution, event="execution_stage_failed",
                )
        self._execution_progress(plan, "SCHEDULER_SUBMITTING", "Requesting the scheduler job; an uncertain result must be reconciled, never resubmitted", timeout_seconds=self.config.timeout_seconds)
        result = self.controller.execute_command(
            command, cwd=Path(plan["cwd"]), timeout=self.config.timeout_seconds,
        )
        if result.get("timeout"):
            if is_submission:
                return self._submission_result(
                    plan, execution, result,
                    submit_error="controller timed out after execution intent",
                    stage_result=stage_result,
                )
            execution.update({
                "status": "RECONCILE_REQUIRED", "finished_at": utc_now(),
                "error": "controller timed out after execution intent; inspect status before retry",
                "result": _redact(result),
                "resolution": "UNKNOWN_DO_NOT_RETRY",
                "safe_to_retry": False,
                "next_action": "RECONCILE",
            })
            return self.store.set_execution(
                plan["action_id"], execution, event="execution_reconcile_required",
            )
        if result.get("returncode") != 0:
            if is_submission:
                detail = str(result.get("stderr") or "controller failed")[:500]
                return self._submission_result(
                    plan, execution, result,
                    submit_error=f"submit controller failed after execution intent: {detail}",
                    stage_result=stage_result,
                )
            execution.update({
                "status": "FAILED", "finished_at": utc_now(),
                "error": str(result.get("stderr") or "controller failed")[:1000],
                "result": _redact(result),
                "resolution": "UNKNOWN_DO_NOT_RETRY",
                "safe_to_retry": False,
                "next_action": "RECONCILE",
            })
            return self.store.set_execution(plan["action_id"], execution, event="execution_failed")
        if is_submission:
            return self._submission_result(
                plan, execution, result, stage_result=stage_result,
            )
        payload = result.get("payload")
        execution.update({
            "status": "VERIFIED", "finished_at": utc_now(), "result": _redact(payload),
            "resolution": "APPLIED",
            "safe_to_retry": False,
            "next_action": "OBSERVE_RESULT",
        })
        return self.store.set_execution(plan["action_id"], execution, event="execution_verified")

    def _execution_progress(self, plan, phase, message, *, timeout_seconds=None):
        from ..runs.execution_progress import record_progress
        record_progress(self.store.directory(plan["action_id"]) / "progress.json", phase, message, timeout_seconds=timeout_seconds)
