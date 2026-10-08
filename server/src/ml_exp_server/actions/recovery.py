"""Recovery for ActionService."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from ..controller_gateway import redact as _redact
from ..storage import utc_now
from .errors import ActionError
from .files import file_sha as _file_sha
from .policy import ActionExecutionPolicy

class ActionRecovery:
    """Recovery operations; state belongs to the application facade."""

    def reconcile(self, action_id: str) -> dict[str, Any]:
        """Observe an uncertain submission without ever issuing submit again."""
        snapshot = self.store.snapshot(action_id)
        execution = snapshot["execution"]
        if execution.get("status") == "VERIFIED":
            return snapshot
        if execution.get("status") == "EXECUTING":
            # In a live daemon, EXECUTING is owned by the background worker
            # that durably claimed it.  Reconciliation must not race that
            # worker.  Startup recovery converts abandoned claims to
            # RECONCILE_REQUIRED before the API begins serving requests.
            return snapshot
        if snapshot.get("operation") == "REBUILD_LOCAL_EVIDENCE":
            if not self.config.allow_local_evidence_rebuild:
                raise ActionError("local evidence rebuild Actions are disabled by daemon policy")
            if execution.get("status") not in {"EXECUTING", "RECONCILE_REQUIRED"}:
                raise ActionError("local evidence rebuild is not awaiting reconciliation")
            collection_path = Path(str(snapshot.get("collection_path") or ""))
            actual_digest = _file_sha(collection_path)
            expected_new = snapshot.get("expected_new_collection_sha256")
            reviewed_old = snapshot.get("expected_collection_sha256")
            previous = execution.get("result")
            result = dict(previous) if isinstance(previous, dict) else {}
            result.update({
                "observed_collection_sha256": actual_digest,
                "expected_new_collection_sha256": expected_new,
                "reviewed_old_collection_sha256": reviewed_old,
                "reconciled_read_only": True,
            })
            now = utc_now()
            if actual_digest == expected_new:
                execution.update({
                    "status": "VERIFIED", "finished_at": now,
                    "last_reconciled_at": now, "error": None,
                    "result": result,
                    "resolution": "APPLIED",
                    "safe_to_retry": False,
                    "next_action": "OBSERVE_RESULT",
                })
                return self.store.set_execution(
                    action_id, execution,
                    event="local_evidence_rebuild_reconciled",
                )
            if (
                actual_digest == reviewed_old
                and snapshot.get("atomic_collection_replace") is True
                and snapshot.get("write_protocol") == "dirfd-fsync-rename-v1"
            ):
                execution.update({
                    "status": "FAILED", "finished_at": now,
                    "last_reconciled_at": now,
                    "error": (
                        "collection remains at the reviewed old digest; "
                        "the atomic local evidence write did not execute"
                    ),
                    "result": result,
                    "resolution": "NOT_APPLIED_SAFE_TO_RETRY",
                    "safe_to_retry": True,
                    "next_action": "PREPARE_NEW_ACTION",
                })
                return self.store.set_execution(
                    action_id, execution,
                    event="local_evidence_rebuild_not_executed",
                )
            execution.update({
                "status": "RECONCILE_REQUIRED", "last_reconciled_at": now,
                "error": (
                    "collection digest is neither the reviewed old digest nor "
                    "the frozen expected new digest; manual investigation is required"
                ),
                "result": result,
                "resolution": "UNKNOWN_DO_NOT_RETRY",
                "safe_to_retry": False,
                "next_action": "RECONCILE",
            })
            return self.store.set_execution(
                action_id, execution,
                event="local_evidence_rebuild_reconcile_blocked",
            )
        if snapshot.get("operation") in ActionExecutionPolicy.PROJECT_WRITE_OPERATIONS:
            if not self.config.allow_project_writes:
                raise ActionError("project writes are disabled by daemon policy")
            if execution.get("status") not in {"EXECUTING", "RECONCILE_REQUIRED"}:
                raise ActionError("project write is not awaiting reconciliation")
            return self._execute_write(snapshot, execution)
        if snapshot.get("operation") == "CANCEL_RUN":
            if execution.get("status") not in {"EXECUTING", "RECONCILE_REQUIRED"}:
                raise ActionError("cancellation is not awaiting reconciliation")
            command = [str(item) for item in snapshot.get("verification_command_preview") or []]
            if not command:
                raise ActionError("cancellation plan has no verification command")
            verification_result = self.controller.execute_command(
                command, cwd=Path(snapshot["verification_cwd"]),
                timeout=self.config.timeout_seconds,
            )
            payload = verification_result.get("payload")
            observed = payload[0] if isinstance(payload, list) and payload else {}
            exact = str(observed.get("backend_job_id") or "") == str(
                snapshot.get("backend_job_id") or ""
            )
            state = str(observed.get("state") or "").upper()
            terminal = state in {
                "CANCELLED", "CANCELED", "COMPLETED", "FAILED", "SUCCEEDED",
            }
            verified = exact and terminal
            execution.update({
                "status": "VERIFIED" if verified else "RECONCILE_REQUIRED",
                "last_reconciled_at": utc_now(),
                "finished_at": utc_now() if verified else execution.get("finished_at"),
                "error": None if verified else "cancellation is not yet terminal",
                "result": {"observation": _redact(observed)},
                "resolution": "APPLIED" if verified else "UNKNOWN_DO_NOT_RETRY",
                "safe_to_retry": False,
                "next_action": "OBSERVE_RESULT" if verified else "RECONCILE",
            })
            return self.store.set_execution(
                action_id, execution,
                event="cancellation_verified" if verified else "cancellation_reconcile_pending",
            )
        if snapshot.get("operation") not in {
            "SUBMIT_RUN", "RETRY_ATTEMPT", "RUN_EVALUATION",
        }:
            raise ActionError("only submission actions support reconciliation")
        if execution.get("status") not in {"EXECUTING", "RECONCILE_REQUIRED"}:
            raise ActionError("submission is not awaiting reconciliation")
        previous = execution.get("result")
        previous = previous if isinstance(previous, dict) else {}
        submitted = previous.get("submission")
        expected_job_id = (
            str(submitted.get("backend_job_id") or "")
            if isinstance(submitted, dict) else ""
        ) or None
        verified, observed, verification_result, detail = self._verify_submission(
            snapshot, expected_job_id=expected_job_id,
        )
        result = {
            **previous,
            "observation": _redact(observed),
            "verification_command": _redact(verification_result),
        }
        if verified:
            execution.update({
                "status": "VERIFIED", "finished_at": utc_now(),
                "last_reconciled_at": utc_now(), "error": None, "result": result,
                "resolution": "SUBMITTED",
                "safe_to_retry": False,
                "next_action": "MONITOR_RUN",
            })
            return self.store.set_execution(
                action_id, execution, event="submission_reconciled",
            )
        observed_state = (
            str(observed.get("state") or "").upper()
            if isinstance(observed, dict) else ""
        )
        confirmed_absent = expected_job_id is None and observed_state == "NOT_SUBMITTED"
        execution.update({
            "status": "FAILED" if confirmed_absent else "RECONCILE_REQUIRED",
            "finished_at": utc_now() if confirmed_absent else execution.get("finished_at"),
            "last_reconciled_at": utc_now(),
            "error": (
                "scheduler confirms that submission did not occur"
                if confirmed_absent else detail
            ),
            "result": result,
            "resolution": (
                "NOT_SUBMITTED_SAFE_TO_RETRY"
                if confirmed_absent else "UNKNOWN_DO_NOT_RETRY"
            ),
            "safe_to_retry": confirmed_absent,
            "next_action": "PREPARE_NEW_ACTION" if confirmed_absent else "RECONCILE",
        })
        return self.store.set_execution(
            action_id, execution, event="submission_reconcile_pending",
        )

    def recover_pending_project_writes(self) -> list[dict[str, Any]]:
        """Roll forward authorized write transactions after daemon restart."""

        pending = [
            action for action in self.store.list_all()
            if (
                action.get("operation") in ActionExecutionPolicy.PROJECT_WRITE_OPERATIONS
                and action.get("execution", {}).get("status")
                in {"EXECUTING", "RECONCILE_REQUIRED"}
            )
        ]
        if not self.config.allow_project_writes:
            return [{
                **action,
                "execution": {
                    **action["execution"],
                    "error": "; ".join(filter(None, (
                        str(action["execution"].get("error") or ""),
                        "startup recovery is blocked because project writes are disabled",
                    ))),
                },
            } for action in pending]
        recovered: list[dict[str, Any]] = []
        for action in pending:
            recovered.append(self._execute_write(
                action, dict(action["execution"]),
            ))
        return recovered

    def recover_interrupted_executions(self) -> list[dict[str, Any]]:
        """Mark claims abandoned by a daemon restart as reconciliation-only."""
        recovered: list[dict[str, Any]] = []
        for action in self.store.list_all():
            execution = action.get("execution")
            execution = dict(execution) if isinstance(execution, dict) else {}
            if execution.get("status") != "EXECUTING":
                continue
            execution.update({
                "status": "RECONCILE_REQUIRED",
                "error": (
                    execution.get("error")
                    or "daemon restarted after execution was claimed; observe effects before retry"
                ),
                "resolution": "UNKNOWN_DO_NOT_RETRY",
                "safe_to_retry": False,
                "next_action": "RECONCILE",
            })
            recovered.append(self.store.set_execution(
                str(action["action_id"]), execution,
                event="execution_interrupted_by_restart",
            ))
        return recovered
