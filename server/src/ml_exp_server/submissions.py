"""First-class experiment submission lifecycle over durable Actions."""

from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .application import ApplicationError, ExperimentServerApplication, evidence_digest
from .runtime import ExperimentServerRuntime
from .schemas import OperationScope, OperationScopeType


_SUBMISSION_OPERATIONS = {"SUBMIT_RUN", "RETRY_ATTEMPT", "RUN_EVALUATION"}
_ACTIVE_EXECUTION_STATES = {
    "PREPARED", "AUTHORIZED", "EXECUTING", "RECONCILE_REQUIRED", "VERIFIED",
}


def _reusable(view: dict[str, Any]) -> bool:
    if view.get("status") not in _ACTIVE_EXECUTION_STATES:
        return False
    if view.get("status") not in {"PREPARED", "AUTHORIZED"}:
        return True
    value = view.get("gate_expires_at")
    if not isinstance(value, str):
        return False
    try:
        expires_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return datetime.now(timezone.utc) < expires_at


class ExperimentSubmissionService:
    """Expose submission as an idempotent, inspectable experiment resource."""

    def __init__(
        self, application: ExperimentServerApplication,
        runtime: ExperimentServerRuntime,
    ) -> None:
        self.application = application
        self.runtime = runtime
        self._prepare_lock = threading.RLock()

    @staticmethod
    def _view(action: dict[str, Any], *, reused: bool = False) -> dict[str, Any]:
        execution = action.get("execution")
        execution = execution if isinstance(execution, dict) else {}
        status = str(execution.get("status") or "UNKNOWN")
        action_id = str(action.get("action_id") or "")
        if status == "PREPARED":
            next_action = "AUTHORIZE"
        elif status == "AUTHORIZED":
            next_action = "EXECUTE"
        elif status in {"EXECUTING", "RECONCILE_REQUIRED"}:
            next_action = "RECONCILE"
        elif status == "BLOCKED":
            next_action = "REPREPARE"
        else:
            next_action = "NONE"
        return {
            "submission_id": action_id,
            "project": (action.get("scope") or {}).get("project"),
            "run_id": action.get("run_id"),
            "attempt_id": action.get("attempt_id"),
            "operation": action.get("operation"),
            "status": status,
            "ready": bool(action.get("ready")),
            "next_action": next_action,
            "reused": reused,
            "confirmation": f"EXECUTE {action_id}" if action_id else None,
            "gate_expires_at": action.get("gate_expires_at"),
            "intent_digest": action.get("intent_digest"),
            "gates": action.get("gates") or [],
            "preflight_summary": action.get("preflight_summary") or {},
            "execution": execution,
        }

    @staticmethod
    def _is_submission(action: dict[str, Any]) -> bool:
        return str(action.get("operation") or "") in _SUBMISSION_OPERATIONS

    def _snapshot(self, submission_id: str) -> dict[str, Any]:
        try:
            action = self.runtime.action_store.snapshot(submission_id)
        except (FileNotFoundError, ValueError) as exc:
            raise ApplicationError(
                "submission not found", status_code=404, code="UNKNOWN_SUBMISSION",
            ) from exc
        if not self._is_submission(action):
            raise ApplicationError(
                "action is not an experiment submission",
                status_code=404, code="UNKNOWN_SUBMISSION",
            )
        return action

    def get(self, submission_id: str) -> dict[str, Any]:
        return self._view(self._snapshot(submission_id))

    def list(self, project: str, run_id: str) -> dict[str, Any]:
        # Authored Runs are submit candidates before a run directory exists,
        # so listing submissions must not require RunIndex resolution.
        scope = OperationScope(
            project=project, scope_type=OperationScopeType.RUN, object_id=run_id,
        )
        actions = [
            action for action in self.runtime.action_store.list_for_scope(scope)
            if self._is_submission(action)
        ]
        return {
            "project": project,
            "run_id": run_id,
            "submissions": [self._view(action) for action in actions],
        }

    def prepare_first_attempt(
        self, project: str, run_id: str, *, max_gpu_hours: float,
        reason: str,
    ) -> dict[str, Any]:
        """Prepare a first-Attempt intent without authorizing or scheduling it."""
        with self._prepare_lock:
            existing = self.list(project, run_id)["submissions"]
            for view in reversed(existing):
                if not _reusable(view):
                    continue
                existing_budget = view["preflight_summary"].get("max_gpu_hours")
                if existing_budget is not None and float(existing_budget) != max_gpu_hours:
                    raise ApplicationError(
                        "an active submission exists with a different GPU-hour budget",
                        code="SUBMISSION_INTENT_EXISTS",
                    )
                view["reused"] = True
                return view
            try:
                configured = self.runtime.project(project)
            except KeyError as exc:
                raise ApplicationError(
                    f"unknown project: {project}", status_code=404,
                    code="UNKNOWN_PROJECT",
                ) from exc
            if configured.controller is None:
                raise ApplicationError(
                    "project has no controller configuration",
                    code="CONTROLLER_UNAVAILABLE",
                )
            materializers = []
            for reference in configured.campaigns:
                revision = reference.current_revision
                if revision is None:
                    continue
                if any(
                    membership.run_id == run_id and membership.kind == "materialize"
                    for membership in revision.memberships
                ):
                    materializers.append(revision)
            if len(materializers) != 1:
                raise ApplicationError(
                    "Run must have exactly one authored materializing Campaign",
                    code="RUN_NOT_MATERIALIZABLE",
                )
            revision = materializers[0]
            campaign = Path(revision.file)
            if not campaign.is_absolute():
                campaign = Path(configured.base_dir or ".") / campaign
            campaign = campaign.resolve()
            if not campaign.is_file():
                raise ApplicationError(
                    "authored Campaign file is unavailable",
                    code="CAMPAIGN_FILE_MISSING",
                )
            row = self.runtime.index.get_run(project, run_id)
            if row is not None:
                state = str(row.scheduler_state or "NOT_SUBMITTED").upper()
                if state != "NOT_SUBMITTED":
                    raise ApplicationError(
                        f"Run state {state} is not eligible for first submission",
                        code="RUN_NOT_SUBMITTABLE",
                    )
                if any(attempt.has_submission for attempt in row.attempts):
                    raise ApplicationError(
                        "Run already has submitted Attempt evidence",
                        code="RUN_ALREADY_SUBMITTED",
                    )
                existing_attempts = [attempt.attempt_id for attempt in row.attempts]
            else:
                existing_attempts = []
            attempt_id = existing_attempts[-1] if existing_attempts else "attempt-001"
            scope = OperationScope(
                project=project, scope_type=OperationScopeType.RUN, object_id=run_id,
            )
            draft = yaml.safe_dump({
                "campaign_file": str(campaign),
                "run_id": run_id,
                "attempt_id": attempt_id,
                "max_gpu_hours": max_gpu_hours,
            }, sort_keys=False)
            digest = evidence_digest({
                "project": project,
                "run_id": run_id,
                "campaign_revision": revision.model_dump(mode="json"),
                "indexed_run": row.model_dump(mode="json") if row is not None else None,
            })
            action = self.runtime.action_service.prepare(scope, configured, {
                "kind": "SUBMIT_RUN",
                "title": f"Launch {run_id} as {attempt_id}",
                "target": f"run://{project}/{run_id}",
                "change_summary": reason or "submit the authored Run's first Attempt",
                "resource_estimate": f"up to {max_gpu_hours:g} GPU-hours",
                "rationale": "materialize one authored Campaign membership",
                "risk": "scheduler mutation requires separate authorization and execution",
                "draft": draft,
                "evidence_digest": digest,
                "idempotency_key": "submission-" + hashlib.sha256(
                    f"{project}:{run_id}:{attempt_id}:{len(existing) + 1}".encode()
                ).hexdigest()[:20],
            })
            return self._view(action)

    def authorize(self, submission_id: str, note: str) -> dict[str, Any]:
        self._snapshot(submission_id)
        return self._view(self.application.authorize_action(submission_id, note))

    def execute(self, submission_id: str, confirmation: str) -> dict[str, Any]:
        self._snapshot(submission_id)
        return self._view(
            self.application.execute_action(submission_id, confirmation),
        )

    def reconcile(self, submission_id: str) -> dict[str, Any]:
        self._snapshot(submission_id)
        return self._view(self.application.reconcile_action(submission_id))
