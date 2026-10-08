"""Stable application facade composed from domain operations."""
from __future__ import annotations
from typing import Any
from .actions.helpers import PendingActionExecution
from .application_errors import ApplicationError
from .ingest.indexer import index_project
from .projects.project_service import ProjectApplicationService
from .projects.project_imports import ProjectImportService
from .projects.source_revisions import SourceRevisionService
from .runtime import ExperimentServerRuntime
from .schemas import OperationScopeType
from .telemetry import Telemetry

from .runs.context import RunContext
from .runs.evidence import RunEvidence
from .runs.validation import RunValidation
from .runs.queries import RunQueries
from .runs.run_operations import RunOperations
from .projects.lifecycle import ProjectLifecycle


class ExperimentServerApplication(RunContext, RunEvidence, RunValidation, RunQueries, RunOperations, ProjectLifecycle):
    """Compose domain operations without introducing additional mutable state."""

    def __init__(self, runtime: ExperimentServerRuntime):
        self.runtime = runtime
        self.telemetry = getattr(runtime, "telemetry", Telemetry())
        self.project_service = ProjectApplicationService(runtime)
        complete_config = callable(
            getattr(getattr(runtime, "config", None), "project_registry_root_path", None)
        )
        self.project_import_service = (
            ProjectImportService(runtime, self.project_service)
            if complete_config else None
        )
        self.source_revision_service = (
            SourceRevisionService(runtime)
            if complete_config else None
        )

    def list_actions(self, project: str, scope_type: OperationScopeType | str,
                     object_id: str) -> dict[str, Any]:
        scope, _, _ = self.resolve_scope(project, scope_type, object_id)
        return {"actions": self.runtime.action_store.list_for_scope(scope),
                "policy": self.runtime.config.action_runtime.model_dump()}

    def authorize_action(self, action_id: str, note: str = "") -> dict[str, Any]:
        with self.telemetry.span("research.action.authorize", {
            "research.action_id": action_id,
        }) as span:
            result = self._authorize_action_local(action_id, note)
            span.set_attribute(
                "research.status",
                str((result.get("execution") or {}).get("status") or "UNKNOWN"),
            )
            return result

    def _authorize_action_local(self, action_id: str, note: str = "") -> dict[str, Any]:
        try:
            return self.runtime.action_service.authorize(action_id, note)
        except FileNotFoundError as exc:
            raise ApplicationError("action not found", status_code=404,
                                   code="UNKNOWN_ACTION") from exc
        except RuntimeError as exc:
            raise ApplicationError(str(exc), code="ACTION_BLOCKED") from exc

    def execute_action(self, action_id: str, confirmation: str) -> dict[str, Any]:
        with self.telemetry.span("research.action.execute", {
            "research.action_id": action_id,
        }) as span:
            result = self._execute_action_local(action_id, confirmation)
            span.set_attribute(
                "research.status",
                str((result.get("execution") or {}).get("status") or "UNKNOWN"),
            )
            return result

    def begin_action_execution(
        self, action_id: str, confirmation: str,
    ) -> tuple[dict[str, Any], PendingActionExecution | None]:
        """Durably claim an Action and return before slow controller work."""
        try:
            return self.runtime.action_service.begin_execute(action_id, confirmation)
        except FileNotFoundError as exc:
            raise ApplicationError(
                "action not found", status_code=404, code="UNKNOWN_ACTION",
            ) from exc
        except RuntimeError as exc:
            raise ApplicationError(str(exc), code="ACTION_BLOCKED") from exc

    def finish_action_execution(
        self, pending: PendingActionExecution,
    ) -> dict[str, Any]:
        """Finish a background Action and refresh its exact project read model."""
        action = pending.plan
        try:
            result = self.runtime.action_service.finish_execute(pending)
        except Exception as exc:
            # The Action is already durably EXECUTING.  Preserve uncertainty;
            # reconciliation must observe effects instead of replaying them.
            execution = self.runtime.action_store.execution(str(action["action_id"]))
            execution.update({
                "status": "RECONCILE_REQUIRED",
                "finished_at": None,
                "error": f"{type(exc).__name__}: {exc}"[:1000],
                "resolution": "UNKNOWN_DO_NOT_RETRY",
                "safe_to_retry": False,
                "next_action": "RECONCILE",
            })
            result = self.runtime.action_store.set_execution(
                str(action["action_id"]), execution,
                event="background_execution_reconcile_required",
            )
        if action.get("operation") in {
            "SUBMIT_RUN", "RETRY_ATTEMPT", "RUN_EVALUATION", "CANCEL_RUN",
        } or result.get("execution", {}).get("status") == "VERIFIED":
            self._refresh_action_project(action)
        return result

    def reconcile_action(self, action_id: str) -> dict[str, Any]:
        """Resolve an uncertain scheduler submission through exact status only."""
        with self.telemetry.span("research.action.reconcile", {
            "research.action_id": action_id,
        }) as span:
            try:
                action = self.runtime.action_store.snapshot(action_id)
                result = self.runtime.action_service.reconcile(action_id)
            except FileNotFoundError as exc:
                raise ApplicationError(
                    "action not found", status_code=404, code="UNKNOWN_ACTION",
                ) from exc
            except RuntimeError as exc:
                raise ApplicationError(str(exc), code="ACTION_BLOCKED") from exc
            self._refresh_action_project(action)
            span.set_attribute(
                "research.status",
                str((result.get("execution") or {}).get("status") or "UNKNOWN"),
            )
            return result

    def _refresh_action_project(self, action: dict[str, Any]) -> None:
        scope = action.get("scope")
        if not isinstance(scope, dict):
            return
        project_name = str(scope.get("project") or "")
        if not project_name:
            return
        try:
            configured = self.runtime.project(project_name)
        except KeyError:
            return
        index_project(self.runtime.index, configured)

    def _execute_action_local(self, action_id: str, confirmation: str) -> dict[str, Any]:
        try:
            action = self.runtime.action_store.snapshot(action_id)
            result = self.runtime.action_service.execute(action_id, confirmation)
        except ApplicationError:
            raise
        except FileNotFoundError as exc:
            raise ApplicationError("action not found", status_code=404,
                                   code="UNKNOWN_ACTION") from exc
        except RuntimeError as exc:
            raise ApplicationError(str(exc), code="ACTION_BLOCKED") from exc
        # Submission may have materialized a durable intent even when its
        # scheduler confirmation is still uncertain. Refresh this exact
        # Project immediately instead of waiting for the next collector cycle.
        if action.get("operation") in {
            "SUBMIT_RUN", "RETRY_ATTEMPT", "RUN_EVALUATION", "CANCEL_RUN",
        } or result.get("execution", {}).get("status") == "VERIFIED":
            self._refresh_action_project(action)
        return result
