"""Transport-neutral use cases owned by the experiment daemon."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import yaml

from .agents.store import utc_now
from .campaign_lifecycle import campaign_snapshot
from .ingest.indexer import index_project
from .ingest.runscan import (
    evaluation_variants,
    parse_iso_ts,
    preferred_attempt_id,
    read_jsonl,
    train_metric_records,
)
from .project_registry import ProjectRegistryError
from .runtime import ExperimentServerRuntime
from .research_operations import (
    GPU_BUDGET,
    OPERATIONS_BY_ID,
    PROPOSAL_OPERATION_IDS,
    OperationAvailability,
    OperationParameter,
    operations_for_scope,
    proposal_scope_error,
)
from .schemas import (
    AgentLifecycleState,
    AgentScope,
    AgentScopeType,
    CampaignRelationship,
    ProjectLifecycleState,
    ProjectRegistrationSource,
    ResearchProject,
)
from .source_identity import file_digest, repository_identity


class ApplicationError(RuntimeError):
    """Stable error shared by transport adapters."""

    def __init__(self, message: str, *, status_code: int = 409,
                 code: str = "APPLICATION_ERROR"):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


_OMITTED_EVIDENCE_KEYS = {
    "stdout_tail", "stderr_tail", "raw_stdout", "raw_stderr", "startup_script",
}


def compact_evidence(value: Any, *, depth: int = 0) -> Any:
    if depth >= 8:
        return "[nested evidence omitted]"
    if isinstance(value, dict):
        return {
            str(key): compact_evidence(item, depth=depth + 1)
            for key, item in value.items()
            if str(key) not in _OMITTED_EVIDENCE_KEYS
        }
    if isinstance(value, list):
        compact = [compact_evidence(item, depth=depth + 1) for item in value[:40]]
        if len(value) > 40:
            compact.append(f"[{len(value) - 40} additional records omitted]")
        return compact
    if isinstance(value, str) and len(value) > 2000:
        return value[:2000] + "…[truncated]"
    return value


_OOM_RE = re.compile(
    r"Tried to allocate (?P<requested>[\d.]+) (?P<requested_unit>[GM]iB).*?"
    r"total capacity of (?P<total>[\d.]+) (?P<total_unit>[GM]iB) of which "
    r"(?P<free>[\d.]+) (?P<free_unit>[GM]iB) is free.*?"
    r"allocated memory (?P<allocated>[\d.]+) (?P<allocated_unit>[GM]iB) is allocated",
    re.IGNORECASE,
)


def _memory_bytes(value: str, unit: str) -> int:
    factor = 1024 ** 3 if unit.lower() == "gib" else 1024 ** 2
    return round(float(value) * factor)


def structured_failure_summary(
    collection: dict[str, Any], decision: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Extract stable failure fields from bounded collected process evidence."""
    process = collection.get("process_evidence")
    process = process if isinstance(process, dict) else {}
    stderr_tail = process.get("stderr_tail")
    stdout_tail = process.get("stdout_tail")
    stderr = "\n".join(str(item) for item in stderr_tail or [])
    stdout = "\n".join(str(item) for item in stdout_tail or [])
    combined = f"{stdout}\n{stderr}"
    failure_class = collection.get("failure_class")
    if failure_class is None and isinstance(decision, dict):
        failure_class = decision.get("failure_class")
    if str(failure_class or "").strip().lower() in {"", "none", "null"}:
        failure_class = None

    oom = _OOM_RE.search(stderr)
    if oom:
        ranks = {int(value) for value in re.findall(r"\[rank(\d+)\].*?OutOfMemoryError", stderr)}
        world_match = re.search(r"(?:world_size|nproc_per_node)=(\d+)", stdout)
        world_size = int(world_match.group(1)) if world_match else None
        phase = "first_backward" if (
            "Performing initial training step" in stdout and ".backward()" in stderr
        ) else ("backward" if ".backward()" in stderr else "unknown")
        return {
            "failure_signature": "CUDA_OOM",
            "failure_class": failure_class or "resource",
            "phase": phase,
            "requested_bytes": _memory_bytes(oom["requested"], oom["requested_unit"]),
            "free_bytes": _memory_bytes(oom["free"], oom["free_unit"]),
            "total_bytes": _memory_bytes(oom["total"], oom["total_unit"]),
            "allocated_bytes": _memory_bytes(oom["allocated"], oom["allocated_unit"]),
            "rank_count": world_size or len(ranks) or None,
            "observed_oom_rank_count": len(ranks) or None,
            "source": "collection.process_evidence.stderr_tail",
        }
    signatures = (
        ("ModuleNotFoundError", "MISSING_PYTHON_MODULE", "configuration"),
        ("no kernel image is available", "UNSUPPORTED_CUDA_KERNEL", "configuration"),
        ("TIMEOUT", "TIMEOUT", "timeout"),
    )
    for needle, signature, default_class in signatures:
        if needle.lower() in combined.lower():
            return {
                "failure_signature": signature,
                "failure_class": failure_class or default_class,
                "phase": "unknown",
                "source": "collection.process_evidence",
            }
    if failure_class or str(collection.get("process_state") or "").upper() == "FAILED":
        return {
            "failure_signature": "UNCLASSIFIED_PROCESS_FAILURE",
            "failure_class": failure_class or "unknown",
            "phase": "unknown",
            "source": "collection.process_evidence",
        }
    return None


def evidence_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ExperimentServerApplication:
    def __init__(self, runtime: ExperimentServerRuntime):
        self.runtime = runtime

    # --------------------------------------------------------------- scopes

    def resolve_scope(
        self, project_name: str, scope_type: AgentScopeType | str, object_id: str,
    ) -> tuple[AgentScope, ResearchProject, Any]:
        try:
            project = self.runtime.project(project_name)
        except KeyError as exc:
            raise ApplicationError(str(exc).strip("'"), status_code=404,
                                   code="UNKNOWN_PROJECT") from exc
        kind = AgentScopeType(scope_type)
        if kind == AgentScopeType.PROJECT:
            if object_id != project.project:
                raise ApplicationError("project agent object_id must equal project",
                                       status_code=404, code="UNKNOWN_PROJECT")
            resolved: Any = project
        elif kind == AgentScopeType.RESEARCH_QUESTION:
            resolved = next((item for item in project.research_questions if item.id == object_id), None)
            if resolved is None:
                raise ApplicationError(f"unknown research_question: {object_id}", status_code=404,
                                       code="UNKNOWN_RESEARCH_QUESTION")
        elif kind == AgentScopeType.CAMPAIGN:
            resolved = next((
                campaign for campaign in project.campaigns if campaign.name == object_id
            ), None)
            if resolved is None:
                raise ApplicationError(f"unknown campaign: {object_id}", status_code=404,
                                       code="UNKNOWN_CAMPAIGN")
        elif kind == AgentScopeType.RUN:
            resolved = self.runtime.index.get_run(project.project, object_id)
            if resolved is None:
                raise ApplicationError(f"unknown run: {object_id}", status_code=404,
                                       code="UNKNOWN_RUN")
        else:
            if "::" not in object_id:
                raise ApplicationError("attempt object_id must be run_id::attempt_id",
                                       status_code=422, code="INVALID_ATTEMPT_ID")
            run_id, attempt_id = object_id.rsplit("::", 1)
            row = self.runtime.index.get_run(project.project, run_id)
            if row is None:
                raise ApplicationError(f"unknown run: {run_id}", status_code=404,
                                       code="UNKNOWN_RUN")
            resolved = next((item for item in row.attempts if item.attempt_id == attempt_id), None)
            if resolved is None:
                raise ApplicationError(f"unknown attempt: {attempt_id}", status_code=404,
                                       code="UNKNOWN_ATTEMPT")
        return AgentScope(project=project.project, scope_type=kind, object_id=object_id), project, resolved

    @staticmethod
    def default_goal(scope: AgentScope, resolved: Any) -> str:
        if scope.scope_type == AgentScopeType.PROJECT:
            return f"推进项目 {getattr(resolved, 'title', scope.object_id)} 的研究目标并管理证据缺口。"
        if scope.scope_type == AgentScopeType.RESEARCH_QUESTION:
            return f"整理研究问题 {scope.object_id} 关联的证据、未决事项和可选下一步。"
        if scope.scope_type == AgentScopeType.CAMPAIGN:
            return f"检查 campaign {scope.object_id} 的对照完整性、可比性和证据完成状态。"
        if scope.scope_type == AgentScopeType.RUN:
            return f"解释不可变 Run {scope.object_id} 的运行与模型证据，不改写其科学身份。"
        return f"诊断 Attempt {scope.object_id} 的执行证据和基础设施状态。"

    @staticmethod
    def row_evidence(row: Any) -> dict[str, Any]:
        return compact_evidence({
            "run_id": row.run_id, "campaign": row.campaign, "role": row.role,
            "campaign_binding": row.campaign_binding.model_dump(mode="json"),
            "campaign_memberships": [
                item.model_dump(mode="json") for item in row.campaign_memberships
            ],
            "scheduler_state": row.scheduler_state,
            "evidence": row.evidence.model_dump(mode="json"),
            "latest_metrics": row.latest_metrics, "eval_metrics": row.eval_metrics,
            "eval_variants": row.eval_variants,
            "canonical_eval_variant_id": row.canonical_eval_variant_id,
            "checkpoint": row.checkpoint, "artifacts": row.artifacts,
            "decision": row.decision, "provenance": row.provenance,
            "warnings": row.warnings, "evidence_conflicts": row.evidence_conflicts,
        })

    def campaign_contexts(self, project: ResearchProject, row: Any) -> list[dict[str, Any]]:
        """Bounded comparator context for a Run reused by authored Campaigns."""
        contexts = []
        for binding in row.campaign_memberships:
            ref = next((item for item in project.campaigns
                        if item.name == binding.campaign), None)
            revision = ref.current_revision if ref else None
            if ref is None:
                contexts.append({
                    "campaign": binding.campaign,
                    "revision_id": binding.revision_id,
                    "membership": binding.membership.model_dump(mode="json"),
                    "comparator_runs": [],
                    "lifecycle": {
                        "lifecycle_state": "UNKNOWN",
                        "reason": "authored Campaign is no longer present in the Project catalog",
                    },
                    "orphaned_campaign": True,
                })
                continue
            peers = []
            for peer in self.runtime.index.list_runs(project.project, binding.campaign):
                peer_binding = next(
                    (item for item in peer.campaign_memberships
                     if item.campaign == binding.campaign), None,
                )
                if peer_binding and not peer_binding.membership.included_in_analysis:
                    continue
                peers.append({
                    "run_id": peer.run_id,
                    "membership": (
                        peer_binding.membership.model_dump(mode="json")
                        if peer_binding else None
                    ),
                    "scheduler_state": peer.scheduler_state,
                    "latest_metrics": peer.latest_metrics,
                    "eval_metrics": peer.eval_metrics,
                    "provenance": peer.provenance,
                })
            lifecycle = campaign_snapshot(
                self.runtime.index, project, binding.campaign,
            )
            contexts.append({
                "campaign": binding.campaign,
                "revision_id": revision.revision_id if revision else None,
                "membership": binding.membership.model_dump(mode="json"),
                "lifecycle_state": lifecycle.get("lifecycle_state"),
                "research_contract": revision.research_contract if revision else None,
                "comparator_runs": peers,
            })
        return compact_evidence(contexts)

    def bounded_evidence(self, scope: AgentScope, project: ResearchProject,
                         resolved: Any) -> dict[str, Any]:
        index = self.runtime.index
        if scope.scope_type == AgentScopeType.PROJECT:
            rows = index.list_runs(project.project)
            return {
                "project": project.project, "title": project.title,
                "research_questions": [
                    {"id": item.id, "title": item.title, "status": item.status}
                    for item in project.research_questions
                ],
                "runs": [{
                    "run_id": row.run_id, "campaign": row.campaign, "role": row.role,
                    "campaign_relationship": row.campaign_binding.relationship.value,
                    "scheduler_state": row.scheduler_state, "decision": row.decision,
                    "stale_layers": [
                        name for name in ("scheduler", "worker", "process", "model", "evaluation")
                        if getattr(row.evidence, name).stale
                    ],
                } for row in rows],
            }
        if scope.scope_type == AgentScopeType.RESEARCH_QUESTION:
            names = set(resolved.links.campaigns)
            rows = [
                row for row in index.list_runs(project.project)
                if row.campaign in names or any(
                    binding.campaign in names for binding in row.campaign_memberships
                )
            ]
            return {"research_question": resolved.model_dump(mode="json"),
                    "runs": [self.row_evidence(row) for row in rows]}
        if scope.scope_type == AgentScopeType.CAMPAIGN:
            rows = index.list_runs(project.project, campaign=scope.object_id)
            return {"campaign": resolved.model_dump(mode="json"),
                    "lifecycle": campaign_snapshot(index, project, scope.object_id),
                    "runs": [self.row_evidence(row) for row in rows]}
        if scope.scope_type == AgentScopeType.RUN:
            return {"run": self.row_evidence(resolved),
                    "campaign_contexts": self.campaign_contexts(project, resolved),
                    "attempts": [
                item.model_dump(mode="json") for item in resolved.attempts
            ]}
        run_id, _ = scope.object_id.rsplit("::", 1)
        row = index.get_run(project.project, run_id)
        return {"run": self.row_evidence(row),
                "campaign_contexts": self.campaign_contexts(project, row),
                "attempt": resolved.model_dump(mode="json")}

    def _snapshot(self, scope: AgentScope, project: ResearchProject,
                  resolved: Any) -> dict[str, Any]:
        payload = self.runtime.agent_store.snapshot(scope)
        payload["actions"] = self.runtime.action_store.list_for_scope(scope)
        payload["current_evidence_digest"] = evidence_digest(
            self.bounded_evidence(scope, project, resolved)
        )
        payload["evidence_checked_at"] = utc_now()
        return payload

    # ------------------------------------------------------- research operations

    def operation_availability(
        self, project: str, scope_type: AgentScopeType | str, object_id: str,
    ) -> list[OperationAvailability]:
        """Return deterministic operation eligibility for one exact scope."""
        scope, configured, resolved = self.resolve_scope(project, scope_type, object_id)
        default_budget = configured.proposal_defaults.get("max_gpu_hours", 1.0)
        default_budget = float(default_budget) if isinstance(default_budget, (int, float)) \
            and default_budget > 0 else 1.0
        result: list[OperationAvailability] = []
        for base_operation in operations_for_scope(scope.scope_type):
            parameters = tuple(
                replace(parameter, default=default_budget)
                if parameter.key == GPU_BUDGET.key else parameter
                for parameter in base_operation.parameters
            )
            operation = replace(base_operation, parameters=parameters)
            try:
                reasons = self._operation_blockers(
                    operation.operation_id, scope, configured, resolved,
                )
            except (ApplicationError, KeyError, OSError, ValueError) as exc:
                reasons = [f"Eligibility evidence is unavailable: {exc}"]
            result.append(OperationAvailability(
                operation=operation, scope=scope,
                status="BLOCKED" if reasons else "AVAILABLE",
                reasons=tuple(reasons), expected_effect=operation.expected_effect,
                metadata={"proposal_kind": operation.proposal_kind},
            ))
        return result

    def _operation_blockers(
        self, operation_id: str, scope: AgentScope,
        project: ResearchProject, resolved: Any,
    ) -> list[str]:
        reasons: list[str] = []
        if operation_id == "question.create":
            if not project.research_questions_dir:
                reasons.append("Project does not declare research_questions_dir")
        elif operation_id in {"campaign.create", "campaign.update", "run.derive"}:
            if project.authored_file is None or not Path(project.authored_file).is_file():
                reasons.append("Authored research_project catalog is unavailable")
            if operation_id == "campaign.update" and getattr(resolved, "current_revision", None) is None:
                reasons.append("Campaign has no resolved current revision")
            if operation_id == "run.derive" and scope.scope_type == AgentScopeType.RUN:
                memberships = getattr(resolved, "campaign_memberships", []) or []
                if not memberships and not getattr(resolved, "campaign", None):
                    reasons.append("Run has no authored Campaign context to derive from")
        elif operation_id == "campaign.complete":
            status = campaign_snapshot(self.runtime.index, project, scope.object_id)
            if status.get("lifecycle_state") != "COMPLETABLE":
                failed = [
                    str(gate.get("name") or gate.get("id") or "gate")
                    for gate in (status.get("completion") or {}).get("gates", [])
                    if gate.get("status") != "PASS"
                ]
                detail = ", ".join(failed) if failed else status.get("lifecycle_state", "UNKNOWN")
                reasons.append(f"Campaign is not completable; blocked by {detail}")
        elif operation_id == "object.archive":
            if scope.scope_type == AgentScopeType.CAMPAIGN:
                status = campaign_snapshot(self.runtime.index, project, scope.object_id)
                if status.get("lifecycle_state") == "ARCHIVED":
                    reasons.append("Campaign is already archived")
            else:
                root = (project.base_dir or Path(".")) / "experiments" / "archive_records"
                if scope.scope_type == AgentScopeType.RUN:
                    record = root / "runs" / f"{scope.object_id}.yml"
                else:
                    run_id, attempt_id = scope.object_id.rsplit("::", 1)
                    record = root / "attempts" / f"{run_id}--{attempt_id}.yml"
                if record.is_file():
                    reasons.append(f"Archive record already exists: {record}")
        elif operation_id == "run.submit":
            if project.controller is None:
                reasons.append("Project has no controller configuration")
            state = str(getattr(resolved, "scheduler_state", None) or "NOT_SUBMITTED").upper()
            if state in {
                "SUBMITTING", "PENDING", "QUEUED", "STARTING", "RUNNING", "EVALUATING",
                "SUCCEEDED", "FAILED", "PREEMPTED", "CANCELLED",
            }:
                reasons.append(f"Run state {state} is not eligible for first submission")
            if any(item.has_submission for item in getattr(resolved, "attempts", []) or []):
                reasons.append("Run already has submitted Attempt evidence; use exact Attempt retry")
            materialized = [
                binding for binding in getattr(resolved, "campaign_memberships", []) or []
                if binding.membership.kind == "materialize"
            ]
            if not materialized and not getattr(resolved, "campaign", None):
                reasons.append("Run is not an authored materialized Campaign membership")
        elif operation_id in {"attempt.retry", "attempt.cancel"}:
            state = str(getattr(resolved, "state", None) or "UNKNOWN").upper()
            decision = getattr(resolved, "decision", {}) or {}
            if operation_id == "attempt.retry":
                if state not in {"FAILED", "PREEMPTED", "CANCELLED"}:
                    reasons.append(f"Attempt state {state} is not retryable")
                if str(decision.get("action") or "").upper() == "DO_NOT_RETRY":
                    reasons.append("Collected decision is DO_NOT_RETRY")
                allowed, used = decision.get("retries_allowed"), decision.get("retries_used", 0)
                if isinstance(allowed, int) and isinstance(used, int) and used >= allowed:
                    reasons.append(f"Retry-count budget exhausted: used={used}, allowed={allowed}")
            else:
                if state not in {
                    "SUBMITTING", "PENDING", "QUEUED", "STARTING", "RUNNING", "EVALUATING",
                }:
                    reasons.append(f"Attempt state {state} is not cancellable")
                if not getattr(resolved, "backend_job_id", None):
                    reasons.append("Attempt has no exact backend_job_id")
        elif operation_id == "run.evaluate":
            if project.controller is None:
                reasons.append("Project has no controller configuration")
            elif not project.controller.capabilities.get("evaluation_as_run"):
                reasons.append("Controller does not declare evaluation_as_run")
            identity = scope.object_id if scope.scope_type == AgentScopeType.ATTEMPT else None
            if identity is None:
                attempt_id = preferred_attempt_id(Path(resolved.run_dir))
                if attempt_id:
                    identity = f"{resolved.run_id}::{attempt_id}"
            if identity is None:
                reasons.append("No exact source Attempt is available for evaluation")
            else:
                checkpoints = self.attempt_checkpoints(project.project, identity)
                if not checkpoints.get("latest_completed_checkpoint"):
                    reasons.append("No completed checkpoint evidence is available")
        return reasons

    def _require_operation_available(
        self, operation_id: str, project: str,
        scope_type: AgentScopeType | str, object_id: str,
    ) -> None:
        availability = next((item for item in self.operation_availability(
            project, scope_type, object_id,
        ) if item.operation.operation_id == operation_id), None)
        if availability is None or not availability.available:
            reasons = availability.reasons if availability else ("operation is not valid in this scope",)
            raise ApplicationError("; ".join(reasons), code="OPERATION_BLOCKED")

    def invoke_direct_operation(
        self, operation_id: str, project: str,
        scope_type: AgentScopeType | str, object_id: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a catalogued direct proposal for one exact scope.

        Agent-backed operations intentionally remain Agent turns.  This method
        is the transport-neutral home for the small set of direct proposal
        operations, so HTTP and terminal clients cannot drift in how they
        bind object identities or apply availability gates.
        """
        parameters = parameters or {}
        scope, configured, _ = self.resolve_scope(project, scope_type, object_id)
        self._require_operation_available(operation_id, project, scope.scope_type, object_id)
        reason = str(parameters.get("reason") or "")
        budget = 0.0
        if operation_id in {"run.submit", "attempt.retry"}:
            budget_value = parameters.get("max_gpu_hours")
            default_budget = configured.proposal_defaults.get("max_gpu_hours", 1.0)
            try:
                budget = float(default_budget if budget_value is None else budget_value)
            except (TypeError, ValueError) as exc:
                raise ApplicationError(
                    "max_gpu_hours must be a number", code="INVALID_OPERATION",
                ) from exc
            if budget <= 0:
                raise ApplicationError("max_gpu_hours must be positive", code="INVALID_OPERATION")

        if operation_id == "campaign.complete":
            return self.propose_campaign_completion(
                project, object_id,
                outcome=str(parameters.get("outcome") or "INCONCLUSIVE"),
                assessment=str(parameters.get("assessment") or ""),
            )
        if operation_id == "object.archive":
            return self.propose_object_archive(project, scope.scope_type, object_id, reason=reason)
        if operation_id == "run.submit":
            return self.propose_run_submit(
                project, object_id, max_gpu_hours=budget,
                reason=reason or "Requested from the scoped research operation catalog",
            )
        if operation_id == "attempt.retry":
            return self.propose_attempt_retry(
                project, object_id,
                new_attempt_id=(
                    str(parameters["new_attempt_id"])
                    if parameters.get("new_attempt_id") else None
                ),
                max_gpu_hours=budget, reason=reason,
            )
        if operation_id == "attempt.cancel":
            return self.propose_attempt_cancel(project, object_id, reason=reason)
        raise ApplicationError(
            f"operation {operation_id} is not a direct proposal operation",
            code="INVALID_OPERATION",
        )

    # --------------------------------------------------------------- agents

    def agent_snapshot(self, project: str, scope_type: AgentScopeType | str,
                       object_id: str) -> dict[str, Any]:
        scope, configured, resolved = self.resolve_scope(project, scope_type, object_id)
        self.runtime.agent_store.ensure(
            scope, default_goal=self.default_goal(scope, resolved)
        )
        return self._snapshot(scope, configured, resolved)

    def update_agent_goal(self, project: str, scope_type: AgentScopeType | str,
                          object_id: str, goal: str) -> dict[str, Any]:
        scope, configured, resolved = self.resolve_scope(project, scope_type, object_id)
        self.runtime.agent_store.ensure(scope, default_goal=self.default_goal(scope, resolved))
        self.runtime.agent_store.set_goal(scope, goal)
        return self._snapshot(scope, configured, resolved)

    def decide_proposal(self, project: str, scope_type: AgentScopeType | str,
                        object_id: str, proposal_id: str, decision: str,
                        note: str = "") -> dict[str, Any]:
        scope, configured, resolved = self.resolve_scope(project, scope_type, object_id)
        store = self.runtime.agent_store
        store.ensure(scope, default_goal=self.default_goal(scope, resolved))
        try:
            proposal = store.proposal(scope, proposal_id)
        except FileNotFoundError as exc:
            raise ApplicationError("unknown proposal", status_code=404,
                                   code="UNKNOWN_PROPOSAL") from exc
        current = evidence_digest(self.bounded_evidence(scope, configured, resolved))
        if decision == "APPROVED" and proposal.get("evidence_digest") != current:
            raise ApplicationError(
                "outdated proposal cannot be approved; regenerate it from current evidence",
                code="OUTDATED_PROPOSAL",
            )
        try:
            approval = store.decide_proposal(scope, proposal_id, decision, note=note)
        except ValueError as exc:
            raise ApplicationError(str(exc), code="INVALID_PROPOSAL") from exc
        snapshot = store.snapshot(scope)
        if not any(item.get("status") == "PENDING" for item in snapshot["proposals"]):
            store.set_state(scope, AgentLifecycleState.IDLE)
        return {"approval": approval, "agent": self._snapshot(scope, configured, resolved)}

    def proposal_show(self, project: str, scope_type: AgentScopeType | str,
                      object_id: str, proposal_id: str) -> dict[str, Any]:
        scope, _, _ = self.resolve_scope(project, scope_type, object_id)
        try:
            return self.runtime.agent_store.proposal(scope, proposal_id)
        except FileNotFoundError as exc:
            raise ApplicationError("unknown proposal", status_code=404,
                                   code="UNKNOWN_PROPOSAL") from exc

    def begin_agent_turn(
        self, project: str, scope_type: AgentScopeType | str, object_id: str,
        message: str, *, enforce_operation_availability: bool = False,
    ) -> dict[str, Any]:
        """Persist a request for an external Agent client; never invoke a model."""
        scope, configured, resolved = self.resolve_scope(project, scope_type, object_id)
        store = self.runtime.agent_store
        snapshot = store.ensure(scope, default_goal=self.default_goal(scope, resolved))
        request = store.create_turn_request(
            scope, message=message,
            enforce_operation_availability=enforce_operation_availability,
        )
        store.append_message(scope, role="user", content=message)
        store.set_state(scope, AgentLifecycleState.ANALYZING,
                        current_task="等待 Agent client 领取并分析")
        return {"turn": request, "agent": self._snapshot(scope, configured, resolved)}

    def claim_agent_turn(
        self, project: str, scope_type: AgentScopeType | str, object_id: str,
        request_id: str, *, client_id: str, provider: str,
    ) -> dict[str, Any]:
        """Bind current evidence to one queued turn and return structured context."""
        scope, configured, resolved = self.resolve_scope(project, scope_type, object_id)
        store = self.runtime.agent_store
        snapshot = store.ensure(scope, default_goal=self.default_goal(scope, resolved))
        try:
            record = store.turn_request(scope, request_id)
        except (FileNotFoundError, ValueError) as exc:
            raise ApplicationError("unknown Agent turn", status_code=404,
                                   code="UNKNOWN_AGENT_TURN") from exc
        if record.get("status") != "PENDING":
            raise ApplicationError("Agent turn is not pending", code="AGENT_TURN_NOT_PENDING")
        previous_provider = snapshot.get("conversation_provider")
        if provider and previous_provider != provider:
            snapshot = store.begin_provider_epoch(
                scope, provider, preserve_messages=True,
            )
            if previous_provider is not None:
                store.append_message(scope, role="user", content=str(record["message"]))
                snapshot = store.snapshot(scope)
        evidence = self.bounded_evidence(scope, configured, resolved)
        digest = evidence_digest(evidence)
        captured = utc_now()
        record = store.set_turn_request(
            scope, request_id, status="RUNNING", evidence_digest=digest,
            evidence_captured_at=captured, client_id=client_id,
        )
        base_dir = Path(configured.base_dir or ".").expanduser().resolve()
        project_file = Path(configured.authored_file).resolve() \
            if configured.authored_file else None
        try:
            project_file_relative = str(project_file.relative_to(base_dir)) \
                if project_file else ""
        except ValueError:
            project_file_relative = ""
        return {
            "turn": record,
            "agent_id": snapshot["agent_id"],
            "session_id": snapshot.get("thread_id"),
            "goal": snapshot["goal"],
            "scope": scope.model_dump(mode="json"),
            "message": record["message"],
            "evidence": compact_evidence(evidence),
            "evidence_digest": digest,
            "evidence_captured_at": captured,
            "project_context": {
                "workspace_root": str(base_dir),
                "project_file": str(project_file or ""),
                "project_file_relative": project_file_relative,
                "project_file_digest": file_digest(project_file) \
                    if project_file and project_file.is_file() else "",
                "source_identity": repository_identity(base_dir),
                "proposal_defaults": configured.proposal_defaults,
                "agent_guidance": configured.agent_guidance,
            },
            "operations": [asdict(item) for item in self.operation_availability(
                scope.project, scope.scope_type, scope.object_id,
            )],
        }

    def claim_next_agent_turn(
        self, *, client_id: str, provider: str, project: str | None = None,
    ) -> dict[str, Any] | None:
        for record in self.runtime.agent_store.pending_turn_requests(project):
            scope = AgentScope.model_validate(record["scope"])
            try:
                return self.claim_agent_turn(
                    scope.project, scope.scope_type, scope.object_id,
                    str(record["request_id"]), client_id=client_id, provider=provider,
                )
            except ApplicationError as exc:
                if exc.code != "AGENT_TURN_NOT_PENDING":
                    raise
        return None

    def complete_agent_turn(
        self, project: str, scope_type: AgentScopeType | str, object_id: str,
        request_id: str, *, status: str, session_id: str | None,
        message: str = "", proposals: list[dict[str, Any]] | None = None,
        evidence_digest_value: str, client_id: str, error: str = "",
    ) -> dict[str, Any]:
        """Validate client output against server-owned scope and evidence, then persist it."""
        scope, configured, resolved = self.resolve_scope(project, scope_type, object_id)
        store = self.runtime.agent_store
        try:
            record = store.turn_request(scope, request_id)
        except (FileNotFoundError, ValueError) as exc:
            raise ApplicationError("unknown Agent turn", status_code=404,
                                   code="UNKNOWN_AGENT_TURN") from exc
        if record.get("status") != "RUNNING":
            raise ApplicationError("Agent turn is not running", code="AGENT_TURN_NOT_RUNNING")
        if record.get("client_id") != client_id:
            raise ApplicationError(
                "Agent turn is owned by a different client", code="AGENT_TURN_OWNER_MISMATCH",
            )
        if status == "FAILED":
            safe_error = (error or "Agent client failed")[:500]
            store.set_turn_request(scope, request_id, status="FAILED", error=safe_error)
            store.set_state(scope, AgentLifecycleState.FAILED, last_error=safe_error)
            return {"turn": store.turn_request(scope, request_id),
                    "agent": self._snapshot(scope, configured, resolved)}
        expected_digest = str(record.get("evidence_digest") or "")
        current_digest = evidence_digest(self.bounded_evidence(scope, configured, resolved))
        if evidence_digest_value != expected_digest or current_digest != expected_digest:
            store.set_turn_request(
                scope, request_id, status="FAILED", error="evidence changed during Agent turn",
            )
            store.set_state(scope, AgentLifecycleState.FAILED,
                            last_error="evidence changed during Agent turn")
            raise ApplicationError(
                "Agent result is stale; evidence changed during analysis",
                code="STALE_AGENT_RESULT",
            )
        availability_by_id = {
            item.operation.operation_id: item for item in self.operation_availability(
                scope.project, scope.scope_type, scope.object_id,
            )
        }
        accepted_proposals = []
        rejected_proposals = []
        for proposal in proposals or []:
            kind = str(proposal.get("kind") or "")
            error = proposal_scope_error(kind, scope.scope_type)
            operation_id = PROPOSAL_OPERATION_IDS.get(kind)
            availability = availability_by_id.get(operation_id) if operation_id else None
            if (bool(record.get("enforce_operation_availability")) and error is None
                    and availability is not None and not availability.available):
                error = f"proposal kind {kind} is currently blocked: " + \
                    "; ".join(availability.reasons)
            if error:
                rejected_proposals.append(error)
            else:
                accepted_proposals.append(proposal)
        assistant_message = message
        if rejected_proposals:
            assistant_message += (
                "\n\nDaemon scope policy rejected proposal output:\n- "
                + "\n- ".join(rejected_proposals)
            )
        store.append_message(scope, role="assistant", content=assistant_message,
                             thread_id=session_id, evidence_digest=expected_digest,
                             evidence_captured_at=record.get("evidence_captured_at"))
        created = store.add_proposals(
            scope, accepted_proposals, evidence_digest=expected_digest,
        )
        store.set_state(scope, AgentLifecycleState.WAITING_FOR_APPROVAL
                        if created else AgentLifecycleState.IDLE)
        response = self._snapshot(scope, configured, resolved)
        response["created_proposals"] = created
        store.set_turn_request(
            scope, request_id, status="COMPLETED",
            result={"created_proposal_ids": [item["proposal_id"] for item in created]},
        )
        response["turn"] = store.turn_request(scope, request_id)
        return response

    # ------------------------------------------------------------- read model

    def object_show(self, project: str, scope_type: AgentScopeType | str,
                    object_id: str) -> dict[str, Any]:
        """Return one of the five canonical objects plus its bounded evidence."""
        scope, configured, resolved = self.resolve_scope(project, scope_type, object_id)
        if hasattr(resolved, "model_dump"):
            scientific_object = resolved.model_dump(mode="json")
        else:
            scientific_object = compact_evidence(resolved)
        return {
            "scope": scope.model_dump(mode="json"),
            "object": scientific_object,
            "evidence": self.bounded_evidence(scope, configured, resolved),
        }

    def campaign_list(self, project: str) -> dict[str, Any]:
        configured = self.runtime.project(project)
        return {
            "project": project,
            "campaigns": [
                campaign_snapshot(self.runtime.index, configured, item.name)
                for item in configured.campaigns
            ],
        }

    def campaign_status(self, project: str, campaign: str) -> dict[str, Any]:
        configured = self.runtime.project(project)
        try:
            return campaign_snapshot(self.runtime.index, configured, campaign)
        except KeyError as exc:
            raise ApplicationError(str(exc).strip("'"), status_code=404,
                                   code="UNKNOWN_CAMPAIGN") from exc

    def propose_campaign_completion(
        self, project: str, campaign: str, *, outcome: str, assessment: str,
    ) -> dict[str, Any]:
        self._require_operation_available(
            "campaign.complete", project, AgentScopeType.CAMPAIGN, campaign,
        )
        scope, configured, resolved = self.resolve_scope(
            project, AgentScopeType.CAMPAIGN, campaign,
        )
        status = campaign_snapshot(self.runtime.index, configured, campaign)
        if status["lifecycle_state"] != "COMPLETABLE":
            raise ApplicationError(
                f"Campaign is not completable: {status['lifecycle_state']}",
                code="CAMPAIGN_NOT_COMPLETABLE",
            )
        if not outcome.strip():
            raise ApplicationError("outcome is required", status_code=422,
                                   code="INVALID_CAMPAIGN_OUTCOME")
        draft = yaml.safe_dump({
            "schema_version": 1,
            "project": project,
            "campaign": campaign,
            "revision_id": status["revision_id"],
            "evidence_digest": status["completion"]["evidence_digest"],
            "membership_run_ids": status["completion"]["membership_run_ids"],
            "outcome": outcome.strip(),
            "assessment": assessment.strip(),
        }, allow_unicode=True, sort_keys=False)
        store = self.runtime.agent_store
        store.ensure(scope, default_goal=self.default_goal(scope, resolved))
        digest = evidence_digest(self.bounded_evidence(scope, configured, resolved))
        created = store.add_proposals(scope, [{
            "kind": "COMPLETE_CAMPAIGN",
            "title": f"Complete Campaign {campaign}",
            "target": f"campaign://{project}/{campaign}@{status['revision_id']}",
            "change_summary": "freeze an evidence-bound Campaign completion record",
            "resource_estimate": "none",
            "rationale": "all Campaign completion gates currently pass",
            "risk": "scientific conclusion becomes an immutable reviewed record",
            "draft": draft,
        }], evidence_digest=digest)
        store.set_state(scope, AgentLifecycleState.WAITING_FOR_APPROVAL)
        return {"proposal": created[0], "campaign": status}

    def propose_campaign_archive(
        self, project: str, campaign: str, *, reason: str,
    ) -> dict[str, Any]:
        self._require_operation_available(
            "object.archive", project, AgentScopeType.CAMPAIGN, campaign,
        )
        scope, configured, resolved = self.resolve_scope(
            project, AgentScopeType.CAMPAIGN, campaign,
        )
        status = campaign_snapshot(self.runtime.index, configured, campaign)
        if status["lifecycle_state"] == "ARCHIVED":
            raise ApplicationError("Campaign is already archived",
                                   code="CAMPAIGN_ALREADY_ARCHIVED")
        if not reason.strip():
            raise ApplicationError("archive reason is required", status_code=422,
                                   code="INVALID_ARCHIVE_REASON")
        draft = yaml.safe_dump({
            "schema_version": 1,
            "project": project,
            "campaign": campaign,
            "revision_id": status["revision_id"],
            "reason": reason.strip(),
            "prior_lifecycle_state": status["lifecycle_state"],
        }, allow_unicode=True, sort_keys=False)
        store = self.runtime.agent_store
        store.ensure(scope, default_goal=self.default_goal(scope, resolved))
        digest = evidence_digest(self.bounded_evidence(scope, configured, resolved))
        created = store.add_proposals(scope, [{
            "kind": "ARCHIVE_CAMPAIGN",
            "title": f"Archive Campaign {campaign}",
            "target": f"campaign://{project}/{campaign}@{status['revision_id']}",
            "change_summary": "freeze an immutable Campaign archive record",
            "resource_estimate": "none",
            "rationale": reason.strip(),
            "risk": "the Campaign revision will leave active research views",
            "draft": draft,
        }], evidence_digest=digest)
        store.set_state(scope, AgentLifecycleState.WAITING_FOR_APPROVAL)
        return {"proposal": created[0], "campaign": status}

    def propose_object_archive(
        self, project: str, scope_type: AgentScopeType | str, object_id: str, *, reason: str,
    ) -> dict[str, Any]:
        self._require_operation_available("object.archive", project, scope_type, object_id)
        scope, configured, resolved = self.resolve_scope(project, scope_type, object_id)
        if scope.scope_type not in {AgentScopeType.RUN, AgentScopeType.ATTEMPT}:
            raise ApplicationError("only Run or Attempt records can be archived here",
                                   code="INVALID_ARCHIVE_SCOPE")
        if not reason.strip():
            raise ApplicationError("archive reason is required", status_code=422,
                                   code="INVALID_ARCHIVE_REASON")
        if scope.scope_type == AgentScopeType.ATTEMPT:
            run_id, attempt_id = object_id.rsplit("::", 1)
            kind = "ARCHIVE_ATTEMPT"
            target = f"attempt://{project}/{run_id}::{attempt_id}"
        else:
            run_id, attempt_id = object_id, None
            kind = "ARCHIVE_RUN"
            target = f"run://{project}/{run_id}"
        digest = evidence_digest(self.bounded_evidence(scope, configured, resolved))
        payload = {
            "schema_version": 1, "project": project, "run_id": run_id,
            "evidence_digest": digest, "reason": reason.strip(),
        }
        if attempt_id is not None:
            payload["attempt_id"] = attempt_id
        store = self.runtime.agent_store
        store.ensure(scope, default_goal=self.default_goal(scope, resolved))
        created = store.add_proposals(scope, [{
            "kind": kind,
            "title": f"Archive {scope.scope_type.value} record {object_id}",
            "target": target,
            "change_summary": "append an evidence-bound archive record without deleting evidence",
            "resource_estimate": "none",
            "rationale": reason.strip(),
            "risk": "object is hidden from default active workflows but immutable evidence remains",
            "draft": yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        }], evidence_digest=digest)
        store.set_state(scope, AgentLifecycleState.WAITING_FOR_APPROVAL)
        return {"proposal": created[0]}

    def _attempt_context(self, project: str, identity: str):
        scope, configured, attempt = self.resolve_scope(
            project, AgentScopeType.ATTEMPT, identity,
        )
        run_id, attempt_id = identity.rsplit("::", 1)
        row = self.runtime.index.get_run(project, run_id)
        assert row is not None
        attempt_dir = Path(row.run_dir) / "attempts" / attempt_id
        return scope, configured, row, attempt, attempt_dir

    @staticmethod
    def _read_mapping(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _read_yaml_mapping(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _gate(gate_id: str, status: str, message: str,
              evidence: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "id": gate_id, "status": status, "message": message,
            "evidence": evidence or {},
        }

    @staticmethod
    def _validation_payload(*, object_type: str, identity: str,
                            gates: list[dict[str, Any]]) -> dict[str, Any]:
        identity_gate_ids = {
            "run.identity", "run.campaign_binding", "run.provenance",
            "run.current_attempt",
            "attempt.identity", "attempt.immutable_provenance", "attempt.current",
            "attempt.evidence_identity", "attempt.backend_job_id",
        }
        execution_gate_ids = {"attempt.execution_layers"}
        for gate in gates:
            gate_id = gate["id"]
            gate["dimension"] = (
                "identity" if gate_id in identity_gate_ids else
                "execution" if gate_id in execution_gate_ids else
                "scientific_evidence"
            )

        def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
            dimension_counts = {
                status: sum(gate["status"] == status for gate in selected)
                for status in ("PASS", "UNKNOWN", "BLOCKED")
            }
            if dimension_counts["BLOCKED"]:
                dimension_result = "BLOCKED"
            elif dimension_counts["UNKNOWN"] or not selected:
                dimension_result = "UNKNOWN"
            else:
                dimension_result = "PASS"
            return {"result": dimension_result, "summary": dimension_counts}

        counts = {
            status: sum(gate["status"] == status for gate in gates)
            for status in ("PASS", "UNKNOWN", "BLOCKED")
        }
        if counts["BLOCKED"]:
            result = "BLOCKED"
        elif counts["UNKNOWN"]:
            result = "UNKNOWN"
        else:
            result = "PASS"
        dimensions = {
            name: summarize([gate for gate in gates if gate["dimension"] == name])
            for name in ("execution", "identity", "scientific_evidence")
        }
        return {
            "schema_version": 1,
            "object_type": object_type,
            "identity": identity,
            "result": result,
            "exit_code": 0 if result == "PASS" else 3,
            "execution_evidence_result": dimensions["execution"]["result"],
            "identity_result": dimensions["identity"]["result"],
            "scientific_evidence_result": dimensions["scientific_evidence"]["result"],
            "dimensions": dimensions,
            "summary": counts,
            "gates": gates,
        }

    @staticmethod
    def _identity_gate(*, gate_id: str, label: str, payload: dict[str, Any],
                       expected: dict[str, Any]) -> dict[str, Any]:
        if not payload:
            return ExperimentServerApplication._gate(
                gate_id, "UNKNOWN", f"{label} is missing or unreadable",
            )
        missing = [key for key in expected if payload.get(key) is None]
        mismatches = {
            key: {"expected": value, "observed": payload.get(key)}
            for key, value in expected.items()
            if payload.get(key) is not None and payload.get(key) != value
        }
        evidence = {"expected": expected, "observed": {
            key: payload.get(key) for key in expected
        }}
        if mismatches:
            evidence["mismatches"] = mismatches
            return ExperimentServerApplication._gate(
                gate_id, "BLOCKED", f"{label} identity conflicts with the requested object",
                evidence,
            )
        if missing:
            evidence["missing_fields"] = missing
            return ExperimentServerApplication._gate(
                gate_id, "UNKNOWN", f"{label} does not record every identity field", evidence,
            )
        return ExperimentServerApplication._gate(
            gate_id, "PASS", f"{label} identity matches", evidence,
        )

    def run_attempts(self, project: str, run_id: str) -> dict[str, Any]:
        """List exact Attempt identities without expanding collected evidence."""
        _, _, row = self.resolve_scope(project, AgentScopeType.RUN, run_id)
        current = preferred_attempt_id(Path(row.run_dir))
        attempts = [{
            "identity": f"{row.run_id}::{item.attempt_id}",
            "attempt_id": item.attempt_id,
            "current": item.attempt_id == current,
            "backend": item.backend,
            "backend_job_id": item.backend_job_id,
            "state": item.state,
            "decision": item.decision.get("action") if item.decision else None,
            "has_submission": item.has_submission,
        } for item in row.attempts]
        return {
            "project": project, "run_id": row.run_id,
            "current_attempt_id": current, "count": len(attempts),
            "attempts": attempts,
        }

    @staticmethod
    def _manifest_at(root: Path, names: tuple[str, ...]) -> tuple[dict[str, Any], Path | None]:
        for name in names:
            path = root / name
            payload = (
                ExperimentServerApplication._read_mapping(path)
                if path.suffix == ".json"
                else ExperimentServerApplication._read_yaml_mapping(path)
            )
            if payload:
                return payload, path
        return {}, None

    def _attempt_validation_gates(self, project: str, row: Any, attempt: Any,
                                  attempt_dir: Path, *, require_current: bool) -> list[dict[str, Any]]:
        gates: list[dict[str, Any]] = []
        run_dir = Path(row.run_dir)
        attempt_id = attempt.attempt_id
        attempt_manifest, attempt_manifest_path = self._manifest_at(
            attempt_dir, ("attempt.yaml", "attempt.json", "control_attempt.yaml"),
        )
        run_manifest, _ = self._manifest_at(
            run_dir, ("manifest.yaml", "manifest.json", "collected_run/manifest.yaml",
                      "control_manifest.yaml"),
        )
        harness_layout = bool(
            attempt_manifest_path and attempt_manifest_path.name == "attempt.json"
            and (run_dir / "manifest.json").is_file()
        )
        contract = run_manifest.get("research_contract")
        contract_source: dict[str, Any] = {"kind": "run_manifest"}
        if not isinstance(contract, dict):
            candidates = []
            configured_project = self.runtime.project(project)
            for binding in row.campaign_memberships:
                ref = next(
                    (item for item in configured_project.campaigns
                     if item.name == binding.campaign), None,
                )
                revision = ref.current_revision if ref else None
                if revision and isinstance(revision.research_contract, dict):
                    candidates.append((binding, revision))
            if len(candidates) == 1:
                binding, revision = candidates[0]
                contract = revision.research_contract
                contract_source = {
                    "kind": "current_campaign_revision",
                    "campaign": binding.campaign,
                    "revision_id": revision.revision_id,
                    "frozen_in_run": False,
                }
        identity_payload = dict(attempt_manifest)
        if harness_layout and identity_payload.get("project") is None:
            # The harness freezes project identity in the parent Run manifest;
            # attempt.json deliberately repeats only run_id and attempt_id.
            identity_payload["project"] = run_manifest.get("project")
        gates.append(self._identity_gate(
            gate_id="attempt.identity",
            label="Attempt record" if harness_layout else "Attempt manifest",
            payload=identity_payload,
            expected={"project": project, "run_id": row.run_id,
                      "attempt_id": attempt_id},
        ))
        gates[-1]["evidence"]["source"] = (
            str(attempt_manifest_path) if attempt_manifest_path else None
        )

        immutable_keys = ("source_id", "image_id", "config_path", "seed", "campaign")
        compared: dict[str, Any] = {}
        missing_immutable: list[str] = []
        provenance_conflicts: dict[str, Any] = {}
        for key in immutable_keys:
            run_value = run_manifest.get(key)
            attempt_value = attempt_manifest.get(key)
            if key == "seed":
                run_resolved = run_manifest.get("resolved_config")
                attempt_resolved = attempt_manifest.get("resolved_config")
                run_value = run_value if run_value is not None else (
                    run_resolved.get(key) if isinstance(run_resolved, dict) else None
                )
                attempt_value = attempt_value if attempt_value is not None else (
                    attempt_resolved.get(key) if isinstance(attempt_resolved, dict) else None
                )
            compared[key] = {"run": run_value, "attempt": attempt_value}
            if run_value is None or attempt_value is None:
                missing_immutable.append(key)
            elif run_value != attempt_value:
                provenance_conflicts[key] = compared[key]
        harness_summary = self._read_mapping(attempt_dir / "summary.json") if harness_layout else {}
        harness_integrity = harness_summary.get("integrity")
        harness_integrity = harness_integrity if isinstance(harness_integrity, dict) else {}
        failed_integrity = {
            key: value for key, value in harness_integrity.items() if value is not True
        }
        if harness_layout and failed_integrity:
            gates.append(self._gate(
                "attempt.immutable_provenance", "BLOCKED",
                "Harness source-integrity checks failed",
                {"integrity": harness_integrity},
            ))
        elif harness_layout and harness_integrity:
            gates.append(self._gate(
                "attempt.immutable_provenance", "PASS",
                "Harness verifies the immutable source snapshot used by this Attempt",
                {"integrity": harness_integrity,
                 "source": str(attempt_dir / "summary.json")},
            ))
        elif provenance_conflicts:
            gates.append(self._gate(
                "attempt.immutable_provenance", "BLOCKED",
                "Attempt changes immutable Run provenance", {"fields": provenance_conflicts},
            ))
        elif missing_immutable:
            gates.append(self._gate(
                "attempt.immutable_provenance", "UNKNOWN",
                "Run/Attempt provenance cannot be fully compared",
                {"missing_fields": missing_immutable, "fields": compared},
            ))
        else:
            gates.append(self._gate(
                "attempt.immutable_provenance", "PASS",
                "Attempt preserves immutable Run provenance", {"fields": compared},
            ))

        current = preferred_attempt_id(run_dir)
        if not require_current:
            current_status = "PASS"
            current_message = "exact Attempt exists; current status is informational"
        elif current is None:
            current_status = "UNKNOWN"
            current_message = "current Attempt cannot be resolved"
        elif current != attempt_id:
            current_status = "BLOCKED"
            current_message = "Run evidence selects a different current Attempt"
        else:
            current_status = "PASS"
            current_message = "Attempt is the Run's current evidence source"
        gates.append(self._gate(
            "attempt.current", current_status, current_message,
            {"expected_attempt_id": attempt_id, "current_attempt_id": current},
        ))

        status = self._read_mapping(attempt_dir / "status.json")
        backend = self._read_mapping(attempt_dir / "backend.json")
        collection = self._read_mapping(attempt_dir / "collection.json")
        decision = self._read_mapping(attempt_dir / "decision.json")
        submission = self._read_mapping(attempt_dir / "submission.json")
        if harness_layout:
            status = attempt_manifest
            backend = submission
            collection = harness_summary
            identity_sources = {
                "attempt": attempt_manifest,
                "submission": submission,
                "summary": harness_summary,
            }
        else:
            identity_sources = {
                "status": status, "backend": backend,
                "collection": collection, "decision": decision,
            }
        conflicts: dict[str, Any] = {}
        unknown_sources: list[str] = []
        for name, payload in identity_sources.items():
            if not payload:
                unknown_sources.append(name)
                continue
            observed = payload.get("attempt_id")
            if observed is None:
                unknown_sources.append(name)
            elif observed != attempt_id:
                conflicts[name] = observed
        if conflicts:
            gates.append(self._gate(
                "attempt.evidence_identity", "BLOCKED",
                "Attempt-local evidence names a different Attempt", {"conflicts": conflicts},
            ))
        elif unknown_sources:
            gates.append(self._gate(
                "attempt.evidence_identity", "UNKNOWN",
                "some Attempt-local evidence is missing or unscoped",
                {"unknown_sources": unknown_sources},
            ))
        else:
            gates.append(self._gate(
                "attempt.evidence_identity", "PASS",
                "all Attempt-local evidence is scoped to the exact Attempt",
            ))

        job_ids = {
            name: payload.get("backend_job_id")
            for name, payload in (("status", status), ("backend", backend),
                                  ("collection", collection))
            if payload.get("backend_job_id") is not None
        }
        unique_job_ids = {str(value) for value in job_ids.values()}
        expected_job = attempt.backend_job_id
        if harness_layout and submission.get("gpu") is not None and not unique_job_ids:
            gates.append(self._gate(
                "attempt.backend_job_id", "PASS",
                "Local CUDA execution has no scheduler job identity",
                {"backend": "local-cuda", "gpu": submission.get("gpu")},
            ))
        elif len(unique_job_ids) > 1 or (
            expected_job is not None and unique_job_ids
            and str(expected_job) not in unique_job_ids
        ):
            gates.append(self._gate(
                "attempt.backend_job_id", "BLOCKED",
                "backend_job_id conflicts across Attempt evidence",
                {"index": expected_job, "sources": job_ids},
            ))
        elif expected_job is None or not unique_job_ids:
            gates.append(self._gate(
                "attempt.backend_job_id", "UNKNOWN",
                "backend_job_id is not available in both index and Attempt evidence",
                {"index": expected_job, "sources": job_ids},
            ))
        else:
            gates.append(self._gate(
                "attempt.backend_job_id", "PASS", "backend_job_id is consistent",
                {"backend_job_id": expected_job, "sources": sorted(job_ids)},
            ))

        if harness_layout:
            root_status = self._read_mapping(run_dir / "status.json")
            summary_metrics = harness_summary.get("metrics")
            states = {
                "scheduler": root_status.get("state"),
                "process": attempt_manifest.get("state"),
                "model": "OBSERVED" if isinstance(summary_metrics, dict) and summary_metrics else None,
            }
        else:
            states = {
                "scheduler": status.get("state"),
                "process": collection.get("process_state"),
                "model": collection.get("model_state"),
            }
        terminal = (status.get("state") or attempt.state or "").upper()
        terminal_success = terminal in {"SUCCEEDED", "COMPLETED"}
        if any(value is None for value in states.values()):
            gates.append(self._gate(
                "attempt.execution_layers", "UNKNOWN",
                "scheduler/process/model state is not fully observed", {"states": states},
            ))
        elif terminal_success and str(states["process"]).upper() not in {"SUCCEEDED", "COMPLETED"}:
            gates.append(self._gate(
                "attempt.execution_layers", "BLOCKED",
                "scheduler succeeded but process evidence does not", {"states": states},
            ))
        else:
            gates.append(self._gate(
                "attempt.execution_layers", "PASS",
                "scheduler/process/model evidence is present and non-conflicting",
                {"states": states},
            ))

        records, metric_source, source_attempt_id = train_metric_records(
            run_dir, attempt_id=attempt_id, exact_attempt=True,
        )
        if source_attempt_id != attempt_id:
            gates.append(self._gate(
                "attempt.model_evidence", "BLOCKED",
                "model evidence resolves to a different Attempt",
                {"expected": attempt_id, "source_attempt_id": source_attempt_id,
                 "source": str(metric_source) if metric_source else None},
            ))
        elif not records:
            gates.append(self._gate(
                "attempt.model_evidence", "UNKNOWN", "no exact-Attempt model metrics found",
                {"source_attempt_id": source_attempt_id},
            ))
        else:
            gates.append(self._gate(
                "attempt.model_evidence", "PASS", "exact-Attempt model metrics are readable",
                {"source_attempt_id": source_attempt_id,
                 "source": str(metric_source), "records": len(records)},
            ))

        checkpoint = collection.get("latest_completed_checkpoint")
        checkpoint_step = collection.get("latest_completed_checkpoint_step")
        local_checkpoints = sorted(
            path.name for path in (attempt_dir / "collected_run").glob("checkpoint*")
        )
        checkpoint_policy = contract.get("checkpoint") if isinstance(contract, dict) else None
        checkpoint_required = (
            isinstance(checkpoint_policy, dict)
            and checkpoint_policy.get("required_on_success") is True
        )
        if checkpoint and checkpoint_step is not None:
            gates.append(self._gate(
                "attempt.checkpoint_evidence", "PASS",
                "completed checkpoint is recorded for the exact Attempt",
                {"path": checkpoint, "step": checkpoint_step,
                 "local_entries": local_checkpoints[:20]},
            ))
        elif checkpoint_required:
            gates.append(self._gate(
                "attempt.checkpoint_evidence", "UNKNOWN",
                "completed checkpoint path or step is missing",
                {"path": checkpoint, "step": checkpoint_step,
                 "local_entries": local_checkpoints[:20]},
            ))
        else:
            gates.append(self._gate(
                "attempt.checkpoint_evidence", "PASS",
                "a completed checkpoint is not required by the applicable research contract",
                {"path": checkpoint, "step": checkpoint_step,
                 "local_entries": local_checkpoints[:20]},
            ))

        artifacts = collection.get("artifacts")
        if harness_layout and harness_summary:
            artifacts = {
                "summary": {"records": 1, "nonempty_records": 1},
                "integrity": {
                    "records": len(harness_integrity),
                    "nonempty_records": sum(value is True for value in harness_integrity.values()),
                },
            }
        if not isinstance(artifacts, dict) or not artifacts:
            gates.append(self._gate(
                "attempt.artifact_evidence", "UNKNOWN",
                "artifact summary is missing for the exact Attempt",
            ))
        else:
            evidence_records = sum(
                int(item.get("records") or 0) for item in artifacts.values()
                if isinstance(item, dict)
            )
            status_value = "PASS" if evidence_records > 0 else "UNKNOWN"
            gates.append(self._gate(
                "attempt.artifact_evidence", status_value,
                "artifact summary contains records" if evidence_records > 0
                else "artifact summary contains no records",
                {"records": evidence_records, "groups": sorted(artifacts)},
            ))

        evidence_conflicts = collection.get("evidence_conflicts")
        evidence_conflicts = evidence_conflicts if isinstance(evidence_conflicts, list) else []
        gates.append(self._gate(
            "attempt.evidence_conflicts",
            "BLOCKED" if evidence_conflicts else "PASS",
            "scientific evidence contains conflicting values" if evidence_conflicts
            else "no scientific evidence conflicts are recorded",
            {"count": len(evidence_conflicts), "conflicts": evidence_conflicts[:50]},
        ))

        if not isinstance(contract, dict):
            gates.append(self._gate(
                "attempt.scientific_contract", "UNKNOWN",
                "Run has no frozen research_contract",
            ))
        else:
            required = contract.get("required_metrics")
            required = required if isinstance(required, dict) else {}
            required_names = set(required.get("common") or [])
            by_role = required.get("by_role")
            if isinstance(by_role, dict) and row.role:
                required_names.update(by_role.get(row.role) or [])
            available_names = set(records[-1]) if records else set()
            metric_values: dict[str, list[Any]] = {}
            if records:
                for key, value in records[-1].items():
                    metric_values.setdefault(key, []).append(value)
            variants, _ = evaluation_variants(
                run_dir, attempt_id=attempt_id, exact_attempt=True,
            )
            for variant in variants:
                latest = variant.get("latest")
                if isinstance(latest, dict):
                    available_names.update(latest)
                    for key, value in latest.items():
                        metric_values.setdefault(key, []).append(value)
            missing_metrics = sorted(required_names - available_names)

            terminal_results: list[dict[str, Any]] = []
            terminal_failures: list[dict[str, Any]] = []
            unresolved_checks: list[dict[str, Any]] = []
            operators = {
                "lt": lambda left, right: left < right,
                "le": lambda left, right: left <= right,
                "lte": lambda left, right: left <= right,
                "gt": lambda left, right: left > right,
                "ge": lambda left, right: left >= right,
                "gte": lambda left, right: left >= right,
                "eq": lambda left, right: left == right,
            }

            def unique_metric(name: Any) -> tuple[Any, str | None]:
                values = metric_values.get(str(name), [])
                unique = []
                for value in values:
                    if value not in unique:
                        unique.append(value)
                if not unique:
                    return None, "missing"
                if len(unique) > 1:
                    return None, "ambiguous_across_eval_variants"
                return unique[0], None

            for index, check in enumerate(contract.get("terminal_checks") or []):
                if not isinstance(check, dict):
                    unresolved_checks.append({"index": index, "reason": "invalid_check"})
                    continue
                roles = check.get("roles")
                if isinstance(roles, list) and row.role not in roles:
                    continue
                op = check.get("op")
                if op not in {*operators, "finite", "nonfinite"}:
                    unresolved_checks.append({"index": index, "reason": "unsupported_op", "op": op})
                    continue
                left_name = check.get("left") or check.get("metric")
                left, left_error = unique_metric(left_name)
                if op in {"finite", "nonfinite"}:
                    if left_error:
                        unresolved_checks.append({
                            "index": index, "left": left_name, "reason": left_error,
                        })
                        continue
                    try:
                        is_finite = isinstance(left, (int, float)) and math.isfinite(float(left))
                    except (TypeError, ValueError):
                        is_finite = False
                    passed = is_finite if op == "finite" else not is_finite
                    result = {"index": index, "op": op, "left": left,
                              "right": None, "passed": passed}
                    terminal_results.append(result)
                    if not passed:
                        terminal_failures.append(result)
                    continue
                if check.get("right") is not None:
                    right, right_error = unique_metric(check.get("right"))
                else:
                    right, right_error = check.get("value"), None
                if left_error or right_error or right is None:
                    unresolved_checks.append({
                        "index": index, "left": left_name, "right": check.get("right"),
                        "reason": left_error or right_error or "missing_threshold",
                    })
                    continue
                try:
                    passed = bool(operators[op](left, right))
                except TypeError:
                    unresolved_checks.append({"index": index, "reason": "non_comparable_values"})
                    continue
                result = {"index": index, "op": op, "left": left, "right": right,
                          "passed": passed}
                terminal_results.append(result)
                if not passed:
                    terminal_failures.append(result)

            required_artifacts = contract.get("required_artifacts")
            required_artifacts = required_artifacts if isinstance(required_artifacts, dict) else {}
            common_artifacts = required_artifacts.get("common")
            common_artifacts = common_artifacts if isinstance(common_artifacts, dict) else {}
            missing_artifacts: list[dict[str, Any]] = []
            artifact_payload = artifacts if isinstance(artifacts, dict) else {}
            for name, rule in common_artifacts.items():
                observed = artifact_payload.get(name)
                observed = observed if isinstance(observed, dict) else {}
                rule = rule if isinstance(rule, dict) else {}
                for key, threshold_key in (
                    ("records", "min_records"),
                    ("nonempty_records", "min_nonempty_records"),
                ):
                    minimum = rule.get(threshold_key)
                    if isinstance(minimum, (int, float)) and (observed.get(key) or 0) < minimum:
                        missing_artifacts.append({
                            "artifact": name, "field": key,
                            "required": minimum, "observed": observed.get(key) or 0,
                        })
            if terminal_failures:
                contract_status = "BLOCKED"
            elif missing_metrics or missing_artifacts or unresolved_checks:
                contract_status = "UNKNOWN"
            else:
                contract_status = "PASS"
            gates.append(self._gate(
                "attempt.scientific_contract", contract_status,
                "required scientific evidence is present" if contract_status == "PASS"
                else "required scientific evidence is incomplete",
                {
                    "required_metrics": sorted(required_names),
                    "missing_metrics": missing_metrics,
                    "missing_artifacts": missing_artifacts,
                    "terminal_checks_declared": len(contract.get("terminal_checks") or []),
                    "terminal_results": terminal_results,
                    "terminal_failures": terminal_failures,
                    "unresolved_checks": unresolved_checks,
                    "contract_source": contract_source,
                },
            ))
        return gates

    def attempt_validate(self, project: str, identity: str) -> dict[str, Any]:
        _, _, row, attempt, attempt_dir = self._attempt_context(project, identity)
        gates = self._attempt_validation_gates(
            project, row, attempt, attempt_dir, require_current=False,
        )
        return self._validation_payload(
            object_type="attempt", identity=identity, gates=gates,
        )

    def run_validate(self, project: str, run_id: str) -> dict[str, Any]:
        _, _, row = self.resolve_scope(project, AgentScopeType.RUN, run_id)
        run_dir = Path(row.run_dir)
        manifest, manifest_path = self._manifest_at(
            run_dir, ("manifest.yaml", "collected_run/manifest.yaml", "control_manifest.yaml"),
        )
        gates = [self._identity_gate(
            gate_id="run.identity", label="Run manifest", payload=manifest,
            expected={"project": project, "run_id": row.run_id,
                      "campaign": row.campaign},
        )]
        gates[-1]["evidence"]["source"] = str(manifest_path) if manifest_path else None

        relationship = row.campaign_binding.relationship
        blocking_relationships = {
            CampaignRelationship.DUPLICATE_RUN_ID,
            CampaignRelationship.PROJECT_MISMATCH,
            CampaignRelationship.ROLE_MISMATCH,
            CampaignRelationship.UNDECLARED_RUN,
        }
        relationship_status = (
            "PASS" if relationship == CampaignRelationship.MATCHED
            else "BLOCKED" if relationship in blocking_relationships
            else "UNKNOWN"
        )
        gates.append(self._gate(
            "run.campaign_binding",
            relationship_status,
            "Run matches its authored Campaign revision"
            if relationship_status == "PASS"
            else "Run-to-Campaign relationship requires reconciliation",
            row.campaign_binding.model_dump(mode="json"),
        ))

        resolved = manifest.get("resolved_config")
        seed = manifest.get("seed")
        if seed is None and isinstance(resolved, dict):
            seed = resolved.get("seed")
        provenance = {
            "source_id": manifest.get("source_id") or manifest.get("git_commit"),
            "image_id": manifest.get("image_id"),
            "config_path": manifest.get("config_path"),
            "seed": seed,
        }
        missing = [key for key, value in provenance.items() if value is None]
        gates.append(self._gate(
            "run.provenance", "UNKNOWN" if missing else "PASS",
            "immutable provenance is incomplete" if missing
            else "source, image, config, and seed provenance is recorded",
            {"provenance": provenance, "missing_fields": missing},
        ))

        current = preferred_attempt_id(run_dir)
        attempt = next((item for item in row.attempts if item.attempt_id == current), None)
        if current is None or attempt is None:
            gates.append(self._gate(
                "run.current_attempt", "UNKNOWN", "current Attempt cannot be resolved",
                {"current_attempt_id": current,
                 "indexed_attempt_ids": [item.attempt_id for item in row.attempts]},
            ))
        else:
            gates.append(self._gate(
                "run.current_attempt", "PASS", "current Attempt resolves exactly",
                {"current_attempt_id": current,
                 "identity": f"{row.run_id}::{current}"},
            ))
            gates.extend(self._attempt_validation_gates(
                project, row, attempt, run_dir / "attempts" / current,
                require_current=True,
            ))

        return self._validation_payload(
            object_type="run", identity=row.run_id, gates=gates,
        )

    def attempt_show(self, project: str, identity: str) -> dict[str, Any]:
        scope, _, row, attempt, attempt_dir = self._attempt_context(project, identity)
        collection = self._read_mapping(attempt_dir / "collection.json")
        decision = self._read_mapping(attempt_dir / "decision.json")
        if not collection and (attempt_dir / "attempt.json").is_file():
            attempt_record = self._read_mapping(attempt_dir / "attempt.json")
            summary = self._read_mapping(attempt_dir / "summary.json")
            root_status = self._read_mapping(Path(row.run_dir) / "status.json")
            summary_metrics = summary.get("metrics")
            collection = {
                "attempt_id": attempt.attempt_id,
                "scheduler_state": root_status.get("state"),
                "worker_state": "LOCAL_GPU",
                "process_state": attempt_record.get("state"),
                "model_state": (
                    "OBSERVED" if isinstance(summary_metrics, dict) and summary_metrics else None
                ),
                "evaluation_state": (
                    "OBSERVED" if isinstance(summary_metrics, dict)
                    and summary_metrics.get("val_bpb") is not None else None
                ),
                "failure_class": attempt_record.get("failure_class"),
            }
        return {
            "scope": scope.model_dump(mode="json"),
            "attempt": attempt.model_dump(mode="json"),
            "run_id": row.run_id,
            "run_dir": row.run_dir,
            "attempt_dir": str(attempt_dir),
            "failure_summary": structured_failure_summary(collection, decision),
            "collection": compact_evidence(collection),
        }

    def attempt_logs(self, project: str, identity: str, *, stream: str = "both",
                     lines: int = 80) -> dict[str, Any]:
        if lines <= 0 or lines > 10000:
            raise ApplicationError("lines must be between 1 and 10000",
                                   status_code=422, code="INVALID_LINE_COUNT")
        _, _, row, attempt, attempt_dir = self._attempt_context(project, identity)
        collection = self._read_mapping(attempt_dir / "collection.json")
        process = collection.get("process_evidence")
        process = process if isinstance(process, dict) else {}
        wanted = ("stdout", "stderr") if stream == "both" else (stream,)
        result: dict[str, Any] = {}
        for name in wanted:
            candidates = (
                attempt_dir / f"{name}.log",
                attempt_dir / "collected_run" / f"{name}.log",
            )
            local = next((path for path in candidates if path.is_file()), None)
            if local is not None:
                content = local.read_text(encoding="utf-8", errors="replace").splitlines()
                result[name] = {
                    "source": str(local), "mode": "local_file",
                    "lines": content[-lines:], "available_lines": len(content),
                }
                continue
            archived = process.get(f"{name}_tail")
            archived = archived if isinstance(archived, list) else []
            sources = process.get("sources") if isinstance(process.get("sources"), dict) else {}
            result[name] = {
                "source": sources.get(name), "mode": "collected_tail",
                "lines": [str(item) for item in archived[-lines:]],
                "available_lines": len(archived),
            }
        return {
            "project": project, "run_id": row.run_id,
            "attempt_id": attempt.attempt_id, "streams": result,
            "follow_supported": any(item["mode"] == "local_file" for item in result.values()),
        }

    def attempt_checkpoints(self, project: str, identity: str) -> dict[str, Any]:
        _, _, row, attempt, attempt_dir = self._attempt_context(project, identity)
        collection = self._read_mapping(attempt_dir / "collection.json")
        latest = collection.get("latest_completed_checkpoint") or row.checkpoint.get(
            "latest_completed_checkpoint"
        )
        step = collection.get("latest_completed_checkpoint_step") or row.checkpoint.get(
            "latest_completed_checkpoint_step"
        )
        local_root = attempt_dir / "collected_run"
        local = []
        if local_root.is_dir():
            local = [str(path.relative_to(attempt_dir)) for path in sorted(
                local_root.glob("checkpoint*"))[:200]
            ]
        return {
            "project": project, "run_id": row.run_id,
            "attempt_id": attempt.attempt_id,
            "latest_completed_checkpoint": latest,
            "latest_completed_checkpoint_step": step,
            "local_entries": local,
        }

    def attempt_artifacts(self, project: str, identity: str) -> dict[str, Any]:
        _, _, row, attempt, attempt_dir = self._attempt_context(project, identity)
        collection = self._read_mapping(attempt_dir / "collection.json")
        artifacts = collection.get("artifacts")
        if not isinstance(artifacts, dict):
            artifacts = row.artifacts
        roots = []
        collected = attempt_dir / "collected_run"
        if collected.is_dir():
            roots = [str(path.relative_to(attempt_dir)) for path in sorted(
                item for item in collected.iterdir() if item.is_dir()
            )]
        return {
            "project": project, "run_id": row.run_id,
            "attempt_id": attempt.attempt_id,
            "summary": artifacts, "local_roots": roots,
        }

    def attempt_metrics(self, project: str, identity: str, *, keys: str | None = None,
                        max_points: int = 2000) -> dict[str, Any]:
        _, _, row, attempt, _ = self._attempt_context(project, identity)
        records, source, source_attempt_id = train_metric_records(
            Path(row.run_dir), attempt_id=attempt.attempt_id, exact_attempt=True,
        )
        payload = self._metric_payload(
            records, keys=keys, max_points=max_points, source=source,
            source_attempt_id=source_attempt_id,
        )
        return {"project": project, "run_id": row.run_id,
                "attempt_id": attempt.attempt_id, **payload}

    def attempt_eval(self, project: str, identity: str) -> dict[str, Any]:
        _, _, row, attempt, _ = self._attempt_context(project, identity)
        variants, source_attempt_id = evaluation_variants(
            Path(row.run_dir), attempt_id=attempt.attempt_id, exact_attempt=True,
        )
        return {"project": project, "run_id": row.run_id,
                "attempt_id": attempt.attempt_id,
                "source_attempt_id": source_attempt_id, "variants": variants}

    def attempt_events(self, project: str, identity: str) -> dict[str, Any]:
        _, _, row, attempt, attempt_dir = self._attempt_context(project, identity)
        events: list[dict[str, Any]] = []
        sources: list[str] = []
        seen: set[str] = set()
        candidates = (
            attempt_dir / "collected_run" / "events.jsonl",
            attempt_dir / "events.jsonl",
            Path(row.run_dir) / "events.jsonl",
        )
        for candidate in candidates:
            records = read_jsonl(candidate)
            if not records:
                continue
            is_run_timeline = candidate == Path(row.run_dir) / "events.jsonl"
            added = False
            for event in records:
                event_attempt = event.get("attempt_id")
                if event_attempt != attempt.attempt_id and (
                    is_run_timeline or event_attempt is not None
                ):
                    continue
                identity_key = json.dumps(event, sort_keys=True, default=str)
                if identity_key in seen:
                    continue
                seen.add(identity_key)
                events.append(event)
                added = True
            if added:
                sources.append(str(candidate))
        for event in events:
            event["ts"] = parse_iso_ts(event.get("timestamp"))
        events.sort(key=lambda event: event.get("ts") or 0)
        return {"project": project, "run_id": row.run_id,
                "attempt_id": attempt.attempt_id,
                "sources": sources, "events": events}

    @staticmethod
    def _campaign_file(project: ResearchProject, campaign_name: str | None) -> Path:
        reference = next((
            campaign for campaign in project.campaigns if campaign.name == campaign_name
        ), None)
        if reference is None or not reference.file:
            raise ApplicationError("attempt run has no authored campaign file",
                                   code="CAMPAIGN_FILE_MISSING")
        path = Path(reference.file)
        if not path.is_absolute():
            path = Path(project.base_dir or ".") / path
        return path.resolve()

    def propose_run_submit(self, project: str, run_id: str, *,
                           max_gpu_hours: float, reason: str = "") -> dict[str, Any]:
        """Create a reviewable first-submission proposal for an authored Run."""
        self._require_operation_available(
            "run.submit", project, AgentScopeType.RUN, run_id,
        )
        if max_gpu_hours <= 0:
            raise ApplicationError("max_gpu_hours must be positive", status_code=422,
                                   code="INVALID_GPU_BUDGET")
        scope, configured, row = self.resolve_scope(project, AgentScopeType.RUN, run_id)
        if configured.controller is None:
            raise ApplicationError("project has no controller configuration",
                                   code="CONTROLLER_UNAVAILABLE")
        state = (row.scheduler_state or "NOT_SUBMITTED").upper()
        if state in {
            "SUBMITTING", "PENDING", "QUEUED", "STARTING", "RUNNING", "EVALUATING",
            "SUCCEEDED", "FAILED", "PREEMPTED", "CANCELLED",
        }:
            raise ApplicationError(
                f"run state {state} is not eligible for first submission",
                code="RUN_NOT_SUBMITTABLE",
            )
        if any(item.has_submission for item in row.attempts):
            raise ApplicationError(
                "Run already has submitted Attempt evidence; use exact Attempt retry when eligible",
                code="RUN_ALREADY_SUBMITTED",
            )
        authored = [
            binding for binding in row.campaign_memberships
            if binding.membership.kind == "materialize"
        ]
        campaign_name = authored[0].campaign if len(authored) == 1 else row.campaign
        campaign = self._campaign_file(configured, campaign_name)
        existing = [item.attempt_id for item in row.attempts]
        attempt_id = existing[-1] if existing else "attempt-001"
        draft = yaml.safe_dump({
            "campaign_file": str(campaign), "run_id": row.run_id,
            "attempt_id": attempt_id, "max_gpu_hours": max_gpu_hours,
        }, sort_keys=False)
        store = self.runtime.agent_store
        store.ensure(scope, default_goal=self.default_goal(scope, row))
        digest = evidence_digest(self.bounded_evidence(scope, configured, row))
        created = store.add_proposals(scope, [{
            "kind": "SUBMIT_RUN",
            "title": f"Launch {row.run_id} as {attempt_id}",
            "target": f"run://{configured.project}/{row.run_id}",
            "change_summary": reason or "submit the authored Run's first Attempt",
            "resource_estimate": f"up to {max_gpu_hours:g} GPU-hours",
            "rationale": "materialize an authored Campaign membership",
            "risk": "scheduler mutation; approval and prepared gates are required before execution",
            "draft": draft,
        }], evidence_digest=digest)
        store.set_state(scope, AgentLifecycleState.WAITING_FOR_APPROVAL)
        return {
            "proposal": created[0], "created_proposals": created,
            "agent": self._snapshot(scope, configured, row),
        }

    def propose_attempt_retry(self, project: str, identity: str, *,
                              new_attempt_id: str | None,
                              max_gpu_hours: float, reason: str) -> dict[str, Any]:
        if max_gpu_hours <= 0:
            raise ApplicationError("max_gpu_hours must be positive", status_code=422,
                                   code="INVALID_GPU_BUDGET")
        scope, configured, row, attempt, _ = self._attempt_context(project, identity)
        if (attempt.state or "").upper() not in {"FAILED", "PREEMPTED", "CANCELLED"}:
            raise ApplicationError(
                f"attempt state {attempt.state or 'UNKNOWN'} is not retryable",
                code="ATTEMPT_NOT_RETRYABLE",
            )
        decision = attempt.decision if isinstance(attempt.decision, dict) else {}
        if str(decision.get("action") or "").upper() == "DO_NOT_RETRY":
            raise ApplicationError(
                "attempt decision explicitly forbids retry",
                code="ATTEMPT_RETRY_FORBIDDEN",
            )
        allowed = decision.get("retries_allowed")
        used = decision.get("retries_used", 0)
        if isinstance(allowed, int) and isinstance(used, int) and used >= allowed:
            raise ApplicationError(
                f"attempt retry budget exhausted: used={used}, allowed={allowed}",
                code="ATTEMPT_RETRY_BUDGET_EXHAUSTED",
            )
        self._require_operation_available(
            "attempt.retry", project, AgentScopeType.ATTEMPT, identity,
        )
        existing = {item.attempt_id for item in row.attempts}
        if new_attempt_id is None:
            numbers = [int(match.group(1)) for item in existing
                       if (match := re.fullmatch(r"attempt-(\d+)", item))]
            new_attempt_id = f"attempt-{max(numbers, default=0) + 1:03d}"
        if not re.fullmatch(r"attempt-[0-9]{3,}", new_attempt_id):
            raise ApplicationError("new attempt_id must match attempt-NNN",
                                   code="INVALID_ATTEMPT_ID")
        if new_attempt_id in existing:
            raise ApplicationError("new attempt_id already exists",
                                   code="DUPLICATE_ATTEMPT_ID")
        campaign = self._campaign_file(configured, row.campaign)
        draft = yaml.safe_dump({
            "campaign_file": str(campaign), "run_id": row.run_id,
            "source_attempt_id": attempt.attempt_id,
            "attempt_id": new_attempt_id, "max_gpu_hours": max_gpu_hours,
        }, sort_keys=False)
        return self._create_attempt_proposal(
            scope, configured, row, attempt, kind="RETRY_ATTEMPT", draft=draft,
            title=f"Retry {row.run_id} from {attempt.attempt_id} as {new_attempt_id}",
            change_summary=reason or "retry failed attempt with a new attempt identity",
            resource_estimate=f"up to {max_gpu_hours:g} GPU-hours",
            risk="scheduler mutation; review checkpoint and failure classification before approval",
        )

    def propose_attempt_cancel(self, project: str, identity: str, *,
                               reason: str) -> dict[str, Any]:
        scope, configured, row, attempt, _ = self._attempt_context(project, identity)
        if (attempt.state or "").upper() not in {
            "SUBMITTING", "PENDING", "QUEUED", "STARTING", "RUNNING", "EVALUATING",
        }:
            raise ApplicationError(
                f"attempt state {attempt.state or 'UNKNOWN'} is not cancellable",
                code="ATTEMPT_NOT_CANCELLABLE",
            )
        if not attempt.backend_job_id:
            raise ApplicationError("attempt has no backend_job_id",
                                   code="BACKEND_JOB_ID_MISSING")
        self._require_operation_available(
            "attempt.cancel", project, AgentScopeType.ATTEMPT, identity,
        )
        campaign = self._campaign_file(configured, row.campaign)
        draft = yaml.safe_dump({
            "campaign_file": str(campaign), "run_id": row.run_id,
            "attempt_id": attempt.attempt_id,
            "backend_job_id": attempt.backend_job_id,
        }, sort_keys=False)
        return self._create_attempt_proposal(
            scope, configured, row, attempt, kind="CANCEL_RUN", draft=draft,
            title=f"Cancel {row.run_id} {attempt.attempt_id}",
            change_summary=reason or "cancel the exact observed backend job",
            resource_estimate="none", risk="scheduler cancellation",
        )

    def _create_attempt_proposal(
        self, scope: AgentScope, configured: ResearchProject, row: Any, attempt: Any,
        *, kind: str, draft: str, title: str, change_summary: str,
        resource_estimate: str, risk: str,
    ) -> dict[str, Any]:
        store = self.runtime.agent_store
        store.ensure(scope, default_goal=self.default_goal(scope, attempt))
        digest = evidence_digest(self.bounded_evidence(scope, configured, attempt))
        created = store.add_proposals(scope, [{
            "kind": kind, "title": title,
            "target": f"attempt://{configured.project}/{scope.object_id}",
            "change_summary": change_summary,
            "resource_estimate": resource_estimate,
            "rationale": f"operate on immutable attempt evidence for {row.run_id}",
            "risk": risk, "draft": draft,
        }], evidence_digest=digest)
        store.set_state(scope, AgentLifecycleState.WAITING_FOR_APPROVAL)
        return {"proposal": created[0],
                "agent": self._snapshot(scope, configured, attempt)}

    def run_detail(self, project: str, run_id: str) -> dict[str, Any]:
        _, _, row = self.resolve_scope(project, AgentScopeType.RUN, run_id)
        payload = row.model_dump()
        payload["is_terminal"] = row.is_terminal
        return payload

    def run_metrics(self, project: str, run_id: str, *, keys: str | None = None,
                    max_points: int = 2000) -> dict[str, Any]:
        _, _, row = self.resolve_scope(project, AgentScopeType.RUN, run_id)
        records, source, source_attempt_id = train_metric_records(Path(row.run_dir))
        return self._metric_payload(
            records, keys=keys, max_points=max_points, source=source,
            source_attempt_id=source_attempt_id,
        )

    @staticmethod
    def _metric_payload(records: list[dict[str, Any]], *, keys: str | None,
                        max_points: int, source: Path | None,
                        source_attempt_id: str | None) -> dict[str, Any]:
        if max_points <= 0:
            raise ApplicationError("max_points must be positive", status_code=422,
                                   code="INVALID_MAX_POINTS")
        wanted = [key.strip() for key in keys.split(",") if key.strip()] if keys else None
        total = len(records)
        available = sorted({key for record in records for key, value in record.items()
                            if key not in ("step", "timestamp")
                            and isinstance(value, (int, float))})
        if total <= max_points:
            sampled = records
        elif max_points == 1:
            sampled = [records[-1]]
        else:
            indices = [(index * (total - 1)) // (max_points - 1)
                       for index in range(max_points)]
            sampled = [records[index] for index in indices]
        points = []
        for record in sampled:
            point: dict[str, Any] = {"step": record.get("step"),
                                     "timestamp": record.get("timestamp")}
            for key in wanted or [k for k in record if k not in ("step", "timestamp")]:
                if isinstance(record.get(key), (int, float)):
                    point[key] = record[key]
            points.append(point)
        return {
            "points": points, "keys": wanted or available,
            "missing_keys": [key for key in wanted or [] if key not in available],
            "total_records": total, "downsampled": total > len(sampled),
            "source": str(source) if source else None,
            "source_attempt_id": source_attempt_id,
        }

    def run_eval(self, project: str, run_id: str) -> dict[str, Any]:
        _, _, row = self.resolve_scope(project, AgentScopeType.RUN, run_id)
        variants, source_attempt_id = evaluation_variants(Path(row.run_dir))
        return {"source_attempt_id": source_attempt_id, "variants": variants}

    def run_events(self, project: str, run_id: str) -> dict[str, Any]:
        _, _, row = self.resolve_scope(project, AgentScopeType.RUN, run_id)
        events = read_jsonl(Path(row.run_dir) / "events.jsonl")
        for event in events:
            event["ts"] = parse_iso_ts(event.get("timestamp"))
        return {"events": events}

    # --------------------------------------------------------------- actions

    def list_actions(self, project: str, scope_type: AgentScopeType | str,
                     object_id: str) -> dict[str, Any]:
        scope, _, _ = self.resolve_scope(project, scope_type, object_id)
        return {"actions": self.runtime.action_store.list_for_scope(scope),
                "policy": self.runtime.config.action_runtime.model_dump()}

    def prepare_action(self, project: str, scope_type: AgentScopeType | str,
                       object_id: str, proposal_id: str) -> dict[str, Any]:
        return self._prepare_action_local(project, scope_type, object_id, proposal_id)

    def _prepare_action_local(self, project: str, scope_type: AgentScopeType | str,
                              object_id: str, proposal_id: str) -> dict[str, Any]:
        scope, configured, _ = self.resolve_scope(project, scope_type, object_id)
        try:
            proposal = self.runtime.agent_store.proposal(scope, proposal_id)
            if proposal.get("kind") in {"COMPLETE_CAMPAIGN", "ARCHIVE_CAMPAIGN"}:
                payload = yaml.safe_load(str(proposal.get("draft") or ""))
                payload = payload if isinstance(payload, dict) else {}
                status = campaign_snapshot(self.runtime.index, configured, object_id)
                if payload.get("revision_id") != status.get("revision_id"):
                    raise ApplicationError(
                        "Campaign revision changed after proposal approval",
                        code="OUTDATED_CAMPAIGN_REVISION",
                    )
                if proposal.get("kind") == "COMPLETE_CAMPAIGN" and (
                    status.get("lifecycle_state") != "COMPLETABLE"
                    or payload.get("evidence_digest")
                    != status.get("completion", {}).get("evidence_digest")
                ):
                    raise ApplicationError(
                        "Campaign completion evidence changed or no longer passes",
                        code="OUTDATED_CAMPAIGN_COMPLETION",
                    )
            return self.runtime.action_service.prepare(scope, configured, proposal)
        except ApplicationError:
            raise
        except FileNotFoundError as exc:
            raise ApplicationError("proposal not found", status_code=404,
                                   code="UNKNOWN_PROPOSAL") from exc
        except (RuntimeError, ValueError) as exc:
            raise ApplicationError(str(exc), code="ACTION_BLOCKED") from exc

    def authorize_action(self, action_id: str, note: str = "") -> dict[str, Any]:
        return self._authorize_action_local(action_id, note)

    def _authorize_action_local(self, action_id: str, note: str = "") -> dict[str, Any]:
        try:
            return self.runtime.action_service.authorize(action_id, note)
        except FileNotFoundError as exc:
            raise ApplicationError("action not found", status_code=404,
                                   code="UNKNOWN_ACTION") from exc
        except RuntimeError as exc:
            raise ApplicationError(str(exc), code="ACTION_BLOCKED") from exc

    def execute_action(self, action_id: str, confirmation: str) -> dict[str, Any]:
        return self._execute_action_local(action_id, confirmation)

    def _execute_action_local(self, action_id: str, confirmation: str) -> dict[str, Any]:
        try:
            action = self.runtime.action_store.snapshot(action_id)
            if action.get("proposal_kind") == "COMPLETE_CAMPAIGN":
                scope = action.get("scope") or {}
                configured = self.runtime.project(str(scope.get("project") or ""))
                index_project(self.runtime.index, configured)
                status = campaign_snapshot(
                    self.runtime.index, configured, str(scope.get("object_id") or ""),
                )
                payload = yaml.safe_load(str(action.get("proposed_content") or ""))
                payload = payload if isinstance(payload, dict) else {}
                if status.get("lifecycle_state") != "COMPLETABLE" or (
                    payload.get("evidence_digest")
                    != status.get("completion", {}).get("evidence_digest")
                ):
                    raise ApplicationError(
                        "Campaign evidence changed after authorization; prepare a fresh completion",
                        code="CAMPAIGN_COMPLETION_DRIFT",
                    )
            result = self.runtime.action_service.execute(action_id, confirmation)
        except ApplicationError:
            raise
        except FileNotFoundError as exc:
            raise ApplicationError("action not found", status_code=404,
                                   code="UNKNOWN_ACTION") from exc
        except RuntimeError as exc:
            raise ApplicationError(str(exc), code="ACTION_BLOCKED") from exc
        if result.get("execution", {}).get("status") == "VERIFIED":
            for project in self.runtime.projects:
                index_project(self.runtime.index, project)
        return result

    # ------------------------------------------------------------- projects

    def project_lifecycle_list(self) -> dict[str, Any]:
        """Return daemon-owned registrations, including inactive Projects."""
        return {
            "projects": [record.model_dump(mode="json")
                         for record in self.runtime.project_records()],
            "events": self.runtime.project_registry.events(),
        }

    def project_register(self, project_file: Path) -> dict[str, Any]:
        try:
            project = self.runtime.register_project(
                project_file, source=ProjectRegistrationSource.MANUAL,
            )
            index_project(self.runtime.index, project)
        except (ProjectRegistryError, FileNotFoundError, ValueError) as exc:
            raise ApplicationError(str(exc), code="PROJECT_REGISTRATION_BLOCKED") from exc
        record = next(
            item for item in self.runtime.project_records() if item.project == project.project
        )
        return {
            "project": record.model_dump(mode="json"),
            "effect": "project is active; initial indexing completed and collector observation may run",
        }

    def project_lifecycle_transition(
        self, project: str, target: ProjectLifecycleState, *, reason: str = "",
    ) -> dict[str, Any]:
        if target == ProjectLifecycleState.ARCHIVED and not reason.strip():
            raise ApplicationError(
                "archiving a project requires a reason",
                code="PROJECT_LIFECYCLE_BLOCKED",
            )
        try:
            record = self.runtime.transition_project(project, target, reason=reason)
        except (ProjectRegistryError, FileNotFoundError, ValueError) as exc:
            raise ApplicationError(
                str(exc), status_code=404 if "unknown" in str(exc) else 409,
                code="PROJECT_LIFECYCLE_BLOCKED",
            ) from exc
        return {
            "project": record.model_dump(mode="json"),
            "active": target == ProjectLifecycleState.ACTIVE,
            "effect": (
                "project is active; indexing and collector observation may resume"
                if target == ProjectLifecycleState.ACTIVE
                else "project is inactive; this daemon stops indexing and collecting it"
            ),
        }

    def project_unregister(self, project: str, *, reason: str = "") -> dict[str, Any]:
        try:
            record = self.runtime.unregister_project(project, reason=reason)
        except ProjectRegistryError as exc:
            raise ApplicationError(str(exc), status_code=404, code="UNKNOWN_PROJECT") from exc
        return {
            "unregistered": record.model_dump(mode="json"),
            "effect": "Daemon tracking was removed; repository files, runs, artifacts, and jobs were not changed",
        }

    def project_unregister_all(self, *, reason: str = "") -> dict[str, Any]:
        records = self.runtime.unregister_all_projects(reason=reason)
        return {
            "unregistered": [record.model_dump(mode="json") for record in records],
            "effect": "Daemon tracking was removed; repository files, runs, artifacts, and jobs were not changed",
        }
