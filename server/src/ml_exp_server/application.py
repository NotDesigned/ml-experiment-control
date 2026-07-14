"""Transport-neutral use cases owned by the experiment daemon."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from .authored_runs import authored_run_placeholder
from .application_errors import ApplicationError
from .campaign_lifecycle import campaign_snapshot
from .code_identity import project_code_identity
from .ingest.indexer import index_project
from .ingest.runscan import (
    evaluation_snapshot,
    parse_iso_ts,
    preferred_attempt_id,
    read_jsonl,
    train_metric_records,
)
from .evidence_conflicts import classify_evidence_conflicts
from .project_service import ProjectApplicationService
from .runtime import ExperimentServerRuntime
from .outward import attempt_dto, operational_decision, run_dto, sanitized_outward
from .operations import (
    GPU_BUDGET,
    OBSERVABILITY_TARGET,
    WANDB_CLOUD_SYNC,
    OPERATIONS_BY_ID,
    OperationAvailability,
    OperationParameter,
    operations_for_scope,
)
from .schemas import (
    OperationScope,
    OperationScopeType,
    CampaignRelationship,
    ProjectLifecycleState,
    ResearchProject,
)
from .telemetry import Telemetry


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


def _normalized_failure_class(value: Any) -> str | None:
    normalized = str(value or "").strip()
    return None if normalized.lower() in {"", "none", "null"} else normalized


def _evidence_timestamp(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return parse_iso_ts(value)


def _raw_process_failure_summary(
    collection: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract a process signal without deciding whether it is applicable."""
    process = collection.get("process_evidence")
    process = process if isinstance(process, dict) else {}
    stderr_tail = process.get("stderr_tail")
    stdout_tail = process.get("stdout_tail")
    stderr = "\n".join(str(item) for item in stderr_tail or [])
    stdout = "\n".join(str(item) for item in stdout_tail or [])
    combined = f"{stdout}\n{stderr}"
    failure_class = _normalized_failure_class(collection.get("failure_class"))

    oom = _OOM_RE.search(combined)
    if oom:
        ranks = {
            int(value)
            for value in re.findall(r"\[rank(\d+)\].*?OutOfMemoryError", combined)
        }
        world_match = re.search(r"(?:world_size|nproc_per_node)=(\d+)", stdout)
        world_size = int(world_match.group(1)) if world_match else None
        phase = "first_backward" if (
            "Performing initial training step" in combined and ".backward()" in combined
        ) else ("backward" if ".backward()" in combined else "unknown")
        return {
            "failure_signature": "CUDA_OOM",
            # Concrete process OOM evidence must override a coarse/stale
            # scheduler or log-transport classification.
            "failure_class": "resource",
            "phase": phase,
            "requested_bytes": _memory_bytes(oom["requested"], oom["requested_unit"]),
            "free_bytes": _memory_bytes(oom["free"], oom["free_unit"]),
            "total_bytes": _memory_bytes(oom["total"], oom["total_unit"]),
            "allocated_bytes": _memory_bytes(oom["allocated"], oom["allocated_unit"]),
            "rank_count": world_size or len(ranks) or None,
            "observed_oom_rank_count": len(ranks) or None,
            "source": "collection.process_evidence",
        }
    signatures = (
        ("OutOfMemoryError", "CUDA_OOM", "resource"),
        ("CUDA out of memory", "CUDA_OOM", "resource"),
        ("ModuleNotFoundError", "MISSING_PYTHON_MODULE", "configuration"),
        ("no kernel image is available", "UNSUPPORTED_CUDA_KERNEL", "configuration"),
        ("TIMEOUT", "TIMEOUT", "timeout"),
    )
    for needle, signature, default_class in signatures:
        if needle.lower() in combined.lower():
            return {
                "failure_signature": signature,
                "failure_class": (
                    default_class if signature == "CUDA_OOM"
                    else failure_class or default_class
                ),
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


_FAILED_ATTEMPT_STATES = frozenset({"FAILED", "PREEMPTED"})
_FAILED_PROCESS_STATES = frozenset({"FAILED", "PREEMPTED"})
_FAILED_SCHEDULER_STATES = frozenset({
    "FAILED", "PREEMPTED", "NODE_FAIL", "OUT_OF_MEMORY", "TIMEOUT",
})
_FAILED_WORKER_STATES = frozenset({"FAILED", "LOST", "ERROR", "NODE_FAIL"})


def attempt_failure_evidence_assessment(
    collection: dict[str, Any],
    decision: dict[str, Any] | None = None,
    *,
    attempt_id: str,
    attempt_state: str | None,
    attempt_started_at: Any,
    observed_at: Any = None,
    domain_observed_at: dict[str, Any] | None = None,
    domain_attempt_ids: dict[str, Any] | None = None,
    domain_evidence_sources: dict[str, str | None] | None = None,
    decision_observed_at: Any = None,
    decision_source_binding: str = "EXACT_ATTEMPT",
    decision_evidence_source: str | None = None,
) -> dict[str, Any]:
    """Separate applicable terminal failure evidence from diagnostic signals.

    Failure classification is fail-closed: collected evidence must bind to the
    exact Attempt, be no older than that Attempt, and the exact Attempt must be
    in a failed terminal state.  Signals that do not meet all three conditions
    remain visible as explicitly non-applicable diagnostics.
    """
    expected_attempt_id = str(attempt_id)
    state = str(attempt_state or "UNKNOWN").upper()
    process_state = str(collection.get("process_state") or "UNKNOWN").upper()
    scheduler_state = str(collection.get("scheduler_state") or "UNKNOWN").upper()
    worker_state = str(collection.get("worker_state") or "UNKNOWN").upper()
    started_ts = _evidence_timestamp(attempt_started_at)
    terminal_failure = state in _FAILED_ATTEMPT_STATES
    observed_by_domain = dict(domain_observed_at or {})
    identities_by_domain = dict(domain_attempt_ids or {})
    sources_by_domain = dict(domain_evidence_sources or {})
    for domain in ("process", "scheduler", "worker"):
        observed_by_domain.setdefault(domain, observed_at)
        identities_by_domain.setdefault(domain, collection.get("attempt_id"))

    def applicability_for(domain: str, *, domain_terminal: bool) \
            -> tuple[str, str, str | None, float | None]:
        observed_id = str(identities_by_domain.get(domain) or "") or None
        observed_ts = _evidence_timestamp(observed_by_domain.get(domain))
        if observed_id != expected_attempt_id:
            return (
                "ATTEMPT_MISMATCH",
                f"{domain} evidence names {observed_id or 'no Attempt'} instead of "
                f"{expected_attempt_id}", observed_id, observed_ts,
            )
        if started_ts is None or observed_ts is None:
            return (
                "UNKNOWN_APPLICABILITY",
                f"actual Attempt start or {domain} observation time is unavailable",
                observed_id, observed_ts,
            )
        if observed_ts < started_ts:
            return (
                "STALE", f"{domain} evidence predates the exact Attempt start",
                observed_id, observed_ts,
            )
        if not terminal_failure:
            return (
                "NON_APPLICABLE",
                f"exact Attempt state {state} is not a failed terminal state",
                observed_id, observed_ts,
            )
        if not domain_terminal:
            return (
                "NON_APPLICABLE", f"{domain} layer state is not terminal failed",
                observed_id, observed_ts,
            )
        return (
            "APPLICABLE", f"exact, fresh terminal {domain} evidence",
            observed_id, observed_ts,
        )

    failure_summary: dict[str, Any] | None = None
    diagnostics: list[dict[str, Any]] = []
    raw_process = _raw_process_failure_summary(collection)

    candidates: list[tuple[str, dict[str, Any], bool]] = []
    process_terminal = process_state in _FAILED_PROCESS_STATES
    if process_terminal or raw_process is not None:
        process_candidate = raw_process or {
            "failure_signature": "UNCLASSIFIED_PROCESS_FAILURE",
            "failure_class": _normalized_failure_class(collection.get("failure_class"))
            or "unknown",
            "phase": "unknown",
            "source": "collection.process_evidence",
        }
        candidates.append((
            "process", {**process_candidate, "failure_domain": "process"},
            process_terminal,
        ))
    scheduler_terminal = scheduler_state in _FAILED_SCHEDULER_STATES
    if scheduler_terminal:
        candidates.append(("scheduler", {
            "failure_signature": "SCHEDULER_TERMINAL_FAILURE",
            "failure_class": _normalized_failure_class(collection.get("failure_class"))
            or "scheduler",
            "failure_domain": "scheduler",
            "phase": "unknown",
            "source": "collection.scheduler_state",
            "scheduler_state": scheduler_state,
        }, True))
    worker_terminal = worker_state in _FAILED_WORKER_STATES
    if worker_terminal:
        candidates.append(("worker", {
            "failure_signature": "WORKER_TERMINAL_FAILURE",
            "failure_class": _normalized_failure_class(collection.get("failure_class"))
            or "worker",
            "failure_domain": "worker",
            "phase": "unknown",
            "source": "collection.worker_state",
            "worker_state": worker_state,
        }, True))

    assessed: list[tuple[str, dict[str, Any], str, str, str | None, float | None]] = []
    for domain, candidate, domain_terminal in candidates:
        applicability, reason, observed_id, observed_ts = applicability_for(
            domain, domain_terminal=domain_terminal,
        )
        assessed.append((
            domain, candidate, applicability, reason, observed_id, observed_ts,
        ))

    selected = next((item for item in assessed if item[2] == "APPLICABLE"), None)
    if selected is not None:
        _, candidate, _, _, _, observed_ts = selected
        failure_summary = {
            **candidate,
            "attempt_id": expected_attempt_id,
            "observed_at": observed_ts,
            "applicability": "APPLICABLE",
            "evidence_source": sources_by_domain.get(selected[0]),
        }
    for domain, candidate, applicability, reason, observed_id, observed_ts in assessed:
        if selected is not None and candidate is selected[1]:
            continue
        if applicability != "APPLICABLE":
            diagnostics.append({
                "kind": "failure_signal",
                **candidate,
                "attempt_id": observed_id,
                "expected_attempt_id": expected_attempt_id,
                "attempt_state": state,
                "observed_at": observed_ts,
                "applicability": applicability,
                "evidence_source": sources_by_domain.get(domain),
                "reason": reason,
            })

    decision_class = (
        _normalized_failure_class(decision.get("failure_class"))
        if isinstance(decision, dict) else None
    )
    if decision_class is not None and failure_summary is None:
        diagnostics.append({
            "kind": "preliminary_failure_classification",
            "failure_class": decision_class,
            "source": "decision.failure_class",
            "attempt_id": expected_attempt_id,
            "attempt_state": state,
            "observed_at": _evidence_timestamp(decision_observed_at),
            "applicability": "NON_APPLICABLE",
            "source_binding": decision_source_binding,
            "evidence_source": decision_evidence_source,
            "reason": (
                "decision metadata is contextual only and is not exact, fresh, "
                "terminal failure evidence"
            ),
        })

    return {
        "failure_summary": failure_summary,
        "diagnostic_evidence": compact_evidence(diagnostics),
    }


def structured_failure_summary(
    collection: dict[str, Any], decision: dict[str, Any] | None = None, *,
    attempt_id: str | None = None, attempt_state: str | None = None,
    attempt_started_at: Any = None, observed_at: Any = None,
) -> dict[str, Any] | None:
    """Return only exact, fresh, terminal failure evidence.

    Callers without the Attempt applicability context receive no failure rather
    than silently promoting historical process text into a current failure.
    """
    if attempt_id is None:
        return None
    return attempt_failure_evidence_assessment(
        collection, decision,
        attempt_id=attempt_id,
        attempt_state=attempt_state,
        attempt_started_at=attempt_started_at,
        observed_at=observed_at,
    )["failure_summary"]


def evidence_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ExperimentServerApplication:
    def __init__(self, runtime: ExperimentServerRuntime):
        self.runtime = runtime
        self.telemetry = getattr(runtime, "telemetry", Telemetry())
        self.project_service = ProjectApplicationService(runtime)

    def recover_observability_policies(self) -> None:
        """Idempotently close the VERIFIED-action/target crash window."""
        for action in self.runtime.action_store.list_all():
            if str(action.get("operation") or "") in {
                "SUBMIT_RUN", "RETRY_ATTEMPT", "RUN_EVALUATION",
            }:
                self._activate_observability_policy(action, action)

    # --------------------------------------------------------------- scopes

    def resolve_scope(
        self, project_name: str, scope_type: OperationScopeType | str, object_id: str,
    ) -> tuple[OperationScope, ResearchProject, Any]:
        try:
            project = self.runtime.project(project_name)
        except KeyError as exc:
            raise ApplicationError(str(exc).strip("'"), status_code=404,
                                   code="UNKNOWN_PROJECT") from exc
        kind = OperationScopeType(scope_type)
        if kind == OperationScopeType.PROJECT:
            if object_id != project.project:
                raise ApplicationError("project object_id must equal project",
                                       status_code=404, code="UNKNOWN_PROJECT")
            resolved: Any = project
        elif kind == OperationScopeType.RESEARCH_QUESTION:
            resolved = next((item for item in project.research_questions if item.id == object_id), None)
            if resolved is None:
                raise ApplicationError(f"unknown research_question: {object_id}", status_code=404,
                                       code="UNKNOWN_RESEARCH_QUESTION")
        elif kind == OperationScopeType.CAMPAIGN:
            resolved = next((
                campaign for campaign in project.campaigns if campaign.name == object_id
            ), None)
            if resolved is None:
                raise ApplicationError(f"unknown campaign: {object_id}", status_code=404,
                                       code="UNKNOWN_CAMPAIGN")
        elif kind == OperationScopeType.RUN:
            resolved = self.runtime.index.get_run(project.project, object_id)
            if resolved is None:
                resolved = authored_run_placeholder(project, object_id)
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
        return OperationScope(project=project.project, scope_type=kind, object_id=object_id), project, resolved

    @staticmethod
    def _operational_decision(value: Any) -> dict[str, Any]:
        return operational_decision(value)

    @staticmethod
    def row_evidence(row: Any) -> dict[str, Any]:
        return sanitized_outward(compact_evidence({
            "run_id": row.run_id, "campaign": row.campaign, "role": row.role,
            "campaign_binding": row.campaign_binding.model_dump(mode="json"),
            "campaign_memberships": [
                item.model_dump(mode="json") for item in row.campaign_memberships
            ],
            "scheduler_state": row.scheduler_state,
            "evidence": row.evidence.model_dump(mode="json"),
            "latest_metrics": row.latest_metrics, "eval_metrics": row.eval_metrics,
            "eval_variants": row.eval_variants,
            "evaluation_snapshot": getattr(row, "evaluation_snapshot", {}),
            "canonical_eval_variant_id": row.canonical_eval_variant_id,
            "checkpoint": row.checkpoint, "artifacts": row.artifacts,
            "decision": ExperimentServerApplication._operational_decision(row.decision),
            "provenance": row.provenance,
            "warnings": row.warnings, "evidence_conflicts": row.evidence_conflicts,
        }))

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
                    "evaluation_snapshot": getattr(peer, "evaluation_snapshot", {}),
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

    def bounded_evidence(self, scope: OperationScope, project: ResearchProject,
                         resolved: Any) -> dict[str, Any]:
        index = self.runtime.index
        if scope.scope_type == OperationScopeType.PROJECT:
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
                    "scheduler_state": row.scheduler_state,
                    "decision": self._operational_decision(row.decision),
                    "failure_assessment": self._agent_failure_assessment(
                        self.run_failure_assessment(row),
                    ),
                    "stale_layers": [
                        name for name in ("scheduler", "worker", "process", "model", "evaluation")
                        if getattr(row.evidence, name).stale
                    ],
                } for row in rows],
            }
        if scope.scope_type == OperationScopeType.RESEARCH_QUESTION:
            names = set(resolved.links.campaigns)
            rows = [
                row for row in index.list_runs(project.project)
                if row.campaign in names or any(
                    binding.campaign in names for binding in row.campaign_memberships
                )
            ]
            return {"research_question": resolved.model_dump(mode="json"),
                    "runs": [self._agent_row_evidence(row) for row in rows]}
        if scope.scope_type == OperationScopeType.CAMPAIGN:
            rows = index.list_runs(project.project, campaign=scope.object_id)
            return {"campaign": resolved.model_dump(mode="json"),
                    "lifecycle": campaign_snapshot(index, project, scope.object_id),
                    "runs": [self._agent_row_evidence(row) for row in rows]}
        if scope.scope_type == OperationScopeType.RUN:
            failure_assessment = self.run_failure_assessment(resolved)
            return {"run": self.row_evidence(resolved),
                    "campaign_contexts": self.campaign_contexts(project, resolved),
                    "attempts": [
                        attempt_dto(item) for item in resolved.attempts
                    ],
                    "failure_assessment": self._agent_failure_assessment(
                        failure_assessment,
                    )}
        run_id, attempt_id = scope.object_id.rsplit("::", 1)
        row = index.get_run(project.project, run_id)
        attempt_dir = Path(row.run_dir) / "attempts" / attempt_id
        _, assessment = self._attempt_failure_assessment(row, resolved, attempt_dir)
        return {"run": self.row_evidence(row),
                "campaign_contexts": self.campaign_contexts(project, row),
                "attempt": attempt_dto(resolved),
                "failure_assessment": self._agent_failure_assessment(assessment)}

    # -------------------------------------------------------------- operations

    def operation_availability(
        self, project: str, scope_type: OperationScopeType | str, object_id: str,
    ) -> list[OperationAvailability]:
        """Return deterministic operation eligibility for one exact scope."""
        scope, configured, resolved = self.resolve_scope(project, scope_type, object_id)
        default_budget = 1.0
        result: list[OperationAvailability] = []
        for base_operation in operations_for_scope(scope.scope_type):
            available_targets = self._publication_targets_available()
            parameters = tuple(
                replace(parameter, default=default_budget)
                if parameter.key == GPU_BUDGET.key else
                replace(
                    parameter,
                    choices=tuple(
                        choice for choice in parameter.choices
                        if choice[1] in available_targets
                    ),
                    default=(available_targets[0] if available_targets else None),
                ) if parameter.key == OBSERVABILITY_TARGET.key else parameter
                for parameter in base_operation.parameters
                if (
                    parameter.key != WANDB_CLOUD_SYNC.key
                    or self._cloud_publication_available()
                )
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
                metadata={"intent_kind": (
                    {
                        OperationScopeType.CAMPAIGN: "ARCHIVE_CAMPAIGN",
                        OperationScopeType.RUN: "ARCHIVE_RUN",
                        OperationScopeType.ATTEMPT: "ARCHIVE_ATTEMPT",
                    }[scope.scope_type]
                    if operation.operation_id == "object.archive"
                    else operation.intent_kind
                )},
            ))
        return result

    def _cloud_publication_available(self) -> bool:
        policy = self.runtime.config.observability.wandb_cloud
        if not policy.enabled or not policy.default_credential_ref or not policy.entity:
            return False
        return self.runtime.credential_store.status(
            policy.default_credential_ref,
        ).configured

    def _local_publication_available(self) -> bool:
        policy = self.runtime.config.observability.local_wandb
        return bool(
            policy.enabled and policy.publisher_entity
            and policy.publisher_credential_ref
            and self.runtime.credential_store.status(
                policy.publisher_credential_ref,
            ).configured
        )

    def _publication_targets_available(self) -> tuple[str, ...]:
        return tuple(
            target for target, available in (
                ("local", self._local_publication_available()),
                ("cloud", self._cloud_publication_available()),
            ) if available
        )

    def _operation_blockers(
        self, operation_id: str, scope: OperationScope,
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
            if operation_id == "run.derive" and scope.scope_type == OperationScopeType.RUN:
                memberships = getattr(resolved, "campaign_memberships", []) or []
                if not memberships and not getattr(resolved, "campaign", None):
                    reasons.append("Run has no authored Campaign context to derive from")
        elif operation_id == "object.archive":
            if scope.scope_type == OperationScopeType.CAMPAIGN:
                status = campaign_snapshot(self.runtime.index, project, scope.object_id)
                if status.get("lifecycle_state") == "ARCHIVED":
                    reasons.append("Campaign is already archived")
            else:
                root = (project.base_dir or Path(".")) / "experiments" / "archive_records"
                if scope.scope_type == OperationScopeType.RUN:
                    record = root / "runs" / f"{scope.object_id}.yml"
                else:
                    run_id, attempt_id = scope.object_id.rsplit("::", 1)
                    record = root / "attempts" / f"{run_id}--{attempt_id}.yml"
                if record.is_file():
                    reasons.append(f"Archive record already exists: {record}")
        elif operation_id == "observability.backfill":
            if not self.runtime.config.action_runtime.allow_observability_mutations:
                reasons.append("Observability mutations are disabled by daemon policy")
            if not self._publication_targets_available():
                reasons.append("No authenticated W&B publisher target is available")
            try:
                attempts = self._observability_attempts(scope, project, resolved)
            except (KeyError, ValueError):
                attempts = []
            if not attempts:
                reasons.append("Scope has no observed Attempts to backfill")
            elif len(attempts) > 500:
                reasons.append("Scope exceeds the 500-Attempt backfill limit")
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
            if len(materialized) != 1:
                reasons.append(
                    "Run must have exactly one authored materialized Campaign membership"
                )
            else:
                status = campaign_snapshot(
                    self.runtime.index, project, materialized[0].campaign,
                )
                lifecycle = str(status.get("lifecycle_state") or "UNKNOWN").upper()
                validation = str(
                    (status.get("validation") or {}).get("status") or "UNKNOWN"
                ).upper()
                if lifecycle != "ACTIVE" or validation != "PASS":
                    reasons.append(
                        "Materializing Campaign is not submit-ready: "
                        f"lifecycle={lifecycle}; validation={validation}"
                    )
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
            identity = scope.object_id if scope.scope_type == OperationScopeType.ATTEMPT else None
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
        scope_type: OperationScopeType | str, object_id: str,
    ) -> None:
        availability = next((item for item in self.operation_availability(
            project, scope_type, object_id,
        ) if item.operation.operation_id == operation_id), None)
        if availability is None or not availability.available:
            reasons = availability.reasons if availability else ("operation is not valid in this scope",)
            raise ApplicationError("; ".join(reasons), code="OPERATION_BLOCKED")

    def invoke_direct_operation(
        self, operation_id: str, project: str,
        scope_type: OperationScopeType | str, object_id: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare a catalogued direct Action for one exact scope."""
        parameters = parameters or {}
        scope, configured, _ = self.resolve_scope(project, scope_type, object_id)
        self._require_operation_available(operation_id, project, scope.scope_type, object_id)
        definition = OPERATIONS_BY_ID.get(operation_id)
        availability = next((item for item in self.operation_availability(
            project, scope.scope_type, object_id,
        ) if item.operation.operation_id == operation_id), None)
        if definition is None or availability is None:
            raise ApplicationError(
                f"unknown operation {operation_id}", code="INVALID_OPERATION",
            )
        allowed_parameters = {item.key for item in availability.operation.parameters}
        unsupported = sorted(set(parameters) - allowed_parameters)
        if unsupported:
            raise ApplicationError(
                "unsupported operation parameters: " + ", ".join(unsupported),
                code="INVALID_OPERATION",
            )
        reason = str(parameters.get("reason") or "")
        cloud_sync = str(parameters.get("wandb_cloud_sync") or "no").lower()
        if cloud_sync not in {"yes", "no"}:
            raise ApplicationError(
                "wandb_cloud_sync must be 'yes' or 'no'", code="INVALID_OPERATION",
            )
        budget = 0.0
        if operation_id in {"run.submit", "attempt.retry"}:
            budget_value = parameters.get("max_gpu_hours")
            try:
                budget = float(1.0 if budget_value is None else budget_value)
            except (TypeError, ValueError) as exc:
                raise ApplicationError(
                    "max_gpu_hours must be a number", code="INVALID_OPERATION",
                ) from exc
            if budget <= 0:
                raise ApplicationError("max_gpu_hours must be positive", code="INVALID_OPERATION")
        if operation_id == "object.archive":
            if scope.scope_type == OperationScopeType.CAMPAIGN:
                return self.prepare_campaign_archive(project, object_id, reason=reason)
            return self.prepare_object_archive(
                project, scope.scope_type, object_id, reason=reason,
            )
        if operation_id == "observability.backfill":
            target = str(parameters.get("target") or "")
            return self.prepare_observability_backfill(
                project, scope.scope_type, object_id,
                target=target, reason=reason,
            )
        if operation_id == "run.submit":
            return self.prepare_run_submit(
                project, object_id, max_gpu_hours=budget,
                reason=reason or "Requested from the scoped operation catalog",
                wandb_cloud_sync=cloud_sync == "yes",
            )
        if operation_id == "attempt.retry":
            return self.prepare_attempt_retry(
                project, object_id,
                new_attempt_id=(
                    str(parameters["new_attempt_id"])
                    if parameters.get("new_attempt_id") else None
                ),
                max_gpu_hours=budget, reason=reason,
                wandb_cloud_sync=cloud_sync == "yes",
            )
        if operation_id == "attempt.cancel":
            return self.prepare_attempt_cancel(project, object_id, reason=reason)
        raise ApplicationError(
            f"operation {operation_id} requires a client-authored intent",
            code="INVALID_OPERATION",
        )

    # ------------------------------------------------------------- read model

    def object_show(self, project: str, scope_type: OperationScopeType | str,
                    object_id: str) -> dict[str, Any]:
        """Return one of the five canonical objects plus its bounded evidence."""
        scope, configured, resolved = self.resolve_scope(project, scope_type, object_id)
        if scope.scope_type == OperationScopeType.RUN and hasattr(resolved, "run_id"):
            object_payload = self.row_evidence(resolved)
        elif scope.scope_type == OperationScopeType.ATTEMPT:
            object_payload = attempt_dto(resolved)
        elif hasattr(resolved, "model_dump"):
            object_payload = resolved.model_dump(mode="json")
        else:
            object_payload = compact_evidence(resolved)
        bounded = self.bounded_evidence(scope, configured, resolved)
        return {
            "scope": scope.model_dump(mode="json"),
            "object": object_payload,
            "evidence": bounded,
            "evidence_digest": evidence_digest(bounded),
            "code_identity": project_code_identity(configured),
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

    def _prepare_action_intent(
        self, scope: OperationScope, project: ResearchProject, intent: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            return self.runtime.action_service.prepare(scope, project, intent)
        except (RuntimeError, ValueError) as exc:
            raise ApplicationError(str(exc), code="ACTION_BLOCKED") from exc

    def prepare_campaign_archive(
        self, project: str, campaign: str, *, reason: str,
    ) -> dict[str, Any]:
        self._require_operation_available(
            "object.archive", project, OperationScopeType.CAMPAIGN, campaign,
        )
        scope, configured, resolved = self.resolve_scope(
            project, OperationScopeType.CAMPAIGN, campaign,
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
        digest = evidence_digest(self.bounded_evidence(scope, configured, resolved))
        action = self._prepare_action_intent(scope, configured, {
            "kind": "ARCHIVE_CAMPAIGN",
            "title": f"Archive Campaign {campaign}",
            "target": f"campaign://{project}/{campaign}@{status['revision_id']}",
            "change_summary": "freeze an immutable Campaign archive record",
            "resource_estimate": "none",
            "rationale": reason.strip(),
            "risk": "the Campaign revision will leave active research views",
            "draft": draft,
            "evidence_digest": digest,
        })
        return {"action": action, "campaign": status}

    def prepare_object_archive(
        self, project: str, scope_type: OperationScopeType | str, object_id: str, *, reason: str,
    ) -> dict[str, Any]:
        self._require_operation_available("object.archive", project, scope_type, object_id)
        scope, configured, resolved = self.resolve_scope(project, scope_type, object_id)
        if scope.scope_type not in {OperationScopeType.RUN, OperationScopeType.ATTEMPT}:
            raise ApplicationError("only Run or Attempt records can be archived here",
                                   code="INVALID_ARCHIVE_SCOPE")
        if not reason.strip():
            raise ApplicationError("archive reason is required", status_code=422,
                                   code="INVALID_ARCHIVE_REASON")
        if scope.scope_type == OperationScopeType.ATTEMPT:
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
        action = self._prepare_action_intent(scope, configured, {
            "kind": kind,
            "title": f"Archive {scope.scope_type.value} record {object_id}",
            "target": target,
            "change_summary": "append an evidence-bound archive record without deleting evidence",
            "resource_estimate": "none",
            "rationale": reason.strip(),
            "risk": "object is hidden from default active workflows but immutable evidence remains",
            "draft": yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            "evidence_digest": digest,
        })
        return {"action": action}

    def _observability_attempts(
        self, scope: OperationScope, project: ResearchProject, resolved: Any,
    ) -> list[tuple[str, str]]:
        if scope.scope_type == OperationScopeType.PROJECT:
            rows = self.runtime.index.list_runs(project.project)
        elif scope.scope_type == OperationScopeType.CAMPAIGN:
            rows = self.runtime.index.list_runs(
                project.project, campaign=scope.object_id,
            )
        elif scope.scope_type == OperationScopeType.RUN:
            rows = [resolved]
        elif scope.scope_type == OperationScopeType.ATTEMPT:
            run_id, attempt_id = scope.object_id.rsplit("::", 1)
            return [(run_id, attempt_id)]
        else:
            return []
        return sorted({
            (str(row.run_id), str(attempt.attempt_id))
            for row in rows for attempt in (row.attempts or [])
            if attempt.attempt_id
        })

    def prepare_observability_backfill(
        self, project: str, scope_type: OperationScopeType | str,
        object_id: str, *, target: str, reason: str,
    ) -> dict[str, Any]:
        self._require_operation_available(
            "observability.backfill", project, scope_type, object_id,
        )
        scope, configured, resolved = self.resolve_scope(
            project, scope_type, object_id,
        )
        available = self._publication_targets_available()
        if target not in available:
            raise ApplicationError(
                f"publisher target {target!r} is unavailable",
                code="PUBLISHER_UNAVAILABLE",
            )
        if not reason.strip():
            raise ApplicationError(
                "backfill reason is required", status_code=422,
                code="INVALID_BACKFILL_REASON",
            )
        attempts = self._observability_attempts(scope, configured, resolved)
        digest = evidence_digest(self.bounded_evidence(scope, configured, resolved))
        action = self._prepare_action_intent(scope, configured, {
            "kind": "OBSERVABILITY_BACKFILL",
            "title": (
                f"Backfill {len(attempts)} Attempts to {target} W&B"
            ),
            "target": f"wandb-{target}://{project}/{scope.object_id}",
            "change_summary": (
                f"enable {target} publication and replay sanitized history"
            ),
            "resource_estimate": f"{len(attempts)} Attempts",
            "rationale": reason.strip(),
            "risk": "external publication and potentially large durable backlog",
            "draft": yaml.safe_dump({
                "schema_version": 1,
                "project": project,
                "target": target,
                "reason": reason.strip(),
                "attempts": [
                    {"run_id": run_id, "attempt_id": attempt_id}
                    for run_id, attempt_id in attempts
                ],
            }, allow_unicode=True, sort_keys=False),
            "evidence_digest": digest,
        })
        return {
            "action": action,
            "preflight": {"target": target, "attempt_count": len(attempts)},
        }

    def _attempt_context(self, project: str, identity: str):
        scope, configured, attempt = self.resolve_scope(
            project, OperationScopeType.ATTEMPT, identity,
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
    def _path_mtime(path: Path) -> float | None:
        try:
            return path.stat().st_mtime
        except OSError:
            return None

    def _attempt_failure_assessment(
        self, row: Any, attempt: Any, attempt_dir: Path,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        run_dir = Path(row.run_dir)
        collection_path = attempt_dir / "collection.json"
        root_collection_path = run_dir / "collection.json"
        decision_path = attempt_dir / "decision.json"
        root_decision_path = run_dir / "decision.json"
        status_path = attempt_dir / "status.json"
        root_status_path = run_dir / "status.json"
        attempt_record_path = attempt_dir / "attempt.json"

        collection = self._read_mapping(collection_path)
        selected_collection_path = collection_path
        if not collection:
            collection = self._read_mapping(root_collection_path)
            selected_collection_path = root_collection_path

        local_decision = self._read_mapping(decision_path)
        root_decision = self._read_mapping(root_decision_path)
        if local_decision:
            decision = local_decision
            selected_decision_path = decision_path
            decision_binding = (
                "EXACT_ATTEMPT" if decision.get("attempt_id") == attempt.attempt_id
                else "EXACT_ATTEMPT_PATH_UNSCOPED" if decision.get("attempt_id") is None
                else "ATTEMPT_MISMATCH"
            )
        else:
            decision = root_decision
            selected_decision_path = root_decision_path
            decision_binding = (
                "EXACT_ROOT_MIRROR" if decision.get("attempt_id") == attempt.attempt_id
                else "BOUND_BY_EXACT_ROOT_COLLECTION"
                if decision.get("attempt_id") is None
                and collection.get("attempt_id") == attempt.attempt_id
                and selected_collection_path == root_collection_path
                else "ROOT_MIRROR_UNSCOPED" if decision.get("attempt_id") is None
                else "ATTEMPT_MISMATCH"
            )

        status = self._read_mapping(status_path)
        root_status = self._read_mapping(root_status_path)
        attempt_record = self._read_mapping(attempt_record_path)
        domain_observed_at: dict[str, Any] = {}
        domain_attempt_ids: dict[str, Any] = {}
        domain_sources: dict[str, str | None] = {}

        def observed(payload: dict[str, Any], path: Path, *keys: str) -> Any:
            return next(
                (payload.get(key) for key in keys if payload.get(key) is not None),
                self._path_mtime(path),
            )

        if not collection and attempt_record:
            summary = self._read_mapping(attempt_dir / "summary.json")
            summary_metrics = summary.get("metrics")
            collection = {
                "attempt_id": attempt_record.get("attempt_id"),
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
            process_observed = observed(
                attempt_record, attempt_record_path,
                "finished_at", "observed_at", "updated_at",
            )
            scheduler_observed = observed(
                root_status, root_status_path, "observed_at", "updated_at", "finished_at",
            )
            domain_observed_at.update({
                "process": process_observed,
                "worker": process_observed,
                "scheduler": scheduler_observed,
            })
            domain_attempt_ids.update({
                "process": attempt_record.get("attempt_id"),
                "worker": attempt_record.get("attempt_id"),
                "scheduler": root_status.get("attempt_id"),
            })
            domain_sources.update({
                "process": str(attempt_record_path),
                "worker": str(attempt_record_path),
                "scheduler": str(root_status_path),
            })
        elif collection:
            process = collection.get("process_evidence")
            process = process if isinstance(process, dict) else {}
            collection_observed = observed(
                collection, selected_collection_path, "observed_at", "collected_at",
            )
            domain_observed_at.update({
                "process": process.get("observed_at")
                or collection.get("process_observed_at") or collection_observed,
                "worker": collection.get("worker_observed_at") or collection_observed,
                "scheduler": collection.get("scheduler_observed_at") or collection_observed,
            })
            domain_attempt_ids.update({
                domain: collection.get("attempt_id")
                for domain in ("process", "worker", "scheduler")
            })
            domain_sources.update({
                domain: str(selected_collection_path)
                for domain in ("process", "worker", "scheduler")
            })
            status_candidate = status if status else root_status
            status_candidate_path = status_path if status else root_status_path
            if (
                status_candidate.get("attempt_id") == attempt.attempt_id
                and status_candidate.get("state") is not None
            ):
                domain_observed_at["scheduler"] = observed(
                    status_candidate, status_candidate_path,
                    "observed_at", "updated_at", "finished_at",
                )
                domain_attempt_ids["scheduler"] = status_candidate.get("attempt_id")
                domain_sources["scheduler"] = str(status_candidate_path)

        manifest, _ = self._manifest_at(
            attempt_dir, ("attempt.yaml", "attempt.json", "control_attempt.yaml"),
        )
        attempt_started_at = (
            status.get("started_at") or attempt_record.get("started_at")
            or manifest.get("started_at")
            or (
                root_status.get("started_at")
                if root_status.get("attempt_id") == attempt.attempt_id else None
            )
        )
        decision_observed_at = (
            decision.get("observed_at") or decision.get("created_at")
            or self._path_mtime(selected_decision_path)
        )
        assessment = attempt_failure_evidence_assessment(
            collection, decision,
            attempt_id=attempt.attempt_id,
            attempt_state=attempt.state,
            attempt_started_at=attempt_started_at,
            domain_observed_at=domain_observed_at,
            domain_attempt_ids=domain_attempt_ids,
            domain_evidence_sources=domain_sources,
            decision_observed_at=decision_observed_at,
            decision_source_binding=decision_binding,
            decision_evidence_source=(
                str(selected_decision_path) if decision else None
            ),
        )
        return collection, assessment

    def run_failure_assessment(self, row: Any) -> dict[str, Any]:
        """Return the one daemon-owned assessment for a Run's current Attempt."""
        empty = {"failure_summary": None, "diagnostic_evidence": []}
        current_id = preferred_attempt_id(Path(row.run_dir))
        if current_id is None:
            return empty
        current_attempt = next(
            (
                item for item in getattr(row, "attempts", [])
                if item.attempt_id == current_id
            ),
            None,
        )
        if current_attempt is None:
            return empty
        _, assessment = self._attempt_failure_assessment(
            row, current_attempt, Path(row.run_dir) / "attempts" / current_id,
        )
        return assessment

    @staticmethod
    def _agent_failure_assessment(assessment: dict[str, Any]) -> dict[str, Any]:
        return {
            **assessment,
            "agent_instruction": (
                "Only failure_summary is applicable failure evidence. "
                "diagnostic_evidence is explicitly non-applicable and MUST NOT "
                "be treated as failure, retry, or classification evidence."
            ),
        }

    def _agent_row_evidence(self, row: Any) -> dict[str, Any]:
        return {
            **self.row_evidence(row),
            "failure_assessment": self._agent_failure_assessment(
                self.run_failure_assessment(row),
            ),
        }

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
                "evidence"
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
            for name in ("execution", "identity", "evidence")
        }
        return {
            "schema_version": 1,
            "object_type": object_type,
            "identity": identity,
            "result": result,
            "exit_code": 0 if result == "PASS" else 3,
            "execution_evidence_result": dimensions["execution"]["result"],
            "identity_result": dimensions["identity"]["result"],
            "evidence_result": dimensions["evidence"]["result"],
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
        _, _, row = self.resolve_scope(project, OperationScopeType.RUN, run_id)
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
        if checkpoint and checkpoint_step is not None:
            gates.append(self._gate(
                "attempt.checkpoint_evidence", "PASS",
                "completed checkpoint is recorded for the exact Attempt",
                {"path": checkpoint, "step": checkpoint_step,
                 "local_entries": local_checkpoints[:20]},
            ))
        else:
            gates.append(self._gate(
                "attempt.checkpoint_evidence", "UNKNOWN",
                "completed checkpoint path or step is missing",
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

        evidence_conflicts, reclassified = classify_evidence_conflicts(
            collection.get("evidence_conflicts"),
            project=project, run_id=row.run_id, attempt_id=attempt_id,
        )
        gates.append(self._gate(
            "attempt.evidence_conflicts",
            "BLOCKED" if evidence_conflicts else "PASS",
            "exact variant-bound evidence contains conflicting values"
            if evidence_conflicts else
            "no exact-identity evidence conflicts are recorded",
            {
                "count": len(evidence_conflicts),
                "conflicts": evidence_conflicts[:50],
                "reclassified_cross_binding": reclassified[:50],
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
        _, _, row = self.resolve_scope(project, OperationScopeType.RUN, run_id)
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
        collection, assessment = self._attempt_failure_assessment(
            row, attempt, attempt_dir,
        )
        return {
            "scope": scope.model_dump(mode="json"),
            "attempt": attempt_dto(attempt),
            "run_id": row.run_id,
            "run_dir": row.run_dir,
            "attempt_dir": str(attempt_dir),
            "failure_assessment": assessment,
            "collection": sanitized_outward(compact_evidence(collection)),
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
        source_attempt_id = row.evidence.evaluation.attempt_id
        exact = source_attempt_id == attempt.attempt_id
        variants = row.eval_variants if exact else []
        snapshot = row.evaluation_snapshot if exact else evaluation_snapshot([])
        return {"project": project, "run_id": row.run_id,
                "attempt_id": attempt.attempt_id,
                "source_attempt_id": source_attempt_id, "variants": variants,
                "evaluation_snapshot": snapshot,
                "evidence_status": "INDEXED_EXACT_ATTEMPT" if exact
                else "EXACT_ATTEMPT_NOT_INDEXED"}

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

    def prepare_run_submit(self, project: str, run_id: str, *,
                           max_gpu_hours: float, reason: str = "",
                           wandb_cloud_sync: bool = False) -> dict[str, Any]:
        """Prepare a reviewable first-submission Action for an authored Run."""
        self._require_operation_available(
            "run.submit", project, OperationScopeType.RUN, run_id,
        )
        if max_gpu_hours <= 0:
            raise ApplicationError("max_gpu_hours must be positive", status_code=422,
                                   code="INVALID_GPU_BUDGET")
        scope, configured, row = self.resolve_scope(project, OperationScopeType.RUN, run_id)
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
            "wandb_cloud_sync": wandb_cloud_sync,
        }, sort_keys=False)
        digest = evidence_digest(self.bounded_evidence(scope, configured, row))
        action = self._prepare_action_intent(scope, configured, {
            "kind": "SUBMIT_RUN",
            "title": f"Launch {row.run_id} as {attempt_id}",
            "target": f"run://{configured.project}/{row.run_id}",
            "change_summary": reason or "submit the authored Run's first Attempt",
            "resource_estimate": f"up to {max_gpu_hours:g} GPU-hours",
            "rationale": "materialize an authored Campaign membership",
            "risk": "scheduler mutation; approval and prepared gates are required before execution",
            "draft": draft,
            "evidence_digest": digest,
        })
        return {"action": action}

    def prepare_attempt_retry(self, project: str, identity: str, *,
                              new_attempt_id: str | None,
                              max_gpu_hours: float, reason: str,
                              wandb_cloud_sync: bool = False) -> dict[str, Any]:
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
            "attempt.retry", project, OperationScopeType.ATTEMPT, identity,
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
            "wandb_cloud_sync": wandb_cloud_sync,
        }, sort_keys=False)
        result = self._prepare_attempt_action(
            scope, configured, row, attempt, kind="RETRY_ATTEMPT", draft=draft,
            title=f"Retry {row.run_id} from {attempt.attempt_id} as {new_attempt_id}",
            change_summary=reason or "retry failed attempt with a new attempt identity",
            resource_estimate=f"up to {max_gpu_hours:g} GPU-hours",
            risk="scheduler mutation; review checkpoint and failure classification before approval",
        )
        return result

    def prepare_attempt_cancel(self, project: str, identity: str, *,
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
            "attempt.cancel", project, OperationScopeType.ATTEMPT, identity,
        )
        campaign = self._campaign_file(configured, row.campaign)
        draft = yaml.safe_dump({
            "campaign_file": str(campaign), "run_id": row.run_id,
            "attempt_id": attempt.attempt_id,
            "backend_job_id": attempt.backend_job_id,
        }, sort_keys=False)
        return self._prepare_attempt_action(
            scope, configured, row, attempt, kind="CANCEL_RUN", draft=draft,
            title=f"Cancel {row.run_id} {attempt.attempt_id}",
            change_summary=reason or "cancel the exact observed backend job",
            resource_estimate="none", risk="scheduler cancellation",
        )

    def _prepare_attempt_action(
        self, scope: OperationScope, configured: ResearchProject, row: Any, attempt: Any,
        *, kind: str, draft: str, title: str, change_summary: str,
        resource_estimate: str, risk: str,
    ) -> dict[str, Any]:
        digest = evidence_digest(self.bounded_evidence(scope, configured, attempt))
        action = self._prepare_action_intent(scope, configured, {
            "kind": kind, "title": title,
            "target": f"attempt://{configured.project}/{scope.object_id}",
            "change_summary": change_summary,
            "resource_estimate": resource_estimate,
            "rationale": f"operate on immutable attempt evidence for {row.run_id}",
            "risk": risk, "draft": draft, "evidence_digest": digest,
        })
        return {"action": action}

    def run_detail(self, project: str, run_id: str) -> dict[str, Any]:
        _, _, row = self.resolve_scope(project, OperationScopeType.RUN, run_id)
        payload = run_dto(row)
        payload["is_terminal"] = row.is_terminal
        payload["failure_assessment"] = self.run_failure_assessment(row)
        return payload

    def run_metrics(self, project: str, run_id: str, *, keys: str | None = None,
                    max_points: int = 2000) -> dict[str, Any]:
        _, _, row = self.resolve_scope(project, OperationScopeType.RUN, run_id)
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
        _, _, row = self.resolve_scope(project, OperationScopeType.RUN, run_id)
        return {
            "source_attempt_id": row.evidence.evaluation.attempt_id,
            "variants": row.eval_variants,
            "evaluation_snapshot": row.evaluation_snapshot,
        }

    def run_events(self, project: str, run_id: str) -> dict[str, Any]:
        _, _, row = self.resolve_scope(project, OperationScopeType.RUN, run_id)
        events = read_jsonl(Path(row.run_dir) / "events.jsonl")
        for event in events:
            event["ts"] = parse_iso_ts(event.get("timestamp"))
        return {"events": events}

    # --------------------------------------------------------------- actions

    def list_actions(self, project: str, scope_type: OperationScopeType | str,
                     object_id: str) -> dict[str, Any]:
        scope, _, _ = self.resolve_scope(project, scope_type, object_id)
        return {"actions": self.runtime.action_store.list_for_scope(scope),
                "policy": self.runtime.config.action_runtime.model_dump()}

    def prepare_action(self, project: str, scope_type: OperationScopeType | str,
                       object_id: str, intent: dict[str, Any]) -> dict[str, Any]:
        scope, configured, resolved = self.resolve_scope(project, scope_type, object_id)
        with self.telemetry.span("research.action.prepare", {
            "research.project": scope.project,
            "research.scope_type": scope.scope_type.value,
            "research.object_id": scope.object_id,
        }) as span:
            current_digest = evidence_digest(
                self.bounded_evidence(scope, configured, resolved)
            )
            if intent.get("evidence_digest") != current_digest:
                raise ApplicationError(
                    "intent evidence digest does not match current bounded evidence",
                    code="STALE_EVIDENCE",
                )
            result = self._prepare_action_intent(scope, configured, intent)
            span.set_attribute("research.action_id", str(result.get("action_id") or ""))
            span.set_attribute(
                "research.status",
                str((result.get("execution") or {}).get("status") or "UNKNOWN"),
            )
            return result

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
            self._activate_observability_policy(action, result)
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

    def _activate_observability_policy(
        self, action: dict[str, Any], result: dict[str, Any],
    ) -> None:
        status = str((result.get("execution") or {}).get("status") or "")
        if status != "VERIFIED":
            return
        preflight = action.get("preflight_summary")
        if not isinstance(preflight, dict) or not preflight.get("wandb_cloud_sync"):
            return
        scope = action.get("scope")
        if not isinstance(scope, dict):
            return
        project = str(scope.get("project") or "")
        run_id = str(preflight.get("run_id") or "")
        attempt_id = str(preflight.get("attempt_id") or "")
        if project and run_id and attempt_id:
            self.runtime.observability.enable_cloud(project, run_id, attempt_id)

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
        self._activate_observability_policy(action, result)
        return result

    # ------------------------------------------------------------- projects

    def project_lifecycle_list(self) -> dict[str, Any]:
        return self.project_service.lifecycle_list()

    def project_register(self, project_file: Path) -> dict[str, Any]:
        return self.project_service.register(project_file)

    def project_lifecycle_transition(
        self, project: str, target: ProjectLifecycleState, *, reason: str = "",
    ) -> dict[str, Any]:
        return self.project_service.transition(project, target, reason=reason)

    def project_unregister(self, project: str, *, reason: str = "") -> dict[str, Any]:
        return self.project_service.unregister(project, reason=reason)

    def project_unregister_all(self, *, reason: str = "") -> dict[str, Any]:
        return self.project_service.unregister_all(reason=reason)
