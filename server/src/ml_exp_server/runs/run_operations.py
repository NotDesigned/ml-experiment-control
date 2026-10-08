"""Run operations for ExperimentServerApplication."""
from __future__ import annotations
import json
import re
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any
import yaml
from ..application_errors import ApplicationError
from ..projects.campaign_lifecycle import campaign_snapshot
from ..ingest.runscan import preferred_attempt_id
from .operations import GPU_BUDGET, RESOURCE_APPROVAL, OPERATIONS_BY_ID, OperationAvailability, operations_for_scope
from ..schemas import OperationScope, OperationScopeType, CampaignRelationship, ActionRuntimeConfig, ResearchProject, TERMINAL_RUN_STATES
from .failures import _reviewable_unknown_retry, evidence_digest

class RunOperations:
    """Run operations operations; state belongs to the application facade."""

    def operation_availability(
        self, project: str, scope_type: OperationScopeType | str, object_id: str,
    ) -> list[OperationAvailability]:
        """Return deterministic operation eligibility for one exact scope."""
        scope, configured, resolved = self.resolve_scope(project, scope_type, object_id)
        resource_policy = getattr(
            getattr(self.runtime, "config", None),
            "action_runtime", ActionRuntimeConfig(),
        )
        result: list[OperationAvailability] = []
        for base_operation in operations_for_scope(scope.scope_type):
            parameters = tuple(
                parameter for parameter in base_operation.parameters
                if parameter.key not in {GPU_BUDGET.key, RESOURCE_APPROVAL.key}
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
                metadata={
                    "intent_kind": (
                        {
                            OperationScopeType.CAMPAIGN: "ARCHIVE_CAMPAIGN",
                            OperationScopeType.RUN: "ARCHIVE_RUN",
                            OperationScopeType.ATTEMPT: "ARCHIVE_ATTEMPT",
                        }[scope.scope_type]
                        if operation.operation_id == "object.archive"
                        else operation.intent_kind
                    ),
                    **({
                        "resource_policy": {
                            "approval": resource_policy.scheduler_resource_approval,
                            "max_gpu_hours": (
                                resource_policy.max_gpu_hours_per_action
                                if resource_policy.scheduler_resource_approval == "budget_cap"
                                else None
                            ),
                            "owner": "daemon",
                        },
                    } if operation.operation_id in {"run.submit", "attempt.retry"} else {}),
                },
            ))
        return result

    def _operation_blockers(
        self, operation_id: str, scope: OperationScope,
        project: ResearchProject, resolved: Any,
    ) -> list[str]:
        reasons: list[str] = []
        if operation_id in {
            "campaign.create", "campaign.update", "run.derive", "run.clone",
        }:
            if project.authored_file is None or not Path(project.authored_file).is_file():
                reasons.append("Authored research_project catalog is unavailable")
            if operation_id == "campaign.update" and getattr(resolved, "current_revision", None) is None:
                reasons.append("Campaign has no resolved current revision")
            if operation_id in {"run.derive", "run.clone"} and (
                scope.scope_type == OperationScopeType.RUN
            ):
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
        elif operation_id == "evidence.rebuild_local":
            if not self.runtime.config.action_runtime.allow_local_evidence_rebuild:
                reasons.append("Local evidence rebuild Actions are disabled by daemon policy")
            if project.controller is None:
                reasons.append("Project has no controller configuration")
            elif not project.controller.capabilities.get("refresh_evidence_local"):
                reasons.append(
                    "Project controller does not declare refresh_evidence_local"
                )
            state = str(getattr(resolved, "state", None) or "UNKNOWN").upper()
            if state not in TERMINAL_RUN_STATES:
                reasons.append(f"Attempt state {state} is not terminal")
            run_id, attempt_id = scope.object_id.rsplit("::", 1)
            row = self.runtime.index.get_run(project.project, run_id)
            if row is None:
                reasons.append("Exact Run evidence is unavailable")
            else:
                relationship = row.campaign_binding.relationship
                if relationship != CampaignRelationship.MATCHED:
                    reasons.append(
                        "Run is not exactly bound to the current authored Campaign: "
                        f"{relationship.value}"
                    )
                if not row.campaign:
                    reasons.append("Run has no exact authored Campaign identity")
                attempt_dir = Path(row.run_dir) / "attempts" / attempt_id
                required = (
                    Path(row.run_dir) / "manifest.yaml",
                    attempt_dir / "attempt.yaml",
                    attempt_dir / "backend.json",
                )
                missing = [str(path) for path in required if not path.is_file()]
                if missing:
                    reasons.append(
                        "Exact Attempt durable identity files are unavailable: "
                        + ", ".join(missing)
                    )
                durable = attempt_dir / "collected_run"
                try:
                    has_durable_artifact = durable.is_dir() and any(
                        path.is_file() for path in durable.rglob("*")
                    )
                except OSError:
                    has_durable_artifact = False
                if not has_durable_artifact:
                    reasons.append(
                        "Exact Attempt has no already-local collected_run artifacts"
                    )
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
                policy = getattr(
                    getattr(self.runtime, "config", None),
                    "action_runtime", ActionRuntimeConfig(),
                )
                if (
                    str(decision.get("action") or "").upper() == "DO_NOT_RETRY"
                    and not _reviewable_unknown_retry(
                        decision,
                        getattr(policy, "scheduler_resource_approval", "budget_cap"),
                    )
                ):
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
        resource_approval = "budget_cap"
        budget: float | None = None
        if operation_id in {"run.submit", "attempt.retry"}:
            policy = getattr(
                getattr(self.runtime, "config", None),
                "action_runtime", ActionRuntimeConfig(),
            )
            resource_approval = policy.scheduler_resource_approval
            budget = (
                policy.max_gpu_hours_per_action
                if resource_approval == "budget_cap" else None
            )
        if operation_id == "object.archive":
            if scope.scope_type == OperationScopeType.CAMPAIGN:
                return self.prepare_campaign_archive(project, object_id, reason=reason)
            return self.prepare_object_archive(
                project, scope.scope_type, object_id, reason=reason,
            )
        if operation_id == "evidence.rebuild_local":
            return self.prepare_local_evidence_rebuild(
                project, object_id, reason=reason,
            )
        if operation_id == "run.clone":
            return self.prepare_run_clone(
                project,
                object_id,
                new_run_id=str(parameters.get("new_run_id") or ""),
                profiles=str(parameters.get("profiles") or ""),
                config_overrides=str(parameters.get("config_overrides") or ""),
                reason=reason,
            )
        if operation_id == "run.submit":
            return self.prepare_run_submit(
                project, object_id, max_gpu_hours=budget,
                resource_approval=resource_approval,
                reason=reason or "Requested from the scoped operation catalog",
            )
        if operation_id == "attempt.retry":
            return self.prepare_attempt_retry(
                project, object_id,
                new_attempt_id=(
                    str(parameters["new_attempt_id"])
                    if parameters.get("new_attempt_id") else None
                ),
                max_gpu_hours=budget, reason=reason,
                resource_approval=resource_approval,
            )
        if operation_id == "attempt.cancel":
            return self.prepare_attempt_cancel(project, object_id, reason=reason)
        raise ApplicationError(
            f"operation {operation_id} requires a client-authored intent",
            code="INVALID_OPERATION",
        )

    def _prepare_action_intent(
        self, scope: OperationScope, project: ResearchProject, intent: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            return self.runtime.action_service.prepare(scope, project, intent)
        except (RuntimeError, ValueError) as exc:
            raise ApplicationError(str(exc), code="ACTION_BLOCKED") from exc

    def prepare_run_clone(
        self, project: str, source_run_id: str, *, new_run_id: str,
        profiles: str = "", config_overrides: str = "", reason: str = "",
    ) -> dict[str, Any]:
        """Prepare one deterministic Campaign update from an authored Run."""
        self._require_operation_available(
            "run.clone", project, OperationScopeType.RUN, source_run_id,
        )
        scope, configured, resolved = self.resolve_scope(
            project, OperationScopeType.RUN, source_run_id,
        )
        materializers = [
            binding for binding in getattr(resolved, "campaign_memberships", []) or []
            if binding.membership.kind == "materialize"
        ]
        if len(materializers) != 1:
            raise ApplicationError(
                "source Run must have exactly one authored materializing Campaign",
                code="RUN_NOT_CLONEABLE",
            )
        campaign_name = materializers[0].campaign
        reference = next(
            (item for item in configured.campaigns if item.name == campaign_name), None,
        )
        revision = reference.current_revision if reference is not None else None
        if revision is None:
            raise ApplicationError(
                "source Campaign has no current authored revision",
                code="CAMPAIGN_REVISION_MISSING",
            )
        campaign_path = Path(revision.file)
        if not campaign_path.is_absolute():
            campaign_path = (configured.base_dir or Path(".")) / campaign_path
        try:
            payload = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ApplicationError(
                f"source Campaign is unreadable: {exc}", code="CAMPAIGN_FILE_MISSING",
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
            raise ApplicationError(
                "source Campaign has no materialized runs list",
                code="RUN_NOT_CLONEABLE",
            )
        source = next((
            item for item in payload["runs"]
            if isinstance(item, dict) and item.get("run_id") == source_run_id
        ), None)
        if source is None:
            raise ApplicationError(
                "source Run is not a materialized Campaign entry",
                code="RUN_NOT_CLONEABLE",
            )

        profile_names = [item.strip() for item in profiles.split(",") if item.strip()]
        if any(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", item) is None
            for item in profile_names
        ):
            raise ApplicationError(
                "profiles must contain safe profile identities",
                code="INVALID_OPERATION",
            )
        if len(profile_names) > 1 and "{profile}" not in new_run_id:
            raise ApplicationError(
                "new_run_id must contain {profile} when profiles contains multiple entries",
                code="INVALID_OPERATION",
            )
        if not profile_names:
            profile_names = [""]
        try:
            override_items = json.loads(config_overrides) if config_overrides.strip() else []
        except json.JSONDecodeError as exc:
            raise ApplicationError(
                f"config_overrides must be a JSON list: {exc}",
                code="INVALID_OPERATION",
            ) from exc
        if not isinstance(override_items, list):
            raise ApplicationError(
                "config_overrides must be a JSON list of KEY=VALUE strings",
                code="INVALID_OPERATION",
            )
        normalized_overrides: list[str] = []
        override_keys: set[str] = set()
        for item in override_items:
            key, separator, value = item.partition("=") if isinstance(item, str) else ("", "", "")
            if (
                not separator
                or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", key) is None
                or key in override_keys
            ):
                raise ApplicationError(
                    "config_overrides must use unique safe KEY=VALUE strings",
                    code="INVALID_OPERATION",
                )
            override_keys.add(key)
            normalized_overrides.append(f"{key}={value}")

        existing_ids = {
            str(item.get("run_id") or "")
            for item in payload["runs"] if isinstance(item, dict)
        }
        clones = []
        for profile in profile_names:
            clone_id = new_run_id.replace("{profile}", profile)
            if (
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", clone_id) is None
                or clone_id in existing_ids
            ):
                raise ApplicationError(
                    f"derived Run ID is unsafe or already exists: {clone_id}",
                    code="INVALID_OPERATION",
                )
            clone = deepcopy(source)
            clone["run_id"] = clone_id
            if profile:
                clone["profile"] = [profile]
            current_overrides = [
                str(item) for item in clone.get("config_overrides") or []
            ]
            clone["config_overrides"] = [
                item for item in current_overrides
                if item.partition("=")[0] not in override_keys
            ] + normalized_overrides
            clones.append(clone)
            existing_ids.add(clone_id)
        payload["runs"] = [*payload["runs"], *clones]
        action = self._prepare_action_intent(scope, configured, {
            "kind": "DERIVE_RUN_DRAFT",
            "title": f"Clone {source_run_id} into {len(clones)} authored Run(s)",
            "target": f"campaign://{project}/{campaign_name}",
            "change_summary": (
                reason.strip()
                or "clone an authored Run while preserving all unspecified identity fields"
            ),
            "resource_estimate": "none; this Action only updates the authored Campaign",
            "rationale": reason.strip() or "prepare reviewable Run variants",
            "risk": "new Run definitions become submit candidates after Campaign reindexing",
            "draft": yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            "evidence_digest": evidence_digest(
                self.bounded_evidence(scope, configured, resolved),
            ),
        })
        return {
            "action": action,
            "source_run_id": source_run_id,
            "derived_run_ids": [str(item["run_id"]) for item in clones],
            "profiles": profile_names,
        }

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
                           max_gpu_hours: float | None, reason: str = "",
                           resource_approval: str = "budget_cap") -> dict[str, Any]:
        """Prepare a reviewable first-submission Action for an authored Run."""
        self._require_operation_available(
            "run.submit", project, OperationScopeType.RUN, run_id,
        )
        if resource_approval not in {"budget_cap", "review_exact"}:
            raise ApplicationError("invalid resource approval mode", status_code=422,
                                   code="INVALID_GPU_BUDGET")
        if resource_approval == "budget_cap" and (
            max_gpu_hours is None or max_gpu_hours <= 0
        ):
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
        draft_payload = {
            "campaign_file": str(campaign), "run_id": row.run_id,
            "attempt_id": attempt_id, "resource_approval": resource_approval,
            **(
                {"expected_source_id": row.provenance["source_id"]}
                if row.provenance.get("source_binding") == "campaign_file" else {}
            ),
        }
        if max_gpu_hours is not None:
            draft_payload["max_gpu_hours"] = max_gpu_hours
        draft = yaml.safe_dump(draft_payload, sort_keys=False)
        expected_source = (
            str(row.provenance.get("source_id") or "")
            if row.provenance.get("source_binding") == "campaign_file" else ""
        )
        if expected_source.startswith("source."):
            if self.source_revision_service is None:
                raise ApplicationError(
                    "source revision service is unavailable",
                    code="SOURCE_IMPORT_BLOCKED",
                )
            self.source_revision_service.get(configured.project, expected_source)
        digest = evidence_digest(self.bounded_evidence(scope, configured, row))
        action = self._prepare_action_intent(scope, configured, {
            "kind": "SUBMIT_RUN",
            "title": f"Launch {row.run_id} as {attempt_id}",
            "target": f"run://{configured.project}/{row.run_id}",
            "change_summary": reason or "submit the authored Run's first Attempt",
            "resource_estimate": (
                "exact dry-run resources require human Action confirmation"
                if resource_approval == "review_exact"
                else f"up to {max_gpu_hours:g} GPU-hours"
            ),
            "rationale": "materialize an authored Campaign membership",
            "risk": "scheduler mutation; approval and prepared gates are required before execution",
            "draft": draft,
            "evidence_digest": digest,
        })
        return {"action": action}

    def prepare_attempt_retry(self, project: str, identity: str, *,
                              new_attempt_id: str | None,
                              max_gpu_hours: float | None, reason: str,
                              resource_approval: str = "budget_cap") -> dict[str, Any]:
        if resource_approval not in {"budget_cap", "review_exact"}:
            raise ApplicationError("invalid resource approval mode", status_code=422,
                                   code="INVALID_GPU_BUDGET")
        if resource_approval == "budget_cap" and (
            max_gpu_hours is None or max_gpu_hours <= 0
        ):
            raise ApplicationError("max_gpu_hours must be positive", status_code=422,
                                   code="INVALID_GPU_BUDGET")
        scope, configured, row, attempt, _ = self._attempt_context(project, identity)
        if (attempt.state or "").upper() not in {"FAILED", "PREEMPTED", "CANCELLED"}:
            raise ApplicationError(
                f"attempt state {attempt.state or 'UNKNOWN'} is not retryable",
                code="ATTEMPT_NOT_RETRYABLE",
            )
        decision = attempt.decision if isinstance(attempt.decision, dict) else {}
        decision_action = str(decision.get("action") or "").upper()
        decision_failure = str(decision.get("failure_class") or "").lower()
        allowed = decision.get("retries_allowed")
        used = decision.get("retries_used", 0)
        reviewable_unknown_failure = _reviewable_unknown_retry(
            decision, resource_approval, operator_reason=reason,
        )
        if decision_action == "DO_NOT_RETRY" and not reviewable_unknown_failure:
            raise ApplicationError(
                "attempt decision explicitly forbids retry",
                code="ATTEMPT_RETRY_FORBIDDEN",
            )
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
        draft_payload = {
            "campaign_file": str(campaign), "run_id": row.run_id,
            "source_attempt_id": attempt.attempt_id,
            "attempt_id": new_attempt_id, "resource_approval": resource_approval,
        }
        if reviewable_unknown_failure:
            draft_payload["failure_review"] = {
                "required": True,
                "producer_decision": decision_action,
                "producer_failure_class": decision_failure,
                "operator_reason": reason.strip(),
            }
        if getattr(row, "run_dir", None):
            # Keep legacy Runs coherent until an operator migrates the whole
            # Run directory; newly materialized Runs already live in the
            # daemon root and therefore naturally retain that root on retry.
            draft_payload["local_root"] = str(
                Path(row.run_dir).resolve().parent.parent
            )
        if max_gpu_hours is not None:
            draft_payload["max_gpu_hours"] = max_gpu_hours
        draft = yaml.safe_dump(draft_payload, sort_keys=False)
        result = self._prepare_attempt_action(
            scope, configured, row, attempt, kind="RETRY_ATTEMPT", draft=draft,
            title=f"Retry {row.run_id} from {attempt.attempt_id} as {new_attempt_id}",
            change_summary=reason or "retry failed attempt with a new attempt identity",
            resource_estimate=(
                "exact dry-run resources require human Action confirmation"
                if resource_approval == "review_exact"
                else f"up to {max_gpu_hours:g} GPU-hours"
            ),
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

    def prepare_local_evidence_rebuild(
        self, project: str, identity: str, *, reason: str,
    ) -> dict[str, Any]:
        """Prepare an exact terminal-Attempt local evidence rebuild Action."""
        if not reason.strip():
            raise ApplicationError(
                "local evidence rebuild reason is required", status_code=422,
                code="INVALID_EVIDENCE_REBUILD_REASON",
            )
        self._require_operation_available(
            "evidence.rebuild_local", project, OperationScopeType.ATTEMPT, identity,
        )
        scope, configured, row, attempt, _ = self._attempt_context(project, identity)
        campaign = self._campaign_file(configured, row.campaign)
        run_dir = Path(row.run_dir).resolve()
        local_root = run_dir.parent.parent
        draft = yaml.safe_dump({
            "schema_version": 1,
            "project": project,
            "campaign_file": str(campaign),
            "run_id": row.run_id,
            "attempt_id": attempt.attempt_id,
            "run_dir": str(run_dir),
            "local_root": str(local_root),
            "reason": reason.strip(),
        }, sort_keys=False)
        return self._prepare_attempt_action(
            scope, configured, row, attempt,
            kind="REBUILD_LOCAL_EVIDENCE", draft=draft,
            title=f"Rebuild local evidence for {row.run_id} {attempt.attempt_id}",
            change_summary="recompute collection.json from already-local durable artifacts",
            resource_estimate="local CPU and filesystem only",
            risk="replaces exact Attempt collection.json; no backend or scheduler access",
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
