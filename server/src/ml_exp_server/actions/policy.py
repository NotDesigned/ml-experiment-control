"""Execution-time authorization and mutation policy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..schemas import ActionRuntimeConfig
from .errors import ActionError
from .files import file_sha


@dataclass(frozen=True)
class ExecutionDispatch:
    project_write: bool
    internal_mutation: bool
    local_evidence_rebuild: bool


class ActionExecutionPolicy:
    """Validate immutable authorization and choose one executor boundary."""

    PROJECT_WRITE_OPERATIONS = frozenset({
        "WRITE_RESEARCH_QUESTION",
        "WRITE_CAMPAIGN",
        "WRITE_CAMPAIGN_ARCHIVE",
        "WRITE_RUN_ARCHIVE",
        "WRITE_ATTEMPT_ARCHIVE",
    })
    IDENTITY_PINNED_OPERATIONS = frozenset({
        "SUBMIT_RUN", "RETRY_ATTEMPT", "RUN_EVALUATION",
    })

    def __init__(self, config: ActionRuntimeConfig):
        self.config = config

    def validate(
        self, snapshot: dict[str, Any], confirmation: str,
    ) -> ExecutionDispatch:
        action_id = str(snapshot["action_id"])
        execution = snapshot["execution"]
        status = execution.get("status")
        if status in {"EXECUTING", "RECONCILE_REQUIRED"}:
            raise ActionError(
                "execution intent already exists; reconcile instead of retrying"
            )
        if status != "AUTHORIZED":
            raise ActionError("action requires a separate execution authorization")
        if confirmation != f"EXECUTE {action_id}":
            raise ActionError(f"confirmation must equal EXECUTE {action_id}")
        expires_at = datetime.fromisoformat(
            str(snapshot["gate_expires_at"]).replace("Z", "+00:00")
        )
        if datetime.now(timezone.utc) >= expires_at:
            raise ActionError(
                "gate bundle has expired; prepare and approve a fresh intent"
            )
        if execution.get("authorized_intent_digest") != snapshot.get("intent_digest"):
            raise ActionError("authorization does not match the immutable action intent")
        if execution.get("authorized_gate_bundle_digest") != snapshot.get(
            "gate_bundle_digest"
        ):
            raise ActionError("authorization does not match the gate bundle")

        operation = str(snapshot["operation"])
        project_write = operation in self.PROJECT_WRITE_OPERATIONS
        internal_mutation = operation == "OBSERVABILITY_BACKFILL"
        local_evidence_rebuild = operation == "REBUILD_LOCAL_EVIDENCE"
        if project_write and not self.config.allow_project_writes:
            raise ActionError("project writes are disabled by daemon policy")
        if internal_mutation and not self.config.allow_observability_mutations:
            raise ActionError("observability mutations are disabled by daemon policy")
        if local_evidence_rebuild and not self.config.allow_local_evidence_rebuild:
            raise ActionError("local evidence rebuild Actions are disabled by daemon policy")
        if (
            not project_write
            and not internal_mutation
            and not local_evidence_rebuild
            and not self.config.allow_scheduler_mutations
        ):
            raise ActionError("scheduler mutations are disabled by daemon policy")

        if operation in self.IDENTITY_PINNED_OPERATIONS:
            campaign_path = Path(str(
                snapshot.get("execution_campaign_file")
                or snapshot.get("campaign_file") or ""
            ))
            if file_sha(campaign_path) != snapshot.get("execution_campaign_sha256"):
                raise ActionError(
                    "campaign changed after Action preparation; prepare a fresh action"
                )
            authored_sha = snapshot.get("authored_campaign_sha256")
            if authored_sha is not None and file_sha(
                Path(str(snapshot.get("campaign_file") or ""))
            ) != authored_sha:
                raise ActionError(
                    "authored campaign changed after Action preparation; "
                    "prepare a fresh action"
                )
            manifest_path = Path(str(snapshot.get("execution_manifest_path") or ""))
            if file_sha(manifest_path) != snapshot.get("execution_manifest_sha256"):
                raise ActionError(
                    "canonical execution manifest changed after Action preparation; "
                    "prepare a fresh action"
                )
        return ExecutionDispatch(
            project_write=project_write,
            internal_mutation=internal_mutation,
            local_evidence_rebuild=local_evidence_rebuild,
        )
