"""Durable, filesystem-backed state for object-scoped research agents."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from ..schemas import AgentLifecycleState, AgentScope


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _proposal_validation(kind: str, draft: str) -> dict[str, Any]:
    """Validate executable proposal shapes before they enter approval review."""
    if kind not in {
        "CREATE_CAMPAIGN_DRAFT", "UPDATE_CAMPAIGN_DRAFT", "DERIVE_RUN_DRAFT", "SUBMIT_RUN",
        "RETRY_ATTEMPT", "CANCEL_RUN", "RUN_EVALUATION",
        "CREATE_PROJECT_ADAPTER_DRAFT", "CREATE_RESEARCH_QUESTION_DRAFT",
        "COMPLETE_CAMPAIGN", "ARCHIVE_CAMPAIGN", "ARCHIVE_RUN", "ARCHIVE_ATTEMPT",
    }:
        return {"status": "NOT_REQUIRED", "errors": []}
    text = draft.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return {"status": "INVALID", "errors": [f"draft is not valid YAML: {exc}"]}
    if not isinstance(payload, dict):
        return {"status": "INVALID", "errors": ["draft must be a YAML mapping"]}
    errors: list[str] = []
    if kind in {"COMPLETE_CAMPAIGN", "ARCHIVE_CAMPAIGN"}:
        for field in ("project", "campaign", "revision_id"):
            if not str(payload.get(field) or ""):
                errors.append(f"{field} is required")
        if kind == "COMPLETE_CAMPAIGN":
            if not str(payload.get("evidence_digest") or "").startswith("sha256:"):
                errors.append("evidence_digest is required")
            if not str(payload.get("outcome") or ""):
                errors.append("outcome is required")
            if not isinstance(payload.get("assessment"), str):
                errors.append("assessment must be a string")
            if not isinstance(payload.get("membership_run_ids"), list):
                errors.append("membership_run_ids must be a list")
        elif not str(payload.get("reason") or ""):
            errors.append("archive reason is required")
    elif kind == "CREATE_RESEARCH_QUESTION_DRAFT":
        if not _SAFE_ID.fullmatch(str(payload.get("id") or "")):
            errors.append("id is required and must be a safe identity")
        if not str(payload.get("title") or ""):
            errors.append("title is required")
        if payload.get("status") is not None and not isinstance(payload.get("status"), str):
            errors.append("status must be a string")
        if payload.get("links") is not None and not isinstance(payload.get("links"), dict):
            errors.append("links must be a mapping")
        if payload.get("assessments") is not None and not isinstance(
            payload.get("assessments"), list
        ):
            errors.append("assessments must be a list")
    elif kind == "CREATE_PROJECT_ADAPTER_DRAFT":
        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        train = payload.get("train") if isinstance(payload.get("train"), dict) else {}
        container = payload.get("container") if isinstance(payload.get("container"), dict) else {}
        parameters = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}
        outputs = payload.get("outputs") if isinstance(payload.get("outputs"), dict) else {}
        checkpoint = payload.get("checkpoint") if isinstance(payload.get("checkpoint"), dict) else {}
        if not _SAFE_ID.fullmatch(str(payload.get("project") or "")):
            errors.append("project is required and must be a safe identity")
        if not str(payload.get("title") or ""):
            errors.append("title is required")
        if not str(source.get("identity") or "") or not isinstance(source.get("required_paths"), list):
            errors.append("source.identity and source.required_paths list are required")
        if not isinstance(train.get("command"), list) or not train.get("command"):
            errors.append("train.command must be a non-empty argv list")
        allowed_types = {"string", "integer", "number", "float", "boolean"}
        if not parameters or any(
            not isinstance(item, dict)
            or item.get("type") not in allowed_types
            or not isinstance(item.get("required"), bool)
            for item in parameters.values()
        ):
            errors.append("parameters must declare supported type and required boolean")
        if not isinstance(container.get("install_command"), list):
            errors.append("container.install_command must be an argv list")
        if not isinstance(payload.get("backend_profile"), dict):
            errors.append("backend_profile mapping is required")
        if not isinstance(payload.get("assets"), list):
            errors.append("assets must be a list")
        if any(not isinstance(outputs.get(key), str) for key in ("metrics", "checkpoints", "artifacts")):
            errors.append("outputs.metrics/checkpoints/artifacts must be relative path strings")
        if any(checkpoint.get(key) is None for key in (
            "expected_first_minutes", "max_uncheckpointed_minutes"
        )):
            errors.append("checkpoint timing fields are required")
    elif kind in {"ARCHIVE_RUN", "ARCHIVE_ATTEMPT"}:
        if not str(payload.get("project") or ""):
            errors.append("project is required")
        if not _SAFE_ID.fullmatch(str(payload.get("run_id") or "")):
            errors.append("run_id is required and must be a safe identity")
        if kind == "ARCHIVE_ATTEMPT" and not _SAFE_ID.fullmatch(
            str(payload.get("attempt_id") or "")
        ):
            errors.append("attempt_id is required and must be a safe identity")
        if not str(payload.get("reason") or ""):
            errors.append("archive reason is required")
        if not str(payload.get("evidence_digest") or "").startswith("sha256:"):
            errors.append("evidence_digest is required")
    elif kind in {"CREATE_CAMPAIGN_DRAFT", "UPDATE_CAMPAIGN_DRAFT", "DERIVE_RUN_DRAFT"}:
        campaign = str(payload.get("campaign") or "")
        runs = payload.get("runs", [])
        run_refs = payload.get("run_refs", [])
        if payload.get("schema_version") != 1:
            errors.append("schema_version must equal 1")
        if not str(payload.get("project") or ""):
            errors.append("project is required")
        if not _SAFE_ID.fullmatch(campaign):
            errors.append("campaign is required and must be a safe identity")
        if not isinstance(payload.get("research_contract"), dict):
            errors.append("research_contract mapping is required")
        budget = payload.get("budget")
        resources = (
            payload.get("defaults", {}).get("resources")
            if isinstance(payload.get("defaults"), dict) else None
        )
        if isinstance(payload.get("default_resources"), dict):
            errors.append("use defaults.resources; default_resources is not canonical")
        if runs and not (isinstance(budget, dict) or isinstance(resources, dict)):
            errors.append("budget or defaults.resources is required")
        if isinstance(budget, dict):
            max_gpu_hours = budget.get("max_gpu_hours")
            if not isinstance(max_gpu_hours, (int, float)) or max_gpu_hours <= 0:
                errors.append("budget.max_gpu_hours must be a positive number")
        if isinstance(resources, dict):
            gpus = resources.get("gpus")
            if not isinstance(gpus, int) or gpus <= 0:
                errors.append("defaults.resources.gpus must be a positive integer")
        if not isinstance(runs, list):
            errors.append("runs must be a list")
            runs = []
        if not isinstance(run_refs, list):
            errors.append("run_refs must be a list")
            run_refs = []
        if not runs and not run_refs:
            errors.append("runs or run_refs must be a non-empty list")
        else:
            entries = [*runs, *run_refs]
            run_ids = [str(item.get("run_id") or "") if isinstance(item, dict) else ""
                       for item in entries]
            if any(not _SAFE_ID.fullmatch(run_id) for run_id in run_ids):
                errors.append("every run and run_ref requires a safe run_id")
            if len(run_ids) != len(set(run_ids)):
                errors.append("run_id values must be unique")
            if any(isinstance(item, dict) and (
                "role" in item or "membership" in item
            ) for item in entries):
                errors.append(
                    "membership fields must be top-level and use research_role, not role"
                )
            contract = payload.get("research_contract") or {}
            required_roles = contract.get("required_roles", []) \
                if isinstance(contract, dict) else []
            if isinstance(required_roles, list) and required_roles:
                def declared_role(item):
                    if not isinstance(item, dict):
                        return None
                    template = item.get("template")
                    return item.get("research_role") or (
                        template.get("research_role") if isinstance(template, dict) else None
                    )

                declared_roles = [declared_role(item) for item in entries]
                if any(not isinstance(role, str) or not role for role in declared_roles):
                    errors.append(
                        "every membership requires research_role when required_roles is declared"
                    )
                missing_roles = sorted(set(map(str, required_roles)) - set(filter(None, declared_roles)))
                if missing_roles:
                    errors.append(
                        "memberships do not cover required_roles: " + ", ".join(missing_roles)
                    )
    else:
        for field in ("campaign_file", "run_id", "attempt_id"):
            if not str(payload.get(field) or ""):
                errors.append(f"{field} is required")
        if kind == "RETRY_ATTEMPT" and not str(
            payload.get("source_attempt_id") or ""
        ):
            errors.append("source_attempt_id is required")
        if kind != "CANCEL_RUN" and not isinstance(payload.get("max_gpu_hours"), (int, float)):
            errors.append("max_gpu_hours must be numeric")
        if kind == "CANCEL_RUN" and not str(payload.get("backend_job_id") or ""):
            errors.append("backend_job_id is required")
    return {"status": "INVALID" if errors else "VALID", "errors": errors}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


class AgentStore:
    """Persist goals, lifecycle state, chat, journal, and approval proposals.

    Agent records are operational memory. They never replace immutable run
    manifests or evidence files in the science repository.
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def agent_id(scope: AgentScope) -> str:
        raw = f"{scope.project}:{scope.scope_type.value}:{scope.object_id}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        readable = "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in scope.object_id
        ).strip("-")[:48] or "root"
        return f"{scope.project}--{scope.scope_type.value}--{readable}--{digest}"

    def agent_dir(self, scope: AgentScope) -> Path:
        return self.root / self.agent_id(scope)

    def pending_count(self, project: str) -> int:
        """Count pending proposals across every durable scope in one Project."""
        total = 0
        with self._lock:
            for directory in self.root.iterdir():
                if not directory.is_dir():
                    continue
                metadata = _read_json(directory / "scope.json", {})
                if (metadata.get("scope") or {}).get("project") != project:
                    continue
                for path in (directory / "proposals").glob("*.json"):
                    if _read_json(path, {}).get("status") == "PENDING":
                        total += 1
        return total

    def ensure(self, scope: AgentScope, *, default_goal: str) -> dict[str, Any]:
        with self._lock:
            directory = self.agent_dir(scope)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "proposals").mkdir(exist_ok=True)
            (directory / "approvals").mkdir(exist_ok=True)
            (directory / "draft_artifacts").mkdir(exist_ok=True)
            self._migrate_analysis_drafts(directory)
            self._migrate_proposal_validation(directory)
            goal_path = directory / "goal.yaml"
            if not goal_path.exists():
                goal_path.write_text(
                    yaml.safe_dump({"goal": default_goal}, allow_unicode=True, sort_keys=False),
                    encoding="utf-8",
                )
            state_path = directory / "state.json"
            if not state_path.exists():
                _atomic_json(state_path, {
                    "state": AgentLifecycleState.IDLE.value,
                    "current_task": None,
                    "last_error": None,
                    "updated_at": utc_now(),
                })
            conversation_path = directory / "conversation.json"
            if not conversation_path.exists():
                _atomic_json(conversation_path, {
                    "provider": None, "epoch": 0, "thread_id": None, "messages": [],
                })
            metadata_path = directory / "scope.json"
            if not metadata_path.exists():
                _atomic_json(metadata_path, {
                    "agent_id": self.agent_id(scope),
                    "scope": scope.model_dump(mode="json"),
                    "created_at": utc_now(),
                })
            return self.snapshot(scope)

    def begin_provider_epoch(
        self, scope: AgentScope, provider: str, *, preserve_messages: bool = False,
    ) -> dict[str, Any]:
        """Start a clean provider conversation without deleting prior audit history."""
        with self._lock:
            directory = self.agent_dir(scope)
            path = directory / "conversation.json"
            conversation = _read_json(
                path, {"provider": None, "epoch": 0, "thread_id": None, "messages": []},
            )
            if conversation.get("provider") == provider:
                return self.snapshot(scope)
            if conversation.get("provider") is None and preserve_messages:
                conversation["provider"] = provider
                conversation["epoch"] = max(1, int(conversation.get("epoch") or 0))
                _atomic_json(path, conversation)
                self.append_journal(scope, "conversation_epoch_started", {
                    "provider": provider, "epoch": conversation["epoch"],
                    "archived_message_count": 0,
                })
                return self.snapshot(scope)
            epoch = int(conversation.get("epoch") or 0)
            if conversation.get("messages") or conversation.get("thread_id"):
                archive = directory / "conversation_epochs" / f"epoch-{epoch:04d}.json"
                _atomic_json(archive, conversation)
            next_epoch = epoch + 1
            _atomic_json(path, {
                "provider": provider,
                "epoch": next_epoch,
                "thread_id": None,
                "messages": [],
            })
            self.append_journal(scope, "conversation_epoch_started", {
                "provider": provider, "epoch": next_epoch,
                "archived_message_count": len(conversation.get("messages") or []),
            })
            return self.snapshot(scope)

    def create_turn_request(
        self, scope: AgentScope, *, message: str,
        enforce_operation_availability: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            request_id = f"turn-{uuid4().hex[:16]}"
            record = {
                "request_id": request_id,
                "scope": scope.model_dump(mode="json"),
                "message": message,
                "enforce_operation_availability": enforce_operation_availability,
                "status": "PENDING",
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "result": None,
                "error": None,
            }
            _atomic_json(
                self.agent_dir(scope) / "turn_requests" / f"{request_id}.json", record,
            )
            self.append_journal(scope, "turn_requested", {"request_id": request_id})
            return record

    def turn_request(self, scope: AgentScope, request_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"turn-[a-f0-9]{16}", request_id):
            raise ValueError("invalid turn request identity")
        record = _read_json(
            self.agent_dir(scope) / "turn_requests" / f"{request_id}.json", {},
        )
        if not record:
            raise FileNotFoundError(request_id)
        return record

    def set_turn_request(
        self, scope: AgentScope, request_id: str, *, status: str,
        result: dict[str, Any] | None = None, error: str | None = None,
        evidence_digest: str | None = None,
        evidence_captured_at: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            record = self.turn_request(scope, request_id)
            record.update({
                "status": status, "result": result, "error": error,
                "updated_at": utc_now(),
            })
            if evidence_digest is not None:
                record["evidence_digest"] = evidence_digest
            if evidence_captured_at is not None:
                record["evidence_captured_at"] = evidence_captured_at
            if client_id is not None:
                record["client_id"] = client_id
            _atomic_json(
                self.agent_dir(scope) / "turn_requests" / f"{request_id}.json", record,
            )
            self.append_journal(scope, "turn_request_state_changed", {
                "request_id": request_id, "status": status, "error": error,
            })
            return record

    def pending_turn_requests(self, project: str | None = None) -> list[dict[str, Any]]:
        """Return queued client work in creation order; claiming remains explicit."""
        pending: list[dict[str, Any]] = []
        with self._lock:
            for directory in self.root.iterdir():
                if not directory.is_dir():
                    continue
                for path in (directory / "turn_requests").glob("turn-*.json"):
                    record = _read_json(path, {})
                    scope = record.get("scope") if isinstance(record, dict) else None
                    if record.get("status") != "PENDING" or not isinstance(scope, dict):
                        continue
                    if project is not None and scope.get("project") != project:
                        continue
                    pending.append(record)
        return sorted(pending, key=lambda item: str(item.get("created_at") or ""))

    @staticmethod
    def _migrate_analysis_drafts(directory: Path) -> None:
        """Materialize chart/report records from older durable proposals once."""
        for proposal_path in (directory / "proposals").glob("*.json"):
            proposal = _read_json(proposal_path, {})
            kind = proposal.get("kind")
            if kind not in {"CREATE_CHART_SPEC", "CREATE_REPORT_DRAFT"}:
                continue
            prefix = "chart" if kind == "CREATE_CHART_SPEC" else "report"
            artifact_id = proposal.get("artifact_id") or f"{prefix}-{uuid4().hex[:12]}"
            artifact_path = directory / "draft_artifacts" / f"{artifact_id}.json"
            if not artifact_path.exists():
                _atomic_json(artifact_path, {
                    "artifact_id": artifact_id,
                    "artifact_type": prefix,
                    "version": 1,
                    "status": "DRAFT",
                    "proposal_id": proposal.get("proposal_id"),
                    "title": proposal.get("title", "Untitled draft"),
                    "target": proposal.get("target", ""),
                    "content": proposal.get("draft", ""),
                    "evidence_digest": proposal.get("evidence_digest", ""),
                    "created_at": proposal.get("created_at") or utc_now(),
                })
            if not proposal.get("artifact_id"):
                proposal["artifact_id"] = artifact_id
                _atomic_json(proposal_path, proposal)

    @staticmethod
    def _migrate_proposal_validation(directory: Path) -> None:
        for proposal_path in (directory / "proposals").glob("*.json"):
            proposal = _read_json(proposal_path, {})
            validation = _proposal_validation(
                str(proposal.get("kind", "")), str(proposal.get("draft", ""))
            )
            if proposal.get("validation") != validation:
                proposal["validation"] = validation
                _atomic_json(proposal_path, proposal)

    def snapshot(self, scope: AgentScope) -> dict[str, Any]:
        with self._lock:
            directory = self.agent_dir(scope)
            goal_payload: dict[str, Any] = {}
            try:
                loaded = yaml.safe_load((directory / "goal.yaml").read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    goal_payload = loaded
            except OSError:
                pass
            state = _read_json(directory / "state.json", {})
            conversation = _read_json(
                directory / "conversation.json", {"thread_id": None, "messages": []}
            )
            proposals = [
                _read_json(path, {})
                for path in sorted((directory / "proposals").glob("*.json"))
            ] if (directory / "proposals").is_dir() else []
            proposals.sort(key=lambda item: str(item.get("created_at") or ""))
            approvals = [
                _read_json(path, {})
                for path in sorted((directory / "approvals").glob("*.json"))
            ] if (directory / "approvals").is_dir() else []
            draft_artifacts = [
                _read_json(path, {})
                for path in sorted((directory / "draft_artifacts").glob("*.json"))
            ] if (directory / "draft_artifacts").is_dir() else []
            journal = []
            journal_path = directory / "journal.jsonl"
            if journal_path.is_file():
                for line in journal_path.read_text(encoding="utf-8").splitlines()[-100:]:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict):
                        journal.append(item)
            return {
                "agent_id": self.agent_id(scope),
                "scope": scope.model_dump(mode="json"),
                "goal": goal_payload.get("goal", ""),
                **state,
                "thread_id": conversation.get("thread_id"),
                "conversation_provider": conversation.get("provider"),
                "conversation_epoch": int(conversation.get("epoch") or 0),
                "archived_conversation_epochs": len(list(
                    (directory / "conversation_epochs").glob("epoch-*.json")
                )) if (directory / "conversation_epochs").is_dir() else 0,
                "messages": list(conversation.get("messages") or [])[-100:],
                "proposals": [item for item in proposals if item],
                "approvals": [item for item in approvals if item],
                "draft_artifacts": [item for item in draft_artifacts if item],
                "journal": journal,
            }

    def append_journal(self, scope: AgentScope, event: str, payload: dict[str, Any]) -> None:
        with self._lock:
            path = self.agent_dir(scope) / "journal.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {"timestamp": utc_now(), "event": event, "payload": payload}
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def set_state(
        self, scope: AgentScope, state: AgentLifecycleState, *,
        current_task: str | None = None, last_error: str | None = None,
    ) -> None:
        with self._lock:
            _atomic_json(self.agent_dir(scope) / "state.json", {
                "state": state.value,
                "current_task": current_task,
                "last_error": last_error,
                "updated_at": utc_now(),
            })
            self.append_journal(scope, "state_changed", {
                "state": state.value, "current_task": current_task,
                "last_error": last_error,
            })

    def set_goal(self, scope: AgentScope, goal: str) -> None:
        with self._lock:
            path = self.agent_dir(scope) / "goal.yaml"
            path.write_text(
                yaml.safe_dump({"goal": goal}, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            self.append_journal(scope, "goal_updated", {"goal": goal})

    def append_message(
        self, scope: AgentScope, *, role: str, content: str,
        thread_id: str | None = None, evidence_digest: str | None = None,
        evidence_captured_at: str | None = None,
    ) -> None:
        with self._lock:
            path = self.agent_dir(scope) / "conversation.json"
            conversation = _read_json(path, {"thread_id": None, "messages": []})
            if thread_id is not None:
                conversation["thread_id"] = thread_id
            conversation.setdefault("messages", []).append({
                "message_id": uuid4().hex,
                "role": role,
                "content": content,
                "timestamp": utc_now(),
                "evidence_digest": evidence_digest,
                "evidence_captured_at": evidence_captured_at,
            })
            _atomic_json(path, conversation)
            self.append_journal(scope, "message_recorded", {"role": role})

    def add_proposals(
        self, scope: AgentScope, proposals: list[dict[str, Any]], *,
        evidence_digest: str,
    ) -> list[dict[str, Any]]:
        created: list[dict[str, Any]] = []
        with self._lock:
            for proposal in proposals:
                proposal_id = f"proposal-{uuid4().hex[:12]}"
                record = {
                    "proposal_id": proposal_id,
                    "kind": str(proposal.get("kind", "ANALYSIS_ONLY")),
                    "title": str(proposal.get("title", "Untitled proposal")),
                    "target": str(proposal.get("target", "")),
                    "change_summary": str(proposal.get("change_summary", "")),
                    "resource_estimate": str(proposal.get("resource_estimate", "unknown")),
                    "rationale": str(proposal.get("rationale", "")),
                    "risk": str(proposal.get("risk", "")),
                    "draft": str(proposal.get("draft", "")),
                    "status": "PENDING",
                    "evidence_digest": evidence_digest,
                    "created_at": utc_now(),
                }
                record["validation"] = _proposal_validation(
                    record["kind"], record["draft"]
                )
                _atomic_json(self.agent_dir(scope) / "proposals" / f"{proposal_id}.json", record)
                if record["kind"] in {"CREATE_CHART_SPEC", "CREATE_REPORT_DRAFT"}:
                    prefix = "chart" if record["kind"] == "CREATE_CHART_SPEC" else "report"
                    artifact_id = f"{prefix}-{uuid4().hex[:12]}"
                    artifact = {
                        "artifact_id": artifact_id,
                        "artifact_type": prefix,
                        "version": 1,
                        "status": "DRAFT",
                        "proposal_id": proposal_id,
                        "title": record["title"],
                        "target": record["target"],
                        "content": record["draft"],
                        "evidence_digest": evidence_digest,
                        "created_at": record["created_at"],
                    }
                    record["artifact_id"] = artifact_id
                    _atomic_json(
                        self.agent_dir(scope) / "draft_artifacts" / f"{artifact_id}.json",
                        artifact,
                    )
                    _atomic_json(
                        self.agent_dir(scope) / "proposals" / f"{proposal_id}.json", record
                    )
                    self.append_journal(scope, "draft_artifact_created", artifact)
                self.append_journal(scope, "proposal_created", record)
                created.append(record)
        return created

    def decide_proposal(
        self, scope: AgentScope, proposal_id: str, decision: str,
        *, note: str = "",
    ) -> dict[str, Any]:
        if decision not in {"APPROVED", "REJECTED"}:
            raise ValueError("decision must be APPROVED or REJECTED")
        with self._lock:
            proposal_path = self.agent_dir(scope) / "proposals" / f"{proposal_id}.json"
            proposal = _read_json(proposal_path, {})
            if not proposal:
                raise FileNotFoundError(proposal_id)
            if proposal.get("status") != "PENDING":
                raise ValueError("proposal has already been decided")
            if decision == "APPROVED" and proposal.get("validation", {}).get("status") == "INVALID":
                errors = "; ".join(proposal["validation"].get("errors") or [])
                raise ValueError(f"invalid proposal draft cannot be approved: {errors}")
            proposal["status"] = decision
            proposal["decided_at"] = utc_now()
            proposal["decision_note"] = note
            _atomic_json(proposal_path, proposal)
            approval = {
                "proposal_id": proposal_id,
                "decision": decision,
                "note": note,
                "created_at": proposal["decided_at"],
                "execution_enabled": False,
            }
            _atomic_json(
                self.agent_dir(scope) / "approvals" / f"{proposal_id}.json", approval
            )
            self.append_journal(scope, "proposal_decided", approval)
            return approval

    def proposal(self, scope: AgentScope, proposal_id: str) -> dict[str, Any]:
        payload = _read_json(
            self.agent_dir(scope) / "proposals" / f"{proposal_id}.json", {}
        )
        if not payload:
            raise FileNotFoundError(proposal_id)
        return payload
