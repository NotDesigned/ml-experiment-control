"""Prepare immutable Action plans and execute them after explicit authorization."""

from __future__ import annotations

import difflib
import getpass
import hashlib
import json
import os
import re
import socket
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import yaml
from pydantic import ValidationError

from ..campaign_lifecycle import campaign_record_path
from ..controller_gateway import CommandRunner, ProjectControllerGateway, redact as _redact
from ..intent_protocol import OperationIntent
from ..project_config import load_research_question
from ..operations import intent_scope_error
from ..schemas import (
    ActionRuntimeConfig,
    OperationScope,
    OperationScopeType,
    ResearchQuestion,
    ResearchProject,
)
from ..storage import atomic_json, utc_now
from .store import ActionStore


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
class ActionError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str | None:
    return _sha256(path.read_bytes()) if path.is_file() else None


def _manifest_identity_digest(payload: dict[str, Any]) -> str:
    """Digest every execution-relevant manifest field except creation time."""
    identity = {key: value for key, value in payload.items() if key != "created_at"}
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return _sha256(encoded)


def _canonical_manifest_path(
    campaign: dict[str, Any], *, cwd: Path, run_id: str,
) -> Path | None:
    local_root = campaign.get("local_root")
    campaign_name = campaign.get("campaign")
    if not local_root or not campaign_name:
        return None
    root = Path(str(local_root))
    if not root.is_absolute():
        root = cwd / root
    return (root / str(campaign_name) / run_id / "manifest.yaml").resolve()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _parse_mapping(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1])
    try:
        payload = yaml.safe_load(stripped)
    except yaml.YAMLError as exc:
        raise ActionError(f"draft is not valid YAML/JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ActionError("draft must contain a mapping")
    def json_safe(value: Any) -> Any:
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return value
    return json_safe(payload)


def _unified_diff(path: Path, proposed: str) -> str:
    current = path.read_text(encoding="utf-8") if path.is_file() else ""
    return "".join(difflib.unified_diff(
        current.splitlines(keepends=True), proposed.splitlines(keepends=True),
        fromfile=str(path) if current else "/dev/null", tofile=str(path),
    ))


def _semantic_changes(before: Any, after: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(before, dict) and isinstance(after, dict):
        changes: list[dict[str, Any]] = []
        for key in sorted(set(before) | set(after)):
            path = f"{prefix}.{key}" if prefix else str(key)
            changes.extend(_semantic_changes(before.get(key), after.get(key), path))
        return changes
    if before == after:
        return []
    root = prefix.split(".", 1)[0]
    if root in {"links", "research_contract", "runs", "resolved_config"}:
        category = "EXPERIMENT_DESIGN"
    elif root in {"budget", "resources", "backend"}:
        category = "RESOURCE"
    elif root in {"storage", "command", "assets", "checkpoint", "resume_policy"}:
        category = "EXECUTION_IDENTITY"
    else:
        category = "METADATA"
    return [{
        "path": prefix, "category": category,
        "before": _redact(before), "after": _redact(after),
    }]


def _gate(name: str, passed: bool, detail: str, *, warning: bool = False) -> dict[str, Any]:
    return {
        "name": name,
        "status": "WARNING" if warning else ("PASS" if passed else "FAIL"),
        "detail": detail,
    }


class ActionService:
    def __init__(self, store: ActionStore, config: ActionRuntimeConfig,
                 runner: Callable[..., dict[str, Any]] | None = None,
                 actor_provider: Callable[[], str] | None = None,
                 internal_executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None):
        self.store = store
        self.config = config
        self.controller = ProjectControllerGateway(runner)
        self.actor_provider = actor_provider or self._local_actor
        self.internal_executor = internal_executor

    @staticmethod
    def _local_actor() -> str:
        return f"local-process-owner:uid={os.getuid()}:{getpass.getuser()}@{socket.gethostname()}"

    def prepare(self, scope: OperationScope, project: ResearchProject,
                intent: OperationIntent | dict[str, Any]) -> dict[str, Any]:
        """Validate a client intent and materialize a read-only Action plan.

        Preparation is idempotent and never authorizes execution.  When the
        client omits an idempotency key, the canonical intent content becomes
        the key so transport retries resolve to the same Action.
        """
        if not isinstance(intent, OperationIntent):
            try:
                intent = OperationIntent.model_validate(intent)
            except ValidationError as exc:
                raise ActionError(f"invalid operation intent: {exc}") from exc
        payload = intent.model_dump(mode="json")
        scope_error = intent_scope_error(str(payload["kind"]), scope.scope_type)
        if scope_error:
            raise ActionError(scope_error)
        intent_key = payload.pop("idempotency_key") or self._intent_key(scope, payload)
        request_digest = self._request_digest(scope, payload)
        payload["intent_id"] = intent_key
        action_id = self.store.action_id(scope, intent_key)
        try:
            existing = self.store.snapshot(action_id)
        except FileNotFoundError:
            pass
        else:
            if existing.get("request_digest") != request_digest:
                raise ActionError(
                    "idempotency_key is already bound to a different operation intent"
                )
            return existing
        kind = str(payload["kind"])
        if kind == "CREATE_RESEARCH_QUESTION_DRAFT":
            plan = self._prepare_research_question(action_id, scope, project, payload)
        elif kind in {"CREATE_CAMPAIGN_DRAFT", "UPDATE_CAMPAIGN_DRAFT", "DERIVE_RUN_DRAFT"}:
            plan = self._prepare_campaign(action_id, scope, project, payload)
        elif kind == "ARCHIVE_CAMPAIGN":
            plan = self._prepare_campaign_record(action_id, scope, project, payload)
        elif kind in {"SUBMIT_RUN", "RETRY_ATTEMPT", "CANCEL_RUN", "RUN_EVALUATION"}:
            plan = self._prepare_controller(action_id, scope, project, payload)
        elif kind in {"ARCHIVE_RUN", "ARCHIVE_ATTEMPT"}:
            plan = self._prepare_object_archive(action_id, scope, project, payload)
        elif kind == "OBSERVABILITY_BACKFILL":
            plan = self._prepare_observability_backfill(
                action_id, scope, project, payload,
            )
        else:
            raise ActionError(f"intent kind {kind!r} has no executor")
        plan["request_digest"] = request_digest
        canonical_gates = json.dumps(plan.get("gates", []), sort_keys=True, separators=(",", ":"))
        plan["gate_bundle_digest"] = _sha256(canonical_gates.encode("utf-8"))
        expires = datetime.now(timezone.utc) + timedelta(seconds=self.config.gate_ttl_seconds)
        plan["gate_expires_at"] = expires.isoformat().replace("+00:00", "Z")
        canonical_intent = json.dumps(
            {key: value for key, value in plan.items() if key not in {"intent_digest"}},
            sort_keys=True, separators=(",", ":"), default=str,
        )
        plan["intent_digest"] = _sha256(canonical_intent.encode("utf-8"))
        return self.store.save_plan(plan)

    @staticmethod
    def _intent_key(scope: OperationScope, payload: dict[str, Any]) -> str:
        return "intent-" + ActionService._request_digest(scope, payload).split(":", 1)[1][:20]

    @staticmethod
    def _request_digest(scope: OperationScope, payload: dict[str, Any]) -> str:
        canonical = json.dumps(
            {"scope": scope.model_dump(mode="json"), "intent": payload},
            sort_keys=True, separators=(",", ":"), default=str,
        )
        return _sha256(canonical.encode("utf-8"))

    def _base_plan(self, action_id: str, scope: OperationScope,
                   intent: dict[str, Any], operation: str) -> dict[str, Any]:
        return {
            "action_id": action_id,
            "intent_id": intent["intent_id"],
            "intent_kind": intent["kind"],
            "scope": scope.model_dump(mode="json"),
            "operation": operation,
            "target": intent.get("target", ""),
            "risk": intent.get("risk", ""),
            "evidence_digest": intent.get("evidence_digest", ""),
            "created_at": utc_now(),
        }

    def _prepare_research_question(self, action_id: str, scope: OperationScope,
                            project: ResearchProject,
                            intent: dict[str, Any]) -> dict[str, Any]:
        payload = _parse_mapping(str(intent.get("draft", "")))
        research_question_id = str(payload.get("id") or "")
        if not _SAFE_ID.fullmatch(research_question_id):
            raise ActionError("research_question id is not a safe file identity")
        if not project.research_questions_dir:
            raise ActionError("project has no research_questions_dir")
        root = Path(project.research_questions_dir)
        if not root.is_absolute():
            root = (project.base_dir or Path(".")) / root
        target = root.resolve() / f"{research_question_id}.yml"
        schema_error = ""
        try:
            research_question = ResearchQuestion.model_validate(payload)
        except ValidationError as exc:
            research_question = None
            schema_error = "; ".join(
                f"{'.'.join(map(str, item['loc']))}: {item['msg']}" for item in exc.errors()
            )
        proposed = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
        current_payload = _parse_mapping(target.read_text(encoding="utf-8")) if target.is_file() else {}
        gates = [
            _gate("schema", research_question is not None,
                  "research_question YAML validates against schema v1" if research_question else schema_error),
            _gate("safe_target", _inside(target, root), str(target)),
        ]
        plan = self._base_plan(action_id, scope, intent, "WRITE_RESEARCH_QUESTION")
        plan.update({
            "target_path": str(target), "expected_sha256": _file_sha(target),
            "proposed_content": proposed, "diff": _unified_diff(target, proposed),
            "semantic_changes": _semantic_changes(current_payload, payload),
            "gates": gates, "ready": not any(g["status"] == "FAIL" for g in gates),
            "command_preview": ["atomic-write", str(target)],
        })
        return plan

    def _prepare_campaign(self, action_id: str, scope: OperationScope,
                          project: ResearchProject,
                          intent: dict[str, Any]) -> dict[str, Any]:
        payload = _parse_mapping(str(intent.get("draft", "")))
        campaign_id = str(payload.get("campaign") or "")
        if not _SAFE_ID.fullmatch(campaign_id):
            raise ActionError("campaign draft requires a safe campaign identity")
        if payload.get("project") != project.project:
            raise ActionError("campaign project does not match operation scope")
        runs = payload.get("runs", [])
        run_refs = payload.get("run_refs", [])
        if not isinstance(runs, list) or not isinstance(run_refs, list) or not (runs or run_refs):
            raise ActionError("campaign draft requires non-empty runs or run_refs")
        entries = [*runs, *run_refs]
        run_ids = [str(item.get("run_id", "")) for item in entries if isinstance(item, dict)]
        unique = (
            len(run_ids) == len(entries) == len(set(run_ids))
            and all(_SAFE_ID.fullmatch(item) for item in run_ids)
        )
        intent_kind = str(intent.get("kind") or "")
        is_update = intent_kind == "UPDATE_CAMPAIGN_DRAFT"
        root = (project.base_dir or Path(".")) / "experiments" / "campaigns"
        reference = next((item for item in project.campaigns if item.name == campaign_id), None)
        if is_update and reference is not None and reference.current_revision is not None:
            target = Path(reference.current_revision.file).resolve()
        else:
            target = root.resolve() / f"{campaign_id}.yml"
        if is_update and (
            scope.scope_type != OperationScopeType.CAMPAIGN or scope.object_id != campaign_id
        ):
            raise ActionError("Campaign update must use the exact existing Campaign scope")
        proposed = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
        current_payload = _parse_mapping(target.read_text(encoding="utf-8")) if target.is_file() else {}
        canonical_memberships = all(
            isinstance(item, dict) and "role" not in item and "membership" not in item
            for item in entries
        )
        gates = [
            _gate("schema", payload.get("schema_version") == 1,
                  "campaign schema_version must equal 1"),
            _gate("safe_target", _inside(target, root), str(target)),
            _gate("operation_semantics",
                  target.is_file() if is_update else not target.exists() and reference is None,
                  "update requires an existing Campaign; create requires a new identity"),
            _gate("immutable_run_ids", unique,
                  "all run_id values are safe and unique"),
            _gate("membership_schema", canonical_memberships,
                  "membership fields are top-level and use research_role, not role"),
            _gate("resource_budget", not runs or bool(
                payload.get("budget") or payload.get("defaults", {}).get("resources")
            ), "campaign materialization declares a budget or default resources"),
        ]
        plan = self._base_plan(action_id, scope, intent, "WRITE_CAMPAIGN")
        plan.update({
            "target_path": str(target), "expected_sha256": _file_sha(target),
            "proposed_content": proposed, "diff": _unified_diff(target, proposed),
            "semantic_changes": _semantic_changes(current_payload, payload),
            "gates": gates, "ready": all(g["status"] != "FAIL" for g in gates),
            "command_preview": ["atomic-write", str(target)],
        })
        if not is_update:
            project_file = project.authored_file
            if project_file is None or not project_file.is_file():
                plan["gates"].append(_gate(
                    "project_catalog", False,
                    "authored research_project.yaml path is unavailable",
                ))
                plan["ready"] = False
            else:
                catalog = _parse_mapping(project_file.read_text(encoding="utf-8"))
                entries = catalog.get("campaigns")
                entries = list(entries) if isinstance(entries, list) else []
                entries.append({
                    "name": campaign_id,
                    "file": str(target.relative_to(project.base_dir or project_file.parent)),
                })
                catalog["campaigns"] = entries
                catalog_content = yaml.safe_dump(catalog, allow_unicode=True, sort_keys=False)
                plan["files"] = [
                    {"path": str(target), "expected_sha256": _file_sha(target),
                     "content": proposed},
                    {"path": str(project_file), "expected_sha256": _file_sha(project_file),
                     "content": catalog_content},
                ]
                plan["diff"] += _unified_diff(project_file, catalog_content)
                plan["command_preview"] = [
                    "atomic-write", str(target), "and-register", str(project_file),
                ]
        return plan

    def _prepare_campaign_record(self, action_id: str, scope: OperationScope,
                                 project: ResearchProject,
                                 intent: dict[str, Any]) -> dict[str, Any]:
        payload = _parse_mapping(str(intent.get("draft", "")))
        kind = str(intent.get("kind"))
        campaign = str(payload.get("campaign") or "")
        revision_id = str(payload.get("revision_id") or "")
        if payload.get("project") != project.project:
            raise ActionError("Campaign record project does not match operation scope")
        if scope.scope_type != OperationScopeType.CAMPAIGN or scope.object_id != campaign:
            raise ActionError("Campaign record must be prepared in its exact Campaign scope")
        reference = next((item for item in project.campaigns if item.name == campaign), None)
        current = reference.current_revision if reference else None
        if current is None or current.revision_id != revision_id:
            raise ActionError("Campaign record revision is not the current authored revision")
        if kind != "ARCHIVE_CAMPAIGN":
            raise ActionError("only Campaign archive records are supported")
        target = campaign_record_path(project, campaign, revision_id, "archive")
        record = {
            **payload,
            "schema_version": 1,
            "recorded_at": utc_now(),
            "record_kind": "archive",
        }
        proposed = yaml.safe_dump(record, allow_unicode=True, sort_keys=False)
        gates = [
            _gate("exact_scope", True, f"{project.project}/{campaign}@{revision_id}"),
            _gate("record_absent", not target.exists(),
                  "Campaign lifecycle record is immutable and does not already exist"),
            _gate("archive_binding", bool(payload.get("reason")),
                  "archive binds the exact Campaign revision and explicit reason"),
        ]
        plan = self._base_plan(action_id, scope, intent, "WRITE_CAMPAIGN_ARCHIVE")
        plan.update({
            "target_path": str(target),
            "expected_sha256": _file_sha(target),
            "proposed_content": proposed,
            "gates": gates,
            "ready": all(gate["status"] != "FAIL" for gate in gates),
            "command_preview": ["atomic-write", str(target)],
        })
        return plan

    def _prepare_object_archive(self, action_id: str, scope: OperationScope,
                                project: ResearchProject,
                                intent: dict[str, Any]) -> dict[str, Any]:
        payload = _parse_mapping(str(intent.get("draft", "")))
        kind = str(intent.get("kind"))
        run_id = str(payload.get("run_id") or "")
        attempt_id = str(payload.get("attempt_id") or "")
        expected_scope = OperationScopeType.RUN if kind == "ARCHIVE_RUN" else OperationScopeType.ATTEMPT
        expected_object = run_id if kind == "ARCHIVE_RUN" else f"{run_id}::{attempt_id}"
        if payload.get("project") != project.project:
            raise ActionError("archive record project does not match operation scope")
        if scope.scope_type != expected_scope or scope.object_id != expected_object:
            raise ActionError("archive record must be prepared in its exact object scope")
        if not _SAFE_ID.fullmatch(run_id) or (
            kind == "ARCHIVE_ATTEMPT" and not _SAFE_ID.fullmatch(attempt_id)
        ):
            raise ActionError("archive record requires safe Run/Attempt identities")
        record_type = "run" if kind == "ARCHIVE_RUN" else "attempt"
        filename = f"{run_id}.yml" if kind == "ARCHIVE_RUN" else f"{run_id}--{attempt_id}.yml"
        root = (project.base_dir or Path(".")) / "experiments" / "archive_records" / f"{record_type}s"
        target = root.resolve() / filename
        record = {**payload, "schema_version": 1, "record_type": record_type,
                  "recorded_at": utc_now()}
        proposed = yaml.safe_dump(record, allow_unicode=True, sort_keys=False)
        gates = [
            _gate("exact_scope", True, f"{scope.scope_type.value}:{scope.object_id}"),
            _gate("safe_target", _inside(target, root), str(target)),
            _gate("record_absent", not target.exists(),
                  "archive records are immutable and append-only"),
            _gate("evidence_bound", str(payload.get("evidence_digest") or "").startswith("sha256:"),
                  "archive record binds the exact bounded-evidence digest"),
            _gate("reason", bool(str(payload.get("reason") or "").strip()),
                  "archive reason is required"),
        ]
        operation = "WRITE_RUN_ARCHIVE" if kind == "ARCHIVE_RUN" else "WRITE_ATTEMPT_ARCHIVE"
        plan = self._base_plan(action_id, scope, intent, operation)
        plan.update({
            "target_path": str(target), "expected_sha256": _file_sha(target),
            "proposed_content": proposed, "gates": gates,
            "ready": all(gate["status"] != "FAIL" for gate in gates),
            "command_preview": ["atomic-write", str(target)],
        })
        return plan

    def _prepare_observability_backfill(
        self, action_id: str, scope: OperationScope,
        project: ResearchProject, intent: dict[str, Any],
    ) -> dict[str, Any]:
        payload = _parse_mapping(str(intent.get("draft", "")))
        target_kind = str(payload.get("target") or "")
        attempts = payload.get("attempts")
        reason = str(payload.get("reason") or "").strip()
        valid_attempts = (
            isinstance(attempts, list) and 0 < len(attempts) <= 500
            and all(
                isinstance(item, dict)
                and _SAFE_ID.fullmatch(str(item.get("run_id") or "")) is not None
                and _SAFE_ID.fullmatch(str(item.get("attempt_id") or "")) is not None
                for item in attempts
            )
        )
        identities = [
            {"run_id": str(item["run_id"]), "attempt_id": str(item["attempt_id"])}
            for item in attempts or [] if isinstance(item, dict)
        ] if valid_attempts else []
        gates = [
            _gate("exact_project", payload.get("project") == project.project,
                  project.project),
            _gate("target", target_kind in {"local", "cloud"}, target_kind),
            _gate("bounded_attempts", bool(valid_attempts),
                  f"attempt_count={len(attempts) if isinstance(attempts, list) else 0}; max=500"),
            _gate("reason", bool(reason), "backfill reason is required"),
            _gate("evidence_reference", bool(intent.get("evidence_digest")),
                  "intent is bound to bounded scope evidence"),
        ]
        plan = self._base_plan(
            action_id, scope, intent, "OBSERVABILITY_BACKFILL",
        )
        plan.update({
            "target_kind": target_kind,
            "attempts": identities,
            "reason": reason,
            "gates": gates,
            "ready": all(item["status"] != "FAIL" for item in gates),
            "preflight_summary": {
                "target": target_kind, "attempt_count": len(identities),
            },
            "command_preview": [
                "daemon-observability-backfill", target_kind,
                f"attempts={len(identities)}",
            ],
        })
        return plan

    def _run_gate(self, name: str, command: list[str], cwd: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        result = self.controller.execute_command(
            command, cwd=cwd, timeout=self.config.timeout_seconds,
        )
        passed = result.get("returncode") == 0 and not result.get("timeout")
        detail = "controller check passed" if passed else str(
            result.get("stderr") or result.get("stdout") or "controller check failed"
        )[:500]
        return _gate(name, passed, detail), result

    def _prepare_controller(self, action_id: str, scope: OperationScope,
                            project: ResearchProject,
                            intent: dict[str, Any]) -> dict[str, Any]:
        if project.controller is None:
            raise ActionError("project has no controller configuration")
        spec = _parse_mapping(str(intent.get("draft", "")))
        campaign_value = str(spec.get("campaign_file") or "")
        run_id = str(spec.get("run_id") or "")
        attempt_id = str(spec.get("attempt_id") or "attempt-001")
        if not _SAFE_ID.fullmatch(run_id) or not re.fullmatch(r"attempt-[0-9]{3,}", attempt_id):
            raise ActionError("action draft requires safe run_id and attempt_id")
        base = (project.base_dir or Path(".")).resolve()
        campaign = Path(campaign_value)
        if not campaign.is_absolute():
            campaign = base / campaign
        campaign = campaign.resolve()
        experiments = base / "experiments"
        if not campaign.is_file() or not _inside(campaign, experiments):
            raise ActionError("campaign_file must exist under the project's experiments directory")
        kind = str(intent["kind"])
        operation = {
            "SUBMIT_RUN": "SUBMIT_RUN", "RETRY_ATTEMPT": "RETRY_ATTEMPT",
            "CANCEL_RUN": "CANCEL_RUN", "RUN_EVALUATION": "RUN_EVALUATION",
        }[kind]
        if scope.scope_type == OperationScopeType.ATTEMPT:
            source_run_id, source_attempt_id = scope.object_id.rsplit("::", 1)
            if run_id != source_run_id:
                raise ActionError(
                    "attempt-scoped action run_id must equal the scoped run_id"
                )
            if operation == "SUBMIT_RUN":
                raise ActionError("SUBMIT_RUN is not valid in attempt scope")
            if operation == "CANCEL_RUN" and attempt_id != source_attempt_id:
                raise ActionError(
                    "attempt-scoped cancellation must target the scoped attempt_id"
                )
            if operation in {"RETRY_ATTEMPT", "RUN_EVALUATION"}:
                if str(spec.get("source_attempt_id") or "") != source_attempt_id:
                    raise ActionError(
                        "source_attempt_id must equal the scoped attempt_id"
                    )
                if operation == "RETRY_ATTEMPT" and attempt_id == source_attempt_id:
                    raise ActionError("retry must allocate a new attempt_id")
        plan = self._base_plan(action_id, scope, intent, operation)
        preflight_summary: dict[str, Any] = {}
        gates = [
            _gate("safe_target", True, f"{campaign} :: {run_id} :: {attempt_id}"),
            _gate("evidence_reference", bool(intent.get("evidence_digest")),
                  "intent is bound to an evidence digest"),
        ]
        if operation == "RUN_EVALUATION" and not project.controller.capabilities.get("evaluation_as_run"):
            gates.append(_gate("controller_capability", False,
                               "project controller does not declare an evaluate verb"))
            plan.update({
                "campaign_file": str(campaign), "run_id": run_id,
                "attempt_id": attempt_id, "gates": gates, "ready": False,
                "diff": "", "command_preview": [],
            })
            return plan

        actual_verb = "cancel" if operation == "CANCEL_RUN" else "submit"
        call = self.controller.build(
            project, campaign, actual_verb, run_id, attempt_id=attempt_id,
        )
        command, cwd = call.argv, call.cwd
        preview_payload: dict[str, Any] | None = None
        observed: dict[str, Any] = {}
        execution_manifest_path: Path | None = None
        execution_manifest_sha256: str | None = None
        if actual_verb == "submit":
            raw = _parse_mapping(campaign.read_text(encoding="utf-8"))
            preview_campaign = deepcopy(raw)
            preview_root = self.store.directory(action_id) / "preview_runs"
            preview_campaign["local_root"] = str(preview_root)
            preview_path = self.store.directory(action_id) / "campaign.preview.yml"
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            preview_path.write_text(
                yaml.safe_dump(preview_campaign, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            preview_call = self.controller.build(
                project, preview_path, "submit", run_id,
                attempt_id=attempt_id, dry_run=True,
            )
            preview_command, preview_cwd = preview_call.argv, preview_call.cwd
            gate, result = self._run_gate("dry_run", preview_command, preview_cwd)
            gates.append(gate)
            payload = result.get("payload")
            if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                preview_payload = payload[0]
            manifest_path = Path(str((preview_payload or {}).get("manifest_path", "")))
            manifest: dict[str, Any] = {}
            if manifest_path.is_file() and _inside(manifest_path, self.store.directory(action_id)):
                loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    manifest = loaded
            preview_identity = _manifest_identity_digest(manifest)
            execution_manifest_path = _canonical_manifest_path(
                raw, cwd=cwd, run_id=run_id,
            )
            existing_manifest: dict[str, Any] | None = None
            if execution_manifest_path is not None and execution_manifest_path.is_file():
                loaded = yaml.safe_load(
                    execution_manifest_path.read_text(encoding="utf-8")
                )
                if isinstance(loaded, dict):
                    existing_manifest = loaded
                execution_manifest_sha256 = _file_sha(execution_manifest_path)
            execution_identity_matches = (
                execution_manifest_path is not None
                and (
                    existing_manifest is None
                    and execution_manifest_sha256 is None
                    or existing_manifest is not None
                    and _manifest_identity_digest(existing_manifest) == preview_identity
                )
            )
            identity_ok = all(manifest.get(key) for key in ("run_id", "source_id", "image_id"))
            storage_ok = isinstance(manifest.get("storage"), dict) and bool(manifest["storage"].get("run_dir"))
            run_identity_complete = all(
                isinstance(manifest.get(key), dict) and bool(manifest.get(key))
                for key in ("backend", "resources", "storage")
            ) and bool(manifest.get("command")) and manifest.get("identity_version") == 2
            asset_identity_frozen = isinstance(manifest.get("assets"), list) and bool(manifest.get("assets"))
            gates.extend([
                _gate("identity", identity_ok,
                      "run_id, source_id, and immutable image_id are frozen"),
                _gate("storage", storage_ok,
                      "shared run_dir is frozen in preview manifest"),
                _gate("run_identity_complete", run_identity_complete,
                      "Run manifest v2 must freeze backend, resources, full storage, and rendered command"),
                _gate("asset_identity_frozen", asset_identity_frozen,
                      "Run manifest must freeze model/dataset/checkpoint asset identities"),
                _gate(
                    "execution_manifest_match", execution_identity_matches,
                    "no canonical execution manifest exists yet"
                    if execution_manifest_sha256 is None
                    else "canonical execution manifest identity matches preview"
                    if execution_identity_matches
                    else "canonical execution manifest identity conflicts with preview",
                ),
            ])
            if operation == "RUN_EVALUATION":
                evaluation = manifest.get("evaluation")
                evaluation_ready = isinstance(evaluation, dict) and all(
                    evaluation.get(key)
                    for key in ("checkpoint_digest", "spec_digest", "output_namespace")
                )
                gates.append(_gate(
                    "evaluation_identity", evaluation_ready,
                    "evaluation-as-run must freeze checkpoint_digest, spec_digest, and output_namespace",
                ))
            resources = manifest.get("resources") if isinstance(manifest.get("resources"), dict) else {}
            budget_limit = spec.get("max_gpu_hours")
            requested_gpu_hours = self._gpu_hours(resources, manifest.get("backend"))
            budget_ok = (
                isinstance(budget_limit, (int, float))
                and requested_gpu_hours is not None
                and requested_gpu_hours <= float(budget_limit)
            )
            gates.append(_gate("budget", budget_ok,
                               f"requested_gpu_hours={requested_gpu_hours}; max_gpu_hours={budget_limit}"))
            backend = manifest.get("backend") if isinstance(manifest.get("backend"), dict) else {}
            preemptible = bool(backend.get("preemptible")) or backend.get("kind") == "sensecore"
            checkpoint = manifest.get("checkpoint") if isinstance(manifest.get("checkpoint"), dict) else {}
            exposure_ok = not preemptible or all(
                checkpoint.get(key) is not None
                for key in ("expected_first_minutes", "max_uncheckpointed_minutes")
            )
            gates.append(_gate("checkpoint_exposure", exposure_ok,
                               "preemptible work must declare first-checkpoint and uncheckpointed exposure"))
            preflight_summary = _redact({
                "campaign_file": str(campaign),
                "run_id": run_id,
                "attempt_id": attempt_id,
                "source_id": manifest.get("source_id"),
                "image_id": manifest.get("image_id"),
                "config_path": manifest.get("config_path"),
                "resolved_config": manifest.get("resolved_config"),
                "backend": backend,
                "resources": resources,
                "storage": manifest.get("storage"),
                "command": manifest.get("command"),
                "assets": manifest.get("assets"),
                "checkpoint": checkpoint,
                "requested_gpu_hours": requested_gpu_hours,
                "max_gpu_hours": budget_limit,
                "wandb_cloud_sync": bool(spec.get("wandb_cloud_sync", False)),
            })
            for name, verb, extra in (
                ("preflight", "preflight", ["--scope", "submit"]),
                ("duplicate_run_identity", "check-identity", []),
                ("assets", "assets-verify", []),
            ):
                check_call = self.controller.build(
                    project, campaign, verb, run_id,
                    attempt_id=attempt_id, extra=extra,
                )
                gate, _ = self._run_gate(name, check_call.argv, check_call.cwd)
                gates.append(gate)
            capability = project.controller.capabilities
            gates.append(_gate(
                "submit_outbox_capability",
                bool(capability.get("submit_outbox")) and bool(capability.get("run_identity_v2")),
                "controller must declare durable submit reconciliation and Run identity v2",
            ))
        else:
            requested_job_id = spec.get("backend_job_id")
            status_call = self.controller.build(
                project, campaign, "status", run_id, attempt_id=attempt_id,
            )
            status_gate, status_result = self._run_gate(
                "current_scheduler_status", status_call.argv, status_call.cwd,
            )
            gates.append(status_gate)
            status_payload = status_result.get("payload")
            observed = status_payload[0] if isinstance(status_payload, list) and status_payload else {}
            exact = (
                isinstance(observed, dict)
                and str(observed.get("backend_job_id")) == str(requested_job_id)
                and bool(requested_job_id)
            )
            gates.append(_gate("backend_job_identity", exact,
                               f"requested={requested_job_id}; observed={_redact(observed)}"))
            gates.append(_gate(
                "cancel_outbox_capability",
                bool(project.controller.capabilities.get("cancel_outbox")),
                "controller must declare durable cancel intent and reconciliation",
            ))

        plan.update({
            "campaign_file": str(campaign), "run_id": run_id,
            "attempt_id": attempt_id, "backend_job_id": spec.get("backend_job_id"),
            "gates": gates, "ready": all(item["status"] != "FAIL" for item in gates),
            "diff": json.dumps(
                _redact(preview_payload if preview_payload else observed),
                ensure_ascii=False, indent=2,
            ),
            "semantic_changes": _semantic_changes({}, manifest) if actual_verb == "submit" else [],
            "preflight_summary": preflight_summary,
            "command_preview": _redact(command),
            "cwd": str(cwd),
            "execution_campaign_sha256": _file_sha(campaign),
            "execution_manifest_path": (
                str(execution_manifest_path) if execution_manifest_path is not None else None
            ),
            "execution_manifest_sha256": execution_manifest_sha256,
        })
        if actual_verb == "submit":
            verification = self.controller.build(
                project, campaign, "status", run_id, attempt_id=attempt_id,
            )
            plan.update({
                "verification_command_preview": _redact(verification.argv),
                "verification_cwd": str(verification.cwd),
            })
        return plan

    @staticmethod
    def _gpu_hours(resources: Any, backend: Any) -> float | None:
        if not isinstance(resources, dict) or not isinstance(backend, dict):
            return None
        gpus = resources.get("gpus")
        time_value = backend.get("time") or resources.get("max_time")
        if not isinstance(gpus, (int, float)) or not isinstance(time_value, str):
            return None
        parts = time_value.split(":")
        try:
            if len(parts) == 3:
                hours = int(parts[0]) + int(parts[1]) / 60 + int(parts[2]) / 3600
            elif time_value.endswith("h"):
                hours = float(time_value[:-1])
            else:
                return None
        except ValueError:
            return None
        return round(float(gpus) * hours, 4)

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
        return self.store.set_execution(
            action_id, execution, event="execution_authorized",
        )

    def execute(self, action_id: str, confirmation: str) -> dict[str, Any]:
        snapshot = self.store.snapshot(action_id)
        execution = snapshot["execution"]
        if execution.get("status") == "VERIFIED":
            return snapshot
        if execution.get("status") in {"EXECUTING", "RECONCILE_REQUIRED"}:
            raise ActionError("execution intent already exists; reconcile instead of retrying")
        if execution.get("status") != "AUTHORIZED":
            raise ActionError("action requires a separate execution authorization")
        if confirmation != f"EXECUTE {action_id}":
            raise ActionError(f"confirmation must equal EXECUTE {action_id}")
        expires_at = datetime.fromisoformat(str(snapshot["gate_expires_at"]).replace("Z", "+00:00"))
        if datetime.now(timezone.utc) >= expires_at:
            raise ActionError("gate bundle has expired; prepare and approve a fresh intent")
        if execution.get("authorized_intent_digest") != snapshot.get("intent_digest"):
            raise ActionError("authorization does not match the immutable action intent")
        if execution.get("authorized_gate_bundle_digest") != snapshot.get("gate_bundle_digest"):
            raise ActionError("authorization does not match the gate bundle")
        operation = str(snapshot["operation"])
        project_write = operation in {
            "WRITE_RESEARCH_QUESTION", "WRITE_CAMPAIGN", "WRITE_CAMPAIGN_ARCHIVE",
            "WRITE_RUN_ARCHIVE", "WRITE_ATTEMPT_ARCHIVE",
        }
        internal_mutation = operation == "OBSERVABILITY_BACKFILL"
        if project_write and not self.config.allow_project_writes:
            raise ActionError("project writes are disabled by daemon policy")
        if internal_mutation and not self.config.allow_observability_mutations:
            raise ActionError("observability mutations are disabled by daemon policy")
        if (
            not project_write and not internal_mutation
            and not self.config.allow_scheduler_mutations
        ):
            raise ActionError("scheduler mutations are disabled by daemon policy")
        if operation in {"SUBMIT_RUN", "RETRY_ATTEMPT", "RUN_EVALUATION"}:
            campaign_path = Path(str(snapshot.get("campaign_file") or ""))
            if _file_sha(campaign_path) != snapshot.get("execution_campaign_sha256"):
                raise ActionError(
                    "campaign changed after Action preparation; prepare a fresh action"
                )
            manifest_path = Path(str(snapshot.get("execution_manifest_path") or ""))
            if _file_sha(manifest_path) != snapshot.get("execution_manifest_sha256"):
                raise ActionError(
                    "canonical execution manifest changed after Action preparation; "
                    "prepare a fresh action"
                )
        try:
            self.store.claim_execution(action_id)
        except RuntimeError as exc:
            raise ActionError(str(exc)) from exc
        execution.update({"status": "EXECUTING", "started_at": utc_now(), "error": None})
        self.store.set_execution(action_id, execution, event="execution_started")
        if project_write:
            return self._execute_write(snapshot, execution)
        if internal_mutation:
            return self._execute_internal(snapshot, execution)
        return self._execute_controller(snapshot, execution)

    def _execute_internal(
        self, plan: dict[str, Any], execution: dict[str, Any],
    ) -> dict[str, Any]:
        if self.internal_executor is None:
            raise ActionError("daemon internal executor is unavailable")
        try:
            result = self.internal_executor(plan)
        except Exception as exc:
            execution.update({
                "status": "RECONCILE_REQUIRED", "finished_at": utc_now(),
                "error": type(exc).__name__, "result": None,
            })
            return self.store.set_execution(
                plan["action_id"], execution,
                event="internal_execution_reconcile_required",
            )
        execution.update({
            "status": "VERIFIED", "finished_at": utc_now(),
            "error": None, "result": _redact(result),
        })
        return self.store.set_execution(
            plan["action_id"], execution, event="internal_execution_verified",
        )

    def _execute_write(self, plan: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
        if plan.get("files"):
            return self._execute_multi_write(plan, execution)
        target = Path(plan["target_path"])
        if _file_sha(target) != plan.get("expected_sha256"):
            execution.update({
                "status": "FAILED", "finished_at": utc_now(),
                "error": "target changed after diff preparation; prepare a new action",
            })
            return self.store.set_execution(plan["action_id"], execution, event="execution_failed")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        temporary.write_text(plan["proposed_content"], encoding="utf-8")
        os.replace(temporary, target)
        if plan["operation"] == "WRITE_RESEARCH_QUESTION":
            load_research_question(target)
        execution.update({
            "status": "VERIFIED", "finished_at": utc_now(),
            "result": {"target_path": str(target), "sha256": _file_sha(target)},
        })
        return self.store.set_execution(plan["action_id"], execution, event="execution_verified")

    def _execute_multi_write(
        self, plan: dict[str, Any], execution: dict[str, Any],
    ) -> dict[str, Any]:
        """Atomically apply a bounded set of files from one reviewed action."""
        files = plan.get("files") or []
        if not files or any(
            _file_sha(Path(item["path"])) != item.get("expected_sha256") for item in files
        ):
            execution.update({
                "status": "FAILED", "finished_at": utc_now(),
                "error": "one or more targets changed after diff preparation",
            })
            return self.store.set_execution(
                plan["action_id"], execution, event="execution_failed",
            )
        originals = {
            Path(item["path"]): (
                Path(item["path"]).read_bytes() if Path(item["path"]).is_file() else None
            )
            for item in files
        }
        temporaries: list[tuple[Path, Path]] = []
        try:
            for item in files:
                target = Path(item["path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
                temporary.write_text(item["content"], encoding="utf-8")
                temporaries.append((temporary, target))
            for temporary, target in temporaries:
                os.replace(temporary, target)
        except Exception as exc:
            for target, content in originals.items():
                if content is None:
                    target.unlink(missing_ok=True)
                else:
                    rollback = target.with_name(f".{target.name}.{uuid4().hex}.rollback")
                    rollback.write_bytes(content)
                    os.replace(rollback, target)
            execution.update({
                "status": "FAILED", "finished_at": utc_now(),
                "error": f"multi-file write rolled back: {exc}",
            })
            return self.store.set_execution(
                plan["action_id"], execution, event="execution_failed",
            )
        finally:
            for temporary, _ in temporaries:
                temporary.unlink(missing_ok=True)
        execution.update({
            "status": "VERIFIED", "finished_at": utc_now(),
            "result": {"files": [
                {"path": item["path"], "sha256": _file_sha(Path(item["path"]))}
                for item in files
            ]},
        })
        return self.store.set_execution(
            plan["action_id"], execution, event="execution_verified",
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
    ) -> dict[str, Any]:
        submitted = self._single_status_record(submit_result.get("payload"))
        expected_job_id = str((submitted or {}).get("backend_job_id") or "") or None
        verified, observed, verification_result, detail = self._verify_submission(
            plan, expected_job_id=expected_job_id,
        )
        result = {
            "submission": _redact(submitted),
            "submission_command": _redact(submit_result),
            "observation": _redact(observed),
            "verification_command": _redact(verification_result),
        }
        if verified:
            execution.update({
                "status": "VERIFIED", "finished_at": utc_now(),
                "error": None, "result": result,
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
        })
        return self.store.set_execution(
            plan["action_id"], execution, event="execution_reconcile_required",
        )

    def reconcile(self, action_id: str) -> dict[str, Any]:
        """Observe an uncertain submission without ever issuing submit again."""
        snapshot = self.store.snapshot(action_id)
        execution = snapshot["execution"]
        if execution.get("status") == "VERIFIED":
            return snapshot
        if snapshot.get("operation") == "OBSERVABILITY_BACKFILL":
            if not self.config.allow_observability_mutations:
                raise ActionError("observability mutations are disabled by daemon policy")
            if execution.get("status") not in {"EXECUTING", "RECONCILE_REQUIRED"}:
                raise ActionError("observability action is not awaiting reconciliation")
            return self._execute_internal(snapshot, execution)
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
            })
            return self.store.set_execution(
                action_id, execution, event="submission_reconciled",
            )
        execution.update({
            "status": "RECONCILE_REQUIRED", "last_reconciled_at": utc_now(),
            "error": detail, "result": result,
        })
        return self.store.set_execution(
            action_id, execution, event="submission_reconcile_pending",
        )

    def _execute_controller(self, plan: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
        command = [str(item) for item in plan["command_preview"]]
        result = self.controller.execute_command(
            command, cwd=Path(plan["cwd"]), timeout=self.config.timeout_seconds,
        )
        is_submission = plan["operation"] in {
            "SUBMIT_RUN", "RETRY_ATTEMPT", "RUN_EVALUATION",
        }
        if result.get("timeout"):
            if is_submission:
                return self._submission_result(
                    plan, execution, result,
                    submit_error="controller timed out after execution intent",
                )
            execution.update({
                "status": "RECONCILE_REQUIRED", "finished_at": utc_now(),
                "error": "controller timed out after execution intent; inspect status before retry",
                "result": _redact(result),
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
                )
            execution.update({
                "status": "FAILED", "finished_at": utc_now(),
                "error": str(result.get("stderr") or "controller failed")[:1000],
                "result": _redact(result),
            })
            return self.store.set_execution(plan["action_id"], execution, event="execution_failed")
        if is_submission:
            return self._submission_result(plan, execution, result)
        payload = result.get("payload")
        execution.update({
            "status": "VERIFIED", "finished_at": utc_now(), "result": _redact(payload),
        })
        return self.store.set_execution(plan["action_id"], execution, event="execution_verified")
