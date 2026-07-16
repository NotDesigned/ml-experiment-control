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
from ..storage import utc_now
from .errors import ActionError
from .files import file_sha as _file_sha
from .policy import ActionExecutionPolicy
from .project_writes import (
    ProjectWriteConflict,
    ProjectWriteError,
    ProjectWriteTransaction,
)
from .store import ActionStore


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_CAMPAIGN_REVISION = re.compile(r"^campaign\.[0-9a-f]{64}$")


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


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
                 internal_executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
                 source_resolver: Callable[[str, str], Path] | None = None):
        self.store = store
        self.config = config
        self.controller = ProjectControllerGateway(runner)
        self.execution_policy = ActionExecutionPolicy(config)
        self.project_write_transaction = ProjectWriteTransaction(store)
        self.actor_provider = actor_provider or self._local_actor
        self.internal_executor = internal_executor
        self.source_resolver = source_resolver

    @staticmethod
    def _local_actor() -> str:
        return f"local-process-owner:uid={os.getuid()}:{getpass.getuser()}@{socket.gethostname()}"

    def prepare(self, scope: OperationScope, project: ResearchProject,
                intent: OperationIntent | dict[str, Any]) -> dict[str, Any]:
        # Preparation writes preview artifacts before the immutable plan.  Keep
        # that whole sequence under the same cross-process lock as save_plan so
        # competing payloads cannot overwrite the winner's reviewed artifacts.
        with self.store.locked():
            return self._prepare_locked(scope, project, intent)

    def _prepare_locked(self, scope: OperationScope, project: ResearchProject,
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
        elif kind == "REBUILD_LOCAL_EVIDENCE":
            plan = self._prepare_local_evidence_rebuild(
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
        try:
            return self.store.save_plan(plan)
        except RuntimeError as exc:
            raise ActionError(str(exc)) from exc

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

    def _prepare_local_evidence_rebuild(
        self, action_id: str, scope: OperationScope,
        project: ResearchProject, intent: dict[str, Any],
    ) -> dict[str, Any]:
        """Freeze one pure-local exact-Attempt evidence rebuild command."""
        if scope.scope_type != OperationScopeType.ATTEMPT:
            raise ActionError("local evidence rebuild requires exact Attempt scope")
        if project.controller is None:
            raise ActionError("project has no controller configuration")
        spec = _parse_mapping(str(intent.get("draft", "")))
        run_id, scoped_attempt_id = scope.object_id.rsplit("::", 1)
        attempt_id = str(spec.get("attempt_id") or "")
        if (
            spec.get("project") != project.project
            or spec.get("run_id") != run_id
            or attempt_id != scoped_attempt_id
        ):
            raise ActionError("local evidence rebuild draft conflicts with exact scope")
        if not project.controller.capabilities.get("refresh_evidence_local"):
            raise ActionError(
                "project controller does not declare refresh_evidence_local"
            )
        reason = str(spec.get("reason") or "").strip()
        if not reason:
            raise ActionError("local evidence rebuild reason is required")
        base = (project.base_dir or Path(".")).resolve()
        campaign = Path(str(spec.get("campaign_file") or ""))
        if not campaign.is_absolute():
            campaign = base / campaign
        campaign = campaign.resolve()
        if not campaign.is_file() or not _inside(campaign, base / "experiments"):
            raise ActionError(
                "campaign_file must exist under the project's experiments directory"
            )

        run_dir = Path(str(spec.get("run_dir") or ""))
        local_root = Path(str(spec.get("local_root") or ""))
        if not run_dir.is_absolute() or not local_root.is_absolute():
            raise ActionError("local evidence paths must be absolute")
        run_dir = run_dir.resolve()
        local_root = local_root.resolve()
        campaign_payload = _parse_mapping(campaign.read_text(encoding="utf-8"))
        campaign_id = str(campaign_payload.get("campaign") or "")
        expected_run_dir = (local_root / campaign_id / run_id).resolve()
        if (
            campaign_payload.get("project") != project.project
            or not campaign_id
            or run_dir != expected_run_dir
        ):
            raise ActionError("local evidence run path conflicts with campaign identity")
        attempt_dir = run_dir / "attempts" / attempt_id
        reviewed_inputs = {
            "inputs/campaign.yml": campaign,
            "inputs/run/manifest.yaml": run_dir / "manifest.yaml",
            "inputs/attempt/attempt.yaml": attempt_dir / "attempt.yaml",
            "inputs/attempt/backend.json": attempt_dir / "backend.json",
        }
        status_preimage = attempt_dir / "status.json"
        if status_preimage.is_file():
            reviewed_inputs["inputs/attempt/status.json"] = status_preimage
        collection_preimage = attempt_dir / "collection.json"
        if collection_preimage.is_file():
            reviewed_inputs["inputs/attempt/collection.json"] = collection_preimage
        missing = [
            str(path) for path in reviewed_inputs.values() if not path.is_file()
        ]
        collected_run = attempt_dir / "collected_run"
        if missing or not collected_run.is_dir() or not any(collected_run.iterdir()):
            detail = ", ".join(missing) if missing else str(collected_run)
            raise ActionError(f"required already-local evidence is missing: {detail}")
        try:
            controller_snapshot = self.controller.snapshot_execution_bundle(
                project, self.store.directory(action_id) / "controller-input",
                reviewed_inputs=reviewed_inputs,
            )
        except (OSError, ValueError) as exc:
            raise ActionError(f"controller snapshot preparation failed: {exc}") from exc
        snapshot_campaign = "{controller_snapshot}/inputs/campaign.yml"
        snapshot_identity_root = "{controller_snapshot}/inputs"
        preview_arguments = [
            snapshot_campaign, "refresh-evidence-local", "--run", run_id,
            "--attempt-id", attempt_id, "--local-root", str(local_root),
            "--identity-root", snapshot_identity_root, "--dry-run",
        ]
        try:
            preview_result = self.controller.execute_snapshot(
                controller_snapshot, preview_arguments,
                timeout=self.config.timeout_seconds,
            )
        except (OSError, ValueError) as exc:
            preview_result = {
                "returncode": 1, "timeout": False, "payload": None,
                "stdout": "", "stderr": str(exc),
            }
        preview_passed = (
            preview_result.get("returncode") == 0
            and not preview_result.get("timeout")
        )
        preview_gate = _gate(
            "local_evidence_preview", preview_passed,
            "private controller preview passed" if preview_passed else str(
                preview_result.get("stderr") or preview_result.get("stdout")
                or "private controller preview failed"
            )[:500],
        )
        payload = preview_result.get("payload")
        record = (
            payload[0] if isinstance(payload, list) and len(payload) == 1
            and isinstance(payload[0], dict) else {}
        )
        input_digest = str(record.get("input_digest") or "")
        old_digest = record.get("old_digest")
        expected_new_digest = str(
            record.get("expected_new_collection_digest") or ""
        )
        raw_collection = record.get("collection_path")
        collection = None
        if isinstance(raw_collection, str) and raw_collection.strip():
            candidate = Path(raw_collection)
            if candidate.is_absolute():
                collection = candidate.resolve()
        identity_exact = (
            record.get("project") == project.project
            and record.get("run_id") == run_id
            and record.get("attempt_id") == attempt_id
        )
        target_exact = (
            collection is not None and collection == collection_preimage.resolve()
        )
        digest_exact = (
            _SHA256_DIGEST.fullmatch(input_digest) is not None
            and (old_digest is None or (
                isinstance(old_digest, str)
                and _SHA256_DIGEST.fullmatch(old_digest) is not None
            ))
            and _SHA256_DIGEST.fullmatch(expected_new_digest) is not None
            and record.get("new_digest") == expected_new_digest
        )
        local_only = (
            record.get("local_only") is True
            and record.get("backend_accessed") is False
            and record.get("scheduler_accessed") is False
            and record.get("controller_snapshot_sha256")
            == controller_snapshot["manifest_sha256"]
            and record.get("atomic_collection_replace") is True
            and record.get("write_protocol") == "dirfd-fsync-rename-v1"
        )
        gates = [
            _gate("exact_project", spec.get("project") == project.project,
                  project.project),
            _gate("exact_attempt_scope", True,
                  f"{project.project}:{run_id}::{attempt_id}"),
            _gate("preview_exact_identity", identity_exact,
                  f"{project.project}:{run_id}::{attempt_id}"),
            _gate("controller_capability", True, "refresh_evidence_local"),
            _gate("local_evidence_preview", preview_gate["status"] != "FAIL",
                  preview_gate["detail"]),
            _gate(
                "local_collection_target", target_exact,
                str(collection) if collection is not None else "unavailable",
            ),
            _gate("local_input_digest", digest_exact, input_digest or "missing"),
            _gate("no_backend_or_scheduler", local_only,
                  "controller preview declares local-only execution"),
            _gate("evidence_reference", bool(intent.get("evidence_digest")),
                  "intent is bound to exact Attempt evidence"),
        ]
        execution_arguments = [
            snapshot_campaign, "refresh-evidence-local", "--run", run_id,
            "--attempt-id", attempt_id, "--local-root", str(local_root),
            "--identity-root", snapshot_identity_root,
            "--expected-input-digest", input_digest,
        ]
        plan = self._base_plan(
            action_id, scope, intent, "REBUILD_LOCAL_EVIDENCE",
        )
        plan.update({
            "campaign_file": str(campaign),
            "run_id": run_id,
            "attempt_id": attempt_id,
            "reason": reason,
            "input_digest": input_digest,
            "collection_path": str(collection) if collection is not None else None,
            "expected_collection_sha256": old_digest,
            "expected_new_collection_sha256": expected_new_digest,
            "atomic_collection_replace": True,
            "write_protocol": "dirfd-fsync-rename-v1",
            "controller_snapshot": controller_snapshot,
            "snapshot_arguments": execution_arguments,
            "gates": gates,
            "ready": all(item["status"] != "FAIL" for item in gates),
            "preflight_summary": {
                "project": project.project,
                "run_id": run_id,
                "attempt_id": attempt_id,
                "input_digest": input_digest,
                "old_digest": old_digest,
                "expected_new_digest": expected_new_digest,
                "local_only": local_only,
            },
            "command_preview": _redact([
                "private-controller", controller_snapshot["manifest_sha256"],
                *execution_arguments,
            ]),
            "diff": json.dumps(_redact(record), ensure_ascii=False, indent=2),
        })
        return plan

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
        try:
            authored_campaign_bytes = campaign.read_bytes()
            authored_campaign_text = authored_campaign_bytes.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ActionError(f"campaign_file cannot be read as UTF-8: {exc}") from exc
        authored_campaign_payload = _parse_mapping(authored_campaign_text)
        authored_campaign_sha256 = _sha256(authored_campaign_bytes)
        authored_revision_from_bytes = (
            "campaign." + hashlib.sha256(authored_campaign_bytes).hexdigest()
        )
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

        expected_source_id = str(spec.get("expected_source_id") or "")
        imported_source_args: list[str] = []
        if expected_source_id.startswith("source."):
            capability = bool(project.controller.capabilities.get("daemon_source_revision"))
            gates.append(_gate(
                "daemon_source_capability", capability,
                "controller must declare daemon_source_revision for imported source trees",
            ))
            source_root: Path | None = None
            try:
                if self.source_resolver is not None:
                    source_root = self.source_resolver(project.project, expected_source_id)
            except (OSError, ValueError, json.JSONDecodeError):
                source_root = None
            gates.append(_gate(
                "daemon_source_available", source_root is not None,
                "daemon-owned immutable source tree must exist and match its metadata",
            ))
            if source_root is not None:
                imported_source_args = [
                    "--source-root", str(source_root),
                    "--source-id", expected_source_id,
                ]

        actual_verb = "cancel" if operation == "CANCEL_RUN" else "submit"
        execution_campaign = campaign
        execution_payload = deepcopy(authored_campaign_payload)
        authored_revision_capability = bool(
            project.controller.capabilities.get("authored_campaign_revision")
        )
        if actual_verb == "submit" and (
            project.daemon_run_root is not None or authored_revision_capability
        ):
            # The authored Campaign describes science and backend resources;
            # the daemon owns where canonical Run/Attempt control metadata is
            # materialized. Freeze that storage binding into the Action so a
            # later source checkout update cannot redirect an approved submit.
            execution_campaign = (
                self.store.directory(action_id) / "campaign.execution.yml"
            )
            execution_campaign.parent.mkdir(parents=True, exist_ok=True)
            if project.daemon_run_root is not None:
                requested_root = spec.get("local_root")
                execution_root = (
                    Path(str(requested_root)).resolve()
                    if requested_root else project.daemon_run_root.resolve()
                )
                allowed_roots = {
                    root.resolve() for root in project.resolved_run_roots()
                }
                if execution_root not in allowed_roots:
                    raise ActionError(
                        "execution local_root is outside registered project run roots"
                    )
                execution_payload["local_root"] = str(execution_root)
                execution_campaign.write_text(
                    yaml.safe_dump(
                        execution_payload, allow_unicode=True, sort_keys=False,
                    ),
                    encoding="utf-8",
                )
            else:
                # Preserve the exact reviewed bytes. Preview and live execution
                # must never reopen the mutable authored path after catalog
                # revision validation.
                execution_campaign.write_bytes(authored_campaign_bytes)

        # A daemon-owned submit may rewrite only the operational local_root in
        # a private Campaign copy. Controllers that opt into this contract
        # freeze either the exact current authored revision (new Run) or the
        # existing canonical Run revision (retry / Attempt evaluation).
        # Build once without the optional identity solely to resolve the
        # controller cwd used by relative local_root values.
        probe_call = self.controller.build(
            project, execution_campaign, actual_verb, run_id,
            attempt_id=attempt_id, extra=imported_source_args,
        )
        campaign_revision: str | None = None
        campaign_revision_source: str | None = None
        inherits_run_revision = operation == "RETRY_ATTEMPT" or (
            operation == "RUN_EVALUATION"
            and scope.scope_type == OperationScopeType.ATTEMPT
        )
        if actual_verb == "submit" and authored_revision_capability:
            if inherits_run_revision:
                canonical_path = _canonical_manifest_path(
                    execution_payload, cwd=probe_call.cwd, run_id=run_id,
                )
                if canonical_path is None or not canonical_path.is_file():
                    raise ActionError(
                        "retry/evaluation requires the exact canonical Run manifest "
                        "to inherit its immutable Campaign revision"
                    )
                try:
                    canonical_manifest = yaml.safe_load(
                        canonical_path.read_text(encoding="utf-8")
                    )
                except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
                    raise ActionError(
                        f"canonical Run manifest cannot be read: {exc}"
                    ) from exc
                expected_campaign = str(execution_payload.get("campaign") or "")
                exact_scope = (
                    isinstance(canonical_manifest, dict)
                    and canonical_manifest.get("project") == project.project
                    and canonical_manifest.get("campaign") == expected_campaign
                    and canonical_manifest.get("run_id") == run_id
                )
                inherited = (
                    str(canonical_manifest.get("campaign_id") or "")
                    if isinstance(canonical_manifest, dict) else ""
                )
                if not exact_scope or _CAMPAIGN_REVISION.fullmatch(inherited) is None:
                    raise ActionError(
                        "canonical Run manifest does not match the exact project, "
                        "Campaign, and Run scope with an immutable campaign.<sha256> revision"
                    )
                inherited_git_commit = str(canonical_manifest.get("git_commit") or "")
                if re.fullmatch(r"[0-9a-f]{40}", inherited_git_commit) is None:
                    raise ActionError(
                        "canonical Run manifest has no valid immutable git_commit"
                    )
                # The controller checkout may advance for operational fixes
                # between Attempts.  Retry metadata must still identify the
                # exact source commit frozen by the canonical scientific Run.
                execution_payload["git_commit"] = inherited_git_commit
                execution_campaign.write_text(
                    yaml.safe_dump(
                        execution_payload, allow_unicode=True, sort_keys=False,
                    ),
                    encoding="utf-8",
                )
                campaign_revision = inherited
                campaign_revision_source = "canonical_run"
            else:
                reference = next((
                    item for item in project.campaigns
                    if item.current_revision is not None
                    and Path(item.current_revision.file).resolve() == campaign
                ), None)
                if reference is None or reference.current_revision is None:
                    raise ActionError(
                        "authored campaign revision capability requires a current "
                        "catalog Campaign matching campaign_file"
                    )
                catalog_revision = reference.current_revision.revision_id
                if catalog_revision != authored_revision_from_bytes:
                    raise ActionError(
                        "campaign_file changed after the Project catalog was loaded; "
                        "refresh or re-register the Project before preparing submission"
                    )
                campaign_revision = catalog_revision
                campaign_revision_source = "authored_catalog"

        campaign_identity_args = (
            ["--campaign-id", campaign_revision]
            if campaign_revision is not None else []
        )
        controller_extra = [*imported_source_args, *campaign_identity_args]
        call = self.controller.build(
            project, execution_campaign, actual_verb, run_id, attempt_id=attempt_id,
            extra=controller_extra,
        )
        command, cwd = call.argv, call.cwd
        stage_command: list[str] | None = None
        stage_cwd: Path | None = None
        if actual_verb == "submit":
            stage_call = self.controller.build(
                project, execution_campaign, "stage", run_id,
                attempt_id=attempt_id, extra=controller_extra,
            )
            stage_command, stage_cwd = stage_call.argv, stage_call.cwd
        preview_payload: dict[str, Any] | None = None
        observed: dict[str, Any] = {}
        execution_manifest_path: Path | None = None
        execution_manifest_sha256: str | None = None
        if actual_verb == "submit":
            raw = _parse_mapping(execution_campaign.read_text(encoding="utf-8"))
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
                attempt_id=attempt_id, dry_run=True, extra=controller_extra,
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
            if campaign_revision is not None:
                gates.append(_gate(
                    "campaign_revision_binding",
                    manifest.get("campaign_id") == campaign_revision,
                    "preview Run manifest must preserve the reviewed Campaign revision "
                    f"{campaign_revision} from {campaign_revision_source}",
                ))
            if expected_source_id:
                gates.append(_gate(
                    "authored_source_binding",
                    manifest.get("source_id") == expected_source_id,
                    f"preview source_id must equal authored source_id {expected_source_id}",
                ))
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
            resource_approval = str(
                spec.get("resource_approval") or "budget_cap"
            )
            requested_gpu_hours = self._gpu_hours(resources, manifest.get("backend"))
            budget_ok = (
                requested_gpu_hours is not None
                and (
                    resource_approval == "review_exact"
                    or (
                        resource_approval == "budget_cap"
                        and isinstance(budget_limit, (int, float))
                        and requested_gpu_hours <= float(budget_limit)
                    )
                )
            )
            gates.append(_gate("budget", budget_ok,
                               f"resource_approval={resource_approval}; "
                               f"requested_gpu_hours={requested_gpu_hours}; "
                               f"max_gpu_hours={budget_limit}"))
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
                "campaign_id": manifest.get("campaign_id"),
                "resolved_config": manifest.get("resolved_config"),
                "backend": backend,
                "resources": resources,
                "storage": manifest.get("storage"),
                "control_metadata_root": raw.get("local_root"),
                "command": manifest.get("command"),
                "assets": manifest.get("assets"),
                "checkpoint": checkpoint,
                "requested_gpu_hours": requested_gpu_hours,
                "max_gpu_hours": budget_limit,
                "resource_approval": resource_approval,
                "wandb_cloud_sync": bool(spec.get("wandb_cloud_sync", False)),
            })
            for name, verb, extra in (
                ("preflight", "preflight", ["--scope", "submit"]),
                ("duplicate_run_identity", "check-identity", []),
                ("assets", "assets-verify", []),
            ):
                check_call = self.controller.build(
                    project, execution_campaign, verb, run_id,
                    attempt_id=attempt_id, extra=[*extra, *controller_extra],
                )
                gate, _ = self._run_gate(name, check_call.argv, check_call.cwd)
                gates.append(gate)
            controller_capabilities = project.controller.capabilities
            gates.append(_gate(
                "submit_outbox_capability",
                bool(controller_capabilities.get("submit_outbox"))
                and bool(controller_capabilities.get("run_identity_v2")),
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
            "execution_campaign_file": str(execution_campaign),
            "attempt_id": attempt_id, "backend_job_id": spec.get("backend_job_id"),
            "expected_source_id": expected_source_id,
            "campaign_revision": campaign_revision,
            "campaign_revision_source": campaign_revision_source,
            "gates": gates, "ready": all(item["status"] != "FAIL" for item in gates),
            "diff": json.dumps(
                _redact(preview_payload if preview_payload else observed),
                ensure_ascii=False, indent=2,
            ),
            "semantic_changes": _semantic_changes({}, manifest) if actual_verb == "submit" else [],
            "preflight_summary": preflight_summary,
            "command_preview": _redact(command),
            "cwd": str(cwd),
            "stage_command_preview": (
                _redact(stage_command) if stage_command is not None else None
            ),
            "stage_cwd": str(stage_cwd) if stage_cwd is not None else None,
            "authored_campaign_sha256": authored_campaign_sha256,
            "execution_campaign_sha256": (
                authored_campaign_sha256
                if execution_campaign == campaign else _file_sha(execution_campaign)
            ),
            "execution_manifest_path": (
                str(execution_manifest_path) if execution_manifest_path is not None else None
            ),
            "execution_manifest_sha256": execution_manifest_sha256,
        })
        verification = self.controller.build(
            project, execution_campaign, "status", run_id, attempt_id=attempt_id,
            extra=controller_extra,
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
        try:
            return self.store.set_execution(
                action_id, execution, event="execution_authorized",
                expected_status="PREPARED",
            )
        except RuntimeError as exc:
            raise ActionError(str(exc)) from exc

    def execute(self, action_id: str, confirmation: str) -> dict[str, Any]:
        snapshot = self.store.snapshot(action_id)
        execution = snapshot["execution"]
        if execution.get("status") == "VERIFIED":
            return snapshot
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
        execution.update({"status": "EXECUTING", "started_at": utc_now(), "error": None})
        try:
            started = self.store.begin_execution(
                action_id, execution, intent_digest=str(snapshot["intent_digest"]),
            )
        except RuntimeError as exc:
            raise ActionError(str(exc)) from exc
        execution = started["execution"]
        if dispatch.project_write:
            return self._execute_write(snapshot, execution)
        if dispatch.internal_mutation:
            return self._execute_internal(snapshot, execution)
        if dispatch.local_evidence_rebuild:
            return self._execute_local_evidence_rebuild(snapshot, execution)
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
            })
            return self.store.set_execution(
                plan["action_id"], execution,
                event="local_evidence_rebuild_reconcile_required",
            )
        execution.update({
            "status": "VERIFIED", "finished_at": utc_now(), "error": None,
            "result": _redact(record),
        })
        return self.store.set_execution(
            plan["action_id"], execution,
            event="local_evidence_rebuild_verified",
        )

    def _execute_write(self, plan: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
        try:
            result = self.project_write_transaction.apply(plan)
            if plan["operation"] == "WRITE_RESEARCH_QUESTION":
                load_research_question(Path(plan["target_path"]))
            if not plan.get("files"):
                only_file = result["files"][0]
                result = {
                    "target_path": only_file["path"],
                    "sha256": only_file["sha256"],
                }
        except ProjectWriteError as exc:
            execution.update({
                "status": (
                    "FAILED"
                    if isinstance(exc, ProjectWriteConflict) and not exc.partial
                    else "RECONCILE_REQUIRED"
                ),
                "finished_at": utc_now(),
                "last_reconciled_at": utc_now(),
                "error": str(exc),
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
        if snapshot.get("operation") == "OBSERVABILITY_BACKFILL":
            if not self.config.allow_observability_mutations:
                raise ActionError("observability mutations are disabled by daemon policy")
            if execution.get("status") not in {"EXECUTING", "RECONCILE_REQUIRED"}:
                raise ActionError("observability action is not awaiting reconciliation")
            previous = execution.get("result")
            result = dict(previous) if isinstance(previous, dict) else {}
            result["reconciled_read_only"] = True
            execution.update({
                "status": "RECONCILE_REQUIRED",
                "last_reconciled_at": utc_now(),
                "error": (
                    "observability backfill cannot be replayed during reconciliation; "
                    "inspect target status and prepare a new explicit Action if replay is needed"
                ),
                "result": result,
            })
            return self.store.set_execution(
                action_id, execution, event="observability_reconcile_requires_operator",
            )
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
                })
                return self.store.set_execution(
                    plan["action_id"], execution, event="execution_stage_failed",
                )
            stage_result = self.controller.execute_command(
                stage_command,
                cwd=Path(str(stage_cwd)),
                timeout=self.config.timeout_seconds,
            )
            if stage_result.get("timeout") or stage_result.get("returncode") != 0:
                detail = str(
                    stage_result.get("stderr")
                    or stage_result.get("stdout")
                    or "source staging controller failed"
                )[:1000]
                error = (
                    "source staging timed out before scheduler submission"
                    if stage_result.get("timeout")
                    else f"source staging failed before scheduler submission: {detail}"
                )
                execution.update({
                    "status": "FAILED", "finished_at": utc_now(),
                    "error": error,
                    "result": {"stage_command": _redact(stage_result)},
                })
                return self.store.set_execution(
                    plan["action_id"], execution, event="execution_stage_failed",
                )
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
            })
            return self.store.set_execution(plan["action_id"], execution, event="execution_failed")
        if is_submission:
            return self._submission_result(
                plan, execution, result, stage_result=stage_result,
            )
        payload = result.get("payload")
        execution.update({
            "status": "VERIFIED", "finished_at": utc_now(), "result": _redact(payload),
        })
        return self.store.set_execution(plan["action_id"], execution, event="execution_verified")

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
                    "error": (
                        action["execution"].get("error")
                        or "startup recovery is blocked because project writes are disabled"
                    ),
                },
            } for action in pending]
        recovered: list[dict[str, Any]] = []
        for action in pending:
            recovered.append(self.reconcile(str(action["action_id"])))
        return recovered
