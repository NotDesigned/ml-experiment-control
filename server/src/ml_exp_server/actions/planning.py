"""Planning for ActionService."""
from __future__ import annotations
import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import yaml
from pydantic import ValidationError
from ..projects.campaign_lifecycle import campaign_record_path
from ..controller_gateway import redact as _redact
from ..runs.intent_protocol import OperationIntent
from ..runs.operations import intent_scope_error
from ..schemas import OperationScope, OperationScopeType, ResearchProject
from ..storage import utc_now
from .errors import ActionError
from .files import file_sha as _file_sha
from .helpers import _sha256, _manifest_identity_digest, _canonical_manifest_path, _inside, _parse_mapping, _unified_diff, _semantic_changes, _missing_frozen_match_fields, _gate, _SAFE_ID, _SHA256_DIGEST, _CAMPAIGN_REVISION

class ActionPlanning:
    """Planning operations; state belongs to the application facade."""

    def _verified_submission_campaign(
        self,
        *,
        project: str,
        run_id: str,
        attempt_id: str,
        backend_job_id: str,
    ) -> Path | None:
        """Return the immutable Campaign that submitted an exact Attempt.

        Cancellation must remain possible after the authored catalog replaces
        or removes a Run.  Bind it to the already-VERIFIED submission Action,
        exact Attempt and scheduler job instead of reopening mutable authored
        science.
        """
        candidates: list[tuple[str, Path]] = []
        for action in self.store.list_all():
            if (
                action.get("operation")
                not in {"SUBMIT_RUN", "RETRY_ATTEMPT", "RUN_EVALUATION"}
                or action.get("run_id") != run_id
                or action.get("attempt_id") != attempt_id
            ):
                continue
            scope = action.get("scope")
            execution = action.get("execution")
            if (
                not isinstance(scope, dict)
                or scope.get("project") != project
                or not isinstance(execution, dict)
                or execution.get("status") != "VERIFIED"
            ):
                continue
            result = execution.get("result")
            submission = (
                result.get("submission")
                if isinstance(result, dict)
                else None
            )
            observation = (
                result.get("observation")
                if isinstance(result, dict)
                else None
            )
            bound_jobs = {
                str(value.get("backend_job_id") or "")
                for value in (submission, observation)
                if isinstance(value, dict) and value.get("backend_job_id")
            }
            if str(backend_job_id) not in bound_jobs:
                continue
            action_id = str(action.get("action_id") or "")
            try:
                action_root = self.store.directory(action_id).resolve()
                candidate = Path(
                    str(action.get("execution_campaign_file") or ""),
                )
                if candidate.is_symlink():
                    continue
                campaign = candidate.resolve(strict=True)
                campaign.relative_to(action_root)
            except (FileNotFoundError, OSError, ValueError):
                continue
            if (
                not campaign.is_file()
                or action.get("execution_campaign_sha256")
                != _file_sha(campaign)
            ):
                continue
            candidates.append(
                (str(action.get("created_at") or ""), campaign),
            )
        return (
            max(candidates, key=lambda item: item[0])[1]
            if candidates
            else None
        )

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
        if kind in {"CREATE_CAMPAIGN_DRAFT", "UPDATE_CAMPAIGN_DRAFT", "DERIVE_RUN_DRAFT"}:
            plan = self._prepare_campaign(action_id, scope, project, payload)
        elif kind == "ARCHIVE_CAMPAIGN":
            plan = self._prepare_campaign_record(action_id, scope, project, payload)
        elif kind in {"SUBMIT_RUN", "RETRY_ATTEMPT", "CANCEL_RUN", "RUN_EVALUATION"}:
            plan = self._prepare_controller(action_id, scope, project, payload)
        elif kind in {"ARCHIVE_RUN", "ARCHIVE_ATTEMPT"}:
            plan = self._prepare_object_archive(action_id, scope, project, payload)
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
        return "intent-" + ActionPlanning._request_digest(scope, payload).split(":", 1)[1][:20]

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
        experiments_root = (project.base_dir or Path(".")) / "experiments"
        root = experiments_root / "campaigns"
        reference = next((item for item in project.campaigns if item.name == campaign_id), None)
        is_update = (
            intent_kind == "UPDATE_CAMPAIGN_DRAFT"
            or (
                intent_kind == "DERIVE_RUN_DRAFT"
                and reference is not None
                and reference.current_revision is not None
            )
        )
        if is_update and reference is not None and reference.current_revision is not None:
            target = Path(reference.current_revision.file).resolve()
        else:
            target = root.resolve() / f"{campaign_id}.yml"
        if is_update:
            exact_campaign_scope = (
                scope.scope_type == OperationScopeType.CAMPAIGN
                and scope.object_id == campaign_id
            )
            exact_derived_run_scope = (
                intent_kind == "DERIVE_RUN_DRAFT"
                and scope.scope_type == OperationScopeType.RUN
                and scope.object_id in run_ids
            )
            if not exact_campaign_scope and not exact_derived_run_scope:
                raise ActionError(
                    "Campaign update must use the exact existing Campaign scope "
                    "or its source Run scope"
                )
        proposed = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
        current_payload = _parse_mapping(target.read_text(encoding="utf-8")) if target.is_file() else {}
        canonical_memberships = all(
            isinstance(item, dict) and "role" not in item and "membership" not in item
            for item in entries
        )
        gates = [
            _gate("schema", payload.get("schema_version") == 1,
                  "campaign schema_version must equal 1"),
            _gate(
                "safe_target",
                _inside(target, experiments_root if is_update else root),
                str(target),
            ),
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
        if operation == "CANCEL_RUN":
            frozen_campaign = self._verified_submission_campaign(
                project=project.project,
                run_id=run_id,
                attempt_id=attempt_id,
                backend_job_id=str(spec.get("backend_job_id") or ""),
            )
            if frozen_campaign is not None:
                execution_campaign = frozen_campaign
                execution_payload = _parse_mapping(
                    frozen_campaign.read_text(encoding="utf-8"),
                )
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
                if (re.fullmatch(r"[0-9a-f]{40}", inherited_git_commit) is None
                        and not project.controller.capabilities.get("container_execution")):
                    raise ActionError(
                        "canonical Run manifest has no valid immutable git_commit"
                    )
                # The controller checkout may advance for operational fixes
                # between Attempts.  Retry metadata must still identify the
                # exact source commit frozen by the canonical scientific Run.
                if inherited_git_commit:
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
            missing_match_fields = _missing_frozen_match_fields(manifest)
            gates.append(_gate(
                "comparison_identity_fields",
                not missing_match_fields,
                "all frozen comparison match_fields exist in the Run manifest"
                if not missing_match_fields
                else "missing frozen comparison match_fields: "
                + ", ".join(missing_match_fields),
            ))
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
                "git_commit": manifest.get("git_commit"),
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
                project, execution_campaign, "status", run_id,
                attempt_id=attempt_id,
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
