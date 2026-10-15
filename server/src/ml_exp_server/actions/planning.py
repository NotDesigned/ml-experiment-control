"""Planning for ActionService."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import yaml
from pydantic import ValidationError
from ..projects.campaign_lifecycle import campaign_record_path
from ..controller_output import redact as _redact
from ..runs.intent_protocol import OperationIntent
from ..runs.operations import intent_scope_error
from ..schemas import OperationScope, OperationScopeType, ResearchProject
from ..storage import utc_now
from .errors import ActionError
from .files import file_sha as _file_sha
from .helpers import _sha256, _inside, _parse_mapping, _unified_diff, _semantic_changes, _gate, _SAFE_ID, _SHA256_DIGEST

class ActionPlanning:
    """Planning operations; state belongs to the application facade."""

    def prepare(self, scope: OperationScope, project: ResearchProject,
                intent: OperationIntent | dict[str, Any]) -> dict[str, Any]:
        # Preparation writes preview artifacts before the immutable plan.  Keep
        # that whole sequence under the same cross-process lock as save_plan so
        # competing payloads cannot overwrite the winner's reviewed artifacts.
        from ..runs.deletion_guards import action_guard
        kind = intent.get("kind") if isinstance(intent, dict) else intent.kind
        guard = action_guard(self, {"scope": scope.model_dump(mode="json"), "operation": kind})
        with guard, self.store.locked():
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
        elif kind == "DELETE_RUN_RESOURCES" and self.deletion_service is not None:
            plan = self.deletion_service.prepare_action(action_id, scope, project, payload)
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
        if unique:
            from ..runs.deletion_guards import require_action_runs
            require_action_runs(self, project.project, run_ids)
            for run in runs:
                reference = run.get("resume_from")
                if isinstance(reference, dict) and "run_id" in reference:
                    require_action_runs(self, project.project, [reference["run_id"]])
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
