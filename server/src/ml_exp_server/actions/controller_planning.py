"""Controller Action planning: freeze identities, preview, then assemble gates.

The ActionPlanning entry point owns preparation locking and persistence. These
steps only prepare the same reviewed artifacts and controller observations.
"""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from ..controller_output import redact as _redact
from ..schemas import OperationScope, OperationScopeType, ResearchProject
from .errors import ActionError
from .files import file_sha as _file_sha
from .helpers import (
    _CAMPAIGN_REVISION, _SAFE_ID, _canonical_manifest_path, _gate, _inside,
    _manifest_identity_digest, _missing_frozen_match_fields, _parse_mapping,
    _semantic_changes, _sha256,
)


class ControllerPlanning:
    """Plan controller operations using the shared ActionService state."""

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
        imported_source_args = self._controller_source_args(project, expected_source_id, gates)

        actual_verb = "cancel" if operation == "CANCEL_RUN" else "submit"
        execution_campaign, campaign_revision, campaign_revision_source = self._freeze_controller_campaign(
            action_id, scope, project, spec, campaign, authored_campaign_bytes,
            authored_campaign_payload, authored_revision_from_bytes, operation,
            actual_verb, run_id, attempt_id, imported_source_args,
        )

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
            (
                raw, manifest, preview_payload,
                execution_manifest_path, execution_manifest_sha256,
            ) = self._preview_controller_submission(
                action_id, project, execution_campaign, run_id, attempt_id,
                controller_extra, cwd, gates,
            )
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
            preflight_summary = self._controller_resource_gates(
                manifest, raw, spec, campaign, run_id, attempt_id, gates,
            )
            self._controller_submit_checks(
                project, execution_campaign, run_id, attempt_id, controller_extra, gates,
            )
        else:
            observed = self._controller_cancel_gates(
                project, execution_campaign, run_id, attempt_id, spec, gates,
            )

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

    def _controller_source_args(
        self, project: ResearchProject, expected_source_id: str,
        gates: list[dict[str, Any]],
    ) -> list[str]:
        """Bind an imported source before any controller preview."""
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
        return imported_source_args

    def _freeze_controller_campaign(
        self, action_id, scope, project, spec, campaign, authored_campaign_bytes,
        authored_campaign_payload, authored_revision_from_bytes, operation,
        actual_verb, run_id, attempt_id, imported_source_args,
    ):
        """Freeze storage and source revision without reopening authored science."""
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
        campaign_revision, campaign_revision_source = self._controller_campaign_revision(
            project, scope, operation, execution_campaign, execution_payload,
            campaign, authored_revision_from_bytes, actual_verb, run_id, attempt_id,
            imported_source_args, authored_revision_capability,
        )
        return execution_campaign, campaign_revision, campaign_revision_source

    def _controller_campaign_revision(
        self, project, scope, operation, execution_campaign, execution_payload,
        campaign, authored_revision_from_bytes, actual_verb, run_id, attempt_id,
        imported_source_args, authored_revision_capability,
    ):
        """Inherit retry provenance or bind the current authored catalog revision."""
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
        return campaign_revision, campaign_revision_source

    def _preview_controller_submission(
        self, action_id, project, execution_campaign, run_id, attempt_id,
        controller_extra, cwd, gates,
    ):
        """Dry-run in the private preview root and compare canonical identity."""
        preview_payload: dict[str, Any] | None = None
        execution_manifest_sha256: str | None = None
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
        return raw, manifest, preview_payload, execution_manifest_path, execution_manifest_sha256

    def _controller_resource_gates(
        self, manifest, raw, spec, campaign, run_id, attempt_id, gates,
    ) -> dict[str, Any]:
        """Keep exact resource approval and checkpoint exposure in the review."""
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
        return _redact({
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

    def _controller_submit_checks(
        self, project, execution_campaign, run_id, attempt_id, controller_extra, gates,
    ) -> None:
        """Observe controller preflight, duplicate identity and asset gates in order."""
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

    def _controller_cancel_gates(
        self, project, execution_campaign, run_id, attempt_id, spec, gates,
    ):
        """Cancellation is bound to the original scheduler job and outbox support."""
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
        return observed

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
