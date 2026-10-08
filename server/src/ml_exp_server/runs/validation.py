"""Validation for ExperimentServerApplication."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from ..ingest.runscan import preferred_attempt_id, train_metric_records
from .evidence_conflicts import classify_evidence_conflicts
from ..schemas import OperationScopeType, CampaignRelationship

class RunValidation:
    """Validation operations; state belongs to the application facade."""

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
            return RunValidation._gate(
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
            return RunValidation._gate(
                gate_id, "BLOCKED", f"{label} identity conflicts with the requested object",
                evidence,
            )
        if missing:
            evidence["missing_fields"] = missing
            return RunValidation._gate(
                gate_id, "UNKNOWN", f"{label} does not record every identity field", evidence,
            )
        return RunValidation._gate(
            gate_id, "PASS", f"{label} identity matches", evidence,
        )

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
                if run_value is None and attempt_value is None:
                    run_value = (
                        run_resolved.get("seeds")
                        if isinstance(run_resolved, dict)
                        else None
                    )
                    attempt_value = (
                        attempt_resolved.get("seeds")
                        if isinstance(attempt_resolved, dict)
                        else None
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
        elif not records and (
            str(collection.get("model_state") or "").upper() == "OBSERVED"
            and isinstance(collection.get("artifacts"), dict)
            and sum(
                int(item.get("records") or 0)
                for item in collection["artifacts"].values()
                if isinstance(item, dict)
            ) > 0
        ):
            gates.append(self._gate(
                "attempt.model_evidence", "PASS",
                "exact-Attempt model result artifacts are observed",
                {"source_attempt_id": source_attempt_id},
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
        research_contract = run_manifest.get("research_contract")
        required_artifacts = (
            research_contract.get("required_artifacts") or {}
            if isinstance(research_contract, dict)
            else {}
        )
        if checkpoint and checkpoint_step is not None:
            gates.append(self._gate(
                "attempt.checkpoint_evidence", "PASS",
                "completed checkpoint is recorded for the exact Attempt",
                {"path": checkpoint, "step": checkpoint_step,
                 "local_entries": local_checkpoints[:20]},
            ))
        elif (
            terminal_success
            and bool(run_manifest)
            and not run_manifest.get("checkpoint")
            and "checkpoint" not in json.dumps(
                required_artifacts,
                sort_keys=True,
            ).lower()
            and not (
                isinstance(run_manifest.get("storage"), dict)
                and run_manifest["storage"].get("checkpoint_dir")
            )
        ):
            gates.append(self._gate(
                "attempt.checkpoint_evidence", "PASS",
                "Run identity does not declare checkpoint evidence",
                {"applicability": "NOT_REQUIRED", "local_entries": []},
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
        if seed is None:
            seed = manifest.get("seeds")
        if seed is None and isinstance(resolved, dict):
            seed = resolved.get("seeds")
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
