"""Evidence for ExperimentServerApplication."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from ..projects.campaign_lifecycle import campaign_snapshot
from ..ingest.runscan import preferred_attempt_id
from ..outward import attempt_dto, sanitized_outward
from ..schemas import OperationScope, OperationScopeType, ResearchProject
from .failures import compact_evidence, attempt_failure_evidence_assessment
from .context import RunContext

class RunEvidence:
    """Evidence operations; state belongs to the application facade."""

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
            "decision": RunContext._operational_decision(row.decision),
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
