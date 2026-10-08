"""Queries for ExperimentServerApplication."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from ..application_errors import ApplicationError
from ..projects.campaign_lifecycle import campaign_snapshot
from ..projects.code_identity import project_code_identity
from ..ingest.runscan import evaluation_snapshot, metric_contract, parse_iso_ts, preferred_attempt_id, read_jsonl, train_metric_records
from ..outward import attempt_dto, run_dto, sanitized_outward
from ..schemas import OperationScopeType
from .failures import compact_evidence, evidence_digest

class RunQueries:
    """Queries operations; state belongs to the application facade."""

    def object_show(self, project: str, scope_type: OperationScopeType | str,
                    object_id: str) -> dict[str, Any]:
        """Return one of the five canonical objects plus its bounded evidence."""
        scope, configured, resolved = self.resolve_scope(project, scope_type, object_id)
        if scope.scope_type == OperationScopeType.RUN and hasattr(resolved, "run_id"):
            object_payload = self.row_evidence(resolved)
        elif scope.scope_type == OperationScopeType.ATTEMPT:
            object_payload = attempt_dto(resolved)
        elif hasattr(resolved, "model_dump"):
            object_payload = resolved.model_dump(mode="json")
        else:
            object_payload = compact_evidence(resolved)
        bounded = self.bounded_evidence(scope, configured, resolved)
        return {
            "scope": scope.model_dump(mode="json"),
            "object": object_payload,
            "evidence": bounded,
            "evidence_digest": evidence_digest(bounded),
            "code_identity": project_code_identity(configured),
        }

    def campaign_list(self, project: str) -> dict[str, Any]:
        configured = self.runtime.project(project)
        return {
            "project": project,
            "campaigns": [
                campaign_snapshot(self.runtime.index, configured, item.name)
                for item in configured.campaigns
            ],
        }

    def campaign_status(self, project: str, campaign: str) -> dict[str, Any]:
        configured = self.runtime.project(project)
        try:
            return campaign_snapshot(self.runtime.index, configured, campaign)
        except KeyError as exc:
            raise ApplicationError(str(exc).strip("'"), status_code=404,
                                   code="UNKNOWN_CAMPAIGN") from exc

    def run_attempts(self, project: str, run_id: str) -> dict[str, Any]:
        """List exact Attempt identities without expanding collected evidence."""
        _, _, row = self.resolve_scope(project, OperationScopeType.RUN, run_id)
        current = preferred_attempt_id(Path(row.run_dir))
        attempts = [{
            "identity": f"{row.run_id}::{item.attempt_id}",
            "attempt_id": item.attempt_id,
            "current": item.attempt_id == current,
            "backend": item.backend,
            "backend_job_id": item.backend_job_id,
            "state": item.state,
            "decision": item.decision.get("action") if item.decision else None,
            "has_submission": item.has_submission,
        } for item in row.attempts]
        return {
            "project": project, "run_id": row.run_id,
            "current_attempt_id": current, "count": len(attempts),
            "attempts": attempts,
        }

    def attempt_show(self, project: str, identity: str) -> dict[str, Any]:
        scope, _, row, attempt, attempt_dir = self._attempt_context(project, identity)
        collection, assessment = self._attempt_failure_assessment(
            row, attempt, attempt_dir,
        )
        return {
            "scope": scope.model_dump(mode="json"),
            "attempt": attempt_dto(attempt),
            "run_id": row.run_id,
            "run_dir": row.run_dir,
            "attempt_dir": str(attempt_dir),
            "failure_assessment": assessment,
            "collection": sanitized_outward(compact_evidence(collection)),
        }

    def attempt_logs(self, project: str, identity: str, *, stream: str = "both",
                     lines: int = 80) -> dict[str, Any]:
        if lines <= 0 or lines > 10000:
            raise ApplicationError("lines must be between 1 and 10000",
                                   status_code=422, code="INVALID_LINE_COUNT")
        _, _, row, attempt, attempt_dir = self._attempt_context(project, identity)
        collection = self._read_mapping(attempt_dir / "collection.json")
        process = collection.get("process_evidence")
        process = process if isinstance(process, dict) else {}
        wanted = ("stdout", "stderr") if stream == "both" else (stream,)
        result: dict[str, Any] = {}
        for name in wanted:
            candidates = (
                attempt_dir / f"{name}.log",
                attempt_dir / "collected_run" / f"{name}.log",
            )
            local = next((path for path in candidates if path.is_file()), None)
            if local is not None:
                content = local.read_text(encoding="utf-8", errors="replace").splitlines()
                result[name] = {
                    "source": str(local), "mode": "local_file",
                    "lines": content[-lines:], "available_lines": len(content),
                }
                continue
            archived = process.get(f"{name}_tail")
            archived = archived if isinstance(archived, list) else []
            sources = process.get("sources") if isinstance(process.get("sources"), dict) else {}
            result[name] = {
                "source": sources.get(name), "mode": "collected_tail",
                "lines": [str(item) for item in archived[-lines:]],
                "available_lines": len(archived),
            }
        return {
            "project": project, "run_id": row.run_id,
            "attempt_id": attempt.attempt_id, "streams": result,
            "follow_supported": any(item["mode"] == "local_file" for item in result.values()),
        }

    def attempt_checkpoints(self, project: str, identity: str) -> dict[str, Any]:
        _, _, row, attempt, attempt_dir = self._attempt_context(project, identity)
        collection = self._read_mapping(attempt_dir / "collection.json")
        latest = collection.get("latest_completed_checkpoint") or row.checkpoint.get(
            "latest_completed_checkpoint"
        )
        step = collection.get("latest_completed_checkpoint_step") or row.checkpoint.get(
            "latest_completed_checkpoint_step"
        )
        local_root = attempt_dir / "collected_run"
        local = []
        if local_root.is_dir():
            local = [str(path.relative_to(attempt_dir)) for path in sorted(
                local_root.glob("checkpoint*"))[:200]
            ]
        return {
            "project": project, "run_id": row.run_id,
            "attempt_id": attempt.attempt_id,
            "latest_completed_checkpoint": latest,
            "latest_completed_checkpoint_step": step,
            "local_entries": local,
        }

    def attempt_artifacts(self, project: str, identity: str) -> dict[str, Any]:
        _, _, row, attempt, attempt_dir = self._attempt_context(project, identity)
        collection = self._read_mapping(attempt_dir / "collection.json")
        artifacts = collection.get("artifacts")
        if not isinstance(artifacts, dict):
            artifacts = row.artifacts
        roots = []
        collected = attempt_dir / "collected_run"
        if collected.is_dir():
            roots = [str(path.relative_to(attempt_dir)) for path in sorted(
                item for item in collected.iterdir() if item.is_dir()
            )]
        return {
            "project": project, "run_id": row.run_id,
            "attempt_id": attempt.attempt_id,
            "summary": artifacts, "local_roots": roots,
        }

    def attempt_metrics(self, project: str, identity: str, *, keys: str | None = None,
                        max_points: int = 2000) -> dict[str, Any]:
        _, _, row, attempt, _ = self._attempt_context(project, identity)
        records, source, source_attempt_id = train_metric_records(
            Path(row.run_dir), attempt_id=attempt.attempt_id, exact_attempt=True,
        )
        from ..tracking.tracking_service import store_for
        tracking = store_for(self.runtime)
        live = tracking.metric_records(project, row.run_id, attempt.attempt_id)
        if live:
            if source is None or source.name not in {"metrics.jsonl", "train_metrics.jsonl"}:
                records = live
                source = tracking.db
            source_attempt_id = attempt.attempt_id
        payload = self._metric_payload(
            records, keys=keys, max_points=max_points, source=source,
            source_attempt_id=source_attempt_id, contract=metric_contract(Path(row.run_dir)),
        )
        return {"project": project, "run_id": row.run_id,
                "attempt_id": attempt.attempt_id, **payload}

    def attempt_eval(self, project: str, identity: str) -> dict[str, Any]:
        _, _, row, attempt, _ = self._attempt_context(project, identity)
        source_attempt_id = row.evidence.evaluation.attempt_id
        exact = source_attempt_id == attempt.attempt_id
        variants = row.eval_variants if exact else []
        snapshot = row.evaluation_snapshot if exact else evaluation_snapshot([])
        return {"project": project, "run_id": row.run_id,
                "attempt_id": attempt.attempt_id,
                "source_attempt_id": source_attempt_id, "variants": variants,
                "evaluation_snapshot": snapshot,
                "evidence_status": "INDEXED_EXACT_ATTEMPT" if exact
                else "EXACT_ATTEMPT_NOT_INDEXED"}

    def attempt_events(self, project: str, identity: str) -> dict[str, Any]:
        _, _, row, attempt, attempt_dir = self._attempt_context(project, identity)
        events: list[dict[str, Any]] = []
        sources: list[str] = []
        seen: set[str] = set()
        candidates = (
            attempt_dir / "collected_run" / "events.jsonl",
            attempt_dir / "events.jsonl",
            Path(row.run_dir) / "events.jsonl",
        )
        for candidate in candidates:
            records = read_jsonl(candidate)
            if not records:
                continue
            is_run_timeline = candidate == Path(row.run_dir) / "events.jsonl"
            added = False
            for event in records:
                event_attempt = event.get("attempt_id")
                if event_attempt != attempt.attempt_id and (
                    is_run_timeline or event_attempt is not None
                ):
                    continue
                identity_key = json.dumps(event, sort_keys=True, default=str)
                if identity_key in seen:
                    continue
                seen.add(identity_key)
                events.append(event)
                added = True
            if added:
                sources.append(str(candidate))
        for event in events:
            event["ts"] = parse_iso_ts(event.get("timestamp"))
        events.sort(key=lambda event: event.get("ts") or 0)
        return {"project": project, "run_id": row.run_id,
                "attempt_id": attempt.attempt_id,
                "sources": sources, "events": events}

    def run_detail(self, project: str, run_id: str) -> dict[str, Any]:
        _, _, row = self.resolve_scope(project, OperationScopeType.RUN, run_id)
        payload = run_dto(row)
        payload["is_terminal"] = row.is_terminal
        payload["failure_assessment"] = self.run_failure_assessment(row)
        return payload

    def run_metrics(self, project: str, run_id: str, *, keys: str | None = None,
                    max_points: int = 2000) -> dict[str, Any]:
        _, _, row = self.resolve_scope(project, OperationScopeType.RUN, run_id)
        records, source, source_attempt_id = train_metric_records(Path(row.run_dir))
        from ..tracking.tracking_service import store_for
        current = self.run_attempts(project, run_id)["current_attempt_id"]
        tracking = store_for(self.runtime)
        live = tracking.metric_records(project, run_id, current) if current else []
        if live:
            if source_attempt_id != current or source is None or source.name not in {"metrics.jsonl", "train_metrics.jsonl"}:
                records = live
                source = tracking.db
            source_attempt_id = current
        return self._metric_payload(
            records, keys=keys, max_points=max_points, source=source,
            source_attempt_id=source_attempt_id, contract=metric_contract(Path(row.run_dir)),
        )

    def run_eval(self, project: str, run_id: str) -> dict[str, Any]:
        _, _, row = self.resolve_scope(project, OperationScopeType.RUN, run_id)
        return {
            "source_attempt_id": row.evidence.evaluation.attempt_id,
            "variants": row.eval_variants,
            "evaluation_snapshot": row.evaluation_snapshot,
        }

    def run_events(self, project: str, run_id: str) -> dict[str, Any]:
        _, _, row = self.resolve_scope(project, OperationScopeType.RUN, run_id)
        events = read_jsonl(Path(row.run_dir) / "events.jsonl")
        for event in events:
            event["ts"] = parse_iso_ts(event.get("timestamp"))
        return {"events": events}
