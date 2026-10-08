"""Context for ExperimentServerApplication."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import yaml
from ..projects.authored_runs import authored_run_placeholder
from ..application_errors import ApplicationError
from ..outward import operational_decision
from ..schemas import OperationScope, OperationScopeType, ResearchProject
from ..tracking.metric_contract import metric_result, finite

class RunContext:
    """Context operations; state belongs to the application facade."""

    def resolve_scope(
        self, project_name: str, scope_type: OperationScopeType | str, object_id: str,
    ) -> tuple[OperationScope, ResearchProject, Any]:
        try:
            project = self.runtime.project(project_name)
        except KeyError as exc:
            raise ApplicationError(str(exc).strip("'"), status_code=404,
                                   code="UNKNOWN_PROJECT") from exc
        kind = OperationScopeType(scope_type)
        if kind == OperationScopeType.PROJECT:
            if object_id != project.project:
                raise ApplicationError("project object_id must equal project",
                                       status_code=404, code="UNKNOWN_PROJECT")
            resolved: Any = project
        elif kind == OperationScopeType.CAMPAIGN:
            resolved = next((
                campaign for campaign in project.campaigns if campaign.name == object_id
            ), None)
            if resolved is None:
                raise ApplicationError(f"unknown campaign: {object_id}", status_code=404,
                                       code="UNKNOWN_CAMPAIGN")
        elif kind == OperationScopeType.RUN:
            resolved = self.runtime.index.get_run(project.project, object_id)
            if resolved is None:
                resolved = authored_run_placeholder(project, object_id)
            if resolved is None:
                raise ApplicationError(f"unknown run: {object_id}", status_code=404,
                                       code="UNKNOWN_RUN")
        else:
            if "::" not in object_id:
                raise ApplicationError("attempt object_id must be run_id::attempt_id",
                                       status_code=422, code="INVALID_ATTEMPT_ID")
            run_id, attempt_id = object_id.rsplit("::", 1)
            row = self.runtime.index.get_run(project.project, run_id)
            if row is None:
                raise ApplicationError(f"unknown run: {run_id}", status_code=404,
                                       code="UNKNOWN_RUN")
            resolved = next((item for item in row.attempts if item.attempt_id == attempt_id), None)
            if resolved is None:
                raise ApplicationError(f"unknown attempt: {attempt_id}", status_code=404,
                                       code="UNKNOWN_ATTEMPT")
        return OperationScope(project=project.project, scope_type=kind, object_id=object_id), project, resolved

    @staticmethod
    def _operational_decision(value: Any) -> dict[str, Any]:
        return operational_decision(value)

    def _attempt_context(self, project: str, identity: str):
        scope, configured, attempt = self.resolve_scope(
            project, OperationScopeType.ATTEMPT, identity,
        )
        run_id, attempt_id = identity.rsplit("::", 1)
        row = self.runtime.index.get_run(project, run_id)
        assert row is not None
        attempt_dir = Path(row.run_dir) / "attempts" / attempt_id
        return scope, configured, row, attempt, attempt_dir

    @staticmethod
    def _read_mapping(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _read_yaml_mapping(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _path_mtime(path: Path) -> float | None:
        try:
            return path.stat().st_mtime
        except OSError:
            return None

    @staticmethod
    def _manifest_at(root: Path, names: tuple[str, ...]) -> tuple[dict[str, Any], Path | None]:
        for name in names:
            path = root / name
            payload = (
                RunContext._read_mapping(path)
                if path.suffix == ".json"
                else RunContext._read_yaml_mapping(path)
            )
            if payload:
                return payload, path
        return {}, None

    @staticmethod
    def _metric_payload(records: list[dict[str, Any]], *, keys: str | None,
                        max_points: int, source: Path | None,
                        source_attempt_id: str | None, contract: dict | None = None) -> dict[str, Any]:
        if max_points <= 0:
            raise ApplicationError("max_points must be positive", status_code=422,
                                   code="INVALID_MAX_POINTS")
        wanted = [key.strip() for key in keys.split(",") if key.strip()] if keys else None
        total = len(records)
        available = sorted({
            key
            for record in records
            for key, value in record.items()
            if key not in ("step", "timestamp")
            and isinstance(value, (str, int, float, bool))
        })
        numeric_available = sorted({
            key
            for record in records
            for key, value in record.items()
            if key not in ("step", "timestamp")
            and isinstance(value, (int, float))
        })
        if total <= max_points:
            sampled = records
        elif max_points == 1:
            sampled = [records[-1]]
        else:
            indices = [(index * (total - 1)) // (max_points - 1)
                       for index in range(max_points)]
            sampled = [records[index] for index in indices]
        points = []
        for record in sampled:
            point: dict[str, Any] = {"step": record.get("step"),
                                     "timestamp": record.get("timestamp")}
            for key in wanted or [k for k in record if k not in ("step", "timestamp")]:
                value = record.get(key)
                if (
                    finite(value)
                    or wanted is not None and isinstance(value, (str, bool))
                ):
                    point[key] = value
            points.append(point)
        contract = contract or {}
        result = metric_result(sampled, contract.get("metrics_schema"),
                               protocol_id=contract.get("protocol_id"), source=str(source) if source else None)
        return {
            "metrics": result, "evaluation": contract,
            "points": points, "keys": wanted or numeric_available,
            "missing_keys": [key for key in wanted or [] if key not in available],
            "total_records": total, "downsampled": total > len(sampled),
            "source": str(source) if source else None,
            "source_attempt_id": source_attempt_id,
        }
