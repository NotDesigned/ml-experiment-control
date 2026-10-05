"""Read-only REST routes. Contract: docs/api_contract.md."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from .sse import IndexEventResponse
from pydantic import BaseModel, Field, model_validator

from ..application import ApplicationError
from ..api_contract import DaemonHealth, ProjectRegistrationResponse
from ..campaign_lifecycle import campaign_snapshot
from ..collectord import Collector
from ..ingest.indexer import RunIndex, index_project
from ..ingest.runscan import parse_iso_ts, read_jsonl
from ..outward import operational_decision, sanitized_outward
from ..schemas import (
    OperationScopeType,
    ProjectLifecycleState,
    ResearchProject,
    RunIndexRow,
    TERMINAL_RUN_STATES,
)
from ..terminal_snapshot import (build_snapshot, is_current_collector_error,
                                 snapshot_payload)
from .errors import application_http_error

router = APIRouter(prefix="/api")

class ArchiveCampaignRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=4000)


class ProjectLifecycleRequest(BaseModel):
    reason: str = Field(default="", max_length=4000)


class DaemonPathProjectSource(BaseModel):
    """An existing manifest path in the daemon host filesystem namespace."""

    kind: Literal["daemon_path"] = "daemon_path"
    manifest_path: str = Field(min_length=1, max_length=4096)


class ProjectRegisterRequest(BaseModel):
    # `project_file` remains a protocol-v1 compatibility field. New clients use
    # the explicit source locator so a client-local path is never mistaken for
    # a path on a remote daemon host.
    project_file: Optional[str] = Field(default=None, min_length=1, max_length=4096)
    source: Optional[DaemonPathProjectSource] = None

    @model_validator(mode="after")
    def require_one_source(self) -> "ProjectRegisterRequest":
        if (self.project_file is None) == (self.source is None):
            raise ValueError("provide exactly one of project_file or source")
        return self

    def daemon_manifest_path(self) -> Path:
        value = self.source.manifest_path if self.source is not None else self.project_file
        assert value is not None
        return Path(value)


class ProjectImportPreviewRequest(BaseModel):
    source: dict[str, Any]
    project: Optional[str] = Field(default=None, max_length=128)
    title: Optional[str] = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_source(self) -> "ProjectImportPreviewRequest":
        if self.source.get("kind") != "daemon_path":
            raise ValueError("only daemon_path Project discovery is currently supported")
        root = self.source.get("repository_root")
        if not isinstance(root, str) or not root or len(root) > 4096:
            raise ValueError("source.repository_root must be a non-empty path")
        return self

    def repository_root(self) -> Path:
        return Path(str(self.source["repository_root"]))


class ProjectImportExecuteRequest(BaseModel):
    confirmation: str = Field(min_length=1, max_length=256)


class SourceRevisionPreviewRequest(BaseModel):
    project: str = Field(min_length=1, max_length=128)
    proposal: dict[str, Any]


class SourceRevisionExecuteRequest(BaseModel):
    confirmation: str = Field(min_length=1, max_length=256)


class RefreshRequest(BaseModel):
    project: Optional[str] = Field(default=None, max_length=256)


class AttemptRetryRequest(BaseModel):
    max_gpu_hours: float = Field(gt=0)
    new_attempt_id: Optional[str] = Field(default=None, pattern=r"^attempt-[0-9]{3,}$")
    reason: str = Field(default="", max_length=4000)


class AttemptCancelRequest(BaseModel):
    reason: str = Field(default="", max_length=4000)


@router.get("/health", response_model=DaemonHealth)
def health(request: Request) -> DaemonHealth:
    from importlib.metadata import PackageNotFoundError, version

    try:
        server_version = version("ml-experiment-server")
    except PackageNotFoundError:  # source-tree execution without installation
        server_version = "0.1.0+source"
    return DaemonHealth(
        status="ok",
        server_version=server_version,
        authentication=request.app.state.auth_mode,
        transport_security=request.url.scheme,
        projects=len(request.app.state.projects),
        workspace_id=request.app.state.runtime.workspace_id,
        collector_enabled=(
            request.app.state.collector is not None
            and bool(getattr(request.app.state, "collector_owner", False))
        ),
        collector_requested=request.app.state.collector is not None,
        collector_error=(
            getattr(request.app.state, "collector_error", None)
            or request.app.state.index.get_meta("collector_last_error")
            or None
        ),
        project_write_recovery_errors=(
            request.app.state.project_write_recovery_errors
        ),
        project_writes=request.app.state.config.action_runtime.allow_project_writes,
        source_imports=request.app.state.config.action_runtime.allow_source_imports,
        scheduler_mutations=(
            request.app.state.config.action_runtime.allow_scheduler_mutations
        ),
        local_evidence_rebuild=(
            request.app.state.config.action_runtime.allow_local_evidence_rebuild
        ),
        telemetry_enabled=request.app.state.runtime.telemetry.enabled,
    )












def _state(request: Request) -> tuple[RunIndex, list[ResearchProject], Optional[Collector]]:
    st = request.app.state
    return st.index, st.projects, getattr(st, "collector", None)


def _find_project(projects: list[ResearchProject], name: str) -> ResearchProject:
    for project in projects:
        if project.project == name:
            return project
    raise HTTPException(status_code=404, detail=f"unknown project: {name}")


def _project_lifecycle_response(request: Request, project: str, action: str,
                                state: ProjectLifecycleState,
                                data: ProjectLifecycleRequest):
    try:
        return request.app.state.application.project_lifecycle_transition(
            project, action, state, reason=data.reason,
        )
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


def _run_or_404(index: RunIndex, project: str, run_id: str) -> RunIndexRow:
    row = index.get_run(project, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"unknown run: {project}/{run_id}")
    return row


@router.get("/project-lifecycle")
def project_lifecycle_list(request: Request):
    return request.app.state.application.project_lifecycle_list()


@router.post(
    "/project-lifecycle/register", response_model=ProjectRegistrationResponse,
)
def project_lifecycle_register(
    data: ProjectRegisterRequest, request: Request,
) -> ProjectRegistrationResponse:
    try:
        return ProjectRegistrationResponse.model_validate(
            request.app.state.application.project_register(data.daemon_manifest_path())
        )
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.post("/project-imports/preview")
def project_import_preview(data: ProjectImportPreviewRequest, request: Request):
    try:
        return request.app.state.application.project_import_preview(
            data.repository_root(), project=data.project, title=data.title,
        )
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.post("/project-imports/{import_id}/execute")
def project_import_execute(
    import_id: str, data: ProjectImportExecuteRequest, request: Request,
):
    try:
        return request.app.state.application.project_import_execute(
            import_id, data.confirmation,
        )
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.post("/source-revisions/preview")
def source_revision_preview(data: SourceRevisionPreviewRequest, request: Request):
    try:
        return request.app.state.application.source_revision_preview(
            data.project, data.proposal,
        )
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.post("/source-revisions/{import_id}/execute")
def source_revision_execute(
    import_id: str, data: SourceRevisionExecuteRequest, request: Request,
):
    try:
        return request.app.state.application.source_revision_execute(
            import_id, data.confirmation,
        )
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.get("/projects/{project}/source-revisions/{source_id}")
def source_revision_get(project: str, source_id: str, request: Request):
    try:
        return request.app.state.application.source_revision_get(project, source_id)
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.post("/project-lifecycle/{project}/pause")
def project_lifecycle_pause(project: str, data: ProjectLifecycleRequest, request: Request):
    return _project_lifecycle_response(
        request, project, "pause", ProjectLifecycleState.PAUSED, data,
    )


@router.post("/project-lifecycle/{project}/resume")
def project_lifecycle_resume(project: str, data: ProjectLifecycleRequest, request: Request):
    return _project_lifecycle_response(
        request, project, "resume", ProjectLifecycleState.ACTIVE, data,
    )


@router.post("/project-lifecycle/{project}/archive")
def project_lifecycle_archive(project: str, data: ProjectLifecycleRequest, request: Request):
    return _project_lifecycle_response(
        request, project, "archive", ProjectLifecycleState.ARCHIVED, data,
    )


@router.post("/project-lifecycle/{project}/restore")
def project_lifecycle_restore(project: str, data: ProjectLifecycleRequest, request: Request):
    return _project_lifecycle_response(
        request, project, "restore", ProjectLifecycleState.PAUSED, data,
    )


@router.post("/project-lifecycle/{project}/unregister")
def project_lifecycle_unregister(project: str, data: ProjectLifecycleRequest, request: Request):
    try:
        return request.app.state.application.project_unregister(project, reason=data.reason)
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.post("/project-lifecycle/unregister-all")
def project_lifecycle_unregister_all(data: ProjectLifecycleRequest, request: Request):
    return request.app.state.application.project_unregister_all(reason=data.reason)


def _stale_layers(row: RunIndexRow) -> list[str]:
    layers = row.evidence
    return [name for name, layer in (
        ("scheduler", layers.scheduler), ("worker", layers.worker),
        ("process", layers.process), ("model", layers.model),
        ("evaluation", layers.evaluation)) if layer.stale]


def _attention(
    rows: list[RunIndexRow], collector_statuses,
    failure_assessments: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    failure_assessments = failure_assessments or {}
    items: list[dict[str, Any]] = []
    for row in rows:
        state = (row.scheduler_state or "").upper()
        if state in {"FAILED", "PREEMPTED"}:
            summary = failure_assessments.get(row.run_id, {}).get("failure_summary")
            summary = summary if isinstance(summary, dict) else {}
            failure = (
                summary.get("failure_class")
                if str(summary.get("applicability") or "").upper() == "APPLICABLE"
                else None
            )
            items.append({"kind": "failed_run", "run_id": row.run_id,
                          "detail": f"{state}" + (f" ({failure})" if failure else "")})
        stale = _stale_layers(row)
        if stale:
            reasons = "; ".join(
                filter(None, [getattr(row.evidence, layer).stale_reason for layer in stale]))
            items.append({"kind": "stale_evidence", "run_id": row.run_id,
                          "detail": reasons or f"stale layers: {', '.join(stale)}"})
    rows_by_id = {row.run_id: row for row in rows}
    for status in collector_statuses:
        row = rows_by_id.get(status.run_id)
        if row is not None and is_current_collector_error(row, status):
            items.append({"kind": "collector_error", "run_id": status.run_id,
                          "detail": status.last_error})
    return items


def _campaign_membership(row: RunIndexRow, campaign: str):
    return next(
        (binding.membership for binding in row.campaign_memberships
         if binding.campaign == campaign),
        None,
    )


def _belongs_to_campaign(row: RunIndexRow, campaign: str) -> bool:
    return row.campaign == campaign or _campaign_membership(row, campaign) is not None


def _role_summaries(rows: list[RunIndexRow], campaign: str | None = None) -> list[dict[str, Any]]:
    summaries = []
    resolved = [
        (row, (_campaign_membership(row, campaign).role
               if campaign and _campaign_membership(row, campaign) else row.role))
        for row in rows
    ]
    for row, role in sorted(resolved, key=lambda item: (item[1] or "~", item[0].run_id)):
        summaries.append({
            "role": role,
            "run_id": row.run_id,
            "scheduler_state": row.scheduler_state,
            "stale": bool(_stale_layers(row)),
        })
    return summaries


@router.get("/projects")
def list_projects(request: Request):
    index, projects, _ = _state(request)
    payload = []
    for project in projects:
        rows = index.list_runs(project.project)
        counts: dict[str, int] = {}
        for row in rows:
            key = row.scheduler_state or "UNKNOWN"
            counts[key] = counts.get(key, 0) + 1
        statuses = index.collector_statuses(project.project)
        assessments = {
            row.run_id: request.app.state.application.run_failure_assessment(row)
            for row in rows
        }
        payload.append({
            "project": project.project,
            "title": project.title,
            "run_counts": counts,
            "attention_count": len(_attention(rows, statuses, assessments)),
        })
    return payload


@router.get("/terminal/snapshot")
def terminal_snapshot(request: Request, project: Optional[str] = None):
    """Serve the server-owned read model to a terminal renderer.

    This endpoint intentionally does not initiate a collection cycle. The
    server collector is the sole owner of scheduler observation; the TUI only
    consumes the already-indexed snapshot.
    """
    projects = [
        item for item in request.app.state.projects
        if project is None or item.project == project
    ]
    if project is not None and not projects:
        raise HTTPException(status_code=404, detail=f"unknown project: {project}")
    payload = snapshot_payload(build_snapshot(
        request.app.state.index,
        projects,
    ))
    for project_name, rows in payload["runs"].items():
        for row in rows:
            run_id = str(row.get("run_id") or "")
            indexed_row = request.app.state.index.get_run(project_name, run_id)
            row["failure_assessment"] = (
                request.app.state.application.run_failure_assessment(indexed_row)
                if indexed_row is not None else
                {"failure_summary": None, "diagnostic_evidence": []}
            )
    payload["scale"] = {
        "projects": len(payload["projects"]),
        "runs": sum(len(rows) for rows in payload["runs"].values()),
        "runs_by_project": {
            project: len(rows) for project, rows in payload["runs"].items()
        },
    }
    return payload


@router.post("/terminal/refresh")
def terminal_refresh(data: RefreshRequest, request: Request):
    """Reindex server-owned files without creating a second client runtime."""
    projects = request.app.state.projects
    selected = [item for item in projects if data.project in {None, item.project}]
    if data.project is not None and not selected:
        raise HTTPException(status_code=404, detail=f"unknown project: {data.project}")
    for project in selected:
        index_project(request.app.state.index, project)
    return terminal_snapshot(request, project=data.project)


@router.get("/objects")
def object_show(
    request: Request, project: str, scope_type: OperationScopeType, object_id: str,
):
    try:
        return request.app.state.application.object_show(project, scope_type, object_id)
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.get("/campaigns/{project_name}")
def list_campaign_lifecycle(project_name: str, request: Request):
    try:
        return request.app.state.application.campaign_list(project_name)
    except (ApplicationError, KeyError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/campaigns/{project_name}/{campaign_name}")
def campaign_lifecycle(project_name: str, campaign_name: str, request: Request):
    try:
        return request.app.state.application.campaign_status(project_name, campaign_name)
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.post("/campaigns/{project_name}/{campaign_name}/archive")
def prepare_campaign_archive(
    project_name: str, campaign_name: str, data: ArchiveCampaignRequest, request: Request,
):
    try:
        return request.app.state.application.prepare_campaign_archive(
            project_name, campaign_name, reason=data.reason,
        )
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.get("/projects/{project_name}/overview")
def project_overview(project_name: str, request: Request):
    index, projects, collector = _state(request)
    project = _find_project(projects, project_name)
    rows = index.list_runs(project.project)
    statuses = index.collector_statuses(project.project)
    assessments = {
        row.run_id: request.app.state.application.run_failure_assessment(row)
        for row in rows
    }

    campaigns = []
    for campaign in project.campaigns:
        campaign_rows = [row for row in rows if _belongs_to_campaign(row, campaign.name)]
        relationship_counts: dict[str, int] = {}
        for row in campaign_rows:
            relationship = row.campaign_binding.relationship.value
            relationship_counts[relationship] = relationship_counts.get(relationship, 0) + 1
        revision = campaign.current_revision
        lifecycle = campaign_snapshot(index, project, campaign.name)
        campaigns.append({
            "name": campaign.name,
            "lifecycle_state": lifecycle["lifecycle_state"],
            "current_revision_id": revision.revision_id if revision else None,
            "declared_run_count": len(revision.memberships) if revision else None,
            "relationship_counts": relationship_counts,
            "roles": _role_summaries(campaign_rows, campaign.name),
        })

    counts: dict[str, int] = {}
    for row in rows:
        key = row.scheduler_state or "UNKNOWN"
        counts[key] = counts.get(key, 0) + 1

    last_cycle = index.get_meta("collector_last_cycle_at")
    cycle_started = index.get_meta("collector_cycle_started_at")
    collector_error = index.get_meta("collector_last_error") or None
    return {
        "project": project.project,
        "title": project.title,
        "campaigns": campaigns,
        "run_states": counts,
        "attention": _attention(rows, statuses, assessments),
        "collector": {
            "enabled": collector is not None,
            "last_cycle_at": float(last_cycle) if last_cycle else None,
            "cycle_started_at": float(cycle_started) if cycle_started else None,
            "cycle_in_progress": bool(cycle_started),
            "last_error": collector_error,
        },
    }




@router.get("/runs/{project_name}/{run_id}")
def run_detail(project_name: str, run_id: str, request: Request):
    try:
        return request.app.state.application.run_detail(project_name, run_id)
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.get("/runs/{project_name}/{run_id}/attempts")
def run_attempts(project_name: str, run_id: str, request: Request):
    try:
        return request.app.state.application.run_attempts(project_name, run_id)
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.get("/runs/{project_name}/{run_id}/validate")
def run_validate(project_name: str, run_id: str, request: Request):
    try:
        return request.app.state.application.run_validate(project_name, run_id)
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.get("/runs/{project_name}/{run_id}/metrics")
def run_metrics(project_name: str, run_id: str, request: Request,
                keys: Optional[str] = None, max_points: int = 2000):
    try:
        return request.app.state.application.run_metrics(
            project_name, run_id, keys=keys, max_points=max_points,
        )
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.get("/runs/{project_name}/{run_id}/eval")
def run_eval(project_name: str, run_id: str, request: Request):
    try:
        return request.app.state.application.run_eval(project_name, run_id)
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.get("/runs/{project_name}/{run_id}/events")
def run_events(project_name: str, run_id: str, request: Request):
    try:
        return request.app.state.application.run_events(project_name, run_id)
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.get("/attempts/{project_name}/{attempt_id}/bundle")
def attempt_bundle(project_name: str, attempt_id: str, request: Request):
    """Return exact Attempt evidence for any authorized client."""
    try:
        application = request.app.state.application
        return {
            "show": application.attempt_show(project_name, attempt_id),
            "validation": application.attempt_validate(project_name, attempt_id),
            "metrics": application.attempt_metrics(project_name, attempt_id, max_points=50),
            "checkpoints": application.attempt_checkpoints(project_name, attempt_id),
            "artifacts": application.attempt_artifacts(project_name, attempt_id),
        }
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.get("/attempts/{project_name}/{attempt_id}/{view}")
def attempt_view(
    project_name: str, attempt_id: str, view: str, request: Request,
    stream: str = "both", lines: int = 80, keys: Optional[str] = None,
    max_points: int = 2000,
):
    """Expose the bounded Attempt read operations through one resource route."""
    application = request.app.state.application
    try:
        if view == "show":
            return application.attempt_show(project_name, attempt_id)
        if view == "logs":
            return application.attempt_logs(
                project_name, attempt_id, stream=stream, lines=lines,
            )
        if view == "checkpoints":
            return application.attempt_checkpoints(project_name, attempt_id)
        if view == "artifacts":
            return application.attempt_artifacts(project_name, attempt_id)
        if view == "metrics":
            return application.attempt_metrics(
                project_name, attempt_id, keys=keys, max_points=max_points,
            )
        if view == "eval":
            return application.attempt_eval(project_name, attempt_id)
        if view == "events":
            return application.attempt_events(project_name, attempt_id)
        if view == "validate":
            return application.attempt_validate(project_name, attempt_id)
    except ApplicationError as exc:
        raise application_http_error(exc) from exc
    raise HTTPException(status_code=404, detail=f"unknown Attempt view: {view}")


@router.post("/attempts/{project_name}/{attempt_id}/retry")
def attempt_retry(
    project_name: str, attempt_id: str, data: AttemptRetryRequest, request: Request,
):
    try:
        application = request.app.state.application
        policy = request.app.state.config.action_runtime
        resource_approval = policy.scheduler_resource_approval
        return application.prepare_attempt_retry(
            project_name, attempt_id, new_attempt_id=data.new_attempt_id,
            max_gpu_hours=(
                data.max_gpu_hours if resource_approval == "budget_cap" else None
            ),
            reason=data.reason, resource_approval=resource_approval,
        )
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.post("/attempts/{project_name}/{attempt_id}/cancel")
def attempt_cancel(
    project_name: str, attempt_id: str, data: AttemptCancelRequest, request: Request,
):
    try:
        return request.app.state.application.prepare_attempt_cancel(
            project_name, attempt_id, reason=data.reason,
        )
    except ApplicationError as exc:
        raise application_http_error(exc) from exc


@router.get("/collector/status")
def collector_status(request: Request):
    index, _, collector = _state(request)
    last_cycle = index.get_meta("collector_last_cycle_at")
    cycle_started = index.get_meta("collector_cycle_started_at")
    last_error = (getattr(request.app.state, "collector_error", None)
                  or index.get_meta("collector_last_error") or None)
    return {
        "enabled": collector is not None and bool(
            getattr(request.app.state, "collector_owner", False)
        ),
        "requested": collector is not None,
        "owner": bool(getattr(request.app.state, "collector_owner", False)),
        "poll_interval_seconds": collector.config.poll_interval_seconds if collector else None,
        "last_cycle_at": float(last_cycle) if last_cycle else None,
        "cycle_started_at": float(cycle_started) if cycle_started else None,
        "cycle_in_progress": bool(cycle_started),
        "last_error": last_error,
        "runs": [s.model_dump() for s in index.collector_statuses()],
    }


@router.get("/stream")
async def stream(request: Request):
    broker = request.app.state.broker
    return IndexEventResponse(broker.stream(), ping=15, send_timeout=15, sep="\n")
