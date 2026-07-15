"""Read-only polling collector.

Periodically invokes the project's experimentctl with OBSERVATION VERBS ONLY,
then rescans run directories into the index. Mutation verbs are rejected at
command-construction time — there is deliberately no collector path to
submit/cancel/stage/prepare.
"""

from __future__ import annotations

import errno
import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .actions.store import ActionStore
from .campaign_lifecycle import campaign_snapshot
from .controller_gateway import ControllerCall, ProjectControllerGateway
from .ingest.indexer import RunIndex, index_project
from .schemas import (
    CampaignRelationship,
    ResearchProject,
    RunIndexRow,
    TERMINAL_RUN_STATES,
)

# Hard allowlist. Anything else — most importantly submit/cancel/stage/prepare —
# must raise before a subprocess is even constructed.
OBSERVATION_VERBS = frozenset({"observe", "decide", "status", "collect"})

# These states mean that invoking today's authored campaign file could target
# the wrong scientific object or reinterpret a historical Run. Legacy and
# unresolved rows retain the v1 behavior until they can be migrated.
_UNSAFE_CAMPAIGN_RELATIONSHIPS = frozenset({
    CampaignRelationship.CAMPAIGN_REVISION_DRIFT,
    CampaignRelationship.DUPLICATE_RUN_ID,
    CampaignRelationship.PROJECT_MISMATCH,
    CampaignRelationship.ROLE_MISMATCH,
    CampaignRelationship.UNDECLARED_RUN,
})

_SUBPROCESS_TIMEOUT = 300
_SUBMISSION_OPERATIONS = frozenset({
    "SUBMIT_RUN", "RETRY_ATTEMPT", "RUN_EVALUATION",
})


class ForbiddenVerbError(ValueError):
    """Raised when a non-observation verb reaches the collector."""


PlannedCall = ControllerCall


@dataclass
class CollectorConfig:
    poll_interval_seconds: int = 20
    dry_run: bool = False


class CollectorLease:
    """Non-blocking, process-wide ownership guard for one workspace collector.

    A collector updates canonical observation and derived-decision files, so
    two ``serve`` processes must not both run cycles for the same workspace.
    The lock remains held for the owner lifetime and is released automatically
    if that process exits.
    """

    def __init__(self, index_db: Path) -> None:
        self.path = Path(str(index_db) + ".collector.lock")
        self._fd: int | None = None

    def acquire(self) -> bool:
        if self._fd is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        except OSError as exc:
            os.close(fd)
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                return False
            raise
        try:
            os.ftruncate(fd, 0)
            payload = f"pid={os.getpid()}\n".encode("ascii")
            offset = 0
            while offset < len(payload):
                written = os.write(fd, payload[offset:])
                if written <= 0:
                    raise OSError(errno.EIO, "could not write workspace lease metadata")
                offset += written
            os.fsync(fd)
        except BaseException:
            # Closing the descriptor also releases flock.  Do not publish it
            # through ``self._fd`` until the complete lease record is durable.
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


@dataclass
class Collector:
    index: RunIndex
    projects: list[ResearchProject]
    config: CollectorConfig = field(default_factory=CollectorConfig)
    controller: ProjectControllerGateway = field(default_factory=ProjectControllerGateway)
    action_store: ActionStore | None = None

    @staticmethod
    def _current_attempt_id(row: RunIndexRow) -> str | None:
        attempt_id = row.evidence.scheduler.attempt_id
        if attempt_id:
            return attempt_id
        submitted = [item.attempt_id for item in row.attempts if item.has_submission]
        return submitted[-1] if submitted else None

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _verified_execution_campaign(
        self, project: ResearchProject, row: RunIndexRow,
        actions: list[dict[str, object]],
    ) -> Path | None:
        """Resolve the immutable campaign that actually submitted a drifted Run.

        Daemon-owned control roots require an execution copy of an authored
        Campaign.  That operational-only copy has a different byte identity,
        so the resulting Run correctly appears revision-drifted relative to
        today's authored file.  It is safe to observe only through the exact
        VERIFIED Action copy that froze and submitted the current Attempt.
        """
        if (
            self.action_store is None
            or row.campaign_binding.relationship
            != CampaignRelationship.CAMPAIGN_REVISION_DRIFT
        ):
            return None
        attempt_id = self._current_attempt_id(row)
        origin_revision = str(row.campaign_binding.origin_revision or "")
        if (
            not attempt_id
            or not origin_revision.startswith("campaign.")
            or row.campaign_binding.origin_project != project.project
            or row.campaign_binding.origin_campaign != row.campaign
        ):
            return None
        attempt = next(
            (item for item in row.attempts if item.attempt_id == attempt_id), None,
        )
        if attempt is None or not attempt.has_submission or not attempt.backend_job_id:
            return None

        candidates: list[tuple[str, Path]] = []
        for action in actions:
            execution = action.get("execution")
            scope = action.get("scope")
            operation = action.get("operation")
            scope_type = scope.get("scope_type") if isinstance(scope, dict) else None
            scope_object = str(scope.get("object_id") or "") if isinstance(scope, dict) else ""
            exact_scope = (
                (
                    operation == "SUBMIT_RUN"
                    and scope_type == "run"
                    and scope_object == row.run_id
                )
                or (
                    operation == "RETRY_ATTEMPT"
                    and scope_type == "attempt"
                    and scope_object.startswith(f"{row.run_id}::")
                )
                or (
                    operation == "RUN_EVALUATION"
                    and (
                        (scope_type == "run" and scope_object == row.run_id)
                        or (
                            scope_type == "attempt"
                            and scope_object.startswith(f"{row.run_id}::")
                        )
                    )
                )
            )
            if (
                operation not in _SUBMISSION_OPERATIONS
                or not exact_scope
                or not isinstance(execution, dict)
                or execution.get("status") != "VERIFIED"
                or not isinstance(scope, dict)
                or scope.get("project") != project.project
                or action.get("run_id") != row.run_id
                or action.get("attempt_id") != attempt_id
            ):
                continue
            result = execution.get("result")
            submission = result.get("submission") if isinstance(result, dict) else None
            observation = result.get("observation") if isinstance(result, dict) else None
            bound_jobs = {
                str(value.get("backend_job_id") or "")
                for value in (submission, observation)
                if isinstance(value, dict) and value.get("backend_job_id")
            }
            if bound_jobs != {str(attempt.backend_job_id)}:
                continue
            action_id = str(action.get("action_id") or "")
            try:
                action_root = self.action_store.directory(action_id).resolve()
                candidate = Path(str(action.get("execution_campaign_file") or ""))
                if candidate.is_symlink():
                    continue
                campaign = candidate.resolve(strict=True)
                campaign.relative_to(action_root)
            except (FileNotFoundError, OSError, ValueError):
                continue
            if not campaign.is_file():
                continue
            try:
                actual = self._file_sha256(campaign)
            except OSError:
                continue
            if (
                action.get("execution_campaign_sha256") != f"sha256:{actual}"
                or origin_revision != f"campaign.{actual}"
            ):
                continue
            candidates.append((str(action.get("created_at") or ""), campaign))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]
    last_cycle_at: Optional[float] = None

    def build_command(self, project: ResearchProject, campaign_file: Path,
                      run_id: str, verb: str, *,
                      attempt_id: str | None = None) -> PlannedCall:
        if verb not in OBSERVATION_VERBS:
            raise ForbiddenVerbError(
                f"verb {verb!r} is not an observation verb; collector refuses "
                f"anything outside {sorted(OBSERVATION_VERBS)}"
            )
        return self.controller.build(
            project, campaign_file, verb, run_id, attempt_id=attempt_id,
        )

    def _campaign_files(self, project: ResearchProject) -> dict[str, Path]:
        """campaign name → campaign YAML from the project-owned catalog."""
        base = project.base_dir or Path(".")
        mapping: dict[str, Path] = {}
        for campaign in project.campaigns:
            if not campaign.file:
                continue
            path = Path(campaign.file)
            if not path.is_absolute():
                path = (base / path).resolve()
            if path.is_file():
                mapping[campaign.name] = path
        return mapping

    def plan_cycle(self) -> list[PlannedCall]:
        """Decide which (run, verb) pairs the next cycle would execute.

        Only non-terminal runs whose campaign file is declared by the project
        are actively polled; everything else stays passive file scanning.
        """
        calls: list[PlannedCall] = []
        actions_by_run: dict[tuple[str, str], list[dict[str, object]]] = {}
        if self.action_store is not None:
            for action in self.action_store.list_all():
                scope = action.get("scope")
                if not isinstance(scope, dict):
                    continue
                key = (str(scope.get("project") or ""), str(action.get("run_id") or ""))
                actions_by_run.setdefault(key, []).append(action)
        for project in self.projects:
            campaign_files = self._campaign_files(project)
            inactive_campaigns = {
                campaign.name for campaign in project.campaigns
                if campaign_snapshot(self.index, project, campaign.name)["lifecycle_state"]
                in {"COMPLETED", "ARCHIVED"}
            }
            for row in self.index.list_runs(project.project):
                if (row.scheduler_state or "").upper() in TERMINAL_RUN_STATES:
                    continue
                execution_campaign = self._verified_execution_campaign(
                    project, row,
                    actions_by_run.get((project.project, row.run_id), []),
                )
                if (
                    row.campaign_binding.relationship in _UNSAFE_CAMPAIGN_RELATIONSHIPS
                    and execution_campaign is None
                ):
                    continue
                if (row.campaign or "") in inactive_campaigns:
                    continue
                campaign_file = (
                    execution_campaign or campaign_files.get(row.campaign or "")
                )
                if campaign_file is None:
                    continue
                attempt_id = self._current_attempt_id(row)
                for verb in ("observe", "decide"):
                    calls.append(self.build_command(project, campaign_file,
                                                    row.run_id, verb,
                                                    attempt_id=attempt_id))
        return calls

    def run_cycle(self) -> list[PlannedCall]:
        """Execute one poll cycle: observe+decide per active run, then reindex."""
        calls = self.plan_cycle()
        if self.config.dry_run:
            return calls
        by_project = {p.project: p for p in self.projects}
        failed_observations: set[tuple[str, str]] = set()
        for call in calls:
            run_key = (call.project, call.run_id)
            if call.verb == "decide" and run_key in failed_observations:
                self.index.record_poll(
                    call.project,
                    call.run_id,
                    call.verb,
                    "skipped because observe failed in this collector cycle",
                    outcome="skipped",
                )
                continue
            error: Optional[str] = None
            result = self.controller.execute(call, timeout=_SUBPROCESS_TIMEOUT)
            if result.get("returncode") != 0:
                error = str(result.get("stderr") or result.get("stdout") or "").strip()[
                    -500:
                ] or f"exit code {result.get('returncode')}"
            self.index.record_poll(call.project, call.run_id, call.verb, error)
            if call.verb == "observe" and error is not None:
                failed_observations.add(run_key)
        for project in by_project.values():
            index_project(self.index, project)
        self.last_cycle_at = time.time()
        self.index.set_meta("collector_last_cycle_at", str(self.last_cycle_at))
        return calls
