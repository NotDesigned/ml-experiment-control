"""Read-only polling collector.

Periodically invokes the project's experimentctl with OBSERVATION VERBS ONLY,
then rescans run directories into the index. Mutation verbs are rejected at
command-construction time — there is deliberately no collector path to
submit/cancel/stage/prepare.
"""

from __future__ import annotations

import errno
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .campaign_lifecycle import campaign_snapshot
from .controller_gateway import ControllerCall, ProjectControllerGateway
from .ingest.indexer import RunIndex, index_project
from .schemas import CampaignRelationship, ResearchProject, TERMINAL_RUN_STATES

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
    last_cycle_at: Optional[float] = None

    def build_command(self, project: ResearchProject, campaign_file: Path,
                      run_id: str, verb: str) -> PlannedCall:
        if verb not in OBSERVATION_VERBS:
            raise ForbiddenVerbError(
                f"verb {verb!r} is not an observation verb; collector refuses "
                f"anything outside {sorted(OBSERVATION_VERBS)}"
            )
        return self.controller.build(project, campaign_file, verb, run_id)

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
                if row.campaign_binding.relationship in _UNSAFE_CAMPAIGN_RELATIONSHIPS:
                    continue
                if (row.campaign or "") in inactive_campaigns:
                    continue
                campaign_file = campaign_files.get(row.campaign or "")
                if campaign_file is None:
                    continue
                for verb in ("observe", "decide"):
                    calls.append(self.build_command(project, campaign_file,
                                                    row.run_id, verb))
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
