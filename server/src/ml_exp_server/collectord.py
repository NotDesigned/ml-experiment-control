"""Read-only polling collector.

Periodically invokes the project's experimentctl with OBSERVATION VERBS ONLY,
then rescans run directories into the index. Mutation verbs are rejected at
command-construction time — there is deliberately no collector path to
submit/cancel/stage/prepare.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

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
from .storage import atomic_text

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
    execution_campaign_root: Path | None = None


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
        """Resolve the immutable campaign that actually submitted the current Attempt.

        Daemon-owned control roots require an execution copy of an authored
        Campaign.  The copy can differ only in operational fields such as
        ``local_root`` while retaining the authored Campaign revision in the
        Run manifest.  Therefore both MATCHED and drifted Runs must prefer the
        exact VERIFIED Action copy that submitted the current Attempt.
        """
        if self.action_store is None:
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
            approved_revision = str(action.get("campaign_revision") or "")
            revision_matches = (
                approved_revision == origin_revision
                if approved_revision else origin_revision == f"campaign.{actual}"
            )
            if (
                action.get("execution_campaign_sha256") != f"sha256:{actual}"
                or not revision_matches
            ):
                continue
            candidates.append((str(action.get("created_at") or ""), campaign))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]
    last_cycle_at: Optional[float] = None

    def build_command(self, project: ResearchProject, campaign_file: Path,
                      run_id: str, verb: str, *,
                      attempt_id: str | None = None,
                      campaign_id: str | None = None) -> PlannedCall:
        if verb not in OBSERVATION_VERBS:
            raise ForbiddenVerbError(
                f"verb {verb!r} is not an observation verb; collector refuses "
                f"anything outside {sorted(OBSERVATION_VERBS)}"
            )
        extra = ["--campaign-id", campaign_id] if campaign_id else []
        return self.controller.build(
            project, campaign_file, verb, run_id, attempt_id=attempt_id,
            extra=extra,
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

    @staticmethod
    def _campaign_payload(path: Path) -> dict[str, Any] | None:
        try:
            source = path.read_text(encoding="utf-8")
            payload = (
                json.loads(source) if path.suffix == ".json"
                else yaml.safe_load(source)
            )
        except (
            OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError,
        ):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _controller_workdir(project: ResearchProject) -> Path | None:
        if project.controller is None:
            return None
        base = (project.base_dir or Path(".")).resolve()
        workdir = Path(project.controller.workdir)
        return (workdir if workdir.is_absolute() else base / workdir).resolve()

    def _authored_run_dir(
        self, project: ResearchProject, payload: dict[str, Any], row: RunIndexRow,
    ) -> Path | None:
        workdir = self._controller_workdir(project)
        campaign = str(payload.get("campaign") or "")
        local_root = payload.get("local_root", "outputs/experiment_campaigns")
        if workdir is None or campaign != row.campaign or not local_root:
            return None
        root = Path(str(local_root))
        if not root.is_absolute():
            root = workdir / root
        return (root / campaign / row.run_id).resolve()

    @staticmethod
    def _contains_exact_run(payload: dict[str, Any], run_id: str) -> bool:
        runs = payload.get("runs")
        if not isinstance(runs, list):
            return False
        matches = [
            item for item in runs
            if isinstance(item, dict) and str(item.get("run_id") or "") == run_id
        ]
        return len(matches) == 1

    def _materialize_canonical_campaign(
        self,
        project: ResearchProject,
        row: RunIndexRow,
        authored: Path,
        payload: dict[str, Any],
    ) -> tuple[Path, str] | None:
        """Create a daemon-private execution copy for one canonical imported Run.

        The authored Campaign remains the scientific identity.  The copy changes
        only the operational ``local_root`` so observation verbs update the exact
        indexed Run instead of creating a second tree in the source checkout.
        """
        root = self.config.execution_campaign_root
        daemon_run_root = project.daemon_run_root
        binding = row.campaign_binding
        if root is None or daemon_run_root is None:
            return None
        try:
            canonical_root = daemon_run_root.resolve(strict=True)
            run_dir = Path(row.run_dir).resolve(strict=True)
            expected_run_dir = (
                canonical_root / str(row.campaign) / row.run_id
            ).resolve()
        except OSError:
            return None
        if run_dir != expected_run_dir:
            return None
        if (
            binding.relationship != CampaignRelationship.MATCHED
            or binding.origin_project != project.project
            or binding.origin_campaign != row.campaign
            or not self._contains_exact_run(payload, row.run_id)
            or payload.get("project") != project.project
        ):
            return None
        try:
            authored_sha = self._file_sha256(authored)
        except OSError:
            return None
        revision = f"campaign.{authored_sha}"
        if (
            binding.origin_revision != revision
            or binding.current_revision != revision
        ):
            return None

        execution_payload = dict(payload)
        execution_payload["local_root"] = str(canonical_root)
        try:
            encoded = yaml.safe_dump(
                execution_payload, allow_unicode=True, sort_keys=True,
            )
        except yaml.YAMLError:
            return None
        execution_sha = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        target = (
            Path(root).resolve()
            / project.project
            / str(row.campaign)
            / f"campaign.{execution_sha}.yml"
        )
        if target.exists():
            existing = self._campaign_payload(target)
            if existing != execution_payload:
                return None
        else:
            try:
                atomic_text(target, encoded)
            except OSError:
                return None
        return target, revision

    def _poll_campaign(
        self, project: ResearchProject, row: RunIndexRow, authored: Path,
    ) -> tuple[Path, str | None] | None:
        """Bind an authored Campaign to the indexed Run without path ambiguity."""
        # Embedded/test Projects without a daemon root retain the original v1
        # behavior. A live daemon always binds ``daemon_run_root`` in runtime.py.
        if project.daemon_run_root is None:
            return authored, None
        payload = self._campaign_payload(authored)
        if payload is None:
            return None
        try:
            indexed = Path(row.run_dir).resolve(strict=True)
        except OSError:
            return None
        authored_run = self._authored_run_dir(project, payload, row)
        if authored_run == indexed:
            return authored, None
        return self._materialize_canonical_campaign(
            project, row, authored, payload,
        )

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
                campaign_id: str | None = None
                if execution_campaign is not None:
                    campaign_file = execution_campaign
                    campaign_id = row.campaign_binding.origin_revision
                else:
                    authored = campaign_files.get(row.campaign or "")
                    if authored is None:
                        continue
                    resolved = self._poll_campaign(project, row, authored)
                    if resolved is None:
                        continue
                    campaign_file, campaign_id = resolved
                if campaign_file is None:
                    continue
                attempt_id = self._current_attempt_id(row)
                for verb in ("observe", "decide"):
                    calls.append(self.build_command(project, campaign_file,
                                                    row.run_id, verb,
                                                    attempt_id=attempt_id,
                                                    campaign_id=campaign_id))
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
