"""Collector integration for sanitized archives and independent W&B mirrors."""

from __future__ import annotations

import re
import socket
from pathlib import Path
from typing import Iterable

from .observability_archive import (
    ArchiveSource,
    ObservabilityArchive,
    SourceCursor as ArchiveCursor,
)
from .observability_store import (
    ArchiveRejection,
    AttemptRef,
    ObservabilityStore,
    OutboxRecord,
    SourceRef,
)
from .schemas import LocalWandbConfig, RunIndexRow, WandbCloudConfig
from .wandb_publisher import (
    AttemptIdentity,
    PublicationItem,
    SubprocessWandbAdapter,
    TargetConfig,
    TargetKind,
    WandbPublisher,
)


_LOG_NAMES = re.compile(r"(?:^|[._-])(stdout|stderr|slurm|job)(?:[._-]|$)|\.(?:log|out|err)$", re.I)


def _project_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return (safe or "ml-expd")[:128]


class ObservabilityCoordinator:
    """Archive canonical files and drain durable publisher outboxes."""

    def __init__(
        self,
        *,
        workspace_id: str,
        archive_root: Path,
        store: ObservabilityStore,
        local: LocalWandbConfig,
        cloud: WandbCloudConfig,
        credential_provider,
    ) -> None:
        self.workspace_id = workspace_id
        self.archive = ObservabilityArchive(archive_root)
        self.store = store
        self.local = local
        self.cloud = cloud
        self._credential_provider = credential_provider
        self.publisher = WandbPublisher(
            SubprocessWandbAdapter(timeout_seconds=20.0),
            credential_provider=credential_provider,
        )
        self.worker_id = f"{socket.gethostname()}-{id(self):x}"

    def enable_cloud(self, project: str, run_id: str, attempt_id: str) -> None:
        attempt = AttemptRef(self.workspace_id, project, run_id, attempt_id)
        self.store.activate_target_and_rewind(
            attempt, TargetKind.CLOUD.value,
            records=self._archived_outbox_records(attempt),
        )

    def backfill(
        self, target: str, attempts: Iterable[AttemptRef],
    ) -> dict[str, object]:
        try:
            kind = TargetKind(target)
        except ValueError as exc:
            raise ValueError("unsupported observability target") from exc
        if not self._target_enabled(kind):
            raise RuntimeError("PublisherUnavailable")
        unique: dict[tuple[str, str, str, str], AttemptRef] = {}
        for attempt in attempts:
            if attempt.workspace_id != self.workspace_id:
                raise ValueError("Attempt workspace does not match coordinator")
            unique[attempt.values()] = attempt
        if not unique or len(unique) > 500:
            raise ValueError("backfill requires between 1 and 500 Attempts")
        for attempt in unique.values():
            self.store.backfill_target(
                attempt, kind.value, records=self._archived_outbox_records(attempt),
            )
        return {
            "target": kind.value,
            "attempt_count": len(unique),
            "rewound_attempts": len(unique),
        }

    def _archived_outbox_records(self, attempt: AttemptRef) -> tuple[OutboxRecord, ...]:
        archived = self.archive.load(self.store.source_keys(attempt))
        return tuple(OutboxRecord(
            record_key=item.idempotency_key,
            kind=item.kind,
            payload=item.payload,
            observed_at=_observed_at(item.payload),
        ) for item in archived)

    def collect_rows(self, rows: Iterable[RunIndexRow]) -> None:
        """Archive complete new records; malformed records never stop collection."""
        for row in rows:
            if not row.run_dir:
                continue
            for source in self._sources(row):
                self._collect_source(source)

    def publish_once(self, *, limit_per_target: int = 50) -> None:
        if limit_per_target < 1:
            raise ValueError("limit_per_target must be positive")
        for target in (TargetKind.LOCAL, TargetKind.CLOUD):
            if not self._target_enabled(target):
                continue
            self.store.revive_terminal(target.value)
            # The store leases only the earliest undelivered item per Attempt,
            # preserving strict W&B step order while allowing other Attempts
            # to make progress independently.
            items = self.store.claim(
                target.value, self.worker_id, limit=limit_per_target,
            )
            for item in items:
                self.store.set_target_state(item.attempt, target.value, "SYNCING")
                config = self._target_config(target, item.attempt.project)
                if config is None:
                    self.store.retry(
                        item.id, self.worker_id, "PublisherUnavailable",
                    )
                    self.store.set_target_state(
                        item.attempt, target.value, "DEGRADED",
                        error="PublisherUnavailable",
                    )
                    continue
                identity = AttemptIdentity(*item.attempt.values())
                kind = {"metrics": "metric", "events": "event", "log": "log"}.get(
                    item.kind, item.kind,
                )
                try:
                    publication = PublicationItem(
                        target=target,
                        record_key=item.record_key,
                        sequence=item.id,
                        kind=kind,
                        payload=item.payload,
                        timestamp=item.observed_at,
                    )
                    result = self.publisher.publish(config, identity, publication)
                except Exception as exc:
                    terminal = self.store.retry(
                        item.id, self.worker_id, type(exc).__name__,
                    )
                    self.store.set_target_state(
                        item.attempt, target.value,
                        "FAILED" if terminal else "DEGRADED",
                        error=type(exc).__name__,
                    )
                    continue
                if result.acknowledged:
                    self.store.acknowledge(item.id, self.worker_id)
                    target_status = next((
                        status for status in self.store.statuses(
                            attempt=item.attempt, limit=10,
                        ) if status.target == target.value
                    ), None)
                    remaining = target_status.pending if target_status else 0
                    terminal = target_status.terminal if target_status else 0
                    self.store.set_target_state(
                        item.attempt, target.value,
                        "FAILED" if terminal else "PENDING" if remaining else "READY",
                        dashboard_url=result.dashboard_url,
                    )
                else:
                    terminal = self.store.retry(
                        item.id, self.worker_id,
                        result.error_class or "PublisherError",
                    )
                    self.store.set_target_state(
                        item.attempt, target.value,
                        "FAILED" if terminal else "DEGRADED",
                        error=result.error_class,
                    )

    def _collect_source(self, source: ArchiveSource) -> None:
        attempt = AttemptRef(
            source.workspace_id, source.project, source.run_id, source.attempt_id,
        )
        reference = SourceRef(attempt, source.source_id)
        stored = self.store.get_cursor(reference)
        cursor = None
        if stored is not None and stored.file_identity and stored.anchor_digest:
            cursor = ArchiveCursor(
                source_id=source.source_id,
                generation=stored.generation,
                file_identity=stored.file_identity,
                offset=stored.byte_offset,
                anchor_start=stored.anchor_start,
                anchor_digest=stored.anchor_digest,
            )
        try:
            batch = self.archive.scan(source, cursor)
            self.archive.persist(batch.records)
        except (OSError, ValueError, RuntimeError) as exc:
            # The collector loop remains healthy. The next pass retries the
            # exact source because no durable cursor has advanced.
            self.store.record_archive_error(reference, type(exc).__name__)
            return
        targets = []
        if self._target_requested(TargetKind.LOCAL):
            targets.append(TargetKind.LOCAL.value)
        statuses = self.store.statuses(attempt=attempt, limit=10)
        if any(item.target == TargetKind.CLOUD.value for item in statuses):
            targets.append(TargetKind.CLOUD.value)
        records = [OutboxRecord(
            record_key=item.idempotency_key,
            kind=item.kind,
            payload=item.payload,
            observed_at=_observed_at(item.payload),
        ) for item in batch.records]
        self.store.enqueue_and_advance(
            reference,
            expected=stored,
            generation=batch.cursor.generation,
            byte_offset=batch.cursor.offset,
            file_identity=batch.cursor.file_identity,
            anchor_start=batch.cursor.anchor_start,
            anchor_digest=batch.cursor.anchor_digest,
            records=records,
            targets=targets,
            rejections=[ArchiveRejection(
                generation=batch.cursor.generation,
                byte_start=item.byte_start,
                byte_end=item.byte_end,
                reason=item.reason,
            ) for item in batch.issues],
        )

    def _target_enabled(self, kind: TargetKind) -> bool:
        if not self._target_requested(kind):
            return False
        if kind is TargetKind.LOCAL:
            reference = self.local.publisher_credential_ref
        else:
            reference = self.cloud.default_credential_ref
        if reference is None:
            return False
        try:
            return bool(self._credential_provider(reference))
        except Exception:
            return False

    def _target_requested(self, kind: TargetKind) -> bool:
        if kind is TargetKind.LOCAL:
            return bool(
                self.local.enabled and self.local.publisher_entity
                and self.local.publisher_credential_ref
            )
        return bool(
            self.cloud.enabled and self.cloud.default_credential_ref
            and self.cloud.entity
        )

    def _target_config(self, kind: TargetKind, project: str) -> TargetConfig | None:
        if kind is TargetKind.LOCAL:
            if (
                not self.local.enabled or not self.local.publisher_entity
                or not self.local.publisher_credential_ref
            ):
                return None
            return TargetConfig(
                kind=kind,
                api_url=self.local.url(),
                dashboard_url=self.local.url(),
                entity=self.local.publisher_entity,
                project=_project_name(project),
                working_dir=self.local.data_path() / "publisher",
                credential_ref=self.local.publisher_credential_ref,
            )
        if (
            not self.cloud.enabled
            or not self.cloud.default_credential_ref
            or not self.cloud.entity
        ):
            return None
        return TargetConfig(
            kind=kind,
            api_url=self.cloud.api_url,
            dashboard_url=self.cloud.dashboard_url,
            entity=self.cloud.entity,
            project=_project_name(project),
            working_dir=self.archive.archive_root.parent / "wandb-cloud",
            credential_ref=self.cloud.default_credential_ref,
        )

    def _sources(self, row: RunIndexRow) -> list[ArchiveSource]:
        root = Path(row.run_dir).expanduser()
        if not root.is_dir() or root.is_symlink():
            return []
        attempt_ids = [item.attempt_id for item in row.attempts]
        for layer in (row.evidence.scheduler, row.evidence.model, row.evidence.evaluation):
            if layer.attempt_id and layer.attempt_id not in attempt_ids:
                attempt_ids.append(layer.attempt_id)
        if not attempt_ids:
            attempt_ids = ["attempt-001"]
        sources: list[ArchiveSource] = []
        seen: set[tuple[str, Path]] = set()

        def add(attempt_id: str, path: Path, kind: str) -> None:
            try:
                relative = path.relative_to(root)
            except ValueError:
                return
            key = (attempt_id, path)
            if key in seen or not path.is_file() or path.is_symlink():
                return
            seen.add(key)
            sources.append(ArchiveSource(
                workspace_id=self.workspace_id,
                project=row.project,
                run_id=row.run_id,
                attempt_id=attempt_id,
                name=str(relative),
                path=path,
                kind=kind,  # type: ignore[arg-type]
            ))

        for attempt_id in attempt_ids:
            attempt_root = root / "attempts" / attempt_id
            if not attempt_root.is_dir() or attempt_root.is_symlink():
                continue
            for path in attempt_root.rglob("*"):
                if not path.is_file():
                    continue
                if path.name in {"train_metrics.jsonl", "metrics.jsonl"}:
                    add(attempt_id, path, "metrics")
                elif path.name == "events.jsonl":
                    add(attempt_id, path, "events")
                elif _LOG_NAMES.search(path.name):
                    add(attempt_id, path, "log")
        selected = attempt_ids[-1]
        for base in (root, root / "collected_run"):
            if not base.is_dir():
                continue
            for path in base.rglob("*"):
                if not path.is_file() or "attempts" in path.relative_to(root).parts:
                    continue
                if path.name in {"train_metrics.jsonl", "metrics.jsonl"}:
                    add(selected, path, "metrics")
                elif path.name == "events.jsonl" and len(attempt_ids) == 1:
                    add(selected, path, "events")
                elif _LOG_NAMES.search(path.name):
                    add(selected, path, "log")
        return sources


def _observed_at(payload) -> float | None:
    for key in ("timestamp", "ts", "time"):
        value = payload.get(key) if hasattr(payload, "get") else None
        if isinstance(value, (int, float)):
            return float(value)
    return None
