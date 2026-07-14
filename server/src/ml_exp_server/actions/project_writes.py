"""Crash-recoverable project-file effects for reviewed Actions.

Multiple filesystem paths cannot be replaced atomically as one operation.  A
durable write-ahead record therefore binds every target to its reviewed old and
new digest before the first replacement.  Recovery may only roll that exact
intent forward; an unrelated edit always fails closed.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..storage import atomic_json, read_json, utc_now
from .files import file_sha
from .store import ActionStore


class ProjectWriteError(RuntimeError):
    """A reviewed project write could not be completed safely."""

    def __init__(self, message: str, *, partial: bool):
        super().__init__(message)
        self.partial = partial


class ProjectWriteConflict(ProjectWriteError):
    """A target contains neither the reviewed old nor reviewed new content."""


@dataclass(frozen=True)
class PlannedWrite:
    path: Path
    expected_sha256: str | None
    proposed_content: str
    proposed_sha256: str

    def record(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "expected_sha256": self.expected_sha256,
            "proposed_sha256": self.proposed_sha256,
        }


def _content_sha(content: str) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


def _planned_writes(plan: dict[str, Any]) -> list[PlannedWrite]:
    raw_files = plan.get("files")
    if raw_files is None:
        raw_files = [{
            "path": plan["target_path"],
            "expected_sha256": plan.get("expected_sha256"),
            "content": plan["proposed_content"],
        }]
    writes: list[PlannedWrite] = []
    for item in raw_files:
        path = Path(str(item["path"]))
        content = str(item["content"])
        writes.append(PlannedWrite(
            path=path,
            expected_sha256=item.get("expected_sha256"),
            proposed_content=content,
            proposed_sha256=_content_sha(content),
        ))
    if not writes:
        raise ProjectWriteError("project write plan contains no files", partial=False)
    return writes


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_text(target: Path, content: str) -> None:
    """Replace one file durably while preserving an existing file mode."""

    target.parent.mkdir(parents=True, exist_ok=True)
    mode = 0o644
    try:
        mode = stat.S_IMODE(target.stat().st_mode)
    except FileNotFoundError:
        pass
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode,
        )
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = None
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


class ProjectWriteTransaction:
    """Persist and roll forward one immutable project-write Action."""

    SCHEMA_VERSION = 1

    def __init__(self, store: ActionStore):
        self.store = store

    def _path(self, action_id: str) -> Path:
        return self.store.directory(action_id) / "write_transaction.json"

    @staticmethod
    def _identity(plan: dict[str, Any], writes: list[PlannedWrite]) -> dict[str, Any]:
        return {
            "schema_version": ProjectWriteTransaction.SCHEMA_VERSION,
            "action_id": str(plan["action_id"]),
            "intent_digest": str(plan["intent_digest"]),
            "operation": str(plan["operation"]),
            "files": [item.record() for item in writes],
        }

    def _prepare(
        self, plan: dict[str, Any], writes: list[PlannedWrite],
    ) -> dict[str, Any]:
        path = self._path(str(plan["action_id"]))
        identity = self._identity(plan, writes)
        existing = read_json(path, {})
        if existing:
            if any(existing.get(key) != value for key, value in identity.items()):
                raise ProjectWriteError(
                    "durable project-write transaction does not match Action intent",
                    partial=True,
                )
            return existing
        record = {
            **identity,
            "phase": "PREPARED",
            "prepared_at": utc_now(),
            "updated_at": utc_now(),
            "last_error": None,
        }
        atomic_json(path, record)
        return record

    def _update(
        self, action_id: str, record: dict[str, Any], *, phase: str,
        error: str | None = None,
    ) -> dict[str, Any]:
        updated = {
            **record,
            "phase": phase,
            "updated_at": utc_now(),
            "last_error": error,
        }
        atomic_json(self._path(action_id), updated)
        return updated

    def apply(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Apply or recover the reviewed files, never an unreviewed variant."""

        action_id = str(plan["action_id"])
        writes = _planned_writes(plan)
        with self.store.locked():
            record = self._prepare(plan, writes)
            states = [(item, file_sha(item.path)) for item in writes]
            already_applied = [
                item for item, current in states if current == item.proposed_sha256
            ]
            conflicts = [
                (item, current) for item, current in states
                if current not in {item.expected_sha256, item.proposed_sha256}
            ]
            if conflicts:
                detail = ", ".join(
                    f"{item.path}={current or 'missing'}"
                    for item, current in conflicts
                )
                message = "project write target changed outside reviewed intent: " + detail
                self._update(action_id, record, phase="CONFLICT", error=message)
                raise ProjectWriteConflict(message, partial=bool(already_applied))

            record = self._update(action_id, record, phase="APPLYING")
            try:
                for item, current in states:
                    if current == item.proposed_sha256:
                        continue
                    _atomic_write_text(item.path, item.proposed_content)
                    if file_sha(item.path) != item.proposed_sha256:
                        raise OSError(f"durable replacement verification failed: {item.path}")
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                self._update(action_id, record, phase="APPLYING", error=message)
                # A failed fsync may occur after replace, so the effect is
                # conservatively treated as uncertain even on the first file.
                raise ProjectWriteError(message, partial=True) from exc

            result = {
                "files": [
                    {"path": str(item.path), "sha256": item.proposed_sha256}
                    for item in writes
                ],
            }
            self._update(action_id, record, phase="APPLIED")
            return result
