"""Small atomic JSON helpers shared by server-owned stores."""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class StorageError(RuntimeError):
    """A durable store exists but cannot be read without losing state."""


class TransitionConflict(StorageError):
    """A compare-and-set transition observed a different durable revision."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
        )
        encoded = json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True,
        ) + "\n"
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = None
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StorageError(f"durable JSON is unreadable: {path}") from exc


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Durably append one bounded event to an owner-private JSONL ledger."""

    path.parent.mkdir(parents=True, exist_ok=True)
    _normalize_jsonl_tail(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        encoded = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _normalize_jsonl_tail(path: Path) -> None:
    """Make a crash-truncated JSONL tail safe for the next append.

    A complete JSON object that merely lacks its final newline is finalized.
    An incomplete final object is truncated back to the preceding newline. The
    authoritative durable state can then replay its embedded transition.
    """

    try:
        descriptor = os.open(path, os.O_RDWR)
    except FileNotFoundError:
        return
    try:
        size = os.fstat(descriptor).st_size
        if size == 0:
            return
        chunks: list[bytes] = []
        remaining = size
        os.lseek(descriptor, 0, os.SEEK_SET)
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if data.endswith(b"\n"):
            return
        boundary = data.rfind(b"\n") + 1
        tail = data[boundary:]
        try:
            decoded = json.loads(tail.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            decoded = None
        if isinstance(decoded, dict):
            os.lseek(descriptor, 0, os.SEEK_END)
            os.write(descriptor, b"\n")
        elif decoded is not None:
            raise StorageError(
                f"durable transition journal tail is not a mapping: {path}"
            )
        else:
            os.ftruncate(descriptor, boundary)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _jsonl_mappings(path: Path) -> list[dict[str, Any]]:
    _normalize_jsonl_tail(path)
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise StorageError(f"durable transition journal is unreadable: {path}") from exc
    items: list[dict[str, Any]] = []
    transition_ids: set[str] = set()
    previous_revision = 0
    for line_number, line in enumerate(lines, start=1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StorageError(
                "durable transition journal contains an invalid complete "
                f"record: {path}:{line_number}"
            ) from exc
        if not isinstance(item, dict):
            raise StorageError(
                "durable transition journal record must be a mapping: "
                f"{path}:{line_number}"
            )
        if "transition_id" in item:
            transition_id = item.get("transition_id")
            revision = item.get("revision")
            if not isinstance(transition_id, str) or not transition_id:
                raise StorageError(
                    "durable transition identity is invalid: "
                    f"{path}:{line_number}"
                )
            if transition_id in transition_ids:
                raise StorageError(
                    "durable transition identity is duplicated: "
                    f"{path}:{line_number}"
                )
            expected_revision = previous_revision + 1
            if (
                not isinstance(revision, int)
                or isinstance(revision, bool)
                or revision != expected_revision
            ):
                raise StorageError(
                    "durable transition revision is not contiguous; "
                    f"expected {expected_revision}: {path}:{line_number}"
                )
            transition_ids.add(transition_id)
            previous_revision = revision
        items.append(item)
    return items


_DURABILITY_KEY = "_durability"


@dataclass(frozen=True)
class DurableSnapshot:
    value: dict[str, Any]
    revision: int
    last_transition: dict[str, Any] | None = None
    journal_pending: bool = False


class DurableJsonState:
    """Recoverable JSON state plus a compare-and-set transition journal.

    The JSON file is authoritative.  Each commit embeds the complete last
    transition before appending that transition to the audit JSONL.  A crash
    between those writes is repaired on the next locked read/commit, so state
    can never depend on a best-effort journal append.

    Callers must hold their store's cross-process lock around these methods.
    """

    def __init__(self, path: Path, journal_path: Path):
        self.path = path
        self.journal_path = journal_path

    def snapshot(self, default: dict[str, Any]) -> DurableSnapshot:
        raw = read_json(self.path, default)
        if not isinstance(raw, dict):
            raise StorageError(f"durable state must be a mapping: {self.path}")
        value = dict(raw)
        metadata = value.pop(_DURABILITY_KEY, None)
        if metadata is None:
            return DurableSnapshot(value=value, revision=0)
        if not isinstance(metadata, dict):
            raise StorageError(f"durability metadata is invalid: {self.path}")
        revision = metadata.get("revision")
        transition = metadata.get("last_transition")
        if not isinstance(revision, int) or revision < 1:
            raise StorageError(f"durability revision is invalid: {self.path}")
        if not isinstance(transition, dict):
            raise StorageError(f"last transition is invalid: {self.path}")
        transition_id = transition.get("transition_id")
        transition_revision = transition.get("revision")
        if not isinstance(transition_id, str) or not transition_id:
            raise StorageError(f"last transition identity is invalid: {self.path}")
        if (
            not isinstance(transition_revision, int)
            or isinstance(transition_revision, bool)
            or transition_revision != revision
        ):
            raise StorageError(
                f"last transition revision does not match durable state: {self.path}"
            )
        return DurableSnapshot(value=value, revision=revision, last_transition=transition)

    def repair_journal(self, snapshot: DurableSnapshot) -> None:
        """Append the authoritative last transition if a crash omitted it."""

        transition = snapshot.last_transition
        if transition is None:
            return
        transition_id = transition.get("transition_id")
        if not isinstance(transition_id, str) or not transition_id:
            raise StorageError(f"last transition identity is invalid: {self.path}")
        items = _jsonl_mappings(self.journal_path)
        last_transition = next(
            (
                item for item in reversed(items)
                if item.get("transition_id")
            ),
            None,
        )
        if last_transition is not None and last_transition.get(
            "transition_id"
        ) == transition_id:
            if last_transition != transition:
                raise StorageError(
                    "durable transition journal disagrees with authoritative "
                    f"state: {self.journal_path}"
                )
            return

        prior_revision = (
            last_transition.get("revision") if last_transition is not None else 0
        )
        if last_transition is None and snapshot.revision != 1:
            raise StorageError(
                f"durable transition journal history is missing: {self.journal_path}"
            )
        if prior_revision != snapshot.revision - 1:
            raise StorageError(
                "durable transition journal is not the immediate predecessor "
                f"of authoritative state: {self.journal_path}"
            )
        if last_transition is None or last_transition.get(
            "transition_id"
        ) != transition_id:
            try:
                append_jsonl(self.journal_path, transition)
            except OSError as exc:
                raise StorageError(
                    f"durable transition journal cannot be repaired: {self.journal_path}"
                ) from exc
            repaired = _jsonl_mappings(self.journal_path)
            repaired_transition = next(
                (
                    item for item in reversed(repaired)
                    if item.get("transition_id")
                ),
                None,
            )
            if repaired_transition != transition:
                raise StorageError(
                    f"durable transition journal repair could not be verified: "
                    f"{self.journal_path}"
                )

    def append_event(
        self, event: dict[str, Any], *, event_id: str | None = None,
    ) -> dict[str, Any]:
        """Append a non-state event after repairing any pending transition."""

        snapshot = self.snapshot({})
        self.repair_journal(snapshot)
        identity = event_id or uuid4().hex
        existing = _jsonl_mappings(self.journal_path)
        if any(item.get("journal_event_id") == identity for item in existing):
            return next(
                item for item in existing
                if item.get("journal_event_id") == identity
            )
        record = {
            **event,
            "journal_event_id": identity,
            "state_revision": snapshot.revision,
        }
        try:
            append_jsonl(self.journal_path, record)
        except OSError as exc:
            raise StorageError(
                f"durable transition journal cannot append an event: {self.journal_path}"
            ) from exc
        persisted = _jsonl_mappings(self.journal_path)
        if not persisted or persisted[-1].get("journal_event_id") != identity:
            raise StorageError(
                f"durable transition journal event cannot be verified: {self.journal_path}"
            )
        return record

    def commit(
        self,
        value: dict[str, Any],
        *,
        event: dict[str, Any],
        expected_revision: int | None = None,
    ) -> DurableSnapshot:
        current = self.snapshot({})
        self.repair_journal(current)
        if expected_revision is not None and current.revision != expected_revision:
            raise TransitionConflict(
                f"durable state changed; expected revision {expected_revision}, "
                f"found {current.revision}"
            )
        revision = current.revision + 1
        transition = {
            **event,
            "transition_id": uuid4().hex,
            "revision": revision,
        }
        raw = {
            **value,
            _DURABILITY_KEY: {
                "revision": revision,
                "last_transition": transition,
            },
        }
        try:
            atomic_json(self.path, raw)
        except OSError as exc:
            raise StorageError(
                f"durable state cannot be committed: {self.path}"
            ) from exc
        journal_pending = False
        try:
            append_jsonl(self.journal_path, transition)
            persisted = _jsonl_mappings(self.journal_path)
            if not persisted or persisted[-1].get("transition_id") != transition[
                "transition_id"
            ]:
                journal_pending = True
        except (OSError, StorageError):
            # The authoritative state embeds this complete transition.  A
            # later commit must repair it before advancing, so acknowledging
            # this commit cannot lose or reorder audit history.
            journal_pending = True
        return DurableSnapshot(
            value=dict(value), revision=revision, last_transition=transition,
            journal_pending=journal_pending,
        )


@contextmanager
def exclusive_file_lock(path: Path):
    """Serialize a workspace mutation across daemon processes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
