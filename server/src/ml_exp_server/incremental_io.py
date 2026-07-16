"""Safe incremental append-file cursor primitive.

Extracted from ``observability_archive`` so the sanitizing archive scanner
and any other bounded, append-only ingest can share one hardened algorithm:
open only real, non-symlinked regular files, detect truncation, replacement,
and in-place rewrite through a trailing anchor digest, and return newly
completed line bytes without ever re-emitting (or losing) a partial trailing
line. This module owns no domain semantics (payload parsing, redaction,
persistence) -- only the byte-cursor mechanics.
"""

from __future__ import annotations

import errno
import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ANCHOR_BYTES = 128
_DISCARD_MARKER = "discard:"


@dataclass(frozen=True)
class SourceCursor:
    """Durable position for one append-only source."""

    source_id: str
    generation: str
    file_identity: str
    offset: int
    anchor_start: int
    anchor_digest: str

    def __post_init__(self) -> None:
        if self.offset < 0 or self.anchor_start < 0 or self.anchor_start > self.offset:
            raise ValueError("invalid source cursor offsets")


@dataclass(frozen=True)
class IncrementalIssue:
    """A read-layer anomaly, independent of any record-level parsing."""

    byte_start: int
    byte_end: int
    reason: str


@dataclass(frozen=True)
class IncrementalRead:
    """One bounded increment of newly completed bytes for a source."""

    cursor: SourceCursor
    generation_changed: bool
    record_start: int
    data: bytes
    issues: tuple[IncrementalIssue, ...]


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def open_regular_nofollow(path: Path) -> tuple[int, os.stat_result]:
    """Open ``path`` for reading, refusing symlinks and non-regular files."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(
                "observability sources must not be symbolic links"
            ) from exc
        raise
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise ValueError("observability source must be a regular file")
    return descriptor, metadata


def scan_increment(
    source_id: str,
    path: Path,
    cursor: SourceCursor | None,
    *,
    max_read_bytes: int,
) -> IncrementalRead:
    """Advance ``cursor`` over ``path`` and return newly completed line bytes.

    The returned ``data`` never includes a trailing partial line; callers
    receive exactly the complete lines available in this bounded read,
    starting at ``record_start``. A file whose identity or trailing anchor
    bytes no longer match the supplied cursor starts a new generation from
    byte zero. An unterminated record larger than ``max_read_bytes`` is
    dropped (with an ``oversized_record`` issue) rather than buffered
    unbounded in memory.
    """

    if max_read_bytes < 1:
        raise ValueError("max_read_bytes must be positive")
    if cursor is not None and cursor.source_id != source_id:
        raise ValueError("cursor does not belong to source")
    descriptor, metadata = open_regular_nofollow(path.expanduser())
    with os.fdopen(descriptor, "rb") as handle:
        file_identity = f"{metadata.st_dev}:{metadata.st_ino}"
        offset, generation, changed = _start_position(
            handle, source_id, cursor, file_identity, metadata.st_size,
        )
        handle.seek(offset)
        data = handle.read(max_read_bytes)
        discarding = bool(
            cursor is not None
            and cursor.anchor_digest.startswith(_DISCARD_MARKER)
            and not changed
        )
        issues: list[IncrementalIssue] = []
        record_start = offset
        if discarding:
            newline = data.find(b"\n")
            if newline < 0:
                new_offset = offset + len(data)
                complete = b""
                still_discarding = True
                if data:
                    issues.append(IncrementalIssue(offset, new_offset, "oversized_record"))
            else:
                record_start = offset + newline + 1
                remaining = data[newline + 1:]
                complete_length = remaining.rfind(b"\n") + 1
                complete = remaining[:complete_length]
                new_offset = record_start + complete_length
                still_discarding = False
        else:
            complete_length = data.rfind(b"\n") + 1
            complete = data[:complete_length]
            new_offset = offset + complete_length
            still_discarding = False
            if complete_length == 0 and len(data) == max_read_bytes:
                # Drop the entire logical record, including future chunks,
                # until its terminating newline arrives.
                new_offset = offset + len(data)
                complete = b""
                still_discarding = True
                issues.append(IncrementalIssue(offset, new_offset, "oversized_record"))
        anchor_start = max(0, new_offset - _ANCHOR_BYTES)
        handle.seek(anchor_start)
        anchor = handle.read(new_offset - anchor_start)

    new_cursor = SourceCursor(
        source_id=source_id,
        generation=generation,
        file_identity=file_identity,
        offset=new_offset,
        anchor_start=anchor_start,
        anchor_digest=(_DISCARD_MARKER if still_discarding else "") + digest(anchor),
    )
    return IncrementalRead(
        cursor=new_cursor, generation_changed=changed,
        record_start=record_start, data=complete, issues=tuple(issues),
    )


def _start_position(
    handle: Any, source_id: str, cursor: SourceCursor | None,
    file_identity: str, size: int,
) -> tuple[int, str, bool]:
    if cursor is None:
        generation = digest(f"initial\x00{source_id}\x00{file_identity}".encode())
        return 0, generation, True

    replacement_reason: str | None = None
    if cursor.file_identity != file_identity:
        replacement_reason = "replace"
    elif size < cursor.offset:
        replacement_reason = "truncate"
    else:
        handle.seek(cursor.anchor_start)
        anchor = handle.read(cursor.offset - cursor.anchor_start)
        expected_anchor = cursor.anchor_digest.removeprefix(_DISCARD_MARKER)
        if digest(anchor) != expected_anchor:
            replacement_reason = "rewrite"
    if replacement_reason is None:
        return cursor.offset, cursor.generation, False

    handle.seek(0)
    prefix_digest = digest(handle.read(min(size, 4096)))
    generation = digest(
        (f"{cursor.generation}\x00{replacement_reason}\x00{file_identity}\x00"
         f"{size}\x00{prefix_digest}").encode()
    )
    return 0, generation, True
