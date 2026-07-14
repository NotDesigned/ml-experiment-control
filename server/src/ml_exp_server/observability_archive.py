"""Incremental, sanitized archival of canonical Run/Attempt observability files.

The module deliberately does not own cursor persistence.  Callers load a
``SourceCursor`` from their durable store, call :meth:`ObservabilityArchive.scan`,
persist the returned records, and atomically enqueue those records together
with ``batch.cursor``.  No raw source bytes are written below ``archive_root``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


SourceKind = Literal["log", "metrics", "events"]
IssueReason = Literal[
    "invalid_json", "invalid_payload", "secret_key", "nonfinite", "oversized_record",
]

_SECRET_KEY = re.compile(
    r"(?i)(?:^|[_-])(?:secret|token|password|passwd|credential|authorization|"
    r"cookie|api[_-]?key|access[_-]?key|private[_-]?key|client[_-]?secret|proxy)"
    r"(?:$|[_-])"
)
_ASSIGNMENT = re.compile(
    r"(?i)\b((?:wandb[_-]?)?(?:api[_-]?key|access[_-]?key|secret|token|"
    r"password|passwd|credential|authorization|cookie))\s*([=:])\s*"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_ENV_ASSIGNMENT = re.compile(
    r"(?i)\b([A-Z][A-Z0-9_]*(?:SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|"
    r"AUTHORIZATION|COOKIE|API_KEY|ACCESS_KEY|PRIVATE_KEY|CLIENT_SECRET)"
    r"[A-Z0-9_]*)\s*([=:])\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_BEARER = re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+\-/]+=*")
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?"
    r"-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)
_URL = re.compile(r"https?://[^\s<>'\"]+")
_ANCHOR_BYTES = 128
_DISCARD_MARKER = "discard:"


def _secret_key_matches(key: Any) -> bool:
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key)).lower()
    return _SECRET_KEY.search(normalized) is not None


class CursorStore(Protocol):
    """Minimal store boundary needed by a collector integration."""

    def source_cursor(self, source_id: str) -> "SourceCursor | None": ...

    def commit_archive_batch(self, batch: "ArchiveBatch") -> None: ...


@dataclass(frozen=True)
class ArchiveSource:
    """One canonical file selected by the caller's Run/Attempt resolver."""

    workspace_id: str
    project: str
    run_id: str
    attempt_id: str
    name: str
    path: Path
    kind: SourceKind

    def __post_init__(self) -> None:
        if self.kind not in {"log", "metrics", "events"}:
            raise ValueError("unsupported observability source kind")
        for value in (
            self.workspace_id, self.project, self.run_id, self.attempt_id, self.name,
        ):
            if not value or "\x00" in value:
                raise ValueError("source identity fields must be non-empty")

    @property
    def source_id(self) -> str:
        identity = "\x00".join(
            (self.workspace_id, self.project, self.run_id, self.attempt_id,
             self.kind, self.name)
        )
        return hashlib.sha256(identity.encode()).hexdigest()


@dataclass(frozen=True)
class SourceCursor:
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
class ArchiveRecord:
    """A sanitized record suitable for durable archive and outbox insertion."""

    idempotency_key: str
    source_id: str
    generation: str
    kind: SourceKind
    name: str
    byte_start: int
    byte_end: int
    payload_digest: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class ArchiveIssue:
    byte_start: int
    byte_end: int
    reason: IssueReason


@dataclass(frozen=True)
class ArchiveBatch:
    source: ArchiveSource
    cursor: SourceCursor
    records: tuple[ArchiveRecord, ...]
    issues: tuple[ArchiveIssue, ...]
    generation_changed: bool


class ObservabilityArchive:
    """Scan complete lines and persist only sanitized, idempotent record files."""

    def __init__(self, archive_root: Path, *, max_read_bytes: int = 4 * 1024 * 1024):
        if max_read_bytes < 1:
            raise ValueError("max_read_bytes must be positive")
        root = archive_root.expanduser().absolute()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("archive_root must be a real directory")
        self.archive_root = root.resolve(strict=True)
        try:
            os.chmod(self.archive_root, 0o700)
        except OSError:
            pass
        self.max_read_bytes = max_read_bytes

    def scan(
        self, source: ArchiveSource, cursor: SourceCursor | None = None,
    ) -> ArchiveBatch:
        """Return newly completed records and the cursor to commit with them."""
        if cursor is not None and cursor.source_id != source.source_id:
            raise ValueError("cursor does not belong to source")
        path = source.path.expanduser()
        if path.is_symlink():
            raise ValueError("observability sources must not be symbolic links")
        with path.open("rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("observability source must be a regular file")
            file_identity = f"{info.st_dev}:{info.st_ino}"
            offset, generation, changed = self._start_position(
                handle, source, cursor, file_identity, info.st_size,
            )
            handle.seek(offset)
            data = handle.read(self.max_read_bytes)
            discarding = bool(
                cursor is not None
                and cursor.anchor_digest.startswith(_DISCARD_MARKER)
                and not changed
            )
            issues: list[ArchiveIssue] = []
            record_start = offset
            if discarding:
                newline = data.find(b"\n")
                if newline < 0:
                    new_offset = offset + len(data)
                    complete = b""
                    still_discarding = True
                    if data:
                        issues.append(ArchiveIssue(offset, new_offset, "oversized_record"))
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
                if complete_length == 0 and len(data) == self.max_read_bytes:
                    # Drop the entire logical record, including future chunks,
                    # until its terminating newline arrives.
                    new_offset = offset + len(data)
                    complete = b""
                    still_discarding = True
                    issues.append(ArchiveIssue(offset, new_offset, "oversized_record"))
            records, record_issues = self._records(
                source, generation, record_start, complete,
            )
            issues.extend(record_issues)
            anchor_start = max(0, new_offset - _ANCHOR_BYTES)
            handle.seek(anchor_start)
            anchor = handle.read(new_offset - anchor_start)

        new_cursor = SourceCursor(
            source_id=source.source_id,
            generation=generation,
            file_identity=file_identity,
            offset=new_offset,
            anchor_start=anchor_start,
            anchor_digest=(
                _DISCARD_MARKER if still_discarding else ""
            ) + _digest(anchor),
        )
        return ArchiveBatch(
            source=source, cursor=new_cursor, records=tuple(records),
            issues=tuple(issues), generation_changed=changed,
        )

    def persist(self, records: Sequence[ArchiveRecord]) -> tuple[Path, ...]:
        """Persist sanitized records once, using content-addressed filenames.

        Existing identical records are harmless after a crash/replay.  A key
        collision with different content fails closed.
        """
        paths: list[Path] = []
        for record in records:
            _validate_record(record)
            if not re.fullmatch(r"[0-9a-f]{64}", record.idempotency_key):
                raise ValueError("invalid archive record key")
            directory = self._safe_child(record.source_id[:2], record.source_id)
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            _fsync_directory(directory.parent)
            _fsync_directory(directory)
            path = self._safe_child(
                record.source_id[:2], record.source_id,
                f"{record.idempotency_key}.json",
            )
            if path.is_symlink():
                raise ValueError("archive record path must not be a symbolic link")
            encoded = _canonical_json(asdict(record)) + b"\n"
            try:
                descriptor = os.open(
                    path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
                )
            except FileExistsError:
                if path.read_bytes() != encoded:
                    raise RuntimeError("archive idempotency collision")
            else:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                _fsync_directory(directory)
            paths.append(path)
        return tuple(paths)

    def _safe_child(self, *parts: str) -> Path:
        if any(not re.fullmatch(r"[A-Za-z0-9._-]+", part) for part in parts):
            raise ValueError("unsafe archive path component")
        candidate = self.archive_root.joinpath(*parts)
        if (
            not candidate.is_relative_to(self.archive_root)
            or not candidate.resolve(strict=False).is_relative_to(self.archive_root)
        ):
            raise ValueError("archive path escapes archive_root")
        return candidate

    def _start_position(
        self, handle: Any, source: ArchiveSource, cursor: SourceCursor | None,
        file_identity: str, size: int,
    ) -> tuple[int, str, bool]:
        if cursor is None:
            generation = _digest(
                f"initial\x00{source.source_id}\x00{file_identity}".encode()
            )
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
            if _digest(anchor) != expected_anchor:
                replacement_reason = "rewrite"
        if replacement_reason is None:
            return cursor.offset, cursor.generation, False

        handle.seek(0)
        prefix_digest = _digest(handle.read(min(size, 4096)))
        generation = _digest(
            (f"{cursor.generation}\x00{replacement_reason}\x00{file_identity}\x00"
             f"{size}\x00{prefix_digest}").encode()
        )
        return 0, generation, True

    def _records(
        self, source: ArchiveSource, generation: str, base: int, data: bytes,
    ) -> tuple[list[ArchiveRecord], list[ArchiveIssue]]:
        records: list[ArchiveRecord] = []
        issues: list[ArchiveIssue] = []
        position = base
        for raw_line in data.splitlines(keepends=True):
            end = position + len(raw_line)
            content = raw_line[:-1] if raw_line.endswith(b"\n") else raw_line
            if content.endswith(b"\r"):
                content = content[:-1]
            if source.kind == "log":
                payload: Mapping[str, Any] = {
                    "stream": source.name,
                    "text": sanitize_log_text(content.decode("utf-8", errors="replace")),
                }
            else:
                payload, reason = _structured_payload(content)
                if reason is not None:
                    issues.append(ArchiveIssue(position, end, reason))
                    position = end
                    continue
                assert payload is not None
            payload_bytes = _canonical_json(payload)
            payload_digest = _digest(payload_bytes)
            key_material = (
                f"{source.source_id}\x00{generation}\x00{position}\x00{end}\x00"
                f"{payload_digest}"
            ).encode()
            records.append(ArchiveRecord(
                idempotency_key=_digest(key_material),
                source_id=source.source_id,
                generation=generation,
                kind=source.kind,
                name=source.name,
                byte_start=position,
                byte_end=end,
                payload_digest=payload_digest,
                payload=payload,
            ))
            position = end
        return records, issues


def sanitize_log_text(text: str) -> str:
    """Remove common credential and URL secret shapes before persistence."""
    text = _PRIVATE_KEY.sub("[REDACTED PRIVATE KEY]", text)
    text = _BEARER.sub(r"\1 [REDACTED]", text)
    text = _ENV_ASSIGNMENT.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", text,
    )
    text = _ASSIGNMENT.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", text,
    )
    return _URL.sub(lambda match: _sanitize_url(match.group(0)), text)


def _sanitize_url(value: str) -> str:
    # Preserve trailing punctuation outside the parsed URL.
    suffix = ""
    while value and value[-1] in ".,;)":
        suffix = value[-1] + suffix
        value = value[:-1]
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        netloc = host
        if parsed.port is not None:
            netloc += f":{parsed.port}"
        if parsed.username is not None or parsed.password is not None:
            # Brackets are invalid in URL userinfo and would make a second
            # sanitizer pass misparse this as an IPv6 host.
            netloc = f"redacted@{netloc}"
        query = urlencode([
            (key, "[REDACTED]" if _secret_key_matches(key) else item)
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        ])
        return urlunsplit((parsed.scheme, netloc, parsed.path, query, "")) + suffix
    except (ValueError, UnicodeError):
        return "[REDACTED URL]" + suffix


def _structured_payload(
    content: bytes,
) -> tuple[Mapping[str, Any] | None, IssueReason | None]:
    try:
        value = json.loads(
            content.decode("utf-8"),
            # Python's decoder accepts these JavaScript extensions.  Preserve
            # them just long enough for the explicit finite-value check below
            # to classify and reject them as non-finite payloads.
            parse_constant=lambda _value: float("nan"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None, "invalid_json"
    if not isinstance(value, dict):
        return None, "invalid_payload"
    if _has_secret_key(value):
        return None, "secret_key"
    if not _finite(value):
        return None, "nonfinite"
    return _sanitize_structured(value), None


def _has_secret_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            _secret_key_matches(key) or _has_secret_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_has_secret_key(item) for item in value)
    return False


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _sanitize_structured(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_structured(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_structured(item) for item in value]
    if isinstance(value, str):
        return sanitize_log_text(value)
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _validate_record(record: ArchiveRecord) -> None:
    payload = record.payload
    if record.kind == "log":
        text = payload.get("text") if isinstance(payload, Mapping) else None
        if not isinstance(text, str) or sanitize_log_text(text) != text:
            raise ValueError("unsanitized log archive record")
    elif _has_secret_key(payload):
        raise ValueError("secret-like key in archive record")
    if not _finite(payload):
        raise ValueError("non-finite archive record")
    payload_digest = _digest(_canonical_json(payload))
    if payload_digest != record.payload_digest:
        raise ValueError("archive payload digest mismatch")
    expected_key = _digest(
        (f"{record.source_id}\x00{record.generation}\x00{record.byte_start}\x00"
         f"{record.byte_end}\x00{payload_digest}").encode()
    )
    if expected_key != record.idempotency_key:
        raise ValueError("archive idempotency key mismatch")


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
