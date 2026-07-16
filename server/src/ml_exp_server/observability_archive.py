"""Incremental, sanitized archival of canonical Run/Attempt observability files.

The module deliberately does not own cursor persistence.  Callers load a
``SourceCursor`` from their durable store, call :meth:`ObservabilityArchive.scan`,
persist the returned records, and atomically enqueue those records together
with ``batch.cursor``.  No raw source bytes are written below ``archive_root``.

Persistence is dual-read, new-write. Every new ``persist`` call writes only
the immutable v2 layout: ``<source>/segments/segment-<sha256>.jsonl`` holds
one canonical-JSON record per line and ``segment-<sha256>.idx`` is the
crash-safe commit marker recording each record's offsets and digests. The
legacy v1 layout, one ``<source>/<record-key>.json`` file per record, is
still read so archives written by an older daemon remain queryable, but it
is never written again.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .incremental_io import SourceCursor, digest as _digest, scan_increment


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
_KEY = re.compile(r"[0-9a-f]{64}")
_SEGMENT_NAME = re.compile(r"segment-([0-9a-f]{64})")


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
        return _digest(identity.encode())


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
        increment = scan_increment(
            source.source_id, source.path, cursor, max_read_bytes=self.max_read_bytes,
        )
        records, record_issues = self._records(
            source, increment.cursor.generation, increment.record_start, increment.data,
        )
        issues = tuple(
            ArchiveIssue(item.byte_start, item.byte_end, "oversized_record")
            for item in increment.issues
        ) + tuple(record_issues)
        return ArchiveBatch(
            source=source, cursor=increment.cursor, records=tuple(records),
            issues=issues, generation_changed=increment.generation_changed,
        )

    def persist(self, records: Sequence[ArchiveRecord]) -> tuple[Path, ...]:
        """Persist sanitized records once, using content-addressed v2 segments.

        Records are grouped by ``source_id``; each source gets at most one
        new immutable ``segments/segment-<sha256>.jsonl`` file per call,
        covering only the records not already durable in either the legacy
        v1 layout or an existing v2 segment. The returned tuple mirrors
        ``records`` positionally, so multiple records commonly resolve to
        the same segment path. A key collision against different full
        canonical record bytes -- in this batch, or already on disk --
        fails closed.
        """
        if not records:
            return ()
        order: list[ArchiveRecord] = []
        for record in records:
            if not _KEY.fullmatch(record.idempotency_key):
                raise ValueError("invalid archive record key")
            _validate_record(record)
            order.append(record)

        groups: dict[str, list[ArchiveRecord]] = {}
        for record in order:
            groups.setdefault(record.source_id, []).append(record)

        resolved: dict[str, Path] = {}
        for source_id, group in groups.items():
            resolved.update(self._persist_source_group(source_id, group))
        return tuple(resolved[record.idempotency_key] for record in order)

    def load(self, source_ids: Sequence[str]) -> tuple[ArchiveRecord, ...]:
        """Load previously sanitized records for an audited target replay.

        Both the legacy v1 layout and v2 segments are read; output is
        stably ordered by source id, then by record key.
        """
        records: list[ArchiveRecord] = []
        for source_id in sorted(set(source_ids)):
            if not _KEY.fullmatch(source_id):
                # Pre-archive and embedded integrations may use readable
                # cursor keys. They have no content-addressed archive path,
                # but their canonical source cursor should still be rewound.
                continue
            directory = self._safe_child(source_id[:2], source_id)
            if not directory.exists():
                continue
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("archive source directory must be a real directory")
            existing = self._existing_records(source_id, directory)
            for key in sorted(existing):
                records.append(existing[key][0])
        return tuple(records)

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

    def _persist_source_group(
        self, source_id: str, group: Sequence[ArchiveRecord],
    ) -> dict[str, Path]:
        directory = self._safe_child(source_id[:2], source_id)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        _fsync_directory(directory.parent)
        _fsync_directory(directory)

        encoded_by_key: dict[str, bytes] = {}
        new_order: list[tuple[str, bytes]] = []
        for record in group:
            encoded = _canonical_json(asdict(record)) + b"\n"
            existing = encoded_by_key.get(record.idempotency_key)
            if existing is None:
                encoded_by_key[record.idempotency_key] = encoded
                new_order.append((record.idempotency_key, encoded))
            elif existing != encoded:
                raise RuntimeError("archive idempotency collision")

        already = self._existing_records(source_id, directory)
        resolved: dict[str, Path] = {}
        pending: list[tuple[str, bytes]] = []
        for key, encoded in new_order:
            found = already.get(key)
            if found is None:
                pending.append((key, encoded))
                continue
            _record, found_encoded, found_path = found
            if found_encoded != encoded:
                raise RuntimeError("archive idempotency collision")
            resolved[key] = found_path

        if pending:
            segment_path = self._commit_v2_segment(source_id, directory, pending)
            for key, _encoded in pending:
                resolved[key] = segment_path
        return resolved

    def _commit_v2_segment(
        self, source_id: str, directory: Path, pending: Sequence[tuple[str, bytes]],
    ) -> Path:
        segments_dir = self._safe_child(source_id[:2], source_id, "segments")
        # ``_persist_source_group`` always calls ``_existing_records`` (which
        # runs the identical symlink/real-directory check via
        # ``_v2_committed_records``) for this exact source/directory before
        # ever reaching here, so a second check here would be unreachable.
        segments_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

        segment_bytes = b"".join(encoded for _key, encoded in pending)
        segment_digest = _digest(segment_bytes)
        segment_path = self._safe_child(
            source_id[:2], source_id, "segments", f"segment-{segment_digest}.jsonl",
        )
        index_path = self._safe_child(
            source_id[:2], source_id, "segments", f"segment-{segment_digest}.idx",
        )
        if segment_path.is_symlink() or index_path.is_symlink():
            raise ValueError("archive segment path must not be a symbolic link")

        if segment_path.exists():
            if not segment_path.is_file() or segment_path.read_bytes() != segment_bytes:
                raise RuntimeError("archive segment collision")
        else:
            _atomic_publish(segments_dir, segment_path, segment_bytes)

        index_bytes = _canonical_json(_v2_index_payload(
            source_id, segment_digest, len(segment_bytes), pending,
        )) + b"\n"
        # Unlike the segment above, an index is only ever discoverable
        # through ``_v2_committed_records``, which ``_existing_records``
        # already consulted for this exact source/directory: a pre-existing,
        # valid index at this content-addressed path means these keys would
        # already have resolved above and never reached ``pending``, and an
        # invalid one would already have raised. It cannot exist here.
        _atomic_publish(segments_dir, index_path, index_bytes)

        _fsync_directory(segments_dir)
        return segment_path

    def _existing_records(
        self, source_id: str, directory: Path,
    ) -> dict[str, tuple[ArchiveRecord, bytes, Path]]:
        """Merge legacy v1 files and committed v2 segments for one source.

        A key present in both formats (or across segments) must carry
        identical full canonical record bytes; any mismatch fails closed.
        """
        records: dict[str, tuple[ArchiveRecord, bytes, Path]] = {}
        for path in sorted(directory.glob("*.json")):
            record, raw = self._load_v1_record(source_id, path)
            records[record.idempotency_key] = (record, raw, path)
        for key, entry in self._v2_committed_records(source_id, directory).items():
            record, encoded, segment_path = entry
            if key in records and records[key][1] != encoded:
                raise RuntimeError("archive idempotency collision")
            records[key] = (record, encoded, segment_path)
        return records

    def _load_v1_record(self, source_id: str, path: Path) -> tuple[ArchiveRecord, bytes]:
        if path.is_symlink() or not path.is_file():
            raise ValueError("archive record path must be a regular file")
        raw = path.read_bytes()
        try:
            payload = json.loads(raw.decode("utf-8"))
            record = ArchiveRecord(**payload)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"invalid archived observability record: {path}") from exc
        _validate_record(record)
        if record.source_id != source_id or path.stem != record.idempotency_key:
            raise ValueError("archived observability record identity mismatch")
        return record, raw

    def _v2_committed_records(
        self, source_id: str, directory: Path,
    ) -> dict[str, tuple[ArchiveRecord, bytes, Path]]:
        segments_dir = self._safe_child(source_id[:2], source_id, "segments")
        if not segments_dir.exists():
            return {}
        if segments_dir.is_symlink() or not segments_dir.is_dir():
            raise ValueError("archive segments directory must be a real directory")

        records: dict[str, tuple[ArchiveRecord, bytes, Path]] = {}
        # Only ``.idx`` files are enumerated: the index is the commit
        # marker, so a segment written without one (a crash between the two
        # publishes) is correctly invisible here. A random leftover temp
        # file never matches this glob either.
        for index_path in sorted(segments_dir.glob("segment-*.idx")):
            match = _SEGMENT_NAME.fullmatch(index_path.stem)
            if index_path.is_symlink() or not index_path.is_file() or not match:
                raise ValueError(f"invalid archive segment index: {index_path}")
            segment_path = segments_dir / f"{index_path.stem}.jsonl"
            for key, (record, chunk) in self._load_v2_segment(
                source_id, match.group(1), segment_path, index_path,
            ).items():
                if key in records:
                    raise ValueError("archive record duplicated across segments")
                records[key] = (record, chunk, segment_path)
        return records

    def _load_v2_segment(
        self, source_id: str, expected_digest: str, segment_path: Path, index_path: Path,
    ) -> dict[str, tuple[ArchiveRecord, bytes]]:
        if segment_path.is_symlink() or not segment_path.is_file():
            raise ValueError(f"archive segment missing or invalid: {segment_path}")
        segment_bytes = segment_path.read_bytes()
        if _digest(segment_bytes) != expected_digest:
            raise ValueError(f"archive segment digest mismatch: {segment_path}")
        try:
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid archive segment index: {index_path}") from exc
        if (
            not isinstance(index_payload, dict)
            or index_payload.get("version") != 2
            or index_payload.get("source_id") != source_id
            or index_payload.get("segment_sha256") != expected_digest
            or index_payload.get("segment_size") != len(segment_bytes)
            or not isinstance(index_payload.get("records"), list)
        ):
            raise ValueError(f"invalid archive segment index: {index_path}")

        entries: dict[str, tuple[ArchiveRecord, bytes]] = {}
        position = 0
        for entry in index_payload["records"]:
            key, chunk = self._validate_index_entry(entry, segment_bytes, position, index_path)
            if not chunk.endswith(b"\n"):
                raise ValueError(f"invalid archive segment index offsets: {index_path}")
            try:
                payload = json.loads(chunk[:-1].decode("utf-8"))
                record = ArchiveRecord(**payload)
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
                raise ValueError(
                    f"invalid archived observability record: {index_path}"
                ) from exc
            _validate_record(record)
            if record.source_id != source_id or record.idempotency_key != key:
                raise ValueError("archived observability record identity mismatch")
            if key in entries:
                raise ValueError(f"duplicate record key within segment: {index_path}")
            entries[key] = (record, chunk)
            position += len(chunk)
        if position != len(segment_bytes):
            raise ValueError(f"archive segment index incomplete: {index_path}")
        return entries

    @staticmethod
    def _validate_index_entry(
        entry: Any, segment_bytes: bytes, position: int, index_path: Path,
    ) -> tuple[str, bytes]:
        if not isinstance(entry, dict):
            raise ValueError(f"invalid archive segment index: {index_path}")
        key = entry.get("idempotency_key")
        start = entry.get("byte_start")
        end = entry.get("byte_end")
        entry_digest = entry.get("sha256")
        if (
            not isinstance(key, str) or not _KEY.fullmatch(key)
            or not isinstance(start, int) or isinstance(start, bool)
            or not isinstance(end, int) or isinstance(end, bool)
            or start != position or end < start or end > len(segment_bytes)
        ):
            raise ValueError(f"invalid archive segment index offsets: {index_path}")
        chunk = segment_bytes[start:end]
        if _digest(chunk) != entry_digest:
            raise ValueError(f"archive record digest mismatch: {index_path}")
        return key, chunk

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


def _v2_index_payload(
    source_id: str, segment_digest: str, segment_size: int,
    pending: Sequence[tuple[str, bytes]],
) -> dict[str, Any]:
    entries = []
    position = 0
    for key, encoded in pending:
        end = position + len(encoded)
        entries.append({
            "idempotency_key": key, "byte_start": position, "byte_end": end,
            "sha256": _digest(encoded),
        })
        position = end
    return {
        "version": 2, "source_id": source_id, "segment_sha256": segment_digest,
        "segment_size": segment_size, "records": entries,
    }


def _atomic_publish(directory: Path, final_path: Path, data: bytes) -> None:
    """Durably write ``data`` then publish it at ``final_path``.

    The temp file lives in the same directory (same filesystem, so the
    publish step below is a metadata-only operation), is opened
    ``O_EXCL``/``O_NOFOLLOW``, and is fsynced before publication. Publication
    uses a hardlink rather than a rename so an existing final path -- always
    expected to be byte-identical because callers name it by content digest
    -- is never overwritten.
    """
    token = os.urandom(16).hex()
    temp_path = directory / f".tmp-{token}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temp_path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, final_path)
        except FileExistsError:
            # A concurrent or previously crashed writer already published
            # this exact content-addressed path.
            pass
    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass


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
