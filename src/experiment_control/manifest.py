#!/usr/bin/env python
"""Create durable run/attempt metadata before a training process starts."""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import yaml
from .run_manifest import build_run_manifest, comparable_manifest

IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SECRET_KEY_RE = re.compile(
    r"(?:^|[_-])(?:secret|token|password|credential|access[_-]?key|api[_-]?key|"
    r"proxy|authorization|cookie)(?:$|[_-])",
    re.IGNORECASE,
)
URL_USERINFO_RE = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)[^/@\s]+@")


class RunState(str, Enum):
    """Controller/runtime states persisted in ``status.json``.

    ``NOT_SUBMITTED`` is a read-model value for a run without an attempt or
    submission record. ``SUBMITTING`` is the durable outbox state written
    before contacting a scheduler, closing the otherwise unavoidable crash
    window between scheduler acceptance and local bookkeeping.
    """

    NOT_SUBMITTED = "NOT_SUBMITTED"
    CREATED = "CREATED"
    SUBMITTING = "SUBMITTING"
    QUEUED = "QUEUED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    EVALUATING = "EVALUATING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    PREEMPTED = "PREEMPTED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class LifecycleStatus:
    """Typed normalized status read from or written to ``status.json``."""

    project: str | None
    run_id: str | None
    attempt_id: str | None
    state: RunState
    updated_at: str | None = None
    exit_code: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        return {key: value for key, value in payload.items() if value is not None}


def utc_now() -> str:
    """Return the current UTC timestamp in RFC 3339 form with a ``Z`` suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sanitize_command(command: list[str]) -> list[str]:
    """Redact secret-bearing ``KEY=value`` and flag-following command arguments."""
    sanitized: list[str] = []
    redact_next = False
    for argument in command:
        if redact_next:
            sanitized.append("<redacted>")
            redact_next = False
            continue
        if "=" in argument:
            key, value = argument.split("=", 1)
            sanitized.append(f"{key}=<redacted>" if SECRET_KEY_RE.search(key) else argument)
            continue
        sanitized.append(argument)
        if argument.startswith("-") and SECRET_KEY_RE.search(argument):
            redact_next = True
    return sanitized


def _fsync_dir(path: Path) -> None:
    """Persist a directory entry update after an atomic file replacement."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _durable_temp(path: Path, payload: Any, *, yaml_format: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".yaml" if yaml_format else ".json"
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=suffix, dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        if yaml_format:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
        else:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return temp_name


def atomic_write(path: Path, payload: Any, *, yaml_format: bool = False) -> None:
    """Durably replace a JSON or YAML file using fsync plus atomic rename."""
    temp_name = _durable_temp(path, payload, yaml_format=yaml_format)
    try:
        os.replace(temp_name, path)
        _fsync_dir(path.parent)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def atomic_create(path: Path, payload: Any, *, yaml_format: bool = False) -> None:
    """Durably create an immutable file, failing instead of overwriting it."""
    temp_name = _durable_temp(path, payload, yaml_format=yaml_format)
    try:
        # Hard-linking a complete temporary file makes publication atomic while
        # preserving O_EXCL-like no-overwrite behavior for concurrent writers.
        os.link(temp_name, path)
        _fsync_dir(path.parent)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def append_event(path: Path, event: dict[str, Any]) -> None:
    """Append and fsync one compact JSON object to a lifecycle event stream."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)


def append_event_once(path: Path, event: dict[str, Any], event_id: str) -> bool:
    """Append an event once, using a filesystem lock as the idempotency gate.

    Returns ``True`` when a new line was appended and ``False`` when the event
    ID was already present. The lock is separate from ``events.jsonl`` so an
    atomic replacement or log rotation cannot invalidate the lock inode.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.is_file():
            with path.open(encoding="utf-8") as existing:
                for line in existing:
                    try:
                        if json.loads(line).get("event_id") == event_id:
                            return False
                    except json.JSONDecodeError:
                        continue
        append_event(path, {**event, "event_id": event_id})
        return True


def validate_identity(label: str, value: str) -> None:
    """Require a scheduler/filesystem-safe run or attempt identity."""
    if not IDENTITY_RE.fullmatch(value):
        raise ValueError(
            f"{label}={value!r} is invalid; use 1-128 letters, digits, '.', '_' or '-'"
        )


def require_immutable(label: str, value: str) -> None:
    """Reject missing, mutable, or placeholder source/image identities."""
    if not value or value.lower() in {"unknown", "latest", "runtime", "seed"}:
        raise ValueError(f"{label} must be an immutable, non-placeholder identity")


def _sanitize_mapping(value: Any, key: str = "") -> Any:
    """Return a JSON-safe submission request with secret-bearing values removed."""
    if SECRET_KEY_RE.search(key):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitize_mapping(item, str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_mapping(item) for item in value]
    if isinstance(value, str):
        return URL_USERINFO_RE.sub(r"\g<scheme><redacted>@", value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise TypeError(f"cannot serialize submission request value of type {type(value).__name__}")


class ExperimentStateStore:
    """Reusable durable store shared by controllers and training runtimes.

    Immutable run/attempt manifests are kept separate from mutable scheduler
    submission and lifecycle observations. Submission uses a tiny durable
    outbox: write ``SUBMITTING`` first, contact the scheduler, then reconcile
    the returned job ID. Repeating either operation with the same data is safe.
    """

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir).resolve()

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.yaml"

    @property
    def legacy_manifest_path(self) -> Path:
        """Pre identity-v2 run manifest, accepted for observation only."""
        return self.run_dir / "control_manifest.yaml"

    @property
    def events_path(self) -> Path:
        return self.run_dir / "events.jsonl"

    @property
    def status_path(self) -> Path:
        return self.run_dir / "status.json"

    @property
    def backend_path(self) -> Path:
        return self.run_dir / "backend.json"

    def attempt_dir(self, attempt_id: str) -> Path:
        validate_identity("attempt_id", attempt_id)
        return self.run_dir / "attempts" / attempt_id

    def attempt_path(self, attempt_id: str) -> Path:
        return self.attempt_dir(attempt_id) / "attempt.yaml"

    def legacy_attempt_path(self, attempt_id: str) -> Path:
        """Pre identity-v2 attempt manifest, accepted for observation only."""
        return self.attempt_dir(attempt_id) / "control_attempt.yaml"

    def readable_manifest_path(self) -> Path:
        """Resolve canonical state first, then the immutable legacy snapshot."""
        if self.manifest_path.is_file():
            return self.manifest_path
        return self.legacy_manifest_path

    def readable_attempt_path(self, attempt_id: str) -> Path:
        """Resolve canonical state first, then the immutable legacy snapshot."""
        path = self.attempt_path(attempt_id)
        if path.is_file():
            return path
        return self.legacy_attempt_path(attempt_id)

    def submission_path(self, attempt_id: str) -> Path:
        return self.attempt_dir(attempt_id) / "submission.json"

    def attempt_status_path(self, attempt_id: str) -> Path:
        """Return the canonical mutable status path for one attempt."""
        return self.attempt_dir(attempt_id) / "status.json"

    def attempt_backend_path(self, attempt_id: str) -> Path:
        """Return the canonical scheduler identity path for one attempt."""
        return self.attempt_dir(attempt_id) / "backend.json"

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"state record is not an object: {path}")
        return payload

    def _current_attempt_id(self) -> str | None:
        """Return the attempt represented by root mirrors, failing on drift."""
        attempt_ids = {
            str(payload["attempt_id"])
            for path in (self.backend_path, self.status_path)
            if path.is_file()
            for payload in (self._load_json(path),)
            if payload.get("attempt_id")
        }
        if len(attempt_ids) > 1:
            raise ValueError(
                "root backend/status mirrors disagree about the current attempt"
            )
        return next(iter(attempt_ids), None)

    def _snapshot_root_mirrors(self) -> None:
        """Preserve legacy/current root state before selecting a new attempt."""
        current = self._current_attempt_id()
        if not current:
            return
        for root_path, canonical_path in (
            (self.backend_path, self.attempt_backend_path(current)),
            (self.status_path, self.attempt_status_path(current)),
        ):
            if not root_path.is_file():
                continue
            root_payload = self._load_json(root_path)
            if canonical_path.is_file():
                if self._load_json(canonical_path) != root_payload:
                    raise ValueError(
                        f"root mirror conflicts with canonical attempt state: {root_path}"
                    )
            else:
                atomic_create(canonical_path, root_payload)

    def _mirror_if_current(self, attempt_id: str, path: Path, payload: Mapping[str, Any]) -> None:
        current = self._current_attempt_id()
        if current in {None, attempt_id}:
            atomic_write(path, dict(payload))

    def load_backend(self, attempt_id: str | None = None) -> dict[str, Any] | None:
        """Load one attempt scheduler record, or the current root mirror."""
        if attempt_id is None:
            return self._load_json(self.backend_path) if self.backend_path.is_file() else None
        path = self.attempt_backend_path(attempt_id)
        if path.is_file():
            payload = self._load_json(path)
            if payload.get("attempt_id") != attempt_id:
                raise ValueError(f"backend record conflicts with selected attempt: {path}")
            return payload
        if self.backend_path.is_file():
            payload = self._load_json(self.backend_path)
            if payload.get("attempt_id") == attempt_id:
                return payload
        return None

    def load_status_payload(self, attempt_id: str | None = None) -> dict[str, Any] | None:
        """Load one attempt status record, or the current root mirror."""
        if attempt_id is None:
            return self._load_json(self.status_path) if self.status_path.is_file() else None
        path = self.attempt_status_path(attempt_id)
        if path.is_file():
            payload = self._load_json(path)
            if payload.get("attempt_id") != attempt_id:
                raise ValueError(f"status record conflicts with selected attempt: {path}")
            return payload
        if self.status_path.is_file():
            payload = self._load_json(self.status_path)
            if payload.get("attempt_id") == attempt_id:
                return payload
        return None

    def write_status_payload(
        self, attempt_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Write canonical attempt status and refresh its root mirror if current."""
        self.load_attempt(attempt_id)
        normalized = dict(payload)
        recorded_attempt = normalized.get("attempt_id")
        if recorded_attempt not in {None, attempt_id}:
            raise ValueError("status payload conflicts with selected attempt")
        normalized["attempt_id"] = attempt_id
        atomic_write(self.attempt_status_path(attempt_id), normalized)
        self._mirror_if_current(attempt_id, self.status_path, normalized)
        return normalized

    def load_manifest(self) -> dict[str, Any]:
        path = self.readable_manifest_path()
        if not path.is_file():
            raise FileNotFoundError(f"run manifest does not exist: {self.manifest_path}")
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def ensure_manifest(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        """Publish one immutable run identity or validate the existing identity."""
        candidate = dict(manifest)
        if self.manifest_path.is_file():
            existing = self.load_manifest()
            if comparable_manifest(existing) != comparable_manifest(candidate):
                raise ValueError("existing run manifest conflicts")
            return existing
        if self.legacy_manifest_path.is_file():
            raise ValueError(
                "legacy control_manifest.yaml is observation-only; migrate to a new "
                "run_id before preparing or submitting another attempt"
            )
        try:
            atomic_create(self.manifest_path, candidate, yaml_format=True)
        except FileExistsError:
            return self.ensure_manifest(candidate)
        return candidate

    def load_attempt(self, attempt_id: str) -> dict[str, Any]:
        path = self.readable_attempt_path(attempt_id)
        if not path.is_file():
            raise FileNotFoundError(
                f"attempt manifest does not exist: {self.attempt_path(attempt_id)}"
            )
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def create_attempt(self, attempt: Mapping[str, Any]) -> Path:
        """Atomically publish one immutable attempt manifest."""
        attempt_id = str(attempt.get("attempt_id", ""))
        validate_identity("attempt_id", attempt_id)
        if not self.manifest_path.is_file():
            raise FileNotFoundError("cannot create an attempt before the run manifest")
        manifest = self.load_manifest()
        if (
            attempt.get("project") != manifest.get("project")
            or attempt.get("run_id") != manifest.get("run_id")
        ):
            raise ValueError("attempt identity conflicts with run manifest")
        path = self.attempt_path(attempt_id)
        atomic_create(path, dict(attempt), yaml_format=True)
        return path

    def initialize_attempt_records(self, attempt_id: str) -> LifecycleStatus:
        """Idempotently initialize derived status, backend, and event records."""
        self._snapshot_root_mirrors()
        attempt = self.load_attempt(attempt_id)
        timestamp = attempt["created_at"]
        attempt_backend = attempt["backend"]
        backend_kind = (
            attempt_backend["kind"]
            if isinstance(attempt_backend, Mapping)
            else attempt_backend
        )
        backend = self.load_backend(attempt_id)
        if backend is None:
            backend = {
                "backend": backend_kind,
                "backend_job_id": attempt.get("backend_job_id"),
                "attempt_id": attempt_id,
            }
            atomic_write(self.attempt_backend_path(attempt_id), backend)
        status_payload = self.load_status_payload(attempt_id)
        if status_payload is None:
            status = self._write_status(
                project=attempt["project"],
                run_id=attempt["run_id"],
                attempt_id=attempt_id,
                state=RunState.CREATED,
                timestamp=timestamp,
                mirror=False,
            )
        else:
            status = self.read_status(attempt_id)
        # Preparing an attempt explicitly makes it current.  The root records
        # are read-model mirrors only; canonical history remains below attempts/.
        atomic_write(self.backend_path, backend)
        atomic_write(
            self.status_path,
            self._load_json(self.attempt_status_path(attempt_id)),
        )
        append_event_once(
            self.events_path,
            {
                "timestamp": timestamp,
                "project": attempt["project"],
                "run_id": attempt["run_id"],
                "attempt_id": attempt_id,
                "backend": backend_kind,
                "backend_job_id": attempt.get("backend_job_id"),
                "event": "attempt_created",
                "payload": {
                    "command": attempt["command"],
                    "output_dir": str(self.run_dir),
                    "resume_from": attempt.get("resume_from"),
                },
            },
            f"attempt-created:{attempt_id}",
        )
        return status

    def read_status(self, attempt_id: str | None = None) -> LifecycleStatus:
        """Read normalized state without raising for an unsubmitted run."""
        payload = self.load_status_payload(attempt_id)
        if payload is not None:
            return LifecycleStatus(
                project=payload.get("project"),
                run_id=payload.get("run_id"),
                attempt_id=payload.get("attempt_id"),
                state=RunState(payload.get("state", RunState.UNKNOWN.value)),
                updated_at=payload.get("updated_at"),
                exit_code=payload.get("exit_code"),
            )
        if attempt_id and self.readable_attempt_path(attempt_id).is_file():
            attempt = self.load_attempt(attempt_id)
            return LifecycleStatus(
                project=attempt.get("project"),
                run_id=attempt.get("run_id"),
                attempt_id=attempt_id,
                state=RunState.CREATED,
            )
        manifest_path = self.readable_manifest_path()
        manifest = (
            yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file() else {}
        )
        return LifecycleStatus(
            project=manifest.get("project"),
            run_id=manifest.get("run_id"),
            attempt_id=attempt_id,
            state=RunState.NOT_SUBMITTED,
        )

    def read_submission(self, attempt_id: str) -> dict[str, Any] | None:
        path = self.submission_path(attempt_id)
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def _validate_attempt_identity(
        self, project: str, run_id: str, attempt_id: str
    ) -> dict[str, Any]:
        manifest = self.load_manifest()
        attempt = self.load_attempt(attempt_id)
        expected = (project, run_id, attempt_id)
        actual = (attempt.get("project"), attempt.get("run_id"), attempt.get("attempt_id"))
        if (manifest.get("project"), manifest.get("run_id")) != expected[:2] or actual != expected:
            raise ValueError("submission identity conflicts with run or attempt manifest")
        return attempt

    def _write_status(
        self,
        *,
        project: str,
        run_id: str,
        attempt_id: str,
        state: RunState,
        exit_code: int | None = None,
        timestamp: str | None = None,
        mirror: bool = True,
    ) -> LifecycleStatus:
        status = LifecycleStatus(
            project=project,
            run_id=run_id,
            attempt_id=attempt_id,
            state=state,
            updated_at=timestamp or utc_now(),
            exit_code=exit_code,
        )
        payload = status.to_dict()
        atomic_write(self.attempt_status_path(attempt_id), payload)
        if mirror:
            self._mirror_if_current(attempt_id, self.status_path, payload)
        return status

    def begin_submission(
        self,
        *,
        project: str,
        run_id: str,
        attempt_id: str,
        backend: str,
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Persist a submission intent before making an external scheduler call."""
        self._validate_attempt_identity(project, run_id, attempt_id)
        sanitized_request = _sanitize_mapping(request)
        path = self.submission_path(attempt_id)
        existing = self.read_submission(attempt_id)
        if existing:
            immutable = {
                "project": project,
                "run_id": run_id,
                "attempt_id": attempt_id,
                "backend": backend,
                "request": sanitized_request,
            }
            conflicts = [key for key, value in immutable.items() if existing.get(key) != value]
            if conflicts:
                raise ValueError("existing submission intent conflicts in " + ", ".join(conflicts))
            intent = existing
        else:
            intent = {
                "project": project,
                "run_id": run_id,
                "attempt_id": attempt_id,
                "backend": backend,
                "request": sanitized_request,
                "state": "SUBMITTING",
                "created_at": utc_now(),
            }
            try:
                atomic_create(path, intent)
            except FileExistsError:
                # A concurrent controller won publication. Re-enter through
                # the same conflict checks instead of overwriting its intent.
                return self.begin_submission(
                    project=project,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    backend=backend,
                    request=request,
                )

        # Repair derived files/events after a crash, but never regress a
        # reconciled submission back to SUBMITTING.
        if intent.get("state") == "SUBMITTING":
            timestamp = intent["created_at"]
            backend_payload = {
                "backend": backend, "backend_job_id": None, "attempt_id": attempt_id
            }
            atomic_write(self.attempt_backend_path(attempt_id), backend_payload)
            self._mirror_if_current(attempt_id, self.backend_path, backend_payload)
            self._write_status(
                project=project,
                run_id=run_id,
                attempt_id=attempt_id,
                state=RunState.SUBMITTING,
                timestamp=timestamp,
            )
            append_event_once(
                self.events_path,
                {
                    "timestamp": timestamp,
                    "project": project,
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "backend": backend,
                    "backend_job_id": None,
                    "event": "submission_intent_created",
                    "payload": {"request": sanitized_request},
                },
                f"submission-intent:{attempt_id}",
            )
        return intent

    def reconcile_submission(
        self,
        *,
        project: str,
        run_id: str,
        attempt_id: str,
        backend_job_id: str,
        state: RunState = RunState.QUEUED,
    ) -> dict[str, Any]:
        """Attach a scheduler job to a prior intent, safely and idempotently."""
        self._validate_attempt_identity(project, run_id, attempt_id)
        if not backend_job_id:
            raise ValueError("backend_job_id must not be empty")
        intent = self.read_submission(attempt_id)
        if not intent:
            raise FileNotFoundError("cannot reconcile scheduler job before submission intent")
        existing_job_id = intent.get("backend_job_id")
        if existing_job_id and existing_job_id != backend_job_id:
            raise ValueError(
                f"attempt is already reconciled to backend job {existing_job_id!r}"
            )
        timestamp = intent.get("reconciled_at") or utc_now()
        reconciled = {
            **intent,
            "state": "SUBMITTED",
            "backend_job_id": backend_job_id,
            "reconciled_at": timestamp,
        }
        atomic_write(self.submission_path(attempt_id), reconciled)
        backend_payload = {
            "backend": intent["backend"],
            "backend_job_id": backend_job_id,
            "attempt_id": attempt_id,
        }
        atomic_write(self.attempt_backend_path(attempt_id), backend_payload)
        self._mirror_if_current(attempt_id, self.backend_path, backend_payload)
        self._write_status(
            project=project,
            run_id=run_id,
            attempt_id=attempt_id,
            state=state,
            timestamp=timestamp,
        )
        append_event_once(
            self.events_path,
            {
                "timestamp": timestamp,
                "project": project,
                "run_id": run_id,
                "attempt_id": attempt_id,
                "backend": intent["backend"],
                "backend_job_id": backend_job_id,
                "event": "submission_reconciled",
                "payload": {"state": state.value},
            },
            f"submission-reconciled:{attempt_id}:{backend_job_id}",
        )
        return reconciled

    def transition(
        self,
        *,
        project: str,
        run_id: str,
        attempt_id: str,
        state: RunState,
        event: str,
        payload: Mapping[str, Any] | None = None,
        event_id: str | None = None,
        exit_code: int | None = None,
    ) -> LifecycleStatus:
        """Write status atomically and append an optionally idempotent event."""
        self._validate_attempt_identity(project, run_id, attempt_id)
        timestamp = utc_now()
        status = self._write_status(
            project=project,
            run_id=run_id,
            attempt_id=attempt_id,
            state=state,
            exit_code=exit_code,
            timestamp=timestamp,
        )
        backend = self.load_backend(attempt_id) or {}
        record = {
            "timestamp": timestamp,
            "project": project,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "backend": backend.get("backend"),
            "backend_job_id": backend.get("backend_job_id"),
            "event": event,
            "payload": _sanitize_mapping(payload or {}),
        }
        if event_id:
            append_event_once(self.events_path, record, event_id)
        else:
            append_event(self.events_path, record)
        return status
