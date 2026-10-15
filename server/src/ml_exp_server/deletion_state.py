"""Permanent Run deletion fences shared by every daemon process.

Readers hold a shared lock while touching a Run. A delete claims a short,
nonblocking exclusive lock before publishing its tombstone; external deletion
effects run after that lock is released. Tombstones never expire or authorize
reuse of an old scientific identity.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
from pathlib import Path
import re
import threading

from .application_errors import ApplicationError
from .storage import atomic_json, read_json, utc_now

_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class RunFence:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.lock_path = self.root / ".run-deletion.lock"
        self._held = threading.local()

    def path(self, project, run):
        run_valid = isinstance(run, str) and (bool(_IDENTITY.fullmatch(run)) or any(
            run.endswith(suffix) and bool(_IDENTITY.fullmatch(run[:-len(suffix)]))
            for suffix in ("--preparation", "--backfill")))
        if not isinstance(project, str) or not _IDENTITY.fullmatch(project) or not run_valid:
            raise ValueError("invalid Run deletion identity")
        return self.root / "deleted-runs" / project / (run + ".json")

    @contextmanager
    def _locked(self, exclusive, blocking):
        current = getattr(self._held, "mode", None)
        if current is not None:
            if exclusive and current != "exclusive":
                raise ApplicationError("Run is busy; retry deletion", code="RUN_DELETION_BUSY")
            yield
            return
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.lock_path.open("a") as lock:
            operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            try:
                fcntl.flock(lock, operation | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError:
                raise ApplicationError("Run is busy; retry deletion", code="RUN_DELETION_BUSY") from None
            self._held.mode = "exclusive" if exclusive else "shared"
            try:
                yield
            finally:
                self._held.mode = None

    def shared(self):
        return self._locked(False, True)

    def exclusive(self, *, blocking=False):
        return self._locked(True, blocking)

    def is_deleted(self, project, run):
        # Atomic tombstone publication makes this safe inside an exclusive
        # section too. A malformed or unreadable tombstone still fences the ID.
        return self.path(project, run).exists()

    def require_live(self, project, run):
        if self.is_deleted(project, run):
            raise ApplicationError("Run was deleted by an administrator", status_code=410, code="RUN_DELETED")

    @contextmanager
    def guard(self, project, run):
        self.path(project, run)
        with self.shared():
            self.require_live(project, run)
            yield

    def mark(self, project, run, metadata=None):
        path = self.path(project, run)
        with self.exclusive():
            if path.exists():
                return read_json(path, {})
            value = {**(metadata or {}), "project": project, "run_id": run,
                     "status": "DELETED", "deleted_at": utc_now()}
            atomic_json(path, value)
            return value
