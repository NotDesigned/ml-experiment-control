"""Durable creation of an immutable JSON file."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _fsync_dir(path: Path) -> None:
    """Persist a directory entry update after an atomic file replacement."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_create(path: Path, payload: Any) -> None:
    """Durably create an immutable file, failing instead of overwriting it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".json", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        # Hard-linking a complete temporary file makes publication atomic while
        # preserving O_EXCL-like no-overwrite behavior for concurrent writers.
        os.link(temp_name, path)
        _fsync_dir(path.parent)
    finally:
        Path(temp_name).unlink(missing_ok=True)
