"""Cheap repository identity checks shared by server and Agent clients."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any


def repository_identity(root: Path) -> dict[str, Any]:
    workspace = root.expanduser().resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(workspace), "status", "--porcelain=v1", "-z"],
            check=True, capture_output=True, timeout=10,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"kind": "directory", "commit": None, "dirty": None,
                "status_digest": None}
    return {
        "kind": "git", "commit": commit, "dirty": bool(status),
        "status_digest": "sha256:" + hashlib.sha256(status).hexdigest(),
    }


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def require_matching_repository(expected: dict[str, Any], root: Path) -> None:
    actual = repository_identity(root)
    for key in ("kind", "commit", "dirty", "status_digest"):
        if expected.get(key) != actual.get(key):
            raise ValueError(
                f"Agent workspace identity mismatch for {key}: "
                f"server={expected.get(key)!r}, client={actual.get(key)!r}"
            )
