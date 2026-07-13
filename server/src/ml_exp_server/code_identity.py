"""Observed identity of the Project code checkout managed by the daemon."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from .schemas import ResearchProject


def _digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def project_code_identity(project: ResearchProject) -> dict[str, Any]:
    root = Path(project.base_dir or ".").expanduser().resolve()
    authored = Path(project.authored_file).resolve() if project.authored_file else None
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1", "-z"],
            check=True, capture_output=True, timeout=10,
        ).stdout
        repository = {
            "kind": "git", "commit": commit, "dirty": bool(status),
            "status_digest": "sha256:" + hashlib.sha256(status).hexdigest(),
        }
    except (FileNotFoundError, subprocess.SubprocessError):
        repository = {
            "kind": "directory", "commit": None, "dirty": None,
            "status_digest": None,
        }
    try:
        relative = str(authored.relative_to(root)) if authored else None
    except ValueError:
        relative = None
    return {
        "repository": repository,
        "project_file_relative": relative,
        "project_file_digest": _digest(authored) if authored else None,
    }
