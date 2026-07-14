"""Small file-identity helpers shared by action preparation and policy."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
