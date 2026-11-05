"""Validation of Dockerfile paths."""
from __future__ import annotations

from pathlib import PurePosixPath


def relative_source_path(value: str) -> str:
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or "\\" in value or "\x00" in value
            or any(p in {".", ".."} or p.startswith(".") for p in value.split("/"))):
        raise ValueError("source file must use a relative, non-hidden path")
    return value
