"""Validation shared by Dockerfile, data scripts and Run arguments."""
from __future__ import annotations

from pathlib import PurePosixPath


def relative_source_path(value: str) -> str:
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or "\\" in value or "\x00" in value
            or any(p in {".", ".."} or p.startswith(".") for p in value.split("/"))):
        raise ValueError("source file must use a relative, non-hidden path")
    return value


def argument_vector(value: list[str]) -> list[str]:
    if any(not item or "\x00" in item or len(item) > 8192 for item in value):
        raise ValueError("arguments contain an invalid value")
    return value
