"""Native bearer authentication for the daemon HTTP boundary."""

from __future__ import annotations

import os
from pathlib import Path
import stat


class HttpAuthError(ValueError):
    """The configured HTTP credential cannot be used safely."""


def load_bearer_token(path: Path) -> str:
    """Read a daemon token only from a private, owner-controlled regular file."""

    expanded = path.expanduser()
    try:
        descriptor = os.open(
            expanded,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise HttpAuthError(
                    "HTTP bearer token file must be a regular file, not a symlink"
                )
            if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
                raise HttpAuthError("HTTP bearer token file must be owned by the daemon user")
            if stat.S_IMODE(metadata.st_mode) & 0o077:
                raise HttpAuthError("HTTP bearer token file permissions must be 0600 or stricter")
            if metadata.st_size > 4096:
                raise HttpAuthError("HTTP bearer token file is unexpectedly large")
            raw = os.read(descriptor, 4097)
            if len(raw) > 4096:
                raise HttpAuthError("HTTP bearer token file is unexpectedly large")
        finally:
            os.close(descriptor)
    except HttpAuthError:
        raise
    except OSError as exc:
        if expanded.is_symlink():
            raise HttpAuthError(
                "HTTP bearer token file must be a regular file, not a symlink"
            ) from exc
        raise HttpAuthError(
            f"HTTP bearer token file is unavailable: {expanded}"
        ) from exc
    try:
        token = raw.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise HttpAuthError("HTTP bearer token file is unreadable") from exc
    if len(token) < 32 or any(character.isspace() for character in token):
        raise HttpAuthError(
            "HTTP bearer token must contain at least 32 non-whitespace characters"
        )
    return token
