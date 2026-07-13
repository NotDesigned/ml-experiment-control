"""Stable identities for independently running daemon workspaces."""

from __future__ import annotations

import hashlib

from .schemas import ServerConfig


def workspace_identity(config: ServerConfig) -> str:
    """Bind a daemon instance to its durable index, not mutable membership."""
    identity = str(config.index_db_path().resolve())
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
