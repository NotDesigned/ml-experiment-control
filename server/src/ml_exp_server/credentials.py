"""Daemon-host credential storage for optional publication targets.

Secrets deliberately have no HTTP representation.  Operators provision them
through the local ``ml-expd credential`` command and publishers resolve them
only at the point of use.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path


_REFERENCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class CredentialError(ValueError):
    """A credential reference or secret failed validation."""


@dataclass(frozen=True)
class CredentialStatus:
    reference: str
    configured: bool


class CredentialStore:
    """Small file-backed secret store with strict ownership permissions."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    @staticmethod
    def validate_reference(reference: str) -> str:
        value = reference.strip()
        if not _REFERENCE.fullmatch(value):
            raise CredentialError(
                "credential reference must use only letters, digits, '.', '_' or '-'"
            )
        return value

    def _path(self, reference: str) -> Path:
        return self.root / f"{self.validate_reference(reference)}.wandb-api-key"

    def set_wandb_api_key(self, reference: str, secret: str) -> None:
        value = secret.strip()
        if not value or "\n" in value or "\r" in value or "\0" in value:
            raise CredentialError("W&B API key must be one non-empty line")
        if len(value) > 4096:
            raise CredentialError("W&B API key is too long")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        destination = self._path(reference)
        fd, temporary = tempfile.mkstemp(prefix=".credential-", dir=self.root)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            os.chmod(destination, 0o600)
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def resolve_wandb_api_key(self, reference: str) -> str:
        path = self._path(reference)
        try:
            value = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise CredentialError("W&B credential is not configured") from exc
        if not value:
            raise CredentialError("W&B credential is empty")
        return value

    def clear_wandb_api_key(self, reference: str) -> bool:
        try:
            self._path(reference).unlink()
        except FileNotFoundError:
            return False
        return True

    def status(self, reference: str) -> CredentialStatus:
        normalized = self.validate_reference(reference)
        return CredentialStatus(normalized, self._path(normalized).is_file())
