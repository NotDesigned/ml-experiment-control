"""Daemon-host credential storage for optional publication targets.

Secrets deliberately have no HTTP representation.  Operators provision them
through the local ``ml-expd credential`` command and publishers resolve them
only at the point of use.
"""

from __future__ import annotations

import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path


_REFERENCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_MAX_SECRET_BYTES = 4096


class CredentialError(ValueError):
    """A credential reference or secret failed validation."""


@dataclass(frozen=True)
class CredentialStatus:
    reference: str
    configured: bool


class CredentialStore:
    """Small file-backed secret store with strict ownership permissions."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().absolute()

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

    @staticmethod
    def _reject_symlink(path: Path, label: str) -> None:
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise CredentialError(f"{label} must not be a symlink")

    def _secure_root(self, *, create: bool) -> None:
        self._reject_symlink(self.root, "credential directory")
        if create:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._reject_symlink(self.root, "credential directory")
            try:
                os.chmod(self.root, 0o700)
            except OSError as exc:
                raise CredentialError("cannot secure credential directory") from exc
        if not self.root.exists():
            return
        metadata = self.root.stat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise CredentialError("credential directory path is not a directory")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise CredentialError("credential directory permissions must be 0700")
        if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
            raise CredentialError("credential directory must be owned by the current user")

    def _validate_path(self, path: Path) -> None:
        self._reject_symlink(path, "credential file")
        try:
            metadata = path.stat()
        except FileNotFoundError:
            return
        if not stat.S_ISREG(metadata.st_mode):
            raise CredentialError("credential path is not a regular file")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise CredentialError("credential file permissions must be 0600")
        if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
            raise CredentialError("credential file must be owned by the current user")
        if metadata.st_size > _MAX_SECRET_BYTES:
            raise CredentialError("W&B credential is too long")

    def set_wandb_api_key(self, reference: str, secret: str) -> None:
        value = secret.strip()
        if not value or "\n" in value or "\r" in value or "\0" in value:
            raise CredentialError("W&B API key must be one non-empty line")
        if len(value.encode("utf-8")) > _MAX_SECRET_BYTES:
            raise CredentialError("W&B API key is too long")
        self._secure_root(create=True)
        destination = self._path(reference)
        self._validate_path(destination)
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
        self._secure_root(create=False)
        path = self._path(reference)
        self._validate_path(path)
        try:
            value = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise CredentialError("W&B credential is not configured") from exc
        except (OSError, UnicodeDecodeError) as exc:
            raise CredentialError("W&B credential is unreadable") from exc
        if not value or any(char in value for char in "\n\r\0"):
            raise CredentialError("W&B credential has an invalid shape")
        return value

    def clear_wandb_api_key(self, reference: str) -> bool:
        self._secure_root(create=False)
        path = self._path(reference)
        self._validate_path(path)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True

    def status(self, reference: str) -> CredentialStatus:
        normalized = self.validate_reference(reference)
        self._secure_root(create=False)
        path = self._path(normalized)
        self._validate_path(path)
        configured = path.is_file()
        if configured:
            self.resolve_wandb_api_key(normalized)
        return CredentialStatus(normalized, configured)
