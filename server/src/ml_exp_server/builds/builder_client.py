"""Shared private image-builder identities and errors."""
from __future__ import annotations

import re


IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}$")
PROJECT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
MANIFEST_TYPE = "application/vnd.docker.distribution.manifest.v2+json"


class BuildStorageError(ValueError):
    """Only reviewed storage diagnostics may cross the private builder socket."""

    def __init__(self, code: str, details: dict):
        super().__init__(code)
        self.code, self.details = code, details


class BuildTransportError(ValueError):
    """Bounded transport evidence; uncertain publications are never replayed."""

    def __init__(self, code, details):
        super().__init__(code)
        self.code, self.details = code, details


class BuildReceiptMissingError(ValueError):
    """The builder verified the lookup identity and found no published receipt."""

    def __init__(self, message, *, retry_proof=None):
        super().__init__(message)
        self.retry_proof = retry_proof
