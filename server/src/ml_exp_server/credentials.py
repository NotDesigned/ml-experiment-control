"""Small daemon-local write-only credential store."""
from __future__ import annotations

import os
import re
from pathlib import Path


class CredentialStore:
    def __init__(self, root: Path):
        self.root = root.expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)

    def _path(self, ref: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", ref):
            raise ValueError("invalid credential_ref")
        return self.root / ref

    def put(self, ref: str, value: str) -> None:
        secret = value.strip()
        if not secret:
            raise ValueError("credential must not be empty")
        path = self._path(ref)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(secret + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(path)

    def configured(self, ref: str | None) -> bool:
        return bool(ref and self._path(ref).is_file())

