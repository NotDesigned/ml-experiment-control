"""Short-lived, sealed source archives on the owned desktop staging volume."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import shutil

from .storage import atomic_json

CONTEXT_ID = re.compile(r"^context\.[0-9a-f]{32}$")
MAX_BYTES = 300 * 1024 ** 2


class BuildContexts:
    def __init__(self, root):
        self.root = Path(root) / "build-contexts"

    def path(self, identity):
        if not isinstance(identity, str) or not CONTEXT_ID.fullmatch(identity):
            raise ValueError("invalid source context identity")
        path = self.root / identity
        if self.root.is_symlink() or path.is_symlink():
            raise ValueError("invalid source context directory")
        return path

    def call(self, operation, data):
        path = self.path(data["context_id"])
        if operation == "context-remove":
            if path.exists():
                shutil.rmtree(path)
            return {"removed": True}
        if operation == "context-create":
            size, sha = data["bytes"], data["sha256"]
            if (not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= MAX_BYTES
                    or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)):
                raise ValueError("invalid source context manifest")
            self.root.mkdir(parents=True, exist_ok=True)
            if size + 64 * 1024 ** 2 > shutil.disk_usage(self.root).free:
                raise ValueError("insufficient source context storage")
            path.mkdir(mode=0o700)
            atomic_json(path / "manifest.json", {"bytes": size, "sha256": sha, "sealed": False})
            return {"created": True}
        if operation != "context-seal":
            raise ValueError("unknown source context operation")
        import json
        value = json.loads((path / "manifest.json").read_text())
        archive = path / "context.tar.gz"
        if archive.is_symlink() or not archive.is_file():
            raise ValueError("source context archive is absent")
        sha, size = hashlib.sha256(), 0
        with archive.open("rb") as stream:
            while chunk := stream.read(1024 ** 2):
                sha.update(chunk)
                size += len(chunk)
        if size != value["bytes"] or sha.hexdigest() != value["sha256"]:
            raise ValueError("source context checksum differs")
        archive.chmod(0o444)
        atomic_json(path / "manifest.json", {**value, "sealed": True})
        return {"sealed": True, "bytes": size, "sha256": sha.hexdigest()}

    def archive(self, identity):
        import json
        path = self.path(identity)
        value = json.loads((path / "manifest.json").read_text())
        archive = path / "context.tar.gz"
        if (not value.get("sealed") or archive.is_symlink() or not archive.is_file()
                or archive.stat().st_size != value["bytes"]):
            raise ValueError("source context is not sealed")
        return archive
