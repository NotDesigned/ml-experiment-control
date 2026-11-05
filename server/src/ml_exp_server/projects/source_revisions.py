"""Content fingerprint of an immutable source tree."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def _tree_digest(root: Path, *, require_read_only: bool = False) -> str:
    """Hash a source tree without following links or trusting filesystem modes."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("source revision tree is unavailable")
    if require_read_only and root.stat().st_mode & 0o222:
        raise ValueError("source revision tree is writable")
    digest = hashlib.sha256()

    def field(value: bytes) -> None:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)

    stack = [root]
    while stack:
        directory = stack.pop()
        children = sorted(directory.iterdir(), key=lambda item: item.name)
        directories: list[Path] = []
        for path in children:
            relative = path.relative_to(root).as_posix().encode("utf-8")
            if path.is_symlink():
                target = os.readlink(path)
                if Path(target).is_absolute():
                    raise ValueError("source revision tree contains an escaping symlink")
                try:
                    resolved = path.resolve(strict=False)
                except (OSError, RuntimeError) as exc:
                    raise ValueError("source revision tree contains an invalid symlink") from exc
                if resolved != root and root not in resolved.parents:
                    raise ValueError("source revision tree contains an escaping symlink")
                field(b"L")
                field(relative)
                field(target.encode("utf-8"))
            elif path.is_dir():
                if require_read_only and path.stat().st_mode & 0o222:
                    raise ValueError("source revision tree is writable")
                field(b"D")
                field(relative)
                directories.append(path)
            elif path.is_file():
                mode = path.stat().st_mode
                if require_read_only and mode & 0o222:
                    raise ValueError("source revision tree is writable")
                field(b"F")
                field(relative)
                field(b"X" if mode & 0o111 else b"-")
                with path.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        field(chunk)
            else:
                raise ValueError("source revision tree contains a special file")
        stack.extend(reversed(directories))
    return "sha256:" + digest.hexdigest()
