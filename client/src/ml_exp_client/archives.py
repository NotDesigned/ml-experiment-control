"""Credential-safe source packaging and reproducible data archives."""
from __future__ import annotations

import gzip
import hashlib
import io
from pathlib import Path
import tarfile

from .api import ClientError


def source_archive(directory: Path) -> bytes:
    if not directory.is_dir() or directory.is_symlink():
        raise ClientError("source must be a regular directory")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(directory)
            # Skip developer metadata; never silently skip credential-looking files.
            if any(p in {".git", ".venv", "__pycache__"} for p in relative.parts):
                continue
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                raise ClientError("source contains a link or special file")
            if (any(p.startswith(".env") or p.lower() in {"credentials", "secrets", ".ssh"}
                    for p in relative.parts) or path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}):
                raise ClientError("remove credential-looking paths before uploading source")
            if path.is_file():
                archive.add(path, arcname=relative.as_posix(), recursive=False)
    value = stream.getvalue()
    if len(value) > 64 * 1024 ** 2:
        raise ClientError("client limits source uploads to 64 MiB")
    return value


def copy_stream(source, target=None):
    digest, size = hashlib.sha256(), 0
    while chunk := source.read(1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
        if target is not None:
            target.write(chunk)
    return digest.hexdigest(), size


def data_archive(directory: Path, stream):
    if not directory.is_dir() or directory.is_symlink():
        raise ClientError("data asset must be a regular directory")
    with gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as compressed, \
            tarfile.open(fileobj=compressed, mode="w") as archive:
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(directory)
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                raise ClientError("data asset contains a link or special file")
            if any(p.startswith(".") for p in relative.parts) or path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}:
                raise ClientError("data asset contains a hidden or credential-looking path")
            if path.is_file():
                member = archive.gettarinfo(str(path), arcname=relative.as_posix())
                member.uid = member.gid = member.mtime = 0
                member.uname = member.gname = ""
                member.mode = 0o400
                with path.open("rb") as body:
                    archive.addfile(member, body)
    length = stream.tell()
    stream.seek(0)
    digest, _ = copy_stream(stream)
    stream.seek(0)
    return digest, length


def data_transport_identity(stream, length):
    """Declare file identities and the deterministic, uncompressed OCI layer.

    The backend still verifies the actual archive/files before training. This
    lets CCR verify an upload after a server restart without storing its bytes.
    """
    inventory = []
    stream.seek(0)
    with tarfile.open(fileobj=stream, mode="r|*") as archive:
        for member in archive:
            if not member.isfile():
                raise ClientError("data archive must contain regular files")
            body = archive.extractfile(member)
            sha, size = copy_stream(body)
            if size != member.size:
                raise ClientError("data archive file is truncated")
            inventory.append({"path": member.name, "sha256": sha, "bytes": size})
    member = tarfile.TarInfo("payload/dataset.tar")
    member.size, member.mode = length, 0o444
    sha = hashlib.sha256(member.tobuf(format=tarfile.GNU_FORMAT))
    stream.seek(0)
    while chunk := stream.read(1024 ** 2):
        sha.update(chunk)
    sha.update(b"\0" * ((-length) % 512 + 1024))
    stream.seek(0)
    return sorted(inventory, key=lambda item: item["path"]), sha.hexdigest()
