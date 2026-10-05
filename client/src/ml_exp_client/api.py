"""Protocol-2 HTTP transport, source packaging and verified artifact retrieval.

No request is retried automatically. See docs/api-quickstart.md.
"""
from __future__ import annotations

import hashlib
import io
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class ClientError(RuntimeError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        # Never forward a bearer token to another URL after a proxy redirect.
        return None


def segment(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}", value):
        raise ClientError("invalid API identity")
    return quote(value, safe="")


class Client:
    def __init__(self, base: str, token: str, *, timeout: float = 60):
        url = urlsplit(base)
        loopback = url.hostname == "localhost"
        try:
            loopback = loopback or ipaddress.ip_address(url.hostname or "").is_loopback
        except ValueError:
            pass
        if (not url.hostname or url.username or url.password or url.query or url.fragment
                or not (url.scheme == "https" or url.scheme == "http" and loopback)):
            raise ClientError("API URL must use HTTPS (HTTP is allowed on loopback only)")
        if not token or any(c.isspace() for c in token):
            raise ClientError("set ML_EXPD_API_TOKEN or ML_EXPD_API_TOKEN_FILE")
        self.base, self.token, self.timeout = base.rstrip("/"), token, timeout
        self.opener = build_opener(NoRedirect())

    def open(self, path: str, *, data=None, raw: bytes | None = None):
        if not path.startswith("/api/") or "\x00" in path:
            raise ClientError("request must stay within /api/")
        headers = {"Authorization": "Bearer " + self.token,
                   "X-ML-Expd-Client-Protocol": "2"}
        body = raw
        if data is not None:
            body = json.dumps(data).encode()
            headers["Content-Type"] = "application/json"
        elif raw is not None:
            headers["Content-Type"] = "application/octet-stream"
        method = "GET" if body is None else "POST"
        try:
            return self.opener.open(Request(self.base + path, body, headers, method=method),
                                    timeout=self.timeout)
        except HTTPError as exc:
            code = exc.headers.get("X-ML-Expd-Error-Code", "API_ERROR")
            exc.close()
            raise ClientError(f"{method} {path}: HTTP {exc.code} {code}; inspect saved IDs before retrying") from None
        except (URLError, TimeoutError, OSError):
            raise ClientError(f"{method} {path}: connection failed; inspect saved IDs before retrying") from None

    def call(self, path: str, **kwargs):
        with self.open(path, **kwargs) as response:
            return json.load(response)

    def negotiate(self):
        health = self.call("/api/health")
        if not health["min_client_protocol_version"] <= 2 <= health["api_protocol_version"]:
            raise ClientError("daemon does not support protocol 2")
        return health

    def wait(self, path: str, *, pending=("EXECUTING",), seconds=900, interval=2):
        deadline = time.monotonic() + seconds
        while True:
            value = self.call(path)
            if value["status"] not in pending:
                return value
            if time.monotonic() >= deadline:
                raise ClientError(f"poll timed out: GET {path}; execution continues on the server")
            time.sleep(interval)


def save(path: Path, value):
    # State contains recovery IDs and sanitized API responses, never auth headers.
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as output:
        os.chmod(temporary, 0o600)
        json.dump(value, output, indent=2)
        output.write("\n")
    temporary.replace(path)


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


def download(client: Client, project: str, run: str, attempt: str, out: Path):
    endpoint = f"/api/runs/{segment(project)}/{segment(run)}/attempts/{segment(attempt)}"
    listing = client.call(endpoint + "/files")
    if not listing["files"] or listing.get("truncated"):
        raise ClientError("files are unavailable or listing is truncated; inspect this exact Attempt")
    out.mkdir(parents=True, exist_ok=False)
    downloaded = {}
    for item in listing["files"]:
        path = PurePosixPath(item["path"])
        if (path.is_absolute() or ".." in path.parts or "\\" in item["path"]
                or not path.parts or path.parts[0] != "outputs"
                or any(p.startswith(".") for p in path.parts)):
            continue  # Backend logs/metadata are separate from uploaded outputs.
        target = out.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with client.open(endpoint + "/files/" + quote(item["path"], safe="/")) as response:
            with target.open("xb") as output:
                digest, size = copy_stream(response, output)
        if size != item["bytes"]:
            raise ClientError("downloaded file size differs from listing")
        downloaded[path.as_posix()] = {"sha256": digest, "bytes": size}
    with client.open(endpoint + "/artifacts/archive") as response:
        expected = response.headers.get("ETag", "").strip('"')
        with (out / "artifacts.tar").open("xb") as output:
            digest, size = copy_stream(response, output)
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or digest != expected:
        raise ClientError("artifact archive SHA256 differs from server receipt")
    members = set()
    with tarfile.open(out / "artifacts.tar", mode="r:") as archive:
        for member in archive:
            if member.isdir():
                continue
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in member.name:
                raise ClientError("archive contains an unsafe path")
            name = "outputs/" + relative.as_posix()
            if not member.isfile() or name in members or name not in downloaded:
                raise ClientError("archive contains an unexpected or duplicate file")
            members.add(name)
            with archive.extractfile(member) as body:
                actual, count = copy_stream(body)
            if downloaded[name] != {"sha256": actual, "bytes": count}:
                raise ClientError("individual file differs from uploaded archive")
    if members != set(downloaded):
        raise ClientError("archive and downloaded file sets differ")
    report = {"project": project, "run_id": run, "attempt_id": attempt,
              "archive_sha256": digest, "archive_bytes": size, "files": downloaded}
    save(out / "verification.json", report)
    return report


def copy_stream(source, target=None):
    digest, size = hashlib.sha256(), 0
    while chunk := source.read(1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
        if target is not None:
            target.write(chunk)
    return digest.hexdigest(), size
