"""Protocol-2 HTTP transport, source packaging and verified artifact retrieval.

Only resumable archive transfers retry transient failures. Scheduler operations
are never replayed automatically. See docs/api-quickstart.md.
"""
from __future__ import annotations

import hashlib
import gzip
import io
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class ClientError(RuntimeError):
    def __init__(self, message, *, status=None, retryable=False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        # Never forward a bearer token to another URL after a proxy redirect.
        return None


def segment(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}", value):
        raise ClientError("invalid API identity")
    return quote(value, safe="")


class Client:
    def __init__(self, base: str, token: str, *, timeout: float = 60, upload_timeout: float = 1200):
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
        self.upload_timeout = max(timeout, upload_timeout)
        self.opener = build_opener(NoRedirect())

    def open(self, path: str, *, data=None, raw=None, length: int | None = None, method=None, timeout=None):
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
            if length is not None:
                headers["Content-Length"] = str(length)
        method = method or ("GET" if body is None else "POST")
        try:
            return self.opener.open(Request(self.base + path, body, headers, method=method),
                                    timeout=self.timeout if timeout is None else timeout)
        except HTTPError as exc:
            code = exc.headers.get("X-ML-Expd-Error-Code", "API_ERROR")
            exc.close()
            raise ClientError(f"{method} {path}: HTTP {exc.code} {code}; inspect saved IDs before retrying", status=exc.code) from None
        except (URLError, TimeoutError, OSError):
            raise ClientError(f"{method} {path}: connection failed; inspect saved IDs before retrying", retryable=True) from None

    def call(self, path: str, **kwargs):
        with self.open(path, **kwargs) as response:
            return json.load(response)

    def open_object(self, url: str):
        target = urlsplit(url)
        if (target.scheme != "https" or not target.hostname or target.username or target.password or target.fragment):
            raise ClientError("object storage download must use HTTPS")
        try:
            # The signed URL is the only credential. Do not forward API Bearer
            # headers or follow redirects; never echo the URL in errors/state.
            return self.opener.open(Request(url), timeout=self.timeout)
        except (HTTPError, URLError, TimeoutError, OSError):
            raise ClientError("object storage download failed; request a fresh download link") from None

    def negotiate(self):
        health = self.call("/api/health")
        if not health["min_client_protocol_version"] <= 2 <= health["api_protocol_version"]:
            raise ClientError("daemon does not support protocol 2")
        return health

    def wait(self, path: str, *, pending=("EXECUTING",), seconds=900, interval=2, observer=None):
        deadline = time.monotonic() + seconds
        while True:
            value = self.call(path)
            if value["status"] not in pending:
                return value
            if observer is not None:
                observer(value)
            if time.monotonic() >= deadline:
                raise ClientError(f"poll timed out: GET {path}; execution continues on the server")
            time.sleep(interval)


def save(path: Path, value):
    # State contains recovery IDs and sanitized API responses, never auth headers.
    output = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name + ".", delete=False)
    temporary = Path(output.name)
    try:
        with output:
            json.dump(value, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


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
    try:
        ticket = client.call(endpoint + "/artifacts/download")
    except ClientError as exc:
        if exc.status != 404:
            raise
        return _download_proxy(client, project, run, attempt, out)
    return _download_object(client, project, run, attempt, out, ticket)


def _download_object(client, project, run, attempt, out, ticket):
    if ticket.get("transport") != "s3-presigned-get" or not re.fullmatch(r"[0-9a-f]{64}", ticket.get("sha256", "")):
        raise ClientError("invalid object storage download receipt")
    expected = {}
    for item in ticket["files"]:
        path = PurePosixPath(item["path"])
        if (not path.parts or path.is_absolute() or ".." in path.parts or "\\" in item["path"]
                or any(p.startswith(".") for p in path.parts) or path.as_posix() in expected):
            raise ClientError("download receipt contains an unsafe or duplicate path")
        expected[path.as_posix()] = item
    out.mkdir(parents=True, exist_ok=False)
    with client.open_object(ticket["url"]) as response, (out / "artifacts.tar").open("xb") as output:
        digest, size = copy_stream(response, output)
    if digest != ticket["sha256"] or size != ticket["bytes"]:
        raise ClientError("artifact archive differs from download receipt")
    downloaded = {}
    with tarfile.open(out / "artifacts.tar", mode="r:") as archive:
        for member in archive:
            if member.isdir():
                continue
            path = PurePosixPath(member.name)
            name = path.as_posix()
            if not member.isfile() or name not in expected or name in downloaded:
                raise ClientError("archive contains an unexpected or duplicate file")
            target = out / "outputs" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as body, target.open("xb") as output:
                sha, count = copy_stream(body, output)
            if count != expected[name]["bytes"] or expected[name].get("sha256", sha) != sha:
                raise ClientError("artifact file differs from download receipt")
            downloaded[name] = {"sha256": sha, "bytes": count}
    if set(downloaded) != set(expected):
        raise ClientError("archive and download receipt file sets differ")
    report = {"project": project, "run_id": run, "attempt_id": attempt,
              "archive_sha256": digest, "archive_bytes": size, "transport": "s3-presigned-get",
              "files": {"outputs/" + name: value for name, value in downloaded.items()}}
    save(out / "verification.json", report)
    return report


def _download_proxy(client: Client, project: str, run: str, attempt: str, out: Path):
    endpoint = f"/api/runs/{segment(project)}/{segment(run)}/attempts/{segment(attempt)}"
    listing = client.call(endpoint + "/files?checksums=true")
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
        if item.get("sha256", digest) != digest:
            raise ClientError("downloaded file SHA256 differs from listing")
        downloaded[path.as_posix()] = {"sha256": digest, "bytes": size}
    try:
        response = client.open(endpoint + "/artifacts/archive")
    except ClientError as error:
        if error.status != 404:
            raise
        expected = {"outputs/" + item["path"].removeprefix("outputs/"): item for item in listing["files"] if item["path"].startswith("outputs/")}
        if not downloaded or set(expected) != set(downloaded) or any(expected[name].get("sha256") != value["sha256"] for name, value in downloaded.items()):
            raise ClientError("archive is unavailable and complete file SHA256 evidence is missing") from error
        report = {"project": project, "run_id": run, "attempt_id": attempt,
                  "archive_sha256": None, "archive_bytes": None, "transport": "http-files",
                  "files": downloaded}
        save(out / "verification.json", report)
        return report
    with response:
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


def upload_asset_parts(client, project, stream, digest, length, state):
    endpoint = f"/api/projects/{segment(project)}/asset-uploads"
    for attempt in range(3):
        try:
            value = client.call(endpoint, data={"sha256": digest, "bytes": length})
            part_bytes = value["part_bytes"]
            if (not 1024 <= part_bytes <= 64 * 1024 ** 2 or value["sha256"] != digest or value["bytes"] != length
                    or value["part_count"] != (length + part_bytes - 1) // part_bytes
                    or not re.fullmatch(r"upload\.[0-9a-f]{64}", value["upload_id"])):
                raise ClientError("invalid multipart upload receipt")
            if value["status"] == "COMPLETED":
                return value["result"]
            save(state, {"project": project, "asset_id": "asset." + digest, "archive_bytes": length,
                         "upload_id": value["upload_id"], "status": "UPLOADING"})
            root = endpoint + "/" + value["upload_id"]
            stream.seek(0)
            for number in range(value["part_count"]):
                data = stream.read(part_bytes)
                sha = hashlib.sha256(data).hexdigest()
                previous = value["parts"].get(str(number))
                if previous:
                    if previous != {"sha256": sha, "bytes": len(data)}:
                        raise ClientError("sealed upload part differs from the local archive")
                    continue
                client.call(root + "/parts/" + str(number) + "?sha256=" + sha, raw=data, method="PUT",
                            timeout=getattr(client, "upload_timeout", 1200))
            return client.call(root + "/complete", data={}, timeout=getattr(client, "upload_timeout", 1200))
        except ClientError as exc:
            if attempt == 2 or not (exc.retryable or exc.status == 429 or exc.status is not None and exc.status >= 500):
                raise
            time.sleep(2 ** attempt)
