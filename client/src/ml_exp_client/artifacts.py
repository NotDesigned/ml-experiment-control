"""Verified exact-Attempt downloads and durable local acknowledgement."""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import tarfile
from urllib.parse import quote

from .api import Client, ClientError, segment
from .archives import copy_stream
from .state import save


def download(client: Client, project: str, run: str, attempt: str, out: Path, *, keep_server_copy=False):
    endpoint = f"/api/runs/{segment(project)}/{segment(run)}/attempts/{segment(attempt)}"
    try:
        ticket = client.call(endpoint + "/artifacts/download")
    except ClientError as exc:
        if exc.status != 404:
            raise
        report = _download_proxy(client, project, run, attempt, out)
        if "artifact-retention.v1" in getattr(client, "capabilities", []) and report.get("archive_sha256"):
            return _acknowledge_report(client, out, report, keep_server_copy=keep_server_copy)
        return report
    report = _download_object(client, project, run, attempt, out, ticket)
    if ticket.get("acknowledgement", {}).get("contract") == "artifact-retention.v1":
        return _acknowledge_report(client, out, report, keep_server_copy=keep_server_copy)
    return report


def _sync_download(out):
    # Confirm durable local bytes before granting remote deletion. Windows does
    # not expose POSIX directory fsync, but file fsync is still required.
    for path in out.rglob("*"):
        if path.is_file():
            with path.open("r+b") as stream:
                os.fsync(stream.fileno())
    if os.name != "nt":
        for directory in [*reversed(sorted(p for p in out.rglob("*") if p.is_dir())), out]:
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def _acknowledge_report(client, out, report, *, keep_server_copy=False):
    _sync_download(out)
    report["acknowledgement"] = {"status": "PENDING", "keep_server_copy": keep_server_copy}
    save(out / "verification.json", report)
    endpoint = f"/api/runs/{segment(report['project'])}/{segment(report['run_id'])}/attempts/{segment(report['attempt_id'])}/artifacts/ack"
    proof = {k: report[k] for k in ("archive_sha256", "archive_bytes", "files")}
    try:
        report["acknowledgement"] = client.call(endpoint, data={**proof, "release": not keep_server_copy})
    except ClientError as error:
        # Local results remain usable. Retry only the acknowledgement, never
        # redownload or replay training because its response was lost.
        report["acknowledgement"].update(http_status=error.status, retry_command=shlex.join(
            ["ml-exp", "run", "acknowledge", "--directory", str(out)]))
    save(out / "verification.json", report)
    return report


def acknowledge(client, out: Path, *, keep_server_copy=None):
    """Revalidate saved local results and retry only their exact acknowledgement."""
    report = json.loads((out / "verification.json").read_text())
    if keep_server_copy is None:
        previous = report.get("acknowledgement", {})
        keep_server_copy = previous.get("keep_server_copy", previous.get("status") == "KEEP")
    if not report.get("archive_sha256") or not report.get("archive_bytes"):
        raise ClientError("archive proof is unavailable; keep the server copy")
    archive = out / "artifacts.tar"
    if archive.is_symlink():
        raise ClientError("downloaded archive must be a regular local file")
    with archive.open("rb") as stream:
        if copy_stream(stream) != (report["archive_sha256"], report["archive_bytes"]):
            raise ClientError("saved archive differs from verification report")
    for name, expected in report["files"].items():
        relative = PurePosixPath(name)
        if not relative.parts or relative.parts[0] != "outputs" or ".." in relative.parts or "\\" in name:
            raise ClientError("saved verification path is invalid")
        file = out / relative
        if file.is_symlink() or not file.resolve().is_relative_to(out.resolve()):
            raise ClientError("saved verification path is outside the download")
        with file.open("rb") as stream:
            if copy_stream(stream) != (expected["sha256"], expected["bytes"]):
                raise ClientError("saved file differs from verification report")
    return _acknowledge_report(client, out, report, keep_server_copy=keep_server_copy)


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
