"""Verified, dependency-free updates for a standalone client environment."""
from __future__ import annotations

from email.parser import BytesParser
from hashlib import sha256
from importlib import metadata, util
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zipfile import BadZipFile, ZipFile
import zlib

from .api import ClientError


REPOSITORY = "NotDesigned/ml-experiment-control"
CLIENT = "ml-experiment-client"
VERSION = r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
WHEEL = re.compile(r"ml_experiment_client-(" + VERSION + r")-py3-none-any\.whl\Z")
METADATA_LIMIT = 1024 * 1024
WHEEL_LIMIT = 8 * 1024 * 1024


def _parts(value):
    if not isinstance(value, str) or not re.fullmatch(VERSION, value):
        raise ClientError("client updates require stable N.N.N versions")
    return tuple(int(part) for part in value.split("."))


def _installed(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _github_url(url):
    target = urlsplit(url)
    return (target.scheme == "https" and target.hostname in {
        "api.github.com", "github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"}
        and not target.username and not target.password and target.port in (None, 443))


class _GithubRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _github_url(newurl):
            raise ClientError("official release download redirected outside GitHub HTTPS")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download(url, limit):
    """No service Client, API token, credential file or authentication headers."""
    try:
        if not _github_url(url):
            raise ClientError("invalid official release download address")
        accept = "application/vnd.github+json" if urlsplit(url).hostname == "api.github.com" else "application/octet-stream"
        request = Request(url, headers={"User-Agent": "ml-expd-client-updater", "Accept": accept})
        with build_opener(_GithubRedirect()).open(request, timeout=30) as response:
            if not _github_url(response.geturl()):
                raise ClientError("invalid official release download address")
            body = response.read(limit + 1)
        if len(body) > limit:
            raise ClientError("official release download exceeds its size limit")
        return body
    except ClientError:
        raise
    except (OSError, ValueError):
        raise ClientError("cannot download the official release; check GitHub connectivity and retry explicitly") from None


def _release(release_version):
    if release_version is not None:
        _parts(release_version)
    suffix = "latest" if release_version is None else "tags/v" + release_version
    try:
        release = json.loads(_download(f"https://api.github.com/repos/{REPOSITORY}/releases/{suffix}", METADATA_LIMIT))
    except (ValueError, UnicodeError):
        raise ClientError("official release metadata is invalid") from None
    if (not isinstance(release, dict) or release.get("draft") is not False or release.get("prerelease") is not False
            or not isinstance(release.get("tag_name"), str) or not re.fullmatch("v" + VERSION, release["tag_name"])
            or release_version is not None and release["tag_name"] != "v" + release_version
            or not isinstance(release.get("assets"), list)):
        raise ClientError("official release must be a stable published release")
    assets = release["assets"]
    wheels = [asset for asset in assets if isinstance(asset, dict) and WHEEL.fullmatch(str(asset.get("name", "")))]
    sums = [asset for asset in assets if isinstance(asset, dict) and asset.get("name") == "SHA256SUMS"]
    if len(wheels) != 1 or len(sums) != 1:
        raise ClientError("official release must contain one standalone client wheel and SHA256SUMS")
    for asset in (wheels[0], sums[0]):
        expected = f"https://github.com/{REPOSITORY}/releases/download/{release['tag_name']}/{asset['name']}"
        if asset.get("browser_download_url") != expected:
            raise ClientError("release asset does not belong to the official repository")
    return release["tag_name"][1:], wheels[0], sums[0]


def _verify(wheel, name, target, checksums):
    try:
        rows = [re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
                for line in checksums.decode("ascii").splitlines() if line]
        if not rows or any(row is None for row in rows):
            raise ValueError
        matches = [row.group(1).lower() for row in rows if row.group(2) in (name, "./" + name)]
        if len(matches) != 1 or sha256(wheel).hexdigest() != matches[0]:
            raise ValueError
    except (ValueError, UnicodeError):
        raise ClientError("client wheel SHA256 verification failed; nothing was installed") from None
    try:
        with ZipFile(BytesIO(wheel)) as archive:
            entries = archive.infolist()
            identities = [entry for entry in entries if entry.filename.endswith(".dist-info/METADATA")]
            if (len(identities) != 1 or identities[0].filename != f"ml_experiment_client-{target}.dist-info/METADATA"
                    or identities[0].file_size > METADATA_LIMIT or len(entries) > 512
                    or sum(entry.file_size for entry in entries) > 32 * METADATA_LIMIT):
                raise ValueError
            identity = BytesParser().parsebytes(archive.read(identities[0]), headersonly=True)
            if identity.get_all("Name") != [CLIENT] or identity.get_all("Version") != [target]:
                raise ValueError
    except (BadZipFile, KeyError, OSError, ValueError, RuntimeError, EOFError, zlib.error):
        raise ClientError("downloaded wheel has an invalid client package identity; nothing was installed") from None


def _installer():
    if util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "--isolated", "--disable-pip-version-check", "--no-input",
                "install", "--no-deps", "--no-index", "--no-cache-dir"]
    uv = shutil.which("uv")
    if uv:
        return [uv, "--no-config", "--offline", "--no-cache", "pip", "install", "--python", sys.executable,
                "--no-deps", "--no-index"]
    raise ClientError("no installer is available; install pip or uv in the standalone client environment, then retry")


def update_client(version=None, check=False):
    """`version` selects an official release; its client wheel has its own version."""
    current = _installed(CLIENT)
    current_parts = _parts(current)
    release_version, asset, checksums = _release(version)
    target = WHEEL.fullmatch(asset["name"]).group(1)
    target_parts = _parts(target)
    result = {"current_version": current, "target_version": target, "release_version": release_version,
              "update_available": target_parts > current_parts, "check_only": bool(check), "updated": False}
    if check or target_parts == current_parts:
        return result
    if target_parts < current_parts:
        raise ClientError("refusing to downgrade the client; the selected release contains an older client wheel")
    if _installed("ml-experiment-server") is not None:
        raise ClientError("this environment contains ml-experiment-server; use the server deployment process or a separate client environment")
    command = _installer()
    content = _download(asset["browser_download_url"], WHEEL_LIMIT)
    _verify(content, asset["name"], target, _download(checksums["browser_download_url"], METADATA_LIMIT))
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("PIP_", "UV_", "ML_EXPD_")) and key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}}
    environment["PIP_CONFIG_FILE"] = os.devnull
    try:
        with TemporaryDirectory(prefix="ml-exp-client-update-") as directory:
            path = Path(directory) / asset["name"]
            path.write_bytes(content)
            completed = subprocess.run(command + [str(path)], check=False, capture_output=True,
                                       timeout=300, env=environment)
            if completed.returncode != 0:
                raise ClientError("client installation failed; inspect the standalone environment before retrying; installer output was withheld")
    except (OSError, subprocess.SubprocessError):
        raise ClientError("client installation could not finish; inspect the standalone environment before retrying") from None
    if _installed(CLIENT) != target:
        raise ClientError("client installer finished but version verification failed; inspect the standalone environment")
    return {**result, "updated": True, "installed_version": target}
