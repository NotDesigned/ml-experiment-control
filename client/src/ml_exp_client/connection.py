"""Private local connection settings; constructing these helpers sends no HTTP."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat

from .api import ClientError, validate_base_url
from .state import save


MAX_CONFIG_BYTES = 32 * 1024
MAX_TOKEN_BYTES = 8192


def config_path(explicit=None):
    if explicit is not None:
        return Path(explicit).expanduser()
    configured = os.environ.get("ML_EXPD_CLIENT_CONFIG")
    if configured:
        return Path(configured).expanduser()
    base = Path(os.environ["XDG_CONFIG_HOME"]) if os.environ.get("XDG_CONFIG_HOME") else Path.home() / ".config"
    return base.expanduser() / "ml-expd" / "client.json"


def normalize_url(value):
    try:
        if (not isinstance(value, str) or len(value) > MAX_TOKEN_BYTES
                or any(char.isspace() or not char.isprintable() for char in value)):
            raise ValueError
        return validate_base_url(value)
    except (ClientError, TypeError, ValueError):
        raise ClientError("invalid API URL; use HTTPS (HTTP is allowed on loopback only)") from None


def validate_token(value):
    if (not isinstance(value, str) or not value or len(value) > MAX_TOKEN_BYTES
            or any(not 33 <= ord(char) <= 126 for char in value)):
        raise ClientError("invalid API token; use a nonempty token without whitespace")
    return value


def _private_file(metadata):
    if not stat.S_ISREG(metadata.st_mode) or os.name == "posix" and metadata.st_mode & 0o077:
        raise ValueError


def _contents(path, limit, *, private=False):
    path = Path(path).expanduser()
    if private and path.is_symlink():
        raise ValueError
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    if private:
        flags |= getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if private:
            _private_file(metadata)
        elif not stat.S_ISREG(metadata.st_mode):
            raise ValueError
        value = stream.read(limit + 1)
        if len(value) > limit:
            raise ValueError
        return value.decode("utf-8")


def read_token_file(path):
    try:
        return validate_token(_contents(path, MAX_TOKEN_BYTES).strip())
    except (OSError, ValueError, TypeError, ClientError):
        raise ClientError("cannot read API token file; use a readable regular file containing one valid token") from None


def _settings(value):
    if (not isinstance(value, dict) or set(value) - {"url", "token", "token_file"}
            or any(not isinstance(item, str) or not item for item in value.values())
            or "token" in value and "token_file" in value
            or ("token" in value or "token_file" in value) and "url" not in value):
        raise ValueError
    result = dict(value)
    if "url" in result:
        result["url"] = normalize_url(result["url"])
    if "token" in result:
        result["token"] = validate_token(result["token"])
    if "token_file" in result and (len(result["token_file"]) > MAX_TOKEN_BYTES
                                   or not result["token_file"].isprintable()):
        raise ValueError
    return result


def read_config(path):
    try:
        return _settings(json.loads(_contents(path, MAX_CONFIG_BYTES, private=True)))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, TypeError, ClientError):
        raise ClientError("cannot read private client configuration; use a regular private JSON file with valid connection settings") from None


def write_config(path, value):
    try:
        path, value = Path(path).expanduser(), _settings(value)
        if "token_file" in value:
            value["token_file"] = str(Path(value["token_file"]).expanduser().absolute())
        if len(json.dumps(value).encode()) > MAX_CONFIG_BYTES:
            raise ValueError
        if path.exists() or path.is_symlink():
            _private_file(path.lstat())
        missing, parent = [], path.parent
        while not parent.exists():
            missing.append(parent)
            parent = parent.parent
        for directory in reversed(missing):
            directory.mkdir(mode=0o700)
        save(path, value)
    except (OSError, ValueError, TypeError, ClientError):
        raise ClientError("cannot write private client configuration; check the path, permissions and connection settings") from None


def public_config(path, value):
    return {"config_file": str(Path(path).expanduser().absolute()), "url": value.get("url"),
            "token_configured": bool(value.get("token") or value.get("token_file")),
            "token_file": value.get("token_file")}


def resolve_connection(url, settings):
    selected = normalize_url(url) if url else ""
    token, token_file = os.environ.get("ML_EXPD_API_TOKEN", ""), os.environ.get("ML_EXPD_API_TOKEN_FILE", "")
    if token and token_file:
        raise ClientError("set only one of ML_EXPD_API_TOKEN_FILE and ML_EXPD_API_TOKEN")
    if token or token_file:
        return selected, read_token_file(token_file) if token_file else validate_token(token)
    if settings.get("token") or settings.get("token_file"):
        if not selected or selected != normalize_url(settings.get("url")):
            raise ClientError("saved credentials belong to a different API URL; configure credentials for the selected server")
        return selected, read_token_file(settings["token_file"]) if settings.get("token_file") else validate_token(settings["token"])
    return selected, ""
