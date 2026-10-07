#!/usr/bin/env python3
"""Small, stdlib-only entrypoint for a sealed, exact-Attempt launch manifest."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from urllib.parse import urlsplit

CONTRACT = "launcher-manifest.v1"
MAX_BYTES = 256 * 1024
FETCH_SECONDS = 30
LAUNCH_PATH = re.compile(r"^(?:/[A-Za-z0-9][A-Za-z0-9_.-]{0,127})*/api/launch-transfers/([A-Za-z0-9][A-Za-z0-9_.-]{0,127})/([A-Za-z0-9][A-Za-z0-9_.-]{0,127})/(attempt-[0-9]{3,})/([0-9a-f]{64})$")


class LaunchError(ValueError):
    """Fixed public diagnostics, with no raw response or credential values."""


def encode(document):
    """Validate and canonically encode; never include transfer credentials."""
    if not isinstance(document, dict) or set(document) != {
        "contract", "identity", "environment", "argv", "workdir", "timeout_seconds"
    } or document["contract"] != CONTRACT:
        raise LaunchError("invalid launch manifest contract")
    identity = document["identity"]
    if (not isinstance(identity, dict) or set(identity) != {"project", "run_id", "attempt_id"}
            or any(not isinstance(value, str) for value in identity.values())
            or not LAUNCH_PATH.fullmatch("/api/launch-transfers/" + "/".join(
                str(identity.get(key, "")) for key in ("project", "run_id", "attempt_id")) + "/" + "0" * 64)):
        raise LaunchError("invalid launch manifest identity")
    environment = document["environment"]
    if not isinstance(environment, dict) or any(
        not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
        or not isinstance(value, str) or "\x00" in value
        or key in {"ML_EXPD_UPLOAD_TOKEN", "ML_EXPD_BOOTSTRAP_TOKEN"}
        for key, value in environment.items()
    ) or not environment.get("OUTPUT_DIR", "").startswith("/"):
        raise LaunchError("invalid launch manifest environment")
    argv = document["argv"]
    if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or "\x00" in arg for arg in argv) or not argv[0]:
        raise LaunchError("invalid launch manifest argv")
    if not isinstance(document["workdir"], str) or not document["workdir"].startswith("/") or "\x00" in document["workdir"]:
        raise LaunchError("invalid launch manifest workdir")
    seconds = document["timeout_seconds"]
    if type(seconds) is not int or not 1 <= seconds < 1000 * 3600:
        raise LaunchError("invalid launch manifest timeout")
    body = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    if len(body) > MAX_BYTES:
        raise LaunchError("launch manifest exceeds size limit")
    return body


def endpoint(url):
    target = urlsplit(url)
    match = LAUNCH_PATH.fullmatch(target.path)
    if (target.scheme != "https" or not target.hostname or target.username or target.password
            or target.query or target.fragment or not match):
        raise LaunchError("launch manifest requires an exact HTTPS endpoint")
    return target, match


def fetch(url, token):
    target, match = endpoint(url)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", token):
        raise LaunchError("invalid launch capability")
    connection = http.client.HTTPSConnection(target.hostname, target.port or 443, timeout=FETCH_SECONDS)
    deadline = time.monotonic() + FETCH_SECONDS
    try:
        connection.request("GET", target.path, headers={"Authorization": "Bearer " + token})
        socket = connection.sock
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("launch manifest download timed out")
        socket.settimeout(remaining)
        response = connection.getresponse()
        if response.status != 200:
            raise LaunchError("launch manifest request rejected")
        body = bytearray()
        while not response.isclosed():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("launch manifest download timed out")
            socket.settimeout(remaining)
            chunk = response.read1(min(64 * 1024, MAX_BYTES + 1 - len(body)))
            if not chunk:
                break
            body.extend(chunk)
            if len(body) > MAX_BYTES:
                raise LaunchError("launch manifest exceeds size limit")
        if hashlib.sha256(body).hexdigest() != match[4]:
            raise LaunchError("launch manifest checksum differs")
        document = json.loads(body)
        if encode(document) != body or document["identity"] != dict(zip(
                ("project", "run_id", "attempt_id"), match.groups()[:3])):
            raise LaunchError("launch manifest identity or encoding differs")
        return document
    finally:
        connection.close()


def execute(document, token, *, worker="/usr/local/lib/ml-expd/worker.py"):
    encode(document)
    environment = dict(os.environ)
    environment.pop("ML_EXPD_BOOTSTRAP_TOKEN", None)
    environment.update(document["environment"])
    environment["ML_EXPD_UPLOAD_TOKEN"] = token
    Path(environment["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    child = subprocess.Popen([
        "timeout", "--signal=TERM", "--kill-after=30s", str(document["timeout_seconds"]) + "s",
        sys.executable, worker, *document["argv"]
    ], cwd=document["workdir"], env=environment, start_new_session=True)
    def forward(signum, _frame):
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass
    previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        return child.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-url", required=True)
    args = parser.parse_args(argv)
    token = os.environ.pop("ML_EXPD_BOOTSTRAP_TOKEN", "")
    try:
        document = fetch(args.manifest_url, token)
        identity = document["identity"]
        print("ML_EXPD_LAUNCH READY " + identity["run_id"] + "/" + identity["attempt_id"], flush=True)
        return execute(document, token)
    except (ValueError, OSError) as exc:
        # Neither HTTP response bodies nor URLs/capabilities enter logs.
        print("ML_EXPD_LAUNCH FAILED " + (str(exc) if isinstance(exc, LaunchError) else type(exc).__name__), file=sys.stderr, flush=True)
        return 78


if __name__ == "__main__":
    raise SystemExit(main())
