"""Fixed ACP CPU entrypoint: verify data-image payload into the shared NAS."""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
from pathlib import Path
import signal
from urllib.parse import urlsplit

try:
    from .data_input import deliver
except ImportError:
    from data_input import deliver


def notify(url, token, value):
    target = urlsplit(url)
    if (target.scheme != "https" or not target.hostname or target.username or target.password
            or target.query or target.fragment
            or not re.fullmatch(r"(?:/[A-Za-z0-9_-]+)*/api/data-copy-transfers/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?", target.path)):
        raise ValueError("invalid data copy callback")
    connection = http.client.HTTPSConnection(target.hostname, target.port or 443, timeout=30)
    try:
        connection.request("PUT", target.path, body=json.dumps(value, sort_keys=True).encode(),
                           headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        response = connection.getresponse()
        response.read(4096)
        if response.status != 200:
            raise ValueError("data copy callback rejected")
    finally:
        connection.close()


def main():
    url, token = os.environ.pop("ML_EXPD_DATA_COPY_URL"), os.environ.pop("ML_EXPD_DATA_COPY_TOKEN")
    root = Path(os.environ["ML_EXPD_DATA_COPY_ROOT"])
    image = os.environ["ML_EXPD_DATA_COPY_IMAGE"]
    seconds = int(os.environ["ML_EXPD_DATA_COPY_SECONDS"])
    if not 5 <= seconds <= 3600 or not root.is_absolute() or ".." in root.parts:
        raise ValueError("invalid data copy budget or destination")
    def expired(_signum, _frame):
        raise TimeoutError("data copy time budget exceeded")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        asset = json.loads(Path("/payload/asset.json").read_text())
        value = {"asset_id": asset["asset_id"], "archive_sha256": asset["sha256"], "image": image,
                 "files_sha256": hashlib.sha256(json.dumps(asset["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                 "data_path": str(root / asset["asset_id"])}
        notify(url, token, {**value, "status": "COPYING"})
        print("ML_EXPD_DATA_COPY=START " + asset["asset_id"], flush=True)
        with Path("/payload/dataset.tar").open("rb") as stream:
            location = deliver(asset, "", root, archive_stream=stream)
        if str(location) != value["data_path"]:
            raise ValueError("data copy destination differs")
        notify(url, token, {**value, "status": "READY"})
        print("ML_EXPD_DATA_COPY=READY " + asset["asset_id"], flush=True)
        return 0
    except Exception as exc:
        try:
            notify(url, token, {"status": "FAILED", "error_class": type(exc).__name__})
        except Exception:
            pass
        print("ML_EXPD_DATA_COPY=FAILED " + type(exc).__name__, flush=True)
        return 65
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
