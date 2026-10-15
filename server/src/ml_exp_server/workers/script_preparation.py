"""CPU-only script wrapper: verify a NAS receipt and report it independently of logs.

Executed inline by the scheduler; use only stdlib and immutable image helpers.
"""
from __future__ import annotations

import json
import http.client
import os
from pathlib import Path
import stat
import sys
import time
from urllib.parse import urlsplit


TOKEN_ENV = "ML_EXPD_SCRIPT_RECEIPT_TOKEN"
REQUEST_ENV = "ML_EXPD_SCRIPT_RECEIPT_REQUEST"
LIMIT = 8 * 1024 ** 2


def sealed_receipt(tree, definition):
    from data_preparation import inventory, identity
    path = tree.parent / "receipt.json"
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > LIMIT or before.st_mode & 0o222:
            raise ValueError("DATA_SCRIPT_RECEIPT_NOT_SEALED")
        raw = stream.read(LIMIT + 1)
        after = os.fstat(stream.fileno())
    if len(raw) > LIMIT or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("DATA_SCRIPT_RECEIPT_CHANGED")
    value = json.loads(raw)
    files = inventory(tree)
    if (value.get("schema_version") != 1 or value.get("status") != "READY"
            or value.get("definition") != definition
            or value.get("preparation_id") != definition["preparation_id"]
            or value.get("files") != files or value.get("content_sha256") != identity(files)
            or value.get("dataset_id") != "dataset." + identity(files)
            or value.get("bytes") != sum(item["bytes"] for item in files)):
        raise ValueError("DATA_SCRIPT_RECEIPT_IDENTITY")
    return value


def report(request, token, receipt):
    from worker_http import https_connection
    target = urlsplit(request["url"])
    if target.scheme != "https" or not target.hostname or target.username or target.password or target.query or target.fragment:
        raise ValueError("DATA_SCRIPT_CALLBACK_ORIGIN")
    payload = {key: value for key, value in request.items() if key != "url"}
    payload["receipt"] = receipt
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    for delay in (0, 1, 2, 4, 8):
        time.sleep(delay)
        connection = https_connection(target, timeout=5)
        try:
            connection.request("PUT", target.path, body=raw, headers={
                "Authorization": "Bearer " + token, "Content-Type": "application/json",
                "Content-Length": str(len(raw)),
            })
            response = connection.getresponse()
            body = response.read(8193)
            if response.status == 200 and len(body) <= 8192:
                reply = json.loads(body)
                if (isinstance(reply, dict) and reply.get("operation_id") == request["operation_id"]
                        and reply.get("preparation_id") == receipt["preparation_id"]
                        and reply.get("status") in {"READY", "PENDING_UID"}):
                    return
        except (OSError, ValueError, http.client.HTTPException):
            pass
        finally:
            connection.close()
    raise ValueError("DATA_SCRIPT_CALLBACK_PENDING")


def main(argv=None):
    # Remove both capabilities before importing or executing project code.
    token = os.environ.pop(TOKEN_ENV)
    request = json.loads(os.environ.pop(REQUEST_ENV))
    definition = json.loads((sys.argv if argv is None else argv)[1])
    sys.path.insert(0, "/usr/local/lib/ml-expd")
    from data_preparation import prepare
    from data_input import deliver
    from worker import link_path
    for item in definition["definition"]["inputs"]:
        link_path(deliver({**item, "require_cached": True}, "", Path(definition["cache"]).parent / "data-assets"), Path(item["mount_path"]))
    tree, _ = prepare(definition["definition"], Path(definition["cache"]))
    receipt = sealed_receipt(tree, definition["definition"])
    report(request, token, receipt)
    print("ML_EXPD_SCRIPT_RECEIPT=ACCEPTED", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("ML_EXPD_SCRIPT_RECEIPT=FAILED", flush=True)
        raise SystemExit(74) from None
