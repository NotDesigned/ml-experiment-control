"""Bounded CPU-only recovery of an exact Attempt's persistent outputs."""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import tempfile

if __package__:
    from .worker_artifacts import archive_outputs, upload_parts
    from .persistent_state import restore, open_file
    from .worker_http import https_connection, error_code
else:
    from artifacts import archive_outputs, upload_parts
    from persistent_state import restore, open_file
    from worker_http import https_connection, error_code

from urllib.parse import urlsplit


def sha(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def training_result(root, url, token, code):
    """Persist the process result before attempting any expensive transfer."""
    value = {"contract": "training-result.v1", "project": os.environ["PROJECT_NAME"],
             "run_id": os.environ["RUN_ID"], "attempt_id": os.environ["ATTEMPT_ID"],
             "exit_code": code}
    path = root.parent / "training-result.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True))
    temporary.replace(path)
    target = urlsplit(url.replace("/artifact-transfers/", "/result-ready-transfers/", 1))
    connection = https_connection(target, timeout=30)
    try:
        connection.request("PUT", target.path, body=json.dumps(value).encode(),
                           headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError("training result registration rejected")
        response.read(256 * 1024)
    finally:
        connection.close()


def checked_directory(root):
    current = Path("/")
    if not root.is_absolute():
        raise ValueError("recovery path must be absolute")
    for part in root.parts[1:]:
        current /= part
        if not stat.S_ISDIR(current.lstat().st_mode):
            raise ValueError("recovery directory is a link or special file")


def recover(value):
    outputs = Path(value["outputs_root"])
    checked_directory(outputs)
    checkpoint = value.get("checkpoint")
    if checkpoint is not None:
        checked_directory(Path(checkpoint["state_root"]))
        restore(checkpoint)
        # Check matching exports against immutable state, without imposing any
        # project-specific filename, metric or scientific completion rule.
        for item in checkpoint["files"]:
            relative = "/".join(item["path"].split("/")[2:])
            if any(fnmatch.fnmatch(relative, p) or p.startswith("**/") and fnmatch.fnmatch(relative, p[3:])
                   for p in value["patterns"]) and (outputs / relative).exists():
                with open_file(outputs, relative) as stream:
                    if os.fstat(stream.fileno()).st_size != item["bytes"] or sha(stream) != item["sha256"]:
                        raise ValueError("output differs from registered checkpoint")
    with tempfile.TemporaryFile(dir=outputs.parent) as stream:
        count = archive_outputs(outputs, stream, value["limit"], value["patterns"])
        if not count:
            raise ValueError("Attempt has no declared outputs")
        length = stream.tell()
        stream.seek(0)
        checksum = sha(stream)
        expected = value.get("expected_archive")
        if expected is not None and (checksum, length) != (expected["sha256"], expected["bytes"]):
            raise ValueError("recovered outputs differ from unfinished upload")
        print("ML_EXPD_RESULT_RECOVERY=VERIFIED", flush=True)
        upload_parts(value["url"], value["token"], stream, length)
    print("ML_EXPD_RESULT_RECOVERY=COMPLETE", flush=True)


def main(value):
    def expired(_signal, _frame):
        raise TimeoutError("result recovery exceeded its fixed budget")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(value["seconds"])
    try:
        recover(value)
        return 0
    except Exception as error:
        print("ML_EXPD_RESULT_RECOVERY=FAILED error=" + error_code(error), flush=True)
        return 74
    finally:
        signal.alarm(0)
