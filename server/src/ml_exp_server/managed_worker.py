"""Stdlib-only data delivery and explicit, immutable checkpoint publication."""
from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
from pathlib import Path, PurePosixPath
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.parse import urlsplit

if __package__:
    from .worker_artifacts import archive_outputs, upload_parts as upload
    from .data_input import digest_stream, relative_path, verify_tree, fetch, deliver, DeliveryTrace
    from .data_preparation import prepare as prepare_data
    from . import persistent_state
    from .worker_http import error_code
else:
    from artifacts import archive_outputs, upload_parts as upload
    from data_input import digest_stream, relative_path, verify_tree, fetch, deliver, DeliveryTrace
    from data_preparation import prepare as prepare_data
    import persistent_state
    from worker_http import error_code


def link_path(target, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and path.exists() and path.samefile(target):
        return
    if path.is_symlink():
        raise ValueError("container path is already bound to another location")
    path.symlink_to(target, target_is_directory=True)


def open_output(root, name):
    parts = relative_path(name).parts
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
        if not stat.S_ISREG(os.fstat(file).st_mode):
            os.close(file)
            raise ValueError("checkpoint must be a regular file")
        return os.fdopen(file, "rb")
    finally:
        os.close(descriptor)


def read_checkpoint_ready(root):
    with open_output(root, "checkpoint.ready.json") as body:
        ready = body.read(256 * 1024 + 1)
    if len(ready) > 256 * 1024:
        raise ValueError("checkpoint manifest is invalid")
    return ready


def checkpoint_archive(root, stream, limit=0):
    ready = read_checkpoint_ready(root)
    document = json.loads(ready)
    files = document["files"]
    if not files or len(files) > 20000:
        raise ValueError("checkpoint manifest file count is invalid")
    seen = set()
    total = 0
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for item in files:
            name = relative_path(item["path"]).as_posix()
            if name in seen or name == "checkpoint.ready.json":
                raise ValueError("checkpoint manifest contains duplicate/reserved paths")
            seen.add(name)
            total += int(item["bytes"])
            if limit and total > limit - 1024 * 1024:
                raise ValueError("checkpoint archive exceeds upload limit")
            with open_output(root, name) as body:
                before = os.fstat(body.fileno())
                digest, size = digest_stream(body)
                if not stat.S_ISREG(before.st_mode) or digest != item["sha256"] or size != item["bytes"]:
                    raise ValueError("checkpoint file differs from ready manifest")
                body.seek(0)
                info = tarfile.TarInfo(name)
                info.size, info.mode = size, 0o400
                archive.addfile(info, body)
                after = os.fstat(body.fileno())
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    raise ValueError("checkpoint file changed during publication")
        info = tarfile.TarInfo("checkpoint.ready.json")
        info.size, info.mode = len(ready), 0o400
        archive.addfile(info, io.BytesIO(ready))
    return hashlib.sha256(ready).hexdigest()


def cache_checkpoint(stream, cache):
    """Seed the same verified asset cache before publishing the upload receipt."""
    stream.seek(0)
    digest, length = digest_stream(stream)
    stream.seek(0)
    with tarfile.open(fileobj=stream, mode="r:") as archive:
        ready = archive.extractfile("checkpoint.ready.json").read()
    files = json.loads(ready)["files"] + [{"path": "checkpoint.ready.json", "bytes": len(ready),
                                         "sha256": hashlib.sha256(ready).hexdigest()}]
    item = {"asset_id": "asset." + digest, "sha256": digest, "archive_bytes": length, "files": files}
    return deliver(item, None, cache, archive_stream=stream)


def main(argv=None):
    url = os.environ.pop("ML_EXPD_UPLOAD_URL")
    token = os.environ.pop("ML_EXPD_UPLOAD_TOKEN")
    limit = int(os.environ.pop("ML_EXPD_UPLOAD_LIMIT"))
    patterns = json.loads(os.environ.pop("ML_EXPD_OUTPUT_PATTERNS", '["**/*"]'))
    inputs = json.loads(os.environ.pop("ML_EXPD_INPUT_ASSETS", "[]"))
    snapshot_url = os.environ.pop("ML_EXPD_SNAPSHOT_URL", "")
    interval = int(os.environ.pop("ML_EXPD_SNAPSHOT_INTERVAL", "60"))
    preparation = json.loads(os.environ.pop("ML_EXPD_DATA_PREPARATION", "null"))
    preparation_cached = os.environ.pop("ML_EXPD_DATA_PREPARATION_CACHED", "") == "1"
    state_context = json.loads(os.environ.pop("ML_EXPD_CHECKPOINT_STATE", "null"))
    state_url = os.environ.pop("ML_EXPD_CHECKPOINT_STATE_URL", "")
    state_interval = int(os.environ.pop("ML_EXPD_CHECKPOINT_STATE_INTERVAL", "60"))
    checkpoint_restore = json.loads(os.environ.pop("ML_EXPD_CHECKPOINT_RESTORE", "null"))
    root = Path(os.environ["OUTPUT_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    trace = DeliveryTrace()
    trace.report("INITIALIZING")
    try:
        link_path(root, Path("/outputs"))
        if state_context is not None:
            state_root = Path(state_context["state_root"])
            if state_root != root.parent / "state" or state_root.is_symlink():
                raise ValueError("state directory differs from the Attempt")
            state_root.mkdir(parents=True, exist_ok=True)
            os.environ["STATE_DIR"] = str(state_root)
            patterns = [*patterns, "checkpoints.json"]
        if checkpoint_restore is not None:
            location = persistent_state.restore(checkpoint_restore)
            link_path(location, Path("/inputs/resume"))
            os.environ["RESUME_DIR"] = "/inputs/resume"
            print("ML_EXPD_CHECKPOINT_RESTORE=READY " + checkpoint_restore["checkpoint_id"], flush=True)
        for item in inputs:
            cache = root.parents[4] / "data-assets"
            print("ML_EXPD_INPUT_ASSET=START " + item["asset_id"] + " archive_bytes=" + str(item["archive_bytes"]), flush=True)
            trace = DeliveryTrace(item["asset_id"], item["archive_bytes"])
            location = deliver(item, token, cache, trace=trace)
            trace.report("MOUNTING")
            link_path(location, Path(item["mount_path"]))
            trace.report("READY")
            print("ML_EXPD_INPUT_ASSET=READY " + item["asset_id"], flush=True)
    except Exception as error:
        trace.failed(error)
        print("ML_EXPD_INPUT_DELIVERY=FAILED", file=sys.stderr, flush=True)
        return 65
    print("ML_EXPD_INPUT_DELIVERY_SECONDS=" + str(round(time.monotonic() - started, 3)), flush=True)
    if preparation is not None:
        try:
            location, receipt = prepare_data(preparation, root.parents[4] / "data-preparations",
                                            require_cached=preparation_cached)
            os.environ["DATA_DIR"] = str(location)
        except Exception as error:
            message = str(error)
            receipt = {"status": "FAILED", "definition": preparation,
                       "error": message if message.startswith("DATA_") and len(message) < 128 else type(error).__name__}
            print("ML_EXPD_DATA_PREPARATION=FAILED", file=sys.stderr, flush=True)
        (root / "data-preparation.json").write_text(json.dumps(receipt, sort_keys=True))
        patterns = [*patterns, "data-preparation.json"]
        if receipt["status"] != "READY":
            try:
                with tempfile.TemporaryFile(dir=root.parent) as stream:
                    archive_outputs(root, stream, max(1, limit - 1024 * 1024) if limit else 0, ["data-preparation.json"])
                    upload(url, token, stream, stream.tell())
            except Exception as error:
                print("ML_EXPD_ARTIFACT_UPLOAD=FAILED error=" + error_code(error), file=sys.stderr, flush=True)
            return 65
    child = subprocess.Popen(argv or sys.argv[1:], start_new_session=True)
    def forward(signum, _frame):
        if child.poll() is None:
            os.killpg(child.pid, signum)
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    previous, last_poll, state_previous, state_last_poll = None, 0.0, None, 0.0
    state_receipts = []
    if state_context is not None:
        (root / "checkpoints.json").write_text(json.dumps({"checkpoints": [], "resume_from": checkpoint_restore}))
    while True:
        code = child.poll()
        if state_context is not None and (code is not None or time.monotonic() - state_last_poll >= state_interval):
            state_last_poll = time.monotonic()
            if (state_root / "checkpoint.ready.json").exists():
                try:
                    ready = persistent_state.read_ready(state_root)
                    marker = persistent_state.digest(ready)
                    if marker != state_previous:
                        receipt = persistent_state.publish(state_context, state_url, token)
                        state_receipts.append(receipt)
                        (root / "checkpoints.json").write_text(json.dumps({"checkpoints": state_receipts, "resume_from": checkpoint_restore}))
                        state_previous = marker
                        print("ML_EXPD_CHECKPOINT_STATE=REGISTERED " + receipt["checkpoint_id"], flush=True)
                except Exception as error:
                    print("ML_EXPD_CHECKPOINT_STATE=REGISTRATION_FAILED error=" + error_code(error), file=sys.stderr, flush=True)
        if snapshot_url and (code is not None or time.monotonic() - last_poll >= interval):
            last_poll = time.monotonic()
            checkpoint_root = state_root if state_context is not None else root
            if (checkpoint_root / "checkpoint.ready.json").exists():
                try:
                    marker = hashlib.sha256(read_checkpoint_ready(checkpoint_root)).hexdigest()
                    if marker != previous:
                        with tempfile.TemporaryFile(dir=root.parent) as stream:
                            if checkpoint_archive(checkpoint_root, stream, limit) != marker:
                                raise ValueError("checkpoint publication marker changed")
                            length = stream.tell()
                            if length > limit:
                                raise ValueError("checkpoint archive exceeds upload limit")
                            try:
                                cache_checkpoint(stream, root.parents[4] / "data-assets")
                                print("ML_EXPD_CHECKPOINT_CACHE=READY", flush=True)
                            except Exception:
                                # A cache is optional. Upload remains authoritative.
                                print("ML_EXPD_CHECKPOINT_CACHE=FAILED", file=sys.stderr, flush=True)
                            upload(snapshot_url, token, stream, length)
                            previous = marker
                            print("ML_EXPD_CHECKPOINT_UPLOAD=COMPLETE", flush=True)
                except Exception as error:
                    print("ML_EXPD_CHECKPOINT_UPLOAD=FAILED error=" + error_code(error), file=sys.stderr, flush=True)
        if code is not None:
            break
        time.sleep(1)
    try:
        with tempfile.TemporaryFile(dir=root.parent) as stream:
            archive_outputs(root, stream, max(1, limit - 1024 * 1024) if limit else 0, patterns)
            length = stream.tell()
            if length > limit:
                raise ValueError("artifact archive limit exceeded")
            upload(url, token, stream, length)
        print("ML_EXPD_ARTIFACT_UPLOAD=COMPLETE", flush=True)
    except Exception as error:
        print("ML_EXPD_ARTIFACT_UPLOAD=FAILED error=" + error_code(error), file=sys.stderr, flush=True)
        return code if code > 0 else 74
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    raise SystemExit(main())
