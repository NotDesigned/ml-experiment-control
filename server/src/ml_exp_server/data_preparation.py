"""Backend-local script preparation, atomic receipts and verified shared caches.

This module is also copied into managed images; use only the standard library.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import tempfile


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_record(path, name):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("DATA_FILE_NOT_REGULAR")
        sha = hashlib.sha256()
        while chunk := stream.read(1024 * 1024):
            sha.update(chunk)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("DATA_FILE_CHANGED")
    return {"path": name, "bytes": after.st_size, "sha256": sha.hexdigest()}


def inventory(tree):
    if tree.is_symlink() or not tree.is_dir():
        raise ValueError("DATA_DIRECTORY_INVALID")
    files = []
    for path in sorted(tree.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if len(files) >= 20000:
            raise ValueError("DATA_FILE_COUNT_LIMIT")
        files.append(file_record(path, path.relative_to(tree).as_posix()))
    if not files:
        raise ValueError("DATA_DIRECTORY_EMPTY")
    return files


def run_script(command, environment, workdir, seconds):
    child = subprocess.Popen(command, env=environment, cwd=workdir, start_new_session=True)
    def forward(signum, _frame):
        if child.poll() is None:
            os.killpg(child.pid, signum)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise ValueError("DATA_PREPARATION_CANCELLED")
    previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        try:
            code = child.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise ValueError("DATA_PREPARATION_TIMEOUT") from None
        if code:
            raise ValueError("DATA_SCRIPT_EXIT_" + str(code))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def prepare(definition, cache, *, workspace=Path("/workspace"), require_cached=False):
    spec = dict(definition)
    preparation_id = spec.pop("preparation_id")
    if preparation_id != "preparation." + identity(spec):
        raise ValueError("DATA_PREPARATION_IDENTITY_MISMATCH")
    script = workspace / spec["script"]
    if not script.resolve().is_relative_to(workspace.resolve()) or file_record(script, spec["script"])["sha256"] != spec["script_sha256"]:
        raise ValueError("DATA_SCRIPT_SHA256_MISMATCH")
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / preparation_id
    print("ML_EXPD_DATA_PREPARATION=WAITING " + preparation_id, flush=True)
    with (cache / (preparation_id + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink():
                raise ValueError("DATA_CACHE_INVALID")
            receipt_path = destination / "receipt.json"
            if receipt_path.is_symlink() or receipt_path.stat().st_size > 8 * 1024 ** 2:
                raise ValueError("DATA_RECEIPT_INVALID")
            receipt = json.loads(receipt_path.read_text())
            print("ML_EXPD_DATA_PREPARATION=VERIFYING " + preparation_id, flush=True)
            files = inventory(destination / "tree")
            if (receipt.get("definition") != definition or receipt.get("files") != files
                    or receipt.get("content_sha256") != identity(files)
                    or receipt.get("dataset_id") != "dataset." + identity(files)
                    or receipt.get("preparation_id") != preparation_id
                    or receipt.get("schema_version") != 1
                    or receipt.get("bytes") != sum(item["bytes"] for item in files)
                    or receipt.get("status") != "READY"):
                raise ValueError("DATA_CACHE_CHECKSUM_MISMATCH")
            reused = True
        else:
            if require_cached:
                raise ValueError("DATA_PREPARED_CACHE_MISSING")
            temporary = Path(tempfile.mkdtemp(prefix=".preparation-", dir=cache))
            try:
                tree = temporary / "tree"
                tree.mkdir()
                environment = {key: value for key, value in os.environ.items()
                               if not key.startswith("ML_EXPD_") and key not in {"OUTPUT_DIR", "RUN_ID", "ATTEMPT_ID", "BACKEND_JOB_ID", "DATA_DIR", "INPUTS_DIR", "STATE_DIR", "RESUME_DIR"}}
                environment.update(spec["env"], DATA_DIR=str(tree), SOURCE_ID=spec["source_id"])
                print("ML_EXPD_DATA_PREPARATION=RUNNING " + preparation_id, flush=True)
                run_script([spec["interpreter"], str(script), *spec["arguments"]], environment,
                           workspace / Path(spec["workdir"]).relative_to("/workspace"), spec["timeout_seconds"])
                print("ML_EXPD_DATA_PREPARATION=VERIFYING " + preparation_id, flush=True)
                files = inventory(tree)
                content = identity(files)
                if spec.get("expected_content_sha256") and content != spec["expected_content_sha256"]:
                    raise ValueError("DATA_EXPECTED_SHA256_MISMATCH")
                receipt = {"schema_version": 1, "status": "READY", "preparation_id": preparation_id,
                           "definition": definition, "files": files, "bytes": sum(item["bytes"] for item in files),
                           "content_sha256": content, "dataset_id": "dataset." + content,
                           "created_at": datetime.now(timezone.utc).isoformat()}
                raw = json.dumps(receipt, sort_keys=True)
                if len(raw.encode()) > 8 * 1024 ** 2:
                    raise ValueError("DATA_RECEIPT_SIZE_LIMIT")
                (temporary / "receipt.json").write_text(raw)
                for path in temporary.rglob("*"):
                    path.chmod(0o555 if path.is_dir() else 0o444)
                os.rename(temporary, destination)
                destination.chmod(0o555)
                reused = False
            finally:
                if temporary.exists():
                    for path in temporary.rglob("*"):
                        if path.is_dir() and not path.is_symlink():
                            path.chmod(0o755)
                    shutil.rmtree(temporary)
    print("ML_EXPD_DATA_PREPARATION=READY " + preparation_id + " reused=" + str(reused).lower(), flush=True)
    return destination / "tree", {**receipt, "cache_reused": reused}
