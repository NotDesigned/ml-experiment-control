"""Explicit, bounded native SDK sync for one finished Attempt (never a daemon loop)."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time

from .credentials import CredentialStore
from .identity import workspace_identity
from .schemas import TERMINAL_RUN_STATES
from .tracking import native_files, read_tracking, tracking_root


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sync_attempt(config, project: str, run_id: str, attempt_id: str, *, timeout=300) -> dict:
    tracking = config.tracking
    if not (tracking.entity and tracking.project and tracking.credential_ref):
        raise ValueError("configure tracking entity, project and credential_ref first")
    # Read-only connection: do not start a second runtime or acquire scheduler ownership.
    db = sqlite3.connect(config.index_db_path().resolve().as_uri() + "?mode=ro", uri=True)
    try:
        record = db.execute("SELECT row_json FROM runs WHERE project=? AND run_id=?",
                            (project, run_id)).fetchone()
    finally:
        db.close()
    row = json.loads(record[0]) if record else {}
    attempt = next((a for a in row.get("attempts", []) if a.get("attempt_id") == attempt_id), {})
    if attempt.get("state") not in TERMINAL_RUN_STATES:
        raise ValueError("sync requires an indexed terminal Attempt")
    root = Path(row["run_dir"]) / "attempts" / attempt_id
    status = read_tracking(root, workspace_identity(config), project, run_id, attempt_id, tracking)
    if status["state"] != "OFFLINE_READY":
        raise ValueError("sync requires complete native offline tracking evidence")
    source = tracking_root(root, attempt_id)
    assert source is not None
    files = native_files(source, status["wandb_run_id"])
    key = CredentialStore(Path(tracking.credential_root)).resolve_wandb_api_key(tracking.credential_ref)
    for path in (root / ".tracking-sync.lock", root / "tracking-upload", root / "tracking-sync.json"):
        if path.is_symlink():
            raise ValueError("sync state must not be a symlink")
    with (root / ".tracking-sync.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Retain SDK-owned sync markers outside the collector's rsync --delete tree.
        staged = root / "tracking-upload"
        staged.mkdir(mode=0o700, exist_ok=True)
        destination = {"api_url": tracking.api_url, "entity": tracking.entity, "project": tracking.project}
        target_manifest = staged / "destination.json"
        if target_manifest.is_symlink():
            raise ValueError("sync destination must not be a symlink")
        if target_manifest.exists() and json.loads(target_manifest.read_text()) != destination:
            raise ValueError("sync destination changed; original SDK markers belong to another target")
        target_manifest.write_text(json.dumps(destination))
        selected = []
        for source_file in files:
            digest = file_digest(source_file)
            target = staged / (source_file.parent.name + ".wandb")
            if target.is_symlink() or target.with_suffix(target.suffix + ".synced").is_symlink():
                raise ValueError("sync segments and markers must not be symlinks")
            if target.exists() and file_digest(target) != digest:
                raise ValueError("native segment changed after sync staging")
            if not target.exists():
                shutil.copyfile(source_file, target)
                target.chmod(0o600)
            if not target.with_suffix(target.suffix + ".synced").exists():
                selected.append(target)
        result = {"wandb_run_id": status["wandb_run_id"], "state": "CLI_COMPLETED", "destination": destination,
                  "segments": {str(p.relative_to(source)): [p.stat().st_size, p.stat().st_mtime_ns] for p in files}}
        with tempfile.TemporaryDirectory(prefix="ml-expd-wandb-") as home:
            env = {"PATH": os.defpath, "HOME": home, "LANG": "C.UTF-8",
                   "WANDB_API_KEY": key, "WANDB_BASE_URL": tracking.api_url,
                   "WANDB_CONSOLE": "off", "WANDB_DISABLE_CODE": "true",
                   "WANDB_DISABLE_GIT": "true"}
            deadline = time.monotonic() + timeout
            for path in selected:
                command = [sys.executable, "-m", "wandb", "sync", "--append",
                           "--id", status["wandb_run_id"], "--entity", tracking.entity,
                           "--project", tracking.project, "--no-sync-tensorboard",
                           "--skip-console", str(path)]
                try:
                    child = subprocess.Popen(command, env=env, cwd=home, start_new_session=True,
                                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                             stderr=subprocess.DEVNULL)
                except OSError:
                    result["state"] = "SYNC_FAILED"
                    break
                try:
                    successful = child.wait(timeout=max(0.01, deadline - time.monotonic())) == 0
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                    successful = False
                if not successful or not path.with_suffix(path.suffix + ".synced").is_file():
                    result["state"] = "SYNC_FAILED"
                    break
        temporary = root / ".tracking-sync.json.tmp"
        if temporary.is_symlink():
            raise ValueError("sync state must not be a symlink")
        temporary.write_text(json.dumps(result))
        temporary.replace(root / "tracking-sync.json")
        return result
