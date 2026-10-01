"""Read native project tracking evidence; no log parser, archive, or retry queue."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import stat
from pathlib import Path
from urllib.parse import urlsplit


def attempt_run_id(workspace: str, project: str, run: str, attempt: str) -> str:
    return hashlib.sha256("\0".join((workspace, project, run, attempt)).encode()).hexdigest()[:32]


def safe_url(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > 2048 or any(ord(c) <= 32 for c in value):
        return None
    try:
        url = urlsplit(value)
        url.port
        if (not url.hostname or url.username is not None or url.password is not None
                or url.query or url.fragment or "\\" in value):
            return None
        if url.scheme == "https":
            return value
        host = url.hostname
        if url.scheme == "http" and (host == "localhost" or ipaddress.ip_address(host).is_loopback):
            return value
    except ValueError:
        pass
    return None


def read_json(root: Path, path: Path) -> dict:
    """Bounded regular-file read confined to this Attempt; never follow symlinks."""
    relative = path.relative_to(root)
    for parent in (root, *(root / Path(*relative.parts[:i]) for i in range(1, len(relative.parts) + 1))):
        if parent.is_symlink():
            raise ValueError("tracking evidence must not be a symlink")
    if not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size > 8192:
        raise ValueError("invalid tracking evidence size/type")
    with path.open() as handle:
        data = json.loads(handle.read(8193))
    if not isinstance(data, dict):
        raise ValueError("tracking evidence must be an object")
    return data


def native_files(root: Path, run_id: str) -> list[Path]:
    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise ValueError("invalid tracking run identity")
    files = sorted(root.glob(f"wandb/offline-run-*/run-{run_id}.wandb"))
    if not files or len(files) > 32:
        raise ValueError("expected 1..32 native offline segments")
    for path in files:
        for item in (root, path.parent.parent, path.parent, path):
            if item.is_symlink():
                raise ValueError("native tracking files must not be symlinks")
        if not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size == 0:
            raise ValueError("invalid native offline segment")
    return files


def tracking_root(attempt_dir: Path, attempt: str) -> Path | None:
    # Native per-Attempt output, or a host's already-normalized Attempt mirror.
    for base in (attempt_dir / "collected_run", attempt_dir):
        for root in (base / "tracking" / attempt, base):
            if (root / "tracking.json").exists():
                return root
    return None


def read_tracking(attempt_dir: Path, workspace: str, project: str, run: str, attempt: str, config) -> dict:
    identity = {"workspace_id": workspace, "project_name": project,
                "run_id": run, "attempt_id": attempt}
    expected = attempt_run_id(workspace, project, run, attempt)
    result = {"state": "NOT_RECORDED", "wandb_run_id": expected,
              "mode": None, "dashboard_url": None, "native_segments": 0}
    try:
        root = tracking_root(attempt_dir, attempt)
        if root is None:
            return result
        data = read_json(attempt_dir, root / "tracking.json")
        if (data.get("schema_version") != 1 or data.get("identity") != identity
                or data.get("wandb_run_id") != expected):
            raise ValueError("tracking identity mismatch")
        state, mode = data.get("state"), data.get("mode")
        if mode not in {"offline", "online"} or state not in {"RECORDING", "FINISHED", "ERROR"}:
            raise ValueError("invalid tracking state")
        result.update(state=state, mode=mode)
        if state == "ERROR":
            return result  # raw exception text is deliberately never exposed
        if mode == "offline":
            segments = native_files(root, expected)
            result["native_segments"] = len(segments)
            result["state"] = "OFFLINE_READY" if state == "FINISHED" else "RECORDING"
            sync = attempt_dir / "tracking-sync.json"
            if sync.exists():
                outcome = read_json(attempt_dir, sync)
                if (outcome.get("wandb_run_id") == expected and outcome.get("state") in {"CLI_COMPLETED", "SYNC_FAILED"}
                    and outcome.get("destination") == {"api_url": config.api_url, "entity": config.entity, "project": config.project}):
                    stamps = {str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns] for p in segments}
                    if outcome.get("segments") == stamps:
                        result["sync_state"] = outcome["state"]
                        if outcome["state"] == "CLI_COMPLETED":
                            result["dashboard_url"] = f"{config.dashboard_url}/{config.entity}/{config.project}/runs/{expected}"
                    else:
                        result["sync_state"] = "SOURCE_CHANGED"
        else:
            # Project URLs must refer to the configured destination and exact run.
            expected_url = (f"{config.dashboard_url}/{config.entity}/{config.project}/runs/{expected}"
                            if config.entity and config.project else None)
            if data.get("url") == expected_url and safe_url(expected_url):
                result["dashboard_url"] = expected_url
        return result
    except (OSError, ValueError, TypeError):
        return {**result, "state": "INVALID_EVIDENCE", "dashboard_url": None}
