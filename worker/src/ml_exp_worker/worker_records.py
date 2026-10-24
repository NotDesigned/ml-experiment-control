"""Stdlib-only resumable JSONL telemetry; no W&B credential or dependency."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
from urllib.parse import urlsplit
import uuid
from datetime import datetime, timezone

if __package__:
    from .worker_http import https_connection
else:
    from worker_http import https_connection


def safe_numbers(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: safe_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe_numbers(item) for item in value]
    return value


class WorkerRecords:
    def __init__(self, root, url, token):
        self.root, self.url, self.token = root, url, token
        self.path = root.parent / "worker-records.json"
        self.last = -10.0
        self.state = {"generation": uuid.uuid4().hex, "offset": 0, "inode": None, "queue": [], "event": 0}
        try:
            if url and self.path.exists():
                self.state = json.loads(self.path.read_text())
        except Exception:
            self.url = ""  # corrupt local telemetry cannot block training

    def save(self):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, sort_keys=True))
        temporary.chmod(0o600)
        temporary.replace(self.path)

    def emit(self, kind, data):
        if not self.url:
            return
        self.state["event"] += 1
        if kind == "lifecycle":
            data = {**data, "timestamp": datetime.now(timezone.utc).isoformat()}
        self.state["queue"].append({"event_id": "worker." + self.state["generation"] + "." + str(self.state["event"]),
                                     "kind": kind, "data": data})
        try:
            self.save()
        except OSError:
            pass  # retain memory queue; final JSONL remains a second evidence source

    def flush(self, *, final=False):
        if not self.url or not final and time.monotonic() - self.last < 5:
            return
        self.last = time.monotonic()
        events = self.state["queue"][:64]
        end = self.state["offset"]
        path = self.root / "metrics.jsonl"
        try:
            if path.is_symlink():
                raise ValueError("metric file must be regular")
            if path.is_file():
                info = path.stat()
                identity = [info.st_dev, info.st_ino]
                if self.state["inode"] != identity or info.st_size < end:
                    self.state.update(inode=identity, offset=0, generation=uuid.uuid4().hex)
                    end = 0
                    self.save()
                with path.open("rb") as stream:
                    stream.seek(end)
                    while len(events) < 64:
                        offset = stream.tell()
                        line = stream.readline(65537)
                        if not line or len(line) <= 65536 and not line.endswith(b"\n"):
                            break  # a writer may still be completing this record
                        event_id = "metric." + self.state["generation"] + "." + str(offset)
                        try:
                            if len(line) > 65536:
                                raise ValueError("metric record exceeds size limit")
                            data = json.loads(line)
                            if not isinstance(data, dict):
                                raise ValueError("metric record must be an object")
                            event = {"event_id": event_id, "kind": "metrics", "data": safe_numbers(data)}
                        except (ValueError, UnicodeError):
                            event = {"event_id": event_id, "kind": "lifecycle", "data": {"diagnostic": "INVALID_METRIC_JSONL"}}
                        if len(json.dumps({"records": [*events, event]}, allow_nan=False).encode()) > 480 * 1024:
                            break
                        events.append(event)
                        end = stream.tell()
            if not events:
                return
            target = urlsplit(self.url)
            if target.scheme != "https" or not target.hostname or target.username or target.password or target.query or target.fragment:
                raise ValueError("record transport requires HTTPS")
            connection = https_connection(target, timeout=2)
            try:
                connection.request("POST", target.path, body=json.dumps({"records": events}, allow_nan=False).encode(),
                                   headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"})
                response = connection.getresponse()
                if response.status != 200:
                    raise ValueError("record batch rejected")
                response.read(4096)
            finally:
                connection.close()
            self.state["queue"] = self.state["queue"][min(64, len(self.state["queue"])):]
            self.state["offset"] = end
            self.save()
        except Exception:
            # The original JSONL and durable cursor survive. Transient failures
            # do not stop training; retry the same IDs, with no cursor advance.
            pass
