"""Resumable asset transfer and exact-identity backend data delivery."""
from __future__ import annotations

import hashlib
import json
import re
import time

from .api import ClientError, segment
from .state import save


def upload_asset_parts(client, project, stream, digest, length, state):
    endpoint = f"/api/projects/{segment(project)}/asset-uploads"
    for attempt in range(3):
        try:
            value = client.call(endpoint, data={"sha256": digest, "bytes": length})
            part_bytes = value["part_bytes"]
            if (not 1024 <= part_bytes <= 64 * 1024 ** 2 or value["sha256"] != digest or value["bytes"] != length
                    or value["part_count"] != (length + part_bytes - 1) // part_bytes
                    or not re.fullmatch(r"upload\.[0-9a-f]{64}", value["upload_id"])):
                raise ClientError("invalid multipart upload receipt")
            if value["status"] == "COMPLETED":
                return value["result"]
            save(state, {"project": project, "asset_id": "asset." + digest, "archive_bytes": length,
                         "upload_id": value["upload_id"], "status": "UPLOADING"})
            root = endpoint + "/" + value["upload_id"]
            stream.seek(0)
            for number in range(value["part_count"]):
                data = stream.read(part_bytes)
                sha = hashlib.sha256(data).hexdigest()
                previous = value["parts"].get(str(number))
                if previous:
                    if previous != {"sha256": sha, "bytes": len(data)}:
                        raise ClientError("sealed upload part differs from the local archive")
                    continue
                client.call(root + "/parts/" + str(number) + "?sha256=" + sha, raw=data, method="PUT",
                            timeout=getattr(client, "upload_timeout", 1200))
            return client.call(root + "/complete", data={}, timeout=getattr(client, "upload_timeout", 1200))
        except ClientError as exc:
            if attempt == 2 or not (exc.retryable or exc.status == 429 or exc.status is not None and exc.status >= 500):
                raise
            time.sleep(2 ** attempt)


def deliver_data(client, project, asset_id, executor, state_path, seconds):
    binding = {"project": project, "asset_id": asset_id, "executor": executor}
    state = json.loads(state_path.read_text()) if state_path.exists() else binding
    if any(state.get(key) != value for key, value in binding.items()):
        raise ClientError("saved data delivery belongs to another asset or executor")
    if "delivery" not in state:
        state["delivery"] = client.call(f"/api/projects/{segment(project)}/assets/{segment(asset_id)}/deliveries/prepare",
                                       data={"executor": executor})
        save(state_path, state)
    endpoint = f"/api/projects/{segment(project)}/data-deliveries/{segment(state['delivery']['delivery_id'])}"
    value = client.call(endpoint)
    if value["status"] == "PREPARED":
        if state.get("requested"):
            raise ClientError("data-copy request outcome is uncertain; inspect the saved delivery, never resubmit")
        state["requested"] = True
        save(state_path, state)
        client.call(endpoint + "/execute", data={"confirmation": value["confirmation"]})
    deadline = time.monotonic() + seconds
    while True:
        value = client.call(endpoint)
        state["delivery"] = value
        save(state_path, state)
        if value["status"] == "READY":
            return value
        if value["status"] in {"FAILED", "RECONCILE_REQUIRED"}:
            raise ClientError("data copy requires inspection/reconciliation; no GPU Run was created")
        if time.monotonic() >= deadline:
            raise ClientError("data copy is still preparing; resume with the saved state, never resubmit")
        time.sleep(5)
