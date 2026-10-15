"""Resumable asset transfer and exact-identity backend data delivery."""
from __future__ import annotations

import hashlib
import json
import re
import time

from .api import ClientError, segment
from .state import save
from pathlib import Path
from .archives import data_transport_identity


def upload_asset_parts(client, project, stream, digest, length, state, *, target=None):
    endpoint = f"/api/projects/{segment(project)}/asset-uploads"
    request = {"sha256": digest, "bytes": length}
    saved = json.loads(Path(state).read_text()) if Path(state).exists() else {}
    upload_id = saved.get("upload_id")
    frozen_target = saved.get("target_binding")
    def completed_asset(result):
        if target is not None:
            expected = {"project": project, "asset_id": "asset." + digest, "sha256": digest,
                        "archive_bytes": length, "status": "READY"}
            if not isinstance(result, dict) or any(result.get(key) != item for key, item in expected.items()):
                raise ClientError("completed asset identity differs from the uploaded archive")
        return result
    if upload_id is not None and (not re.fullmatch(r"upload\.[0-9a-f]{64}", str(upload_id))
            or saved.get("project") != project or saved.get("asset_id") != "asset." + digest
            or saved.get("archive_bytes") != length):
        raise ClientError("saved upload belongs to another archive")
    if target is not None:
        if Path(state).exists():
            if saved.get("target", target) != target:
                raise ClientError("saved upload belongs to another executor")
        # An existing session owns its original target and inventory. A GET
        # resume must not compute a new publication request or select storage.
        if upload_id is None:
            inventory, transport_sha = data_transport_identity(stream, length)
            request.update(target=target, files=inventory, transport_sha256=transport_sha)
    for attempt in range(3):
        try:
            value = client.call(endpoint + "/" + upload_id) if upload_id is not None else client.call(endpoint, data=request)
            part_bytes = value["part_bytes"]
            if (not 1024 <= part_bytes <= 64 * 1024 ** 2 or value["sha256"] != digest or value["bytes"] != length
                    or value["part_count"] != (length + part_bytes - 1) // part_bytes
                    or not re.fullmatch(r"upload\.[0-9a-f]{64}", value["upload_id"])
                    or upload_id is not None and value["upload_id"] != upload_id):
                raise ClientError("invalid multipart upload receipt")
            if target is not None:
                binding = value.get("binding")
                actual = binding.get("target") if isinstance(binding, dict) else None
                if (not isinstance(actual, dict) or actual.get("backend") not in {"slurm", "sensecore"}
                        or not re.fullmatch(r"storage\.[0-9a-f]{64}", str(actual.get("storage_scope", "")))
                        or not re.fullmatch(r"[0-9a-f]{64}", str(actual.get("profile_sha256", "")))
                        or actual.get("executor") != target.get("executor")
                        or target.get("storage_scope", actual.get("storage_scope")) != actual.get("storage_scope")):
                    raise ClientError("server upload receipt belongs to another executor storage")
                actual = {key: actual[key] for key in ("executor", "backend", "storage_scope", "profile_sha256")}
                if frozen_target is not None and actual != frozen_target:
                    raise ClientError("server upload receipt differs from the frozen target")
                frozen_target = actual
            upload_id = value["upload_id"]
            upload_state = {"project": project, "asset_id": "asset." + digest, "archive_bytes": length,
                            "upload_id": upload_id, "status": value["status"],
                            **({"target": target, "target_binding": frozen_target} if target else {})}
            if value["status"] == "COMPLETED":
                result = completed_asset(value["result"])
                save(state, upload_state)
                return result
            save(state, upload_state)
            root = endpoint + "/" + value["upload_id"]
            stream.seek(0)
            for number in range(value["part_count"]):
                data = stream.read(part_bytes)
                sha = hashlib.sha256(data).hexdigest()
                previous = value["parts"].get(str(number))
                if previous:
                    if previous.get("sha256") != sha or previous.get("bytes") != len(data):
                        raise ClientError("sealed upload part differs from the local archive")
                    continue
                receipt = client.call(root + "/parts/" + str(number) + "?sha256=" + sha, raw=data, method="PUT",
                                      timeout=getattr(client, "upload_timeout", 1200))
                if target is not None and (receipt.get("contract") != "upload-part.v1" or receipt.get("durable") is not True
                        or receipt.get("upload_id") != value["upload_id"] or receipt.get("number") != number
                        or receipt.get("sha256") != sha or receipt.get("bytes") != len(data)):
                    raise ClientError("backend has not confirmed this upload part")
            result = completed_asset(client.call(root + "/complete", data={}, timeout=getattr(client, "upload_timeout", 1200)))
            save(state, {**upload_state, "status": "COMPLETED"})
            return result
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
