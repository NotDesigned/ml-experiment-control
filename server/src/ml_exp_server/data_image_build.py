"""Fixed data-only OCI recipe, with its context read on the remote builder."""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
import re
import tempfile

from .storage import atomic_json
from .data_image_recipe import RECIPE, recipe, data_worker_digest



def build(builder, request):
    from .image_builder import IMAGE, PROJECT
    from .image_build_context import BUILD_LOG, BUILD_PROGRESS, BUILD_REMOTE_CONTEXT, BUILD_CONTEXT_BYTES
    from .execution_progress import record_progress, progress_view
    project, asset_id = request.get("project", ""), request.get("asset_id", "")
    base = builder.config.get("data_base_image", "")
    if (not PROJECT.fullmatch(project) or not re.fullmatch(r"asset\.[0-9a-f]{64}", asset_id)
            or not IMAGE.fullmatch(base) or builder.config.get("publisher") != "buildkit"):
        raise ValueError("invalid data image identity or publisher")
    prefix = builder.config.get("base_image_prefixes", [])
    if prefix and not any(base.startswith(item) for item in prefix):
        raise ValueError("data image base is outside packaging policy")
    def remote(operation, value):
        result = builder.desktop_stage(operation, value, b"")
        if not result.get("ok"):
            raise ValueError("desktop data image input is unavailable")
        return result["result"]
    asset = remote("asset", {"binding": {"project": project, "kind": "asset"}, "asset_id": asset_id})
    files_digest = hashlib.sha256(json.dumps(asset["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if asset.get("project") != project or asset.get("asset_id") != asset_id:
        raise ValueError("desktop data image input identity differs")
    worker = data_worker_digest()
    identity = hashlib.sha256(json.dumps([project, asset_id, asset["archive_bytes"], files_digest, base, worker, RECIPE], separators=(",", ":")).encode()).hexdigest()
    root = builder.root / "data-images"
    root.mkdir(mode=0o700, exist_ok=True)
    path, progress = root / (identity + ".json"), root / (identity + ".progress.json")
    result_base = {"project": project, "asset_id": asset_id, "data_image_id": "data-image." + identity,
                   "base_image": base, "data_worker_sha256": worker, "files_sha256": files_digest, "recipe": RECIPE}
    operation = request.get("action", "build")
    if operation == "progress":
        return {**result_base, "progress": progress_view(progress, "READY" if path.exists() else "EXECUTING", active=not path.exists())}
    with (root / (identity + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            value = json.loads(path.read_text())
            if any(value.get(key) != item for key, item in result_base.items()) or not IMAGE.fullmatch(value.get("image", "")):
                raise ValueError("data image receipt identity differs")
            return value
        if operation != "build":
            raise ValueError("data image publication requires reconciliation")
        proof = remote("info", {})
        if proof.get("data_worker_sha256") != worker:
            raise ValueError("desktop data-copy worker changed")
        container = builder.config["data_upload_container"]
        context_url = f"http://{container}:8080/contexts/{project}/{asset_id}.tar"
        text = recipe(base)
        with tempfile.TemporaryDirectory(prefix="data-control-", dir=builder.root) as directory:
            control = Path(directory)
            (control / "Dockerfile").write_text(text)
            tokens = [
                (BUILD_LOG, BUILD_LOG.set(root / (identity + ".log"))),
                (BUILD_PROGRESS, BUILD_PROGRESS.set(progress)),
                (BUILD_REMOTE_CONTEXT, BUILD_REMOTE_CONTEXT.set(context_url)),
                (BUILD_CONTEXT_BYTES, BUILD_CONTEXT_BYTES.set(asset["archive_bytes"])),
            ]
            try:
                record_progress(progress, "PREPARING_DATA_IMAGE", "Validated desktop data; build context stays on the desktop")
                image = builder._publish_buildkit(builder.config["repository"] + ":data-" + identity, control)
            finally:
                for variable, token in reversed(tokens):
                    variable.reset(token)
        result = {**result_base, "image": image, "dockerfile_sha256": hashlib.sha256(text.encode()).hexdigest()}
        atomic_json(path, result)
        record_progress(progress, "IMAGE_READY", "Data image manifest verified in CCR")
        return result
