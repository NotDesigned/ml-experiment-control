"""Select an existing asset copy in the executor's exact storage namespace."""
from __future__ import annotations

from ..application_errors import ApplicationError
from ..results.checkpoint_registry import storage_scope

REMOTE = frozenset({"desktop-builder", "wyd", "sensecore-ccr", "oss"})


def for_backend(asset, profile, project):
    scope = storage_scope(profile["backend"], profile["storage_root"].rstrip("/") + "/" + project)
    locations = asset.get("remote_locations", {})
    if not isinstance(locations, dict):
        raise ApplicationError("data location metadata is invalid", code="INPUT_STORAGE_SCOPE_MISMATCH")
    location = locations.get(scope)
    if location is not None:
        expected = {"slurm": "wyd", "sensecore": "sensecore-ccr", "pai-dlc": "oss"}.get(profile["backend"]["kind"])
        if not isinstance(location, dict) or location.get("storage_scope") != scope or location.get("remote_storage") != expected:
            raise ApplicationError("data copy belongs to another backend", code="INPUT_STORAGE_SCOPE_MISMATCH")
        return {**asset, **location}
    if asset.get("remote_storage") in {"wyd", "sensecore-ccr", "oss"}:
        raise ApplicationError("upload this asset to the selected executor's storage before submission", code="INPUT_STORAGE_SCOPE_MISMATCH")
    return asset
