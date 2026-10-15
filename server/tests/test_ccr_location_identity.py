"""One content image can serve executor aliases and independent NAS scopes."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from ml_exp_server.data.ccr_uploads import CCRDataUploads
from ml_exp_server.data.data_assets import AssetStore
from ml_exp_server.data.remote_data_uploads import public_target
from tests.test_ccr_uploads import upload


def transferred(fixture, assets, target):
    binding = {"project": "trial", "kind": "asset", "target": public_target(target)}
    driver = CCRDataUploads(assets, target, fixture.files, hashlib.sha256(fixture.layer).hexdigest())
    value = driver.create(binding, hashlib.sha256(fixture.body).hexdigest(), len(fixture.body), None)
    for number in range(value["part_count"]):
        body = fixture.body[number * value["part_bytes"]:(number + 1) * value["part_bytes"]]
        part = fixture.root / "target-part"; part.write_bytes(body)
        driver.part(value["upload_id"], binding, number, part, hashlib.sha256(body).hexdigest(), len(body))
    return driver.complete(value["upload_id"], binding)


@pytest.mark.parametrize("separate_scope", [False, True])
def test_same_content_has_same_image_and_can_bind_multiple_executor_locations(upload, separate_scope):
    config = upload.root / "objects.json"
    config.write_text(json.dumps({**upload.assets.objects.config, "endpoint": "http://unused.invalid", "bucket": "unused",
                                  "access_key": "fixture", "secret_key": "fixture", "local_storage_headroom_bytes": 0}))
    assets = AssetStore(config, upload.root / "registry")
    first_target = {**upload.target, "storage_scope": "storage." + "a" * 64}
    first = transferred(upload, assets, first_target)
    original = assets.root / "trial" / first["asset_id"] / "asset.json"
    frozen = original.read_bytes()
    alias = {**first_target, "executor": "cloud-alias", "profile_sha256": "b" * 64,
             "storage_scope": "storage." + ("c" if separate_scope else "a") * 64}
    second = transferred(upload, assets, alias)
    assert first["remote_locator"]["image"] == second["remote_locator"]["image"]
    assert first["remote_locator"]["data_image_id"] == second["remote_locator"]["data_image_id"]
    assert second["executor"] == "cloud-alias" and second["storage_scope"] == alias["storage_scope"]
    assert len(second["remote_locations"]) == (2 if separate_scope else 1)
    assert original.read_bytes() == frozen
    manifest = json.loads(upload.memory.manifests[second["remote_locator"]["image"].rsplit("@", 1)[1]])
    control = upload.memory.blobs[manifest["layers"][-1]["digest"]]
    with tarfile.open(fileobj=io.BytesIO(control)) as archive:
        identity = json.loads(archive.extractfile("payload/asset.json").read())
    assert set(identity) == {"project", "asset_id", "sha256", "archive_bytes", "files"}
    assert identity["files"] == upload.files
    assert len([key for key in upload.memory.manifests if key.startswith("data-")]) == 1
