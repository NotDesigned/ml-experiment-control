"""Stream a fixed data image to CCR without Docker or a local dataset copy."""
from __future__ import annotations

from ml_exp_worker import source_path

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import io
import json
from pathlib import Path
import tarfile

from ..application_errors import ApplicationError
from ..archive_limits import exceeds
from ..data_image_recipe import DATA_WORKERS, data_worker_digest
from ..storage import atomic_json, utc_now
from .multipart_upload import UploadStore, UPLOAD_ID
from .registry_transport import Registry, DIGEST, MANIFEST, digest, encoded

RECIPE = "acp-data-image.stream.v1"


def payload_prefix(size):
    member = tarfile.TarInfo("payload/dataset.tar")
    member.size, member.mode = size, 0o444
    return member.tobuf(format=tarfile.GNU_FORMAT)


def payload_suffix(size):
    return b"\0" * ((-size) % 512 + 1024)


class CCRDataUploads:
    remote = True
    part_size = staticmethod(UploadStore.part_size)

    def __init__(self, assets, target, files=None, transport_sha256=None):
        self.assets, self.target = assets, target
        self.files, self.transport_sha256 = files, transport_sha256
        self.root = assets.root.parent / "ccr-data-uploads"
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def target_identity(self):
        return {key: self.target[key] for key in ("executor", "backend", "storage_scope", "profile_sha256")}

    @contextmanager
    def state(self, upload_id, binding):
        if not UPLOAD_ID.fullmatch(upload_id):
            raise ValueError("invalid CCR upload identity")
        path = self.root / (upload_id + ".json")
        if path.is_symlink():
            raise ValueError("CCR upload state is a link")
        with (self.root / (upload_id + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            value = json.loads(path.read_text()) if path.exists() else None
            if value is not None and (value["binding"] != binding or value["target"] != self.target_identity()):
                raise ValueError("CCR upload belongs to another target")
            yield path, value

    @staticmethod
    def public(value):
        return {key: value[key] for key in ("upload_id", "binding", "sha256", "bytes", "part_bytes", "part_count",
                                          "parts", "status", "expires_at", "result") if key in value}

    @staticmethod
    def active(value):
        if value is None:
            raise ApplicationError("unknown CCR upload", status_code=404, code="UNKNOWN_UPLOAD")
        if value["status"] not in {"UPLOADING", "FINALIZING"}:
            raise ApplicationError("CCR upload is not accepting data", code="UPLOAD_NOT_ACTIVE")
        if datetime.fromisoformat(value["expires_at"]) <= datetime.now(timezone.utc):
            raise ApplicationError("CCR upload has expired", status_code=410, code="UPLOAD_EXPIRED")

    def registry(self, value):
        settings = value["registry"]
        return Registry(settings["repository"], settings["registry_auth_file"], blob_hosts=settings.get("blob_hosts", []))

    def create(self, binding, sha, size, limit):
        part_bytes = int(self.assets.objects.config.get("upload_part_bytes", 1024 ** 2))
        maximum = int(self.assets.objects.config.get("upload_max_parts", 65536))
        if (not DIGEST.fullmatch("sha256:" + sha) or type(size) is not int or size <= 0 or exceeds(size, limit)
                or not 1024 <= part_bytes <= 64 * 1024 ** 2 or (size + part_bytes - 1) // part_bytes > maximum
                or not DIGEST.fullmatch("sha256:" + str(self.transport_sha256)) or not self.files):
            raise ApplicationError("CCR upload requires archive, layer and file identities", status_code=413, code="UPLOAD_LIMIT")
        self.assets.validate_inventory(self.files)
        upload_id = "upload." + hashlib.sha256(json.dumps([binding, sha, size], sort_keys=True).encode()).hexdigest()
        with self.state(upload_id, binding) as (path, value):
            generation = value.get("generation", 1) + 1 if value is not None else 1
            if value is not None:
                if value["files"] != self.files or value["transport_sha256"] != self.transport_sha256:
                    raise ValueError("CCR upload inventory or layer identity changed")
                if value["status"] != "ABORTED":
                    return self.public(value)
            settings = self.assets.objects.config.get("data_registry", {})
            if (not isinstance(settings, dict) or not all(settings.get(k) for k in ("repository", "base_image", "registry_auth_file"))
                    or not settings["base_image"].startswith(settings["repository"] + "@sha256:")):
                raise ApplicationError("streaming data registry is not configured", status_code=503, code="CCR_DATA_DISABLED")
            value = {"upload_id": upload_id, "binding": binding, "target": self.target_identity(), "sha256": sha, "bytes": size,
                     "part_bytes": part_bytes, "part_count": (size + part_bytes - 1) // part_bytes,
                     "files": self.files, "transport_sha256": self.transport_sha256,
                     "registry": {key: settings[key] for key in ("repository", "base_image", "registry_auth_file", "blob_hosts") if key in settings},
                     "parts": {}, "status": "CREATING", "offset": 0, "generation": generation, "created_at": utc_now(),
                     "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=int(self.assets.objects.config.get("upload_session_seconds", 86400)))).isoformat()}
            atomic_json(path, value)
            location = self.registry(value).begin()
            value.update(location=location, status="UPLOADING")
            atomic_json(path, value)
            return self.public(value)

    def recover_part(self, path, value, registry):
        pending = value.get("pending")
        if pending is None:
            return
        offset, location = registry.offset(value["location"])
        if not pending["start"] <= offset <= pending["end"]:
            raise ApplicationError("CCR offset differs; inspect the original upload", code="CCR_UPLOAD_UNCERTAIN")
        value.update(offset=offset, location=location)
        if offset == pending["end"]:
            if pending["number"] == "tail":
                value["tail_complete"] = True
            else:
                value["parts"][str(pending["number"])] = {"sha256": pending["sha256"], "bytes": pending["bytes"]}
            value.pop("pending")
        atomic_json(path, value)

    def read(self, upload_id, binding):
        with self.state(upload_id, binding) as (path, value):
            if value is None:
                raise ApplicationError("unknown CCR upload", status_code=404, code="UNKNOWN_UPLOAD")
            if value.get("pending"):
                self.recover_part(path, value, self.registry(value))
            return self.public(value)

    def part(self, upload_id, binding, number, temporary, sha, size):
        with self.state(upload_id, binding) as (path, value):
            completed = value is not None and value["status"] == "COMPLETED"
            if not completed:
                self.active(value)
            if self.part_size(value, number) != size:
                raise ValueError("CCR upload part size differs")
            with Path(temporary).open("rb") as source:
                body = source.read(size + 1)
            if len(body) != size or hashlib.sha256(body).hexdigest() != sha:
                raise ValueError("CCR upload part checksum differs")
            if not completed:
                registry = self.registry(value)
                self.recover_part(path, value, registry)
            item = {"sha256": sha, "bytes": size}
            previous = value["parts"].get(str(number))
            if completed and previous is None:
                raise ValueError("completed CCR upload has no matching part")
            if previous is not None:
                if previous != item:
                    raise ValueError("CCR upload part is already sealed")
            else:
                if number != len(value["parts"]):
                    raise ApplicationError("CCR requires ordered parts", code="UPLOAD_PART_ORDER")
                body = payload_prefix(value["bytes"]) + body if number == 0 else body
                pending = value.get("pending")
                if pending is None:
                    pending = {"number": number, **item, "start": value["offset"], "end": value["offset"] + len(body)}
                    value["pending"] = pending
                    atomic_json(path, value)
                elif pending["number"] != number or any(pending[k] != item[k] for k in item):
                    raise ValueError("CCR pending part differs")
                remaining = body[value["offset"] - pending["start"]:]
                value["location"] = registry.append(value["location"], value["offset"], remaining)
                value["offset"] = pending["end"]
                value["parts"][str(number)] = item
                value.pop("pending")
                atomic_json(path, value)
            return {"contract": "upload-part.v1", "upload_id": upload_id, "number": number, **item, "durable": True}

    def complete(self, upload_id, binding):
        with self.state(upload_id, binding) as (path, value):
            if value is not None and value["status"] == "COMPLETED":
                return self.assets.publish_remote(binding["project"], value["result"])
            self.active(value)
            registry = self.registry(value)
            self.recover_part(path, value, registry)
            if len(value["parts"]) != value["part_count"]:
                raise ValueError("CCR upload is missing parts")
            if not value.get("tail_complete"):
                tail = payload_suffix(value["bytes"])
                pending = value.get("pending")
                if pending is None:
                    pending = {"number": "tail", "start": value["offset"], "end": value["offset"] + len(tail)}
                    value["pending"] = pending
                    atomic_json(path, value)
                remaining = tail[value["offset"] - pending["start"]:]
                value["location"] = registry.append(value["location"], value["offset"], remaining)
                value.update(offset=pending["end"], tail_complete=True)
                value.pop("pending")
                atomic_json(path, value)
            layer_digest = "sha256:" + value["transport_sha256"]
            if value["status"] == "FINALIZING":
                try:
                    registry.check_blob(layer_digest)
                except ApplicationError as error:
                    if error.code != "CCR_HTTP_404":
                        raise
                    # Only a definite missing blob plus the exact complete
                    # upload offset permits retrying its content-addressed PUT.
                    offset, location = registry.offset(value["location"])
                    if offset != value["offset"]:
                        raise ApplicationError("CCR finalization is unconfirmed", code="CCR_UPLOAD_UNCERTAIN")
                    registry.finish(location, layer_digest)
            else:
                value["status"] = "FINALIZING"
                atomic_json(path, value)
                registry.finish(value["location"], layer_digest)
            result = self.publish_image(value, registry)
            result = self.assets.publish_remote(binding["project"], result)
            value.update(status="COMPLETED", result=result)
            value.pop("location", None)
            atomic_json(path, value)
            return result

    def publish_image(self, value, registry):
        project, sha = value["binding"]["project"], value["sha256"]
        asset = {"project": project, "asset_id": "asset." + sha, "status": "READY", "sha256": sha,
                 "archive_bytes": value["bytes"], "bytes": sum(f["bytes"] for f in value["files"]),
                 "files": value["files"], "created_at": value["created_at"], "provenance": None,
                 "remote_storage": "sensecore-ccr", "storage_scope": self.target["storage_scope"],
                 "executor": self.target["executor"], "verification": "PENDING_BACKEND"}
        base = registry.manifest(value["registry"]["base_image"].rsplit("@", 1)[1])
        config = registry.json_blob(base["config"]["digest"])
        if config.get("architecture") != "amd64" or config.get("os") != "linux" or config.get("rootfs", {}).get("type") != "layers":
            raise ValueError("data base image is not linux/amd64")
        control = io.BytesIO()
        workers = source_path(DATA_WORKERS[0]).parent
        # The layer identity belongs to content, not an executor alias or an
        # upload session. Keep presentation metadata outside the pinned image.
        payload_asset = {key: asset[key] for key in ("project", "asset_id", "sha256", "archive_bytes", "files")}
        files = {"payload/asset.json": encoded(payload_asset), **{"usr/local/lib/ml-expd/" + name: (workers / name).read_bytes() for name in DATA_WORKERS}}
        with tarfile.open(fileobj=control, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name, raw in sorted(files.items()):
                item = tarfile.TarInfo(name)
                item.size, item.mode = len(raw), 0o444
                archive.addfile(item, io.BytesIO(raw))
        metadata = registry.blob(control.getvalue())
        layer = {"digest": "sha256:" + value["transport_sha256"],
                 "size": len(payload_prefix(value["bytes"])) + value["bytes"] + len(payload_suffix(value["bytes"]))}
        config["rootfs"]["diff_ids"] += [layer["digest"], metadata["digest"]]
        config.setdefault("config", {}).update(Entrypoint=[], Cmd=["/bin/true"])
        config.setdefault("history", []).extend([{"created_by": RECIPE}, {"created_by": "ml-expd verified-data worker"}])
        config_descriptor = {**registry.blob(encoded(config)), "mediaType": "application/vnd.docker.container.image.v1+json"}
        manifest = {"schemaVersion": 2, "mediaType": MANIFEST, "config": config_descriptor,
                    "layers": base["layers"] + [{**item, "mediaType": "application/vnd.docker.image.rootfs.diff.tar"} for item in (layer, metadata)]}
        files_sha = hashlib.sha256(encoded(value["files"])).hexdigest()
        worker = data_worker_digest()
        identity = hashlib.sha256(encoded([project, sha, files_sha, value["registry"]["base_image"], worker, RECIPE])).hexdigest()
        image = registry.publish("data-" + identity, manifest)
        asset["remote_locator"] = {"project": project, "asset_id": asset["asset_id"], "image": image,
                                   "data_image_id": "data-image." + identity, "files_sha256": files_sha,
                                   "data_worker_sha256": worker, "recipe": RECIPE}
        return asset

    def abort(self, upload_id, binding):
        with self.state(upload_id, binding) as (path, value):
            if value is None:
                raise ApplicationError("unknown CCR upload", status_code=404, code="UNKNOWN_UPLOAD")
            if value["status"] == "COMPLETED":
                raise ValueError("a published data image cannot be aborted")
            if value.get("location"):
                self.registry(value).request("DELETE", value["location"], expected=(204, 404))
            value.update(status="ABORTED")
            value.pop("location", None)
            atomic_json(path, value)
            return self.public(value)
