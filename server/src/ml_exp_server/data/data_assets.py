"""Immutable data archives in the existing object store, separate from source."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from types import SimpleNamespace

from ..application_errors import ApplicationError
from ..results.artifact_store import ArtifactStore, stream_digest
from ..projects.source_imports import IDENTITY, SHA256, unpack_source, seal_tree, remove_staging
from ..storage import atomic_json, utc_now
from ..archive_limits import byte_limit, exceeds, wire_limit

ASSET_ID = re.compile(r"^asset\.[0-9a-f]{64}$")
WORKER_PATH = re.compile(r"^/api/(?:asset-transfers/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/attempt-[0-9]{3,}/asset\.[0-9a-f]{64}|(?:snapshot|checkpoint)-transfers/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/attempt-[0-9]{3,})$")


class AssetStore:
    def __init__(self, config_file: Path, registry_root: Path):
        self.objects = ArtifactStore(config_file, registry_root)
        self.root = registry_root / "data-assets"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.limit = byte_limit(self.objects.config.get("max_asset_archive_bytes", self.objects.limit))
        self.expanded_limit = byte_limit(self.objects.config.get("max_asset_bytes", self.limit))
        self.file_limit = int(self.objects.config.get("max_asset_files", 20000))

    @contextmanager
    def record(self, project: str, asset_id: str):
        if not IDENTITY.fullmatch(project) or not ASSET_ID.fullmatch(asset_id):
            raise ValueError("invalid data asset identity")
        parent = self.root / project
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (parent / (asset_id + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield parent / asset_id

    def read(self, project: str, asset_id: str) -> dict:
        with self.record(project, asset_id) as path:
            if not (path / "asset.json").is_file():
                raise ApplicationError("unknown data asset", status_code=404, code="UNKNOWN_ASSET")
            value = json.loads((path / "asset.json").read_text())
            if value["project"] != project or value["asset_id"] != asset_id:
                raise ValueError("data asset metadata identity mismatch")
            locator = self.root.parent / "desktop-data-assets" / project / (asset_id + ".json")
            if locator.exists():
                staged = json.loads(locator.read_text())
                if any(staged[key] != value[key] for key in ("sha256", "archive_bytes", "files")):
                    raise ValueError("desktop asset locator differs")
                value = {**value, "remote_storage": "desktop-builder"}
            return value

    def list(self, project: str) -> dict:
        if not IDENTITY.fullmatch(project):
            raise ValueError("invalid project identity")
        parent = self.root / project
        return {"assets": [self.read(project, p.name) for p in sorted(parent.glob("asset.*"))
                           if p.is_dir() and ASSET_ID.fullmatch(p.name)]}

    def limits(self) -> dict:
        value = {"asset_archive_bytes": wire_limit(self.limit), "asset_expanded_bytes": wire_limit(self.expanded_limit),
                "asset_file_bytes": wire_limit(self.expanded_limit), "asset_files": self.file_limit,
                "artifact_archive_bytes": wire_limit(self.objects.limit),
                "configured_byte_limits": {"asset_archive_bytes": self.limit, "asset_expanded_bytes": self.expanded_limit,
                                           "artifact_archive_bytes": self.objects.limit},
                "artifact_files": 20000, "local_storage_free_bytes": shutil.disk_usage(self.root).free,
                "upload_part_bytes": int(self.objects.config.get("upload_part_bytes", 16 * 1024 ** 2)),
                "upload_session_seconds": int(self.objects.config.get("upload_session_seconds", 86400)),
                "upload_part_number_base": 0,
                "upload_max_parts": int(self.objects.config.get("upload_max_parts", 65536)),
                "scheduler_input_manifest_bytes": 32768,
                "storage_quota_bytes": self.objects.config.get("storage_quota_bytes"),
                "checkpoint_publication": "atomic-ready-manifest", "input_write_protection": "file-permissions"}
        if self.objects.config.get("data_upload_storage") == "desktop-builder":
            from .remote_data_uploads import stage_request
            remote = stage_request(self.objects.config["data_upload_socket"], "info", {})
            value.update(data_upload_storage=remote["storage"], data_upload_free_bytes=remote["free_bytes"],
                         data_upload_total_bytes=remote["total_bytes"])
        else:
            value["data_upload_storage"] = "api-server"
        return value

    def download(self, project: str, asset_id: str):
        value = self.read(project, asset_id)
        if value.get("remote_storage") == "desktop-builder":
            raise ApplicationError("desktop assets use the authenticated archive endpoint", status_code=404,
                                   code="DIRECT_DOWNLOAD_UNAVAILABLE")
        return self.objects.download('/'.join(['data-assets', project, asset_id + '.tar']),
                                     value['sha256'], value['archive_bytes'], files=value['files'])

    def receive(self, project: str, stream, digest: str, size: int, *, provenance: dict | None = None):
        if not SHA256.fullmatch(digest) or size <= 0 or exceeds(size, self.limit):
            raise ValueError("invalid data asset digest or archive size")
        stream.seek(0)
        if stream_digest(stream) != digest:
            raise ValueError("data asset archive SHA256 mismatch")
        stream.seek(0)
        asset_id = "asset." + digest
        with self.record(project, asset_id) as destination:
            if destination.exists():
                value = json.loads((destination / "asset.json").read_text())
                if value["sha256"] != digest or value["archive_bytes"] != size:
                    raise ValueError("data asset identity is already bound")
                return value
            temporary = Path(tempfile.mkdtemp(prefix=".asset-", dir=destination.parent))
            try:
                tree = temporary / "tree"
                tree.mkdir()
                unpack_source(stream, tree, SimpleNamespace(max_source_bytes=self.expanded_limit,
                                                           max_source_files=self.file_limit))
                files = []
                for path in sorted(tree.rglob("*")):
                    if path.is_file():
                        with path.open("rb") as body:
                            sha = stream_digest(body)
                        files.append({"path": path.relative_to(tree).as_posix(),
                                      "bytes": path.stat().st_size, "sha256": sha})
                key = "/".join(["data-assets", project, asset_id + ".tar"])
                stream.seek(0)
                self.objects.client().upload_fileobj(stream, self.objects.config["bucket"], key,
                    ExtraArgs={"ContentType": "application/x-tar", "Metadata": {"sha256": digest}},
                    Config=self.objects._transfer_config())
                value = {"project": project, "asset_id": asset_id, "status": "READY",
                         "sha256": digest, "archive_bytes": size, "bytes": sum(f["bytes"] for f in files),
                         "files": files, "created_at": utc_now(), "provenance": provenance}
                atomic_json(temporary / "asset.json", value)
                seal_tree(temporary)
                temporary.chmod(0o700)
                os.rename(temporary, destination)
                destination.chmod(0o500)
                return value
            finally:
                remove_staging(temporary)

    def archive(self, project: str, asset_id: str):
        value = self.read(project, asset_id)
        if value.get("remote_storage") == "desktop-builder":
            from .remote_data_uploads import remote_archive
            return value, remote_archive(self.objects.config["data_upload_socket"], project, asset_id, value["archive_bytes"])
        response = self.objects.client().get_object(Bucket=self.objects.config["bucket"],
                    Key="/".join(["data-assets", project, asset_id + ".tar"]))
        return value, response["Body"]

    def publish_remote(self, project, value):
        if (value.get("project") != project or value.get("remote_storage") != "desktop-builder"
                or value.get("asset_id") != "asset." + value.get("sha256", "")
                or not SHA256.fullmatch(value.get("sha256", "")) or value.get("archive_bytes", 0) <= 0):
            raise ValueError("desktop asset identity differs")
        files = value.get("files", [])
        if not files or len(files) > self.file_limit or len({item["path"] for item in files}) != len(files):
            raise ValueError("desktop asset file inventory differs")
        from ..workers.managed_worker import relative_path
        for item in files:
            relative_path(item["path"])
            if not SHA256.fullmatch(item["sha256"]) or item["bytes"] < 0:
                raise ValueError("desktop asset file identity differs")
        with self.record(project, value["asset_id"]) as destination:
            if destination.exists():
                previous = json.loads((destination / "asset.json").read_text())
                if any(previous[key] != value[key] for key in ("sha256", "archive_bytes", "files")):
                    raise ValueError("desktop asset is already sealed")
                atomic_json(self.root.parent / "desktop-data-assets" / project / (value["asset_id"] + ".json"), value)
                return {**previous, "remote_storage": "desktop-builder"}
            destination.mkdir(mode=0o700)
            atomic_json(destination / "asset.json", value)
            destination.chmod(0o500)
            return value

    def authorize_input(self, project: str, run: str, attempt: str, asset_id: str, token: str):
        value = self.objects.authorize(project, run, attempt, token)
        import yaml
        manifest = yaml.safe_load((Path(value["run_dir"]) / "manifest.yaml").read_text())
        identities = {item["identity"] for item in manifest["assets"] if item["kind"] == "data_asset"}
        if manifest["project"] != project or manifest["run_id"] != run or asset_id not in identities:
            raise ValueError("input capability is not bound to this asset")
        return self.read(project, asset_id)

    def snapshot(self, project: str, run: str, attempt: str, token: str, stream, size: int):
        value = self.objects.authorize(project, run, attempt, token)
        if not value.get("checkpoint_upload"):
            raise ValueError("checkpoint publication is not enabled for this Attempt")
        stream.seek(0)
        digest = stream_digest(stream)
        result = self.receive(project, stream, digest, size,
                              provenance={"kind": "checkpoint", "run_id": run, "attempt_id": attempt})
        with self.objects.record(project, run, attempt) as (path, current):
            snapshots = current.setdefault("snapshots", [])
            if result["asset_id"] not in snapshots:
                snapshots.append(result["asset_id"])
                atomic_json(path, current)
        return result

    def snapshots(self, project: str, run: str, attempt: str):
        with self.objects.record(project, run, attempt) as (_, value):
            ids = value.get("snapshots", []) if value else []
        return {"snapshots": [self.read(project, asset_id) for asset_id in ids]}
