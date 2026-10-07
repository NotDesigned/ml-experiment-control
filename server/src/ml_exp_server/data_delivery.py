"""Durable data-image publication and bounded, zero-GPU ACP preparation."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import hmac
import json
from pathlib import Path
import re
import secrets
import shlex
import time
from urllib.parse import urlsplit

from experiment_control.backends.sensecore_rest import SenseCoreREST, RESTError, create_document

from .application_errors import ApplicationError
from .data_assets import AssetStore, ASSET_ID
from .data_image_build import data_worker_digest
from .image_builder import builder_request, IMAGE
from .source_imports import IDENTITY
from .storage import DurableJsonState, utc_now

DELIVERY = re.compile(r"^delivery\.[0-9a-f]{64}$")
COPY_TRANSFER = re.compile(r"^/api/data-copy-transfers/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/delivery\.[0-9a-f]{64}$")
CONTROL_REVISION = "acp-data-copy.v5"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def recover_data_deliveries(runtime):
    root = runtime.config.project_registry_root_path() / "data-deliveries"
    recovered = 0
    for path in root.glob("*/*.json"):
        if not IDENTITY.fullmatch(path.parent.name) or not DELIVERY.fullmatch(path.stem):
            continue
        service = DataDeliveryService(runtime)
        value = service.read(path.parent.name, path.stem)
        if value["status"] in {"BUILDING_IMAGE", "IMAGE_READY", "SUBMITTING", "QUEUED", "COPYING"}:
            service.update(path.parent.name, path.stem, status="RECONCILE_REQUIRED", error="daemon interrupted; inspect exact publication/job before reconciliation")
            recovered += 1
    return recovered


class DataDeliveryService:
    def __init__(self, runtime):
        self.runtime = runtime
        file = runtime.config.container_execution.artifact_store_file
        if not file:
            raise ApplicationError("data delivery is not configured", code="DATA_DELIVERY_DISABLED")
        self.assets = AssetStore(Path(file), runtime.config.project_registry_root_path())
        self.config = self.assets.objects.config.get("data_delivery")
        if not self.config:
            raise ApplicationError("data delivery is not configured", code="DATA_DELIVERY_DISABLED")
        self.root = runtime.config.project_registry_root_path() / "data-deliveries"

    @contextmanager
    def state(self, project, delivery_id):
        if not IDENTITY.fullmatch(project) or not DELIVERY.fullmatch(delivery_id):
            raise ValueError("invalid data delivery identity")
        root = self.root / project
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        with (root / (delivery_id + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            store = DurableJsonState(root / (delivery_id + ".json"), root / (delivery_id + ".jsonl"))
            snapshot = store.snapshot({})
            store.repair_journal(snapshot)
            yield store, snapshot

    @staticmethod
    def public(value):
        return {key: item for key, item in value.items() if key != "copy_token"}

    def read(self, project, delivery_id):
        with self.state(project, delivery_id) as (_, snapshot):
            if not snapshot.value:
                raise ApplicationError("unknown data delivery", status_code=404, code="UNKNOWN_DATA_DELIVERY")
            return self.public(snapshot.value)

    def update(self, project, delivery_id, **fields):
        with self.state(project, delivery_id) as (store, snapshot):
            if snapshot.value.get("status") in {"READY", "FAILED"}:
                return self.public(snapshot.value)
            value = {**snapshot.value, **fields, "last_progress_at": utc_now()}
            store.commit(value, expected_revision=snapshot.revision,
                         event={"event": "data_delivery_progress", "phase": value["status"], "timestamp": utc_now()})
            return self.public(value)

    def prepare(self, project, asset_id, executor):
        from .container_execution import ContainerExecutionService
        containers = ContainerExecutionService(self.runtime)
        containers.require_enabled()
        self.runtime.project(project)
        if not ASSET_ID.fullmatch(asset_id):
            raise ValueError("invalid data asset identity")
        profile = containers.profiles().get(executor)
        if not profile or profile["backend"]["kind"] != "sensecore":
            raise ApplicationError("CCR data preparation requires a SenseCore executor", code="DATA_BACKEND_UNSUPPORTED")
        asset = self.assets.read(project, asset_id)
        copy = {**self.config, "workspace": profile["backend"]["workspace"],
                "storage_mount": profile["backend"]["storage_mount"],
                "data_root": profile["storage_root"].rstrip("/") + "/" + project + "/data-assets"}
        copy.setdefault("quota_type", "reserved")
        if (copy.get("gpus") != 0 or copy.get("cpus") != 2 or copy.get("memory_gb") != 4
                or not re.fullmatch(r"[A-Za-z0-9_.-]+\.2c4g", copy.get("worker_spec", ""))
                or not 5 <= copy.get("copy_timeout_seconds", 0) <= 3600
                or not 5 <= copy.get("queue_timeout_seconds", 0) <= 3600
                or copy["quota_type"] not in {"reserved", "spot"}
                or not IDENTITY.fullmatch(copy.get("aec2", ""))):
            raise ValueError("data preparation requires the verified 2CPU/4GiB/0GPU profile and bounded time")
        definition = {"project": project, "asset_id": asset_id, "archive_sha256": asset["sha256"],
                      "files_sha256": digest(asset["files"]), "copy_profile": copy,
                      "worker_sha256": data_worker_digest(), "control_revision": CONTROL_REVISION}
        delivery_id = "delivery." + digest(definition)
        with self.state(project, delivery_id) as (store, snapshot):
            if snapshot.value:
                return self.public(snapshot.value)
            if "debug" in copy["aec2"].casefold():
                raise ValueError("new ACP data-copy jobs must not use a debug cluster")
            value = {**definition, "delivery_id": delivery_id, "status": "PREPARED",
                     "confirmation": "prepare-data:" + digest(definition), "created_at": utc_now(),
                     "last_progress_at": utc_now(), "scheduler_name": "ml-expd-data-" + delivery_id[-40:],
                     "copy_token": secrets.token_urlsafe(32), "image": None, "error": None}
            store.commit(value, expected_revision=snapshot.revision,
                         event={"event": "data_delivery_prepared", "timestamp": utc_now()})
            return self.public(value)

    def begin(self, project, delivery_id, confirmation, *, reconcile=False):
        with self.state(project, delivery_id) as (store, snapshot):
            value = snapshot.value
            if not value or confirmation != value["confirmation"]:
                raise ValueError("data preparation confirmation differs")
            if value["status"] == "READY":
                return None
            if value["status"] != "PREPARED" and not reconcile:
                raise ApplicationError("data delivery requires inspection/reconciliation", code="DATA_DELIVERY_UNCERTAIN")
            if reconcile and value["status"] != "RECONCILE_REQUIRED":
                raise ValueError("only an interrupted delivery can be reconciled")
            if not reconcile and value["worker_sha256"] != data_worker_digest():
                raise ValueError("data-copy worker changed; prepare a new delivery")
            if not reconcile and value.get("control_revision") != CONTROL_REVISION:
                raise ValueError("data-copy control revision changed; prepare a new delivery")
            current = {**value, "status": "BUILDING_IMAGE", "started_at": value.get("started_at", utc_now()), "last_progress_at": utc_now(), "error": None}
            store.commit(current, expected_revision=snapshot.revision,
                         event={"event": "data_delivery_started", "timestamp": utc_now()})
            return project, delivery_id, current, reconcile

    @property
    def rest(self):
        if not hasattr(self, "_rest"):
            self._rest = SenseCoreREST.from_environment()
        return self._rest

    def find(self, value):
        matches = self.rest.find(value["copy_profile"], value["scheduler_name"])
        if len(matches) > 1:
            raise ValueError("ambiguous data-copy scheduler identity")
        return matches[0] if matches else None

    def verify_cpu(self, copy):
        specs = [r for r in self.rest.specs(copy) if r["name"] == copy["worker_spec"]]
        if len(specs) != 1:
            raise ValueError("ACP CPU-only data-copy spec is unavailable")
        spec = specs[0]
        if (spec["device"]["number"], spec["cpu"]["vcpu_allocatable"], spec["memory"]["allocatable"]) != (0, 2, 4):
            raise ValueError("ACP data-copy spec unexpectedly allocates GPUs or different CPU resources")

    def create_document(self, value):
        copy = value["copy_profile"]
        endpoint = urlsplit(self.assets.objects.config["public_transfer_base"])
        if (endpoint.scheme != "https" or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment
                or not endpoint.path.endswith("/api/artifact-transfers")):
            raise ValueError("data copy callback requires a fixed HTTPS endpoint")
        prefix = endpoint.path.removesuffix("/api/artifact-transfers")
        url = endpoint.scheme + "://" + endpoint.netloc + prefix + "/api/data-copy-transfers/" + value["project"] + "/" + value["delivery_id"]
        command = ["env", "ML_EXPD_DATA_COPY_URL=" + url, "ML_EXPD_DATA_COPY_TOKEN=" + value["copy_token"],
                   "ML_EXPD_DATA_COPY_ROOT=" + copy["data_root"], "ML_EXPD_DATA_COPY_IMAGE=" + value["image"],
                   "ML_EXPD_DATA_COPY_SECONDS=" + str(copy["copy_timeout_seconds"]),
                   "python", "/usr/local/lib/ml-expd/data_copy_worker.py"]
        body = create_document(copy, value["scheduler_name"], value["image"], shlex.join(command))
        body["roles"][0]["resource_spec"][0].update(
            requests={"cpu": "2", "memory": "3Gi"}, limits={"cpu": "2", "memory": "4Gi"})
        return body

    def finish(self, pending):
        project, delivery_id, value, reconcile = pending
        try:
            image = ({"image": value["image"], "asset_id": value["asset_id"], "project": project,
                      "files_sha256": value["files_sha256"], "data_worker_sha256": value["worker_sha256"],
                      "data_image_id": value["data_image_id"]} if reconcile and value.get("image") else builder_request(self.runtime.config.container_execution.builder_socket,
                                    {"operation": "data-image", "action": "get" if reconcile else "build",
                                     "project": project, "asset_id": value["asset_id"]}))
            if (not IMAGE.fullmatch(image.get("image", "")) or image.get("asset_id") != value["asset_id"]
                    or image.get("project") != project or image.get("files_sha256") != value["files_sha256"]
                    or image.get("data_worker_sha256") != value["worker_sha256"]):
                raise ValueError("published data image identity differs")
            value.update(image=image["image"], data_image_id=image["data_image_id"])
            self.update(project, delivery_id, status="IMAGE_READY", image=value["image"], data_image_id=value["data_image_id"])
            found = self.find(value)
            if found is None:
                if reconcile:
                    raise ValueError("data-copy submission is absent; no automatic replay")
                self.verify_cpu(value["copy_profile"])
                self.update(project, delivery_id, status="SUBMITTING")
                self.rest.create(value["copy_profile"], self.create_document(value), timeout=60)
                self.update(project, delivery_id, status="QUEUED")
            else:
                self.update(project, delivery_id, status="QUEUED")
            deadline = time.monotonic() + value["copy_profile"]["queue_timeout_seconds"] + value["copy_profile"]["copy_timeout_seconds"]
            while time.monotonic() < deadline:
                current = self.read(project, delivery_id)
                if current["status"] in {"READY", "FAILED"}:
                    return current
                job = self.find(value)
                if job and job.get("state", "").upper() in {"SUCCEEDED", "FAILED", "STOPPED", "SUSPENDED", "DELETED"}:
                    raise ValueError("data-copy job ended without a verified READY callback")
                time.sleep(2)
            self.rest.stop(value["copy_profile"], value["scheduler_name"])
            return self.update(project, delivery_id, status="FAILED", error="DATA_COPY_TIMEOUT")
        except Exception as exc:
            current = self.read(project, delivery_id)
            if current["status"] in {"READY", "FAILED"}:
                return current
            return self.update(project, delivery_id, status="RECONCILE_REQUIRED", error="data publication or ACP submission needs inspection; no automatic replay",
                               control_error=exc.details if isinstance(exc, RESTError) else None)

    @staticmethod
    def check_capability(value, token):
        if not value or not hmac.compare_digest(value["copy_token"], token):
            raise ValueError("invalid data-copy capability")
        issued = value.get("started_at", value["created_at"])
        if (datetime.now(timezone.utc) - datetime.fromisoformat(issued.replace("Z", "+00:00"))).total_seconds() > 86400:
            raise ValueError("expired data-copy capability")

    def authorize(self, project, delivery_id, token):
        with self.state(project, delivery_id) as (_, snapshot):
            self.check_capability(snapshot.value, token)

    def callback(self, project, delivery_id, token, receipt):
        with self.state(project, delivery_id) as (store, snapshot):
            value = snapshot.value
            self.check_capability(value, token)
            status = receipt.get("status")
            if (status not in {"COPYING", "READY", "FAILED"} or not value.get("image")
                    or value["status"] not in {"BUILDING_IMAGE", "IMAGE_READY", "SUBMITTING", "QUEUED", "COPYING", "RECONCILE_REQUIRED", "READY"}):
                raise ValueError("invalid data-copy status transition")
            if status != "FAILED" and any(receipt.get(key) != expected for key, expected in {
                    "asset_id": value["asset_id"], "archive_sha256": value["archive_sha256"],
                    "files_sha256": value["files_sha256"], "image": value["image"],
                    "data_path": value["copy_profile"]["data_root"] + "/" + value["asset_id"]}.items()):
                raise ValueError("data-copy receipt identity differs")
            if value["status"] == "READY":
                if status != "READY" or receipt != value["receipt"]:
                    raise ValueError("data-copy receipt is already sealed")
                return self.public(value)
            updated = {**value, "status": status, "last_progress_at": utc_now(), "receipt": receipt}
            store.commit(updated, expected_revision=snapshot.revision,
                         event={"event": "data_copy_callback", "phase": status, "timestamp": utc_now()})
            return self.public(updated)

    def ready_for(self, project, asset_id, executor):
        prepared = self.prepare(project, asset_id, executor)
        if prepared["status"] != "READY":
            raise ApplicationError("prepare data on the backend before creating a GPU Run", code="INPUT_NOT_READY")
        return {"delivery_id": prepared["delivery_id"], "image": prepared["image"], "files_sha256": prepared["files_sha256"]}
