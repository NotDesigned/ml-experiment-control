"""Frozen source/image/argv definitions shared by every execution backend."""

from __future__ import annotations

from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
import yaml

from .application_errors import ApplicationError
from .image_builder import IMAGE, RECIPE, builder_request
from .source_imports import IDENTITY, source_lock
from .source_revisions import resolve_source_tree
from .storage import DurableJsonState, atomic_text, utc_now


SECRET_KEY = re.compile(r"(?i)(?:^|_)(?:token|secret|password|credential|api_key|proxy|authorization)(?:$|_)")


class RuntimeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(pattern=r"^source\.[0-9a-f]{64}$")
    image: str
    entrypoint: list[str] = Field(min_length=1, max_length=128)
    workdir: str = "/workspace"
    packaging_revision: str = RECIPE

    @field_validator("packaging_revision")
    @classmethod
    def reviewed_recipe(cls, value: str) -> str:
        if value != RECIPE:
            raise ValueError("packaging_revision must match the current reviewed recipe")
        return value

    @field_validator("image")
    @classmethod
    def immutable_image(cls, value: str) -> str:
        if not IMAGE.fullmatch(value):
            raise ValueError("image must be a registry reference pinned by sha256 digest")
        return value

    @field_validator("entrypoint")
    @classmethod
    def argument_vector(cls, value: list[str]) -> list[str]:
        if any(not item or "\x00" in item or len(item) > 8192 for item in value):
            raise ValueError("entrypoint contains an invalid argument")
        return value

    @field_validator("workdir")
    @classmethod
    def source_workdir(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (not path.is_absolute() or ".." in path.parts
                or path.parts[:2] != ("/", "workspace") or "\x00" in value):
            raise ValueError("workdir must stay within /workspace")
        return value


class Resources(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gpus: int = Field(default=1, ge=1, le=64)
    cpus: int = Field(default=8, ge=1, le=512)
    memory_gb: int = Field(default=32, ge=1, le=4096)
    max_time: str = Field(default="00:10:00", pattern=r"^[0-9]{2,3}:[0-5][0-9]:[0-5][0-9]$")


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    runtime_id: str = Field(pattern=r"^runtime\.[0-9a-f]{64}$")
    executor: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    arguments: list[str] = Field(default_factory=list, max_length=128)
    env: dict[str, str] = Field(default_factory=dict)
    resources: Resources = Field(default_factory=Resources)
    outputs: list[str] = Field(default_factory=lambda: ["**/*"], max_length=64)
    checkpoint: dict = Field(default_factory=lambda: {"expected_first_minutes": 5, "max_uncheckpointed_minutes": 10})
    max_infra_retries: int = Field(default=1, ge=0, le=10)

    @field_validator("arguments")
    @classmethod
    def arguments_valid(cls, value):
        return RuntimeSpec.argument_vector(value)

    @field_validator("env")
    @classmethod
    def no_credentials(cls, value):
        reserved = {"OUTPUT_DIR", "PROJECT_NAME", "RUN_ID", "ATTEMPT_ID", "SOURCE_ID", "BACKEND_JOB_ID"}
        for key, item in value.items():
            if (not re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", key) or SECRET_KEY.search(key)
                    or key in reserved or key.startswith("ML_EXPD_") or "\x00" in item or len(item) > 8192):
                raise ValueError("environment contains a reserved or credential-bearing field")
        return value

    @field_validator("outputs")
    @classmethod
    def output_paths(cls, value):
        for item in value:
            path = PurePosixPath(item)
            if (not item or path.is_absolute() or ".." in path.parts or "\\" in item
                    or any(part.startswith(".") for part in path.parts)):
                raise ValueError("outputs must be relative patterns within the Attempt output directory")
        return value


class ContainerExecutionService:
    def __init__(self, runtime):
        self.runtime = runtime
        self.root = runtime.config.project_registry_root_path() / "runtime-bundles"

    def require_enabled(self):
        if not self.runtime.config.action_runtime.allow_project_writes:
            raise ApplicationError("project writes are disabled", code="CONTAINER_EXECUTION_BLOCKED")

    def profiles(self) -> dict:
        value = self.runtime.config.container_execution.profiles_file
        if not value:
            return {}
        data = yaml.safe_load(Path(value).read_text())
        profiles = data.get("executors", {}) if isinstance(data, dict) else {}
        if not isinstance(profiles, dict):
            raise ValueError("invalid execution profile configuration")
        return profiles

    def public_profiles(self) -> dict:
        return {"executors": [{"id": name, "title": profile.get("title", name),
                                "kind": profile["backend"]["kind"],
                                "capacity": profile.get("capacity"),
                                "artifact_transport": bool(self.runtime.config.container_execution.artifact_store_file or profile.get("artifact_ssh") or profile["backend"]["kind"] == "slurm")}
                               for name, profile in sorted(self.profiles().items())]}

    @contextmanager
    def state(self, project: str, runtime_id: str):
        if not IDENTITY.fullmatch(project) or not re.fullmatch(r"runtime\.[0-9a-f]{64}", runtime_id):
            raise ValueError("invalid runtime identity")
        directory = self.root / project
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (directory / f".{runtime_id}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            store = DurableJsonState(directory / f"{runtime_id}.json", directory / f"{runtime_id}.jsonl")
            snapshot = store.snapshot({})
            store.repair_journal(snapshot)
            yield store, snapshot

    def prepare(self, project: str, spec: RuntimeSpec) -> dict:
        self.require_enabled()
        configured = self.runtime.project(project)
        if not configured.controller or not configured.controller.capabilities.get("container_execution"):
            raise ApplicationError("project uses its own controller; container import requires a managed project", code="CONTAINER_EXECUTION_BLOCKED")
        resolve_source_tree(self.runtime.config, project, spec.source_id)
        identity = hashlib.sha256(json.dumps([project, spec.model_dump()], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        runtime_id = "runtime." + identity
        with self.state(project, runtime_id) as (store, snapshot):
            if snapshot.value:
                return snapshot.value
            value = {"project": project, "runtime_id": runtime_id, "spec": spec.model_dump(),
                     "status": "PREPARED", "confirmation": "BUILD " + runtime_id,
                     "created_at": utc_now(), "image": None}
            store.commit(value, expected_revision=snapshot.revision, event={"event": "runtime_prepared", "timestamp": utc_now()})
            return value

    def read(self, project: str, runtime_id: str) -> dict:
        with self.state(project, runtime_id) as (_, snapshot):
            if not snapshot.value:
                raise ApplicationError("unknown runtime", status_code=404, code="UNKNOWN_RUNTIME")
            return snapshot.value

    def execute(self, project: str, runtime_id: str, confirmation: str, *, reconcile: bool = False) -> dict:
        self.require_enabled()
        with self.state(project, runtime_id) as (store, snapshot):
            value = dict(snapshot.value)
            if not value or confirmation != value["confirmation"]:
                raise ApplicationError("runtime confirmation mismatch", code="CONTAINER_EXECUTION_BLOCKED")
            if value["status"] == "READY":
                return value
            if value["status"] == "EXECUTING" and not reconcile:
                raise ApplicationError("runtime packaging is already executing; inspect or reconcile", code="CONTAINER_EXECUTION_BLOCKED")
            value.update(status="EXECUTING", error=None)
            executing = store.commit(value, expected_revision=snapshot.revision, event={"event": "runtime_execution_started", "timestamp": utc_now()})
        spec = RuntimeSpec.model_validate(value["spec"])
        socket = self.runtime.config.container_execution.builder_socket
        try:
            if not socket:
                raise ValueError("image packaging worker is not configured")
            resolve_source_tree(self.runtime.config, project, spec.source_id)
            result = builder_request(socket, {"operation": "get" if reconcile else "build", "project": project,
                                              "source_id": spec.source_id, "base_image": spec.image,
                                              "packaging_revision": spec.packaging_revision})
            if (result.get("project") != project or result.get("source_id") != spec.source_id
                    or result.get("base_image") != spec.image or not IMAGE.fullmatch(result.get("image", ""))):
                raise ValueError("image packaging result identity mismatch")
            value.update(status="READY", image=result["image"], bundle_id=result["bundle_id"], completed_at=utc_now())
        except Exception:
            value.update(status="RECONCILE_REQUIRED", error="image packaging failed or is uncertain; inspect worker and reconcile")
        with self.state(project, runtime_id) as (store, snapshot):
            if snapshot.revision != executing.revision:
                return snapshot.value
            store.commit(value, expected_revision=executing.revision, event={"event": "runtime_execution_finished", "timestamp": utc_now()})
        return value

    def create_run(self, project: str, request: RunRequest) -> dict:
        self.require_enabled()
        bundle = self.read(project, request.runtime_id)
        if bundle["status"] != "READY":
            raise ApplicationError("runtime image is not ready", code="CONTAINER_EXECUTION_BLOCKED")
        source_id = bundle["spec"]["source_id"]
        resolve_source_tree(self.runtime.config, project, source_id)
        profile = copy.deepcopy(self.profiles().get(request.executor))
        if not profile:
            raise ApplicationError("unknown execution profile", status_code=404, code="UNKNOWN_EXECUTOR")
        configured = self.runtime.project(project)
        if not configured.controller or not configured.controller.capabilities.get("container_execution"):
            raise ApplicationError("project is not container-managed", code="CONTAINER_EXECUTION_BLOCKED")
        root = Path(configured.base_dir)
        backend = profile["backend"]
        storage_root = profile["storage_root"].rstrip("/") + "/" + project
        backend["time"] = request.resources.max_time
        run_dir = storage_root + "/runs/" + request.run_id
        if backend["kind"] == "slurm":
            gpu_type = backend["gres"].split(":")[1]
            backend.update(gres=f"gpu:{gpu_type}:{request.resources.gpus}",
                           source_dir=storage_root + "/sources/" + source_id,
                           sif_path=storage_root + "/images/" + bundle["bundle_id"] + ".sif",
                           oci_image=bundle["image"])
        elif backend["kind"] == "sensecore":
            expected_gpus = int(profile.get("gpus", 1))
            if request.resources.gpus != expected_gpus:
                raise ApplicationError("requested GPU count does not match execution profile", code="CONTAINER_EXECUTION_BLOCKED")
            reference, digest = bundle["image"].split("@", 1)
            scheduler_name = "ml-" + re.sub(r"[^a-z0-9-]", "-", request.run_id.lower())[:30].strip("-") + "-" + hashlib.sha256(request.run_id.encode()).hexdigest()[:10]
            backend.update(image=reference + ":bundle-" + bundle["bundle_id"],
                           job_name=scheduler_name, display_name=request.run_id)
        else:
            raise ApplicationError("execution profile is not a supported container backend", code="CONTAINER_EXECUTION_BLOCKED")
        campaign_name = "run-" + request.run_id
        resources = request.resources.model_dump()
        if backend["kind"] == "sensecore" and profile.get("capacity"):
            capacity = profile["capacity"]
            if any(resources[key] > capacity[key] for key in ("gpus", "cpus", "memory_gb")):
                raise ApplicationError("requested resources exceed execution profile capacity", code="CONTAINER_EXECUTION_BLOCKED")
            resources.update({key: capacity[key] for key in ("gpus", "cpus", "memory_gb")})
        run = {"run_id": request.run_id, "source_id": source_id,
               "image_id": bundle["image"].split("@", 1)[1], "backend": backend,
               "resources": resources,
               "storage": {"run_dir": run_dir, "project_data_root": storage_root,
                           "data_root": profile.get("data_root", "/data")},
               "container": {**bundle["spec"], "image": bundle["image"], "runtime_id": request.runtime_id},
               "arguments": request.arguments, "env": request.env, "outputs": request.outputs,
               "checkpoint": request.checkpoint, "max_infra_retries": request.max_infra_retries,
               "artifact_ssh": profile.get("artifact_ssh")}
        campaign = {"schema_version": 1, "project": project, "campaign": campaign_name,
                    "source_store": str(self.runtime.config.project_registry_root_path()),
                    "artifact_store": self.runtime.config.container_execution.artifact_store_file,
                    "registry_pull": self.runtime.config.container_execution.registry_pull_file,
                    "runs": [run]}
        # The controller receives only frozen fields; profile edits never rebind a Run.
        path = root / "experiments" / "campaigns" / f"{campaign_name}.yaml"
        with source_lock(self.root, project):
            existing = yaml.safe_load(path.read_text()) if path.exists() else None
            if existing is not None and existing != campaign:
                raise ApplicationError("run_id is already bound to another execution definition", code="CONTAINER_EXECUTION_BLOCKED")
            atomic_text(path, yaml.safe_dump(campaign, sort_keys=False))
            manifest = root / "experiments" / "research_project.yaml"
            payload = yaml.safe_load(manifest.read_text())
            reference = {"name": campaign_name, "file": path.relative_to(root).as_posix()}
            if reference not in payload["campaigns"]:
                payload["campaigns"].append(reference)
                atomic_text(manifest, yaml.safe_dump(payload, sort_keys=False))
            self.runtime.register_project(manifest)
        return {"project": project, "run_id": request.run_id, "campaign": campaign_name,
                "runtime_id": request.runtime_id, "source_id": source_id, "image": bundle["image"],
                "entrypoint": [*bundle["spec"]["entrypoint"], *request.arguments], "executor": request.executor,
                "resources": resources, "state": "NOT_SUBMITTED"}
