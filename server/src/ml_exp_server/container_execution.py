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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import yaml

from .application_errors import ApplicationError
from .image_builder import IMAGE, RECIPE, builder_request, bundle_id
from .environment_build import DEPENDENCY_RECIPE, dockerfile, inspect_requirements, requirements_path, installer_digest
from .dockerfile_build import DOCKERFILE_RECIPE, inspect_dockerfile, managed_dockerfile, worker_digest
from .data_assets import AssetStore
from .source_imports import IDENTITY, source_lock
from .source_revisions import resolve_source_tree
from .storage import DurableJsonState, atomic_text, utc_now
from .runtime_jobs import PendingRuntimeBuild
from .worker_contract import CAPABILITIES, WORKER_CONTRACT


SECRET_KEY = re.compile(r"(?i)(?:^|_)(?:token|secret|password|credential|api_key|proxy|authorization)(?:$|_)")


class RuntimeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(pattern=r"^source\.[0-9a-f]{64}$")
    image: str | None = None
    environment_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    requirements: str | None = None
    dockerfile: str | None = None
    entrypoint: list[str] = Field(min_length=1, max_length=128)
    workdir: str = "/workspace"
    packaging_revision: str = RECIPE

    @field_validator("packaging_revision")
    @classmethod
    def reviewed_recipe(cls, value: str) -> str:
        if value not in {RECIPE, DEPENDENCY_RECIPE, DOCKERFILE_RECIPE}:
            raise ValueError("packaging_revision must match the current reviewed recipe")
        return value

    @field_validator("image")
    @classmethod
    def immutable_image(cls, value: str | None) -> str | None:
        if value is not None and not IMAGE.fullmatch(value):
            raise ValueError("image must be a registry reference pinned by sha256 digest")
        return value

    @field_validator("requirements", "dockerfile")
    @classmethod
    def relative_requirements(cls, value: str | None) -> str | None:
        return requirements_path(value) if value is not None else None

    @model_validator(mode="after")
    def build_definition(self):
        if self.dockerfile is not None:
            if self.image is not None or self.environment_id is not None or self.requirements is not None:
                raise ValueError("Dockerfile builds define their own base images and dependency installation")
            self.packaging_revision = DOCKERFILE_RECIPE
            return self
        if self.packaging_revision == DOCKERFILE_RECIPE:
            raise ValueError("Dockerfile recipe requires a Dockerfile path")
        if (self.image is None) == (self.environment_id is None):
            raise ValueError("provide exactly one of image or environment_id")
        if self.requirements is not None:
            self.packaging_revision = DEPENDENCY_RECIPE
        elif self.packaging_revision == DEPENDENCY_RECIPE:
            raise ValueError("dependency recipe requires a requirements file")
        return self

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


class DockerfileRuntimeSpec(RuntimeSpec):
    """New public builds have one path. RuntimeSpec still reads frozen history."""
    image: None = None
    environment_id: None = None
    requirements: None = None
    dockerfile: str = "Dockerfile"
    packaging_revision: Literal[DOCKERFILE_RECIPE] = DOCKERFILE_RECIPE


class Resources(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gpus: int = Field(default=1, ge=1, le=64)
    cpus: int = Field(default=8, ge=1, le=512)
    memory_gb: int = Field(default=32, ge=1, le=4096)
    max_time: str = Field(default="00:10:00", pattern=r"^[0-9]{2,3}:[0-5][0-9]:[0-5][0-9]$")


class InputAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asset_id: str = Field(pattern=r"^asset\.[0-9a-f]{64}$")
    mount_path: str = Field(pattern=r"^/inputs/[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class CheckpointUpload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    interval_seconds: int = Field(default=60, ge=5, le=3600)


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
    inputs: list[InputAsset] = Field(default_factory=list, max_length=32)
    checkpoint_upload: CheckpointUpload | None = None

    @field_validator("inputs")
    @classmethod
    def distinct_mounts(cls, value):
        if len({item.mount_path for item in value}) != len(value):
            raise ValueError("input mount paths must be distinct")
        return value

    @field_validator("arguments")
    @classmethod
    def arguments_valid(cls, value):
        return RuntimeSpec.argument_vector(value)

    @field_validator("env")
    @classmethod
    def no_credentials(cls, value):
        reserved = {"OUTPUT_DIR", "INPUTS_DIR", "PROJECT_NAME", "RUN_ID", "ATTEMPT_ID", "SOURCE_ID", "BACKEND_JOB_ID"}
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
                                "dockerfile_execution": True,
                                "data_asset_transport": "worker-http-shared-storage",
                                "artifact_transport": bool(self.runtime.config.container_execution.artifact_store_file or profile.get("artifact_ssh") or profile["backend"]["kind"] == "slurm")}
                               for name, profile in sorted(self.profiles().items())]}

    def environments(self) -> dict:
        path = self.runtime.config.container_execution.environments_file
        data = yaml.safe_load(Path(path).read_text()) if path else {}
        entries = data.get("environments", {}) if isinstance(data, dict) else {}
        if not isinstance(entries, dict):
            raise ValueError("invalid environment catalogue")
        result = []
        for name, entry in sorted(entries.items()):
            if not IDENTITY.fullmatch(name) or not isinstance(entry, dict) or not IMAGE.fullmatch(entry.get("image", "")):
                raise ValueError("invalid environment catalogue entry")
            result.append({"id": name, **{k: entry[k] for k in
                           ("title", "image", "versions", "validation", "description") if k in entry}})
        return {"environments": result}

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
        environment = None
        if spec.environment_id is not None:
            environment = next((e for e in self.environments()["environments"] if e["id"] == spec.environment_id), None)
            if environment is None:
                raise ApplicationError("unknown environment", status_code=404, code="UNKNOWN_ENVIRONMENT")
            spec = RuntimeSpec.model_validate({**spec.model_dump(exclude_none=True), "image": environment["image"], "environment_id": None})
        tree = resolve_source_tree(self.runtime.config, project, spec.source_id)
        dependencies = inspect_requirements(tree, spec.requirements) if spec.requirements is not None else None
        inspection = inspect_dockerfile(tree, spec.dockerfile) if spec.dockerfile is not None else None
        frozen = spec.model_dump(exclude_none=True)
        build_id = bundle_id(project, spec.source_id, inspection["base_images"][-1] if inspection else spec.image,
                             spec.requirements, dockerfile_path=spec.dockerfile)
        identity = hashlib.sha256(json.dumps([project, frozen, build_id], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        runtime_id = "runtime." + identity
        with self.state(project, runtime_id) as (store, snapshot):
            if snapshot.value:
                return snapshot.value
            value = {"project": project, "runtime_id": runtime_id, "spec": frozen,
                     "status": "PREPARED", "confirmation": "BUILD " + runtime_id,
                     "created_at": utc_now(), "image": None, "build_bundle_id": build_id,
                     "worker_contract": WORKER_CONTRACT}
            value.update(dockerfile=dockerfile(spec.image, spec.source_id, spec.requirements))
            if inspection is not None:
                value.update(dockerfile=managed_dockerfile(inspection, spec.source_id),
                             client_dockerfile={k: v for k, v in inspection.items() if k != "text"},
                             base_image=inspection["base_images"][-1])
            if dependencies is not None:
                value["dependencies"] = dependencies
            if environment is not None:
                value["environment"] = environment
            store.commit(value, expected_revision=snapshot.revision, event={"event": "runtime_prepared", "timestamp": utc_now()})
            return value

    def read(self, project: str, runtime_id: str) -> dict:
        with self.state(project, runtime_id) as (_, snapshot):
            if not snapshot.value:
                raise ApplicationError("unknown runtime", status_code=404, code="UNKNOWN_RUNTIME")
            return snapshot.value

    def require_current_build(self, project: str, value: dict):
        if "build_bundle_id" in value:
            prepared = RuntimeSpec.model_validate(value["spec"])
            current = bundle_id(project, prepared.source_id, value.get("base_image", prepared.image),
                                prepared.requirements, dockerfile_path=prepared.dockerfile)
            if value["build_bundle_id"] != current:
                raise ApplicationError("packaging implementation changed; prepare a new Runtime", code="CONTAINER_EXECUTION_BLOCKED")

    def execute(self, project: str, runtime_id: str, confirmation: str, *, reconcile: bool = False) -> dict:
        pending = self.begin_execute(project, runtime_id, confirmation, reconcile=reconcile)
        return self.finish_execute(pending) if pending is not None else self.read(project, runtime_id)

    def begin_execute(self, project: str, runtime_id: str, confirmation: str, *, reconcile: bool = False) -> PendingRuntimeBuild | None:
        self.require_enabled()
        with self.state(project, runtime_id) as (store, snapshot):
            value = dict(snapshot.value)
            if not value or confirmation != value["confirmation"]:
                raise ApplicationError("runtime confirmation mismatch", code="CONTAINER_EXECUTION_BLOCKED")
            if value["status"] == "READY":
                return None
            if value["status"] == "EXECUTING" and not reconcile:
                raise ApplicationError("runtime packaging is already executing; inspect or reconcile", code="CONTAINER_EXECUTION_BLOCKED")
            self.require_current_build(project, value)
            value.update(status="EXECUTING", error=None)
            from .execution_progress import record_progress
            record_progress(self.root / project / (runtime_id + ".progress.json"), "WAITING_BUILD_WORKER",
                            "Build claimed by daemon; waiting for builder request or receipt reconciliation")
            executing = store.commit(value, expected_revision=snapshot.revision, event={"event": "runtime_execution_started", "timestamp": utc_now()})
        return PendingRuntimeBuild(project, runtime_id, value, executing.revision, reconcile)

    def finish_execute(self, pending: PendingRuntimeBuild) -> dict:
        project, runtime_id, value = pending.project, pending.runtime_id, dict(pending.value)
        socket = self.runtime.config.container_execution.builder_socket
        try:
            spec = RuntimeSpec.model_validate(value["spec"])
            if not socket:
                raise ValueError("image packaging worker is not configured")
            resolve_source_tree(self.runtime.config, project, spec.source_id)
            payload = {"operation": "get" if pending.reconcile else "build", "project": project,
                                              "source_id": spec.source_id, "base_image": value.get("base_image", spec.image),
                                              "packaging_revision": spec.packaging_revision}
            if spec.requirements is not None:
                payload["requirements"] = spec.requirements
            if spec.dockerfile is not None:
                payload["dockerfile"] = spec.dockerfile
            result = builder_request(socket, payload)
            if (result.get("project") != project or result.get("source_id") != spec.source_id
                    or result.get("base_image") != value.get("base_image", spec.image) or not IMAGE.fullmatch(result.get("image", ""))):
                raise ValueError("image packaging result identity mismatch")
            if "build_bundle_id" in value and result.get("bundle_id") != value["build_bundle_id"]:
                raise ValueError("image packaging result does not match the prepared build identity")
            if "worker_contract" in value and (result.get("worker_contract") != WORKER_CONTRACT
                    or result.get("worker_sha256") != worker_digest()
                    or result.get("capabilities") != CAPABILITIES
                    or result.get("dockerfile_sha256") != hashlib.sha256(value["dockerfile"].encode()).hexdigest()):
                raise ValueError("managed worker receipt does not match the prepared contract")
            if "worker_contract" not in value and (spec.requirements is not None or spec.dockerfile is not None) and (
                    result.get("dockerfile_sha256") != hashlib.sha256(value["dockerfile"].encode()).hexdigest()):
                raise ValueError("legacy build receipt does not match the prepared recipe")
            if spec.requirements is not None and (result.get("dependencies") != value["dependencies"]
                    or result.get("installer_sha256") != installer_digest()):
                raise ValueError("dependency build receipt does not match the prepared recipe")
            if spec.dockerfile is not None and (result.get("dockerfile") != value["client_dockerfile"]
                    or "worker_contract" not in value and (result.get("worker_sha256") != worker_digest()
                    or result.get("capabilities") != CAPABILITIES)):
                raise ValueError("Dockerfile build receipt does not match the frozen definition")
            value.update(status="READY", image=result["image"], bundle_id=result["bundle_id"], completed_at=utc_now())
            if spec.dockerfile is not None or "worker_contract" in value:
                value["capabilities"] = result["capabilities"]
        except Exception:
            value.update(status="RECONCILE_REQUIRED", error="image packaging failed or is uncertain; inspect worker and reconcile")
        return self.finish_state(pending, value)

    def finish_state(self, pending: PendingRuntimeBuild, value: dict) -> dict:
        with self.state(pending.project, pending.runtime_id) as (store, snapshot):
            if snapshot.revision != pending.revision:
                return snapshot.value
            store.commit(value, expected_revision=pending.revision, event={"event": "runtime_execution_finished", "timestamp": utc_now()})
        return value

    def submission_failed(self, pending: PendingRuntimeBuild) -> None:
        self.finish_state(pending, {**pending.value, "status": "RECONCILE_REQUIRED",
                                   "error": "packaging could not be queued; inspect or reconcile"})

    def create_run(self, project: str, request: RunRequest) -> dict:
        self.require_enabled()
        bundle = self.read(project, request.runtime_id)
        if bundle["status"] != "READY":
            raise ApplicationError("runtime image is not ready", code="CONTAINER_EXECUTION_BLOCKED")
        inputs = []
        if request.inputs or request.checkpoint_upload:
            config = self.runtime.config.container_execution.artifact_store_file
            if not config or "data-assets.v1" not in bundle.get("capabilities", []):
                raise ApplicationError("data/checkpoint delivery requires object storage and a runtime with the managed worker", code="CONTAINER_EXECUTION_BLOCKED")
            assets = AssetStore(Path(config), self.runtime.config.project_registry_root_path())
            for binding in request.inputs:
                asset = assets.read(project, binding.asset_id)
                inputs.append({**binding.model_dump(), "sha256": asset["sha256"],
                               "archive_bytes": asset["archive_bytes"], "files": asset["files"]})
            if len(json.dumps(inputs).encode()) > 32768:
                raise ApplicationError("input manifests exceed the scheduler command limit", code="CONTAINER_EXECUTION_BLOCKED")
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
        if "worker_contract" in bundle:
            run["container"]["worker_contract"] = bundle["worker_contract"]
        if inputs:
            run["inputs"] = inputs
        if request.checkpoint_upload is not None:
            run["checkpoint_upload"] = request.checkpoint_upload.model_dump()
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
