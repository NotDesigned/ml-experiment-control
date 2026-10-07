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
from .image_builder import IMAGE, builder_request, bundle_id, BuildStorageError, BuildTransportError
from .source_paths import relative_source_path, argument_vector
from .dockerfile_build import DOCKERFILE_RECIPE, inspect_dockerfile, managed_dockerfile, worker_digest
from .data_assets import AssetStore
from .source_imports import IDENTITY, source_lock
from .source_revisions import resolve_source_tree
from .storage import DurableJsonState, atomic_text, utc_now
from .runtime_jobs import PendingRuntimeBuild
from .worker_contract import CAPABILITIES, WORKER_CONTRACT
from .metric_contract import MetricSchema, protocol_identity
from .data_preparation import identity as data_identity, file_record
from .checkpoint_registry import CheckpointRegistry, storage_scope
from .worker_http import relay_environment
from .executor_capabilities import declaration, ExecutionRequirements, mismatches, validate_worker


SECRET_KEY = re.compile(r"(?i)(?:^|_)(?:token|secret|password|credential|api_key|proxy|authorization)(?:$|_)")


class DockerfileRuntimeSpec(BaseModel):
    """The only build request: frozen source plus a client Dockerfile."""
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(pattern=r"^source\.[0-9a-f]{64}$")
    dockerfile: str = "Dockerfile"
    entrypoint: list[str] = Field(min_length=1, max_length=128)
    workdir: str = "/workspace"
    packaging_revision: Literal[DOCKERFILE_RECIPE] = DOCKERFILE_RECIPE

    @field_validator("dockerfile")
    @classmethod
    def source_file(cls, value: str) -> str:
        return relative_source_path(value)

    @field_validator("entrypoint")
    @classmethod
    def arguments_valid(cls, value: list[str]) -> list[str]:
        return argument_vector(value)

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

    @field_validator("max_time")
    @classmethod
    def bounded_duration(cls, value):
        if not any(int(part) for part in value.split(":")):
            raise ValueError("max_time must be a positive duration")
        return value


class InputAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asset_id: str = Field(pattern=r"^asset\.[0-9a-f]{64}$")
    mount_path: str = Field(pattern=r"^/inputs/[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class CheckpointUpload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    interval_seconds: int = Field(default=60, ge=5, le=3600)


class CheckpointReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    attempt_id: str = Field(pattern=r"^attempt-[0-9]{3,}$")
    checkpoint_id: str = Field(pattern=r"^checkpoint\.[0-9a-f]{64}$")


class DataPreparation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    script: str = Field(max_length=4096)
    interpreter: Literal["python3", "/bin/sh"] = "python3"
    arguments: list[str] = Field(default_factory=list, max_length=128)
    timeout_seconds: int = Field(default=1200, ge=5, le=86400)
    expected_content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("script")
    @classmethod
    def source_script(cls, value):
        return relative_source_path(value)

    @field_validator("arguments")
    @classmethod
    def script_arguments(cls, value):
        return argument_vector(value)


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
    metrics_schema: MetricSchema | None = None
    evaluation: dict = Field(default_factory=dict)
    data_preparation: DataPreparation | None = None
    checkpoint_persistence: CheckpointUpload | None = None
    resume_from: CheckpointReference | None = None

    @field_validator("evaluation")
    @classmethod
    def evaluation_context(cls, value):
        if len(json.dumps(value, allow_nan=False).encode()) > 16384:
            raise ValueError("evaluation context exceeds 16 KiB")
        return value

    @field_validator("inputs")
    @classmethod
    def distinct_mounts(cls, value):
        if len({item.mount_path for item in value}) != len(value):
            raise ValueError("input mount paths must be distinct")
        return value

    @field_validator("arguments")
    @classmethod
    def arguments_valid(cls, value):
        return argument_vector(value)

    @field_validator("env")
    @classmethod
    def no_credentials(cls, value):
        reserved = {"OUTPUT_DIR", "INPUTS_DIR", "DATA_DIR", "STATE_DIR", "RESUME_DIR", "PROJECT_NAME", "RUN_ID", "ATTEMPT_ID", "SOURCE_ID", "BACKEND_JOB_ID"}
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

    def metrics_schema(self, project):
        configured = self.runtime.project(project)
        return {"project": project, "metrics_schema":
                (configured.metrics_schema or MetricSchema()).model_dump(mode="json")}

    def set_metrics_schema(self, project, schema):
        self.require_enabled()
        with source_lock(self.root, project):
            configured = self.runtime.project(project)
            path = configured.authored_file
            if path is None:
                raise ApplicationError("project manifest is unavailable", code="PROJECT_SCHEMA_UNAVAILABLE")
            value = yaml.safe_load(Path(path).read_text())
            value["metrics_schema"] = schema.model_dump(mode="json")
            atomic_text(Path(path), yaml.safe_dump(value, sort_keys=False))
            self.runtime.register_project(Path(path))
        return self.metrics_schema(project)

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
        profiles = self.profiles()
        capabilities = self.profile_capabilities(profiles)
        return {"executors": [{"id": name, "title": profile.get("title", name),
                                "kind": profile["backend"]["kind"],
                                "capacity": profile.get("capacity"),
                                "capabilities": capabilities[name],
                                "dockerfile_execution": capabilities[name]["runtime"]["materialization"] in {"oci", "oci_to_sif"},
                                "data_asset_transport": ("ccr-cpu-nas" if profile["backend"]["kind"] == "sensecore" else "ssh-stream-shared-storage") if capabilities[name]["data"]["asset_preparation"] == "before_job" else "worker-http-shared-storage",
                                "data_preparation": "script-shared-storage.v1" if capabilities[name]["data"]["download_script"] else None,
                                "checkpoint_persistence": "backend-shared-storage.v1" if capabilities[name]["storage"]["checkpoint_registration"] else None,
                                "artifact_transport": capabilities[name]["outputs"]["transfer"]}
                               for name, profile in sorted(profiles.items())]}

    def profile_capabilities(self, profiles=None):
        profiles = self.profiles() if profiles is None else profiles
        store_file = self.runtime.config.container_execution.artifact_store_file
        desktop = bool(store_file and json.loads(Path(store_file).read_text()).get("data_upload_storage") == "desktop-builder")
        return {name: declaration(profile, artifact_store=bool(store_file), desktop=desktop)
                for name, profile in profiles.items()}

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

    def prepare(self, project: str, spec: DockerfileRuntimeSpec) -> dict:
        self.require_enabled()
        configured = self.runtime.project(project)
        if not configured.controller or not configured.controller.capabilities.get("container_execution"):
            raise ApplicationError("project uses its own controller; container import requires a managed project", code="CONTAINER_EXECUTION_BLOCKED")
        tree = resolve_source_tree(self.runtime.config, project, spec.source_id)
        inspection = inspect_dockerfile(tree, spec.dockerfile)
        frozen = spec.model_dump()
        build_id = bundle_id(project, spec.source_id, inspection["base_images"][-1], dockerfile_path=spec.dockerfile)
        identity = hashlib.sha256(json.dumps([project, frozen, build_id], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        runtime_id = "runtime." + identity
        with self.state(project, runtime_id) as (store, snapshot):
            if snapshot.value:
                return snapshot.value
            value = {"project": project, "runtime_id": runtime_id, "spec": frozen,
                     "status": "PREPARED", "confirmation": "BUILD " + runtime_id,
                     "created_at": utc_now(), "image": None, "build_bundle_id": build_id,
                     "worker_contract": WORKER_CONTRACT,
                     "dockerfile": managed_dockerfile(inspection, spec.source_id),
                     "client_dockerfile": {k: v for k, v in inspection.items() if k != "text"},
                     "base_image": inspection["base_images"][-1]}
            store.commit(value, expected_revision=snapshot.revision, event={"event": "runtime_prepared", "timestamp": utc_now()})
            return value

    def read(self, project: str, runtime_id: str) -> dict:
        with self.state(project, runtime_id) as (_, snapshot):
            if not snapshot.value:
                raise ApplicationError("unknown runtime", status_code=404, code="UNKNOWN_RUNTIME")
            return snapshot.value

    def require_current_build(self, project: str, value: dict):
        try:
            prepared = DockerfileRuntimeSpec.model_validate(value["spec"])
        except ValueError:
            raise ApplicationError("retired build path; prepare a new Dockerfile Runtime", code="CONTAINER_EXECUTION_BLOCKED") from None
        current = bundle_id(project, prepared.source_id, value["base_image"], dockerfile_path=prepared.dockerfile)
        if value.get("build_bundle_id") != current:
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
            if (value["status"] == "RECONCILE_REQUIRED" and not reconcile
                    and not (value.get("build_error") or {}).get("retry_safe", False)):
                raise ApplicationError("runtime publication is uncertain; reconcile before preparing a new build", code="CONTAINER_EXECUTION_BLOCKED")
            self.require_current_build(project, value)
            value.update(status="EXECUTING", error=None, build_error=None)
            from .execution_progress import record_progress
            record_progress(self.root / project / (runtime_id + ".progress.json"), "WAITING_BUILD_WORKER",
                            "Build claimed by daemon; waiting for builder request or receipt reconciliation")
            executing = store.commit(value, expected_revision=snapshot.revision, event={"event": "runtime_execution_started", "timestamp": utc_now()})
        return PendingRuntimeBuild(project, runtime_id, value, executing.revision, reconcile)

    def finish_execute(self, pending: PendingRuntimeBuild) -> dict:
        project, runtime_id, value = pending.project, pending.runtime_id, dict(pending.value)
        socket = self.runtime.config.container_execution.builder_socket
        try:
            spec = DockerfileRuntimeSpec.model_validate(value["spec"])
            if not socket:
                raise ValueError("image packaging worker is not configured")
            resolve_source_tree(self.runtime.config, project, spec.source_id)
            payload = {"operation": "get" if pending.reconcile else "build", "project": project,
                                              "source_id": spec.source_id, "base_image": value["base_image"],
                                              "packaging_revision": spec.packaging_revision}
            payload["dockerfile"] = spec.dockerfile
            result = builder_request(socket, payload)
            if (result.get("project") != project or result.get("source_id") != spec.source_id
                    or result.get("base_image") != value["base_image"] or not IMAGE.fullmatch(result.get("image", ""))):
                raise ValueError("image packaging result identity mismatch")
            if (result.get("bundle_id") != value["build_bundle_id"]
                    or result.get("worker_contract") != WORKER_CONTRACT
                    or result.get("worker_sha256") != worker_digest()
                    or result.get("capabilities") != CAPABILITIES
                    or result.get("dockerfile_sha256") != hashlib.sha256(value["dockerfile"].encode()).hexdigest()
                    or result.get("dockerfile") != value["client_dockerfile"]):
                raise ValueError("Dockerfile worker receipt does not match the frozen definition")
            value.update(status="READY", image=result["image"], bundle_id=result["bundle_id"],
                         completed_at=utc_now(), capabilities=result["capabilities"])
        except BuildStorageError as exc:
            value.update(status="RECONCILE_REQUIRED", error=exc.code,
                         build_error={"code": exc.code, "details": exc.details,
                                      "retry_safe": exc.code in {"BUILD_STORAGE_INSUFFICIENT", "BUILD_STORAGE_UNCHECKED"}})
        except BuildTransportError as exc:
            value.update(status="RECONCILE_REQUIRED", error=exc.code,
                         build_error={"code": exc.code, "details": exc.details, "retry_safe": False})
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

    def create_run(self, project: str, request: RunRequest, *, profile_snapshot=None, prepared_assets=None) -> dict:
        self.require_enabled()
        bundle = self.read(project, request.runtime_id)
        if bundle["status"] != "READY":
            raise ApplicationError("runtime image is not ready", code="CONTAINER_EXECUTION_BLOCKED")
        selected_profile = profile_snapshot if profile_snapshot is not None else self.profiles().get(request.executor)
        if not selected_profile:
            raise ApplicationError("unknown execution profile", status_code=404, code="UNKNOWN_EXECUTOR")
        capability = self.profile_capabilities({request.executor: selected_profile})[request.executor]
        problems = mismatches(capability, selected_profile, request, ExecutionRequirements())
        if problems:
            raise ApplicationError("execution profile cannot satisfy: " + ", ".join(problems), code="CONTAINER_EXECUTION_BLOCKED")
        validate_worker(request, bundle, capability)
        relay = relay_environment(selected_profile["backend"].get("api_relay"))
        if relay:
            from urllib.parse import urlsplit
            store = self.runtime.config.container_execution.artifact_store_file
            target = urlsplit(json.loads(Path(store).read_text())["public_transfer_base"]) if store else None
            origin = urlsplit(relay["ML_EXPD_API_ORIGIN"])
            if target is None or (target.hostname, target.port or 443) != (origin.hostname, origin.port or 443):
                raise ApplicationError("API relay must target the configured artifact API origin", code="EXECUTOR_TRANSPORT_INVALID")
        inputs = []
        if request.inputs or request.checkpoint_upload:
            config = self.runtime.config.container_execution.artifact_store_file
            assets = AssetStore(Path(config), self.runtime.config.project_registry_root_path())
            for binding in request.inputs:
                asset = assets.read(project, binding.asset_id)
                inputs.append({**binding.model_dump(), "sha256": asset["sha256"],
                               "archive_bytes": asset["archive_bytes"], "files": asset["files"]})
                if prepared_assets and binding.asset_id in prepared_assets:
                    validate_worker(request, bundle, capability, cache_required=True)
                    inputs[-1]["require_cached"] = True
                if asset.get("remote_storage") == "desktop-builder" and selected_profile["backend"]["kind"] == "sensecore":
                    if "data-cache-required.v1" not in bundle.get("capabilities", []):
                        raise ApplicationError("CCR data delivery requires a newly built runtime with data-cache-required.v1", code="CONTAINER_EXECUTION_BLOCKED")
                    from .data_delivery import DataDeliveryService
                    inputs[-1].update(require_cached=True, data_delivery=DataDeliveryService(self.runtime).ready_for(project, binding.asset_id, request.executor))
            if len(json.dumps(inputs).encode()) > 32768:
                raise ApplicationError("input manifests exceed the scheduler command limit", code="CONTAINER_EXECUTION_BLOCKED")
        source_id = bundle["spec"]["source_id"]
        tree = resolve_source_tree(self.runtime.config, project, source_id)
        preparation = None
        if request.data_preparation is not None:
            spec = request.data_preparation.model_dump(exclude_none=True)
            path = tree / spec["script"]
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(tree.resolve()):
                raise ApplicationError("data preparation script is missing from frozen source", code="CONTAINER_EXECUTION_BLOCKED")
            preparation = {**spec, "script_sha256": file_record(path, spec["script"])["sha256"],
                           "source_id": source_id, "image": bundle["image"],
                           "workdir": bundle["spec"]["workdir"], "env": request.env,
                           "inputs": inputs}
            preparation["preparation_id"] = "preparation." + data_identity(preparation)
            if len(json.dumps([inputs, preparation]).encode()) > 32768:
                raise ApplicationError("data preparation exceeds the scheduler command limit", code="CONTAINER_EXECUTION_BLOCKED")
        profile = copy.deepcopy(selected_profile)
        configured = self.runtime.project(project)
        if not configured.controller or not configured.controller.capabilities.get("container_execution"):
            raise ApplicationError("project is not container-managed", code="CONTAINER_EXECUTION_BLOCKED")
        root = Path(configured.base_dir)
        backend = profile["backend"]
        storage_root = profile["storage_root"].rstrip("/") + "/" + project
        persistence = request.checkpoint_persistence
        resume = None
        if persistence is not None or request.resume_from is not None:
            config = self.runtime.config.container_execution.artifact_store_file
            persistence = persistence or CheckpointUpload()
            if request.resume_from is not None:
                ref = request.resume_from
                resume = CheckpointRegistry(Path(config), self.runtime.config.project_registry_root_path()).read(project, ref.run_id, ref.attempt_id, ref.checkpoint_id)
                if resume["storage_scope"] != storage_scope(backend, storage_root):
                    raise ApplicationError("checkpoint is on different backend storage; explicitly export/upload it as an asset", code="CHECKPOINT_STORAGE_MISMATCH")
                if any(item.mount_path == "/inputs/resume" for item in request.inputs):
                    raise ApplicationError("/inputs/resume is reserved for the checkpoint reference", code="CONTAINER_EXECUTION_BLOCKED")
            if len(json.dumps([inputs, preparation, resume]).encode()) > 32768:
                raise ApplicationError("checkpoint manifests exceed the scheduler command limit", code="CONTAINER_EXECUTION_BLOCKED")
        backend["time"] = request.resources.max_time
        run_dir = storage_root + "/runs/" + request.run_id
        if backend["kind"] == "slurm":
            gpu_type = backend["gres"].split(":")[1]
            backend.update(gres=f"gpu:{gpu_type}:{request.resources.gpus}",
                           source_dir=storage_root + "/sources/" + source_id,
                           sif_path=storage_root + "/images/" + bundle["bundle_id"] + ".sif",
                           oci_image=bundle["image"])
        else:
            reference, digest = bundle["image"].split("@", 1)
            scheduler_name = "ml-" + re.sub(r"[^a-z0-9-]", "-", request.run_id.lower())[:30].strip("-") + "-" + hashlib.sha256(request.run_id.encode()).hexdigest()[:10]
            backend.update(image=reference + ":bundle-" + bundle["bundle_id"],
                           job_name=scheduler_name, display_name=request.run_id)
        campaign_name = "run-" + request.run_id
        outputs = list(request.outputs)
        if preparation is not None and "data-preparation.json" not in outputs:
            outputs.append("data-preparation.json")
        if persistence is not None and "checkpoints.json" not in outputs:
            outputs.append("checkpoints.json")
        resources = request.resources.model_dump()
        if backend["kind"] == "sensecore" and profile.get("capacity"):
            capacity = profile["capacity"]
            resources.update({key: capacity[key] for key in ("gpus", "cpus", "memory_gb")})
        run = {"run_id": request.run_id, "source_id": source_id,
               "image_id": bundle["image"].split("@", 1)[1], "backend": backend,
               "resources": resources,
               "storage": {"run_dir": run_dir, "project_data_root": storage_root,
                           "data_root": profile.get("data_root", "/data")},
               "container": {**bundle["spec"], "image": bundle["image"], "runtime_id": request.runtime_id},
               "arguments": request.arguments, "env": request.env, "outputs": outputs,
               "checkpoint": request.checkpoint, "max_infra_retries": request.max_infra_retries,
               "artifact_ssh": profile.get("artifact_ssh")}
        if "worker_contract" in bundle:
            run["container"]["worker_contract"] = bundle["worker_contract"]
        if "launcher-manifest.v1" in bundle.get("capabilities", []):
            run["container"]["launcher_contract"] = "launcher-manifest.v1"
        if inputs:
            run["inputs"] = inputs
        if request.checkpoint_upload is not None:
            run["checkpoint_upload"] = request.checkpoint_upload.model_dump()
        if preparation is not None:
            run["data_preparation"] = preparation
        if persistence is not None:
            run["checkpoint_persistence"] = {**persistence.model_dump(), "storage_scope": storage_scope(backend, storage_root)}
        if resume is not None:
            run["resume_from"] = resume
        metric_schema = request.metrics_schema or getattr(configured, "metrics_schema", None)
        if metric_schema is not None or request.evaluation:
            context = {"source_id": source_id, "image": bundle["image"],
                       "inputs": inputs, "evaluation": request.evaluation,
                       "entrypoint": bundle["spec"]["entrypoint"], "arguments": request.arguments,
                       "env": request.env,
                       "metrics_schema": (metric_schema or MetricSchema()).model_dump(mode="json")}
            if preparation is not None:
                context["data_preparation"] = preparation
            if resume is not None:
                context["resume_from"] = resume
            run["evaluation"] = {"metrics_schema": context["metrics_schema"],
                                 "protocol": context, "protocol_id": protocol_identity(context)}
        campaign = {"schema_version": 1, "project": project, "campaign": campaign_name,
                    "source_store": str(self.runtime.config.project_registry_root_path()),
                    "artifact_store": self.runtime.config.container_execution.artifact_store_file,
                    "registry_pull": self.runtime.config.container_execution.registry_pull_file,
                    "runs": [run]}
        # The controller receives only frozen fields; profile edits never rebind a Run.
        path = root / "experiments" / "campaigns" / f"{campaign_name}.yaml"
        with source_lock(self.root, project):
            existing = yaml.safe_load(path.read_text()) if path.exists() else None
            if backend["kind"] == "sensecore" and "pool_selection" not in backend:
                # Historical fixed Runs keep their original definition; new Runs
                # default to the account's requested SPOT selection policy.
                if existing is None or existing["runs"][0]["backend"].get("pool_selection") == "highest_spot":
                    backend["pool_selection"] = "highest_spot"
            if backend["kind"] == "sensecore" and backend.get("pool_selection") not in {None, "fixed"}:
                from experiment_control.backends.sensecore_rest import SenseCoreREST
                if backend["pool_selection"] != "highest_spot":
                    raise ApplicationError("unsupported SenseCore pool selection", code="CONTAINER_EXECUTION_BLOCKED")
                if existing is not None:
                    previous = existing["runs"][0]["backend"]
                    evidence = previous.get("pool_selection_evidence", {})
                    if not isinstance(evidence, dict):
                        raise ApplicationError("frozen pool selection evidence is invalid", code="CONTAINER_EXECUTION_BLOCKED")
                    if (evidence.get("configured_aec2") == backend["aec2"]
                            and evidence.get("policy") == "highest_spot"
                            and evidence.get("selected") == previous["aec2"]):
                        backend.update(aec2=previous["aec2"], pool_selection_evidence=evidence)
                else:
                    try:
                        backend.update(SenseCoreREST.from_environment().select_pool(backend, gpus=resources["gpus"]))
                    except (RuntimeError, ValueError) as error:
                        raise ApplicationError("SenseCore SPOT cluster selection is unavailable", code="CONTAINER_EXECUTION_BLOCKED") from error
            if existing is None and backend["kind"] == "sensecore" and "debug" in backend["aec2"].casefold():
                raise ApplicationError("new ACP jobs must not use a debug cluster", code="CONTAINER_EXECUTION_BLOCKED")
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
                "resources": resources, "state": "NOT_SUBMITTED", "evaluation": run.get("evaluation", {}),
                **({"pool_selection": backend["pool_selection_evidence"]} if "pool_selection_evidence" in backend else {}),
                **({"data_preparation": preparation} if preparation is not None else {}),
                **({"checkpoint_persistence": run["checkpoint_persistence"]} if persistence is not None else {}),
                **({"resume_from": resume} if resume is not None else {})}
