"""Server-owned preparation of one immutable experiment; no scheduler replay."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import time

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..application_errors import ApplicationError
from ..results.checkpoint_registry import CheckpointRegistry
from ..container_execution import ContainerExecutionService, DockerfileRuntimeSpec, RunRequest
from ..data.data_assets import AssetStore
from .executor_capabilities import ExecutionRequirements, ExecutorSelector, match, validate_worker, validate_script_asset
from ..tracking.tracking_service import store_for, preparation_record
from ..projects.source_imports import IDENTITY
from ..storage import DurableJsonState, utc_now
from ..backends.operations import backend_operations
from .deletion_guards import request_run
from ..deletion_state import RunFence


PREPARATION_ID = re.compile(r"^preparation-[0-9a-f]{32}$")


def observe_dependency(read, active, timeout=1800):
    deadline = time.monotonic() + timeout
    while True:
        value = read()
        if value["status"] not in active or time.monotonic() >= deadline:
            return value
        time.sleep(2)


class PreparationRun(RunRequest):
    runtime_id: str | None = Field(default=None, pattern=r"^runtime\.[0-9a-f]{64}$")
    executor: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class ExperimentPreparationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runtime: DockerfileRuntimeSpec | None = None
    run: PreparationRun
    executor_selector: ExecutorSelector | None = None
    requirements: ExecutionRequirements = Field(default_factory=ExecutionRequirements)
    max_gpu_hours: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def one_runtime(self):
        if (self.runtime is None) == (self.run.runtime_id is None):
            raise ValueError("provide a runtime build spec or an existing runtime_id")
        return self


class ExperimentPreparationService:
    def __init__(self, runtime, submissions, *, backends=None):
        self.runtime = runtime
        self.backends = backends
        self.submissions = submissions
        self.containers = ContainerExecutionService(runtime)
        self.root = runtime.config.project_registry_root_path() / "experiment-preparations"

    @contextmanager
    def state(self, project, preparation_id):
        if not IDENTITY.fullmatch(project) or not PREPARATION_ID.fullmatch(preparation_id):
            raise ValueError("invalid experiment preparation identity")
        fence = RunFence(self.runtime.config.project_registry_root_path())
        # Preparation IDs do not contain their Run ID. Hold the workspace
        # shared fence before reading that binding or repairing its journal,
        # then keep it through the caller's entire state transition.
        with fence.shared():
            directory = self.root / project
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            with (directory / (preparation_id + ".lock")).open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                store = DurableJsonState(directory / (preparation_id + ".json"), directory / (preparation_id + ".jsonl"))
                snapshot = store.snapshot({})
                if snapshot.value:
                    fence.require_live(project, snapshot.value["run_id"])
                store.repair_journal(snapshot)
                yield store, snapshot

    @staticmethod
    def public(value):
        result = {key: item for key, item in value.items() if key not in {"request", "runtime_requested", "delivery_requested"}}
        result["matching"] = {key: item for key, item in value["matching"].items() if key != "profile"}
        return result

    def read(self, project, preparation_id):
        with self.state(project, preparation_id) as (_, snapshot):
            if not snapshot.value:
                raise ApplicationError("unknown experiment preparation", status_code=404, code="UNKNOWN_PREPARATION")
            return self.public(snapshot.value)

    def update(self, project, preparation_id, **fields):
        with self.state(project, preparation_id) as (store, snapshot):
            if not snapshot.value:
                raise ApplicationError("unknown experiment preparation", status_code=404, code="UNKNOWN_PREPARATION")
            value = {**snapshot.value, **fields, "updated_at": utc_now()}
            if value["phase"] != snapshot.value["phase"]:
                value["phase_started_at"] = value["updated_at"]
            store.commit(value, expected_revision=snapshot.revision,
                         event={"event": "preparation_progress", "phase": value["phase"], "status": value["status"], "timestamp": value["updated_at"]})
            preparation_record(self.runtime, value)
            return value

    @request_run
    def accept(self, project, request):
        self.containers.require_enabled()
        request_body = request.model_dump(mode="json")
        if not request.requirements.data_preparation_before_job:
            request_body["requirements"].pop("data_preparation_before_job")
        # Absent new options preserve the request hash of pre-upgrade intents.
        for key in ("wandb", "parameters"):
            if request_body["run"][key] is None:
                request_body["run"].pop(key)
        request_sha = hashlib.sha256(json.dumps(request_body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        preparation_id = "preparation-" + hashlib.sha256((project + "/" + request.run.run_id).encode()).hexdigest()[:32]
        with self.state(project, preparation_id) as (store, snapshot):
            if snapshot.value:
                if snapshot.value["request_sha256"] != request_sha:
                    raise ApplicationError("Run already has a different preparation request; use a new run_id", code="PREPARATION_INTENT_CONFLICT")
                return self.public(snapshot.value), None
            matching = self.select(project, request.run, request.requirements, request.executor_selector)
            objects = self.runtime.config.container_execution.artifact_store_file
            h, m, s = (int(part) for part in request.run.resources.max_time.split(":"))
            if request.run.resources.gpus * (h * 3600 + m * 60 + s) / 3600 > request.max_gpu_hours:
                raise ApplicationError("GPU-hour budget is smaller than requested GPU count times duration", code="GPU_BUDGET_EXCEEDED")
            cache_required = False
            for item in request.run.inputs:
                from ..data.locations import for_backend, REMOTE
                asset = for_backend(AssetStore(Path(objects), self.runtime.config.project_registry_root_path()).read(project, item.asset_id), matching["profile"], project)
                validate_script_asset(request.run, matching["capabilities"], asset)
                cache_required |= asset.get("remote_storage") in REMOTE and matching["capabilities"]["data"]["asset_preparation"] == "before_job"
            bundle = (self.containers.prepare(project, request.runtime) if request.runtime is not None
                      else self.containers.read(project, request.run.runtime_id))
            if bundle["status"] == "READY":
                validate_worker(request.run, bundle, matching["capabilities"], request.requirements, cache_required=cache_required)
            now = utc_now()
            value = {"preparation_id": preparation_id, "project": project, "run_id": request.run.run_id,
                     "tracking": store_for(self.runtime).route(request.run.wandb, project),
                     "request": request_body, "request_sha256": request_sha, "matching": matching,
                     "runtime_id": bundle["runtime_id"], "status": "EXECUTING", "phase": "ACCEPTED",
                     "created_at": now, "updated_at": now, "deliveries": {},
                     "phase_started_at": now, "data_receipts": {},
                     "runtime_requested": False, "delivery_requested": [], "next_action": "OBSERVE"}
            store.commit(value, expected_revision=snapshot.revision,
                         event={"event": "preparation_accepted", "timestamp": now})
            preparation_record(self.runtime, value)
            return self.public(value), (project, preparation_id)

    def select(self, project, run, requirements, selector):
        profiles = self.containers.profiles()
        declarations = self.containers.profile_capabilities(profiles)
        objects = self.runtime.config.container_execution.artifact_store_file
        resume = None
        if run.resume_from is not None:
            if not objects:
                raise ApplicationError("checkpoint registry is not configured", code="CONTAINER_EXECUTION_BLOCKED")
            ref = run.resume_from
            resume = CheckpointRegistry(Path(objects), self.runtime.config.project_registry_root_path()).read(project, ref.run_id, ref.attempt_id, ref.checkpoint_id)["storage_scope"]
        return match(profiles, declarations, run, requirements, selector, resume_scope=resume, project=project)

    def progress(self, project, preparation_id):
        value = self.read(project, preparation_id)
        script = value.get("data_script")
        if script:
            with self.state(project, preparation_id) as (_, snapshot):
                kind = snapshot.value["matching"]["profile"]["backend"]["kind"]
            script = backend_operations(kind, self.backends).script_progress(self, project, script)
        started = datetime.fromisoformat(value["phase_started_at"].replace("Z", "+00:00"))
        return {"phase": value["phase"], "phase_started_at": value["phase_started_at"],
                "phase_seconds": max(0, (datetime.now(timezone.utc) - started).total_seconds()),
                "last_progress_at": value["updated_at"], "status": value["status"],
                "runtime_id": value["runtime_id"], "deliveries": value["deliveries"],
                "data_script": script,
                "diagnostic": value.get("error_code"), "message": "Server preparation; scheduler and training have separate observations"}

    def continue_preparation(self, project, preparation_id):
        with self.state(project, preparation_id) as (store, snapshot):
            value = dict(snapshot.value)
            if not value:
                raise ApplicationError("unknown experiment preparation", status_code=404, code="UNKNOWN_PREPARATION")
            self.containers.require_enabled()
            if value["status"] != "RECONCILE_REQUIRED":
                return self.public(value), None
            # The original per-effect request markers are retained. Continuing
            # can observe READY dependencies, never recreate an uncertain effect.
            value.update(status="EXECUTING", next_action="OBSERVE", updated_at=utc_now())
            store.commit(value, expected_revision=snapshot.revision,
                         event={"event": "preparation_continued", "timestamp": value["updated_at"]})
            return self.public(value), (project, preparation_id)

    def finish(self, pending):
        project, preparation_id = pending
        with self.state(project, preparation_id) as (_, snapshot):
            value = dict(snapshot.value)
        if not value:
            raise ApplicationError("unknown experiment preparation", status_code=404, code="UNKNOWN_PREPARATION")
        if value["status"] != "EXECUTING":
            return self.public(value)
        try:
            request = ExperimentPreparationRequest.model_validate(value["request"])
            selected = value["matching"]
            self.check_profile(selected)
            bundle = self.containers.read(project, value["runtime_id"])
            self.update(project, preparation_id, phase="BUILDING_RUNTIME")
            if bundle["status"] == "PREPARED" and not value["runtime_requested"]:
                self.update(project, preparation_id, runtime_requested=True)
                try:
                    bundle = self.containers.execute(project, bundle["runtime_id"], bundle["confirmation"])
                except ApplicationError:
                    bundle = self.containers.read(project, value["runtime_id"])
                    if bundle["status"] != "EXECUTING":
                        raise
            if bundle["status"] == "EXECUTING":
                bundle = observe_dependency(lambda: self.containers.read(project, value["runtime_id"]), {"EXECUTING"})
            if bundle["status"] != "READY":
                raise ApplicationError("inspect the saved Runtime; only a verified READY dependency can continue", code="PREPARATION_RUNTIME_NOT_READY")
            validate_worker(request.run, bundle, selected["capabilities"], request.requirements)
            self.update(project, preparation_id, phase="PREPARING_DATA")
            for item in request.run.inputs:
                from ..data.locations import for_backend, REMOTE
                asset = for_backend(AssetStore(Path(self.runtime.config.container_execution.artifact_store_file), self.runtime.config.project_registry_root_path()).read(project, item.asset_id), selected["profile"], project)
                if selected["capabilities"]["data"]["asset_preparation"] != "before_job" or asset.get("remote_storage") not in REMOTE:
                    continue
                validate_worker(request.run, bundle, selected["capabilities"], request.requirements, cache_required=True)
                backend_operations(selected["profile"]["backend"]["kind"], self.backends).prepare_asset(
                    self, project, preparation_id, value, selected, item.asset_id, observe_dependency)
            self.check_profile(selected)
            self.update(project, preparation_id, phase="FREEZING_RUN")
            definition = request.run.model_dump(exclude_none=True)
            definition.update(executor=selected["executor"], runtime_id=bundle["runtime_id"])
            run = self.containers.create_run(project, RunRequest.model_validate(definition), profile_snapshot=selected["profile"], prepared_assets=set(value.get("data_receipts", {})),
                tracking=value.get("tracking", {"enabled": False, "requested": True, "reason": "HISTORICAL_RUN_NOT_BOUND", "entity": None, "project": project}))
            if request.run.data_preparation is not None:
                backend_operations(selected["profile"]["backend"]["kind"], self.backends).prepare_script(
                    self, project, preparation_id, value, run)
            self.update(project, preparation_id, run=run, phase="CHECKING_SUBMISSION")
            submission = self.submissions.prepare_first_attempt(project, run["run_id"], max_gpu_hours=request.max_gpu_hours, reason="server-prepared experiment")
            return self.public(self.update(project, preparation_id, submission=submission, status="READY", phase="READY",
                                           error=None, error_code=None, next_action=submission["next_action"]))
        except Exception as exc:
            return self.public(self.update(project, preparation_id, status="RECONCILE_REQUIRED",
                                           error="preparation stopped; inspect the referenced dependency before continuing",
                                           error_code=exc.code if isinstance(exc, ApplicationError) else "PREPARATION_FAILED",
                                           next_action="INSPECT_DEPENDENCY"))

    def check_profile(self, selected):
        profiles = self.containers.profiles()
        name = selected["executor"]
        if profiles.get(name) != selected["profile"] or self.containers.profile_capabilities(profiles).get(name) != selected["capabilities"]:
            raise ApplicationError("executor configuration changed after matching; inspect the frozen preparation", code="EXECUTOR_CONFIGURATION_CHANGED")

    def submission_failed(self, pending):
        self.update(*pending, status="RECONCILE_REQUIRED", error="preparation executor is unavailable; inspect saved dependencies", next_action="INSPECT_DEPENDENCY")

    def recover_interrupted(self):
        count = 0
        for path in sorted(self.root.glob("*/*.json")):
            if not IDENTITY.fullmatch(path.parent.name) or not PREPARATION_ID.fullmatch(path.stem):
                continue
            try:
                with self.state(path.parent.name, path.stem) as (store, snapshot):
                    value = dict(snapshot.value)
                    if value.get("status") != "EXECUTING":
                        continue
                    value.update(status="RECONCILE_REQUIRED", next_action="INSPECT_DEPENDENCY",
                                 error="daemon interrupted preparation; inspect exact saved dependencies", updated_at=utc_now())
                    store.commit(value, expected_revision=snapshot.revision,
                                 event={"event": "preparation_interrupted", "timestamp": value["updated_at"]})
                    count += 1
            except ApplicationError as error:
                if error.code != "RUN_DELETED":
                    raise
        return count
