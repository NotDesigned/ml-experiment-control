"""Durable exact-Attempt result recovery; never restart a training job."""
from __future__ import annotations

import base64
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tempfile
import time

import yaml
from experiment_control.backends.sensecore_rest import SenseCoreREST, RESTError, create_document

from .application_errors import ApplicationError
from .artifact_store import ArtifactStore, ATTEMPT, artifact_released
from .artifacts import ArtifactService
from .checkpoint_registry import CheckpointRegistry
from .container_execution import ContainerExecutionService
from .data_delivery import DataDeliveryService
from .result_worker import sha
from .source_imports import IDENTITY, remove_staging
from .storage import DurableJsonState, utc_now
from .worker_artifacts import archive_outputs

TERMINAL = {"SUCCEEDED", "COMPLETED", "FAILED", "CANCELLED", "STOPPED", "SUSPENDED", "DELETED", "TIMEOUT", "PREEMPTED", "OUT_OF_MEMORY", "NODE_FAIL"}
ACTIVE = {"QUEUED", "COLLECTING", "SUBMITTING", "WAITING"}


class RecoveryError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class ResultCollectionService:
    def __init__(self, runtime):
        self.runtime = runtime
        config = runtime.config.container_execution.artifact_store_file
        if not config:
            raise ApplicationError("result storage is not configured", code="RESULT_COLLECTION_DISABLED")
        self.objects = ArtifactStore(Path(config), runtime.config.project_registry_root_path())
        self.root = runtime.config.project_registry_root_path() / "result-collections"
        self.stopping = lambda: False

    @contextmanager
    def state(self, project, run, attempt):
        if not IDENTITY.fullmatch(project) or not IDENTITY.fullmatch(run) or not ATTEMPT.fullmatch(attempt):
            raise ValueError("invalid collection identity")
        parent = self.root / project / run
        parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        with (parent / (attempt + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            store = DurableJsonState(parent / (attempt + ".json"), parent / (attempt + ".jsonl"))
            snapshot = store.snapshot({})
            store.repair_journal(snapshot)
            yield store, snapshot

    def update(self, identity, **changes):
        with self.state(*identity) as (store, snapshot):
            value = {**snapshot.value, **changes, "updated_at": utc_now()}
            store.commit(value, event={"type": "RESULT_COLLECTION_UPDATED"}, expected_revision=snapshot.revision)
            return value

    def evidence(self, project, run, attempt):
        row = self.runtime.index.get_run(project, run)
        matches = [] if row is None else [a for a in row.attempts if a.attempt_id == attempt]
        if not matches:
            raise ApplicationError("unknown Run/Attempt", status_code=404, code="UNKNOWN_ATTEMPT")
        root = Path(row.run_dir)
        manifest = yaml.safe_load((root / "manifest.yaml").read_text())
        definition = yaml.safe_load((root / "attempts" / attempt / "attempt.yaml").read_text())
        for value in (manifest, definition):
            if (value.get("project"), value.get("run_id")) != (project, run):
                raise ValueError("collection Run identity differs")
        if definition.get("attempt_id") != attempt:
            raise ValueError("collection Attempt identity differs")
        with self.objects.record(project, run, attempt) as (_, transfer):
            if not transfer or transfer["run_dir"] != str(root):
                raise ValueError("collection transfer identity differs")
        return root, definition, transfer, matches[0]

    def read(self, project, run, attempt):
        _, _, transfer, _ = self.evidence(project, run, attempt)
        with self.state(project, run, attempt) as (_, snapshot):
            value = snapshot.value
        receipt = transfer.get("receipt")
        training = transfer.get("results_ready")
        released = artifact_released(transfer)
        return {"contract": "result-collection.v1", "project": project, "run_id": run, "attempt_id": attempt,
                "status": "RELEASED" if released else "AVAILABLE" if receipt else value.get("status", "NOT_COLLECTED"),
                "phase": "RELEASED" if released else "VERIFIED" if receipt else value.get("phase", "NOT_STARTED"),
                "training": {"status": "UNKNOWN" if training is None else "COMPLETED" if training["exit_code"] == 0 else "FAILED",
                             "exit_code": None if training is None else training["exit_code"]},
                "verification": "SHA256_VERIFIED" if receipt else "PENDING",
                "download_available": bool(receipt) and not released, "attempts": value.get("attempts", 0),
                "retention": transfer.get("retention"),
                "updated_at": value.get("updated_at"), "job_state": value.get("job_state"),
                "diagnostic": value.get("diagnostic"), "scheduler_name": value.get("scheduler_name"),
                "archive": None if receipt is None else {**{k: receipt[k] for k in ("sha256", "bytes")},
                                                       "file_count": len(receipt["files"])}}

    def begin(self, project, run, attempt, *, retry=False, reconcile=False):
        ContainerExecutionService(self.runtime).require_enabled()
        identity = (project, run, attempt)
        _, _, transfer, observed = self.evidence(*identity)
        if transfer.get("receipt"):
            return self.read(*identity), None
        if observed.state not in TERMINAL:
            raise ApplicationError("result recovery requires a terminal Attempt", code="ATTEMPT_NOT_TERMINAL")
        with self.state(*identity) as (store, snapshot):
            old = snapshot.value
            if old.get("status") in ACTIVE:
                return self.read_unlocked(identity, old), None
            if old.get("status") == "RECONCILE_REQUIRED":
                if not reconcile:
                    return self.read_unlocked(identity, old), None
                value = {**old, "status": "QUEUED", "reconcile": "cpu_profile" in old, "updated_at": utc_now()}
            else:
                if old and not retry:
                    return self.read_unlocked(identity, old), None
                count = old.get("attempts", 0) + 1
                if count > 3:
                    raise ApplicationError("result recovery retry budget exhausted", code="RESULT_RETRY_LIMIT")
                value = {"project": project, "run_id": run, "attempt_id": attempt,
                         "status": "QUEUED", "phase": "CHECKING", "attempts": count,
                         "scheduler_name": "ml-expd-result-" + hashlib.sha256(json.dumps([identity, count]).encode()).hexdigest()[:32],
                         "reconcile": False, "created_at": utc_now(), "updated_at": utc_now()}
            store.commit(value, event={"type": "RESULT_COLLECTION_REQUESTED"}, expected_revision=snapshot.revision)
        return self.read(*identity), identity

    @staticmethod
    def read_unlocked(identity, value):
        return {"project": identity[0], "run_id": identity[1], "attempt_id": identity[2],
                **{k: value.get(k) for k in ("status", "phase", "attempts", "scheduler_name", "diagnostic")}}

    def recover_interrupted(self):
        count = 0
        for path in self.root.glob("*/*/attempt-*.json"):
            with self.state(path.parent.parent.name, path.parent.name, path.stem) as (store, snapshot):
                if snapshot.value.get("status") in ACTIVE:
                    value = {**snapshot.value, "status": "RECONCILE_REQUIRED", "diagnostic": "DAEMON_INTERRUPTED",
                             "updated_at": utc_now()}
                    store.commit(value, event={"type": "RESULT_COLLECTION_INTERRUPTED"}, expected_revision=snapshot.revision)
                    count += 1
        return count

    def register_training(self, project, run, attempt, token, value):
        if (not isinstance(value, dict) or set(value) != {"contract", "project", "run_id", "attempt_id", "exit_code"}
                or value["contract"] != "training-result.v1"
                or (value["project"], value["run_id"], value["attempt_id"]) != (project, run, attempt)
                or type(value["exit_code"]) is not int or not -255 <= value["exit_code"] <= 255):
            raise ApplicationError("invalid training result", status_code=422, code="TRAINING_RESULT_INVALID")
        self.objects.authorize(project, run, attempt, token)
        with self.objects.record(project, run, attempt) as (path, transfer):
            if transfer.get("results_ready", value) != value:
                raise ValueError("training result is already frozen")
            from .storage import atomic_json
            transfer["results_ready"] = value
            atomic_json(path, transfer)
        return {"accepted": True}

    def automatic(self, submit):
        # Historical workers have no training-result record; they are never
        # enrolled automatically. A scientific completion rule is not inferred.
        for path in self.objects.root.glob("*/*/attempt-*.json"):
            transfer = json.loads(path.read_text())
            if transfer.get("receipt") or transfer.get("results_ready", {}).get("exit_code") != 0:
                continue
            identity = tuple(transfer[k] for k in ("project", "run_id", "attempt_id"))
            try:
                _, pending = self.begin(*identity)
            except ApplicationError:
                continue
            if pending is not None:
                try:
                    submit(self.finish, pending)
                except RuntimeError:
                    self.update(identity, status="RECONCILE_REQUIRED", diagnostic="EXECUTOR_UNAVAILABLE")

    def finish(self, identity):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Serialize recoveries to avoid keeping several large incomplete
        # archives plus their validation trees on the API host at once.
        with (self.root / "worker.lock").open("a") as lock:
            while True:
                if self.stopping():
                    self.update(identity, status="RECONCILE_REQUIRED", diagnostic="DAEMON_INTERRUPTED")
                    return
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.1)
            self.finish_one(identity)

    def finish_one(self, identity):
        phase = "CHECKING"
        try:
            root, definition, transfer, _ = self.evidence(*identity)
            with self.state(*identity) as (_, snapshot):
                value = snapshot.value
            backend = definition["backend"]
            local = self.local_outputs(root, identity[2])
            if backend["kind"] == "sensecore":
                rest = SenseCoreREST.from_environment()
                job = json.loads((root / "attempts" / identity[2] / "backend.json").read_text())["backend_job_id"]
                if rest.describe(backend, job)["state"] not in TERMINAL:
                    raise RecoveryError("ORIGINAL_JOB_NOT_TERMINAL", "provider Attempt is still running")
            elif backend["kind"] == "slurm":
                job = json.loads((root / "attempts" / identity[2] / "backend.json").read_text())["backend_job_id"]
                if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", str(job)) or not IDENTITY.fullmatch(backend["ssh_alias"]):
                    raise ValueError("invalid scheduler identity")
                result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                                         backend["ssh_alias"], "sacct -X -n -P -j " + str(job) + " --format=State"],
                                        check=True, capture_output=True, text=True, timeout=30)
                states = [line.split("|")[0].split()[0].split("+")[0] for line in result.stdout.splitlines() if line.strip()]
                if not states or any(state not in TERMINAL for state in states):
                    raise RecoveryError("ORIGINAL_JOB_NOT_TERMINAL", "provider Attempt is not confirmed terminal")
            # Preserve original capability, Run, job and pending upload identity.
            with self.objects.record(*identity) as (path, current):
                from .storage import atomic_json
                current["cache_outputs"] = False
                atomic_json(path, current)
            if local is None and backend["kind"] == "slurm":
                phase = "READING_SHARED_STORAGE"
                self.update(identity, status="COLLECTING", phase=phase)
                local = root / "attempts" / identity[2] / "recovered_outputs"
                remote = self.remote_outputs(definition, identity[2])
                alias = backend["ssh_alias"]
                temporary = Path(tempfile.mkdtemp(prefix=".result-recovery-", dir=local.parent))
                try:
                    subprocess.run(["rsync", "-a", "--protect-args", "--safe-links", "--exclude=.*",
                                    "-e", "ssh -o BatchMode=yes -o ConnectTimeout=15",
                                    alias + ":" + str(remote) + "/", str(temporary) + "/"],
                                   check=True, capture_output=True, timeout=300)
                    temporary.rename(local)
                finally:
                    remove_staging(temporary)
            if local is not None:
                phase = "VERIFYING_AND_PUBLISHING"
                self.update(identity, status="COLLECTING", phase=phase)
                self.publish_local(identity, local, transfer)
            elif backend["kind"] == "sensecore":
                phase = "CPU_NAS_RECOVERY"
                if not self.sensecore(identity, definition, transfer, value, rest):
                    return
            else:
                raise ValueError("backend does not support independent result collection")
            self.update(identity, status="AVAILABLE", phase="VERIFIED", diagnostic=None)
        except RESTError as error:
            uncertain = error.details["uncertain"] or self.provider_effect_pending(identity)
            self.update(identity, status="RECONCILE_REQUIRED" if uncertain else "FAILED",
                        phase=phase, diagnostic="PROVIDER_REQUEST_UNCERTAIN" if uncertain else "PROVIDER_REQUEST_FAILED")
        except Exception as error:
            self.update(identity, status="RECONCILE_REQUIRED" if self.provider_effect_pending(identity) else "FAILED",
                        phase=phase, diagnostic=getattr(error, "code", type(error).__name__))

    def provider_effect_pending(self, identity):
        with self.state(*identity) as (_, snapshot):
            value = snapshot.value
        return value.get("status") in {"SUBMITTING", "WAITING"} and value.get("job_state") not in TERMINAL

    @staticmethod
    def remote_outputs(definition, attempt):
        base = PurePosixPath(definition["storage"]["run_dir"])
        if not base.is_absolute() or ".." in base.parts:
            raise ValueError("invalid persistent output root")
        return base / "attempts" / attempt / "outputs"

    @staticmethod
    def local_outputs(root, attempt):
        parent = root / "attempts" / attempt
        for path in (parent / "outputs", parent / "recovered_outputs", parent / "collected_run" / "attempts" / attempt / "outputs"):
            if path.is_dir():
                return path
        return None

    def expected_archive(self, identity):
        found = []
        for path in (self.objects.root.parent / "multipart-uploads").glob("*/upload.json"):
            value = json.loads(path.read_text())
            if value.get("binding") == dict(zip(("project", "run_id", "attempt_id"), identity), kind="artifacts") and value["status"] == "UPLOADING":
                found.append({k: value[k] for k in ("sha256", "bytes")})
        if len({(v["sha256"], v["bytes"]) for v in found}) > 1:
            raise RecoveryError("ARCHIVE_IDENTITY_CONFLICT", "multiple conflicting unfinished result archives")
        return found[0] if found else None

    def publish_local(self, identity, outputs, transfer):
        from .result_worker import checked_directory
        checked_directory(outputs)
        with tempfile.TemporaryFile() as stream:
            count = archive_outputs(outputs, stream, self.objects.limit or 0, transfer["outputs"])
            if not count:
                raise RecoveryError("RESULT_FILES_NOT_FOUND", "Attempt has no declared outputs")
            length = stream.tell()
            stream.seek(0)
            expected = self.expected_archive(identity)
            if expected is not None and (sha(stream), length) != (expected["sha256"], expected["bytes"]):
                raise RecoveryError("OUTPUT_ARCHIVE_CHANGED", "local outputs differ from unfinished upload")
            stream.seek(0)
            self.objects.receive(*identity, transfer["token"], stream, length)

    def sensecore(self, identity, definition, transfer, operation, rest):
        if not self.runtime.config.action_runtime.allow_scheduler_mutations:
            raise ApplicationError("CPU result recovery is disabled by scheduler policy", code="RESULT_RECOVERY_BLOCKED")
        backend = definition["backend"]
        copy = {**self.objects.config["data_delivery"], "workspace": backend["workspace"],
                "storage_mount": backend["storage_mount"], "quota_type": "reserved", "worker_nodes": 1}
        if operation.get("reconcile"):
            copy = operation["cpu_profile"]
        if "debug" not in copy["aec2"].lower():
            raise ValueError("NAS recovery requires the configured debug CPU pool")
        DataDeliveryService(self.runtime).verify_cpu(copy)
        image = backend["image"]
        for path in sorted((self.objects.root.parent / "data-deliveries" / identity[0]).glob("delivery.*.json")):
            value = json.loads(path.read_text())
            scope = value.get("copy_profile", {})
            if value.get("status") == "READY" and all(scope.get(k) == copy[k] for k in ("workspace", "storage_mount")):
                image = value["image"]
                break
        checkpoints = CheckpointRegistry(Path(self.runtime.config.container_execution.artifact_store_file),
                                         self.objects.root.parent).list(*identity)["checkpoints"]
        checkpoint = max(checkpoints, key=lambda v: v["step"]) if checkpoints else None
        payload = {"outputs_root": str(self.remote_outputs(definition, identity[2])), "patterns": transfer["outputs"],
                   "checkpoint": checkpoint, "expected_archive": self.expected_archive(identity),
                   "url": self.objects.config["public_transfer_base"].rstrip("/") + "/" + "/".join(identity),
                   "token": transfer["token"], "limit": self.objects.limit or 0, "seconds": 900}
        sources = {name: Path(__file__).with_name(file).read_text() for name, file in
                   (("worker_http", "worker_http.py"), ("artifacts", "worker_artifacts.py"),
                    ("persistent_state", "persistent_state.py"), ("result_worker", "result_worker.py"))}
        # Use the existing Python image; no image build and no training entrypoint.
        script = "import sys,types\n" + "\n".join(
            f"m=types.ModuleType({name!r});sys.modules[{name!r}]=m;exec({source!r},m.__dict__)"
            for name, source in sources.items()) + "\nraise SystemExit(sys.modules['result_worker'].main(" + repr(payload) + "))"
        encoded = base64.b64encode(gzip.compress(script.encode(), mtime=0)).decode()
        command = ["python", "-u", "-c", "import base64,gzip;exec(gzip.decompress(base64.b64decode(" + repr(encoded) + ")))"]
        body = create_document(copy, operation["scheduler_name"], image, shlex.join(command), cpu_copy=True)
        body["roles"][0]["resource_spec"][0].update(requests={"cpu": "2", "memory": "3Gi"}, limits={"cpu": "2", "memory": "4Gi"})
        if operation.get("reconcile"):
            jobs = rest.find(copy, operation["scheduler_name"])
            if len(jobs) != 1:
                self.update(identity, status="RECONCILE_REQUIRED", phase="CPU_SUBMISSION", diagnostic="EXACT_CPU_JOB_UNCONFIRMED")
                return False
        else:
            self.update(identity, status="SUBMITTING", phase="CPU_SUBMISSION", cpu_profile=copy,
                        deadline_at=time.time() + 1500)
            rest.create(copy, body, timeout=120)
        self.update(identity, status="WAITING", phase="CPU_NAS_RECOVERY", cpu_profile=copy)
        with self.state(*identity) as (_, snapshot):
            deadline_at = snapshot.value["deadline_at"]
        deadline = time.monotonic() + max(0, deadline_at - time.time())
        while True:
            with self.objects.record(*identity) as (_, current):
                if current.get("receipt"):
                    return True
            job = rest.describe(copy, operation["scheduler_name"])
            self.update(identity, job_state=job["state"])
            if job["state"] in TERMINAL:
                raise RecoveryError("CPU_JOB_EXITED_WITHOUT_RECEIPT", "CPU result recovery ended without verified archive")
            if self.stopping():
                self.update(identity, status="RECONCILE_REQUIRED", diagnostic="DAEMON_INTERRUPTED")
                return False
            if time.monotonic() >= deadline:
                rest.stop(copy, operation["scheduler_name"])
                raise TimeoutError("CPU result collection exceeded its fixed budget")
            time.sleep(2)
