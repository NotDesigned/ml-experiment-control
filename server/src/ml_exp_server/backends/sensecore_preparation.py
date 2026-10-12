"""Prepare download-script data on a bounded CPU job before GPU scheduling."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import re
import shlex
import time

from experiment_control.backends.sensecore_rest import SenseCoreREST, create_document

from ..application_errors import ApplicationError
from ..projects.source_imports import IDENTITY
from ..storage import DurableJsonState, utc_now
from ..workers.data_preparation import identity
from .sensecore_data import SenseCoreDataOperations
from .states import TERMINAL


PREFIX = "ML_EXPD_SCRIPT_READY="


class SenseCoreScriptPreparation:
    def __init__(self, config_file, registry_root, *, rest=None):
        self.config = json.loads(Path(config_file).read_text())
        self.root = Path(registry_root) / "data-script-preparations"
        self._rest = rest

    @property
    def rest(self):
        if self._rest is None:
            self._rest = SenseCoreREST.from_environment()
        return self._rest

    @contextmanager
    def state(self, project, operation):
        if not IDENTITY.fullmatch(project) or not re.fullmatch(r"script\.[0-9a-f]{64}", operation):
            raise ValueError("invalid script preparation identity")
        directory = self.root / project
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (directory / (operation + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            store = DurableJsonState(directory / (operation + ".json"), directory / (operation + ".jsonl"))
            snapshot = store.snapshot({})
            store.repair_journal(snapshot)
            yield store, snapshot

    def update(self, project, operation, **fields):
        with self.state(project, operation) as (store, snapshot):
            if snapshot.value and all(snapshot.value.get(key) == value for key, value in fields.items()):
                return snapshot.value
            value = {**snapshot.value, **fields, "updated_at": utc_now()}
            store.commit(value, expected_revision=snapshot.revision,
                         event={"event": "script_preparation", "status": value["status"], "timestamp": value["updated_at"]})
            return value

    @staticmethod
    def public(value):
        return {key: value[key] for key in ("operation_id", "preparation_id", "scheduler_name", "status", "job_state", "receipt", "diagnostic", "updated_at") if key in value}

    def reference(self, run):
        return {"operation_id": "script." + identity(self.definition(run)),
                "preparation_id": run["data_preparation"]["preparation_id"]}

    def read(self, project, operation):
        with self.state(project, operation) as (_, snapshot):
            return self.public(snapshot.value)

    def definition(self, run):
        backend = run["backend"]
        configured = run.get("data_preparation_cpu", self.config.get("data_delivery"))
        if backend["kind"] != "sensecore" or not configured:
            raise ApplicationError("CPU script preparation is not configured", code="DATA_SCRIPT_DISABLED")
        copy = {**configured, "workspace": backend["workspace"], "storage_mount": backend["storage_mount"],
                "quota_type": "reserved", "worker_nodes": 1}
        if ("debug" not in copy["aec2"].casefold()
                or not IDENTITY.fullmatch(copy["aec2"])
                or not re.fullmatch(r"[A-Za-z0-9_.-]+\.2c4g", copy.get("worker_spec", ""))
                or any(type(copy.get(key)) is not int or copy[key] != expected
                       for key, expected in (("gpus", 0), ("cpus", 2), ("memory_gb", 4)))
                or type(copy.get("queue_timeout_seconds")) is not int
                or not 5 <= copy["queue_timeout_seconds"] <= 3600
                or type(run["data_preparation"]["timeout_seconds"]) is not int
                or not 5 <= run["data_preparation"]["timeout_seconds"] <= 3600):
            raise ApplicationError("download scripts require the configured debug CPU pool", code="DATA_SCRIPT_CPU_PROFILE")
        cache = run["storage"]["project_data_root"].rstrip("/") + "/data-preparations"
        mount = Path(backend["storage_mount"].rsplit(":", 1)[1])
        if not Path(cache).is_absolute() or ".." in Path(cache).parts or not Path(cache).is_relative_to(mount):
            raise ValueError("script cache is outside the frozen NAS mount")
        return {"contract": "cpu-script-preparation.v1", "definition": run["data_preparation"],
                "image": run["container"]["image"], "cache": cache, "cpu_profile": copy}

    @staticmethod
    def command(definition):
        # The helper and script are read from the immutable training image.
        code = ("import json,sys;from pathlib import Path;sys.path.insert(0,'/usr/local/lib/ml-expd');"
                "from data_preparation import prepare;"
                "from data_input import deliver;from worker import link_path;"
                "d=json.loads(sys.argv[1]);"
                "[link_path(deliver({**i,'require_cached':True},'',Path(d['cache']).parent/'data-assets'),Path(i['mount_path'])) for i in d['definition']['inputs']];"
                "p,r=prepare(d['definition'],Path(d['cache']));"
                "print('" + PREFIX + "'+json.dumps({k:r[k] for k in "
                "('preparation_id','content_sha256','dataset_id','bytes')},sort_keys=True),flush=True)")
        seconds = definition["definition"]["timeout_seconds"] + 60
        return shlex.join(["timeout", "--signal=TERM", "--kill-after=30s", str(seconds) + "s",
                           "python3", "-u", "-c", code, json.dumps(definition, sort_keys=True)])

    def receipt(self, profile, name, preparation_id):
        logs = self.rest.logs(profile, name, 200)
        for line in reversed(logs.get("text", "").splitlines()):
            if not line.startswith(PREFIX):
                continue
            value = json.loads(line[len(PREFIX):])
            content = value.get("content_sha256", "")
            if (value.get("preparation_id") != preparation_id or not re.fullmatch(r"[0-9a-f]{64}", content)
                    or value.get("dataset_id") != "dataset." + content
                    or type(value.get("bytes")) is not int or value["bytes"] < 0):
                raise ValueError("CPU script receipt identity differs")
            return {key: value[key] for key in ("preparation_id", "content_sha256", "dataset_id", "bytes")}
        return None

    def ensure(self, project, run, *, require_cached=False):
        definition = self.definition(run)
        operation = "script." + identity(definition)
        preparation = definition["definition"]["preparation_id"]
        with self.state(project, operation) as (_, snapshot):
            value = snapshot.value
        if value and (value.get("operation_id") != operation or value.get("preparation_id") != preparation or value.get("definition") != definition):
            raise ValueError("stored script preparation identity differs")
        if value.get("status") == "READY":
            return self.public(value)
        if require_cached:
            raise ApplicationError("download-script data is not verified on the backend", code="DATA_SCRIPT_NOT_READY")
        # A second caller can observe later; it must never submit concurrently.
        worker = self.root / project / (operation + ".worker.lock")
        with worker.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ApplicationError("original CPU data preparation is active; observe it", code="DATA_SCRIPT_ACTIVE") from None
            with self.state(project, operation) as (_, snapshot):
                value = snapshot.value
            if value.get("status") == "READY":
                return self.public(value)
            profile = definition["cpu_profile"]
            name = "ml-expd-data-script-" + operation.split(".", 1)[1][:32]
            try:
                if not value:
                    SenseCoreDataOperations.verify_cpu(self, profile)
                    body = create_document(profile, name, definition["image"], self.command(definition), cpu_copy=True)
                    body["roles"][0]["resource_spec"][0].update(requests={"cpu": "2", "memory": "3Gi"}, limits={"cpu": "2", "memory": "4Gi"})
                    value = self.update(project, operation, operation_id=operation, preparation_id=preparation,
                                        scheduler_name=name, status="SUBMITTING", definition=definition,
                                        deadline_at=time.time() + profile["queue_timeout_seconds"] + definition["definition"]["timeout_seconds"] + 90)
                    # Durable intent is recorded before the only create call.
                    created = self.rest.create(profile, body, timeout=60)
                    if created.get("name") != name or not isinstance(created.get("uid"), str) or not created["uid"]:
                        raise ValueError("CPU preparation create identity differs")
                    value = self.update(project, operation, job_uid=created["uid"])
                elif value["definition"] != definition or value["status"] == "FAILED":
                    raise ApplicationError("inspect the original failed CPU preparation", code="DATA_SCRIPT_FAILED")
                jobs = self.rest.find(profile, name)
                if len(jobs) != 1:
                    raise ApplicationError("original CPU submission is unconfirmed; never replay it", code="DATA_SCRIPT_SUBMISSION_UNKNOWN")
                uid = jobs[0].get("uid")
                if not isinstance(uid, str) or not uid or value.get("job_uid", uid) != uid:
                    raise ValueError("CPU preparation job identity differs")
                value = self.update(project, operation, job_uid=uid)
                while True:
                    job = self.rest.describe(profile, name)
                    if job.get("uid") != uid:
                        raise ValueError("CPU preparation job identity differs")
                    self.update(project, operation, status="WAITING", job_state=job["state"])
                    if job["state"] == "SUCCEEDED":
                        receipt = self.receipt(profile, name, preparation)
                        if receipt is not None:
                            return self.public(self.update(project, operation, status="READY", receipt=receipt, diagnostic=None))
                    elif job["state"] in TERMINAL:
                        self.update(project, operation, status="FAILED", diagnostic="CPU_DATA_SCRIPT_EXITED")
                        raise ApplicationError("CPU data preparation failed; inspect its exact job", code="DATA_SCRIPT_FAILED")
                    if time.time() >= value["deadline_at"]:
                        if job["state"] not in TERMINAL:
                            self.rest.stop(profile, name)
                        raise ApplicationError("CPU data preparation timed out or its receipt is pending", code="DATA_SCRIPT_TIMEOUT")
                    time.sleep(2)
            except Exception as error:
                with self.state(project, operation) as (_, snapshot):
                    current = snapshot.value
                code = error.code if isinstance(error, ApplicationError) else "DATA_SCRIPT_PROVIDER_ERROR"
                self.update(project, operation, operation_id=operation, preparation_id=preparation,
                            scheduler_name=name, status="FAILED" if current.get("status") == "FAILED" else "RECONCILE_REQUIRED",
                            definition=definition, diagnostic=code)
                raise ApplicationError("CPU data preparation requires inspection of its original job", code=code) from None
