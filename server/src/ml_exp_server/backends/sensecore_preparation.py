"""Prepare download-script data on a bounded CPU job before GPU scheduling."""
from __future__ import annotations

from contextlib import contextmanager
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

from experiment_control.backends.sensecore_rest import SenseCoreREST

from ..application_errors import ApplicationError
from ..projects.source_imports import IDENTITY
from ..storage import DurableJsonState, utc_now
from ..workers.data_preparation import identity
from .sensecore_cpu import checked_job, describe_job, document, find_job, stop_job, validate_profile, verify_spec
from .states import TERMINAL


PREFIX = "ML_EXPD_SCRIPT_READY="
SCRIPT_RECEIPT_PYTHON = ("python3", "-I", "-S", "-u", "-c")
CALLBACK_PATH = re.compile(r"^/api/data-script-transfers/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/script\.[0-9a-f]{64}$")


def script_receipt_worker_sha256():
    code = (Path(__file__).parent.parent / "workers" / "script_preparation.py").read_bytes()
    launcher = b"\0".join(argument.encode() for argument in SCRIPT_RECEIPT_PYTHON)
    return hashlib.sha256(launcher + b"\0" + code).hexdigest()


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
            if snapshot.value.get("status") == "READY" and fields.get("status", "READY") != "READY":
                return snapshot.value
            value = {**snapshot.value, **fields, "updated_at": utc_now()}
            store.commit(value, expected_revision=snapshot.revision,
                         event={"event": "script_preparation", "status": value["status"], "timestamp": value["updated_at"]})
            return value

    @staticmethod
    def public(value):
        return {key: value[key] for key in ("operation_id", "preparation_id", "scheduler_name", "status", "job_state", "job_exit_code", "job_observed_at", "receipt", "receipt_source", "control_diagnostic", "diagnostic", "updated_at") if key in value}

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
        try:
            validate_profile(copy, timeouts=("queue_timeout_seconds",), debug=True)
            seconds = run["data_preparation"]["timeout_seconds"]
            if type(seconds) is not int or not 5 <= seconds <= 3600:
                raise ValueError("script duration is outside the CPU preparation budget")
        except ValueError:
            raise ApplicationError("download scripts require the configured debug CPU pool", code="DATA_SCRIPT_CPU_PROFILE") from None
        cache = run["storage"]["project_data_root"].rstrip("/") + "/data-preparations"
        mount = Path(backend["storage_mount"].rsplit(":", 1)[1])
        if not Path(cache).is_absolute() or ".." in Path(cache).parts or not Path(cache).is_relative_to(mount):
            raise ValueError("script cache is outside the frozen NAS mount")
        definition = {"contract": "cpu-script-preparation.v1", "definition": run["data_preparation"],
                      "image": run["container"]["image"], "cache": cache, "cpu_profile": copy}
        if run.get("data_preparation_receipt_mode") == "callback_v1":
            worker_sha = run.get("data_preparation_receipt_worker_sha256")
            if not isinstance(worker_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", worker_sha):
                raise ValueError("script receipt worker must be frozen")
            definition.update(contract="cpu-script-preparation.v2", receipt_worker_sha256=worker_sha)
        return definition

    @staticmethod
    def command(definition, callback=None):
        if definition["contract"] == "cpu-script-preparation.v2":
            if definition["receipt_worker_sha256"] != script_receipt_worker_sha256() or callback is None:
                raise ValueError("frozen script receipt worker differs")
            code = (Path(__file__).parent.parent / "workers" / "script_preparation.py").read_text()
            return shlex.join([
                "timeout", "--signal=TERM", "--kill-after=30s", str(definition["definition"]["timeout_seconds"] + 60) + "s",
                "env", "ML_EXPD_SCRIPT_RECEIPT_TOKEN=" + callback["token"],
                "ML_EXPD_SCRIPT_RECEIPT_REQUEST=" + json.dumps(callback["request"], sort_keys=True),
                *SCRIPT_RECEIPT_PYTHON, code, json.dumps(definition, sort_keys=True),
            ])
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

    def callback_request(self, project, operation, definition, name, token):
        prefix = self.config.get("public_transfer_base", "").rstrip("/").rsplit("/", 1)[0]
        target = urlsplit(prefix)
        if target.scheme != "https" or not target.hostname or target.port == 0 or target.username or target.password or target.query or target.fragment:
            raise ValueError("script callback HTTPS origin is invalid")
        return {"token": token, "request": {
            "url": prefix + "/data-script-transfers/" + project + "/" + operation,
            "project": project, "operation_id": operation, "scheduler_name": name,
            "definition_sha256": identity(definition),
            "source_id": definition["definition"]["source_id"],
            "script_sha256": definition["definition"]["script_sha256"],
            "image": definition["image"], "cpu_profile_sha256": identity(definition["cpu_profile"]),
        }}

    @staticmethod
    def validate_callback(project, operation, value, payload):
        if not isinstance(payload, dict):
            raise ValueError("script callback must be an object")
        definition = value.get("definition", {})
        receipt = payload.get("receipt")
        if (value.get("operation_id") != operation or operation != "script." + identity(definition)
                or definition.get("contract") != "cpu-script-preparation.v2"
                or value.get("scheduler_name") != "ml-expd-data-script-" + operation.split(".", 1)[1][:32]
                or any(payload.get(key) != expected for key, expected in {
                    "project": project, "operation_id": operation,
                    "scheduler_name": value.get("scheduler_name"), "definition_sha256": identity(definition),
                    "source_id": definition["definition"]["source_id"],
                    "script_sha256": definition["definition"]["script_sha256"],
                    "image": definition["image"], "cpu_profile_sha256": identity(definition["cpu_profile"]),
                }.items())
                or not isinstance(receipt, dict)):
            raise ValueError("script callback operation identity differs")
        files = receipt.get("files")
        if not isinstance(files, list) or not 1 <= len(files) <= 20000:
            raise ValueError("script callback inventory is invalid")
        names = set()
        for item in files:
            if (not isinstance(item, dict) or not isinstance(item.get("path"), str)
                    or not item["path"] or Path(item["path"]).is_absolute()
                    or ".." in Path(item["path"]).parts or item["path"] in names
                    or type(item.get("bytes")) is not int or item["bytes"] < 0
                    or not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
                raise ValueError("script callback file identity differs")
            names.add(item["path"])
        content = identity(files)
        if (receipt.get("definition") != definition["definition"]
                or receipt.get("preparation_id") != value.get("preparation_id")
                or receipt.get("schema_version") != 1 or receipt.get("status") != "READY"
                or receipt.get("content_sha256") != content or receipt.get("dataset_id") != "dataset." + content
                or type(receipt.get("bytes")) is not int or receipt["bytes"] != sum(item["bytes"] for item in files)):
            raise ValueError("script callback sealed receipt differs")
        return receipt

    def authorize_callback(self, project, operation, token):
        with self.state(project, operation) as (_, snapshot):
            value = snapshot.value
        digest = value.get("callback_token_sha256")
        if (not isinstance(token, str) or not isinstance(digest, str)
                or not hmac.compare_digest(digest, hashlib.sha256(token.encode()).hexdigest())
                or time.time() >= value.get("callback_expires_at", 0)):
            raise ValueError("invalid or expired script capability")
        return value

    def callback(self, project, operation, token, payload):
        value = self.authorize_callback(project, operation, token)
        receipt = self.validate_callback(project, operation, value, payload)
        with self.state(project, operation) as (store, snapshot):
            current = snapshot.value
            if current.get("callback_receipt") is not None and current["callback_receipt"] != receipt:
                raise ValueError("script callback receipt is already sealed")
            if current.get("callback_receipt") is None:
                current = {**current, "callback_receipt": receipt, "callback_received_at": utc_now()}
                store.commit(current, expected_revision=snapshot.revision,
                             event={"event": "script_receipt_received", "timestamp": utc_now()})
        confirmed = self.confirm_callback(project, operation)
        return {"operation_id": operation, "preparation_id": current["preparation_id"],
                "status": "READY" if confirmed is not None else "PENDING_UID"}

    def confirm_callback(self, project, operation):
        with self.state(project, operation) as (_, snapshot):
            value = snapshot.value
        if value.get("status") == "READY":
            return self.public(value)
        receipt, uid = value.get("callback_receipt"), value.get("job_uid")
        if receipt is None or uid is None:
            return None
        definition = value["definition"]
        self.validate_callback(project, operation, value, {
            "project": project, "operation_id": operation, "scheduler_name": value["scheduler_name"],
            "definition_sha256": identity(definition), "source_id": definition["definition"]["source_id"],
            "script_sha256": definition["definition"]["script_sha256"], "image": definition["image"],
            "cpu_profile_sha256": identity(definition["cpu_profile"]), "receipt": receipt,
        })
        job = describe_job(self.rest, definition["cpu_profile"], value["scheduler_name"], uid)
        return self.public(self.update(project, operation, status="READY", receipt={
            key: receipt[key] for key in ("preparation_id", "content_sha256", "dataset_id", "bytes")
        }, job_uid=uid, job_state=job["state"], job_observed_at=utc_now(),
            job_exit_code=job.get("exit_code") if type(job.get("exit_code")) is int else None,
            control_diagnostic=value.get("diagnostic"), diagnostic=None, receipt_source="NAS_SEALED_CALLBACK"))

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
                    verify_spec(self.rest, profile)
                    deadline = time.time() + profile["queue_timeout_seconds"] + definition["definition"]["timeout_seconds"] + 90
                    callback = None
                    capability = {}
                    if definition["contract"] == "cpu-script-preparation.v2":
                        token = secrets.token_urlsafe(32)
                        callback = self.callback_request(project, operation, definition, name, token)
                        capability = {"callback_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                                      "callback_expires_at": deadline + 300}
                    body = document(profile, name, definition["image"], self.command(definition, callback))
                    value = self.update(project, operation, operation_id=operation, preparation_id=preparation,
                                        scheduler_name=name, status="SUBMITTING", definition=definition,
                                        deadline_at=deadline, **capability)
                    # Durable intent is recorded before the only create call.
                    created = self.rest.create(profile, body, timeout=60)
                    checked_job(created, name)
                    value = self.update(project, operation, job_uid=created["uid"])
                elif value["definition"] != definition or value["status"] == "FAILED":
                    raise ApplicationError("inspect the original failed CPU preparation", code="DATA_SCRIPT_FAILED")
                job = find_job(self.rest, profile, name, value.get("job_uid"))
                if job is None:
                    raise ApplicationError("original CPU submission is unconfirmed; never replay it", code="DATA_SCRIPT_SUBMISSION_UNKNOWN")
                uid = job["uid"]
                value = self.update(project, operation, job_uid=uid)
                while True:
                    confirmed = self.confirm_callback(project, operation) if definition["contract"] == "cpu-script-preparation.v2" else None
                    if confirmed is not None:
                        return confirmed
                    job = describe_job(self.rest, profile, name, uid)
                    self.update(project, operation, status="WAITING", job_state=job["state"])
                    if job["state"] == "SUCCEEDED" and definition["contract"] == "cpu-script-preparation.v1":
                        receipt = self.receipt(profile, name, preparation)
                        if receipt is not None:
                            return self.public(self.update(project, operation, status="READY", receipt=receipt, diagnostic=None))
                    elif job["state"] in TERMINAL and job["state"] != "SUCCEEDED":
                        self.update(project, operation, status="FAILED", diagnostic="CPU_DATA_SCRIPT_EXITED")
                        raise ApplicationError("CPU data preparation failed; inspect its exact job", code="DATA_SCRIPT_FAILED")
                    if time.time() >= value["deadline_at"]:
                        if job["state"] not in TERMINAL:
                            stop_job(self.rest, profile, name, uid)
                        raise ApplicationError("CPU data preparation timed out or its receipt is pending", code="DATA_SCRIPT_TIMEOUT")
                    time.sleep(2)
            except Exception as error:
                with self.state(project, operation) as (_, snapshot):
                    current = snapshot.value
                if current.get("status") == "READY":
                    return self.public(current)
                code = error.code if isinstance(error, ApplicationError) else "DATA_SCRIPT_PROVIDER_ERROR"
                self.update(project, operation, operation_id=operation, preparation_id=preparation,
                            scheduler_name=name, status="FAILED" if current.get("status") == "FAILED" else "RECONCILE_REQUIRED",
                            definition=definition, diagnostic=code)
                raise ApplicationError("CPU data preparation requires inspection of its original job", code=code) from None
