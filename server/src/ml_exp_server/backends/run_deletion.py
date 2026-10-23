"""Exact managed Run purge; no training submission or provider-history deletion."""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import time

from experiment_control.backends.sensecore_rest import SenseCoreREST

from ..application_errors import ApplicationError
from experiment_control.runner import run_process
from ..workers.delete_run import checked_target
from .sensecore_cpu import checked_job, describe_job, document, stop_job, validate_profile, verify_spec
from .states import TERMINAL


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def blocked(code, message):
    return ApplicationError(message, code=code)


def worker_terminal(value):
    if not isinstance(value, dict):
        return False
    if value.get("deleted") is True or value.get("worker_deleted") is True or value.get("workerDeleted") is True:
        return True
    status = value.get("status")
    candidates = [value.get("phase"), value.get("state"), status]
    if isinstance(status, dict):
        candidates.extend([status.get("phase"), status.get("state")])
    if any(isinstance(item, str) and item.upper() in {"SUCCEEDED", "FAILED", "COMPLETED", "DELETED", "TERMINATED", "STOPPED"}
           or isinstance(item, dict) and isinstance(item.get("terminated"), dict) for item in candidates):
        containers = value.get("containers", [])
        return isinstance(containers, list) and all(worker_terminal(item) for item in containers)
    return False


class BackendRunDeletion:
    def __init__(self, runtime, *, runner=None, rest=None, clock=None):
        self.runtime = runtime
        self.runner = runner or run_process
        self._rest = rest
        self.clock = clock or time.time

    @property
    def rest(self):
        if self._rest is None:
            self._rest = SenseCoreREST.from_environment()
        return self._rest

    def plan(self, run, attempts):
        backend = run["backend"]
        root = run["storage"]["project_data_root"]
        project = run.get("project", PurePosixPath(root).name)
        target = run["storage"]["run_dir"]
        checked_target({"project": project, "run_id": run["run_id"], "project_root": root,
                        "target": target, "plan_sha256": "0" * 64, "deletion_id": "plan"})
        kind = backend["kind"]
        if kind not in {"slurm", "sensecore"}:
            raise blocked("RUN_DELETION_BACKEND_UNSUPPORTED", "backend does not support managed Run deletion")
        profile = None
        if kind == "slurm":
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", backend["ssh_alias"]):
                raise ValueError("invalid managed Run SSH origin")
            executor = {"ssh_alias": backend["ssh_alias"]}
        else:
            config = self.runtime.config.container_execution.artifact_store_file
            configured = json.loads(Path(config).read_text()).get("data_delivery") if config else None
            if not configured:
                raise blocked("RUN_DELETION_CPU_UNAVAILABLE", "bounded CPU deletion profile is not configured")
            profile = {**configured, "workspace": backend["workspace"], "storage_mount": backend["storage_mount"],
                       "quota_type": "reserved", "worker_nodes": 1}
            validate_profile(profile, debug=True)
            mount = PurePosixPath(backend["storage_mount"].rsplit(":", 1)[-1])
            if not mount.is_absolute() or ".." in mount.parts or not PurePosixPath(target).is_relative_to(mount):
                raise ValueError("managed Run is outside frozen NAS mount")
            executor = {"workspace": backend["workspace"], "storage_mount": backend["storage_mount"]}
        image = run.get("container", {}).get("image") or backend.get("image")
        from ..builds.image_mirror import delivery_for_run, target_image
        delivery = delivery_for_run(run)
        if delivery is not None:
            image = target_image(delivery)
        if kind == "sensecore" and (not isinstance(image, str) or not image):
            raise blocked("RUN_DELETION_CPU_UNAVAILABLE", "frozen Python Runtime image is unavailable")
        return {"contract": "managed-run-deletion.v1", "backend": kind, "project": project,
                "run_id": run["run_id"], "project_root": root, "target": target,
                "executor": executor, "image": image, "cpu_profile": profile,
                **({'image_delivery': delivery} if delivery is not None else {}),
                "estimated_registered_bytes": None, "provider_history": {"status": "UNSUPPORTED"}}

    def terminal(self, run, attempts):
        backend = run["backend"]
        evidence = []
        for attempt in attempts:
            item = attempt if isinstance(attempt, dict) else attempt.model_dump(mode="json")
            identity, job = item["attempt_id"], item.get("backend_job_id")
            if not re.fullmatch(r"attempt-[0-9]{3,}", identity):
                raise ValueError("invalid Attempt identity")
            if not job:
                if not item.get("has_submission") and item.get("state") == "NOT_SUBMITTED":
                    evidence.append({"attempt_id": identity, "state": "NOT_SUBMITTED"})
                    continue
                raise blocked("RUN_DELETION_JOB_UNKNOWN", "Attempt submission is not positively resolved")
            if backend["kind"] == "slurm":
                if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", str(job)):
                    raise ValueError("invalid original scheduler identity")
                try:
                    result = self.runner(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", backend["ssh_alias"],
                                          shlex.join(["sacct", "-X", "-n", "-P", "-j", str(job), "--format=JobIDRaw,State"])],
                                         check=True, capture_output=True, text=True, timeout=30)
                except Exception:
                    raise blocked("RUN_DELETION_JOB_UNKNOWN", "original scheduler job observation is unavailable") from None
                rows = [line.split("|") for line in result.stdout.splitlines() if line.strip()]
                if (not rows or any(len(row) < 2 or row[0].strip() != str(job)
                                    or not row[1].strip()
                                    or row[1].strip().split()[0].split("+")[0] not in TERMINAL for row in rows)):
                    raise blocked("RUN_DELETION_JOB_NOT_TERMINAL", "original scheduler job is not positively terminal")
                states = sorted({row[1].strip().split()[0].split("+")[0] for row in rows})
                evidence.append({"attempt_id": identity, "backend_job_id": str(job), "states": states, "source": "sacct"})
            elif backend["kind"] == "sensecore":
                uid = item.get("backend_job_uid") or item.get("job_uid") or item.get("provider_uid")
                if not isinstance(uid, str) or not uid:
                    raise blocked("RUN_DELETION_JOB_UID_UNKNOWN", "original SenseCore UID is not durably confirmed")
                observed = describe_job(self.rest, backend, str(job), uid)
                if observed.get("state") not in TERMINAL:
                    raise blocked("RUN_DELETION_JOB_NOT_TERMINAL", "original SenseCore job is not positively terminal")
                workers = self.rest.workers(backend, str(job))
                if (not isinstance(workers, list) or any(not worker_terminal(worker) for worker in workers)
                        or not workers and observed["state"] not in {"SUCCEEDED", "FAILED", "DELETED", "COMPLETED"}
                        and not any(value.get("worker_deleted") is True or value.get("workerDeleted") is True
                                    for value in (item, observed))):
                    raise blocked("RUN_DELETION_WORKERS_UNCONFIRMED", "original SenseCore workers are not positively stopped")
                after = describe_job(self.rest, backend, str(job), uid)
                if after.get("state") not in TERMINAL:
                    raise blocked("RUN_DELETION_JOB_NOT_TERMINAL", "original SenseCore job terminal evidence changed")
                evidence.append({"attempt_id": identity, "backend_job_id": str(job), "job_uid": uid,
                                 "state": after["state"], "source": "sensecore_exact_uid",
                                 "workers_confirmed_terminal": True, "workers_observed": len(workers)})
            else:
                raise blocked("RUN_DELETION_BACKEND_UNSUPPORTED", "backend does not support managed Run deletion")
        return {"observed_at": self.clock(), "attempts": evidence}

    @staticmethod
    def payload(targets, state):
        if digest(targets) != state["plan_sha256"]:
            raise ValueError("backend deletion plan changed")
        value = {key: targets[key] for key in ("project", "run_id", "project_root", "target")}
        value.update(deletion_id=state["deletion_id"], plan_sha256=state["plan_sha256"], seconds=600)
        if targets["backend"] == "sensecore":
            value.update(receipt_url=state["receipt_url"], capability=state["capability"])
        return value

    @staticmethod
    def receipt(payload, value):
        keys = ("contract", "deletion_id", "project", "run_id", "plan_sha256", "target", "status", "target_absent",
                "files_deleted", "bytes_deleted")
        expected = {key: payload[key] for key in ("deletion_id", "project", "run_id", "plan_sha256", "target")}
        expected.update(contract="run-deletion.v1", status="DELETED", target_absent=True)
        if (not isinstance(value, dict) or set(value) != set(keys)
                or any(value.get(key) != item for key, item in expected.items())
                or value.get("target_absent") is not True
                or any(type(value.get(key)) is not int or value[key] < 0 for key in ("files_deleted", "bytes_deleted"))):
            raise ValueError("backend deletion receipt identity differs")
        return value

    def execute(self, targets, state, persist):
        """Advance one durable deletion; caller reloads state between observations."""
        payload = self.payload(targets, state)
        if state.get("stage") == "DELETED" and state.get("receipt"):
            return {"status": "DELETED", "receipt": self.receipt(payload, state["receipt"])}
        source = (Path(__file__).parent.parent / "workers" / "delete_run.py").read_text()

        def update(**fields):
            state.update(fields)
            persist(dict(state))  # Owning journal must merge a concurrent callback receipt.

        if targets["backend"] == "slurm":
            if state.get("receipt"):
                return {"status": "DELETED", "receipt": self.receipt(payload, state["receipt"])}
            payload["mode"] = "check" if state.get("stage") else "delete"
            update(stage="SSH_EXECUTING")
            try:
                result = self.runner(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                                      targets["executor"]["ssh_alias"], shlex.join(["python3", "-c", source])],
                                     check=True, capture_output=True, text=True, input=json.dumps(payload), timeout=630)
                if len(result.stdout) > 65536:
                    raise ValueError("oversized backend deletion receipt")
                receipt = self.receipt(payload, json.loads(result.stdout))
            except Exception:
                update(stage="RECONCILE_REQUIRED", diagnostic="SSH_RUN_DELETION_UNCONFIRMED")
                raise blocked("RUN_DELETION_RECONCILE_REQUIRED", "original SSH deletion requires exact target reconciliation") from None
            update(stage="DELETED", receipt=receipt, diagnostic=None)
            return {"status": "DELETED", "receipt": receipt}

        if targets["backend"] != "sensecore":
            raise blocked("RUN_DELETION_BACKEND_UNSUPPORTED", "backend does not support managed Run deletion")
        profile = targets["cpu_profile"]
        validate_profile(profile, debug=True)
        if not self.runtime.config.action_runtime.allow_scheduler_mutations:
            raise blocked("RUN_DELETION_CPU_DISABLED", "CPU deletion is disabled by scheduler policy")
        name = "ml-expd-delete-" + digest([state["deletion_id"], state["plan_sha256"]])[:32]
        if not state.get("stage"):
            if targets.get('image_delivery') is not None:
                from ..builds.image_mirror import ensure_delivery
                ensure_delivery(targets['image_delivery'])
            verify_spec(self.rest, profile)
            if self.rest.find(profile, name):
                raise blocked("RUN_DELETION_CPU_IDENTITY_CONFLICT", "deletion scheduler name already exists")
            script = source + "\nraise SystemExit(main(" + repr(payload) + "))\n"
            encoded = base64.b64encode(gzip.compress(script.encode(), mtime=0)).decode()
            command = shlex.join(["python3", "-u", "-c", "import base64,gzip;exec(gzip.decompress(base64.b64decode(" + repr(encoded) + ")),{'__name__':'ml_exp_run_deletion'})"])
            body = document(profile, name, targets["image"], command)
            update(stage="CPU_SUBMITTING", scheduler_name=name, deadline_at=self.clock() + 900)
            try:
                created = checked_job(self.rest.create(profile, body, timeout=60), name)
            except Exception:
                update(stage="RECONCILE_REQUIRED", diagnostic="CPU_RUN_DELETION_CREATE_UNCONFIRMED")
                raise blocked("RUN_DELETION_RECONCILE_REQUIRED", "original CPU deletion create is unconfirmed; never recreate") from None
            update(stage="CPU_WAITING", job_uid=created["uid"])
        if state.get("scheduler_name") != name or not state.get("job_uid"):
            raise blocked("RUN_DELETION_RECONCILE_REQUIRED", "original CPU deletion UID is unconfirmed; never recreate")
        try:
            job = describe_job(self.rest, profile, name, state["job_uid"])
        except Exception:
            update(stage="RECONCILE_REQUIRED", diagnostic="CPU_RUN_DELETION_UID_UNCONFIRMED")
            raise blocked("RUN_DELETION_RECONCILE_REQUIRED", "original CPU deletion identity requires reconciliation") from None
        update(job_state=job.get("state"))
        if state.get("receipt") and job.get("state") in TERMINAL:
            receipt = self.receipt(payload, state["receipt"])
            update(stage="DELETED", diagnostic=None)
            return {"status": "DELETED", "receipt": receipt}
        if self.clock() >= state["deadline_at"]:
            if job.get("state") not in TERMINAL:
                # UID guard is repeated by stop_job; the provider stop itself is name based.
                stop_job(self.rest, profile, name, state["job_uid"])
            update(stage="RECONCILE_REQUIRED", diagnostic="CPU_RUN_DELETION_RECEIPT_TIMEOUT")
            raise blocked("RUN_DELETION_RECONCILE_REQUIRED", "CPU deletion deadline exceeded without an accepted receipt")
        return {"status": "PENDING", "scheduler_name": name, "job_uid": state["job_uid"], "job_state": job.get("state")}
