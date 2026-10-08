"""SenseCoreResultOperations; no training submission or replay."""
from __future__ import annotations
import base64
import gzip
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import shlex
import time
from experiment_control.backends.sensecore_rest import create_document, SenseCoreREST, RESTError, component
from experiment_control.redaction import redact_line
from ..application_errors import ApplicationError
from ..results.checkpoint_registry import CheckpointRegistry
from ..data.data_delivery import DataDeliveryService

from ..results.errors import RecoveryError
from .states import TERMINAL


class SenseCoreResultOperations:
    """Provider-specific operations sharing the owning service state."""

    def result_job_diagnostics(self, identity, *, refresh=False):
        """Read the exact recovery job, without submitting or retrying compute."""
        _, definition, transfer, _ = self.evidence(*identity)
        with self.state(*identity) as (_, snapshot):
            operation = snapshot.value
        name, profile = operation.get("scheduler_name"), operation.get("cpu_profile")
        result = {"scheduler_name": name, "job_state": operation.get("job_state"),
                  "job_exit_code": None, "exit_code_conflict": False, "events": [],
                  "logs": {"status": "PENDING", "source": None, "historical": None,
                           "truncated": False, "lines": [], "unavailable_reason": None}, "errors": []}
        if definition["backend"]["kind"] != "sensecore" or not name or not profile:
            result["logs"].update(status="NOT_APPLICABLE")
            return result
        cached = operation.get("cpu_diagnostics")
        if not refresh:
            return cached if isinstance(cached, dict) and cached.get("scheduler_name") == name else result
        try:
            rest = SenseCoreREST.from_environment()
        except RESTError:
            result["logs"].update(status="UNAVAILABLE", unavailable_reason="CPU_CONFIGURATION_UNAVAILABLE")
            return result
        secrets = [transfer["token"]] + [rest.config.get(key) for key in ("access_key_id", "access_key_secret")]
        return self.collect_result_job_diagnostics(rest, profile, name, secrets, result)

    @staticmethod
    def collect_result_job_diagnostics(rest, profile, name, secrets, result):
        """Provider stdout/stderr share a stream; absent container codes stay unknown."""
        component(name)

        def safe_text(value):
            if not isinstance(value, str):
                return ""
            for secret in secrets:
                if isinstance(secret, str) and secret:
                    value = value.replace(secret, "[REDACTED]")
            value = redact_line(value)
            value = re.sub(r"https?://[^\s'\"]+", "[URL REDACTED]", value, flags=re.I)
            value = re.sub(r"[A-Za-z0-9+/=_-]{128,}", "[ENCODED REDACTED]", value)
            return value[:2048]

        def failed(operation, error):
            result["errors"].append({"operation": operation, "error": type(error).__name__,
                                     "http_status": error.status if isinstance(error, RESTError) else None})

        result["observed_at"] = datetime.now(timezone.utc).isoformat()
        try:
            job = rest.describe(profile, name)
            if not isinstance(job, dict):
                raise ValueError("invalid recovery job observation")
        except (RESTError, ValueError) as error:
            failed("describe", error)
            result["logs"].update(status="UNAVAILABLE", unavailable_reason="CPU_JOB_QUERY_FAILED")
            return result
        state = job.get("state")
        result["job_state"] = state if isinstance(state, str) and state in TERMINAL | {"PENDING", "STARTING", "RUNNING", "QUEUING"} else None
        codes = set()

        def exit_codes(value):
            if not isinstance(value, dict):
                return
            for key in ("exit_code", "exitCode"):
                code = value.get(key)
                if type(code) is int and -255 <= code <= 255:
                    codes.add(code)
            for key in ("state", "last_state", "lastState", "terminated", "status"):
                child = value.get(key)
                if isinstance(child, dict):
                    exit_codes(child)

        exit_codes(job)
        try:
            workers = rest.workers(profile, name)
            if not isinstance(workers, list) or any(not isinstance(worker, dict) for worker in workers):
                raise ValueError("invalid recovery worker observation")
            for worker in workers:
                exit_codes(worker)
                for container in worker.get("containers") or []:
                    exit_codes(container)
        except (RESTError, ValueError) as error:
            failed("workers", error)
        try:
            response = rest.request(rest.jobs_url(profile) + "/" + name + "/events")
            if not isinstance(response, dict) or not isinstance(response.get("events", []), list):
                raise ValueError("invalid recovery event observation")
            events = response.get("events", [])
            result["events"] = [{"reason": safe_text(event.get("reason")),
                                 "type": safe_text(event.get("type")),
                                 "message": safe_text(event.get("message"))}
                                for event in events[-30:] if isinstance(event, dict)]
        except (RESTError, ValueError) as error:
            failed("events", error)
        result["job_exit_code"] = next(iter(codes)) if len(codes) == 1 else None
        result["exit_code_conflict"] = len(codes) > 1
        try:
            logs = rest.logs(profile, name, 200)
            if not isinstance(logs, dict):
                raise ValueError("invalid recovery log observation")
            text = logs.get("text", "")
            lines = [safe_text(line) for line in text.splitlines()[-200:]] if isinstance(text, str) else []
            available = bool(lines) and logs.get("available", True)
            # Offline indexing may lag after worker removal. Empty evidence is
            # pending; a failed/forbidden query is a distinct unavailable state.
            unavailable = bool(logs.get("error"))
            result["logs"] = {"status": "AVAILABLE" if available else "UNAVAILABLE" if unavailable else "PENDING",
                              "source": "offline" if logs.get("historical") else "live", "historical": bool(logs.get("historical")),
                              "truncated": bool(logs.get("truncated")), "lines": lines,
                              "unavailable_reason": None if available else safe_text(logs.get("unavailable_reason")) or None}
        except (RESTError, ValueError) as error:
            failed("logs", error)
            result["logs"].update(status="UNAVAILABLE", unavailable_reason="CPU_LOG_QUERY_FAILED")
        return result

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
        sources = {name: (Path(__file__).parent.parent / "workers" / file).read_text() for name, file in
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
