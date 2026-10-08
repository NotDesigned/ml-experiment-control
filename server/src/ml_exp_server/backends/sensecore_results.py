"""SenseCoreResultOperations; no training submission or replay."""
from __future__ import annotations
import base64
import gzip
import json
from pathlib import Path
import shlex
import time
from experiment_control.backends.sensecore_rest import create_document
from ..application_errors import ApplicationError
from ..results.checkpoint_registry import CheckpointRegistry
from ..data.data_delivery import DataDeliveryService

from ..results.errors import RecoveryError
from .states import TERMINAL


class SenseCoreResultOperations:
    """Provider-specific operations sharing the owning service state."""

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
