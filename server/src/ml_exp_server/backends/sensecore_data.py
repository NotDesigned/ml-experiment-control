"""SenseCoreDataOperations; no training submission or replay."""
from __future__ import annotations
import shlex
from urllib.parse import urlsplit
from experiment_control.backends.sensecore_rest import SenseCoreREST, create_document

class SenseCoreDataOperations:
    """Provider-specific operations sharing the owning service state."""

    @property
    def rest(self):
        if not hasattr(self, "_rest"):
            self._rest = SenseCoreREST.from_environment()
        return self._rest

    def find(self, value):
        matches = self.rest.find(value["copy_profile"], value["scheduler_name"])
        if len(matches) > 1:
            raise ValueError("ambiguous data-copy scheduler identity")
        return matches[0] if matches else None

    def verify_cpu(self, copy):
        specs = [r for r in self.rest.specs(copy) if r["name"] == copy["worker_spec"]]
        if len(specs) != 1:
            raise ValueError("ACP CPU-only data-copy spec is unavailable")
        spec = specs[0]
        if (spec["device"]["number"], spec["cpu"]["vcpu_allocatable"], spec["memory"]["allocatable"]) != (0, 2, 4):
            raise ValueError("ACP data-copy spec unexpectedly allocates GPUs or different CPU resources")

    def create_document(self, value):
        copy = value["copy_profile"]
        endpoint = urlsplit(self.assets.objects.config["public_transfer_base"])
        if (endpoint.scheme != "https" or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment
                or not endpoint.path.endswith("/api/artifact-transfers")):
            raise ValueError("data copy callback requires a fixed HTTPS endpoint")
        prefix = endpoint.path.removesuffix("/api/artifact-transfers")
        url = endpoint.scheme + "://" + endpoint.netloc + prefix + "/api/data-copy-transfers/" + value["project"] + "/" + value["delivery_id"]
        command = ["env", "ML_EXPD_DATA_COPY_URL=" + url, "ML_EXPD_DATA_COPY_TOKEN=" + value["copy_token"],
                   "ML_EXPD_DATA_COPY_ROOT=" + copy["data_root"], "ML_EXPD_DATA_COPY_IMAGE=" + value["image"],
                   "ML_EXPD_DATA_COPY_SECONDS=" + str(copy["copy_timeout_seconds"]),
                   "python", "/usr/local/lib/ml-expd/data_copy_worker.py"]
        body = create_document(copy, value["scheduler_name"], value["image"], shlex.join(command), cpu_copy=True)
        body["roles"][0]["resource_spec"][0].update(
            requests={"cpu": "2", "memory": "3Gi"}, limits={"cpu": "2", "memory": "4Gi"})
        return body
