"""SenseCoreDataOperations; no training submission or replay."""
from __future__ import annotations
import shlex
from urllib.parse import urlsplit
from experiment_control.backends.sensecore_rest import SenseCoreREST
from .sensecore_cpu import document, find_job, verify_spec

class SenseCoreDataOperations:
    """Provider-specific operations sharing the owning service state."""

    @property
    def rest(self):
        if not hasattr(self, "_rest"):
            self._rest = SenseCoreREST.from_environment()
        return self._rest

    def find(self, value):
        return find_job(self.rest, value["copy_profile"], value["scheduler_name"], value.get("job_uid"))

    def verify_cpu(self, copy):
        verify_spec(self.rest, copy)

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
        return document(copy, value["scheduler_name"], value["image"], shlex.join(command))
