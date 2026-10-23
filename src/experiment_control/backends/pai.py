"""Single-node DLC scheduling; provider restarts and purchase fallback disabled."""
from __future__ import annotations

import hashlib
import json
import re
import shlex

from .capabilities import PAI
from .services import BackendServices
from .pai_api import PaiAPI, PaiError, component
from .sensecore import digest_pinned_image, scheduler_job_name, submission_resource_name
from ..identity import IdentityReport
from ..preflight import PreflightCheck, PreflightReport
from ..redaction import redact_line
from ..submission import require_submission_intent


ALLOCATION_POLICIES = {"guaranteed": "ForbiddenQuotaOverSold", "allow_idle": "AcceptQuotaOverSold",
                       "idle_only": "ForceQuotaOverSold"}


def normalized(raw):
    return {"Creating": "QUEUED", "Queuing": "QUEUED", "Bidding": "QUEUED",
            "Dequeued": "STARTING", "WaitingConfig": "STARTING", "EnvPreparing": "STARTING",
            "SanityChecking": "STARTING", "Running": "RUNNING", "Restarting": "RUNNING",
            "Stopping": "UNKNOWN", "SucceededReserving": "SUCCEEDED", "Succeeded": "SUCCEEDED",
            "FailedReserving": "FAILED", "Failed": "FAILED", "Stopped": "CANCELLED"}.get(raw, "UNKNOWN")


def job_tags(run, attempt, token=None):
    tags = {"ml-expd.run": run["run_id"], "ml-expd.attempt": attempt,
            "ml-expd.scope": hashlib.sha256(json.dumps(run["backend"], sort_keys=True).encode()).hexdigest()}
    if token:
        tags["ml-expd.submission"] = token
    return tags


def observed_tags(job):
    settings = job.get("Settings")
    tags = settings.get("Tags") if isinstance(settings, dict) else None
    if not isinstance(tags, dict) or not all(tags.get(k) for k in ("ml-expd.run", "ml-expd.attempt", "ml-expd.scope")):
        raise ValueError("PAI job identity tags are unavailable; absence is not proven")
    return tags


def personal_mounts(backend):
    """Operator-approved personal CPFS plus fixed read-only public directories."""
    if "cpfs_uri" not in backend:
        return []
    uri, user = backend["cpfs_uri"], backend.get("workspace_user")
    vpc = backend.get("user_vpc")
    host = r"bmcpfs://cpfs-[a-z0-9-]+\." + re.escape(backend["region"]) + r"\.cpfs\.aliyuncs\.com/"
    if (not isinstance(uri, str) or not re.fullmatch(host, uri)
            or not isinstance(user, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", user)
            or "-vpc-" in uri and not vpc
            or "user_vpc" in backend and (not isinstance(vpc, dict)
                or set(vpc) != {"VpcId", "VSwitchId", "SecurityGroupId"}
                or any(not isinstance(vpc[key], str) or not re.fullmatch(prefix + r"[a-z0-9]+", vpc[key])
                       for key, prefix in (("VpcId", "vpc-"), ("VSwitchId", "vsw-"), ("SecurityGroupId", "sg-"))))):
        raise ValueError("DLC CPFS requires an explicit same-region endpoint, approved user and matching VPC")
    return [{"Uri": uri + path, "MountPath": target, "MountAccess": access}
            for path, target, access in (("users/" + user + "/", "/mnt/workspace", "RW"),
                                        ("public/models/", "/mnt/models", "RO"),
                                        ("public/datasets/", "/mnt/datasets", "RO"))]


def credential_config(backend):
    """DLC uses different field names from DSW's CredentialConfig."""
    if "data_role_arn" not in backend:
        return None
    user, arn = backend.get("workspace_user"), backend["data_role_arn"]
    if not isinstance(user, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", user):
        raise ValueError("DLC data role requires the approved workspace user")
    match = re.fullmatch(r"acs:ram::([0-9]{1,32}):role/pai-data-" + re.escape(user), arn) if isinstance(arn, str) else None
    if not match:
        raise ValueError("DLC data role must be the approved user's pai-data role")
    return {"EnableCredentialInject": True, "AliyunEnvRoleKey": "0", "CredentialConfigItems": [
        {"Key": "0", "Type": "Role", "Roles": [
            {"RoleArn": arn, "RoleType": "service", "AssumeRoleFor": match[1]}]}]}


class PaiBackend:
    kind = "pai-dlc"
    capabilities = PAI

    def __init__(self, services: BackendServices) -> None:
        self.s = services
        self._api = None

    @property
    def api(self):
        if self._api is None:
            self._api = self.s.pai_api()
        return self._api

    def availability(self):
        try:
            api = self.api
            api.workspace(component(api.config["workspace_id"]))
            ok = True
        except (PaiError, ValueError, KeyError):
            ok = False
        return PreflightReport(self.kind, "availability", (PreflightCheck(
            "pai-openapi", "authorization", "PASS" if ok else "FAIL", "PAI workspace query " + ("available" if ok else "unavailable")),))

    def validate(self, run):
        backend = run["backend"]
        for key in ("workspace_id", "quota_id", "gpu_type", "region", "job_name"):
            component(backend[key])
        scheduler_job_name(backend["job_name"], "attempt-001")
        if backend.get("worker_nodes", 1) != 1:
            raise ValueError("PAI first release supports exactly one worker node")
        if "oversold_type" in backend and backend["oversold_type"] not in ALLOCATION_POLICIES.values():
            raise ValueError("invalid PAI idle resource policy")
        if "priority" in backend and (type(backend["priority"]) is not int or not 1 <= backend["priority"] <= 9):
            raise ValueError("PAI task priority must be an integer from 1 to 9")
        if type(backend.get("gpu_type_from_quota", False)) is not bool:
            raise ValueError("PAI quota GPU selection must be explicit")
        resources = run.get("resources", {})
        for key, maximum in (("gpus", 8), ("cpus", 512), ("memory_gb", 4096)):
            if type(resources.get(key)) is not int or not 1 <= resources[key] <= maximum:
                raise ValueError("invalid PAI single-node resources")
        if not re.fullmatch(r"[0-9]{2,3}:[0-5][0-9]:[0-5][0-9]", backend["time"]):
            raise ValueError("invalid PAI walltime")
        if not sum(int(n) for n in backend["time"].split(":")):
            raise ValueError("PAI walltime must be positive")
        digest_pinned_image(backend["image"], run["image_id"])
        sources = backend.get("data_sources", [])
        if not isinstance(sources, list):
            raise ValueError("invalid PAI data sources")
        mounts = {source["MountPath"] for source in personal_mounts(backend)}
        credential_config(backend)
        for source in sources:
            if (not isinstance(source, dict) or set(source) - {"DataSourceId", "DataSourceVersion", "MountPath", "MountAccess"}
                    or not source.get("DataSourceId") or not source.get("DataSourceVersion")
                    or source.get("MountAccess") not in {"RO", "RW"}
                    or not re.fullmatch(r"/(?:[A-Za-z0-9_.-]+/?)+", source.get("MountPath", ""))
                    or any(part in {".", ".."} for part in source["MountPath"].split("/"))):
                raise ValueError("PAI data mounts require explicit dataset version, access and canonical path")
            component(source["DataSourceId"])
            component(source["DataSourceVersion"])
            path = source["MountPath"].rstrip("/")
            if any(path == other or path.startswith(other + "/") or other.startswith(path + "/") for other in mounts):
                raise ValueError("PAI dataset mount paths must be unique")
            mounts.add(path)

    def preflight(self, run, *, scope):
        if scope not in {"stage", "submit", "observe"}:
            raise ValueError("invalid PAI preflight scope")
        self.validate(run)
        try:
            backend = run["backend"]
            if backend["region"] != self.api.config["region"]:
                raise ValueError("PAI host credentials target another region")
            self.api.workspace(backend["workspace_id"])
            quota = self.api.quota(backend["quota_id"])
            if backend.get("gpu_type_from_quota") and quota.get("GPUType") != backend["gpu_type"]:
                raise ValueError("PAI quota GPU type differs from the frozen executor")
            if scope != "observe" and quota.get("Status") not in {"Ready", "Scaling"}:
                raise ValueError("PAI quota is unavailable")
            self.api.jobs(backend, name=backend["job_name"])
            ok = True
        except (PaiError, ValueError):
            ok = False
        return PreflightReport(self.kind, scope, (PreflightCheck(
            "pai-scope", "authorization", "PASS" if ok else "FAIL", "PAI exact configured workspace and quota query " + ("available" if ok else "unavailable")),))

    def environment(self, campaign, run, source_id, attempt_id):
        return {}

    def stage(self, campaign, run, source_id, source_bundle):
        return True  # Code and managed launcher are already in the pinned OCI image.

    def verify_assets(self, run, probes):
        return {"missing": None, "verification": "image-manifest", "verified_on": None}

    def submission_request(self, campaign, run, attempt_id):
        self.validate(run)
        return {"scheduler_name": scheduler_job_name(run["backend"]["job_name"], attempt_id),
                "image_tag": run["backend"]["image"], "image_digest": run["image_id"],
                "image_reference": digest_pinned_image(run["backend"]["image"], run["image_id"])}

    def matches(self, run, attempt, token=None):
        tags = job_tags(run, attempt, token)
        rows = self.api.jobs(run["backend"], tags=tags)
        # The list filter is a hint, never identity evidence by itself.
        return [row for row in rows if all(observed_tags(row).get(k) == v for k, v in tags.items())]

    def identity(self, campaign, run, attempt_id):
        rows = self.matches(run, attempt_id)
        return IdentityReport(not rows, len(rows) > 1, tuple(row["JobId"] for row in rows))

    def recover_submission(self, run, intent, attempt_id):
        token, request = require_submission_intent(intent)
        if request != self.submission_request({}, run, attempt_id):
            raise ValueError("PAI submission intent differs")
        rows = self.matches(run, attempt_id, token)
        if len(rows) > 1:
            raise ValueError("ambiguous PAI submission identity")
        return rows[0]["JobId"] if rows else None

    def create_document(self, manifest, token=None):
        self.validate(manifest)
        backend, resources, attempt = manifest["backend"], manifest["resources"], manifest["attempt_id"]
        name = (submission_resource_name(backend["job_name"], attempt, token) if token
                else scheduler_job_name(backend["job_name"], attempt))
        command = self.s.dispatch_command(manifest) if token else manifest["command"]
        seconds = sum(int(n) * m for n, m in zip(backend["time"].split(":"), (3600, 60, 1)))
        document = {"WorkspaceId": backend["workspace_id"], "ResourceId": backend["quota_id"],
                "DisplayName": name, "JobType": "PyTorchJob", "Accessibility": "PRIVATE",
                "UserCommand": shlex.join(["env", "BACKEND_JOB_ID=" + name, *command]),
                "JobMaxRunningTimeMinutes": (seconds + 59) // 60,
                "DataSources": personal_mounts(backend) + backend.get("data_sources", []),
                "UserVpc": backend.get("user_vpc", {}),
                "Settings": {"Tags": job_tags(manifest, attempt, token), "JobReservedMinutes": 0,
                             "EnableErrorMonitoringInAIMaster": False, "EnableRDMA": False},
                "JobSpecs": [{"Type": "Worker", "PodCount": 1, "UseSpotInstance": False,
                              "RestartPolicy": "Never", "Image": digest_pinned_image(backend["image"], manifest["image_id"]),
                              "ResourceConfig": {"CPU": str(resources["cpus"]), "Memory": str(resources["memory_gb"]) + "Gi",
                                                 "GPU": str(resources["gpus"]), "GPUType": backend["gpu_type"]}}]}
        if "oversold_type" in backend:
            document["JobSpecs"][0]["OversoldType"] = backend["oversold_type"]
        if "priority" in backend:
            document["Priority"] = backend["priority"]
        credentials = credential_config(backend)
        if backend.get("gpu_type_from_quota"):
            del document["JobSpecs"][0]["ResourceConfig"]["GPUType"]
        if credentials is not None:
            document["CredentialConfig"] = credentials
        return document

    def render(self, manifest):
        return json.dumps(self.create_document(manifest), sort_keys=True)

    def submit(self, campaign, run, manifest, *, dry_run, intent=None):
        if manifest.get("image_id") != run.get("image_id") or manifest.get("backend") != run["backend"]:
            raise ValueError("PAI frozen submission definition differs")
        if dry_run:
            self.create_document(manifest)
            return "DRY_RUN"
        token, request = require_submission_intent(intent)
        if request != self.submission_request(campaign, run, manifest["attempt_id"]):
            raise ValueError("PAI submission intent differs")
        if self.matches(run, manifest["attempt_id"]):
            raise FileExistsError("PAI submission is already present; reconcile it")
        value = self.api.create_job(self.create_document(manifest, token))
        return component(value.get("JobId"))

    def status(self, campaign, run):
        job = self.bound_job(campaign, run)
        return {"run_id": run["run_id"], "backend": self.kind, "backend_job_id": job["JobId"],
                "state": normalized(job.get("Status")), "raw_state": job.get("Status"),
                "reason": redact_line(str(job.get("ReasonMessage") or job.get("ReasonCode") or ""))[:2048],
                "exit_code": None, "failure_class": None,
                "detail": {"provider_reason_code": job.get("ReasonCode"), "restart_times": job.get("RestartTimes"),
                           "provider_phase": job.get("Status")}}

    def bound_job(self, campaign, run):
        record = self.s.backend_record(campaign, run)
        job = self.api.job(run["backend"], record["backend_job_id"])
        tags = observed_tags(job)
        if not all(tags.get(k) == v for k, v in job_tags(run, record["attempt_id"]).items()):
            raise ValueError("PAI job belongs to another Run or Attempt")
        return job

    def cancel(self, campaign, run):
        record = self.s.backend_record(campaign, run)
        self.bound_job(campaign, run)
        self.api.request("pai-dlc", "StopJob", "/api/v1/jobs/" + component(record["backend_job_id"]) + "/stop", method="POST")
        return self.status(campaign, run)

    def logs(self, campaign, run, *, tail):
        record = self.s.backend_record(campaign, run)
        job_id = record["backend_job_id"]
        job = self.bound_job(campaign, run)
        pods = job.get("Pods", []) or job.get("HistoryPods", [])
        pods = [pod for group in pods for pod in (group if isinstance(group, list) else [group])]
        lines = []
        for pod in pods[-8:]:
            value = self.api.request("pai-dlc", "GetPodLogs", "/api/v1/jobs/" + component(job_id) + "/pods/" + component(pod["PodId"]) + "/logs",
                                     query={"PodUid": component(pod["PodUid"]), "MaxLines": min(max(tail, 1), 2000),
                                            "StartTime": job["GmtCreateTime"]})
            if not isinstance(value.get("Logs"), list) or any(not isinstance(x, str) for x in value["Logs"]):
                raise ValueError("invalid PAI logs")
            lines.extend(redact_line(line[:8192]) for line in value["Logs"])
        return {"run_id": run["run_id"], "backend": self.kind, "backend_job_id": job_id,
                "tail": tail, "lines": lines[-max(tail, 1):], "expired": False, "stream_exit_code": 0,
                "available": bool(pods), "source": "pai-pod-logs", "historical": normalized(job.get("Status")) in {"FAILED", "SUCCEEDED", "CANCELLED"}}

    def collect(self, campaign, run):
        snapshot = self.logs(campaign, run, tail=200)
        return {"run_id": run["run_id"], "backend": self.kind, "model_observed": False,
                "live_logs_available": snapshot["available"], "process_evidence": {
                    "observed": False, "stdout_tail": snapshot["lines"], "stderr_tail": []},
                "log_source": "pai-pod-logs"}
