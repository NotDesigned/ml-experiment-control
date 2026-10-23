"""An opt-in CPU DSW publisher; existing source and OCI receipt contracts remain.

Control operations use the signed SDK. The PAI CLI supplies its supported MCP
file/command channel, isolated from the operator's CLI profile and credentials.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
import time
import uuid

from experiment_control.backends.pai_api import PaiAPI, PaiError, component
from experiment_control.redaction import redact_line
from experiment_control.runner import run_process

from .builder_client import BuildTransportError, IMAGE
from .image_build_context import BUILD_LOG, BUILD_PROGRESS
from ..data.registry_transport import Registry, MANIFEST, OCI_MANIFEST, digest
from ..runs.execution_progress import record_progress
from ..storage import atomic_json


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(1024**2):
            value.update(chunk)
    return value.hexdigest()


class DswPublisher:
    def __init__(self, builder):
        self.builder, self.config = builder, builder.config["pai_dsw"]
        self.api = PaiAPI.from_environment()
        self.instance = None
        self.identity = None
        self.name = None
        self.secrets = []

    def datasets(self):
        """Approved account workspace and public RO mounts; never mount a root."""
        datasets = []
        if any(key in self.config for key in ("cpfs_uri", "user_vpc")):
            uri, user, vpc = (self.config.get(key) for key in ("cpfs_uri", "workspace_user", "user_vpc"))
            host = r"cpfs-[a-z0-9-]+\." + re.escape(self.api.config["region"]) + r"\.cpfs\.aliyuncs\.com/"
            if (not isinstance(uri, str) or not re.fullmatch("bmcpfs://" + host, uri)
                    or not isinstance(user, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", user)
                    or "-vpc-" in uri and vpc is None
                    or "user_vpc" in self.config and (
                        not isinstance(vpc, dict) or set(vpc) != {"VpcId", "VSwitchId", "SecurityGroupId"}
                        or any(not isinstance(vpc[key], str) or not re.fullmatch(prefix + r"[a-z0-9]+", vpc[key])
                               for key, prefix in (("VpcId", "vpc-"), ("VSwitchId", "vsw-"), ("SecurityGroupId", "sg-"))))):
                raise ValueError("CPFS workspace requires an explicit same-region endpoint, approved user and matching user_vpc")
            datasets = [{"Uri": uri + path, "MountPath": mount, "MountAccess": access}
                        for path, mount, access in (("users/" + user + "/", "/mnt/ml-expd-personal", "RW"),
                                                    ("public/models/", "/mnt/models", "RO"),
                                                    ("public/datasets/", "/mnt/datasets", "RO"))]
        uri = self.config.get("oss_uri")
        mount = self.config.get("oss_mount_path", "/mnt/ml-expd-oss")
        if "oss_uri" not in self.config and "oss_mount_path" not in self.config:
            return datasets
        if not isinstance(uri, str) or not isinstance(mount, str):
            raise ValueError("DSW OSS mount requires a URI and a canonical mount path")
        host = r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]\.oss-" + re.escape(self.api.config["region"]) + r"(?:-internal)?\.aliyuncs\.com"
        match = re.fullmatch(r"oss://" + host + r"(/[A-Za-z0-9._/-]+/)", uri)
        if (not match or any(p in {"", ".", ".."} for p in match[1].split("/")[1:-1])
                or not re.fullmatch(r"/(?:mnt|data)/[A-Za-z0-9][A-Za-z0-9._/-]*", mount)
                or any(p in {"", ".", ".."} for p in mount.split("/")[1:])
                or datasets and (mount == "/mnt/workspace" or mount.startswith("/mnt/workspace/"))
                or any(mount == d["MountPath"] or mount.startswith(d["MountPath"] + "/")
                       or d["MountPath"].startswith(mount + "/") for d in datasets)):
            raise ValueError("DSW OSS mount must bind an explicit same-region prefix to /mnt or /data")
        role = self.role()
        if role is None:
            return datasets + [{"Uri": uri, "MountPath": mount, "MountAccess": "RW"}]
        user = self.config["workspace_user"]
        if match[1] != "/users/" + user + "/" or mount != "/mnt/oss":
            raise ValueError("personal OSS mount must use the approved user's prefix at /mnt/oss")
        root = uri[:-len(match[1])] + "/"
        return datasets + [{"Uri": root + path, "MountPath": target, "MountAccess": access, "RoleArn": role[0]}
                           for path, target, access in (("users/" + user + "/", "/mnt/oss", "RW"),
                                                       ("public/models/", "/mnt/oss-models", "RO"),
                                                       ("public/datasets/", "/mnt/oss-datasets", "RO"))]

    def role(self):
        arn = self.config.get("data_role_arn")
        if "data_role_arn" not in self.config:
            return None
        user = self.config.get("workspace_user")
        if not isinstance(user, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", user):
            raise ValueError("personal data role requires the approved workspace user")
        match = re.fullmatch(r"acs:ram::([0-9]{1,32}):role/pai-data-" + re.escape(user), arn) if isinstance(arn, str) else None
        if not match:
            raise ValueError("data role must be the approved user's pai-data role")
        return arn, match[1]

    def upload(self, source, target, private):
        """Use small parts below MCP's practical request size, then verify SHA256."""
        source = Path(source)
        if source.stat().st_size <= 512 * 1024:
            self.cli(["dsw", "instance", "cp", str(source), "dsw:" + target,
                      "--instance", self.instance], private, timeout=180)
            return
        parts = []
        with tempfile.TemporaryDirectory(prefix="dsw-parts-", dir=self.builder.root) as directory:
            with source.open("rb") as stream:
                while chunk := stream.read(512 * 1024):
                    suffix = ".part-" + str(len(parts))
                    path = Path(directory) / ("payload" + suffix)
                    path.write_bytes(chunk)
                    self.cli(["dsw", "instance", "cp", str(path), "dsw:" + target + suffix,
                              "--instance", self.instance], private, timeout=180)
                    parts.append(target + suffix)
                    path.unlink()
        code = ("from pathlib import Path; import hashlib; "
                "parts=" + repr(parts) + "; target=Path(" + repr(target) + "); "
                "f=target.open('xb'); "
                "[f.write(Path(p).read_bytes()) for p in parts]; f.close(); "
                "actual=hashlib.file_digest(target.open('rb'),'sha256').hexdigest(); "
                "\nif actual != " + repr(file_sha(source)) + ": raise ValueError('DSW transfer SHA256 differs')"
                "\nfor p in parts: Path(p).unlink()")
        self.cli(["dsw", "instance", "exec", self.instance, "--code", code], private)

    def shutdown_timer(self):
        """Cloud-side stop budget survives a lost CLI or API-host process."""
        path = "/api/v2/instances/" + self.instance + "/shutdowntimer"
        self.api.request("pai-dsw", "CreateInstanceShutdownTimer", path,
                         method="POST", body={"RemainingTimeInMs": 1800000})
        value = self.api.request("pai-dsw", "GetInstanceShutdownTimer", path)
        if (value.get("InstanceId") != self.instance or value.get("Success") is not True
                or type(value.get("RemainingTimeInMs")) is not int
                or not 0 < value["RemainingTimeInMs"] <= 1800000):
            raise ValueError("DSW cloud shutdown timer could not be verified")

    def build(self, remote, private):
        # MCP's HTTPS gateway cannot hold a long build open. Start once, then
        # observe durable status and bounded log deltas using short requests.
        code = ("from pathlib import Path; import subprocess; root=Path(" + repr(remote) + "); "
                "(root/'started').open('x').close(); "
                "subprocess.Popen(['python3',str(root/'worker.py'),str(root)], "
                "stdin=subprocess.DEVNULL,stdout=(root/'build.log').open('wb'), "
                "stderr=subprocess.STDOUT,start_new_session=True)")
        self.cli(["dsw", "instance", "exec", self.instance, "--code", code], private, timeout=45)
        deadline, offset = time.monotonic() + 600, 0
        while True:
            code = ("from pathlib import Path; import json; root=Path(" + repr(remote) + "); "
                    "p=root/'build.log'; f=p.open('rb'); f.seek(" + str(offset) + "); chunk=f.read(65536); "
                    "offset=f.tell(); f.close(); print(chunk.decode('utf-8',errors='replace'),end=''); "
                    "s=root/'status.json'; state=json.loads(s.read_text()) if s.exists() else {'status':'RUNNING'}; "
                    "state['log_offset']=offset; print('\\nML_EXPD_DSW_STATUS_JSON='+json.dumps(state))")
            output = self.cli(["dsw", "instance", "exec", self.instance, "--code", code], private, timeout=45)
            lines = [line.removeprefix("ML_EXPD_DSW_STATUS_JSON=") for line in output.splitlines()
                     if line.startswith("ML_EXPD_DSW_STATUS_JSON=")]
            if len(lines) != 1:
                raise ValueError("DSW durable build status is unavailable")
            state = json.loads(lines[0])
            offset = state["log_offset"]
            if type(offset) is not int or offset < 0:
                raise ValueError("DSW build log position is invalid")
            if state["status"] == "SUCCEEDED":
                return state["receipt"]
            if state["status"] != "RUNNING" or time.monotonic() > deadline:
                raise ValueError("DSW build failed or exceeded ten minutes; inspect bounded build logs")
            self.phase("BUILDING_ON_DSW", "Docker build running; observed " + str(offset) + " log bytes")
            time.sleep(3)

    def phase(self, phase, message):
        path = BUILD_PROGRESS.get()
        if path:
            record_progress(path, phase, message)

    def record(self, phase, **details):
        previous = json.loads(self.lease.read_text()) if self.lease.exists() else {}
        value = {**previous, "bundle_id": self.identity, "instance_id": self.instance, "name": self.name,
                 "region": self.api.config["region"], "workspace_id": self.config["workspace_id"],
                 "quota_id": self.config["quota_id"], "datasets": self.datasets(), "phase": phase, "gpu": 0, **details}
        atomic_json(self.lease, value)

    def checked_instance(self):
        value = self.api.request("pai-dsw", "GetInstance", "/api/v2/instances/" + component(self.instance))
        if (value.get("InstanceId") != self.instance or value.get("InstanceName") != self.name
                or str(value.get("WorkspaceId")) != str(self.config["workspace_id"])
                or value.get("ResourceId") != self.config["quota_id"]
                or value.get("RequestedResource", {}).get("GPU") != "0"
                or value.get("RequestedResource", {}).get("CPU") != "4"
                or value.get("RequestedResource", {}).get("Memory") != "8Gi"
                or {"Key": "ml-expd.bundle", "Value": self.identity} not in value.get("Labels", [])):
            raise ValueError("DSW builder ownership or zero-GPU identity differs")
        return value

    def cli(self, args, private, *, timeout=120):
        env = {**os.environ, "PAI_CONFIG_DIR": str(private)}
        result = run_process([self.config.get("pai_bin", "/usr/local/bin/pai"),
                              "--profile", "ml-expd", "--region", self.api.config["region"], *args],
                             env=env, text=True, timeout=timeout)
        text = result.stdout + "\n" + result.stderr
        text = self.log_output(text)
        if result.returncode:
            raise ValueError("DSW transfer or remote build failed; inspect bounded builder logs")
        return text

    def log_output(self, text):
        for key in ("access_key_id", "access_key_secret", "security_token"):
            if self.api.config.get(key):
                text = text.replace(self.api.config[key], "[redacted]")
        for secret in self.secrets:
            text = text.replace(secret, "[redacted]")
        text = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[redacted]", text)
        text = "\n".join(redact_line(line)[:8192] for line in text.splitlines())[-2*1024**2:]
        log = BUILD_LOG.get()
        if log:
            with Path(log).open("a") as f:
                f.write(text + "\n")
        return text

    def startup_failure(self, value):
        details = {"instance_id": self.instance, "status": value.get("Status"),
                   "publication_uncertain": True, "scheduler_submitted": False}
        try:
            lifecycle = self.api.request("pai-dsw", "GetLifecycle",
                                         "/api/v2/instances/" + self.instance + "/lifecycle", query={"Limit": 1})
            rows = [row for session in lifecycle.get("Lifecycle", []) for row in session]
        except PaiError as error:
            rows = []
            details["diagnostic_query_error"] = error.code
        reason = next((row.get("ReasonMessage") for row in reversed(rows) if row.get("ReasonMessage")),
                      value.get("ReasonMessage"))
        if reason:
            details["reason"] = self.log_output(str(reason))[-4096:]
        self.record("STARTUP_FAILED", startup_diagnostic=details)
        raise BuildTransportError("PAI_DSW_STARTUP_FAILED", details)

    def cleanup(self):
        value = self.checked_instance()
        if value.get("Status") not in {"Stopped", "Failed", "Deleted", "Stopping"}:
            self.api.request("pai-dsw", "StopInstance", "/api/v2/instances/" + self.instance + "/stop", method="PUT")
        deadline = time.monotonic() + 180
        while self.checked_instance().get("Status") not in {"Stopped", "Failed", "Deleted"}:
            if time.monotonic() > deadline:
                raise ValueError("owned DSW stop remains uncertain")
            time.sleep(3)
        self.api.request("pai-dsw", "DeleteInstance", "/api/v2/instances/" + self.instance, method="DELETE")
        self.record("DELETE_REQUESTED")

    def publish(self, tag, context):
        match = re.fullmatch(re.escape(self.builder.config["repository"]) + r":bundle-([0-9a-f]{64})", tag)
        if not match or any(not self.config.get(k) for k in ("workspace_id", "quota_id", "image_id")):
            raise ValueError("DSW publisher requires fixed scope and a Docker-enabled image")
        datasets = self.datasets()
        role = self.role()
        self.identity = match[1]
        self.lease = self.builder.root / (self.identity + ".dsw.json")
        if self.lease.exists():
            raise BuildTransportError("PAI_DSW_RECONCILE_REQUIRED", {"publication_uncertain": True, "scheduler_submitted": False})
        host = tag.split("/", 1)[0]
        auths = json.loads(Path(self.builder.config["registry_auth_file"]).read_text())["auths"]
        auth = auths.get(host) or auths.get("https://" + host)
        if not isinstance(auth, dict) or not isinstance(auth.get("auth"), str) or b":" not in base64.b64decode(auth["auth"], validate=True):
            raise ValueError("DSW target registry authentication is unavailable")
        self.secrets = [auth["auth"], base64.b64decode(auth["auth"]).decode().split(":", 1)[1]]
        self.secrets = [secret for secret in self.secrets if secret]
        nonce = uuid.uuid4().hex
        self.name = "ml_exp_build_" + nonce[:12]
        body = {"WorkspaceId": self.config["workspace_id"], "ResourceId": self.config["quota_id"],
                "InstanceName": self.name, "ImageId": self.config["image_id"], "Accessibility": "PRIVATE",
                "RequestedResource": {"CPU": "4", "Memory": "8Gi", "GPU": "0"},
                "DockerConfig": {"Enable": True, "ContainersLimit": 2}, "OversoldType": "ForbiddenQuotaOverSold",
                "Labels": [{"Key": "ml-expd.bundle", "Value": self.identity}]}
        if datasets:
            body["Datasets"] = datasets
        if self.config.get("cpfs_uri"):
            body["WorkspaceSource"] = "/mnt/ml-expd-personal"
        if "user_vpc" in self.config:
            body["UserVpc"] = {**self.config["user_vpc"], "DefaultRoute": "eth0"}
        if role is not None:
            body["CredentialConfig"] = {"Enable": True, "AliyunEnvRoleKey": "0", "Configs": [
                {"Key": "0", "Type": "Role", "Roles": [
                    {"RoleArn": role[0], "RoleType": "service", "AssumeRoleFor": role[1]}]}]}
        self.record("CREATE_INTENT")
        self.phase("STARTING_DSW_BUILDER", "Starting a private 4CPU/8GiB/0GPU DSW in the fixed resource quota")
        try:
            created = self.api.request("pai-dsw", "CreateInstance", "/api/v2/instances", method="POST", body=body)
            self.instance = component(created.get("InstanceId"))
            self.record("CREATED")
            self.checked_instance()
            self.shutdown_timer()
            end = time.monotonic() + 600
            while True:
                value = self.checked_instance()
                if value.get("Status") == "Running":
                    break
                if value.get("Status") in {"Failed", "Stopped", "Deleted"} or time.monotonic() > end:
                    self.startup_failure(value)
                time.sleep(5)
            with tempfile.TemporaryDirectory(prefix="dsw-context-", dir=self.builder.root) as directory:
                local = Path(directory)
                private = local / "cli"
                private.mkdir(mode=0o700)
                cli_config = {k: self.api.config[k] for k in ("access_key_id", "access_key_secret", "security_token", "region", "workspace_id") if k in self.api.config}
                cli_config["workspace_id"] = self.config["workspace_id"]
                atomic_json(private / "config.json", {"profiles": {"ml-expd": cli_config}, "defaults": {"profile": "ml-expd"}})
                archive = local / "context.tar"
                with tarfile.open(archive, "w") as tar:
                    for path in sorted(context.rglob("*")):
                        tar.add(path, arcname=str(path.relative_to(context)), recursive=False)
                if archive.stat().st_size > 300 * 1024**2:
                    raise ValueError("DSW source context exceeds 300 MiB")
                remote = "/tmp/ml-expd-build-" + nonce
                self.phase("TRANSFERRING_DSW_CONTEXT", "Copying the sealed source context over PAI MCP")
                self.cli(["dsw", "instance", "exec", self.instance, "--code",
                          "from pathlib import Path; p=Path(" + repr(remote) + "); p.mkdir(mode=0o700); (p/'auth').mkdir(mode=0o700); (p/'tools').mkdir(mode=0o700)"], private)
                request = {"context_sha256": file_sha(archive), "tag": tag}
                atomic_json(local / "request.json", request)
                atomic_json(local / "config.json", {"auths": {host: auth}})
                worker = Path(__file__).parent.parent / "workers/dsw_build_worker.py"
                files = [(archive, "context.tar"), (local / "request.json", "request.json"), (local / "config.json", "auth/config.json"),
                         (worker, "worker.py")]
                for source, target in files:
                    self.upload(source, remote + "/" + target, private)
                self.phase("BUILDING_ON_DSW", "Building and pushing the runtime using the DSW Docker engine")
                receipt = self.build(remote, private)
                image = receipt.get("image", "")
                if (not IMAGE.fullmatch(image) or not image.startswith(tag.rsplit(":", 1)[0] + "@")
                        or receipt.get("context_sha256") != request["context_sha256"]):
                    raise ValueError("DSW publication identity differs")
                registry = Registry(self.builder.config["repository"], self.builder.config["registry_auth_file"])
                _, _, raw = registry.request("GET", registry.prefix + "/manifests/" + tag.rsplit(":", 1)[1],
                                             headers={"Accept": MANIFEST + ", " + OCI_MANIFEST})
                if image.rsplit("@", 1)[1] != digest(raw):
                    raise ValueError("independent registry manifest differs from DSW receipt")
                self.record("PUBLICATION_VERIFIED", image=image, context_sha256=request["context_sha256"])
                return image
        except BuildTransportError:
            raise
        except (PaiError, OSError, ValueError, subprocess.SubprocessError) as error:
            raise BuildTransportError("PAI_DSW_BUILD_UNCERTAIN", {"error_class": type(error).__name__,
                                      "publication_uncertain": True, "scheduler_submitted": False}) from None
        finally:
            if self.instance is not None:
                self.phase("STOPPING_DSW_BUILDER", "Stopping and deleting only this build's CPU DSW instance")
                try:
                    self.cleanup()
                except (PaiError, OSError, ValueError) as error:
                    self.record("CLEANUP_REQUIRED", cleanup_error_class=type(error).__name__)
                    raise BuildTransportError("PAI_DSW_CLEANUP_REQUIRED", {
                        "instance_id": self.instance, "error_class": type(error).__name__,
                        "publication_uncertain": True, "scheduler_submitted": False,
                        "cleanup_required": True}) from None
