"""Native DSW debug lifecycle, including key injection and a cloud stop timer."""
from copy import deepcopy
import re
import time

from experiment_control.backends.pai_api import PaiAPI, PaiError, component

from ..pai_storage import PaiStorage


class DswDebug:
    def __init__(self, profile, api=None):
        self.profile, self.config = profile, profile["native"]
        self.api = api or PaiAPI.from_environment()

    def document(self, record):
        c, p = self.config, self.profile
        allowed = {"workspace_id", "quota_id", "image_id", "cpfs_uri", "workspace_user",
                   "data_role_arn", "user_vpc", "ssh_nat_gateway_id", "ssh_eip_id", "ssh_access_type"}
        if set(c) - allowed or not all(c.get(k) for k in ("workspace_id", "quota_id", "image_id",
                                                        "cpfs_uri", "workspace_user", "data_role_arn")):
            raise ValueError("DSW debug profile requires an approved personal CPFS, role and image")
        for k in ("workspace_id", "quota_id", "image_id"):
            component(c[k])
        storage = PaiStorage(c, self.api)
        datasets, role = storage.datasets(), storage.role()
        arn, account = role
        body = {"WorkspaceId": c["workspace_id"], "ResourceId": c["quota_id"],
                "InstanceName": record["name"], "ImageId": c["image_id"], "Accessibility": "PRIVATE",
                "RequestedResource": {"CPU": str(p["cpus"]), "Memory": str(p["memory_gib"]) + "Gi",
                                      "GPU": str(p["gpus"])},
                "OversoldType": "ForbiddenQuotaOverSold",
                "Labels": [{"Key": "ml-expd.debug", "Value": record["identity"]}],
                "Datasets": datasets, "WorkspaceSource": "/mnt/ml-expd-personal",
                "CredentialConfig": {"Enable": True, "AliyunEnvRoleKey": "0", "Configs": [
                    {"Key": "0", "Type": "Role", "Roles": [
                        {"RoleArn": arn, "RoleType": "service", "AssumeRoleFor": account}]}]}}
        access = c.get("ssh_access_type", "intranet")
        if access not in {"intranet", "internet"}:
            raise ValueError("invalid DSW SSH access type")
        forward = {"Enable": True, "ContainerName": "dsw-notebook", "ForwardPort": "22",
                   "Name": "ssh", "SSHPublicKey": record["request"]["ssh_public_key"],
                   "AccessType": [access]}
        if access == "internet":
            for field, pattern in (("ssh_nat_gateway_id", r"ngw-[a-z0-9]+"), ("ssh_eip_id", r"eip-[a-z0-9]+")):
                if not isinstance(c.get(field), str) or not re.fullmatch(pattern, c[field]):
                    raise ValueError("public SSH requires an existing configured NAT and EIP")
            # Omit the port for native allocation. Literal "Auto" is rejected
            # by the regional NAT port-range validator.
            forward.update(NatGatewayId=c["ssh_nat_gateway_id"], EipAllocationId=c["ssh_eip_id"])
        vpc = deepcopy(c.get("user_vpc", {}))
        if not vpc:
            raise ValueError("DSW SSH requires the approved VPC configuration")
        body["UserVpc"] = {**vpc, "DefaultRoute": "eth0", "ForwardInfos": [forward]}
        return body

    def lookup(self, record):
        identifier = record.get("provider_id")
        if identifier:
            try:
                value = self.api.request("pai-dsw", "GetInstance", "/api/v2/instances/" + component(identifier))
            except PaiError as exc:
                if exc.status == 404 or exc.status == 400 and exc.code == "InstanceNotExistError":
                    return None
                raise
            self.checked(record, value)
            return value
        matches = []
        for page in range(1, 101):
            result = self.api.request("pai-dsw", "ListInstances", "/api/v2/instances", query={
                "WorkspaceId": self.config["workspace_id"], "InstanceName": record["name"],
                "PageNumber": page, "PageSize": 100})
            rows, total = result.get("Instances"), result.get("TotalCount")
            if not isinstance(rows, list) or type(total) is not int or total < 0:
                raise ValueError("invalid DSW debug pagination")
            matches.extend(row for row in rows if row.get("InstanceName") == record["name"])
            if page * 100 >= total:
                break
        else:
            raise ValueError("DSW debug pagination limit exceeded")
        if len(matches) > 1:
            raise ValueError("ambiguous DSW debug identity")
        if not matches:
            return None
        value = self.api.request("pai-dsw", "GetInstance", "/api/v2/instances/" + component(matches[0].get("InstanceId")))
        self.checked(record, value)
        return value

    def checked(self, record, value):
        c = self.config
        if (not isinstance(value, dict) or not value.get("InstanceId")
                or record.get("provider_id") not in (None, value["InstanceId"])
                or value.get("InstanceName") != record["name"]
                or str(value.get("WorkspaceId")) != str(c["workspace_id"])
                or value.get("ResourceId") != c["quota_id"]
                or {"Key": "ml-expd.debug", "Value": record["identity"]} not in value.get("Labels", [])):
            raise ValueError("DSW debug instance identity differs")

    def provision(self, record, save, claim):
        value = self.lookup(record)
        if value is None:
            if not claim("create"):
                return None
            result = self.api.request("pai-dsw", "CreateInstance", "/api/v2/instances", method="POST", body=self.document(record))
            record["create_accepted"] = True
            record["provider_id"] = component(result.get("InstanceId"))
            save()
            value = self.lookup(record)
        if value is not None:
            record["provider_id"] = value["InstanceId"]
            save()
            self.timer(record)
            record["expiry_enforcement"] = "provider_shutdown_timer"
        return value

    def timer(self, record):
        remaining = max(1, int((record["expires_at"] - time.time()) * 1000))
        path = "/api/v2/instances/" + component(record["provider_id"]) + "/shutdowntimer"
        # Resetting this native timer is safe: its absolute deadline never advances.
        self.api.request("pai-dsw", "CreateInstanceShutdownTimer", path, method="POST",
                         body={"RemainingTimeInMs": remaining})
        value = self.api.request("pai-dsw", "GetInstanceShutdownTimer", path)
        if (value.get("InstanceId") != record["provider_id"] or value.get("Success") is not True
                or type(value.get("RemainingTimeInMs")) is not int
                or not 0 < value["RemainingTimeInMs"] <= remaining):
            raise ValueError("DSW debug cloud shutdown timer was not confirmed")

    def connection(self, value):
        for forward in value.get("UserVpc", {}).get("ForwardInfos", []):
            if forward.get("Name") != "ssh" or forward.get("ForwardPort") != "22":
                continue
            info = forward.get("ConnectInfo", {})
            route = info.get("Internet" if self.config.get("ssh_access_type") == "internet" else "Intranet", {})
            if info.get("Phase") == "Ready" and route.get("Endpoint") and route.get("Port"):
                return {"host": route["Endpoint"], "port": int(route["Port"]), "user": "root",
                        "access": self.config.get("ssh_access_type", "intranet")}
        return None

    def state(self, value):
        state = value.get("Status")
        return {"Running": "RUNNING", "Stopped": "STOPPED", "Failed": "FAILED"}.get(state, "STARTING")

    def stop(self, record, value, delete=False):
        path = "/api/v2/instances/" + component(record["provider_id"])
        self.checked(record, value)
        if delete and value.get("Status") in {"Stopped", "Failed"}:
            self.api.request("pai-dsw", "DeleteInstance", path, method="DELETE")
        elif value.get("Status") not in {"Stopped", "Stopping", "Failed"}:
            self.api.request("pai-dsw", "StopInstance", path + "/stop", method="PUT")
