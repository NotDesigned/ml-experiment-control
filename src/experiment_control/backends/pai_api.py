"""Host-only Alibaba OpenAPI transport using the official signing SDK.

No CLI processes, credential-bearing manifests, implicit write retries or raw
provider exceptions cross this boundary. HTTP 200 error envelopes are errors.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import re
from urllib.parse import quote


class PaiError(RuntimeError):
    def __init__(self, action, *, code="PAI_REQUEST_FAILED", status=None, uncertain=False):
        self.code, self.status, self.uncertain = code, status, uncertain
        self.details = {"operation": action, "provider_code": code,
                        "http_status": status, "uncertain": uncertain}
        super().__init__("PAI " + action + " failed (" + code + ")")


def component(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}", value):
        raise ValueError("invalid PAI resource identity")
    return quote(value, safe="")


class PaiAPI:
    def __init__(self, config):
        if (not isinstance(config, dict)
                or any(not isinstance(config.get(k), str) or not config[k].strip()
                       for k in ("access_key_id", "access_key_secret", "region"))
                or not re.fullmatch(r"[a-z][a-z0-9-]{1,63}", config["region"])):
            raise PaiError("configuration", code="PAI_CONFIGURATION_INVALID")
        self.config = config

    @classmethod
    def from_environment(cls) -> PaiAPI:
        path = Path(os.environ.get("EXPERIMENTCTL_PAI_CONFIG", "/etc/ml-expd/pai.json"))
        try:
            return cls(json.loads(path.read_text()))
        except (OSError, ValueError):
            raise PaiError("configuration", code="PAI_CONFIGURATION_UNAVAILABLE") from None

    def request(self, product, action, path, *, method="GET", query=None, body=None,
                version=None, style="ROA", timeout=30):
        versions = {"pai-dlc": "2020-12-03", "pai-dsw": "2022-01-01", "aiworkspace": "2021-02-04",
                    "paistudio": "2022-01-12", "cr": "2018-12-01"}
        if (product not in versions or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,99}", action)
                or not path.startswith("/") or any(c in path for c in ("?", "#", "\r", "\n"))
                or method not in {"GET", "POST", "PUT", "DELETE"} or style not in {"ROA", "RPC"}):
            raise ValueError("invalid PAI OpenAPI operation")
        try:
            from alibabacloud_tea_openapi.client import Client
            from alibabacloud_tea_openapi import models
            from alibabacloud_tea_util.models import RuntimeOptions
        except ImportError:
            raise PaiError(action, code="PAI_SDK_UNAVAILABLE") from None
        endpoint = self.config.get("endpoints", {}).get(product, ("pai" if product == "paistudio" else product) + "." + self.config["region"] + ".aliyuncs.com")
        if not isinstance(endpoint, str) or not re.fullmatch(r"[a-z0-9.-]+\.aliyuncs\.com", endpoint):
            raise ValueError("invalid configured Alibaba HTTPS endpoint")
        cfg = models.Config(access_key_id=self.config["access_key_id"],
                            access_key_secret=self.config["access_key_secret"],
                            security_token=self.config.get("security_token"),
                            endpoint=endpoint, region_id=self.config["region"], protocol="HTTPS")
        params = models.Params(action=action, version=version or versions[product],
                               protocol="HTTPS", pathname=path, method=method, auth_type="AK",
                               style=style, req_body_type="json" if style == "ROA" else "formData", body_type="json")
        request = models.OpenApiRequest(query={k: str(v) for k, v in (query or {}).items()}, body=body)
        runtime = RuntimeOptions(autoretry=False, max_attempts=1, connect_timeout=5000,
                                 read_timeout=timeout * 1000, ignore_ssl=False)
        try:
            result = Client(cfg).call_api(params, request, runtime)
        except Exception as exc:
            data = getattr(exc, "data", None)
            data = data if isinstance(data, dict) else {}
            code = data.get("Code", data.get("code", "PAI_TRANSPORT_FAILED"))
            code = code if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", code) else "PAI_REQUEST_FAILED"
            status = data.get("statusCode")
            status = status if type(status) is int else None
            raise PaiError(action, code=code, status=status,
                           uncertain=method != "GET" and (status is None or status >= 500)) from None
        value = result.get("body")
        if not isinstance(value, dict):
            raise PaiError(action, code="PAI_RESPONSE_INVALID", uncertain=method != "GET")
        null_success = (product == "pai-dsw" and action in {
            "CreateInstanceShutdownTimer", "GetInstanceShutdownTimer"}
            and value.get("Code") == "null" and value.get("Success") is True)
        if (value.get("Code") not in (None, "Success", "200", 200) and not null_success
                or value.get("IsSuccess") is False or value.get("Success") is False):
            code = value.get("Code")
            code = code if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", code) else "PAI_REQUEST_FAILED"
            raise PaiError(action, code=code, status=result.get("statusCode"), uncertain=method != "GET")
        return value

    def job(self, backend, job_id):
        value = self.request("pai-dlc", "GetJob", "/api/v1/jobs/" + component(job_id))
        if (str(value.get("JobId")) != job_id or str(value.get("WorkspaceId")) != str(backend["workspace_id"])
                or value.get("ResourceId") != backend["quota_id"]):
            raise ValueError("PAI job identity or workspace differs")
        return value

    def create_job(self, document):
        """Inject private pull credentials only at the host transport boundary."""
        document = deepcopy(document)
        auth_file = self.config.get("registry_auth_file")
        if auth_file:
            try:
                auths = json.loads(Path(auth_file).read_text())["auths"]
                for spec in document["JobSpecs"]:
                    host = spec["Image"].split("/", 1)[0]
                    value = auths.get(host) or auths.get("https://" + host)
                    username, password = base64.b64decode(value["auth"], validate=True).decode().split(":", 1)
                    if not username or not password or any(c in username + password for c in ("\r", "\n")):
                        raise ValueError("invalid registry authentication")
                    spec["ImageConfig"] = {"DockerRegistry": host, "Username": username, "Password": password}
            except (OSError, ValueError, KeyError, TypeError):
                raise PaiError("CreateJob", code="PAI_IMAGE_AUTH_UNAVAILABLE") from None
        return self.request("pai-dlc", "CreateJob", "/api/v1/jobs", method="POST", body=document)

    def quota(self, quota_id):
        value = self.request("paistudio", "GetQuota", "/api/v1/quotas/" + component(quota_id), query={"Verbose": "true"})
        quota = value.get("Quota", value)
        if not isinstance(quota, dict) or quota.get("QuotaId") != quota_id:
            raise ValueError("PAI quota identity differs")
        return quota

    def queue_workload(self, backend, job_id):
        """Read admission diagnostics for one already-bound job, never another user/job."""
        value = self.request("paistudio", "ListQuotaWorkloads",
                             "/api/v1/quotas/" + component(backend["quota_id"]) + "/workloads",
                             query={"WorkspaceIds": component(str(backend["workspace_id"])),
                                    "WorkloadType": "dlc", "WorkloadIds": component(job_id),
                                    "PageNumber": 1, "PageSize": 2}, timeout=10)
        rows, total = value.get("Workloads"), value.get("TotalCount")
        if (not isinstance(rows, list) or type(total) is not int or total not in (0, 1)
                or len(rows) != total):
            raise ValueError("invalid PAI exact-job queue response")
        if not rows:
            return None
        row = rows[0]
        if (not isinstance(row, dict) or row.get("WorkloadId") != job_id
                or str(row.get("WorkspaceId")) != str(backend["workspace_id"])
                or row.get("QuotaId") != backend["quota_id"] or row.get("WorkloadType") != "dlc"):
            raise ValueError("PAI queue workload identity or scope differs")
        return row

    def jobs(self, backend, *, tags=None, name=None):
        rows, seen = [], set()
        for page in range(1, 101):
            query = {"WorkspaceId": backend["workspace_id"], "ResourceId": backend["quota_id"],
                     "ShowOwn": "true", "PageNumber": page, "PageSize": 100,
                     "StartTime": "2020-01-01T00:00:00Z"}
            if tags:
                query["Tags"] = json.dumps(tags, separators=(",", ":"))
            if name:
                query["DisplayName"] = name
            data = self.request("pai-dlc", "ListJobs", "/api/v1/jobs", query=query)
            items, total = data.get("Jobs"), data.get("TotalCount")
            if not isinstance(items, list) or type(total) is not int or total < 0:
                raise ValueError("invalid PAI job pagination")
            for item in items:
                job_id = component(item.get("JobId"))
                if job_id in seen:
                    raise ValueError("duplicate PAI job identity")
                seen.add(job_id)
                if str(item.get("WorkspaceId")) != str(backend["workspace_id"]) or item.get("ResourceId") != backend["quota_id"]:
                    raise ValueError("PAI job list escaped configured scope")
                rows.append(item)
            if len(rows) == total:
                return rows
            if not items or len(rows) > total:
                raise ValueError("incomplete PAI job pagination")
        raise ValueError("PAI job pagination limit exceeded")

    def workspace(self, workspace):
        value = self.request("aiworkspace", "GetWorkspace", "/api/v1/workspaces/" + component(workspace))
        if str(value.get("WorkspaceId")) != str(workspace):
            raise ValueError("PAI workspace identity differs")
        return value
