"""Host-only Alibaba OpenAPI transport using the official signing SDK.

No CLI processes, credential-bearing manifests, implicit write retries or raw
provider exceptions cross this boundary. HTTP 200 error envelopes are errors.
"""
from __future__ import annotations

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
        path = Path(os.environ.get("ML_EXP_PAI_CONFIG", "/etc/ml-exp/pai.json"))
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
