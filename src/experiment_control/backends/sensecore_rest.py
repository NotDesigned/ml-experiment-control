"""Host-owned SenseCore REST client. No CLI, redirects, proxy or write retries."""
from __future__ import annotations

import base64
import email.utils
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid


class RESTError(RuntimeError):
    def __init__(self, operation, *, status=None, reason=None, uncertain=False):
        self.status = status
        self.details = {"operation": operation, "http_status": status,
                        "provider_reason": reason, "uncertain": uncertain}
        super().__init__("SenseCore REST request failed" +
                         (f" (HTTP {status})" if status else "") +
                         ("; reconcile before any retry" if uncertain else ""))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def component(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}", value) or value in {".", ".."}:
        raise ValueError("invalid SenseCore resource component")
    return value


def scope_path(record, collection):
    sub, group, zone, name = (component(record[k]) for k in
                              ("subscription_name", "resource_group_name", "zone", "name"))
    return f"/subscriptions/{sub}/resourceGroups/{group}/zones/{zone}/{collection}/{name}"


def origin(service, record):
    region = record["region"]
    if not re.fullmatch(r"cn-[a-z]+-[0-9]+", region):
        raise ValueError("invalid SenseCore region")
    return f"https://{service}.{region}.sensecoreapi.cn"


def create_document(backend, name, image, command):
    """Translate the existing frozen backend definition without credentials."""
    component(name)
    volume, path = backend["storage_mount"].rsplit(":", 1)
    volume, _, subdir = volume.partition("/")
    if not volume or not path.startswith("/") or any(p == ".." for p in (path + "/" + subdir).split("/")):
        raise ValueError("invalid SenseCore storage mount")
    quota = str(backend["quota_type"]).upper()
    nodes = backend.get("worker_nodes", 1)
    if quota not in {"SPOT", "RESERVED"} or isinstance(nodes, bool) or not isinstance(nodes, int) or nodes < 1:
        raise ValueError("invalid SenseCore quota or worker count")
    return {"name": name, "display_name": backend.get("display_name", name),
            "framework": "PYTORCH", "roles": [{"name": "Worker",
                "resource_spec": [{"name": backend["worker_spec"]}],
                "total_replicas": nodes, "startup_script": command, "image_path": image}],
            "resource_pool": {"name": backend["aec2"]},
            "mount": [{"type": "PV_AFS", "id": volume, "mount_path": path,
                       "subdir": "/" + subdir.lstrip("/")}],
            "scheduling": {"priority": backend.get("priority", "NORMAL"), "quota_type": quota},
            "fault_tolerance": {"backoff_limit": 0}}


class SenseCoreREST:
    def __init__(self, config):
        self.config = config
        self._identity = None
        self._workspaces = {}
        self._pools = {}

    @classmethod
    def from_environment(cls) -> SenseCoreREST:
        path = os.environ.get("EXPERIMENTCTL_SENSECORE_REST_CONFIG", "/etc/ml-expd/sensecore-rest.json")
        try:
            config = json.loads(Path(path).read_text())
            if not isinstance(config, dict) or any(not isinstance(config.get(k), str) or not config[k].strip()
                    for k in ("access_key_id", "access_key_secret", "subscription_name", "resource_group_name")):
                raise ValueError
        except (OSError, ValueError):
            raise RESTError("configuration") from None
        return cls(config)

    def request(self, url, *, method="GET", body=None, timeout=30, signed=True):
        parsed = urllib.parse.urlsplit(url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.port not in (None, 443) or parsed.fragment
                or not any(parsed.hostname.endswith("." + d) for d in
                           ("sensecoreapi.cn", "sensecoreapi.tech", "sensecore.cn"))):
            raise ValueError("invalid SenseCore HTTPS endpoint")
        headers = {"Content-Type": "application/json"}
        if signed:
            date = email.utils.formatdate(usegmt=True)
            signature = base64.b64encode(hmac.new(self.config["access_key_secret"].encode(),
                         ("x-date: " + date).encode(), hashlib.sha256).digest()).decode()
            headers.update({"X-Date": date, "Authorization":
                f'hmac accesskey="{self.config["access_key_id"]}", algorithm="hmac-sha256", headers="x-date", signature="{signature}"'})
        req = urllib.request.Request(url, method=method, headers=headers,
                data=None if body is None else json.dumps(body).encode())
        operation = method + " " + parsed.path
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        try:
            with opener.open(req, timeout=timeout) as response:
                raw = response.read(32 * 1024 * 1024 + 1)
                if len(raw) > 32 * 1024 * 1024:
                    raise ValueError
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as error:
            reason = None
            try:
                detail = json.loads(error.read(65536))
                details = detail.get("details", []) if isinstance(detail, dict) else []
                for entry in details if isinstance(details, list) else []:
                    value = entry.get("reason") if isinstance(entry, dict) else None
                    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", value):
                        reason = value
                        break
            except ValueError:
                pass
            raise RESTError(operation, status=error.code, reason=reason,
                            uncertain=method != "GET" and error.code >= 500) from None
        except (OSError, ValueError, urllib.error.URLError):
            raise RESTError(operation, uncertain=method != "GET") from None

    def pages(self, url, field, *, query=None, numbered=False):
        result, tokens, names, token, total_seen = [], set(), set(), "1", 0
        for _ in range(1000):
            if token in tokens:
                raise ValueError("SenseCore repeated pagination token")
            tokens.add(token)
            data = self.request(url + "?" + urllib.parse.urlencode({**(query or {}), "page_size": 100, "page_token": token}))
            if not isinstance(data, dict) or not isinstance(data.get(field), list):
                raise ValueError("invalid SenseCore list response")
            rows, total, following = data[field], data.get("total_size"), data.get("next_page_token", "")
            if (total is not None and (isinstance(total, bool) or not isinstance(total, int) or total < len(rows))
                    or not isinstance(following, str)):
                raise ValueError("invalid SenseCore pagination metadata")
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not row["name"] or row["name"] in names:
                    raise ValueError("invalid or duplicate SenseCore resource identity")
                names.add(row["name"])
                result.append(row)
            total_seen += len(rows)
            if following and following != "0":
                token = following
            elif total is not None and total_seen < total:
                if not numbered or not rows or not token.isdecimal():
                    raise ValueError("incomplete SenseCore pagination")
                token = str(int(token) + 1)
            else:
                return result
        raise ValueError("SenseCore pagination limit exceeded")

    def identity(self):
        if self._identity is None:
            data = self.request("https://iam.sensecoreapi.cn/iam/idp/v1/me")
            try:
                if not isinstance(data, dict) or not uuid.UUID(data.get("id", "")).int:
                    raise ValueError
            except (ValueError, AttributeError, TypeError):
                raise ValueError("invalid SenseCore account identity") from None
            self._identity = data["id"]
        return self._identity

    def resources(self, kind):
        rows = self.pages("https://management.sensecoreapi.cn/rmh/v1/resources", "resources",
                          query={"filter": f"resource_type='{kind}'"})
        return [r for r in rows if r.get("type") == kind and not r.get("deleted")
                and all(r.get(k) == self.config[k] for k in ("subscription_name", "resource_group_name"))]

    def workspace(self, name):
        component(name)
        if name not in self._workspaces:
            rows = [r for r in self.resources("compute.workspace.v1.instance") if r["name"] == name]
            if len(rows) != 1:
                raise ValueError("SenseCore workspace is missing or ambiguous in configured scope")
            self._workspaces[name] = rows[0]
        return self._workspaces[name]

    def pools(self, backend):
        record = self.workspace(backend["workspace"])
        url = origin("aec2", record) + "/compute/workspace/data/v1" + scope_path(record, "workspaces") + "/workspaceAEC2Bindings"
        if url not in self._pools:
            rows = self.pages(url, "aec2s", numbered=True)
            pools = []
            for row in rows:
                if row.get("state") != "ACTIVE":
                    continue
                match = re.fullmatch(r"/subscriptions/([^/]+)/resourceGroups/([^/]+)/zones/([^/]+)/aec2s/([^/]+)", row.get("id", ""))
                if (not match or match[1] != record["subscription_name"] or match[2] != record["resource_group_name"]
                        or match[4] != row["name"] or not re.fullmatch(re.escape(record["region"]) + "[a-z]", match[3])):
                    raise ValueError("invalid SenseCore pool scope")
                pools.append({**row, "region": record["region"], "zone": match[3],
                              "subscription_name": match[1], "resource_group_name": match[2]})
            self._pools[url] = pools
        return self._pools[url]

    def pool(self, backend):
        rows = [r for r in self.pools(backend) if r["name"] == backend["aec2"]]
        if len(rows) != 1:
            raise ValueError("SenseCore pool is not bound to the configured workspace")
        return rows[0]

    def specs(self, backend):
        pool = self.pool(backend)
        rows = self.pages(origin("aec2", pool) + "/compute/aec2/data/v1" + scope_path(pool, "aec2s") + "/resourceSpecs",
                          "resource_specs", numbered=True)
        for r in rows:
            try:
                values = (r["cpu"]["vcpu_allocatable"], r["memory"]["allocatable"], r["device"]["number"])
                if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in values) or not all(values[:2]) or pool["zone"] not in r["zones"]:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                raise ValueError("invalid SenseCore resource specification") from None
        return rows

    def jobs_url(self, backend):
        record = self.workspace(backend["workspace"])
        return origin("aec2", record) + "/compute/acp/data/v2" + scope_path(record, "workspaces") + "/trainingJobs"

    def owned(self, row, name):
        if (not isinstance(row, dict) or row.get("name") != name or not row.get("uid")
                or not isinstance(row.get("ownership"), dict) or row["ownership"].get("user_id") != self.identity()):
            raise ValueError("SenseCore job identity or owner differs")
        return row

    def find(self, backend, name):
        component(name)
        rows = self.pages(self.jobs_url(backend), "training_jobs", numbered=True,
                          query={"filter": f"name='{name}'", "name": name})
        return [self.owned(r, name) for r in rows if r["name"] == name]

    def describe(self, backend, name):
        return self.owned(self.request(self.jobs_url(backend) + "/" + component(name)), name)

    def create(self, backend, document, *, timeout=120):
        pool = self.pool(backend)
        document = {**document, "mount": [{**m, "zone": pool["zone"]} for m in document["mount"]]}
        result = self.request(self.jobs_url(backend) + "?" + urllib.parse.urlencode({"training_job_name": document["name"]}),
                              method="POST", body=document, timeout=timeout)
        if not isinstance(result, dict) or result.get("name") != document["name"] or not result.get("uid"):
            raise RESTError("create confirmation", uncertain=True)
        return result

    def stop(self, backend, name):
        current = self.describe(backend, name)
        if current["state"] in {"SUCCEEDED", "FAILED", "STOPPED", "SUSPENDED", "DELETED"}:
            return
        record = self.workspace(backend["workspace"])
        body = {k: record[k] for k in ("subscription_name", "resource_group_name", "zone")}
        body.update(workspace_name=record["name"], training_job_names=[name])
        self.request(self.jobs_url(backend) + ":batchStop", method="POST", body=body)

    def workers(self, backend, name):
        self.describe(backend, name)
        return self.pages(self.jobs_url(backend) + "/" + component(name) + "/workers", "workers", numbered=True)

    def logs(self, backend, name, tail):
        job = self.describe(backend, name)
        workers = self.workers(backend, name)
        if not workers:
            return {"text": "", "expired": False, "exit_code": 0}
        worker = workers[0]
        record = self.workspace(backend["workspace"])
        stations = [r for r in self.resources("monitor.ts.v1.telemetryStation") if r.get("zone") == record["zone"]]
        personal = [r for r in stations if r["name"] == "ts-user-" + self.identity()]
        if len(personal) != 1:
            raise ValueError("SenseCore personal telemetry station is unavailable")
        station = personal[0]
        container = worker.get("containers", [{}])[0].get("name", "container-worker-0")
        resource_id = component(record["id"])
        body = {"resource_type": "resource_type.compute.workspace.v1.instance", "resource_id": resource_id,
                "filters": [{"key": k, "val": v} for k, v in {
                    "pod": worker["name"], "container": container,
                    "lepton.sensetime.com/workload-name": name,
                    "lepton.sensetime.com/workload-type": "acp",
                    "lepton.sensetime.com/workload-uid": job["uid"]}.items()]}
        try:
            token = self.request("https://monitor.sensecoreapi.cn/monitor/ts/data/v1" + scope_path(station, "telemetryStations") + "/logLivestream/token",
                                 method="POST", body=body, timeout=20)["token"]
            payload = json.loads(base64.urlsafe_b64decode(token.split(".")[1] + "=="))
            endpoint = payload["Endpoint"]
            if not isinstance(endpoint, str) or any(c in endpoint for c in "/?#@"):
                raise ValueError
            data = self.request("https://" + endpoint + "/v1/polling/resources/" + urllib.parse.quote(resource_id, safe=""),
                                method="POST", body={"token": token, "resource_id": resource_id, "tail": tail},
                                timeout=20, signed=False)
        except RESTError as error:
            expired = error.status == 403 and error.details["provider_reason"] in {"ExpiredPodToken", "ExpiredToken", "PodTokenExpired"}
            return {"text": "", "expired": expired, "exit_code": 1,
                    "available": False, "error": error.details}
        except (KeyError, ValueError, TypeError, IndexError):
            raise ValueError("invalid SenseCore log token or response") from None
        if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
            raise ValueError("invalid SenseCore log entries")
        lines = []
        for entry in data["entries"]:
            if not isinstance(entry, dict):
                raise ValueError("invalid SenseCore log entry")
            if str(entry.get("type", "")).upper() in {"BROKEN", "EOF", "HEARTBEAT"}:
                continue
            value = entry.get("message") or entry.get("content") or entry.get("data") or ""
            try:
                lines.append(base64.b64decode(value, validate=True).decode())
            except (ValueError, UnicodeError, TypeError):
                raise ValueError("invalid SenseCore encoded log content") from None
        return {"text": "\n".join(lines), "expired": False, "exit_code": 0}
