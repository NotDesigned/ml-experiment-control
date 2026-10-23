"""CCI debugging through REST; each new port rule belongs to one session UID."""
from copy import deepcopy
import ipaddress
import re
import secrets
import shlex
from urllib.parse import urlencode
import uuid

from experiment_control.backends.sensecore_rest import SenseCoreREST, RESTError, scope_path, component


def subset(expected, actual):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and subset(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(subset(a, b) for a, b in zip(expected, actual))
    return expected == actual


def startup(record):
    """No package installation or private key transfer; image must contain sshd."""
    key = shlex.quote(record["request"]["ssh_public_key"])
    deadline = int(record["expires_at"])
    return f'''set -eu
umask 077
test -x /usr/sbin/sshd
mkdir -p /run/sshd /run/ml-expd-ssh
chmod 700 /run/ml-expd-ssh
printf '%s\\n' {key} > /run/ml-expd-ssh/authorized_keys
test -f /run/ml-expd-ssh/host_ed25519 || ssh-keygen -q -t ed25519 -N '' -f /run/ml-expd-ssh/host_ed25519
cat > /run/ml-expd-ssh/sshd_config <<'EOF'
Port 22
HostKey /run/ml-expd-ssh/host_ed25519
PidFile /run/ml-expd-ssh/sshd.pid
AuthorizedKeysFile /run/ml-expd-ssh/authorized_keys
PermitRootLogin prohibit-password
PubkeyAuthentication yes
AuthenticationMethods publickey
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitEmptyPasswords no
UsePAM yes
AllowUsers root
Subsystem sftp internal-sftp
EOF
remaining=$(({deadline} - $(date +%s)))
test "$remaining" -gt 0
exec timeout "$remaining" /usr/sbin/sshd -D -e -f /run/ml-expd-ssh/sshd_config
'''


class CciDebug:
    def __init__(self, profile, api=None):
        self.profile, self.config = profile, profile["native"]
        self.api = api or SenseCoreREST.from_environment()
        self.workspace = self.api.workspace(self.config["workspace"])
        self.owner = self.api.identity()

    def url(self, record, service=False):
        product, collection = ("service", "services") if service else ("cci", "apps")
        return ("https://cci." + self.workspace["region"] + ".sensecore.cn/compute/" + product
                + "/data/v2" + scope_path(self.workspace, "workspaces") + "/" + collection + "/" + component(record["name"]))

    def document(self, record):
        c, p = self.config, self.profile
        if set(c) != {"workspace", "aec2", "worker_spec", "image", "mounts", "eip_name", "external_ip"}:
            raise ValueError("invalid CCI debug profile fields")
        pool = self.api.pool(c)
        spec = next((x for x in self.api.specs(c) if x["name"] == c["worker_spec"]), None)
        if (spec is None or spec["cpu"]["vcpu_allocatable"] != p["cpus"]
                or spec["memory"]["allocatable"] != p["memory_gib"] or spec["device"]["number"] != p["gpus"]):
            raise ValueError("CCI debug resource specification differs")
        if not isinstance(c["image"], str) or not re.fullmatch(r"[A-Za-z0-9._/@:-]{1,512}", c["image"]):
            raise ValueError("invalid configured CCI image")
        if not isinstance(c["mounts"], list) or not c["mounts"]:
            raise ValueError("CCI debug requires approved persistent mounts")
        for m in c["mounts"]:
            if (set(m) != {"type", "mount_path", "id", "subdir"} or m["type"] != "PV_AFS"
                    or m["mount_path"] != "/data" or not isinstance(m["subdir"], str)
                    or not re.fullmatch(r"/[A-Za-z0-9_-]{1,62}", m["subdir"])):
                raise ValueError("CCI mount must use an explicitly approved personal NAS subdirectory")
            component(m["id"])
        self.eip()  # validate the existing network before any instance creation
        request = {"cpu": str(p["cpus"]), "memory": str(p["memory_gib"]) + "GiB"}
        if p["gpus"]:
            key = spec["device"].get("resource_key")
            if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9.-]+/[A-Za-z0-9_.-]+", key):
                raise ValueError("CCI GPU resource key is unavailable")
            request[key] = str(p["gpus"])
        return {"name": record["name"], "display_name": "ml-expd-" + record["identity"],
                "resource_pool": {"name": pool["name"], "available_zone": pool["zone"], "vpc_id": pool["vpc_id"]},
                "replicas": 1, "template": {"resource_spec": {"name": c["worker_spec"]}, "containers": [
                    {"name": "main", "image_path": c["image"], "resource_request": request,
                     "command": ["/bin/sh", "-c", startup(record)], "env": [], "volume_mounts": c["mounts"]}]},
                "scheduling": {"priority": "NORMAL", "quota_type": "RESERVED"},
                "termination_grace_period_seconds": 15}

    def optional(self, url):
        try:
            return self.api.request(url)
        except RESTError as exc:
            if exc.status == 404:
                return None
            raise

    def checked(self, record, value):
        if (not isinstance(value, dict) or value.get("name") != record["name"] or not value.get("uid")
                or record.get("provider_id") not in (None, value["uid"])
                or value.get("ownership", {}).get("user_id") != record["owner"]
                or record["owner"] != self.owner
                or not subset({k: record["document"][k] for k in ("display_name", "resource_pool", "template")}, value)):
            raise ValueError("CCI debug identity or frozen specification differs")

    def lookup(self, record):
        value = self.optional(self.url(record))
        if value is not None:
            self.checked(record, value)
        return value

    def provision(self, record, save, claim):
        value = self.lookup(record)
        if value is None:
            if not claim("create"):
                return None
            self.api.request(self.url(record).rsplit("/", 1)[0] + "?" + urlencode(
                {"app_name": record["name"], "request_id": record["create_request_id"]}), method="POST", body=record["document"], timeout=90)
            record["create_accepted"] = True
            save()
            value = self.lookup(record)
        if value is not None:
            record["provider_id"] = value["uid"]
            record["expiry_enforcement"] = "server_cleanup_and_absolute_sshd_deadline"
            save()
            url = self.url(record, service=True)
            service = self.optional(url)
            if service and not service.get("uid") and not service.get("id") and service.get("ports") == []:
                service = None
            if service is None:
                if claim("service"):
                    self.api.request(url.rsplit("/", 1)[0] + "?" + urlencode(
                        {"name": record["name"], "request_id": record["service_request_id"]}), method="POST",
                        body={"name": record["name"], "selector": {"app": record["name"]}, "ports": [{"port": 22, "target_port": 22}]})
                service = self.optional(url)
            if service is not None and service.get("uid"):
                if (service.get("selector") != {"app": record["name"]}
                        or record.get("service_uid") not in (None, service["uid"])
                        or not subset([{"port": 22, "target_port": 22}], service.get("ports"))):
                    raise ValueError("CCI SSH service identity differs")
                record["service_uid"] = service["uid"]
                save()
                self.network(record, save, claim)
        return value

    def eip(self):
        rows = [x for x in self.api.resources("network.eip.v1.eip") if x["name"] == self.config["eip_name"]]
        if len(rows) != 1:
            raise ValueError("CCI debug EIP missing or ambiguous")
        value = rows[0]
        props = value.get("properties", {})
        import json
        props = json.loads(props) if isinstance(props, str) else props
        pool = self.api.pool(self.config)
        if (value["zone"] != pool["zone"] or props.get("vpc_id") != pool["vpc_id"] or not props.get("association_id")):
            raise ValueError("CCI debug EIP and pool network differ")
        ipaddress.ip_address(self.config["external_ip"])
        return value, props

    def rules(self):
        e, _ = self.eip()
        url = "https://network." + e["region"] + ".sensecoreapi.cn/network/eip/data/v1" + scope_path(e, "eips") + "/dnatRules"
        return url, [r for r in self.api.pages(url, "dnat_rules") if not r.get("deleted")]

    def rule(self, record):
        url, rows = self.rules()
        row = next((r for r in rows if r["name"] == record["name"]), None)
        if row is not None:
            plan = record.get("network_plan")
            p = row.get("properties", {})
            if (not plan or any(row.get(k) != plan[k] for k in ("uid", "creator_id", "owner_id", "tenant_id"))
                    or any(p.get(k) != plan["properties"][k] for k in ("external_ip", "external_port", "protocol", "eip_id"))
                    or p.get("internal_instance_name") not in (None, "", record.get("provider_id"))
                    or p.get("internal_instance_type") not in (None, "UNSPECIFIED", "CCI_DEPLOYMENT_SERVICE")
                    or p.get("internal_instance_name") and p.get("internal_port") != "22"):
                raise ValueError("CCI debug DNAT ownership or target differs")
        return url + "/" + component(record["name"]), row

    def network(self, record, save, claim):
        if "network_plan" not in record:
            e, ep = self.eip()
            _, rows = self.rules()
            occupied = set()
            for r in rows:
                port = r.get("properties", {}).get("external_port", "")
                if not isinstance(port, str) or not re.fullmatch(r"\d+(?:-\d+)?", port):
                    raise ValueError("invalid existing DNAT port")
                lo, hi = int(port.split("-")[0]), int(port.split("-")[-1])
                if not 1 <= lo <= hi <= 65535:
                    raise ValueError("invalid DNAT port range")
                occupied.update(range(lo, hi + 1))
            available = sorted(set(range(20000, 65536)) - occupied)
            if not available:
                raise ValueError("CCI debug SSH ports are exhausted")
            me = self.api.request("https://iam.sensecoreapi.cn/iam/idp/v1/me")
            if me.get("id") != record["owner"] or not me.get("tenant_id"):
                raise ValueError("CCI debug network account differs")
            base = scope_path(e, "eips") + "/dnatRules/" + record["name"]
            record["network_plan"] = {"name": record["name"], "display_name": record["name"], "id": base,
                "uid": str(uuid.uuid4()), "resource_type": "network.eip.v1.dnatRule", "zone": e["zone"],
                "creator_id": me["id"], "owner_id": me["id"], "tenant_id": me["tenant_id"], "properties": {
                    "external_ip": self.config["external_ip"], "external_port": str(secrets.choice(available)),
                    "internal_port": "22", "protocol": "tcp", "eip_id": e["id"], "nat_gateway_id": ep["association_id"],
                    "internal_ip": "", "internal_instance_name": "", "internal_instance_type": "UNSPECIFIED"}}
            save()
        url, row = self.rule(record)
        if row is None:
            if claim("dnat_create"):
                self.api.request(url, method="POST", body=record["network_plan"])
            return
        if row.get("state") == "CREATED" and not row["properties"].get("internal_instance_name"):
            if claim("dnat_bind"):
                body = deepcopy(row)
                body["properties"].update(internal_instance_type="CCI_DEPLOYMENT_SERVICE", internal_instance_name=record["provider_id"], internal_port="22")
                self.api.request(url + "/bind", method="POST", body=body)
        elif row.get("state") == "ACTIVE" and row["properties"].get("internal_instance_name") == record["provider_id"]:
            record["connection"] = {"host": self.config["external_ip"], "port": int(row["properties"]["external_port"]), "user": "root", "access": "internet"}

    def connection(self, value):
        return None  # network() writes only a checked session-owned DNAT route

    def state(self, value):
        return {"RUNNING": "RUNNING", "SUSPENDED": "STOPPED", "FAILED": "FAILED"}.get(value.get("state"), "STARTING")

    def stop(self, record, value, delete=False):
        if value is not None:
            self.checked(record, value)
        if not delete:
            if value is not None and value.get("state") != "SUSPENDED":
                self.api.request(self.url(record) + ":stop", method="POST", body={})
            return
        if value is not None and value.get("state") not in {"SUSPENDED", "FAILED", "DELETING"}:
            # Release compute before port cleanup: a network error must not
            # keep an expired machine consuming quota.
            if value.get("state") != "SUSPENDING":
                self.api.request(self.url(record) + ":stop", method="POST", body={})
            return False
        if record.get("network_plan"):
            url, row = self.rule(record)
            if row is not None:
                if row["properties"].get("internal_instance_name"):
                    self.api.request(url + "/unbind", method="POST", body={})
                elif row.get("state") == "CREATED":
                    self.api.request(url, method="DELETE")
                return False
        if value is not None:
            self.api.request(self.url(record), method="DELETE")
            return False
        service = self.optional(self.url(record, service=True))
        if service and service.get("uid"):
            if service["uid"] != record.get("service_uid") or service.get("selector") != {"app": record["name"]}:
                raise ValueError("CCI debug service was replaced")
            self.api.request(self.url(record, service=True), method="DELETE")
            return False
        return True
