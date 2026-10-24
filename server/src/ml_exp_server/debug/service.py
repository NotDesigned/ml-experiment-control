"""Durable debug-machine ownership, bounded admission and expiry reconciliation.

An uncertain create is observed by its original name/identity, never replayed.
These records deliberately do not create scientific Runs or training Attempts.
"""
from copy import deepcopy
from pathlib import Path
import re
import shlex
import time
import uuid

from ..application_errors import ApplicationError
from .models import DebugProfile, DebugRequest, identity
from ..storage import DurableJsonState, exclusive_file_lock, read_json


TERMINAL = {"STOPPED", "DELETED", "REJECTED"}


def adapter(profile):
    if profile["provider"] == "pai-dsw":
        from .dsw import DswDebug
        return DswDebug(profile)
    from .cci import CciDebug
    return CciDebug(profile)


def safe_error(exc):
    details = getattr(exc, "details", {})
    result = {"code": "DEBUG_PROVIDER_UNCERTAIN", "error_class": type(exc).__name__}
    for k in ("operation", "provider_code", "provider_reason", "http_status", "uncertain"):
        v = details.get(k)
        if type(v) in (int, bool) or isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", v):
            result[k] = v
    return result


class DebugSessions:
    def __init__(self, config):
        self.config = config
        self.root = config.project_registry_root_path() / "debug-sessions"

    def profiles(self):
        path = self.config.debug_sessions.profiles_file
        data = read_json(Path(path), {}) if path else {}
        if not isinstance(data, dict) or set(data) != {"profiles"} or not isinstance(data["profiles"], dict):
            if not path:
                return {}
            raise ValueError("invalid debug session profile file")
        profiles = {}
        for key, value in data["profiles"].items():
            self.name(key)
            profiles[key] = DebugProfile.model_validate(value).model_dump()
        return profiles

    def catalog(self):
        profiles = self.profiles()
        return {"enabled": any(p["enabled"] for p in profiles.values()) and self.config.action_runtime.allow_scheduler_mutations,
                "max_active": self.config.debug_sessions.max_active,
                "max_duration_seconds": self.config.debug_sessions.max_duration_seconds,
                "profiles": [{"id": k, "operations": ["create", "status", "refresh", "stop"] + (["delete"] if v["provider"] == "pai-dsw" else []),
                              "expiry_action": "delete" if v["provider"] == "pai-dsw" else "stop",
                              **{field: v[field] for field in ("provider", "description", "cpus", "memory_gib", "gpus", "enabled", "unavailable_reason")}}
                             for k, v in profiles.items()]}

    @staticmethod
    def name(value):
        if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,62}", value):
            raise ApplicationError("invalid debug session identity", status_code=422, code="DEBUG_ID_INVALID")
        return value

    def store(self, key):
        key = self.name(key)
        return DurableJsonState(self.root / (key + ".json"), self.root / (key + ".events.jsonl"))

    def load(self, key):
        value = self.store(key).snapshot({}).value
        if not value:
            raise ApplicationError("debug session not found", status_code=404, code="DEBUG_NOT_FOUND")
        if value.get("session_id") != key or value.get("identity") != identity({"request": value.get("request"), "profile": value.get("profile")}):
            raise ValueError("debug session frozen identity differs")
        return value

    def save(self, record):
        record["updated_at"] = time.time()
        self.store(record["session_id"]).commit(record, event={
            "status": record["status"], "at": record["updated_at"], "intent": record.get("intent"), "error": record.get("error")})

    def view(self, record):
        # Never return native cloud request documents, account fields or mounts.
        result = {k: deepcopy(record.get(k)) for k in (
            "session_id", "status", "provider_id", "created_at", "updated_at", "expires_at",
            "error", "expiry_enforcement", "intent", "machine_status", "machine_observed_at")}
        result.update(profile=record["request"]["profile"], provider=record["profile"]["provider"],
                      resources={k: record["profile"][k] for k in ("cpus", "memory_gib", "gpus")})
        route = record.get("connection") if record["status"] == "RUNNING" and record["expires_at"] > time.time() else None
        if route:
            host, port = route["host"], route["port"]
            if (not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.:-]{0,252}", host)
                    or type(port) is not int or not 1 <= port <= 65535):
                raise ValueError("invalid SSH connection information")
            result["connection"] = {**route, "status": "CONFIGURED", "ssh_command": "ssh -p " + str(port) + " " + shlex.quote("root@" + host),
                                    "host_key_verified": False}
        else:
            result["connection"] = None
        return result

    def read(self, key):
        with exclusive_file_lock(self.root / ".lock"):
            return self.view(self.load(key))

    def listing(self):
        with exclusive_file_lock(self.root / ".lock"):
            return {"sessions": [self.view(self.load(p.stem)) for p in sorted(self.root.glob("*.json"))]}

    def writable(self):
        if not self.config.action_runtime.allow_scheduler_mutations:
            raise ApplicationError("scheduler mutations are disabled", status_code=403, code="DEBUG_DISABLED")

    def create(self, request: DebugRequest):
        self.writable()
        with exclusive_file_lock(self.root / ".lock"):
            # Idempotency is bound to the complete canonical request, never just a name.
            if self.store(request.session_id).path.exists():
                existing = self.load(request.session_id)
                if existing["request"] != request.model_dump():
                    raise ApplicationError("session ID already binds another request", code="DEBUG_ID_CONFLICT")
                return self.view(existing)
            profiles = self.profiles()
            if request.profile not in profiles:
                raise ApplicationError("unknown debug profile", status_code=404, code="DEBUG_PROFILE_NOT_FOUND")
            if not profiles[request.profile]["enabled"]:
                raise ApplicationError("debug profile is disabled; inspect the profile catalog", status_code=403, code="DEBUG_PROFILE_DISABLED")
            limit = min(self.config.debug_sessions.max_duration_seconds, 14400) if profiles[request.profile]["provider"] == "pai-dsw" else self.config.debug_sessions.max_duration_seconds
            if request.duration_seconds > limit:
                raise ApplicationError("debug session duration exceeds server limit", status_code=422, code="DEBUG_DURATION_LIMIT")
            active = sum(self.load(p.stem)["status"] not in TERMINAL for p in self.root.glob("*.json"))
            if active >= self.config.debug_sessions.max_active:
                raise ApplicationError("debug session capacity is full", status_code=503, code="DEBUG_CAPACITY")
            now = time.time()
            frozen = {"request": request.model_dump(), "profile": profiles[request.profile]}
            record = {**frozen, "session_id": request.session_id, "identity": identity(frozen),
                      "status": "CREATING", "created_at": now, "expires_at": now + request.duration_seconds,
                      "provider_id": None, "claims": [], "intent": None,
                      "create_request_id": str(uuid.uuid4()), "service_request_id": str(uuid.uuid4())}
            record["name"] = ("ml-expd_" if record["profile"]["provider"] == "pai-dsw" else "ml-expd-") + record["identity"][:18]
            try:
                provider = adapter(record["profile"])
                # Validate all static mounts/resources/network before a cloud write.
                record["document"] = provider.document(record)
            except Exception:
                raise ApplicationError("debug provider preflight failed; no create was sent",
                                       code="DEBUG_PREFLIGHT_FAILED") from None
            if hasattr(provider, "owner"):
                record["owner"] = provider.owner
            self.save(record)
            return self.advance(record, provider)

    def refresh(self, key, *, intent=None):
        self.writable()
        with exclusive_file_lock(self.root / ".lock"):
            record = self.load(key)
            if record["status"] == "DELETED":
                return self.view(record)
            if record["status"] == "REJECTED" and intent is None:
                return self.view(record)
            if intent == "delete" and record["profile"]["provider"] == "sensecore-cci":
                raise ApplicationError("CCI supports stopping only; provider deletion is not enabled",
                                       code="DEBUG_DELETE_UNSUPPORTED")
            if intent:
                # A deletion/expiry can never be downgraded to stop or provisioning.
                if record.get("intent") != "delete":
                    record["intent"] = intent
                self.save(record)
            return self.advance(record)

    def advance(self, record, provider=None):
        previous_error = record.get("error")
        record["error"] = None
        def save():
            self.save(record)

        def claim(effect):
            if effect in record["claims"]:
                return False
            record["claims"].append(effect)
            save()  # Claim is durable BEFORE the irreversible provider request.
            return True

        try:
            provider = provider or adapter(record["profile"])
            if time.time() >= record["expires_at"]:
                record["intent"] = "stop" if record["profile"]["provider"] == "sensecore-cci" else "delete"
            if record["profile"]["provider"] == "sensecore-cci" and record.get("intent") == "delete":
                # Preserve the stopped instance: this backend intentionally
                # offers no app deletion, including for earlier cleanup records.
                record["intent"] = "stop"
            intent = record.get("intent")
            if intent:
                record.pop("connection", None)
                value = provider.lookup(record)
                record["machine_status"] = provider.state(value) if value is not None else (
                    "ABSENT" if record.get("provider_id") or record.get("create_rejected") else "UNKNOWN")
                record["machine_observed_at"] = time.time()
                if value is None and record.get("create_rejected"):
                    record["status"] = "DELETED"
                elif value is None and "create" in record["claims"] and not record.get("provider_id"):
                    # Missing create response plus absent GET is not proof of no resource.
                    record["status"] = "RECONCILE_REQUIRED"
                    record["error"] = previous_error or {"code": "DEBUG_CREATION_UNCERTAIN"}
                elif record["profile"]["provider"] == "sensecore-cci":
                    complete = provider.stop(record, value)
                    record["status"] = "STOPPED" if complete and (value is None or provider.state(value) == "STOPPED") else "STOPPING"
                elif value is None:
                    record["status"] = "DELETED"
                else:
                    provider.stop(record, value, delete=intent == "delete")
                    record["status"] = "DELETING" if intent == "delete" else (
                        "STOPPED" if provider.state(value) == "STOPPED" else "STOPPING")
            else:
                record.pop("connection", None)
                value = provider.provision(record, save, claim)
                if value is None:
                    record["status"] = "RECONCILE_REQUIRED"
                    record["error"] = previous_error or {"code": "DEBUG_CREATION_UNCERTAIN"}
                else:
                    record["status"] = provider.state(value)
                    record["machine_status"] = record["status"]
                    record["machine_observed_at"] = time.time()
                    connection = provider.connection(value)
                    if connection:
                        record["connection"] = connection
                    record["error"] = None
        except Exception as exc:
            record["status"] = "RECONCILE_REQUIRED"
            record.pop("connection", None)
            record["error"] = safe_error(exc)
            details = getattr(exc, "details", {})
            status = details.get("http_status")
            if ("create" in record["claims"] and not record.get("create_accepted") and not record.get("provider_id")
                    and details.get("uncertain") is False and type(status) is int and 400 <= status < 500):
                record["create_rejected"] = True
                record["status"] = "REJECTED"
                record["error"]["code"] = "DEBUG_PROVIDER_REJECTED"
        save()
        return self.view(record)

    def maintain(self):
        """Resume cleanup after restarts, even if profiles were later disabled."""
        if not self.config.action_runtime.allow_scheduler_mutations:
            return
        with exclusive_file_lock(self.root / ".lock"):
            for path in sorted(self.root.glob("*.json")):
                record = self.load(path.stem)
                if (record.get("error") or {}).get("http_status") == 403:
                    # A permission denial is not transient. Keep the resource
                    # and diagnostic; an explicit refresh retries after repair.
                    continue
                if record["status"] != "DELETED" and (record.get("intent") or time.time() >= record["expires_at"]):
                    self.advance(record)
