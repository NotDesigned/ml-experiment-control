"""Isolated, target-specific publication of archived Attempt records to W&B.

The publisher deliberately knows nothing about collection or SQLite.  Its
input is one durable outbox item and its output is an acknowledgement suitable
for committing by the caller.  Backend files remain canonical.

The production adapter invokes the W&B SDK in a short-lived child process with
an explicitly constructed environment.  In particular, it never inherits the
daemon's environment (and therefore cannot accidentally inherit a Cloud API
key, proxy credential, or an unrelated user's W&B configuration).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Protocol
from urllib.parse import quote, urlsplit, urlunsplit


_SECRET_KEY = re.compile(
    r"(?:secret|token|password|credential|access[_-]?key|api[_-]?key|"
    r"authorization|cookie|proxy)",
    re.IGNORECASE,
)
_SAFE_ERROR = re.compile(r"[^A-Za-z0-9_.-]+")
_URL = re.compile(r"https?://[^\s<>'\"]+", re.IGNORECASE)
_SECRET_TEXT = re.compile(
    r"(?i)(?:\bbearer\s+[A-Za-z0-9._~+\-/]+=*|"
    r"\b(?:wandb[_-]?)?(?:api[_-]?key|access[_-]?key|secret|token|password|credential|"
    r"authorization|cookie)\s*[=:]\s*\S+|"
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----)"
)
_ITEM_KINDS = frozenset({"metric", "event", "log"})


class TargetKind(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"


@dataclass(frozen=True)
class AttemptIdentity:
    workspace_id: str
    project: str
    run_id: str
    attempt_id: str

    def __post_init__(self) -> None:
        for field in ("workspace_id", "project", "run_id", "attempt_id"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value or "\x00" in value:
                raise ValueError(f"{field} must be a non-empty string without NUL bytes")

    @property
    def wandb_run_id(self) -> str:
        """Return the target-independent stable ID for this immutable Attempt."""

        encoded = "\x00".join(
            (self.workspace_id, self.project, self.run_id, self.attempt_id)
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:32]

    @property
    def display_name(self) -> str:
        return f"{self.run_id} · {self.attempt_id}"


@dataclass(frozen=True)
class PublicationItem:
    """One sanitized record from a durable, target-specific outbox.

    ``sequence`` must be stable and monotonically increasing for one Attempt
    and target.  The adapter resumes the stable W&B run and logs at this exact
    step, so replaying an unacknowledged item cannot create a second history
    step.
    """

    target: TargetKind
    record_key: str
    sequence: int
    kind: str
    payload: Mapping[str, Any]
    timestamp: Optional[str | float] = None

    def __post_init__(self) -> None:
        if not self.record_key or len(self.record_key) > 256:
            raise ValueError("record_key must be between 1 and 256 characters")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.kind not in _ITEM_KINDS:
            raise ValueError(f"unsupported publication kind: {self.kind}")
        _validate_payload(self.payload)
        # Freeze only the top-level mapping. Nested values are validated and
        # copied again while constructing the subprocess request.
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


@dataclass(frozen=True)
class TargetConfig:
    kind: TargetKind
    api_url: str
    dashboard_url: str
    entity: str
    project: str
    working_dir: Path
    credential_ref: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "api_url", _safe_base_url(self.api_url, "api_url"))
        object.__setattr__(
            self, "dashboard_url", _safe_base_url(self.dashboard_url, "dashboard_url")
        )
        if not self.entity or not self.project:
            raise ValueError("W&B entity and project are required")
        if self.kind is TargetKind.CLOUD and not self.credential_ref:
            raise ValueError("Cloud target requires a credential reference")
        object.__setattr__(self, "working_dir", Path(self.working_dir).expanduser().resolve())

    def run_url(self, identity: AttemptIdentity) -> str:
        parts = [
            quote(self.entity, safe=""),
            quote(self.project, safe=""),
            "runs",
            quote(identity.wandb_run_id, safe=""),
        ]
        return f"{self.dashboard_url}/{'/'.join(parts)}"


class CredentialProvider(Protocol):
    def __call__(self, credential_ref: str) -> Optional[str]: ...


@dataclass(frozen=True)
class PublishRequest:
    target: TargetConfig
    identity: AttemptIdentity
    item: PublicationItem

    def worker_payload(self) -> dict[str, Any]:
        # Revalidate at the process boundary in case a caller mutated a nested
        # container after constructing the frozen top-level item.
        _validate_payload(self.item.payload)
        return {
            "target": {
                "kind": self.target.kind.value,
                "api_url": self.target.api_url,
                "entity": self.target.entity,
                "project": self.target.project,
                "working_dir": str(self.target.working_dir),
            },
            "identity": {
                "workspace_id": self.identity.workspace_id,
                "project": self.identity.project,
                "run_id": self.identity.run_id,
                "attempt_id": self.identity.attempt_id,
                "wandb_run_id": self.identity.wandb_run_id,
                "display_name": self.identity.display_name,
            },
            "item": {
                "record_key": self.item.record_key,
                "sequence": self.item.sequence,
                "kind": self.item.kind,
                "payload": _json_copy(self.item.payload),
                "timestamp": self.item.timestamp,
            },
        }


class PublisherAdapter(Protocol):
    def publish(self, request: PublishRequest, *, environment: Mapping[str, str]) -> None: ...


class PublisherProcessError(RuntimeError):
    """A deliberately detail-free error safe to retain in target state."""

    def __init__(self, error_class: str):
        self.error_class = _sanitize_error_class(error_class)
        super().__init__(self.error_class)


@dataclass(frozen=True)
class PublishResult:
    acknowledged: bool
    target: TargetKind
    record_key: str
    run_id: str
    dashboard_url: Optional[str]
    error_class: Optional[str] = None


class SubprocessWandbAdapter:
    """Run one W&B SDK operation in an environment-isolated child process."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 60.0,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._timeout_seconds = timeout_seconds
        self._runner = runner

    def publish(self, request: PublishRequest, *, environment: Mapping[str, str]) -> None:
        request.target.working_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        payload = json.dumps(
            request.worker_payload(), ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        try:
            result = self._runner(
                [sys.executable, "-m", "ml_exp_server.wandb_publisher", "--worker"],
                input=payload,
                text=True,
                capture_output=True,
                check=False,
                timeout=self._timeout_seconds,
                env=dict(environment),
                cwd=str(request.target.working_dir),
            )
        except subprocess.TimeoutExpired as exc:
            raise PublisherProcessError("PublisherTimeout") from exc
        except OSError as exc:
            raise PublisherProcessError(type(exc).__name__) from exc
        if result.returncode != 0:
            # Child output can contain SDK/server details and is intentionally
            # discarded rather than entering SQLite or an API response.
            raise PublisherProcessError("WandbWorkerFailed")


class WandbPublisher:
    """Publish one outbox item without coupling Local and Cloud target state."""

    def __init__(
        self,
        adapter: PublisherAdapter,
        *,
        credential_provider: Optional[CredentialProvider] = None,
    ) -> None:
        self._adapter = adapter
        self._credential_provider = credential_provider

    def publish(
        self,
        target: TargetConfig,
        identity: AttemptIdentity,
        item: PublicationItem,
    ) -> PublishResult:
        if item.target is not target.kind:
            return self._failure(target, identity, item, "TargetMismatch")

        api_key: Optional[str] = None
        if target.credential_ref is not None:
            if self._credential_provider is None:
                return self._failure(target, identity, item, "CredentialUnavailable")
            try:
                api_key = self._credential_provider(target.credential_ref)
            except Exception as exc:  # provider errors are an external boundary
                return self._failure(target, identity, item, type(exc).__name__)
            if not api_key:
                return self._failure(target, identity, item, "CredentialUnavailable")

        environment = build_publisher_environment(target, api_key=api_key)
        request = PublishRequest(target=target, identity=identity, item=item)
        try:
            self._adapter.publish(request, environment=environment)
        except Exception as exc:  # adapter errors must never leak raw messages
            error_class = getattr(exc, "error_class", type(exc).__name__)
            return self._failure(target, identity, item, error_class)
        return PublishResult(
            acknowledged=True,
            target=target.kind,
            record_key=item.record_key,
            run_id=identity.wandb_run_id,
            dashboard_url=target.run_url(identity),
        )

    @staticmethod
    def _failure(
        target: TargetConfig,
        identity: AttemptIdentity,
        item: PublicationItem,
        error_class: str,
    ) -> PublishResult:
        return PublishResult(
            acknowledged=False,
            target=target.kind,
            record_key=item.record_key,
            run_id=identity.wandb_run_id,
            dashboard_url=None,
            error_class=_sanitize_error_class(error_class),
        )


def build_publisher_environment(
    target: TargetConfig, *, api_key: Optional[str] = None
) -> dict[str, str]:
    """Construct a complete environment; never copy from ``os.environ``."""

    work = str(target.working_dir)
    environment = {
        "HOME": work,
        "WANDB_BASE_URL": target.api_url,
        "WANDB_CACHE_DIR": str(target.working_dir / "cache"),
        "WANDB_CONFIG_DIR": str(target.working_dir / "config"),
        "WANDB_CONSOLE": "off",
        "WANDB_DIR": work,
        "WANDB_MODE": "online",
        "WANDB_SILENT": "true",
    }
    if target.kind is TargetKind.CLOUD and not api_key:
        raise ValueError("Cloud target requires an API key")
    if api_key:
        environment["WANDB_API_KEY"] = api_key
    return environment


def _safe_base_url(value: str, field: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{field} must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{field} must not contain user information")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{field} must not contain a query or fragment")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError(f"{field} contains an invalid port") from exc
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _validate_payload(value: Any, *, key: Optional[str] = None) -> None:
    if key is not None and _SECRET_KEY.search(key):
        raise ValueError("publication payload contains a secret-like key")
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            if not isinstance(child_key, str):
                raise ValueError("publication payload keys must be strings")
            _validate_payload(child, key=child_key)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_payload(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("publication payload contains a non-finite number")
    elif isinstance(value, str):
        if _SECRET_TEXT.search(value) or _contains_url_userinfo(value):
            raise ValueError("publication payload contains credential-like text")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError("publication payload contains a non-JSON value")


def _json_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        value = dict(value)
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _contains_url_userinfo(value: str) -> bool:
    for match in _URL.finditer(value):
        try:
            parsed = urlsplit(match.group(0).rstrip(".,;)"))
        except ValueError:
            return True
        if parsed.username is not None or parsed.password is not None:
            return True
    return False


def _sanitize_error_class(value: Any) -> str:
    safe = _SAFE_ERROR.sub("_", str(value))[:64].strip("_")
    return safe or "PublisherError"


def _worker_log_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    kind = item["kind"]
    payload = item["payload"]
    base: dict[str, Any] = {
        "_ml_expd/record_key": item["record_key"],
        "_ml_expd/kind": kind,
    }
    if item.get("timestamp") is not None:
        base["_ml_expd/timestamp"] = item["timestamp"]
    if kind == "metric":
        base.update(payload)
    elif kind == "event":
        base["_ml_expd/event"] = json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
    else:
        base["_ml_expd/log"] = payload.get("text", "")
        if payload.get("stream") is not None:
            base["_ml_expd/stream"] = payload["stream"]
    return base


def _run_worker(raw: str) -> int:
    """Child entry point. Credentials arrive only through the child env."""

    try:
        data = json.loads(raw)
        import wandb  # imported only inside the isolated worker

        target = data["target"]
        identity = data["identity"]
        item = data["item"]
        run = wandb.init(
            project=target["project"],
            entity=target["entity"],
            id=identity["wandb_run_id"],
            name=identity["display_name"],
            resume="allow",
            reinit="finish_previous",
            dir=target["working_dir"],
            tags=["ml-expd", target["kind"]],
            config={
                "workspace_id": identity["workspace_id"],
                "project": identity["project"],
                "run_id": identity["run_id"],
                "attempt_id": identity["attempt_id"],
            },
            settings=wandb.Settings(silent=True, console="off"),
        )
        if run is None:
            return 2
        run.log(_worker_log_payload(item), step=item["sequence"], commit=True)
        run.finish(exit_code=0)
        return 0
    except Exception:
        # Do not print exception text: it can contain URLs, headers, or values
        # supplied by a remote service. The parent records only a safe class.
        return 1


def _main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--worker":
        return _run_worker(sys.stdin.read())
    return 2


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess adapter
    raise SystemExit(_main())
