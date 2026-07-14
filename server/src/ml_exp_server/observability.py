"""Daemon-owned observability services.

The manager only owns the optional local W&B process.  Run evidence and logs
remain canonical in backend-owned run directories; publication is a separate,
best-effort projection and must never make collection fail.
"""

from __future__ import annotations

import subprocess
import os
import signal
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

from .schemas import LocalWandbConfig


@dataclass
class WandbServiceManager:
    config: LocalWandbConfig
    popen: Callable[..., Any] = subprocess.Popen
    healthcheck: Callable[[str], bool] | None = None
    port_probe: Callable[[str, int, float], bool] | None = None
    terminator: Callable[[Any, int | None], None] | None = None
    _process: Any = None
    _process_group_id: int | None = None
    _process_start_identity: str | None = None
    _descendant_process_identities: dict[int, str] = field(default_factory=dict)
    _state: str = "DISABLED"
    _error: str | None = None
    _started_at: float | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def __post_init__(self) -> None:
        self._state = "STANDBY" if self.config.enabled else "DISABLED"

    def start(self) -> dict[str, Any]:
        with self._lock:
            if not self.config.enabled:
                self._state, self._error = "DISABLED", None
                return self.status()
            if self._process is not None and self._process.poll() is None:
                return self.status()
            if not self.config.managed and self._state == "READY":
                return self.status()
            self._state, self._error = "STARTING", None
            try:
                data_dir = self.config.data_path()
                if self.config.managed:
                    data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                    try:
                        os.chmod(data_dir, 0o700)
                    except OSError:
                        pass
                    self._process = self.popen(
                        self.config.resolved_command(),
                        env=self._environment(data_dir),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    self._process_group_id = getattr(self._process, "pid", None)
                    self._process_start_identity = (
                        _process_start_time(self._process_group_id)
                        if isinstance(self._process_group_id, int) else None
                    )
                    self._capture_descendants()
                deadline = time.monotonic() + self.config.startup_timeout_seconds
                while time.monotonic() < deadline:
                    self._capture_descendants()
                    if self._process is not None and self._process.poll() is not None:
                        raise RuntimeError("local W&B process exited during startup")
                    try:
                        remaining = max(0.01, deadline - time.monotonic())
                        host, port = self._readiness_address()
                        probe_timeout = min(0.2, remaining / 2)
                        port_ready = (
                            _bounded_bool_call(
                                lambda: self.port_probe(host, port, probe_timeout),
                                probe_timeout,
                            )
                            if self.port_probe is not None else _bounded_bool_call(
                                lambda: _port_ready(host, port, probe_timeout),
                                probe_timeout,
                            )
                        )
                        remaining = max(0.01, deadline - time.monotonic())
                        probe_timeout = min(0.5, remaining)
                        if self.healthcheck is not None:
                            http_ready = _bounded_bool_call(
                                lambda: self.healthcheck(self.config.url()),
                                probe_timeout,
                            )
                        else:
                            http_ready = _bounded_bool_call(
                                lambda: _http_ready(self.config.url(), probe_timeout),
                                probe_timeout,
                            )
                        if port_ready and http_ready:
                            self._state = "READY"
                            self._started_at = time.time()
                            return self.status()
                    except (OSError, urllib.error.URLError, ValueError):
                        pass
                    time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
                raise TimeoutError("local W&B startup timed out")
            except Exception as exc:  # optional dependency must be degradable
                self._terminate_process()
                self._state, self._error = "DEGRADED", _safe_error(exc)
                return self.status()

    def stop(self) -> None:
        with self._lock:
            self._terminate_process()
            if self.config.enabled and self._state != "DEGRADED":
                self._state = "STOPPED"

    def status(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            if process is not None and process.poll() is not None and self._state == "READY":
                self._terminate_process()
                self._state = "DEGRADED"
                self._error = "local W&B process exited"
            return {
                "enabled": self.config.enabled,
                "managed": self.config.managed,
                "state": self._state,
                # Only a redacted origin is public.  Health paths can contain
                # deployment-specific data and remain daemon-local.
                "url": _redacted_origin(self.config.url()) if self.config.enabled else None,
                "started_at": self._started_at,
                "error": self._error,
            }

    def _environment(self, data_dir: Path) -> dict[str, str]:
        # Default empty: no proxy tokens, cloud credentials, HOME, shell hooks,
        # or unrelated application configuration cross the subprocess boundary.
        environment = {
            name: os.environ[name]
            for name in self.config.environment_allowlist
            if name in os.environ
        }
        environment["WANDB_DIR"] = str(data_dir)
        return environment

    def _readiness_address(self) -> tuple[str, int]:
        parsed = urlsplit(self.config.url())
        assert parsed.hostname is not None
        return parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)

    def _terminate_process(self) -> None:
        process, self._process = self._process, None
        process_group_id, self._process_group_id = self._process_group_id, None
        parent_identity, self._process_start_identity = self._process_start_identity, None
        descendants = self._descendant_process_identities
        self._descendant_process_identities = {}
        if process is None:
            return
        try:
            if self.terminator is not None:
                self.terminator(process, process_group_id)
            else:
                _terminate_process_tree(
                    process, process_group_id, descendants,
                    parent_start_identity=parent_identity,
                )
        except Exception:
            # Cleanup is best effort, but the owned handle is always forgotten
            # so a later start cannot treat a dead process as READY.
            pass

    def _capture_descendants(self) -> None:
        process = self._process
        pid = getattr(process, "pid", None)
        if isinstance(pid, int) and pid > 0:
            self._descendant_process_identities.update(_descendant_identities(pid))


def _safe_error(exc: BaseException) -> str:
    # Never include exception text: FileNotFoundError, command wrappers, HTTP
    # clients and custom health checks can embed credentials or full URLs.
    if isinstance(exc, FileNotFoundError):
        return "local W&B executable was not found"
    if isinstance(exc, TimeoutError):
        return "local W&B readiness timed out"
    if isinstance(exc, OSError):
        return "local W&B process could not be started"
    if isinstance(exc, RuntimeError):
        return "local W&B process exited during startup"
    return "local W&B startup failed"


def _redacted_origin(url: str) -> str:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    netloc = host
    if parsed.port is not None:
        netloc += f":{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, "", "", ""))


def _port_ready(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_ready(url: str, timeout: float) -> bool:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return 200 <= int(response.status) < 400


def _bounded_bool_call(call: Callable[[], bool], timeout: float) -> bool:
    result = False
    complete = threading.Event()

    def invoke() -> None:
        nonlocal result
        try:
            result = bool(call())
        except Exception:
            result = False
        finally:
            complete.set()

    thread = threading.Thread(target=invoke, name="wandb-readiness", daemon=True)
    thread.start()
    return bool(complete.wait(max(0.001, timeout)) and result)


def _terminate_process_tree(
    process: Any, process_group_id: int | None,
    known_descendants: dict[int, str] | None = None,
    *, parent_start_identity: str | None = None,
) -> None:
    descendants = dict(known_descendants or {})
    pid = getattr(process, "pid", None)
    if isinstance(pid, int) and pid > 0:
        descendants.update(_descendant_identities(pid))
    parent_is_owned = (
        isinstance(pid, int)
        and (
            (_process_start_time(pid) == parent_start_identity)
            if parent_start_identity is not None else process.poll() is None
        )
    )
    if (
        parent_is_owned and process_group_id is not None
        and process_group_id != os.getpgrp()
    ):
        try:
            os.killpg(process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.terminate()
    _signal_processes(descendants, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except (subprocess.TimeoutExpired, TimeoutError):
        if (
            parent_is_owned and process_group_id is not None
            and process_group_id != os.getpgrp()
        ):
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.kill()
    # A child can leave the parent's process group via setsid().  Kill every
    # descendant observed during startup and again at shutdown, even when the
    # foreground parent has already exited.
    _wait_for_processes(descendants, timeout=0.5)
    _signal_processes(descendants, signal.SIGKILL)
    try:
        process.wait(timeout=2)
    except (subprocess.TimeoutExpired, TimeoutError):
        pass


def _signal_processes(identities: dict[int, str], signum: signal.Signals) -> None:
    for pid in sorted(identities, reverse=True):
        if pid == os.getpid():
            continue
        if _process_start_time(pid) != identities[pid]:
            continue
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass


def _wait_for_processes(identities: dict[int, str], *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(
            _process_start_time(pid) == start_time
            for pid, start_time in identities.items()
        ):
            return
        time.sleep(0.05)


def _descendant_pids(root_pid: int) -> set[int]:
    """Return Linux descendants without adding a process-inspection dependency."""
    descendants: set[int] = set()
    pending = [root_pid]
    while pending:
        parent = pending.pop()
        children_file = Path(f"/proc/{parent}/task/{parent}/children")
        try:
            children = [int(value) for value in children_file.read_text().split()]
        except (FileNotFoundError, PermissionError, OSError, ValueError):
            continue
        for child in children:
            if child not in descendants and child != root_pid:
                descendants.add(child)
                pending.append(child)
    return descendants


def _descendant_identities(root_pid: int) -> dict[int, str]:
    return {
        pid: start_time
        for pid in _descendant_pids(root_pid)
        if (start_time := _process_start_time(pid)) is not None
    }


def _process_start_time(pid: int) -> str | None:
    """Read Linux process start time so a recycled PID is never signalled."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields_after_name = stat[stat.rindex(")") + 2:].split()
        return fields_after_name[19]
    except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError):
        return None
