"""Daemon-owned observability services.

The manager only owns the optional local W&B process.  Run evidence and logs
remain canonical in backend-owned run directories; publication is a separate,
best-effort projection and must never make collection fail.
"""

from __future__ import annotations

import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from .schemas import LocalWandbConfig


@dataclass
class WandbServiceManager:
    config: LocalWandbConfig
    popen: Callable[..., Any] = subprocess.Popen
    healthcheck: Callable[[str], bool] | None = None
    _process: Any = None
    _state: str = "DISABLED"
    _error: str | None = None
    _started_at: float | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def start(self) -> dict[str, Any]:
        with self._lock:
            if not self.config.enabled:
                self._state, self._error = "DISABLED", None
                return self.status()
            if not self.config.managed:
                self._state, self._error = "EXTERNAL", None
                return self.status()
            if self._process is not None and self._process.poll() is None:
                self._state = "READY"
                return self.status()
            self._state, self._error = "STARTING", None
            try:
                self._process = self.popen(
                    list(self.config.command),
                    env=self._environment(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                deadline = time.monotonic() + self.config.startup_timeout_seconds
                while time.monotonic() < deadline:
                    if self._process.poll() is not None:
                        raise RuntimeError("local W&B process exited during startup")
                    try:
                        if self.healthcheck is not None:
                            healthy = self.healthcheck(self.config.url())
                        else:
                            request = urllib.request.Request(self.config.url(), method="GET")
                            with urllib.request.urlopen(request, timeout=0.5):
                                healthy = True
                        if healthy:
                            self._state = "READY"
                            self._started_at = time.time()
                            return self.status()
                    except (OSError, urllib.error.URLError):
                        time.sleep(0.1)
                raise TimeoutError("local W&B startup timed out")
            except Exception as exc:  # optional dependency must be degradable
                self._state, self._error = "DEGRADED", _safe_error(exc)
                self._process = None
                return self.status()

    def stop(self) -> None:
        with self._lock:
            process, self._process = self._process, None
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass
            if self.config.enabled and self._state != "DEGRADED":
                self._state = "STOPPED"

    def status(self) -> dict[str, Any]:
        process = self._process
        if process is not None and process.poll() is not None and self._state == "READY":
            self._state = "DEGRADED"
            self._error = "local W&B process exited"
        return {
            "enabled": self.config.enabled,
            "managed": self.config.managed,
            "state": self._state,
            "url": self.config.url() if self.config.enabled else None,
            "bind_host": self.config.bind_host,
            "port": self.config.port,
            "started_at": self._started_at,
            "error": self._error,
        }

    def _environment(self) -> dict[str, str]:
        # Keep the child isolated from accidental cloud credentials. Cloud
        # credentials are injected only into a future publisher worker.
        import os
        environment = dict(os.environ)
        environment.setdefault("WANDB_DIR", self.config.data_dir)
        environment.pop("WANDB_API_KEY", None)
        return environment


def _safe_error(exc: BaseException) -> str:
    text = str(exc).replace("\n", " ").strip()
    return text[:300] or exc.__class__.__name__
