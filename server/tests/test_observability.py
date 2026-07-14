import os
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import ml_exp_server.observability as observability
from ml_exp_server.api.app import create_app
from ml_exp_server.observability import WandbServiceManager
from ml_exp_server.schemas import LocalWandbConfig, ObservabilityConfig, ServerConfig


COMMAND = [
    "fake-wandb", "--bind", "{bind_host}:{port}", "--data", "{data_dir}",
]


class _Process:
    def __init__(self, *, pid: int = 424242):
        self.pid = pid
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def _managed(tmp_path: Path, **overrides) -> LocalWandbConfig:
    values = {
        "enabled": True,
        "command": COMMAND,
        "data_dir": str(tmp_path / "wandb"),
    }
    values.update(overrides)
    return LocalWandbConfig(**values)


def test_local_wandb_manager_uses_minimal_environment_and_resolved_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    process = _Process()
    seen = {}
    stopped = []
    monkeypatch.setenv("WANDB_API_KEY", "secret-cloud-key")
    monkeypatch.setenv("HTTPS_PROXY", "https://token@proxy.invalid")
    monkeypatch.setenv("PATH", "/safe/bin")

    def fake_popen(command, **kwargs):
        seen["command"] = command
        seen["env"] = kwargs["env"]
        seen["new_session"] = kwargs["start_new_session"]
        return process

    manager = WandbServiceManager(
        _managed(tmp_path), popen=fake_popen,
        port_probe=lambda host, port, timeout: True,
        healthcheck=lambda url: True,
        terminator=lambda owned, pgid: stopped.append((owned, pgid)),
    )
    status = manager.start()

    data_dir = (tmp_path / "wandb").resolve()
    assert status["state"] == "READY"
    assert status["url"] == "http://127.0.0.1:8080"
    assert seen["command"] == [
        "fake-wandb", "--bind", "127.0.0.1:8080", "--data", str(data_dir),
    ]
    assert seen["env"] == {"WANDB_DIR": str(data_dir)}
    assert seen["new_session"] is True
    assert data_dir.is_dir()
    assert data_dir.stat().st_mode & 0o777 == 0o700

    manager.stop()
    assert stopped == [(process, process.pid)]


def test_environment_inheritance_is_an_explicit_non_secret_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("PATH", "/safe/bin")
    process = _Process()
    seen = {}
    manager = WandbServiceManager(
        _managed(tmp_path, environment_allowlist=["PATH"]),
        popen=lambda command, **kwargs: seen.setdefault("env", kwargs["env"]) or process,
        port_probe=lambda host, port, timeout: True,
        healthcheck=lambda url: True,
    )
    # Avoid relying on the compact lambda's truthiness as a fake Popen result.
    def popen(command, **kwargs):
        seen["env"] = kwargs["env"]
        return process
    manager.popen = popen
    manager.terminator = lambda process, pgid: None
    manager.start()
    assert seen["env"]["PATH"] == "/safe/bin"
    assert set(seen["env"]) == {"PATH", "WANDB_DIR"}

    with pytest.raises(ValidationError, match="unsupported names"):
        _managed(tmp_path, environment_allowlist=["WANDB_API_KEY"])


@pytest.mark.parametrize("url", [
    "ftp://metrics.example.test",
    "https://user:password@metrics.example.test",
    "https://metrics.example.test/health?api_key=secret",
    "https://metrics.example.test/health#secret",
    "http://metrics.example.test",
])
def test_external_url_rejects_credential_bearing_or_non_http_values(url: str):
    with pytest.raises(ValidationError):
        LocalWandbConfig(enabled=True, managed=False, external_url=url)


def test_external_url_status_redacts_health_path(tmp_path: Path):
    manager = WandbServiceManager(
        LocalWandbConfig(
            enabled=True, managed=False,
            external_url="https://metrics.example.test:9443/private-health-path",
            data_dir=str(tmp_path / "unused"),
        ),
        port_probe=lambda host, port, timeout: host == "metrics.example.test" and port == 9443,
        healthcheck=lambda url: url.endswith("/private-health-path"),
    )
    status = manager.start()
    assert status["state"] == "READY"
    assert status["url"] == "https://metrics.example.test:9443"
    assert "private-health-path" not in repr(status)


def test_external_service_ready_state_is_rechecked(tmp_path: Path):
    healthy = True
    manager = WandbServiceManager(
        LocalWandbConfig(
            enabled=True, managed=False,
            external_url="http://127.0.0.1:8080",
            data_dir=str(tmp_path / "unused"),
        ),
        port_probe=lambda host, port, timeout: healthy,
        healthcheck=lambda url: healthy,
    )
    assert manager.start()["state"] == "READY"
    healthy = False
    status = manager.status()
    assert status["state"] == "DEGRADED"
    assert status["error"] == "local W&B readiness check failed"


def test_managed_service_requires_loopback_and_consistent_command(tmp_path: Path):
    assert "ml_exp_server.local_wandb_service" in LocalWandbConfig(
        enabled=True,
    ).command
    with pytest.raises(ValidationError, match="loopback"):
        _managed(tmp_path, bind_host="0.0.0.0")
    with pytest.raises(ValidationError, match="missing placeholders"):
        _managed(tmp_path, command=["wandb", "server", "start", "--no-daemon"])


def test_startup_timeout_is_bounded_and_cleans_owned_process(tmp_path: Path):
    process = _Process()
    stopped = []
    manager = WandbServiceManager(
        _managed(tmp_path, startup_timeout_seconds=0.05),
        popen=lambda command, **kwargs: process,
        port_probe=lambda host, port, timeout: True,
        # A buggy probe cannot make startup_timeout ineffective.
        healthcheck=lambda url: (time.sleep(10), True)[1],
        terminator=lambda owned, pgid: stopped.append((owned, pgid)),
    )
    started = time.monotonic()
    status = manager.start()
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert status["state"] == "DEGRADED"
    assert status["error"] == "local W&B readiness timed out"
    assert stopped == [(process, process.pid)]


def test_readiness_requires_both_port_and_http_health(tmp_path: Path):
    for port_ready, http_ready in [(False, True), (True, False)]:
        process = _Process()
        stopped = []
        manager = WandbServiceManager(
            _managed(tmp_path / f"{port_ready}-{http_ready}", startup_timeout_seconds=0.02),
            popen=lambda command, **kwargs: process,
            port_probe=lambda host, port, timeout, value=port_ready: value,
            healthcheck=lambda url, value=http_ready: value,
            terminator=lambda owned, pgid: stopped.append(owned),
        )
        assert manager.start()["state"] == "DEGRADED"
        assert stopped == [process]


def test_errors_never_echo_command_url_or_secret(tmp_path: Path):
    secret = "super-secret-value"

    def missing(*args, **kwargs):
        raise FileNotFoundError(f"/private/{secret}/fake-wandb")

    manager = WandbServiceManager(
        _managed(tmp_path, command=[
            f"/private/{secret}/fake-wandb", "{bind_host}", "{port}", "{data_dir}",
        ]),
        popen=missing,
    )
    status = manager.start()
    assert status["state"] == "DEGRADED"
    assert status["error"] == "local W&B executable was not found"
    assert secret not in repr(status)


def test_process_group_and_descendant_cleanup_escalates_to_kill(
    monkeypatch: pytest.MonkeyPatch,
):
    process = _Process(pid=98765)
    waits = 0
    signals = []

    def wait(timeout=None):
        nonlocal waits
        waits += 1
        if waits == 1:
            raise subprocess.TimeoutExpired("secret-command", timeout)
        process.returncode = -9
        return process.returncode

    process.wait = wait
    monkeypatch.setattr(observability.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)))
    monkeypatch.setattr(observability.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(observability.os, "getpgrp", lambda: 123)
    monkeypatch.setattr(
        observability, "_descendant_identities", lambda pid: {111: "a", 222: "b"},
    )
    monkeypatch.setattr(
        observability, "_process_start_time",
        lambda pid: {111: "a", 222: "b"}.get(pid),
    )
    observability._terminate_process_tree(process, process.pid)
    assert signals == [
        (process.pid, observability.signal.SIGTERM),
        (222, observability.signal.SIGTERM),
        (111, observability.signal.SIGTERM),
        (process.pid, observability.signal.SIGKILL),
        (222, observability.signal.SIGKILL),
        (111, observability.signal.SIGKILL),
    ]


def test_process_tree_cleanup_never_signals_a_recycled_descendant_pid(
    monkeypatch: pytest.MonkeyPatch,
):
    process = _Process(pid=98765)
    process.returncode = 0
    signals = []
    monkeypatch.setattr(observability.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)))
    monkeypatch.setattr(observability.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(observability, "_descendant_identities", lambda pid: {})
    monkeypatch.setattr(observability, "_process_start_time", lambda pid: "new-process")

    observability._terminate_process_tree(
        process, process.pid, {333: "old-process"},
        parent_start_identity="old-parent",
    )

    assert signals == []


def test_observability_config_is_optional(tmp_path: Path):
    config = ServerConfig(index_db=str(tmp_path / "index.sqlite"))
    assert config.observability.local_wandb.enabled is False
    assert config.observability.local_wandb.command == []
    assert config.observability.log_archive_root


class _SpyService:
    def __init__(self, owner_check=lambda: True):
        self.starts = 0
        self.stops = 0
        self.owner_check = owner_check
        self.state = "STANDBY"

    def start(self):
        assert self.owner_check()
        self.starts += 1
        self.state = "READY"
        return self.status()

    def stop(self):
        self.stops += 1
        self.state = "STOPPED"

    def status(self):
        return {
            "enabled": True, "managed": True, "state": self.state,
            "url": "http://127.0.0.1:8080", "started_at": None, "error": None,
        }


def _server_config(tmp_path: Path, *, collector_enabled: bool = True) -> ServerConfig:
    return ServerConfig(
        index_db=str(tmp_path / "index.sqlite"),
        action_root=str(tmp_path / "actions"),
        collector_enabled=collector_enabled,
        projects=[],
        observability=ObservabilityConfig(
            local_wandb=_managed(tmp_path),
            credential_root=str(tmp_path / "credentials"),
            log_archive_root=str(tmp_path / "logs"),
        ),
    )


def test_only_collector_lease_owner_starts_and_stops_managed_service(tmp_path: Path):
    config = _server_config(tmp_path)
    first_app = create_app(config)
    second_app = create_app(config)
    first = _SpyService(lambda: first_app.state.collector_owner is True)
    second = _SpyService(lambda: second_app.state.collector_owner is True)
    first_app.state.runtime.wandb_service = first
    second_app.state.runtime.wandb_service = second

    with TestClient(first_app) as first_client, TestClient(second_app) as second_client:
        assert first.starts == 1
        assert second.starts == 0
        assert first_client.get("/api/health").json()["observability"]["state"] == "READY"
        assert second_client.get("/api/health").json()["observability"]["state"] == "STANDBY"

    assert first.stops == 1
    # Non-owners still close their runtime, but never owned a process.
    assert second.stops == 1


def test_snapshot_mode_never_starts_managed_service(tmp_path: Path):
    app = create_app(_server_config(tmp_path, collector_enabled=False))
    service = _SpyService()
    app.state.runtime.wandb_service = service
    with TestClient(app) as client:
        assert client.get("/api/health").json()["collector_requested"] is False
        assert service.starts == 0
    assert service.stops == 1
