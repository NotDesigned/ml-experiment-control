"""Process ownership and readiness failure edges for local observability."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_exp_server import observability
from ml_exp_server.observability import WandbServiceManager
from ml_exp_server.schemas import LocalWandbConfig


class Process:
    def __init__(self, pid=4242, returncode=None):
        self.pid = pid
        self.returncode = returncode
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


def managed(tmp_path, **updates):
    values = {
        "enabled": True,
        "data_dir": str(tmp_path / "wandb"),
        "command": ["wandb", "{bind_host}", "{port}", "{data_dir}"],
        "startup_timeout_seconds": 0.02,
    }
    values.update(updates)
    return LocalWandbConfig(**values)


def external(tmp_path):
    return LocalWandbConfig(
        enabled=True, managed=False, external_url="http://127.0.0.1:8080",
        data_dir=str(tmp_path / "unused"),
    )


def test_start_is_idempotent_for_running_managed_and_ready_external(tmp_path):
    process = Process()
    managed_service = WandbServiceManager(
        managed(tmp_path), popen=lambda *_args, **_kwargs: process,
        port_probe=lambda *_args: True, healthcheck=lambda *_args: True,
        terminator=lambda *_args: None,
    )
    assert managed_service.start()["state"] == "READY"
    assert managed_service.start()["state"] == "READY"

    external_service = WandbServiceManager(
        external(tmp_path), port_probe=lambda *_args: True,
        healthcheck=lambda *_args: True,
    )
    assert external_service.start()["state"] == "READY"
    assert external_service.start()["state"] == "READY"


def test_managed_start_ignores_chmod_failure_and_uses_default_probes(
    monkeypatch, tmp_path,
):
    process = Process()
    monkeypatch.setattr(
        observability.os, "chmod",
        lambda *_args: (_ for _ in ()).throw(OSError("unsupported")),
    )
    monkeypatch.setattr(observability, "_port_ready", lambda *_args: True)
    monkeypatch.setattr(observability, "_http_ready", lambda *_args: True)
    service = WandbServiceManager(
        managed(tmp_path), popen=lambda *_args, **_kwargs: process,
        terminator=lambda *_args: None,
    )
    assert service.start()["state"] == "READY"


def test_start_detects_early_process_exit_and_readiness_exception(tmp_path):
    exited = WandbServiceManager(
        managed(tmp_path / "exit"),
        popen=lambda *_args, **_kwargs: Process(returncode=3),
        terminator=lambda *_args: None,
    )
    assert exited.start()["error"] == "local W&B process exited during startup"

    service = WandbServiceManager(
        managed(tmp_path / "probe"), popen=lambda *_args, **_kwargs: Process(),
        port_probe=lambda *_args: (_ for _ in ()).throw(OSError("probe")),
        healthcheck=lambda *_args: True,
        terminator=lambda *_args: None,
    )
    assert service.start()["state"] == "DEGRADED"


def test_status_degrades_exited_managed_and_exceptional_external(tmp_path):
    process = Process()
    managed_service = WandbServiceManager(
        managed(tmp_path), popen=lambda *_args, **_kwargs: process,
        port_probe=lambda *_args: True, healthcheck=lambda *_args: True,
        terminator=lambda *_args: None,
    )
    managed_service.start()
    process.returncode = 1
    assert managed_service.status()["error"] == "local W&B process exited"

    external_service = WandbServiceManager(
        external(tmp_path),
        port_probe=lambda *_args: (_ for _ in ()).throw(ValueError("bad")),
        healthcheck=lambda *_args: True,
    )
    external_service._state = "READY"
    assert external_service.status()["state"] == "DEGRADED"


def test_readiness_address_failures_are_degradable(monkeypatch, tmp_path):
    managed_service = WandbServiceManager(
        managed(tmp_path), popen=lambda *_args, **_kwargs: Process(),
        terminator=lambda *_args: None,
    )
    monkeypatch.setattr(
        managed_service, "_readiness_address",
        lambda: (_ for _ in ()).throw(ValueError("invalid URL")),
    )
    assert managed_service.start()["state"] == "DEGRADED"

    external_service = WandbServiceManager(external(tmp_path))
    monkeypatch.setattr(
        external_service, "_readiness_address",
        lambda: (_ for _ in ()).throw(ValueError("invalid URL")),
    )
    assert external_service._external_ready() is False


def test_terminate_uses_default_tree_and_suppresses_cleanup_failure(
    monkeypatch, tmp_path,
):
    calls = []
    process = Process()
    service = WandbServiceManager(managed(tmp_path))
    service._process = process
    service._process_group_id = process.pid
    monkeypatch.setattr(
        observability, "_terminate_process_tree",
        lambda *args, **kwargs: calls.append(args) or (_ for _ in ()).throw(
            RuntimeError("cleanup")
        ),
    )
    service._terminate_process()
    assert calls and service._process is None


@pytest.mark.parametrize(("error", "message"), [
    (OSError("disk"), "process could not be started"),
    (RuntimeError("exit"), "process exited during startup"),
    (Exception("other"), "startup failed"),
])
def test_safe_error_classification(error, message):
    assert message in observability._safe_error(error)


def test_readiness_helpers_cover_ipv6_ports_http_and_exceptions(monkeypatch):
    assert observability._redacted_origin("https://[::1]:9443/private") == (
        "https://[::1]:9443"
    )
    assert observability._redacted_origin("https://example.com/private") == (
        "https://example.com"
    )

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        observability.socket, "create_connection", lambda *_args, **_kwargs: Connection(),
    )
    assert observability._port_ready("localhost", 1, 0.1) is True
    monkeypatch.setattr(
        observability.socket, "create_connection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("closed")),
    )
    assert observability._port_ready("localhost", 1, 0.1) is False

    monkeypatch.setattr(
        observability.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: SimpleNamespace(
            status=204, __enter__=lambda self: self,
            __exit__=lambda *_args: False,
        ),
    )
    # SimpleNamespace special methods are not used for protocol lookup; use a
    # concrete context manager for the HTTP response.
    class Response(Connection):
        status = 204

    monkeypatch.setattr(
        observability.urllib.request, "urlopen", lambda *_args, **_kwargs: Response(),
    )
    assert observability._http_ready("http://localhost", 0.1) is True
    assert observability._bounded_bool_call(
        lambda: (_ for _ in ()).throw(RuntimeError("probe")), 0.1,
    ) is False
    assert observability._bounded_bool_call(
        lambda: (time.sleep(0.02) or True), 0.001,
    ) is False


def test_process_tree_falls_back_to_parent_and_kill(monkeypatch):
    process = Process(pid=99)
    monkeypatch.setattr(observability, "_descendant_identities", lambda _pid: {})
    monkeypatch.setattr(observability, "_wait_for_processes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(observability, "_signal_processes", lambda *_args: None)
    observability._terminate_process_tree(process, None)
    assert process.terminated is True

    process = Process(pid=100)
    waits = 0

    def timeout_then_finish(timeout=None):
        nonlocal waits
        waits += 1
        if waits == 1:
            raise subprocess.TimeoutExpired("process", timeout)
        return process.returncode

    process.wait = timeout_then_finish
    monkeypatch.setattr(process, "poll", lambda: None)
    observability._terminate_process_tree(process, None)
    assert process.killed is True

    exited = Process(pid=None, returncode=1)
    exited.wait = lambda timeout=None: (_ for _ in ()).throw(
        subprocess.TimeoutExpired("process", timeout)
    )
    observability._terminate_process_tree(exited, None)


def test_process_group_lookup_errors_and_final_wait_timeout(monkeypatch):
    process = Process(pid=101)
    process.wait = lambda timeout=None: (_ for _ in ()).throw(
        subprocess.TimeoutExpired("process", timeout)
    )
    monkeypatch.setattr(observability, "_descendant_identities", lambda _pid: {})
    monkeypatch.setattr(observability, "_wait_for_processes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(observability, "_signal_processes", lambda *_args: None)
    monkeypatch.setattr(observability.os, "getpgrp", lambda: 1)
    monkeypatch.setattr(
        observability.os, "killpg",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError()),
    )
    observability._terminate_process_tree(process, process.pid)


def test_signal_processes_skips_self_recycled_and_missing(monkeypatch):
    sent = []
    self_pid = os.getpid()
    monkeypatch.setattr(
        observability, "_process_start_time",
        lambda pid: {2: "new", 3: "same"}.get(pid),
    )

    def kill(pid, signum):
        sent.append((pid, signum))
        raise ProcessLookupError()

    monkeypatch.setattr(observability.os, "kill", kill)
    observability._signal_processes(
        {self_pid: "self", 2: "old", 3: "same"}, observability.signal.SIGTERM,
    )
    assert sent[0][0] == 3


def test_descendant_scan_ignores_root_and_duplicate_children(monkeypatch):
    def children(path, *_args, **_kwargs):
        value = str(path)
        if "/1/task/1/" in value:
            return "1 2"
        if "/2/task/2/" in value:
            return "2"
        return ""

    monkeypatch.setattr(Path, "read_text", children)
    assert observability._descendant_pids(1) == {2}


def test_descendant_and_process_start_time_proc_parsing(monkeypatch):
    def children(path: Path):
        value = str(path)
        return "2" if "/proc/1/" in value else ""

    monkeypatch.setattr(Path, "read_text", children)
    assert observability._descendant_pids(1) == {2}

    stat_line = "1 (cmd) S " + " ".join(str(index) for index in range(1, 25))
    monkeypatch.setattr(Path, "read_text", lambda _path: stat_line)
    assert observability._process_start_time(1) == "19"
    monkeypatch.setattr(Path, "read_text", lambda _path: "malformed")
    assert observability._process_start_time(1) is None
