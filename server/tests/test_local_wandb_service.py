from pathlib import Path
from types import SimpleNamespace

import subprocess

from ml_exp_server import local_wandb_service
from ml_exp_server.local_wandb_service import container_name, docker_command
from ml_exp_server.schemas import LocalWandbConfig


def test_managed_config_uses_attached_builtin_wrapper(tmp_path: Path):
    config = LocalWandbConfig(
        enabled=True, data_dir=str(tmp_path / "wandb"), publisher_entity="team",
    )
    command = config.resolved_command()
    assert command[0]
    assert command[1:3] == ["-m", "ml_exp_server.local_wandb_service"]
    assert "127.0.0.1" in command
    assert "8080" in command
    assert str((tmp_path / "wandb").resolve()) in command
    assert "wandb/local" in command


def test_docker_command_binds_loopback_and_persistent_host_data(tmp_path: Path):
    data = tmp_path / "data"
    command = docker_command(
        "/usr/bin/docker", bind_host="127.0.0.1", port=8080,
        data_dir=data, image="wandb/local@sha256:" + "a" * 64,
    )
    assert command[:3] == ["/usr/bin/docker", "run", "--rm"]
    assert "127.0.0.1:8080:8080" in command
    assert f"{data.resolve() / 'server'}:/vol" in command
    assert container_name(data, 8080) in command


def _arguments(tmp_path: Path) -> list[str]:
    return [
        "--bind-host", "127.0.0.1", "--port", "8080",
        "--data-dir", str(tmp_path / "wandb"),
        "--image", "wandb/local@sha256:" + "a" * 64,
        "--docker", "/fake/docker",
    ]


def test_main_fails_closed_when_volume_preparation_fails(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        local_wandb_service.subprocess, "run",
        lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=9),
    )
    monkeypatch.setattr(
        local_wandb_service.subprocess, "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not start")),
    )

    assert local_wandb_service.main(_arguments(tmp_path)) == 2
    assert calls[0][1:4] == ["run", "--rm", "--user"]
    assert not (tmp_path / "wandb" / ".server-volume-initialized").exists()


def test_main_reuses_prepared_volume_and_removes_stale_container(
    monkeypatch, tmp_path,
):
    data = tmp_path / "wandb"
    data.mkdir()
    marker = data / ".server-volume-initialized"
    marker.write_text("wandb uid=999\n", encoding="utf-8")
    runs = []
    monkeypatch.setattr(
        local_wandb_service.subprocess, "run",
        lambda command, **kwargs: runs.append(command) or SimpleNamespace(returncode=0),
    )

    class Process:
        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    starts = []
    monkeypatch.setattr(
        local_wandb_service.subprocess, "Popen",
        lambda command, **kwargs: starts.append(command) or Process(),
    )
    monkeypatch.setattr(local_wandb_service.signal, "signal", lambda *args: None)

    assert local_wandb_service.main(_arguments(tmp_path)) == 0
    assert len(runs) == 1
    assert runs[0][:3] == ["/fake/docker", "rm", "--force"]
    assert starts[0][:3] == ["/fake/docker", "run", "--rm"]


def test_main_forwards_explicit_docker_host(monkeypatch, tmp_path):
    data = tmp_path / "wandb"
    data.mkdir()
    (data / ".server-volume-initialized").write_text("wandb uid=999\n")
    environments = []
    monkeypatch.setenv("DOCKER_HOST", "unix:///run/user/docker.sock")
    monkeypatch.setattr(
        local_wandb_service.subprocess,
        "run",
        lambda _command, **kwargs: (
            environments.append(kwargs["env"]) or SimpleNamespace(returncode=0)
        ),
    )

    class Process:
        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(
        local_wandb_service.subprocess,
        "Popen",
        lambda _command, **kwargs: (
            environments.append(kwargs["env"]) or Process()
        ),
    )
    monkeypatch.setattr(local_wandb_service.signal, "signal", lambda *_args: None)
    assert local_wandb_service.main(_arguments(tmp_path)) == 0
    assert all(
        item["DOCKER_HOST"] == "unix:///run/user/docker.sock"
        for item in environments
    )


def test_main_sigterm_stops_exact_container_once(monkeypatch, tmp_path):
    calls = []
    handlers = {}
    monkeypatch.setattr(
        local_wandb_service.subprocess, "run",
        lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        local_wandb_service.signal, "signal",
        lambda signum, handler: handlers.setdefault(signum, handler),
    )

    class Process:
        returncode = None

        def wait(self, timeout=None):
            if timeout is None:
                handlers[local_wandb_service.signal.SIGTERM](
                    local_wandb_service.signal.SIGTERM, None,
                )
                handlers[local_wandb_service.signal.SIGTERM](
                    local_wandb_service.signal.SIGTERM, None,
                )
                self.returncode = 143
            return self.returncode

        def poll(self):
            return self.returncode

    monkeypatch.setattr(
        local_wandb_service.subprocess, "Popen", lambda *args, **kwargs: Process(),
    )

    assert local_wandb_service.main(_arguments(tmp_path)) == 143
    stops = [command for command in calls if command[1:3] == ["stop", "--time"]]
    assert len(stops) == 1
    assert stops[0][-1] == container_name(tmp_path / "wandb", 8080)


def test_main_kills_child_when_graceful_stop_times_out(monkeypatch, tmp_path):
    monkeypatch.setattr(
        local_wandb_service.subprocess, "run",
        lambda command, **kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(local_wandb_service.signal, "signal", lambda *args: None)

    class Process:
        killed = False

        def wait(self, timeout=None):
            if timeout is None:
                return 0
            raise subprocess.TimeoutExpired("docker", timeout)

        def poll(self):
            return None

        def kill(self):
            self.killed = True

    process = Process()
    monkeypatch.setattr(
        local_wandb_service.subprocess, "Popen", lambda *args, **kwargs: process,
    )

    assert local_wandb_service.main(_arguments(tmp_path)) == 0
    assert process.killed is True
