from pathlib import Path

from ml_exp_server.observability import WandbServiceManager
from ml_exp_server.schemas import LocalWandbConfig, ObservabilityConfig, ServerConfig


class _Process:
    def __init__(self):
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode


def test_local_wandb_manager_is_degradable_and_redacts_environment():
    process = _Process()
    seen = {}

    def fake_popen(command, **kwargs):
        seen["command"] = command
        seen["env"] = kwargs["env"]
        return process

    manager = WandbServiceManager(
        LocalWandbConfig(enabled=True, command=["fake-wandb"]), popen=fake_popen,
        healthcheck=lambda url: True,
    )
    status = manager.start()
    assert status["state"] == "READY"
    assert status["url"] == "http://127.0.0.1:8080"
    assert "WANDB_API_KEY" not in seen["env"]
    manager.stop()
    assert process.terminated


def test_missing_local_wandb_is_degraded():
    def missing(*args, **kwargs):
        raise FileNotFoundError("wandb")

    manager = WandbServiceManager(
        LocalWandbConfig(enabled=True, startup_timeout_seconds=0.1), popen=missing,
    )
    status = manager.start()
    assert status["state"] == "DEGRADED"
    assert "wandb" in status["error"]


def test_observability_config_is_optional(tmp_path: Path):
    config = ServerConfig(index_db=str(tmp_path / "index.sqlite"))
    assert config.observability.local_wandb.enabled is False
    assert config.observability.log_archive_root
