from pathlib import Path

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
