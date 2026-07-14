from pathlib import Path

import pytest

from ml_exp_server.cli import _require_loopback_host, main
from ml_exp_server.credentials import CredentialStore
from ml_exp_server.project_registry import ProjectRegistry


def _write_config(tmp_path: Path, extra: str = "") -> Path:
    config = tmp_path / "server.yaml"
    config.write_text(
        "schema_version: 1\n"
        f"index_db: {tmp_path / 'index.sqlite'}\n"
        "observability:\n"
        f"  credential_root: {tmp_path / 'credentials'}\n"
        + extra,
        encoding="utf-8",
    )
    return config


def test_doctor_reports_defaults_as_informational_not_failing(tmp_path, capsys):
    config = _write_config(tmp_path)

    assert main(["--config", str(config), "doctor"]) == 0
    out = capsys.readouterr().out
    assert "✓ server config" in out
    assert "· action_runtime.allow_scheduler_mutations: disabled (default)" in out
    assert "· local W&B: disabled" in out
    assert "· W&B cloud: disabled" in out
    assert "· projects: registry not initialized; 0 configured for bootstrap" in out


def test_doctor_flags_missing_wandb_credentials_as_hard_failures(tmp_path, capsys):
    config = _write_config(
        tmp_path,
        "  local_wandb:\n"
        "    enabled: true\n"
        "    managed: false\n"
        "    external_url: http://127.0.0.1:9000\n"
        "    publisher_entity: local-team\n"
        "    publisher_credential_ref: wandb-local-default\n"
        "  wandb_cloud:\n"
        "    enabled: true\n"
        "    entity: cloud-team\n"
        "    default_credential_ref: wandb-cloud-default\n",
    )

    exit_code = main(["--config", str(config), "doctor"])
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "✗ local W&B credential: wandb-local-default not set" in out
    assert "✗ W&B cloud credential: wandb-cloud-default not set" in out
    assert "    → ml-expd --config ... credential wandb set wandb-local-default --stdin" in out


def test_doctor_reports_configured_credentials_as_passing(tmp_path, capsys):
    config = _write_config(
        tmp_path,
        "  local_wandb:\n"
        "    enabled: true\n"
        "    managed: false\n"
        "    external_url: http://127.0.0.1:9000\n"
        "    publisher_entity: local-team\n"
        "    publisher_credential_ref: wandb-local-default\n",
    )
    store = CredentialStore(tmp_path / "credentials")
    store.set_wandb_api_key("wandb-local-default", "secret-value")

    exit_code = main(["--config", str(config), "doctor"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "✓ local W&B credential: wandb-local-default" in out
    assert "secret-value" not in out


def test_doctor_rejects_missing_entities_even_when_credentials_exist(tmp_path, capsys):
    config = _write_config(
        tmp_path,
        "  local_wandb:\n"
        "    enabled: true\n"
        "    managed: false\n"
        "    external_url: http://127.0.0.1:9000\n"
        "    publisher_credential_ref: local-ref\n"
        "  wandb_cloud:\n"
        "    enabled: true\n"
        "    default_credential_ref: cloud-ref\n",
    )
    store = CredentialStore(tmp_path / "credentials")
    store.set_wandb_api_key("local-ref", "local-secret")
    store.set_wandb_api_key("cloud-ref", "cloud-secret")

    assert main(["--config", str(config), "doctor"]) == 1
    out = capsys.readouterr().out
    assert "✗ local W&B entity: publisher_entity not set" in out
    assert "✗ W&B cloud entity: entity not set" in out


def test_doctor_reports_invalid_credential_reference_without_crashing(tmp_path, capsys):
    config = _write_config(
        tmp_path,
        "  local_wandb:\n"
        "    enabled: true\n"
        "    managed: false\n"
        "    external_url: http://127.0.0.1:9000\n"
        "    publisher_entity: local-team\n"
        "    publisher_credential_ref: ../bad\n",
    )

    assert main(["--config", str(config), "doctor"]) == 1
    out = capsys.readouterr().out
    assert "✗ local W&B credential: credential reference must use only" in out


def test_doctor_counts_durable_registry_instead_of_static_config(tmp_path, capsys):
    config = _write_config(tmp_path)
    project_file = tmp_path / "research_project.yaml"
    project_file.write_text(
        "schema_version: 1\nproject: live-project\ntitle: Live\nrun_roots: []\n",
        encoding="utf-8",
    )
    registry = ProjectRegistry(tmp_path / "index.projects")
    registry.bootstrap([])
    registry.register("live-project", project_file)

    assert main(["--config", str(config), "doctor"]) == 0
    out = capsys.readouterr().out
    assert "✓ projects: 1 registered" in out


def test_doctor_rejects_registered_project_that_daemon_cannot_load(tmp_path, capsys):
    config = _write_config(tmp_path)
    registry = ProjectRegistry(tmp_path / "index.projects")
    registry.bootstrap([])
    registry.register("ghost", tmp_path / "missing.yaml")

    assert main(["--config", str(config), "doctor"]) == 1
    assert "✗ projects: ghost: config file not found" in capsys.readouterr().out


def test_doctor_reports_non_utf8_config_without_traceback(tmp_path, capsys):
    config = tmp_path / "server.yaml"
    config.write_bytes(b"\xff\xfe")
    assert main(["--config", str(config), "doctor"]) == 1
    assert "✗ server config" in capsys.readouterr().out


def test_doctor_reports_invalid_config_without_crashing(tmp_path, capsys):
    missing = tmp_path / "missing.yaml"

    exit_code = main(["--config", str(missing), "doctor"])
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "✗ server config" in out


@pytest.mark.parametrize(("host", "normalized"), [
    ("127.0.0.1", "127.0.0.1"),
    ("127.1.2.3", "127.1.2.3"),
    ("::1", "::1"),
    ("[::1]", "::1"),
    (" localhost ", "localhost"),
])
def test_daemon_accepts_only_loopback_bind_hosts(host, normalized):
    assert _require_loopback_host(host) == normalized


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "daemon.internal"])
def test_daemon_rejects_unauthenticated_non_loopback_bind_hosts(host):
    with pytest.raises(SystemExit, match="no HTTP authentication"):
        _require_loopback_host(host)
