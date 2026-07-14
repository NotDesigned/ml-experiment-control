from pathlib import Path
import json

import pytest

from ml_exp_server.cli import _require_loopback_host, _validate_bind_host, main
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


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "daemon.internal"])
def test_daemon_allows_authenticated_non_loopback_bind_hosts(host):
    assert _validate_bind_host(host, authenticated=True, tls=True)


def test_daemon_rejects_bearer_remote_bind_without_tls():
    with pytest.raises(SystemExit, match="without --ssl-certfile"):
        _validate_bind_host("0.0.0.0", authenticated=True)


def test_doctor_validates_private_http_bearer_token(tmp_path, capsys):
    token = tmp_path / "daemon.token"
    token.write_text("x" * 40 + "\n", encoding="utf-8")
    token.chmod(0o600)
    config = _write_config(
        tmp_path,
        f"http_auth:\n  bearer_token_file: {token}\n",
    )

    assert main(["--config", str(config), "doctor"]) == 0
    assert "✓ HTTP authentication: bearer token ready" in capsys.readouterr().out

    token.chmod(0o644)
    assert main(["--config", str(config), "doctor"]) == 1
    assert "permissions must be 0600" in capsys.readouterr().out


def test_daemon_doctor_json_report(tmp_path, capsys):
    config = _write_config(tmp_path)
    assert main(["--config", str(config), "doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"
    assert any(item["name"] == "server config" for item in payload["checks"])


def test_daemon_doctor_rejects_remote_plaintext_bind(tmp_path, capsys):
    token = tmp_path / "daemon.token"
    token.write_text("x" * 40, encoding="utf-8")
    token.chmod(0o600)
    config = _write_config(
        tmp_path, f"http_auth:\n  bearer_token_file: {token}\n",
    )
    assert main([
        "--config", str(config), "--host", "0.0.0.0", "doctor",
    ]) == 1
    assert "✗ HTTP bind policy" in capsys.readouterr().out


def test_tls_preflight_fails_before_runtime_initialization(tmp_path, capsys):
    token = tmp_path / "daemon.token"
    token.write_text("x" * 40, encoding="utf-8")
    token.chmod(0o600)
    action_root = tmp_path / "actions"
    config = _write_config(
        tmp_path,
        f"action_root: {action_root}\n"
        f"http_auth:\n  bearer_token_file: {token}\n",
    )
    arguments = [
        "--config", str(config), "--host", "0.0.0.0",
        "--ssl-certfile", str(tmp_path / "missing.crt"),
        "--ssl-keyfile", str(tmp_path / "missing.key"),
    ]

    assert main([*arguments, "doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    tls = next(item for item in report["checks"] if item["name"] == "TLS certificate")
    assert tls["status"] == "FAIL"
    assert not (tmp_path / "index.sqlite").exists()
    assert not action_root.exists()

    with pytest.raises(SystemExit, match="TLS certificate/key cannot be loaded"):
        main(arguments)
    assert not (tmp_path / "index.sqlite").exists()
    assert not action_root.exists()
