"""Remaining CLI and doctor policy branches."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_exp_server import cli
from ml_exp_server.cli import _validate_bind_host, _validate_tls_cert_chain, main
from ml_exp_server.project_registry import ProjectRegistry
from ml_exp_server.schemas import ProjectLifecycleState


def write_config(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "server.yml"
    path.write_text(
        "schema_version: 1\n"
        f"index_db: {tmp_path / 'index.sqlite'}\n"
        f"action_root: {tmp_path / 'actions'}\n"
        f"project_registry_root: {tmp_path / 'registry'}\n"
        "observability:\n"
        f"  credential_root: {tmp_path / 'credentials'}\n"
        + extra,
    )
    return path


@pytest.mark.parametrize(("host", "authenticated", "tls", "expected"), [
    (" ", False, False, None),
    ("localhost", False, False, "localhost"),
    ("shared.example", False, True, None),
    ("shared.example", True, True, "shared.example"),
])
def test_validate_bind_host_remaining_policy_paths(
    host, authenticated, tls, expected,
):
    if expected is None:
        with pytest.raises(SystemExit):
            _validate_bind_host(host, authenticated=authenticated, tls=tls)
    else:
        assert _validate_bind_host(
            host, authenticated=authenticated, tls=tls,
        ) == expected


def test_tls_pair_validation_requires_both_and_returns_resolved_paths(
    monkeypatch, tmp_path,
):
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    with pytest.raises(SystemExit, match="provided together"):
        _validate_tls_cert_chain(cert, None)

    loaded = {}

    class Context:
        def __init__(self, protocol):
            loaded["protocol"] = protocol

        def load_cert_chain(self, *, certfile, keyfile):
            loaded.update(certfile=certfile, keyfile=keyfile)

    monkeypatch.setattr(cli.ssl, "SSLContext", Context)
    assert _validate_tls_cert_chain(cert, key) == (cert.resolve(), key.resolve())
    assert loaded["certfile"] == str(cert.resolve())


def test_credential_cli_clear_status_prompt_and_error(
    monkeypatch, tmp_path, capsys,
):
    config = write_config(tmp_path)
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: "secret")
    prefix = ["--config", str(config), "credential", "wandb"]
    assert main([*prefix, "set", "cloud"]) == 0
    assert main([*prefix, "status", "cloud"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["configured"] is True
    assert main([*prefix, "clear", "cloud"]) == 0
    assert main([*prefix, "status", "../invalid"]) == 2
    assert "credential error" in capsys.readouterr().err


def test_doctor_covers_enabled_actions_managed_local_and_missing_reference(
    tmp_path, capsys,
):
    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\n")
    docker.chmod(0o700)
    config = write_config(
        tmp_path,
        "  local_wandb:\n"
        "    enabled: true\n"
        "    managed: true\n"
        f"    docker_executable: {docker}\n"
        "    publisher_entity: local-team\n"
        "  wandb_cloud:\n"
        "    enabled: true\n"
        "    entity: team\n"
        "    default_credential_ref: cloud\n"
        "action_runtime:\n"
        "  allow_project_writes: true\n"
        "  allow_scheduler_mutations: true\n"
        "  allow_observability_mutations: true\n",
    )
    assert main(["--config", str(config), "doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    checks = {item["name"]: item for item in report["checks"]}
    assert checks["action_runtime.allow_project_writes"]["status"] == "PASS"
    assert checks["local W&B docker"]["status"] == "PASS"
    assert checks["local W&B credential"]["status"] == "FAIL"
    assert checks["W&B cloud credential"]["status"] == "FAIL"


def test_doctor_reports_nonexecutable_docker_inactive_record_and_identity_drift(
    tmp_path, capsys,
):
    docker = tmp_path / "docker"
    docker.write_text("not executable")
    project = tmp_path / "project.yml"
    project.write_text(
        "schema_version: 1\nproject: demo\ntitle: Demo\nrun_roots: []\n",
    )
    config = write_config(
        tmp_path,
        "  local_wandb:\n"
        "    enabled: true\n"
        "    managed: true\n"
        f"    docker_executable: {docker}\n",
    )
    registry = ProjectRegistry(tmp_path / "registry")
    registry.bootstrap([project])
    registry.transition("demo", ProjectLifecycleState.PAUSED)
    assert main(["--config", str(config), "doctor", "--json"]) == 1
    checks = json.loads(capsys.readouterr().out)["checks"]
    assert next(item for item in checks if item["name"] == "local W&B docker")[
        "status"
    ] == "FAIL"

    registry.transition("demo", ProjectLifecycleState.ACTIVE)
    project.write_text(
        "schema_version: 1\nproject: changed\ntitle: Changed\nrun_roots: []\n",
    )
    assert main(["--config", str(config), "doctor", "--json"]) == 1
    assert "identity drift" in capsys.readouterr().out


def test_doctor_maps_corrupt_registry(tmp_path, capsys):
    config = write_config(tmp_path)
    root = tmp_path / "registry"
    root.mkdir()
    (root / "registry.json").write_text("{broken")
    assert main(["--config", str(config), "doctor", "--json"]) == 1
    assert "invalid project registry" in capsys.readouterr().out


def test_main_serves_snapshot_with_tls_arguments(monkeypatch, tmp_path):
    config = write_config(tmp_path)
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    captured = {}
    monkeypatch.setattr(cli, "_validate_tls_cert_chain", lambda *_args: (cert, key))
    monkeypatch.setattr(cli, "_validate_bind_host", lambda *_args, **_kwargs: "host")
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: captured.update(kwargs))
    assert main([
        "--config", str(config), "--snapshot", "--ssl-certfile", str(cert),
        "--ssl-keyfile", str(key),
    ]) == 0
    assert captured["host"] == "host"
    assert captured["ssl_certfile"] == str(cert)
    assert captured["ssl_keyfile"] == str(key)
