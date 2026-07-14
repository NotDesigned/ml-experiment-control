from pathlib import Path

import pytest

from ml_exp_server.cli import main
from ml_exp_server.credentials import CredentialError, CredentialStore


def test_credential_store_round_trip_and_permissions(tmp_path: Path):
    store = CredentialStore(tmp_path / "credentials")
    store.set_wandb_api_key("cloud-primary", "secret-value\n")

    assert store.status("cloud-primary").configured is True
    assert store.resolve_wandb_api_key("cloud-primary") == "secret-value"
    assert (store.root.stat().st_mode & 0o777) == 0o700
    path = store.root / "cloud-primary.wandb-api-key"
    assert (path.stat().st_mode & 0o777) == 0o600
    assert store.clear_wandb_api_key("cloud-primary") is True
    assert store.clear_wandb_api_key("cloud-primary") is False


@pytest.mark.parametrize("reference", ["../escape", "with space", "", "/absolute"])
def test_credential_reference_cannot_escape_root(tmp_path: Path, reference: str):
    with pytest.raises(CredentialError):
        CredentialStore(tmp_path).set_wandb_api_key(reference, "secret")


def test_credential_cli_reads_stdin_without_exposing_secret(tmp_path, monkeypatch, capsys):
    config = tmp_path / "server.yaml"
    config.write_text(
        "schema_version: 1\n"
        f"index_db: {tmp_path / 'index.sqlite'}\n"
        "observability:\n"
        f"  credential_root: {tmp_path / 'credentials'}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("secret-value\n"))
    assert main(["--config", str(config), "credential", "wandb", "set",
                 "cloud-primary", "--stdin"]) == 0
    output = capsys.readouterr()
    assert "secret-value" not in output.out + output.err
    assert '"configured": true' in output.out
