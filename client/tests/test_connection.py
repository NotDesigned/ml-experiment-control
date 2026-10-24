"""Local connection settings never disclose or send credentials during resolution."""
import json
import os
from pathlib import Path
import stat

import pytest

from ml_exp_client import connection
from ml_exp_client.api import Client, ClientError


URL = "https://api.example.org/ml-expd"
TOKEN = "test-private-token-not-for-a-real-service"


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for key in ("ML_EXPD_CLIENT_CONFIG", "XDG_CONFIG_HOME", "ML_EXPD_API_TOKEN", "ML_EXPD_API_TOKEN_FILE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(Client, "open", lambda *args, **kwargs: pytest.fail("connection settings must not send HTTP"))


def private(path, value):
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    return path


def test_config_location_uses_explicit_environment_xdg_then_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert connection.config_path() == tmp_path / ".config/ml-expd/client.json"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert connection.config_path() == tmp_path / "xdg/ml-expd/client.json"
    monkeypatch.setenv("ML_EXPD_CLIENT_CONFIG", str(tmp_path / "selected.json"))
    assert connection.config_path() == tmp_path / "selected.json"
    assert connection.config_path(tmp_path / "explicit.json") == tmp_path / "explicit.json"
    monkeypatch.setenv("ML_EXPD_CLIENT_CONFIG", "~/personal.json")
    assert connection.config_path() == Path("~/personal.json").expanduser()


@pytest.mark.parametrize("url,expected", [(URL + "///", URL), ("http://localhost:8080/api", "http://localhost:8080/api"),
                                        ("http://127.0.0.1:8080", "http://127.0.0.1:8080"),
                                        ("http://[::1]:8080", "http://[::1]:8080")])
def test_url_validation_is_local_and_preserves_proxy_prefix(url, expected):
    assert connection.normalize_url(url) == expected


@pytest.mark.parametrize("url", [None, 123, "", "http://public.example.org", "https://user:SECRET@api.example.org",
    "https://api.example.org?token=SECRET", "https://api.example.org/#SECRET", "https://[broken", "https://api.example.org/ bad",
    "https://api.example.org/\x1bSECRET",
    "https://" + "a" * connection.MAX_TOKEN_BYTES])
def test_invalid_url_errors_do_not_echo_untrusted_urls_or_credentials(url):
    with pytest.raises(ClientError, match="invalid API URL") as error:
        connection.normalize_url(url)
    assert "SECRET" not in str(error.value)


@pytest.mark.parametrize("token", [None, 1, "", "has space", "two\nlines", "\x1bcontrol", "unicode-密钥",
                                   "x" * (connection.MAX_TOKEN_BYTES + 1)])
def test_tokens_are_bounded_single_ascii_values_without_echo(token):
    with pytest.raises(ClientError, match="invalid API token") as error:
        connection.validate_token(token)
    assert str(token) not in str(error.value) if token else True
    assert connection.validate_token(TOKEN) == TOKEN


def test_missing_config_does_not_create_directories_or_change_existing_config(tmp_path):
    missing = tmp_path / "absent/client.json"
    assert connection.read_config(missing) == {} and not missing.parent.exists()
    config = private(tmp_path / "client.json", {"url": URL + "/", "token": TOKEN})
    before = config.read_bytes()
    assert connection.read_config(config) == {"url": URL, "token": TOKEN}
    assert config.read_bytes() == before
    connection.write_config(config, {})
    assert connection.read_config(config) == {}


@pytest.mark.parametrize("value", [[], {"unknown_secret": TOKEN}, {"url": None}, {"url": ""}, {"url": URL, "token": 1},
    {"token": TOKEN}, {"token_file": "private.token"}, {"url": URL, "token": TOKEN, "token_file": "private.token"},
    {"url": "https://SECRET@api.example.org"}, {"url": URL, "token": "SECRET invalid"},
    {"url": URL, "token_file": "bad\nSECRET"}, {"url": URL, "token_file": "x" * (connection.MAX_TOKEN_BYTES + 1)}])
def test_config_schema_rejects_unknown_ambiguous_or_invalid_fields_without_echo(tmp_path, value):
    path = private(tmp_path / "client.json", value)
    with pytest.raises(ClientError, match="cannot read private client configuration") as error:
        connection.read_config(path)
    assert TOKEN not in str(error.value) and "SECRET" not in str(error.value)
    with pytest.raises(ClientError, match="cannot write private client configuration"):
        connection.write_config(path, value)


@pytest.mark.parametrize("body", [b"{SECRET invalid json", b"\xffSECRET", b" " * (connection.MAX_CONFIG_BYTES + 1)])
def test_unreadable_json_unicode_and_oversized_config_have_fixed_errors(tmp_path, body):
    path = tmp_path / "client.json"
    path.write_bytes(body)
    path.chmod(0o600)
    with pytest.raises(ClientError, match="cannot read private client configuration") as error:
        connection.read_config(path)
    assert "SECRET" not in str(error.value)


@pytest.mark.skipif(os.name != "posix", reason="POSIX private-mode contract")
@pytest.mark.parametrize("mode", [0o640, 0o604, 0o666, 0o601])
def test_nonprivate_config_cannot_be_read_or_replaced(tmp_path, mode):
    path = private(tmp_path / "client.json", {"url": URL, "token": TOKEN})
    path.chmod(mode)
    before = path.read_bytes()
    with pytest.raises(ClientError):
        connection.read_config(path)
    with pytest.raises(ClientError):
        connection.write_config(path, {"url": URL})
    assert path.read_bytes() == before


def test_symlink_config_is_refused_including_broken_symlinks(tmp_path):
    target = private(tmp_path / "target.json", {"url": URL, "token": TOKEN})
    path = tmp_path / "linked.json"
    path.symlink_to(target)
    for exists in (True, False):
        if not exists:
            target.unlink()
        with pytest.raises(ClientError):
            connection.read_config(path)
        with pytest.raises(ClientError):
            connection.write_config(path, {"url": URL})
        assert path.is_symlink()


def test_directories_and_fifos_are_refused_without_waiting_for_input(tmp_path):
    paths = [tmp_path]
    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "fifo"
        os.mkfifo(fifo, 0o600)
        paths.append(fifo)
    for path in paths:
        with pytest.raises(ClientError):
            connection.read_config(path)
        with pytest.raises(ClientError):
            connection.read_token_file(path)
        with pytest.raises(ClientError):
            connection.write_config(path, {"url": URL})


def test_atomic_write_creates_private_directories_and_replaces_only_the_config(tmp_path, monkeypatch):
    path = tmp_path / "new/nested/client.json"
    value = {"url": URL + "/", "token": TOKEN}
    connection.write_config(path, value)
    assert connection.read_config(path) == {"url": URL, "token": TOKEN}
    assert value["url"] == URL + "/"
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == stat.S_IMODE(path.parent.parent.stat().st_mode) == 0o700
    before = path.read_bytes()
    def fail_save(*args):
        raise OSError("SECRET disk failure")
    monkeypatch.setattr(connection, "save", fail_save)
    with pytest.raises(ClientError, match="cannot write private client configuration") as error:
        connection.write_config(path, {"url": URL})
    assert "SECRET" not in str(error.value) and path.read_bytes() == before


def test_saved_token_file_is_absolute_but_never_read_by_save_or_show(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "client.json"
    token_file = tmp_path / "not-created.token"
    connection.write_config(config, {"url": URL, "token_file": token_file.name})
    saved = connection.read_config(config)
    assert saved == {"url": URL, "token_file": str(token_file)} and not token_file.exists()
    public = connection.public_config(config, saved)
    assert public == {"config_file": str(config), "url": URL, "token_configured": True, "token_file": str(token_file)}
    assert connection.public_config(config, {})["token_configured"] is False
    public = connection.public_config(config, {"url": URL, "token": TOKEN})
    assert public["token_configured"] is True and TOKEN not in json.dumps(public) and "token" not in public


def test_write_cannot_create_a_config_larger_than_the_read_limit(tmp_path):
    with pytest.raises(ClientError, match="cannot write private client configuration"):
        connection.write_config(tmp_path / "client.json", {"url": URL, "token_file": "密" * 8190})
    assert not (tmp_path / "client.json").exists()


def test_token_file_has_a_bounded_read_and_fixed_errors(tmp_path):
    path = tmp_path / "private.token"
    path.write_text(TOKEN + "\n")
    assert connection.read_token_file(path) == TOKEN
    path.write_text("x" * connection.MAX_TOKEN_BYTES)
    assert connection.read_token_file(path) == "x" * connection.MAX_TOKEN_BYTES
    for body in (b"x" * (connection.MAX_TOKEN_BYTES + 1), b"\xffSECRET", b"SECRET invalid", b" \n "):
        path.write_bytes(body)
        with pytest.raises(ClientError, match="cannot read API token file") as error:
            connection.read_token_file(path)
        assert "SECRET" not in str(error.value)
    with pytest.raises(ClientError, match="cannot read API token file"):
        connection.read_token_file(tmp_path / "absent.token")


def test_empty_environment_defers_missing_connection_to_the_transport():
    assert connection.resolve_connection(None, {}) == ("", "")
    assert connection.resolve_connection(URL + "/", {}) == (URL, "")
    assert connection.resolve_connection(URL, {"url": "https://other.example"}) == (URL, "")


def test_environment_token_overrides_saved_credentials_without_reading_saved_file(monkeypatch):
    monkeypatch.setenv("ML_EXPD_API_TOKEN", TOKEN)
    monkeypatch.setattr(connection, "read_token_file", lambda path: pytest.fail("saved file must not be read"))
    assert connection.resolve_connection(URL, {"url": "https://other.example", "token_file": "absent"}) == (URL, TOKEN)
    assert connection.resolve_connection(None, {}) == ("", TOKEN)


def test_environment_token_file_overrides_saved_token_and_mutual_exclusion_precedes_read(tmp_path, monkeypatch):
    path = tmp_path / "private.token"
    path.write_text(TOKEN)
    monkeypatch.setenv("ML_EXPD_API_TOKEN_FILE", str(path))
    assert connection.resolve_connection(URL, {"url": "https://other.example", "token": "saved"}) == (URL, TOKEN)
    monkeypatch.setenv("ML_EXPD_API_TOKEN", "another-private-token")
    monkeypatch.setattr(connection, "read_token_file", lambda path: pytest.fail("ambiguous credentials must not be read"))
    with pytest.raises(ClientError, match="set only one of"):
        connection.resolve_connection(URL, {})


def test_saved_credentials_are_url_bound_before_reading_a_secret_file(tmp_path, monkeypatch):
    path = tmp_path / "private.token"
    path.write_text(TOKEN)
    settings = {"url": URL + "/", "token_file": str(path)}
    assert connection.resolve_connection(URL, settings) == (URL, TOKEN)
    assert connection.resolve_connection(URL + "/", {"url": URL, "token": TOKEN}) == (URL, TOKEN)
    monkeypatch.setattr(connection, "read_token_file", lambda path: pytest.fail("foreign saved file must not be read"))
    for selected in (None, "https://other.example", URL + "/different-prefix"):
        with pytest.raises(ClientError, match="different API URL"):
            connection.resolve_connection(selected, settings)
    with pytest.raises(ClientError, match="invalid API URL"):
        connection.resolve_connection(URL, {"token": TOKEN})
