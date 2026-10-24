"""Private persistent defaults and offline setup preserve exact-server binding."""
import getpass
import io
import json
from pathlib import Path
import sys

import pytest

from ml_exp_client.cli import main


@pytest.fixture
def setup(monkeypatch):
    for name in ("ML_EXPD_API_URL", "ML_EXPD_API_TOKEN", "ML_EXPD_API_TOKEN_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("ml_exp_client.cli.Client", lambda *a, **k: pytest.fail("local config must not connect"))


def test_prompt_and_show_do_not_echo_token(setup, monkeypatch, capsys):
    stdin = io.StringIO()
    stdin.isatty = lambda: True
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(getpass, "getpass", lambda message: "synthetic-private-token")
    assert main(["config", "set", "--url", "https://api.example/ml-expd/"]) == 0
    saved = json.loads(capsys.readouterr().out)
    assert saved["url"] == "https://api.example/ml-expd" and saved["token_configured"]
    assert "synthetic-private-token" not in json.dumps(saved)
    path = Path(saved["config_file"])
    assert json.loads(path.read_text())["token"] == "synthetic-private-token"
    assert main(["config", "show"]) == 0
    assert "synthetic-private-token" not in capsys.readouterr().out
    # Re-saving the same URL needs no second token prompt.
    monkeypatch.setattr(getpass, "getpass", lambda message: pytest.fail("existing same-server token should be retained"))
    assert main(["config", "set", "--url", saved["url"]]) == 0


def test_token_file_reference_clear_and_custom_config(setup, tmp_path, capsys):
    token = tmp_path / "private.token"
    token.write_text("synthetic-file-token\n")
    path = tmp_path / "nested" / "client.json"
    prefix = ["--config", str(path), "config"]
    assert main([*prefix, "set", "--url", "https://api.example", "--token-file", str(token)]) == 0
    output = capsys.readouterr().out
    assert "synthetic-file-token" not in output and json.loads(path.read_text()) == {
        "url": "https://api.example", "token_file": str(token.resolve())}
    assert main([*prefix, "clear"]) == 0
    assert not path.exists() and token.read_text() == "synthetic-file-token\n"
    assert main([*prefix, "clear"]) == 0


def test_stdin_and_rotation_are_private_atomic_and_server_bound(setup, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("first-token\n"))
    assert main(["config", "set", "--url", "https://first.example", "--token-stdin"]) == 0
    path = Path(json.loads(capsys.readouterr().out)["config_file"])
    before = path.read_bytes()
    # A URL change must not silently copy credentials to a new server.
    assert main(["config", "set", "--url", "https://other.example"]) == 2
    assert path.read_bytes() == before and "terminal" in capsys.readouterr().err
    monkeypatch.setattr(sys, "stdin", io.StringIO("new-token\n"))
    assert main(["config", "set", "--url", "https://other.example", "--token-stdin"]) == 0
    assert json.loads(path.read_text()) == {"url": "https://other.example", "token": "new-token"}


@pytest.mark.parametrize("failure", [EOFError, KeyboardInterrupt, getpass.GetPassWarning])
def test_cancelled_or_insecure_prompt_writes_nothing(setup, monkeypatch, capsys, failure):
    stdin = io.StringIO(); stdin.isatty = lambda: True
    monkeypatch.setattr(sys, "stdin", stdin)
    def fail(message): raise failure()
    monkeypatch.setattr(getpass, "getpass", fail)
    assert main(["config", "set", "--url", "https://api.example"]) == 2
    assert "unchanged" in capsys.readouterr().err


def test_defaults_and_overrides_precede_any_network(monkeypatch, tmp_path, capsys):
    for name in ("ML_EXPD_API_URL", "ML_EXPD_API_TOKEN", "ML_EXPD_API_TOKEN_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "stdin", io.StringIO("saved-token\n"))
    assert main(["config", "set", "--url", "https://api.example", "--token-stdin"]) == 0
    capsys.readouterr()
    calls = []
    class API:
        def __init__(self, url, token): calls.append((url, token))
        def negotiate(self): return {}
    monkeypatch.setattr("ml_exp_client.cli.Client", API)
    monkeypatch.setattr("ml_exp_client.cli.dispatch", lambda *args: {"ok": True})
    assert main(["check", "--json"]) == 0
    assert calls == [("https://api.example", "saved-token")]
    assert main(["--url", "https://other.example", "check", "--json"]) == 2
    assert len(calls) == 1 and "saved-token" not in capsys.readouterr().err
    monkeypatch.setenv("ML_EXPD_API_URL", "https://environment.example")
    monkeypatch.setenv("ML_EXPD_API_TOKEN", "environment-token")
    assert main(["check", "--json"]) == 0
    assert calls[-1] == ("https://environment.example", "environment-token")
    assert main(["--url", "https://explicit.example", "check", "--json"]) == 0
    assert calls[-1] == ("https://explicit.example", "environment-token")
    # A foreign saved state is rejected before even negotiate sends a token.
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"server": "https://foreign.example", "project": "demo", "run_id": "trial", "config_sha256": None}))
    count = len(calls)
    assert main(["status", "--state", str(state)]) == 2
    assert len(calls) == count


def test_update_bypasses_config_auth_and_service_connection(setup, monkeypatch, capsys):
    monkeypatch.setattr("ml_exp_client.cli.read_config", lambda path: pytest.fail("update is independent from service configuration"))
    seen = []
    monkeypatch.setattr("ml_exp_client.cli.update_client", lambda version, check: seen.append((version, check)) or {"updated": False})
    assert main(["update", "--check"]) == 0
    assert main(["update", "--version", "1.5.1"]) == 0
    assert seen == [(None, True), ("1.5.1", False)]
    assert capsys.readouterr().out.count('"updated": false') == 2


def test_missing_url_and_invalid_stdin_are_local_errors(setup, monkeypatch, capsys):
    assert main(["config", "set"]) == 2
    assert "--url" in capsys.readouterr().err
    monkeypatch.setattr(sys, "stdin", io.StringIO("private-token with spaces\n"))
    assert main(["config", "set", "--url", "https://api.example", "--token-stdin"]) == 2
    assert "private-token" not in capsys.readouterr().err


@pytest.mark.parametrize("suffix", [" " * 16385 + "unexpected-token", " " * 8193])
def test_oversized_stdin_is_not_silently_truncated(setup, monkeypatch, capsys, suffix):
    monkeypatch.setattr(sys, "stdin", io.StringIO("private-token" + suffix))
    assert main(["config", "set", "--url", "https://api.example", "--token-stdin"]) == 2
    assert "private-token" not in capsys.readouterr().err
