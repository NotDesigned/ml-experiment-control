"""Untrusted observation state cannot route credentials; use real local HTTP."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from ml_exp_client.cli import main
from ml_exp_client.connection import config_path, write_config
from ml_exp_client.state import save


TOKEN = "synthetic-destination-token"
SUBMISSION = "action-" + "a" * 16
IDENTITY = {"project": "demo", "run_id": "trial", "attempt_id": "attempt-001"}


@pytest.fixture
def destination(monkeypatch):
    for name in ("ML_EXPD_API_URL", "ML_EXPD_API_TOKEN", "ML_EXPD_API_TOKEN_FILE"):
        monkeypatch.delenv(name, raising=False)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization")))
            if self.path == "/service/api/health":
                value = {"min_client_protocol_version": 2, "api_protocol_version": 2}
            elif self.path == "/service/api/submissions/" + SUBMISSION:
                value = {**IDENTITY, "submission_id": SUBMISSION, "status": "VERIFIED"}
            elif self.path.startswith("/service/api/attempts/demo/trial%3A%3Aattempt-001/logs?"):
                value = {**IDENTITY, "stdout": [], "stderr": []}
            else:
                self.send_error(404)
                return
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/service", requests
        finally:
            server.shutdown()
            thread.join(timeout=2)


def observed_state(path, server):
    save(path, {"server": server, "project": "demo", "run_id": "trial", "config_sha256": None,
                "submission": {"submission_id": SUBMISSION, "attempt_id": "attempt-001"}})
    return path


@pytest.mark.parametrize("credential", ["ML_EXPD_API_TOKEN", "ML_EXPD_API_TOKEN_FILE"])
@pytest.mark.parametrize("command", ["status", "logs", "retry", "cancel"])
def test_state_alone_cannot_choose_credential_destination(destination, tmp_path, monkeypatch, capsys,
                                                         credential, command):
    url, requests = destination
    state = observed_state(tmp_path / "received.state.json", url)
    monkeypatch.setenv(credential, TOKEN if credential.endswith("_TOKEN") else str(tmp_path / "private.token"))
    monkeypatch.setattr("ml_exp_client.connection.read_token_file",
                        lambda path: pytest.fail("reject untrusted routing before reading any token file"))
    argv = ["run", command]
    if command in {"retry", "cancel"}:
        argv += ["--from-state", str(state), "--state", str(tmp_path / "operation.json")]
    else:
        argv += ["--state", str(state)]
    original = state.read_bytes()
    assert main(argv) == 2
    error = capsys.readouterr().err
    assert "set --url" in error and "no request was sent" in error and TOKEN not in error
    assert requests == [] and state.read_bytes() == original
    assert not (tmp_path / "operation.json").exists()


@pytest.mark.parametrize("url_source,credential", [
    ("explicit", "token"), ("explicit", "file"),
    ("environment", "token"), ("environment", "file"),
    ("saved-url", "token"), ("saved-url", "file"),
    ("saved-token", None), ("saved-file", None),
])
def test_trusted_url_and_credential_precedence_reach_only_the_bound_attempt(destination, tmp_path, monkeypatch,
                                                                           capsys, url_source, credential):
    url, requests = destination
    state = observed_state(tmp_path / "run.state.json", url)
    token_file = tmp_path / "private.token"
    token_file.write_text(TOKEN + "\n")
    prefix = []
    if url_source == "explicit":
        prefix = ["--url", url]
    elif url_source == "environment":
        monkeypatch.setenv("ML_EXPD_API_URL", url)
    else:
        settings = {"url": url, "token": "unused-saved-token"}
        if url_source == "saved-token":
            settings["token"] = TOKEN
        elif url_source == "saved-file":
            settings = {"url": url, "token_file": str(token_file)}
        write_config(config_path(), settings)
    if credential == "token":
        monkeypatch.setenv("ML_EXPD_API_TOKEN", TOKEN)
    elif credential == "file":
        monkeypatch.setenv("ML_EXPD_API_TOKEN_FILE", str(token_file))
    assert main([*prefix, "run", "logs", "--state", str(state)]) == 0
    output = capsys.readouterr().out
    assert json.loads(output)["attempt_id"] == "attempt-001" and TOKEN not in output
    assert len(requests) == 3 and all(auth == "Bearer " + TOKEN for _, auth in requests)
    assert requests[0][0] == "/service/api/health"


def test_foreign_state_does_not_read_saved_token_file_or_contact_server(destination, tmp_path, monkeypatch, capsys):
    url, requests = destination
    write_config(config_path(), {"url": url, "token_file": str(tmp_path / "private.token")})
    state = observed_state(tmp_path / "foreign.state.json", "https://foreign.invalid")
    monkeypatch.setattr("ml_exp_client.connection.read_token_file", lambda path: pytest.fail("URL binding comes first"))
    assert main(["run", "logs", "--state", str(state)]) == 2
    assert "no request was sent" in capsys.readouterr().err and requests == []
