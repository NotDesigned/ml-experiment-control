"""HTTP diagnostics remain structured, bounded and safe without replaying effects."""
import io
import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from urllib.error import HTTPError, URLError

import pytest

from ml_exp_client.api import Client, ClientError, ERROR_BODY_BYTES


def rejected(monkeypatch, body, *, status=507, headers=None, token="private-bearer"):
    stream = io.BytesIO(body)
    error = HTTPError("https://private.example?signature=hidden", status, "private reason", headers or {}, stream)
    client = Client("https://api.example", token)
    calls = []
    def fail(*args, **kwargs):
        calls.append((args, kwargs))
        raise error
    monkeypatch.setattr(client.opener, "open", fail)
    with pytest.raises(ClientError) as caught:
        client.call("/api/submissions/original/execute?token=do-not-echo", data={"confirmation": "EXECUTE original"})
    assert len(calls) == 1 and stream.closed
    return caught.value


def test_capacity_json_fields_and_safe_error_code_survive_http_error(monkeypatch):
    detail = {"message": "insufficient storage", "code": "UPLOAD_STORAGE",
              "required_free_bytes": 123, "available_bytes": 45, "reserved_bytes": 67}
    error = rejected(monkeypatch, json.dumps({"detail": detail}).encode(),
                     headers={"X-ML-Expd-Error-Code": "UPLOAD_STORAGE"})
    assert error.details == detail
    assert error.response_body == {"detail": detail}
    assert error.status == 507 and error.retryable is False
    assert error.error_code == "UPLOAD_STORAGE" and not error.body_truncated
    assert "do-not-echo" not in str(error) and "required_free_bytes" not in str(error)


def test_nested_json_removes_credentials_request_inputs_and_signed_urls(monkeypatch):
    private = "private-bearer"
    detail = {"message": f"failed {private} Bearer another-secret https://objects.example/x?signature=private-signature password='quoted private password'",
              "token": "nested-token", "API_KEY": "nested-key", "authorization": "other-auth",
              "url": "https://objects.example/private", "headers": {"safe-name": "private-header"},
              "nested": [{"input": {"safe-name": "private-input"}, "available_bytes": 42}],
              private: "safe", "info": "s3://private-bucket/object?signature=hidden",
              "exception": "-----BEGIN RSA PRIVATE KEY-----\nprivate-key-bytes\n-----END RSA PRIVATE KEY-----"}
    error = rejected(monkeypatch, json.dumps({"detail": detail}).encode())
    text = json.dumps(error.response_body) + json.dumps(error.details) + str(error)
    for secret in (private, "another-secret", "objects.example", "private-signature", "quoted private password",
                   "nested-token", "nested-key", "other-auth", "private-header", "private-input", "private-bucket", "private-key-bytes"):
        assert secret not in text
    assert error.details["nested"][0]["available_bytes"] == 42


def test_queue_busy_preserves_top_level_diagnostics_and_sanitizes_siblings(monkeypatch):
    body = {"detail": "background execution queue is full", "code": "EXECUTOR_BUSY",
            "lane": "action", "retry_after_seconds": 2, "token": "sibling-secret",
            "diagnostic": "https://private.example/object?signature=hidden private-bearer"}
    error = rejected(monkeypatch, json.dumps(body).encode(), status=503,
                     headers={"X-ML-Expd-Error-Code": "EXECUTOR_BUSY", "Retry-After": "2"})
    assert error.details == body["detail"]
    assert error.response_body["code"] == "EXECUTOR_BUSY"
    assert error.response_body["lane"] == "action"
    assert error.response_body["retry_after_seconds"] == 2
    assert error.response_body["token"] == "[REDACTED]"
    assert error.retry_after == 2 and error.error_code == "EXECUTOR_BUSY"
    assert error.status == 503 and error.retryable is False
    text = json.dumps(error.response_body) + str(error)
    for secret in ("sibling-secret", "private.example", "signature=hidden", "private-bearer"):
        assert secret not in text
    assert "background execution" not in str(error)


def test_json_without_detail_keeps_complete_body_in_both_fields(monkeypatch):
    body = {"code": "UPLOAD_STORAGE", "required_free_bytes": 123, "available_bytes": 45}
    error = rejected(monkeypatch, json.dumps(body).encode())
    assert error.details == body and error.response_body == body


@pytest.mark.parametrize("body, truncated", [(b"not JSON private-bearer", False), (b'{"detail":', False),
    (b"x" * (ERROR_BODY_BYTES + 100), True)])
def test_non_json_and_oversized_bodies_are_discarded_and_closed(monkeypatch, body, truncated):
    error = rejected(monkeypatch, body)
    assert error.status == 507 and error.details is None and error.response_body is None
    assert error.body_truncated is truncated
    assert "private-bearer" not in str(error)


def test_http_error_body_read_is_bounded(monkeypatch):
    reads = []
    class Body(io.BytesIO):
        def read(self, amount=-1):
            reads.append(amount)
            return super().read(amount)
    body = Body(b"x" * (ERROR_BODY_BYTES * 2))
    client = Client("https://api.example", "private-bearer")
    def fail(*args, **kwargs):
        raise HTTPError("https://private.example", 503, "private", {}, body)
    monkeypatch.setattr(client.opener, "open", fail)
    with pytest.raises(ClientError):
        client.call("/api/submissions/original/execute", data={})
    assert reads == [ERROR_BODY_BYTES + 1] and body.closed


@pytest.mark.parametrize("retry_after, expected", [("3", 3), ("999999", 86400), ("-1", None),
    ("https://private.example?token=secret", None), (None, None)])
def test_retry_after_is_safe_structured_metadata_without_retry(monkeypatch, retry_after, expected):
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    error = rejected(monkeypatch, b'{"detail":{"code":"EXECUTOR_BUSY"}}', status=503, headers=headers)
    assert error.retry_after == expected and error.retryable is False
    assert "private.example" not in str(error)


def test_http_date_retry_after_and_bounded_json_shape(monkeypatch):
    deadline = datetime.now(timezone.utc) + timedelta(seconds=120)
    value = {"items": ["x" * 2000] * 100, "deep": {"one": {"two": {"three": {"four": {"five": {"six": {"seven": "hidden"}}}}}}}}
    # Short encoded input expands structurally only within the explicit limits.
    value["items"] = ["x" * 1100] * 8
    error = rejected(monkeypatch, json.dumps({"detail": value}).encode(), headers={"Retry-After": format_datetime(deadline)})
    assert 118 <= error.retry_after <= 120
    assert len(error.details["items"][0]) == 1024
    assert "hidden" not in json.dumps(error.details)


def test_untrusted_error_code_and_network_error_do_not_echo_credentials(monkeypatch):
    error = rejected(monkeypatch, b"{}", headers={"X-ML-Expd-Error-Code": "https://private.example?token=secret"})
    assert error.error_code == "API_ERROR"
    client = Client("https://api.example", "private-bearer")
    monkeypatch.setattr(client.opener, "open", lambda *a, **kw: (_ for _ in ()).throw(URLError("private-bearer")))
    with pytest.raises(ClientError) as caught:
        client.call("/api/runs/example?token=hidden")
    assert caught.value.retryable and caught.value.status is None and caught.value.details is None
    assert caught.value.response_body is None
    assert "hidden" not in str(caught.value) and "private-bearer" not in str(caught.value)


def test_cli_prints_safe_structured_http_diagnostics(monkeypatch, capsys):
    from ml_exp_client import cli
    body = {"detail": "full", "code": "EXECUTION_CAPACITY_UNAVAILABLE", "lane": "collection",
            "retry_after_seconds": 2, "available_bytes": 42, "required_free_bytes": 123,
            "token": "private-bearer", "info": "https://objects.example/object?signature=hidden"}
    error = rejected(monkeypatch, json.dumps(body).encode(), status=503,
                     headers={"Retry-After": "2", "X-ML-Expd-Error-Code": "EXECUTION_CAPACITY_UNAVAILABLE"})
    class RejectedClient:
        def negotiate(self):
            raise error
    monkeypatch.setenv("ML_EXPD_API_TOKEN", "private-bearer")
    monkeypatch.setattr(cli, "Client", lambda *a, **kw: RejectedClient())
    assert cli.main(["--url", "https://api.example", "check"]) == 2
    output = capsys.readouterr()
    report = json.loads(output.err)
    assert output.out == ""
    assert report["status"] == 503 and report["retryable"] is False
    assert report["error_code"] == "EXECUTION_CAPACITY_UNAVAILABLE" and report["retry_after"] == 2
    assert report["details"] == "full" and report["body_truncated"] is False
    assert report["response_body"]["lane"] == "collection"
    assert report["response_body"]["required_free_bytes"] == 123
    assert report["response_body"]["available_bytes"] == 42
    for secret in ("private-bearer", "objects.example", "signature=hidden", "do-not-echo"):
        assert secret not in output.err


def test_cli_rechecks_diagnostic_sanitization_and_preserves_other_errors(monkeypatch, capsys):
    from ml_exp_client import cli
    error = ClientError("failure private-bearer https://private.example", status=507,
                        details={"password": "private-password", "available_bytes": 2},
                        response_body={"authorization": "private-auth", "info": "Bearer private-second"})
    class RejectedClient:
        def negotiate(self):
            raise error
    monkeypatch.setenv("ML_EXPD_API_TOKEN", "private-bearer")
    monkeypatch.setattr(cli, "Client", lambda *a, **kw: RejectedClient())
    assert cli.main(["--url", "https://api.example", "check"]) == 2
    text = capsys.readouterr().err
    assert json.loads(text)["details"]["available_bytes"] == 2
    for secret in ("private-bearer", "private.example", "private-password", "private-auth", "private-second"):
        assert secret not in text
    error = OSError("local file unavailable")
    assert cli.main(["--url", "https://api.example", "check"]) == 2
    assert capsys.readouterr().err == "error: local file unavailable\n"


def test_optional_storage_check_failure_preserves_successes_and_nonzero_status(tmp_path, monkeypatch, capsys):
    from ml_exp_client.cli import main
    calls = []
    class PartialCheck:
        def __init__(self, url, token):
            self.token = token
        def negotiate(self):
            return {"version": "test", "capabilities": ["data-assets.v1", "environments.v1"],
                    "openapi_path": "/api/v2/openapi.json"}
        def call(self, path):
            calls.append(path)
            if path == "/api/storage-limits":
                raise ClientError("capacity private-bearer https://private.example", status=503,
                    error_code="DESKTOP_RPC_FAILED", retry_after=2,
                    details={"available_bytes": 12, "password": "private-password"},
                    response_body={"detail": "temporarily unavailable", "authorization": "private-auth"})
            return {"path": path}
    monkeypatch.setenv("ML_EXPD_API_TOKEN", "private-bearer")
    monkeypatch.setattr("ml_exp_client.cli.Client", PartialCheck)
    schema = tmp_path / "openapi.json"
    assert main(["--url", "https://api.example", "check", "--json", "--schema", str(schema)]) == 2
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert output.err == ""
    assert report["health"]["version"] == "test"
    assert report["policy"] == {"path": "/api/actions/policy"}
    assert report["executors"] == {"path": "/api/executors"}
    assert report["environments"] == {"path": "/api/environments"}
    error = report["storage_limits_error"]
    assert error["status"] == 503 and error["error_code"] == "DESKTOP_RPC_FAILED" and error["retry_after"] == 2
    assert error["details"]["available_bytes"] == 12
    assert "storage_limits" not in report
    assert json.loads(schema.read_text()) == {"path": "/api/v2/openapi.json"}
    assert calls.count("/api/storage-limits") == 1
    for secret in ("private-bearer", "private.example", "private-password", "private-auth"):
        assert secret not in output.out


def test_check_still_fails_closed_when_required_executor_query_fails(monkeypatch, capsys):
    from ml_exp_client.cli import main
    class FailedCheck:
        def __init__(self, *args):
            pass
        def negotiate(self):
            return {"capabilities": ["data-assets.v1"]}
        def call(self, path):
            if path == "/api/actions/policy":
                return {}
            assert path == "/api/executors"
            raise ClientError("executors unavailable", status=503)
    monkeypatch.setattr("ml_exp_client.cli.Client", FailedCheck)
    assert main(["--url", "https://api.example", "check"]) == 2
    output = capsys.readouterr()
    assert output.out == "" and json.loads(output.err)["status"] == 503
