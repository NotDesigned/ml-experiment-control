"""Protocol-2 HTTP transport with bounded safe diagnostics and no request replay."""
from __future__ import annotations

import ipaddress
import json
import math
import re
import socket
import ssl
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


ERROR_BODY_BYTES = 16 * 1024
_SENSITIVE_FIELD = re.compile(r"token|secret|password|credential|authorization|cookie|(?:api|private|access|client)[_-]?key|signature|url|argv|command|environment|^env$|^input$|headers|body|^auth$|^key$", re.I)
_ATTEMPT = re.compile(r"attempt-[0-9]{3,}\Z")


def _safe_error_text(value, token):
    if token:
        value = value.replace(token, "[REDACTED]")
    value = re.sub(r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----", "[REDACTED]", value, flags=re.S)
    value = re.sub(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s\"'<>]+", "[REDACTED]", value)
    value = re.sub(r"(?i)\bBearer\s+[^\s,;\"']+", "Bearer [REDACTED]", value)
    value = re.sub(r'''(?i)([\w-]*(?:token|secret|password|credential|authorization|cookie|(?:api|private|access|client)[_-]?key|signature)[\w-]*["']?\s*[:=]\s*)(?:"[^"]*"|'[^']*'|[^\s,;}]+)''', r"\1[REDACTED]", value)
    return "".join(char if ord(char) >= 32 else " " for char in value)[:1024]


def _safe_error_json(value, token, *, depth=0, budget=None):
    """Keep bounded diagnostic JSON without request inputs or credentials."""
    budget = [256] if budget is None else budget
    budget[0] -= 1
    if budget[0] < 0 or depth > 6:
        return "[TRUNCATED]"
    if isinstance(value, dict):
        result = {}
        for key, item in list(value.items())[:64]:
            clean_key = _safe_error_text(key, token)[:128]
            result[clean_key] = ("[REDACTED]" if _SENSITIVE_FIELD.search(key) else
                                 _safe_error_json(item, token, depth=depth + 1, budget=budget))
        return result
    if isinstance(value, list):
        return [_safe_error_json(item, token, depth=depth + 1, budget=budget) for item in value[:64]]
    if isinstance(value, str):
        return _safe_error_text(value, token)
    if value is None or type(value) in {bool, int} or type(value) is float and math.isfinite(value):
        return value
    return None


def _retry_after_seconds(value):
    if not value or len(value) > 128:
        return None
    try:
        seconds = int(value) if value.isdecimal() else math.ceil(
            (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        return min(86400, max(0, seconds))
    except (TypeError, ValueError, OverflowError):
        return None


class ClientError(RuntimeError):
    def __init__(self, message, *, status=None, retryable=False, details=None,
                 error_code=None, retry_after=None, body_truncated=False,
                 response_body=None):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.details = details
        # Retain sibling diagnostics while preserving the established FastAPI
        # `detail` convenience field. Both contain only bounded, sanitized JSON.
        self.response_body = response_body
        self.error_code = error_code
        self.retry_after = retry_after
        self.body_truncated = body_truncated


def error_payload(error, token=""):
    """Render one bounded diagnostic consistently in CLI and partial checks."""
    return {"error": _safe_error_text(str(error), token), **{
        key: _safe_error_json(getattr(error, key), token)
        for key in ("status", "retryable", "error_code", "retry_after", "details", "response_body", "body_truncated")
    }}


def _connection_failure(error):
    """Classify typed failures without disclosing exception text or endpoints."""
    reason = error.reason if isinstance(error, URLError) else error
    if isinstance(reason, ssl.SSLCertVerificationError):
        # OpenSSL X509_V_ERR_HOSTNAME_MISMATCH / IP_ADDRESS_MISMATCH.
        if getattr(reason, "verify_code", None) in {62, 64}:
            return ("TLS_HOSTNAME_MISMATCH", "certificate does not match the API hostname; use an address covered by the server certificate", False)
        return ("TLS_CERTIFICATE_VERIFICATION_FAILED", "certificate verification failed; check certificate validity and the trusted CA (SSL_CERT_FILE)", False)
    if isinstance(reason, ssl.SSLError):
        return ("TLS_HANDSHAKE_FAILED", "TLS handshake failed; ask the operator to check the HTTPS endpoint and TLS/proxy configuration", False)
    if isinstance(reason, ConnectionRefusedError):
        return ("CONNECTION_REFUSED", "connection refused; check the API address, port, running service and firewall", True)
    if isinstance(reason, socket.gaierror):
        return ("DNS_LOOKUP_FAILED", "API hostname lookup failed; check the API address, DNS and network", True)
    if isinstance(reason, TimeoutError):
        return ("CONNECTION_TIMEOUT", "connection timed out; check the network, proxy and service availability", True)
    return ("CONNECTION_FAILED", "connection failed; check the API address and network", True)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        # Never forward a bearer token to another URL after a proxy redirect.
        return None


def segment(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}", value):
        raise ClientError("invalid API identity")
    return quote(value, safe="")


def attempt_id(value):
    if not isinstance(value, str) or not _ATTEMPT.fullmatch(value):
        raise ClientError("choose an exact attempt-NNN identity")
    return value


def validate_base_url(base: str) -> str:
    """Share URL policy with local configuration without constructing a transport."""
    url = urlsplit(base)
    loopback = url.hostname == "localhost"
    try:
        loopback = loopback or ipaddress.ip_address(url.hostname or "").is_loopback
    except ValueError:
        pass
    if (not url.hostname or url.username or url.password or url.query or url.fragment
            or not (url.scheme == "https" or url.scheme == "http" and loopback)):
        raise ClientError("API URL must use HTTPS (HTTP is allowed on loopback only)")
    return base.rstrip("/")


class Client:
    def __init__(self, base: str, token: str, *, timeout: float = 60, upload_timeout: float = 1200):
        base = validate_base_url(base)
        if not token or any(c.isspace() for c in token):
            raise ClientError("set ML_EXPD_API_TOKEN or ML_EXPD_API_TOKEN_FILE")
        self.base, self.token, self.timeout = base, token, timeout
        self.upload_timeout = max(timeout, upload_timeout)
        self.opener = build_opener(NoRedirect())

    def open(self, path: str, *, data=None, raw=None, length: int | None = None, method=None, timeout=None):
        if not path.startswith("/api/") or "\x00" in path:
            raise ClientError("request must stay within /api/")
        headers = {"Authorization": "Bearer " + self.token,
                   "X-ML-Expd-Client-Protocol": "2"}
        body = raw
        if data is not None:
            body = json.dumps(data).encode()
            headers["Content-Type"] = "application/json"
        elif raw is not None:
            headers["Content-Type"] = "application/octet-stream"
            if length is not None:
                headers["Content-Length"] = str(length)
        method = method or ("GET" if body is None else "POST")
        try:
            return self.opener.open(Request(self.base + path, body, headers, method=method),
                                    timeout=self.timeout if timeout is None else timeout)
        except HTTPError as exc:
            code = exc.headers.get("X-ML-Expd-Error-Code", "API_ERROR")
            if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", code) or self.token in code:
                code = "API_ERROR"
            details, response_body, truncated = None, None, False
            retry_after = _retry_after_seconds(exc.headers.get("Retry-After"))
            try:
                body = exc.read(ERROR_BODY_BYTES + 1)
                truncated = len(body) > ERROR_BODY_BYTES
                if not truncated:
                    response_body = _safe_error_json(json.loads(body), self.token)
                    details = (response_body.get("detail", response_body)
                               if isinstance(response_body, dict) else response_body)
            except Exception:
                # Malformed/truncated proxy responses must not hide the HTTP
                # status or expose unparsed HTML/plaintext credentials.
                pass
            finally:
                exc.close()
            safe_path = _safe_error_text(path.partition("?")[0], self.token)
            raise ClientError(f"{method} {safe_path}: HTTP {exc.code} {code}; inspect saved IDs before retrying",
                              status=exc.code, details=details, error_code=code,
                              retry_after=retry_after, body_truncated=truncated,
                              response_body=response_body) from None
        except (URLError, TimeoutError, OSError) as exc:
            safe_path = _safe_error_text(path.partition("?")[0], self.token)
            code, hint, retryable = _connection_failure(exc)
            recovery = " For writes, inspect saved IDs before retrying." if method not in {"GET", "HEAD"} else ""
            raise ClientError(f"{method} {safe_path}: {hint}. See docs/configuration.md#connection-errors.{recovery}",
                              retryable=retryable, error_code=code) from None

    def call(self, path: str, **kwargs):
        with self.open(path, **kwargs) as response:
            return json.load(response)

    def open_object(self, url: str):
        target = urlsplit(url)
        if (target.scheme != "https" or not target.hostname or target.username or target.password or target.fragment):
            raise ClientError("object storage download must use HTTPS")
        try:
            # The signed URL is the only credential. Do not forward API Bearer
            # headers or follow redirects; never echo the URL in errors/state.
            return self.opener.open(Request(url), timeout=self.timeout)
        except (HTTPError, URLError, TimeoutError, OSError):
            raise ClientError("object storage download failed; request a fresh download link") from None

    def negotiate(self):
        health = self.call("/api/health")
        if not health["min_client_protocol_version"] <= 2 <= health["api_protocol_version"]:
            raise ClientError("daemon does not support protocol 2")
        self.capabilities = health.get("capabilities", [])
        return health

    def wait(self, path: str, *, pending=("EXECUTING",), seconds=900, interval=2, observer=None):
        deadline = time.monotonic() + seconds
        while True:
            value = self.call(path)
            if value["status"] not in pending:
                return value
            if observer is not None:
                observer(value)
            if time.monotonic() >= deadline:
                raise ClientError(f"poll timed out: GET {path}; execution continues on the server")
            time.sleep(interval)
