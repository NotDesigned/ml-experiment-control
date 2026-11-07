"""Bounded OCI registry requests; credentials never enter public receipts."""
from __future__ import annotations

import base64
import hashlib
import http.client
import json
from pathlib import Path
import re
from urllib.parse import urlencode, urlsplit

from ..application_errors import ApplicationError

MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"
OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
DOCKER_HUB_HOSTS = {"docker.io", "index.docker.io", "registry-1.docker.io"}


def registry_auth(auths, host):
    names = [host, "https://" + host]
    if host in DOCKER_HUB_HOSTS:
        names += [*sorted(DOCKER_HUB_HOSTS), "https://index.docker.io/v1/"]
    return next((auths[name] for name in names if name in auths), None)


def digest(value):
    return "sha256:" + hashlib.sha256(value).hexdigest()


class Registry:
    @classmethod
    def from_config(cls, config, *, timeout=60):
        return cls(config['repository'], config['registry_auth_file'], timeout=timeout,
                   auth_hosts=config.get('registry_auth_hosts', ()),
                   blob_hosts=config.get('registry_blob_hosts', ()))

    def __init__(self, repository, auth_file, *, timeout=60, blob_hosts=(), auth_hosts=()):
        if (not isinstance(repository, str) or not re.fullmatch(r"[A-Za-z0-9.-]+(?::[0-9]+)?/[a-z0-9][a-z0-9._/-]*", repository)
                or ".." in repository.split("/")):
            raise ValueError("invalid data registry repository")
        self.reference = repository
        host, self.repository = repository.split("/", 1)
        self.host = "registry-1.docker.io" if host in DOCKER_HUB_HOSTS else host
        self.origin, self.prefix = "https://" + self.host, "/v2/" + self.repository
        self.timeout, self.bearer = timeout, None
        self.blob_hosts = frozenset(blob_hosts)
        self.auth_hosts = frozenset(auth_hosts)
        if host in DOCKER_HUB_HOSTS:
            self.auth_hosts |= {"auth.docker.io"}
            self.blob_hosts |= {"production.cloudfront.docker.com", "production.cloudflare.docker.com"}
        if any(not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9.-]+", host)
               for host in self.auth_hosts):
            raise ValueError("invalid registry authentication host allowlist")
        config = json.loads(Path(auth_file).read_text())
        auth = registry_auth(config.get("auths", {}), host)
        if not isinstance(auth, dict) or not isinstance(auth.get("auth"), str):
            raise ValueError("data registry credentials are unavailable")
        raw = base64.b64decode(auth["auth"], validate=True)
        if b":" not in raw or b"\n" in raw or b"\r" in raw:
            raise ValueError("invalid data registry credentials")
        self.basic = "Basic " + auth["auth"]

    def _exchange(self, method, path, headers, body, maximum, *, host=None):
        connection = http.client.HTTPSConnection(host or self.host, timeout=self.timeout)
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read(maximum + 1)
            if len(raw) > maximum:
                raise ValueError("registry metadata exceeds its bound")
            return response.status, {key.lower(): value for key, value in response.getheaders()}, raw
        except (OSError, http.client.HTTPException):
            raise ApplicationError("data registry transport is unavailable", status_code=503, code="CCR_TRANSPORT") from None
        finally:
            connection.close()

    def request(self, method, path, *, body=None, headers=None, expected=(200,), maximum=8 * 1024 ** 2):
        if not path.startswith("/") or path.startswith("//") or "\r" in path or "\n" in path:
            raise ValueError("invalid registry request path")
        supplied = dict(headers or {})
        supplied["Authorization"] = "Bearer " + self.bearer if self.bearer else self.basic
        status, response_headers, raw = self._exchange(method, path, supplied, body, maximum)
        if status == 401:
            challenge = response_headers.get("www-authenticate", "")
            fields = dict(re.findall(r'([a-z]+)="([^"\r\n]*)"', challenge))
            realm = urlsplit(fields.get("realm", ""))
            if (not challenge.startswith("Bearer ") or realm.scheme != "https"
                    or realm.netloc not in {self.host, *self.auth_hosts}
                    or realm.username or realm.password or realm.fragment):
                raise ValueError("registry authentication realm is not trusted")
            query = urlencode({"service": fields.get("service", self.host), "scope": "repository:" + self.repository + ":pull,push"})
            token_path = realm.path + ("?" + realm.query + "&" if realm.query else "?") + query
            target = {"host": realm.netloc} if realm.netloc != self.host else {}
            code, _, token_raw = self._exchange("GET", token_path, {"Authorization": self.basic}, None, 65536, **target)
            if code != 200:
                raise ApplicationError("data registry authentication failed", status_code=503, code="CCR_AUTHENTICATION")
            token = json.loads(token_raw)
            self.bearer = token.get("token") or token.get("access_token")
            if not isinstance(self.bearer, str) or not self.bearer or any(c in self.bearer for c in "\r\n"):
                raise ValueError("invalid registry authentication response")
            supplied["Authorization"] = "Bearer " + self.bearer
            status, response_headers, raw = self._exchange(method, path, supplied, body, maximum)
        if status not in expected:
            raise ApplicationError("data registry rejected the request", status_code=503 if status >= 500 else 409,
                                   code="CCR_HTTP_" + str(status))
        return status, response_headers, raw

    def json_blob(self, sha):
        if not DIGEST.fullmatch(sha):
            raise ValueError("invalid registry config digest")
        status, headers, raw = self.request("GET", self.prefix + "/blobs/" + sha, expected=(200, 307))
        if status == 307:
            _, raw = self.redirected_blob("GET", headers.get("location", ""))
        if digest(raw) != sha:
            raise ValueError("registry config identity differs")
        return json.loads(raw)

    def redirected_blob(self, method, location, *, maximum=8 * 1024 ** 2):
        # A registry may issue an HTTP signed object-store URL. Upgrade to TLS to the exact
        # operator-approved host and never forward registry credentials.
        target = urlsplit(location)
        if (target.scheme not in {"http", "https"} or target.hostname not in self.blob_hosts
                or target.username or target.password or target.fragment or target.port not in {None, 80, 443}):
            raise ValueError("registry blob redirect is not trusted")
        connection = http.client.HTTPSConnection(target.hostname, timeout=self.timeout)
        try:
            connection.request(method, target.path + ("?" + target.query if target.query else ""))
            response = connection.getresponse()
            raw = response.read(maximum + 1)
            if response.status != 200 or len(raw) > maximum:
                raise ValueError("registry config download failed")
            return {key.lower(): value for key, value in response.getheaders()}, raw
        except (OSError, http.client.HTTPException):
            raise ApplicationError("data registry blob transport is unavailable", status_code=503, code="CCR_TRANSPORT") from None
        finally:
            connection.close()
