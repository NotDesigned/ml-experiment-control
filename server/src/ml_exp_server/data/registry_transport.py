"""Bounded OCI registry requests; credentials never enter public receipts."""
from __future__ import annotations

import base64
import hashlib
import http.client
import json
from pathlib import Path
import re
from urllib.parse import urlencode, urljoin, urlsplit

from ..application_errors import ApplicationError

MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"
OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
INDEX = "application/vnd.oci.image.index.v1+json"
DOCKER_INDEX = "application/vnd.docker.distribution.manifest.list.v2+json"
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


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
        self.host, self.repository = repository.split("/", 1)
        self.origin, self.prefix = "https://" + self.host, "/v2/" + self.repository
        self.timeout, self.bearer = timeout, None
        self.blob_hosts = frozenset(blob_hosts)
        self.auth_hosts = frozenset(auth_hosts)
        if any(not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9.-]+", host)
               for host in self.auth_hosts):
            raise ValueError("invalid registry authentication host allowlist")
        config = json.loads(Path(auth_file).read_text())
        auth = config.get("auths", {}).get(self.host) or config.get("auths", {}).get(self.origin)
        if not isinstance(auth, dict) or not isinstance(auth.get("auth"), str):
            raise ValueError("data registry credentials are unavailable")
        raw = base64.b64decode(auth["auth"], validate=True)
        if b":" not in raw or b"\n" in raw or b"\r" in raw:
            raise ValueError("invalid data registry credentials")
        self.basic = "Basic " + auth["auth"]

    def location(self, value):
        target = urlsplit(urljoin(self.origin, value))
        if (target.scheme != "https" or target.netloc != self.host or target.username or target.password
                or target.fragment or not target.path.startswith(self.prefix + "/blobs/uploads/")
                or not re.fullmatch(r"/[A-Za-z0-9._/-]+", target.path) or any(p in {".", ".."} for p in target.path.split("/"))):
            raise ValueError("registry upload location escaped its repository")
        return target.path + ("?" + target.query if target.query else "")

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

    def begin(self):
        _, headers, _ = self.request("POST", self.prefix + "/blobs/uploads/", body=b"", expected=(202,))
        return self.location(headers.get("location", ""))

    def offset(self, location):
        _, headers, _ = self.request("GET", self.location(location), expected=(204,))
        value = headers.get("range", "")
        if not re.fullmatch(r"0-[0-9]+", value):
            raise ValueError("registry upload offset is unavailable")
        end = int(value.split("-")[1])
        return (0 if end == 0 else end + 1), self.location(headers.get("location", location))

    def append(self, location, offset, body):
        if not body:
            raise ValueError("registry upload chunk is empty")
        _, headers, _ = self.request("PATCH", self.location(location), body=body,
                                    headers={"Content-Type": "application/octet-stream",
                                             "Content-Range": f"{offset}-{offset + len(body) - 1}"}, expected=(202,))
        if headers.get("range") != f"0-{offset + len(body) - 1}":
            raise ValueError("registry upload acknowledgement has another offset")
        return self.location(headers.get("location", location))

    def finish(self, location, sha):
        if not DIGEST.fullmatch(sha):
            raise ValueError("invalid registry blob digest")
        path = self.location(location)
        self.request("PUT", path + ("&" if "?" in path else "?") + urlencode({"digest": sha}), body=b"", expected=(201,))
        self.check_blob(sha)

    def check_blob(self, sha):
        if not DIGEST.fullmatch(sha):
            raise ValueError("invalid registry blob digest")
        status, headers, _ = self.request("HEAD", self.prefix + "/blobs/" + sha, expected=(200, 307))
        if status == 307:
            # CCR's authenticated digest endpoint redirects even HEAD to OSS.
            # The signed target proves existence; PUT finalized the exact SHA.
            self.redirected_blob("HEAD", headers.get("location", ""), maximum=0)
        elif headers.get("docker-content-digest") != sha:
            raise ValueError("registry blob identity differs")

    def blob(self, value):
        sha = digest(value)
        status, _, _ = self.request("HEAD", self.prefix + "/blobs/" + sha, expected=(200, 307, 404))
        if status == 404:
            location = self.begin()
            location = self.append(location, 0, value)
            self.finish(location, sha)
        else:
            self.check_blob(sha)
        return {"digest": sha, "size": len(value)}

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
        # CCR may issue an HTTP signed OSS URL. Upgrade to TLS to the exact
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

    def manifest(self, sha):
        if not DIGEST.fullmatch(sha):
            raise ValueError("data base image must be immutable")
        _, _, raw = self.request("GET", self.prefix + "/manifests/" + sha,
                                headers={"Accept": ", ".join((MANIFEST, OCI_MANIFEST, INDEX, DOCKER_INDEX))})
        if digest(raw) != sha:
            raise ValueError("registry manifest identity differs")
        value = json.loads(raw)
        if value.get("mediaType") in {INDEX, DOCKER_INDEX}:
            rows = [row for row in value.get("manifests", []) if row.get("platform", {}).get("os") == "linux"
                    and row["platform"].get("architecture") == "amd64"]
            if len(rows) != 1:
                raise ValueError("data base must select one linux/amd64 image")
            return self.manifest(rows[0]["digest"])
        if value.get("mediaType") not in {MANIFEST, OCI_MANIFEST}:
            raise ValueError("unsupported data base manifest")
        return value

    def publish(self, tag, value):
        if not re.fullmatch(r"data-[0-9a-f]{64}", tag):
            raise ValueError("invalid data image tag")
        raw = encoded(value)
        self.request("PUT", self.prefix + "/manifests/" + tag, body=raw,
                     headers={"Content-Type": value["mediaType"]}, expected=(201,))
        sha = digest(raw)
        if self.manifest(sha) != value:
            raise ValueError("published data image differs")
        return self.host + "/" + self.repository + "@" + sha
