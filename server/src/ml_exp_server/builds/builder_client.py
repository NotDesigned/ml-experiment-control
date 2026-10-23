"""Shared private image-builder identities, errors and Unix client."""
from __future__ import annotations

import hashlib
import http.client
import json
import re
import socket

from ..dockerfile_build import DOCKERFILE_RECIPE, worker_digest
from ..worker_contract import WORKER_CONTRACT, recipe_digest


IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}$")
ID = re.compile(r"^[0-9a-f]{64}$")
PROJECT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
BUILD_REVISION = "sealed-context-transfer.v1"
MANIFEST_TYPE = "application/vnd.docker.distribution.manifest.v2+json"


class BuildStorageError(ValueError):
    """Only reviewed storage diagnostics may cross the private builder socket."""

    def __init__(self, code: str, details: dict):
        super().__init__(code)
        self.code, self.details = code, details


class BuildTransportError(ValueError):
    """Bounded transport evidence; uncertain publications are never replayed."""

    def __init__(self, code, details):
        super().__init__(code)
        self.code, self.details = code, details


class BuildReceiptMissingError(ValueError):
    """The builder verified the lookup identity and found no published receipt."""

    def __init__(self, message, *, retry_proof=None):
        super().__init__(message)
        self.retry_proof = retry_proof


def bundle_id(project: str, source_id: str, base_image: str, *, dockerfile_path: str = "Dockerfile") -> str:
    fields = [project, source_id, base_image, DOCKERFILE_RECIPE, WORKER_CONTRACT,
              worker_digest(), recipe_digest(), BUILD_REVISION, dockerfile_path]
    return hashlib.sha256(json.dumps(fields, separators=(",", ":")).encode()).hexdigest()


class UnixConnection(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.host)


def builder_request(path: str, payload: dict, *, timeout: int = 1200) -> dict:
    connection = UnixConnection(path, timeout=timeout)
    try:
        connection.request("POST", "/", body=json.dumps(payload), headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = json.loads(response.read(65536))
        if response.status != 200:
            if data.get("code") == "BUILD_RECEIPT_MISSING":
                raise BuildReceiptMissingError("image bundle has not been published", retry_proof=data.get("retry_proof"))
            if data.get("code") in {"BUILD_STORAGE_INSUFFICIENT", "BUILD_STORAGE_UNCHECKED", "BUILD_DISK_EXHAUSTED"}:
                raise BuildStorageError(data["code"], data["details"])
            if data.get("code") in {"BUILD_SESSION_TIMEOUT", "BUILD_CONTEXT_TRANSPORT", "DESKTOP_RPC_TIMEOUT", "DESKTOP_RPC_UNAVAILABLE", "DESKTOP_RPC_FAILED", "DESKTOP_RPC_RESPONSE", "BUILDER_DOCKER_UNAVAILABLE", "BUILD_PUBLICATION_UNRESOLVED", "BUILD_RECOVERY_UNCHECKED", "PAI_DSW_BUILD_UNCERTAIN", "PAI_DSW_RECONCILE_REQUIRED", "PAI_DSW_CLEANUP_REQUIRED"}:
                raise BuildTransportError(data["code"], data["details"])
            raise ValueError(data.get("error", "image packaging failed"))
        return data
    finally:
        connection.close()
