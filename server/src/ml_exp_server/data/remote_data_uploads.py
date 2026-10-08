"""Private builder RPC for desktop-backed client data uploads."""
from __future__ import annotations

import hashlib
import json
from urllib.parse import urlencode

from ..application_errors import ApplicationError
from ..image_builder import UnixConnection


def stage_request(socket, operation, metadata, body=b""):
    connection = UnixConnection(socket, timeout=1200)
    try:
        connection.request("POST", "/data-stage/" + operation, body=body, headers={
            "Content-Type": "application/octet-stream", "X-ML-Expd-Data": json.dumps(metadata, ensure_ascii=True)})
        response = connection.getresponse()
        raw = response.read(8 * 1024 ** 2 + 1)
        if len(raw) > 8 * 1024 ** 2:
            raise ValueError("desktop upload metadata is too large")
        value = json.loads(raw)
        if response.status != 200:
            raise ApplicationError("desktop upload is unavailable", status_code=value.get("status_code", 409),
                                   code=value.get("error", "DESKTOP_UPLOAD_UNAVAILABLE"))
        return value
    finally:
        connection.close()


class RemoteDataUploads:
    remote = True

    def __init__(self, assets, socket):
        self.assets, self.socket, self.root = assets, socket, assets.root

    def call(self, operation, binding, upload_id=None, **fields):
        value = {"binding": binding, **fields}
        if upload_id is not None:
            value["upload_id"] = upload_id
        return stage_request(self.socket, operation, value)

    def create(self, binding, digest, size, limit):
        value = self.call("create", binding, sha256=digest, bytes=size)
        if value["status"] == "COMPLETED":
            # Desktop completion may have committed before the API host lost
            # its response. Repair only the small locator, never reupload data.
            value["result"] = self.assets.publish_remote(binding["project"], value["result"])
        return value

    def read(self, upload_id, binding):
        return self.call("read", binding, upload_id)

    @staticmethod
    def part_size(value, number):
        from .multipart_upload import UploadStore
        return UploadStore.part_size(value, number)

    def part(self, upload_id, binding, number, body, digest, size):
        if len(body) != size or hashlib.sha256(body).hexdigest() != digest:
            raise ValueError("invalid data upload part")
        return stage_request(self.socket, "part", {"upload_id": upload_id, "binding": binding,
                             "number": number, "sha256": digest, "bytes": size}, body)

    def complete(self, upload_id, binding):
        value = self.call("complete", binding, upload_id)
        return self.assets.publish_remote(binding["project"], value)

    def abort(self, upload_id, binding):
        return self.call("abort", binding, upload_id)


def remote_uploads(assets):
    if assets.objects.config.get("data_upload_storage") == "desktop-builder":
        return RemoteDataUploads(assets, assets.objects.config["data_upload_socket"])
    return None


class RemoteArchive:
    def __init__(self, connection, response, size):
        self.connection, self.response, self.size = connection, response, size

    def iter_chunks(self, chunk_size):
        remaining = self.size
        try:
            while remaining:
                chunk = self.response.read(min(remaining, chunk_size))
                if not chunk:
                    raise ValueError("desktop archive stream is truncated")
                remaining -= len(chunk)
                yield chunk
        finally:
            self.close()

    def close(self):
        self.connection.close()


def remote_archive(socket, project, asset_id, size):
    connection = UnixConnection(socket, timeout=1200)
    try:
        connection.request("GET", "/data-stage/archive?" + urlencode({"project": project, "asset_id": asset_id}))
        response = connection.getresponse()
        if response.status != 200 or response.getheader("Content-Length") != str(size):
            raise ValueError("desktop archive stream is unavailable")
        return RemoteArchive(connection, response, size)
    except Exception:
        connection.close()
        raise
