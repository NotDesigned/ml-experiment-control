"""Verify immutable OSS destination parts in-region, without extracting a tar.

Only this server-authored, zero-GPU job receives the import-scoped capability.
Neither account credentials nor signed object tickets appear in logs or reports.
"""
import hashlib
import io
import json
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


def call(config, operation, value):
    raw = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("REPORT_LIMIT")
    for attempt in range(8):
        request = urllib.request.Request(config["url"] + "/" + operation, data=raw,
                                        headers={"Authorization": "Bearer " + config["token"],
                                                 "Content-Type": "application/json"})
        try:
            with OPENER.open(request, timeout=130) as response:
                result = response.read(8 * 1024 * 1024 + 1)
                if len(result) > 8 * 1024 * 1024:
                    raise ValueError("RESPONSE_LIMIT")
                return json.loads(result)
        except urllib.error.HTTPError as error:
            if error.code not in (408, 429, 500, 502, 503, 504):
                raise ValueError("IMPORT_REQUEST_REJECTED") from None
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(min(2 ** attempt, 20))
    raise ValueError("IMPORT_TRANSPORT_FAILED")


class Parts(io.RawIOBase):
    def __init__(self, config, fetch=None):
        self.config = config
        self.fetch = fetch or self.download
        self.current = b""
        self.offset = 0
        self.number = 0
        self.bytes = 0
        self.whole = hashlib.sha256()
        self.records = {}

    def readable(self):
        return True

    def download(self, number):
        # Refreshing a lost/expired ticket only reads the same completed object.
        for attempt in range(5):
            ticket = call(self.config, "oss-copy", {"part": number})
            address = urllib.parse.urlsplit(ticket["url"])
            if (address.scheme != "https" or address.port not in (None, 443)
                    or not (address.hostname or "").endswith("-internal.aliyuncs.com")
                    or address.username or address.fragment or ticket["part"] != number):
                raise ValueError("OBJECT_TICKET_INVALID")
            try:
                request = urllib.request.Request(ticket["url"], headers=ticket["headers"])
                with OPENER.open(request, timeout=120) as response:
                    expected = min(self.config["part_bytes"], self.config["archive_bytes"]
                                   - number * self.config["part_bytes"])
                    if (response.status != 200 or ticket["bytes"] != expected
                            or response.headers.get("ETag") != ticket["etag"]
                            or int(response.headers.get("Content-Length", "-1")) != expected):
                        raise ValueError("OBJECT_IDENTITY_CHANGED")
                    data = response.read(expected + 1)
                    if len(data) != expected:
                        raise ValueError("OBJECT_SIZE_MISMATCH")
                    return data
            except (urllib.error.URLError, OSError):
                time.sleep(min(2 ** attempt, 10))
        raise ValueError("OBJECT_READ_FAILED")

    def read(self, size=-1):
        # tarfile streaming uses bounded reads; never accumulate the whole tar.
        if size < 0:
            size = 1024 * 1024
        if not size:
            return b""
        if self.offset == len(self.current):
            if self.number >= self.config["part_count"]:
                return b""
            data = self.fetch(self.number)
            expected = min(self.config["part_bytes"], self.config["archive_bytes"]
                           - self.number * self.config["part_bytes"])
            if len(data) != expected:
                raise ValueError("PART_SIZE_MISMATCH")
            self.records[str(self.number)] = {"sha256": hashlib.sha256(data).hexdigest(),
                                              "bytes": len(data)}
            self.current, self.offset = data, 0
            self.number += 1
        data = self.current[self.offset:self.offset + size]
        self.offset += len(data)
        self.bytes += len(data)
        self.whole.update(data)
        return data


def path_valid(path):
    return (bool(path) and len(path.encode()) <= 4096 and "\\" not in path
            and all(ord(c) >= 32 and not 127 <= ord(c) <= 159 for c in path)
            and all(p not in ("", ".", "..") for p in path.split("/"))
            and len(path.split("/")) <= 128)


def verify(config, fetch=None):
    reader = Parts(config, fetch)
    files = {}
    expanded = 0
    with tarfile.open(fileobj=reader, mode="r|") as archive:
        for entry in archive:
            if (entry.type not in (tarfile.REGTYPE, tarfile.AREGTYPE)
                    or entry.sparse is not None or not path_valid(entry.name)
                    or entry.name in files or len(files) >= 20000):
                raise ValueError("ARCHIVE_MEMBER_REFUSED")
            expanded += entry.size
            if expanded > config["archive_bytes"]:
                raise ValueError("ARCHIVE_SIZE_REFUSED")
            digest, size = hashlib.sha256(), 0
            with archive.extractfile(entry) as stream:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    size += len(chunk)
            if size != entry.size:
                raise ValueError("MEMBER_SIZE_MISMATCH")
            files[entry.name] = {"path": entry.name, "bytes": size, "sha256": digest.hexdigest()}
    # The tar parser can stop at its end marker. Hash padding and all remaining
    # object bytes too, including bytes already prefetched into its own buffer.
    while reader.read(1024 * 1024):
        pass
    if (not files or reader.bytes != config["archive_bytes"]
            or reader.whole.hexdigest() != config["sha256"]):
        raise ValueError("ARCHIVE_SHA256_MISMATCH")
    for name in files:
        parent = name
        while "/" in parent:
            parent = parent.rsplit("/", 1)[0]
            if parent in files:
                raise ValueError("ARCHIVE_PATH_CONFLICT")
    return {"definition": {"sha256": config["sha256"], "archive_bytes": reader.bytes,
                           "files": [files[k] for k in sorted(files)]}, "parts": reader.records}


def main():
    config = json.loads(sys.argv[1])
    try:
        result = verify(config)
        result["request_id"] = config["request_id"]
        call(config, "oss-complete", result)
        print("ML_EXPD_OSS_IMPORT=ARCHIVED", flush=True)
        return 0
    except Exception as error:
        # Exception strings may contain a signed URL. Only fixed error classes
        # are recorded; failures never publish a usable dataset.
        text = str(error)
        code = text if (isinstance(error, ValueError) and 0 < len(text) <= 64
                        and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for c in text)) else "ARCHIVE_VERIFICATION_FAILED"
        try:
            call(config, "oss-failed", {"request_id": config["request_id"], "code": code})
        except Exception:
            pass
        print("ML_EXPD_OSS_IMPORT=FAILED", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
