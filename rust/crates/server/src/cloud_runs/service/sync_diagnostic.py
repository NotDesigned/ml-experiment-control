"""Inspect one retained Attempt without running its worker or modifying storage."""
import errno
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import ssl
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

PART_PREFIX = "ML_EXPD_DIAGNOSTIC_PART="
CONTRACT = "sync-diagnostic.v1"
MAX_REPORT_BYTES = 32768
# PAI log lines are treated as limited to 2048 characters. With the prefix, digest,
# counters and room for a timestamp, a whole line stays under that. The API reads a
# 100-line tail per pod, and a report of MAX_REPORT_BYTES needs 19 lines.
CHUNK_BYTES = 1800


def components(path, absolute=False):
    if not isinstance(path, str) or path.startswith("/") != absolute:
        raise ValueError()
    parts = path[1:].split("/") if absolute else path.split("/")
    if not parts or any(p in ("", ".", "..") or any(ord(c) < 32 for c in p) for p in parts):
        raise ValueError()
    return parts


def open_directory(parts, parent=None):
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY) if parent is None else os.dup(parent)
    try:
        for part in parts:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def inspect(root, path, limit=None, lock=False):
    row = {"path": path}
    parent = fd = None
    data = None
    try:
        parts = components(path)
        parent = open_directory(parts[:-1], root)
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            row["status"] = "NOT_REGULAR"
            return row, None
        row.update(status="PRESENT", size=before.st_size, mode=oct(stat.S_IMODE(before.st_mode)),
                   uid=before.st_uid, gid=before.st_gid)
        if lock:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                row["lock"] = "FREE"
                fcntl.flock(fd, fcntl.LOCK_UN)
            except BlockingIOError:
                row["lock"] = "HELD"
            except OSError:
                row["lock"] = "UNAVAILABLE"
        if limit is not None:
            if before.st_size > limit:
                row["status"] = "METADATA_TOO_LARGE"
                return row, None
            with os.fdopen(os.dup(fd), "rb") as stream:
                data = stream.read(limit + 1)
            after = os.fstat(fd)
            if len(data) > limit or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns
            ):
                row["status"] = "CHANGED_DURING_READ"
                return row, None
            row["sha256"] = hashlib.sha256(data).hexdigest()
    except FileNotFoundError as error:
        row["status"] = "MISSING"
        row["errno"] = error.errno
    except PermissionError as error:
        row["status"] = "PERMISSION_DENIED"
        row["errno"] = error.errno
    except (ValueError, OSError) as error:
        row["status"] = "UNSAFE_PATH" if isinstance(error, ValueError) or error.errno in (
            errno.ELOOP, errno.ENOTDIR
        ) else "STORAGE_ERROR"
        if isinstance(error, OSError):
            row["errno"] = error.errno
    finally:
        if fd is not None:
            os.close(fd)
        if parent is not None:
            os.close(parent)
    return row, data


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def transport(channel):
    address = urllib.parse.urlsplit(channel["url"])
    if address.scheme not in ("http", "https") or address.query or address.fragment or address.username:
        raise ValueError()
    if not address.path.startswith("/api/worker-exchange/"):
        raise ValueError()
    request = urllib.request.Request(channel["url"] + "/status", headers={
        "Authorization": "Bearer " + channel["token"]
    }, method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=10) as response:
            raw = response.read(32769)
            if len(raw) > 32768:
                return {"stage": "RESPONSE_TOO_LARGE", "http_status": response.status}
            value = json.loads(raw)
            if value.get("contract") != "sync-diagnostic-status.v1":
                return {"stage": "RESPONSE_INVALID", "http_status": response.status}
            result = {"stage": "HTTP_OK", "http_status": response.status, "exchange": {}, "artifacts": {}}
            for field in ("observed", "complete"):
                if type(value.get("exchange", {}).get(field)) is bool:
                    result["exchange"][field] = value["exchange"][field]
            sequence = value.get("exchange", {}).get("sequence")
            if type(sequence) is int and sequence >= 0:
                result["exchange"]["sequence"] = sequence
            phase = value.get("exchange", {}).get("phase")
            if phase in ("PREPARED", "STARTED", "RUNNING", "FINISHED", "FAILED", "CANCELLED"):
                result["exchange"]["phase"] = phase
            for field in ("prepared", "published", "released"):
                if type(value.get("artifacts", {}).get(field)) is bool:
                    result["artifacts"][field] = value["artifacts"][field]
            return result
    except urllib.error.HTTPError as error:
        return {"stage": "HTTP_ERROR", "http_status": error.code}
    except (urllib.error.URLError, OSError) as error:
        cause = error.reason if isinstance(error, urllib.error.URLError) else error
        stage = "NETWORK_ERROR"
        if isinstance(cause, socket.gaierror):
            stage = "DNS_ERROR"
        elif isinstance(cause, ssl.SSLError):
            stage = "TLS_ERROR"
        elif isinstance(cause, (TimeoutError, socket.timeout)):
            stage = "TIMEOUT"
        elif isinstance(cause, ConnectionError):
            stage = "TCP_ERROR"
        return {"stage": stage}
    except (ValueError, TypeError):
        return {"stage": "RESPONSE_INVALID"}


def diagnose(config):
    if config.get("contract") == "data-diagnostic.v1":
        return diagnose_data(config)
    if config.get("contract") != CONTRACT or not re.fullmatch("action-[0-9a-f]{16}", config["action"]):
        raise ValueError()
    if not re.fullmatch("[0-9a-f]{32}", config["request_id"]):
        raise ValueError()
    package = config["package"]
    report = {"contract": CONTRACT, "action": config["action"], "request_id": config["request_id"], "checks": []}
    root = open_directory(components(package["config"]["attempt_root"], True))
    metadata = {}
    try:
        for path, limit, lock in [
            (".managed-launch.lock", 16384, True),
            ("launch-manifest.json", 262144, False),
            ("managed-receipt.json", 262144, False),
            ("runtime-environment.json", 262144, False),
            (".cloud-exchange.json", 3 * 1024 * 1024, False),
            (".cloud-artifacts/owner", 128, False),
            (".cloud-artifacts/.lock", None, True),
            (".cloud-artifacts/bundle.json", 8 * 1024 * 1024, False),
            (".cloud-artifacts/archive.tar", None, False),
        ]:
            row, raw = inspect(root, path, limit, lock)
            report["checks"].append(row)
            if raw is not None:
                metadata[path] = raw
        output = package["config"].get("output_directory") or package["config"]["attempt_root"] + "/output"
        prefix = package["config"]["attempt_root"] + "/"
        if not output.startswith(prefix):
            raise ValueError()
        for path in (package.get("artifacts") or {}).get("required_files", []):
            components(path)
            row, _ = inspect(root, output[len(prefix):] + "/" + path)
            report["checks"].append(row)
    finally:
        os.close(root)
    checks = {row["path"]: row for row in report["checks"]}
    comparisons = [
        (".cloud-artifacts/owner", lambda raw: raw.decode() == config["package_sha256"]),
        ("launch-manifest.json", lambda raw: hashlib.sha256(raw).hexdigest() == package["config"]["local_binding"]["sha256"] and json.loads(raw) == package["manifest"]),
        (".cloud-artifacts/bundle.json", lambda raw: json.loads(raw).get("intent_sha256") == config["package_sha256"] and json.loads(raw).get("config_sha256") == config["config_sha256"]),
        (".managed-launch.lock", lambda raw: all(json.loads(line).get("binding_sha256") == config["config_binding_sha256"] for line in raw.splitlines()) and bool(raw)),
    ]
    for path, compare in comparisons:
        if path in metadata:
            try:
                checks[path]["identity"] = "MATCH" if compare(metadata[path]) else "MISMATCH"
            except (ValueError, KeyError, TypeError):
                checks[path]["identity"] = "INVALID_METADATA"
    report["transport"] = transport(config["channel"])
    return report


def diagnose_data(config):
    if not re.fullmatch("[0-9a-f]{32}", config["request_id"]):
        raise ValueError()
    if not re.fullmatch("asset\\.[0-9a-f]{64}", config["asset_id"]):
        raise ValueError()
    if len(components(config["project"])) != 1:
        raise ValueError()
    report = {"contract": "data-diagnostic.v1", "project": config["project"],
              "asset_id": config["asset_id"], "request_id": config["request_id"], "checks": []}
    root = open_directory(components(config["root"], True))
    try:
        space = os.fstatvfs(root)
        root_stat = os.fstat(root)
        report["root"] = {"mode": oct(stat.S_IMODE(root_stat.st_mode)),
                          "uid": root_stat.st_uid, "gid": root_stat.st_gid}
        report["capacity"] = {"available_bytes": space.f_bavail * space.f_frsize,
                              "available_inodes": space.f_favail,
                              "user_quota_observed": False}
        slot = config["project"] + "/" + config["asset_id"] + "/"
        metadata = {}
        for path, limit, lock in [(".native-data.json", 4096, False),
                                  (".native-data.lock", None, True),
                                  (slot + "state.json", 8 * 1024 * 1024, False)]:
            row, raw = inspect(root, path, limit, lock)
            report["checks"].append(row)
            if raw is not None:
                metadata[path] = raw
        for row in report["checks"]:
            raw = metadata.get(row["path"])
            if raw is None:
                continue
            try:
                saved = json.loads(raw)
                if row["path"] == ".native-data.json":
                    device, inode = saved["identity"]
                    row["inode_matches"] = inode == root_stat.st_ino
                    row["device_matches"] = device == root_stat.st_dev
                    matches = (saved.get("contract") == "native-dataset.v1" and
                               saved.get("target") == config["target"] and row["inode_matches"])
                else:
                    matches = (saved.get("contract") == "native-dataset.v1" and
                               saved.get("project") == config["project"] and
                               saved.get("asset_id") == config["asset_id"] and
                               saved.get("target") == config["target"] and
                               saved.get("definition") == config["definition"])
                    report["recorded_parts"] = len(saved.get("parts", {}))
                    report["ready_recorded"] = saved.get("ready") is not None
                row["identity"] = "MATCH" if matches else "MISMATCH"
            except (ValueError, TypeError, KeyError):
                row["identity"] = "INVALID_METADATA"
        count = (config["definition"]["archive_bytes"] + 16777215) // 16777216
        # Inspect sizes only, including a last-written but not yet committed part.
        report["parts"] = []
        report["total_parts"] = count
        report["parts_truncated"] = count > 128
        for number in range(min(count, 128)):
            row, _ = inspect(root, slot + "parts/" + str(number))
            row.pop("path")
            report["parts"].append({"number": number, **row})
    finally:
        os.close(root)
    return report


def encoded(report):
    # A log store may trim a line's ends and a chunk can end anywhere, so the text
    # holds no whitespace. json.dumps already escapes everything else.
    return json.dumps(report, separators=(",", ":"), ensure_ascii=True).replace(" ", "\\u0020")


def emit(text):
    digest = hashlib.sha256(text.encode()).hexdigest()
    chunks = [text[i:i + CHUNK_BYTES] for i in range(0, len(text), CHUNK_BYTES)]
    for index, chunk in enumerate(chunks):
        print("{}{}:{}:{}:{}".format(PART_PREFIX, digest, index, len(chunks), chunk), flush=True)


def main():
    start = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(270)
    config = {}
    try:
        config = json.loads(sys.argv[1])
        report = diagnose(config)
    except Exception as error:
        report = {"contract": CONTRACT, "error": "DIAGNOSTIC_UNAVAILABLE"}
        if isinstance(config, dict):
            if config.get("contract") in (CONTRACT, "data-diagnostic.v1"):
                report["contract"] = config["contract"]
            for key, pattern in [("request_id", "[0-9a-f]{32}"),
                                 ("action", "action-[0-9a-f]{16}"),
                                 ("asset_id", "asset\\.[0-9a-f]{64}"),
                                 ("project", "[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")]:
                value = config.get(key)
                if isinstance(value, str) and re.fullmatch(pattern, value):
                    report[key] = value
        if isinstance(error, OSError):
            report["errno"] = error.errno
    report["elapsed_ms"] = round((time.monotonic() - start) * 1000)
    text = encoded(report)
    if len(text) > MAX_REPORT_BYTES:
        small = {key: report[key] for key in ("contract", "action", "project", "asset_id", "request_id") if key in report}
        small["error"] = "REPORT_TOO_LARGE"
        text = encoded(small)
    emit(text)


if __name__ == "__main__":
    main()
