"""Fixed stdlib helper for one explicitly authorized managed Run directory."""
from __future__ import annotations

import http.client
import json
import os
from pathlib import PurePosixPath
import re
import stat
import sys
import time
from urllib.parse import urlsplit


IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def checked_target(payload):
    project, run = payload["project"], payload["run_id"]
    root, target = payload["project_root"], payload["target"]
    if not all(isinstance(value, str) and IDENTITY.fullmatch(value) for value in (project, run)):
        raise ValueError("invalid Run identity")
    for value in (root, target):
        path = PurePosixPath(value)
        if (not path.is_absolute() or path.as_posix() != value or ".." in path.parts
                or "\\" in value or "\x00" in value):
            raise ValueError("invalid managed storage path")
    if (len(PurePosixPath(root).parts) < 3 or PurePosixPath(root).name != project
            or target != root + "/runs/" + run):
        raise ValueError("target is not the exact managed Run")
    if not isinstance(payload["plan_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", payload["plan_sha256"]):
        raise ValueError("invalid deletion plan digest")
    if payload.get("mode", "delete") not in {"delete", "check"}:
        raise ValueError("invalid deletion mode")
    if (not isinstance(payload.get("deletion_id"), str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", payload["deletion_id"])):
        raise ValueError("invalid deletion operation identity")
    return target


def mountpoints():
    """Read mount identities, including same-device bind mounts."""
    with open("/proc/self/mountinfo", encoding="utf-8") as stream:
        return {re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), line.split()[4])
                for line in stream if line.strip()}


def fingerprint(value):
    return value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode)


def remove_tree(target, *, seconds=600, check=False, clock=time.monotonic, mounts=mountpoints):
    """Internal fd-safe primitive; callers must bind their canonical target."""
    path = PurePosixPath(target)
    if (not path.is_absolute() or path.as_posix() != target or len(path.parts) < 3
            or ".." in path.parts or "\\" in target or "\x00" in target):
        raise ValueError("invalid deletion target")
    if type(seconds) is not int or not 1 <= seconds <= 900:
        raise ValueError("invalid deletion time budget")
    deadline = clock() + seconds

    def bounded():
        if clock() >= deadline:
            raise TimeoutError("managed Run deletion deadline exceeded")

    def unmounted():
        # Recheck before each filesystem mutation so a later bind mount is
        # refused too; the device check below also protects each opened child.
        if any(path == target or path.startswith(target + "/") for path in mounts()):
            raise ValueError("managed Run contains a mount")

    receipt = {"target_absent": True, "files_deleted": 0, "bytes_deleted": 0}
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent = os.open("/", flags)
    try:
        for part in PurePosixPath(target).parts[1:-1]:
            try:
                child = os.open(part, flags, dir_fd=parent)
            except FileNotFoundError:
                return receipt
            os.close(parent)
            parent = child
        name = PurePosixPath(target).name
        try:
            original = os.stat(name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return receipt
        if check:
            raise ValueError("managed Run data is still present")
        if not stat.S_ISDIR(original.st_mode):
            raise ValueError("managed Run root is not a directory")
        unmounted()
        root = os.open(name, flags, dir_fd=parent)
        try:
            if fingerprint(os.fstat(root)) != fingerprint(original):
                raise ValueError("managed Run root changed")
            device = original.st_dev

            def walk(fd, *, remove):
                bounded()
                current = os.fstat(fd)
                if current.st_dev != device:
                    raise ValueError("managed Run contains another filesystem")
                entries = os.listdir(fd)
                if remove:
                    bounded(); unmounted()
                    # Published checkpoint generations are sealed 0555.
                    os.fchmod(fd, stat.S_IMODE(current.st_mode) | stat.S_IWUSR | stat.S_IXUSR)
                for entry in entries:
                    bounded()
                    before = os.stat(entry, dir_fd=fd, follow_symlinks=False)
                    if before.st_dev != device:
                        raise ValueError("managed Run contains another filesystem")
                    if stat.S_ISDIR(before.st_mode):
                        child = os.open(entry, flags, dir_fd=fd)
                        try:
                            if fingerprint(os.fstat(child)) != fingerprint(before):
                                raise ValueError("managed Run directory changed")
                            walk(child, remove=remove)
                        finally:
                            os.close(child)
                        if remove:
                            bounded(); unmounted()
                            if fingerprint(os.stat(entry, dir_fd=fd, follow_symlinks=False)) != fingerprint(before):
                                raise ValueError("managed Run directory changed")
                            os.rmdir(entry, dir_fd=fd)
                    elif stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
                        if remove:
                            bounded(); unmounted()
                            after = os.stat(entry, dir_fd=fd, follow_symlinks=False)
                            if fingerprint(after) != fingerprint(before) or after.st_size != before.st_size:
                                raise ValueError("managed Run file changed")
                            os.unlink(entry, dir_fd=fd)
                            receipt["files_deleted"] += 1
                            if stat.S_ISREG(before.st_mode):
                                receipt["bytes_deleted"] += before.st_size
                    else:
                        raise ValueError("managed Run contains a special file")

            # Refuse an existing unsafe tree before deleting its first file.
            walk(root, remove=False)
            walk(root, remove=True)
        finally:
            os.close(root)
        bounded(); unmounted()
        if fingerprint(os.stat(name, dir_fd=parent, follow_symlinks=False)) != fingerprint(original):
            raise ValueError("managed Run root changed")
        os.rmdir(name, dir_fd=parent)
        os.fsync(parent)
        try:
            os.stat(name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return receipt
        raise ValueError("managed Run target absence is unconfirmed")
    finally:
        os.close(parent)


def delete(payload, *, clock=time.monotonic, mounts=mountpoints):
    target = checked_target(payload)
    counts = remove_tree(target, seconds=payload.get("seconds", 600), check=payload.get("mode") == "check",
                         clock=clock, mounts=mounts)
    return {"contract": "run-deletion.v1", "deletion_id": payload["deletion_id"],
            "project": payload["project"], "run_id": payload["run_id"],
            "plan_sha256": payload["plan_sha256"], "target": target, "status": "DELETED", **counts}


def report(payload, receipt, *, connection=http.client.HTTPSConnection):
    endpoint = urlsplit(payload["receipt_url"])
    if (endpoint.scheme != "https" or not endpoint.hostname or endpoint.username is not None
            or endpoint.password is not None or endpoint.fragment or endpoint.query
            or not endpoint.path.endswith("/api/run-deletion-transfers/" + payload["deletion_id"] + "/receipt")):
        raise ValueError("invalid deletion receipt endpoint")
    token = payload["capability"]
    if not isinstance(token, str) or not 32 <= len(token) <= 128 or any(c.isspace() for c in token):
        raise ValueError("invalid deletion receipt capability")
    body = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    last = None
    for _ in range(3):
        client = connection(endpoint.hostname, endpoint.port or 443, timeout=10)
        try:
            client.request("POST", endpoint.path, body=body,
                           headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
            response = client.getresponse()
            response.read(65537)
            if 200 <= response.status < 300:
                return
            last = ValueError("deletion receipt was not accepted")
        except (OSError, http.client.HTTPException) as error:
            last = error
        finally:
            client.close()
    raise ValueError("deletion receipt delivery requires reconciliation") from last


def main(payload=None):
    if payload is None:
        raw = sys.stdin.buffer.read(65537)
        if len(raw) > 65536:
            raise ValueError("deletion instruction is too large")
        payload = json.loads(raw)
    receipt = delete(payload)
    if payload.get("receipt_url"):
        report(payload, receipt)
    else:
        print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
