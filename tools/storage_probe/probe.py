#!/usr/bin/env python3
"""Establish what the shared filesystem really does, using two zero-GPU jobs.

Run this same command as two jobs at once, one with --role a and one with --role b,
with the same --dir on the backend's shared storage. They meet through files there.
It only reads and writes inside DIR/ml-exp-storage-probe, and creates nothing else.

It answers what the code assumes and no test can show:
- does flock exclude across nodes, in both orders?
- do two clients see the same inode, device, mtime and ctime for one file?
- is the job root, so that read-only permissions do not stop it writing?
- how fast is stat, which decides how many files a stat check can cover?
- does the provider wrap the command in one shell string?
- when both sides race to create one name, does exactly one win, for each way of creating?
"""
import argparse
import ctypes
import errno
import fcntl
import json
import os
from pathlib import Path
import socket
import sys
import time

STAT_FILES = 2000
RACE_ROUNDS = 40
RENAME_NOREPLACE = 1


def create_exclusive(target, own):
    os.close(os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))


def make_directory(target, own):
    os.mkdir(target)


def hard_link(target, own):
    own.write_text("x")
    os.link(own, target)


def rename_no_replace(target, own):
    own.write_text("x")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.renameat2(-100, os.fsencode(own), -100, os.fsencode(target), RENAME_NOREPLACE):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


CREATORS = {"o_excl": create_exclusive, "mkdir": make_directory,
            "link": hard_link, "rename_noreplace": rename_no_replace}


def attempt(create, target, own):
    """What one try did: `won`, `lost`, or the errno that means this way does not work here."""
    try:
        create(target, own)
    except FileExistsError:
        return "lost"
    except OSError as error:
        return errno.errorcode.get(error.errno, str(error.errno))
    return "won"


def race(meeting, root, name, create):
    """Both sides create one name at an agreed moment; the two results are compared per round."""
    directory = root / f"race-{name}"
    directory.mkdir(exist_ok=True)
    mine = []
    for number in range(RACE_ROUNDS):
        start = time.time() + 0.5
        if meeting.role == "a":
            meeting.say(f"{name}-{number}-start", start)
        else:
            start = meeting.hear(f"{name}-{number}-start")
        time.sleep(max(0, start - time.time()))
        mine.append(attempt(create, directory / f"round-{number}",
                            directory / f"own-{meeting.role}-{number}"))
        meeting.say(f"{name}-{number}-done")
        meeting.hear(f"{name}-{number}-done")
    meeting.say(f"{name}-results", mine)
    return {"mine": mine, "theirs": meeting.hear(f"{name}-results")}


def race_finding(rounds):
    pairs = list(zip(rounds["mine"], rounds["theirs"]))
    both = sum(pair == ("won", "won") for pair in pairs)
    none = sum("won" not in pair for pair in pairs)
    other = sorted({word for pair in pairs for word in pair} - {"won", "lost"})
    if other:
        return f"UNSUPPORTED: {other}"
    if both or none:
        return f"NOT EXCLUSIVE: both won {both}, neither won {none} of {len(pairs)}"
    won = sum(pair[0] == "won" for pair in pairs)
    return f"exactly one wins in all {len(pairs)} (this side won {won})"


class Meeting:
    """Each side publishes named facts atomically and waits for the other's."""

    def __init__(self, root, role, timeout):
        self.root, self.role, self.deadline = root, role, time.monotonic() + timeout
        self.other = "b" if role == "a" else "a"

    def say(self, name, value=True):
        path = self.root / f"{self.role}.{name}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value))
        os.replace(temporary, path)

    def hear(self, name):
        path = self.root / f"{self.other}.{name}.json"
        while True:
            try:
                return json.loads(path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                if time.monotonic() > self.deadline:
                    raise TimeoutError(f"the other job never said {name}")
                time.sleep(0.2)


def snapshot(path):
    stat = os.stat(path)
    return {"inode": stat.st_ino, "device": stat.st_dev, "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}


def try_lock(path):
    """What a non-blocking exclusive lock does while the other side holds it."""
    with open(path, "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return "refused"
        except OSError as error:
            return f"error {error.errno}"
        fcntl.flock(lock, fcntl.LOCK_UN)
        return "acquired"


def lock_round(meeting, holder, path, name):
    """The holder keeps the lock while the other tries. `refused` means it excludes."""
    if meeting.role == holder:
        with open(path, "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            meeting.say(f"{name}-held")
            result = meeting.hear(f"{name}-tried")
            fcntl.flock(lock, fcntl.LOCK_UN)
        meeting.say(f"{name}-released")
        return result
    meeting.hear(f"{name}-held")
    result = try_lock(path)
    meeting.say(f"{name}-tried", result)
    meeting.hear(f"{name}-released")
    return result


def stat_rate(root):
    directory = root / f"stat-{socket.gethostname()}-{os.getpid()}"
    directory.mkdir()
    try:
        for index in range(STAT_FILES):
            (directory / str(index)).touch()
        start = time.monotonic()
        for index in range(STAT_FILES):
            os.stat(directory / str(index))
        return round(STAT_FILES / max(time.monotonic() - start, 1e-9))
    finally:
        for entry in directory.iterdir():
            entry.unlink()
        directory.rmdir()


def masked(argv):
    """The worker receives its token on the command line, and this report leaves the job."""
    return ["[redacted]" if index and argv[index - 1].endswith("-token") else word
            for index, word in enumerate(argv)]


def command_chain():
    """Who started this process. A shell given one joined string shows here as `sh -c`."""
    chain = []
    pid = os.getpid()
    while pid > 1 and len(chain) < 8:
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
            status = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            break
        chain.append(masked([part[:200] for part in raw.decode("utf-8", "replace").split("\0") if part][:8]))
        pid = int(status.rsplit(")", 1)[1].split()[1])
    return chain


def verdict(result):
    """Plain findings. A check on one host proves nothing about other nodes."""
    findings = {}
    if result["same_host"]:
        findings["nodes"] = "BOTH JOBS RAN ON ONE HOST: the lock and client results prove nothing"
    locks = [result["lock_held_by_a"], result["lock_held_by_b"]]
    findings["flock"] = ("excludes across clients" if locks == ["refused", "refused"]
                         else f"DOES NOT EXCLUDE: {locks}")
    file = result["same_file"]
    findings["inode"] = "same" if file["mine"]["inode"] == file["theirs"]["inode"] else "DIFFERS"
    findings["device"] = "same" if file["mine"]["device"] == file["theirs"]["device"] else "DIFFERS"
    for key in ("mtime_ns", "ctime_ns"):
        findings[key] = "same" if file["mine"][key] == file["theirs"][key] else "DIFFERS"
    findings["root"] = "runs as root" if result["euid"] == 0 else f"runs as uid {result['euid']}"
    findings["stat_per_second"] = result["stat_per_second"]
    for name, rounds in result["races"].items():
        findings[f"race_{name}"] = race_finding(rounds)
    return findings


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=os.environ.get("DATA_DIR"), type=Path,
                        help="shared directory; DATA_DIR unless given")
    parser.add_argument("--role", required=True, choices=["a", "b"])
    parser.add_argument("--session", default="default",
                        help="a new word for each pair of jobs, because old facts stay on disk")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--note", default="", help="pass words with a space to see how the provider quotes them")
    args = parser.parse_args(arguments)
    if args.dir is None:
        parser.error("--dir is required when DATA_DIR is not set")
    root = args.dir / "ml-exp-storage-probe" / args.session
    root.mkdir(parents=True, exist_ok=True)
    meeting = Meeting(root, args.role, args.timeout)
    meeting.say("hello", socket.gethostname())
    other_host = meeting.hear("hello")

    shared, lock = root / "shared.txt", root / "probe.lock"
    if args.role == "a":
        shared.write_text("x")
        meeting.say("shared-written")
        meeting.hear("shared-read")
    else:
        meeting.hear("shared-written")
        shared.read_text()
        meeting.say("shared-read")
    mine = snapshot(shared)
    meeting.say("stat", mine)
    theirs = meeting.hear("stat")

    result = {
        "role": args.role,
        "host": socket.gethostname(),
        "other_host": other_host,
        "same_host": other_host == socket.gethostname(),
        "uid": os.getuid(),
        "euid": os.geteuid(),
        "note": args.note,
        "argv": sys.argv[1:],
        "command_chain": command_chain(),
        "same_file": {"mine": mine, "theirs": theirs},
        "lock_held_by_a": lock_round(meeting, "a", lock, "lock-a"),
        "lock_held_by_b": lock_round(meeting, "b", lock, "lock-b"),
        "stat_per_second": stat_rate(root),
        "races": {name: race(meeting, root, name, create) for name, create in CREATORS.items()},
    }
    result["findings"] = verdict(result)
    meeting.say("result", result)
    print(json.dumps(result, indent=1))
    output = os.environ.get("OUTPUT_DIR")
    if output:
        Path(output, f"storage-probe-{args.role}.json").write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
