"""Run one local command and durably publish its exit result."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .manifest import atomic_write, utc_now


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    try:
        completed = subprocess.run(command, check=False)
        exit_code = completed.returncode
    except OSError as error:
        print(f"local command could not start: {type(error).__name__}", file=sys.stderr)
        exit_code = 127
    atomic_write(args.result, {
        "worker_pid": os.getpid(),
        "exit_code": exit_code,
        "finished_at": utc_now(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
