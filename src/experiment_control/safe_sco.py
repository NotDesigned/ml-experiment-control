#!/usr/bin/env python3
"""Sanitize SCO output and normalize SenseCore scheduler states.

Only explicitly allowlisted job fields are emitted.  Error and log text can be
passed through ``redact-lines`` without first exposing the raw response to the
controller.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, TextIO

from experiment_control.backends.sensecore import normalize_state


SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:secret|token|password|passwd|credential|access[_-]?key(?:[_-]?(?:id|secret))?"
    r"|api[_-]?key|proxy|authorization|cookie)[\w.-]*\b[\s\"']*[=:][\s\"']*)"
    r"([^\s,;\"']+)"
)
BEARER_RE = re.compile(r"(?i)(\b(?:authorization\s*:\s*)?bearer\s+)[^\s,;]+")
URL_USERINFO_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://)[^/@\s]+@")
SENSITIVE_QUERY_RE = re.compile(
    r"(?i)([?&](?:access[_-]?key(?:[_-]?(?:id|secret))?|api[_-]?key|secret|token|signature)=)"
    r"[^&#\s]+"
)

def safe_text(value: Any) -> Any:
    """Redact allowlisted string values as a second line of defense."""
    return redact_line(value) if isinstance(value, str) else value


def job_summary(job: dict[str, Any]) -> dict[str, Any]:
    """Return only non-secret fields needed to observe an exact job."""
    roles = job.get("roles") or []
    first_role = roles[0] if roles and isinstance(roles[0], dict) else {}
    specs = first_role.get("resource_spec") or []
    first_spec = specs[0] if specs and isinstance(specs[0], dict) else {}
    pool = job.get("resource_pool") or {}
    mounts = job.get("mount") or []
    return {
        "name": safe_text(job.get("name")),
        "display_name": safe_text(job.get("display_name")),
        "state": job.get("state"),
        "normalized_state": normalize_state(job.get("state")),
        "create_time": job.get("create_time"),
        "pool": safe_text(pool.get("name")) if isinstance(pool, dict) else None,
        "image": safe_text(first_role.get("image_path")),
        "spec": safe_text(first_spec.get("name")) if isinstance(first_spec, dict) else None,
        "mounts": [
            {
                "id": safe_text(mount.get("id")),
                "subdir": safe_text(mount.get("subdir")),
                "mount_path": safe_text(mount.get("mount_path")),
            }
            for mount in mounts
            if isinstance(mount, dict)
        ],
    }


def redact_line(line: str) -> str:
    """Redact common credential forms while preserving diagnostic context."""
    line = URL_USERINFO_RE.sub(r"\1<redacted>@", line)
    line = BEARER_RE.sub(r"\1<redacted>", line)
    line = SENSITIVE_QUERY_RE.sub(r"\1<redacted>", line)
    return SECRET_ASSIGNMENT_RE.sub(r"\1<redacted>", line)


def read_json(stream: TextIO, *, empty_list: bool = False) -> Any:
    """Read JSON without echoing malformed or potentially sensitive input."""
    raw = stream.read()
    # SCO v1.2 prints this exact success sentinel to stdout even with
    # ``-o json`` when an exact-name list has no matches. Accept only the
    # observed fixed phrase; every other non-JSON response still fails closed.
    if empty_list and raw.strip().casefold() in {"", "no jobs found"}:
        return []
    try:
        return json.loads(raw)
    except Exception:
        raise SystemExit("safe_sco: input was not valid JSON; raw response suppressed")


def write_json(value: Any, stream: TextIO) -> None:
    """Write stable single-line JSON for controller parsing."""
    json.dump(value, stream, ensure_ascii=False, sort_keys=True)
    stream.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("job-summary", "job-list", "redact-lines", "normalize-state"),
    )
    parser.add_argument("value", nargs="?")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested sanitizer mode and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "normalize-state":
        if args.value is None:
            parser.error("normalize-state requires a state value")
        print(normalize_state(args.value))
        return 0
    if args.mode == "redact-lines":
        for line in sys.stdin:
            sys.stdout.write(redact_line(line))
        return 0

    payload = read_json(sys.stdin, empty_list=args.mode == "job-list")
    if args.mode == "job-summary":
        if not isinstance(payload, dict):
            raise SystemExit("safe_sco: expected one JSON job object")
        write_json(job_summary(payload), sys.stdout)
        return 0
    if not isinstance(payload, list):
        raise SystemExit("safe_sco: expected a JSON job array")
    write_json([job_summary(job) for job in payload if isinstance(job, dict)], sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
