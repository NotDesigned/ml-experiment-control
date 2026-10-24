#!/usr/bin/env python3
"""Run pytest and enforce independent repository line/branch coverage gates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "coverage.json"
DAEMON_REPORT = ROOT / "server-coverage.json"
WORKER_REPORT = ROOT / "worker-coverage.json"
COMBINED_REPORT = ROOT / "server-worker-coverage.json"
CLIENT_REPORT = ROOT / "client-coverage.json"
DEFAULT_MIN_LINE = 100.0
DEFAULT_MIN_BRANCH = 100.0


def coverage_percentages(payload: dict[str, Any]) -> tuple[float, float]:
    totals = payload["totals"]
    statements = int(totals["num_statements"])
    branches = int(totals["num_branches"])
    line = 100.0 * int(totals["covered_lines"]) / statements if statements else 100.0
    branch = 100.0 * int(totals["covered_branches"]) / branches if branches else 100.0
    return line, branch


def check_report(path: Path, *, min_line: float, min_branch: float) -> int:
    line, branch = coverage_percentages(json.loads(path.read_text(encoding="utf-8")))
    print(
        f"{path.name} gates: line={line:.1f}% (required {min_line:.1f}%), "
        f"branch={branch:.1f}% (required {min_branch:.1f}%)"
    )
    failed = [
        name for name, actual, required in (
            ("line", line, min_line), ("branch", branch, min_branch),
        ) if actual < required
    ]
    if failed:
        print(f"coverage gate failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def split_daemon_report() -> None:
    """Gate server and worker separately without rerunning their shared flows."""
    payload = json.loads(COMBINED_REPORT.read_text(encoding="utf-8"))
    for package, path in (("ml_exp_server", DAEMON_REPORT), ("ml_exp_worker", WORKER_REPORT)):
        files = {name: row for name, row in payload["files"].items() if package in Path(name).parts}
        if not files:
            raise ValueError(f"coverage did not measure {package}")
        totals = {key: sum(row["summary"][key] for row in files.values())
                  for key in ("num_statements", "covered_lines", "num_branches", "covered_branches")}
        path.write_text(json.dumps({"meta": payload.get("meta", {}), "files": files, "totals": totals}),
                        encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-line", type=float, default=DEFAULT_MIN_LINE)
    parser.add_argument("--min-branch", type=float, default=DEFAULT_MIN_BRANCH)
    parser.add_argument(
        "--suite", choices=("core", "daemon", "client"), default="core",
        help="coverage suite to execute and verify",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1,
                        help="independent pytest workers (loadfile); coverage is combined before gating")
    args = parser.parse_args(argv)
    if args.suite == "daemon":
        report = DAEMON_REPORT
        pytest_args = [
            "server/tests", "worker/tests", "-q", "--cov=ml_exp_server", "--cov=ml_exp_worker", "--cov-branch",
            "--cov-report=term-missing:skip-covered",
            "--cov-report=json:server-worker-coverage.json",
        ]
    elif args.suite == "client":
        report = CLIENT_REPORT
        pytest_args = ["client/tests", "-q", "--cov=ml_exp_client", "--cov-branch",
                       "--cov-report=term-missing:skip-covered", "--cov-report=json:client-coverage.json"]
    else:
        report = REPORT
        pytest_args = [
            "-q", "--cov=experiment_control", "--cov-branch",
            "--cov-report=term-missing", "--cov-report=json:coverage.json",
        ]
    pytest_args += ["--durations=20", "--junitxml=" + (args.suite + "-tests.xml")]
    if args.workers > 1:
        pytest_args += ["-n", str(args.workers), "--dist=loadfile"]
    if not args.check_only:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", *pytest_args],
            cwd=ROOT, check=False, env={**os.environ, "COVERAGE_FILE": str(ROOT / (".coverage." + args.suite))},
        )
        if result.returncode:
            return result.returncode
    if args.suite == "daemon":
        try:
            split_daemon_report()
        except (OSError, KeyError, ValueError) as error:
            print(f"cannot verify independent server/worker coverage: {error}", file=sys.stderr)
            return 2
        results = [check_report(path, min_line=args.min_line, min_branch=args.min_branch)
                   for path in (DAEMON_REPORT, WORKER_REPORT)]
        return max(results)
    if not report.is_file():
        print(f"{report.name} is missing", file=sys.stderr)
        return 2
    return check_report(report, min_line=args.min_line, min_branch=args.min_branch)


if __name__ == "__main__":
    raise SystemExit(main())
