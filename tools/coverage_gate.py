#!/usr/bin/env python3
"""Run pytest and enforce independent repository line/branch coverage gates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "coverage.json"
DEFAULT_MIN_LINE = 100.0
DEFAULT_MIN_BRANCH = 95.0


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
        f"coverage gates: line={line:.1f}% (required {min_line:.1f}%), "
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-line", type=float, default=DEFAULT_MIN_LINE)
    parser.add_argument("--min-branch", type=float, default=DEFAULT_MIN_BRANCH)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.check_only:
        result = subprocess.run([
            sys.executable, "-m", "pytest", "-q",
            "--cov=experiment_control", "--cov-branch",
            "--cov-report=term-missing", "--cov-report=json:coverage.json",
        ], cwd=ROOT, check=False)
        if result.returncode:
            return result.returncode
    if not REPORT.is_file():
        print("coverage.json is missing", file=sys.stderr)
        return 2
    return check_report(REPORT, min_line=args.min_line, min_branch=args.min_branch)


if __name__ == "__main__":
    raise SystemExit(main())
