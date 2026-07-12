"""Independent coverage math and generated CLI documentation contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from coverage_gate import (  # noqa: E402
    DEFAULT_MIN_BRANCH,
    DEFAULT_MIN_LINE,
    check_report,
    coverage_percentages,
)
from generate_cli_reference import main as cli_docs_main  # noqa: E402


def report(tmp_path, *, lines=(90, 100), branches=(80, 100)):
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps({"totals": {
        "covered_lines": lines[0], "num_statements": lines[1],
        "covered_branches": branches[0], "num_branches": branches[1],
    }}))
    return path


def test_coverage_dimensions_are_independent_and_fail_closed(tmp_path, capsys):
    path = report(tmp_path, lines=(89, 100), branches=(81, 100))
    assert coverage_percentages(json.loads(path.read_text())) == (89.0, 81.0)
    assert check_report(path, min_line=90, min_branch=80) == 1
    assert "line" in capsys.readouterr().err

    path = report(tmp_path, lines=(91, 100), branches=(79, 100))
    assert check_report(path, min_line=90, min_branch=80) == 1
    assert "branch" in capsys.readouterr().err


def test_empty_modules_and_current_generated_cli_reference(tmp_path):
    assert (DEFAULT_MIN_LINE, DEFAULT_MIN_BRANCH) == (100.0, 100.0)
    assert check_report(
        report(tmp_path, lines=(0, 0), branches=(0, 0)),
        min_line=100, min_branch=100,
    ) == 0
    assert cli_docs_main(["--check"]) == 0
