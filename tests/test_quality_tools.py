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


def test_empty_modules_and_strict_default_gates(tmp_path):
    assert (DEFAULT_MIN_LINE, DEFAULT_MIN_BRANCH) == (100.0, 100.0)
    assert check_report(
        report(tmp_path, lines=(0, 0), branches=(0, 0)),
        min_line=100, min_branch=100,
    ) == 0


def test_parallel_daemon_report_still_enforces_branch_gate(tmp_path, monkeypatch):
    import subprocess
    import coverage_gate as gate
    path = tmp_path / 'combined.json'
    summary = {'covered_lines': 100, 'num_statements': 100,
               'covered_branches': 99, 'num_branches': 100}
    path.write_text(json.dumps({'files': {
        'server/src/ml_exp_server/service.py': {'summary': summary},
        'worker/src/ml_exp_worker/worker.py': {'summary': {**summary, 'covered_branches': 100}},
    }}))
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    monkeypatch.setattr(gate,'COMBINED_REPORT',path)
    monkeypatch.setattr(gate,'DAEMON_REPORT',tmp_path / 'server.json')
    monkeypatch.setattr(gate,'WORKER_REPORT',tmp_path / 'worker.json')
    calls=[]
    def run(argv,**kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv,0)
    monkeypatch.setattr(gate.subprocess,'run',run)
    assert gate.main(['--suite','daemon','--workers','2'])==1
    assert calls[0][-3:]==['-n','2','--dist=loadfile']
    assert '--cov=ml_exp_worker' in calls[0]
    # Server success cannot mask incomplete worker coverage, nor an omitted package.
    payload = json.loads(path.read_text())
    payload['files']['server/src/ml_exp_server/service.py']['summary']['covered_branches'] = 100
    payload['files']['worker/src/ml_exp_worker/worker.py']['summary']['covered_branches'] = 99
    path.write_text(json.dumps(payload))
    assert gate.main(['--suite','daemon','--check-only']) == 1
    del payload['files']['worker/src/ml_exp_worker/worker.py']
    path.write_text(json.dumps(payload))
    assert gate.main(['--suite','daemon','--check-only']) == 2
