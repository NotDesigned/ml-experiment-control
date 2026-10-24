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


def test_successful_shard_saves_evidence_but_never_gates_incomplete_coverage(tmp_path, monkeypatch):
    import subprocess
    import ci_shards
    import coverage_gate as gate
    monkeypatch.setattr(gate, 'ROOT', tmp_path)
    monkeypatch.setattr(gate, 'partition', lambda root, count: [['server/tests/test_one.py']])
    monkeypatch.setattr(ci_shards, 'checkout_tree', lambda root: 'a' * 40)
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        (tmp_path / '.coverage.daemon').write_bytes(b'coverage data')
        (tmp_path / 'daemon-tests.xml').write_text('<testsuite/>')
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(gate.subprocess, 'run', run)
    assert gate.main(['--suite', 'daemon', '--shard-index', '0', '--shard-count', '1']) == 0
    assert 'server/tests/test_one.py' in calls[0] and 'worker/tests' not in calls[0]
    manifest = json.loads((tmp_path / '.ops/shard/daemon-shard.json').read_text())
    assert manifest['pytest_passed'] is True and manifest['files'] == ['server/tests/test_one.py']
    assert (tmp_path / '.ops/shard/.coverage.daemon').read_bytes() == b'coverage data'
    monkeypatch.setattr(gate.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 1))
    assert gate.main(['--suite', 'daemon', '--shard-index', '0', '--shard-count', '1']) == 1


def test_aggregate_requires_valid_data_and_still_enforces_independent_gates(tmp_path, monkeypatch):
    import coverage_gate as gate
    monkeypatch.setattr(gate, 'ROOT', tmp_path)
    def incomplete(*args):
        raise ValueError('missing shard')
    monkeypatch.setattr(gate, 'combine_daemon', incomplete)
    assert gate.main(['--suite', 'daemon', '--combine-dir', str(tmp_path), '--shard-count', '2']) == 2


def test_sharding_cannot_skip_other_suite_gates():
    import pytest
    import coverage_gate as gate
    for args in (['--shard-count', '2'], ['--suite', 'client', '--shard-index', '0', '--shard-count', '2'],
                 ['--suite', 'daemon', '--shard-index', '2', '--shard-count', '2'],
                 ['--suite', 'daemon', '--shard-index', '0', '--shard-count', '2', '--check-only']):
        with pytest.raises(SystemExit):
            gate.main(args)


def test_real_coverage_data_combines_complementary_branches_and_rejects_missing_ones(tmp_path, monkeypatch):
    import coverage
    import coverage_gate as gate
    roots = [tmp_path / 'server/src/ml_exp_server', tmp_path / 'worker/src/ml_exp_worker']
    for root in roots:
        root.mkdir(parents=True)
        (root / '__init__.py').write_text('')
        (root / 'example.py').write_text('def choose(flag):\n    if flag:\n        return 1\n    return 2\n')
    sources = []
    for index, flag in enumerate((True, False)):
        source = tmp_path / ('.coverage.fixture-' + str(index))
        measurement = coverage.Coverage(data_file=str(source), branch=True, source=list(map(str, roots)))
        measurement.start()
        for root in roots:
            code = root / 'example.py'
            namespace = {}
            exec(compile(code.read_text(), str(code), 'exec'), namespace)
            namespace['choose'](flag)
        measurement.stop()
        measurement.save()
        sources.append(source)
    monkeypatch.setattr(gate, 'ROOT', tmp_path)
    monkeypatch.setattr(gate, 'COMBINED_REPORT', tmp_path / 'combined.json')
    monkeypatch.setattr(gate, 'DAEMON_REPORT', tmp_path / 'server.json')
    monkeypatch.setattr(gate, 'WORKER_REPORT', tmp_path / 'worker.json')
    monkeypatch.setattr(gate, 'validated_data', lambda *args: sources)
    arguments = ['--suite', 'daemon', '--combine-dir', str(tmp_path), '--shard-count', '2']
    assert gate.main(arguments) == 0
    # Each partial data file leaves a branch untested. Old combined data must not
    # fill that hole on a subsequent aggregate attempt.
    monkeypatch.setattr(gate, 'validated_data', lambda *args: sources[:1])
    assert gate.main(arguments) == 1
