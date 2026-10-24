"""CI partitions cannot omit, duplicate or import tests from artifact evidence."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import ci_shards as shards


@pytest.fixture
def tree(tmp_path, monkeypatch):
    for name in ("server/tests/test_slow.py", "server/tests/test_fast.py", "worker/tests/test_worker.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def test_case(): pass\n")
    timings = tmp_path / "timings.json"
    timings.write_text(json.dumps({"server/tests/test_slow.py": 100}))
    monkeypatch.setattr(shards, "TIMINGS", timings)
    monkeypatch.setattr(shards, "checkout_tree", lambda root: "a" * 40)
    return tmp_path


def evidence(tree, count=2):
    directory = tree / "incoming"
    for index, files in enumerate(shards.partition(tree, count)):
        manifest = directory / str(index) / "daemon-shard.json"
        shards.write_manifest(tree, index, count, files, manifest)
        manifest.with_name(".coverage.daemon").write_bytes(b"data, never Python")
    return directory


def test_partitions_are_deterministic_complete_and_include_new_files(tree):
    first = shards.partition(tree, 2)
    assert first == shards.partition(tree, 2)
    assert sorted(name for group in first for name in group) == shards.test_files(tree)
    assert first[0] == ["server/tests/test_slow.py"]
    (tree / "worker/tests/new_test.py").write_text("def test_new(): pass\n")
    groups = shards.partition(tree, 2)
    assert sum("worker/tests/new_test.py" in group for group in groups) == 1
    assert len({name for group in groups for name in group}) == 4
    with pytest.raises(ValueError, match="at least one"):
        shards.partition(tree, 5)
    with pytest.raises(ValueError, match="between"):
        shards.partition(tree, 0)


def test_all_exact_partitions_are_required_before_coverage_combines(tree):
    directory = evidence(tree)
    sources = shards.validated_data(tree, directory, 2)
    assert len(sources) == 2 and all(path.name == ".coverage.daemon" for path in sources)
    manifest = directory / "1/daemon-shard.json"
    original = manifest.read_text()
    manifest.unlink()
    with pytest.raises(ValueError, match="missing or extra"):
        shards.validated_data(tree, directory, 2)
    manifest.write_text(original)
    extra = directory / "extra/daemon-shard.json"
    extra.parent.mkdir()
    extra.write_text(original)
    with pytest.raises(ValueError, match="missing or extra"):
        shards.validated_data(tree, directory, 2)


@pytest.mark.parametrize("change", [
    {"index": 0}, {"index": True}, {"count": 3}, {"tree": "b" * 40},
    {"pytest_passed": False}, {"files": []}, {"files": ["unexpected.py"]},
])
def test_mismatched_or_duplicated_manifest_fails_closed(tree, change):
    directory = evidence(tree)
    path = directory / "1/daemon-shard.json"
    value = json.loads(path.read_text())
    path.write_text(json.dumps({**value, **change}))
    with pytest.raises(ValueError, match="identity or complete"):
        shards.validated_data(tree, directory, 2)


def test_missing_or_symlink_coverage_is_refused(tree):
    directory = evidence(tree)
    source = directory / "1/.coverage.daemon"
    source.unlink()
    with pytest.raises(ValueError, match="coverage data"):
        shards.validated_data(tree, directory, 2)
    source.symlink_to(directory / "0/.coverage.daemon")
    with pytest.raises(ValueError, match="coverage data"):
        shards.validated_data(tree, directory, 2)


def test_repository_partition_is_exhaustive_and_stable():
    root = Path(__file__).parents[1]
    groups = shards.partition(root, 16)
    flattened = [name for group in groups for name in group]
    assert len(groups) == 16 and all(groups)
    assert len(flattened) == len(set(flattened))
    assert sorted(flattened) == shards.test_files(root)
