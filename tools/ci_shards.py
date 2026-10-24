"""Deterministic file partitions and fail-closed coverage-shard manifests.

Timing observations only balance work; discovering tests never depends on them.
Coverage data and manifests are data, never executable artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess


TIMINGS = Path(__file__).with_name("ci_test_timings.json")


def test_files(root):
    return sorted(str(path.relative_to(root)) for folder in ("server/tests", "worker/tests")
                  for path in (root / folder).rglob("*.py")
                  if path.name.startswith("test_") or path.name.endswith("_test.py"))


def partition(root, count):
    if not 1 <= count <= 64:
        raise ValueError("shard count must be between 1 and 64")
    timings = json.loads(TIMINGS.read_text())
    files = test_files(root)
    if len(files) < count:
        raise ValueError("every shard must contain at least one test file")
    weights = {name: max(float(timings.get(name, (root / name).stat().st_size / 4000)), 0.05)
               for name in files}
    shards, loads = [[] for _ in range(count)], [0.0] * count
    for name in sorted(files, key=lambda name: (-weights[name], name)):
        index = min(range(count), key=lambda index: (loads[index], index))
        shards[index].append(name)
        loads[index] += weights[name]
    return [sorted(files) for files in shards]


def checkout_tree(root):
    return subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True).strip()


def write_manifest(root, index, count, files, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"tree": checkout_tree(root), "index": index, "count": count,
                                       "files": files, "pytest_passed": True}) + "\n")


def validated_data(root, directory, count):
    """Require every expected partition once, on this tree, before combining."""
    expected = partition(root, count)
    tree, seen, data = checkout_tree(root), set(), []
    manifests = sorted(directory.rglob("daemon-shard.json"))
    if len(manifests) != count:
        raise ValueError("missing or extra daemon shard manifests")
    for manifest in manifests:
        value = json.loads(manifest.read_text())
        index = value.get("index")
        if (type(index) is not int or not 0 <= index < count or index in seen
                or value.get("count") != count or value.get("tree") != tree
                or value.get("pytest_passed") is not True or value.get("files") != expected[index]):
            raise ValueError("daemon shard identity or complete file partition differs")
        source = manifest.with_name(".coverage.daemon")
        if not source.is_file() or source.is_symlink() or source.stat().st_size > 32 * 1024**2:
            raise ValueError("daemon coverage data is missing or invalid")
        seen.add(index)
        data.append(source)
    return data
