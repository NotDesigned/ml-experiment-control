"""Producer data integrity, filesystem boundaries, credentials and live snapshots."""
import ast
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import tarfile
from types import SimpleNamespace

import pytest

from ml_exp_server import managed_worker as worker
from ml_exp_server import container_worker as legacy


def tar_bytes(files, special=None):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        root = tarfile.TarInfo(".")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        folder = tarfile.TarInfo("nested")
        folder.type = tarfile.DIRTYPE
        archive.addfile(folder)
        for name, body in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(body)
            archive.addfile(member, io.BytesIO(body))
        if special:
            archive.addfile(special)
    return output.getvalue()


def input_item(data, files):
    sha = hashlib.sha256(data).hexdigest()
    return {"asset_id": "asset." + sha, "sha256": sha, "archive_bytes": len(data), "url": "https://api.example/input",
            "files": [{"path": name, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()} for name, body in files.items()],
            "mount_path": "/inputs/data"}


@pytest.mark.parametrize("url", ["http://api.example/input", "https://user:pass@api.example/input", "https://api.example/input?q=x", "https://api.example/input#x"])
def test_input_fetch_requires_fixed_https(url):
    with pytest.raises(ValueError, match="fixed HTTPS"):
        worker.fetch(url, "private-capability", io.BytesIO(), "a" * 64, 10)


@pytest.mark.parametrize("status,size,data,sha,error", [(200, 3, b"abc", hashlib.sha256(b"abc").hexdigest(), None),
    (401, 3, b"abc", "a" * 64, "rejected"), (200, 4, b"abc", "a" * 64, "size"),
    (200, 3, b"ab", "a" * 64, "truncated"), (200, 3, b"abc", "a" * 64, "checksum")])
def test_input_fetch_checks_auth_size_digest_and_closes(monkeypatch, status, size, data, sha, error):
    closed = []
    body = io.BytesIO(data)
    class Connection:
        def __init__(self, *args, **kwargs): pass
        def request(self, method, path, headers):
            assert method == "GET" and headers["Authorization"] == "Bearer exact-capability"
        def getresponse(self):
            return SimpleNamespace(status=status, getheader=lambda *a: str(size), read=body.read)
        def close(self): closed.append(True)
    monkeypatch.setattr(worker.http.client, "HTTPSConnection", Connection)
    output = io.BytesIO()
    if error:
        with pytest.raises(ValueError, match=error): worker.fetch("https://api.example/input", "exact-capability", output, sha, 3)
    else:
        worker.fetch("https://api.example/input", "exact-capability", output, sha, 3)
        assert output.read() == data
    assert closed == [True]


def test_delivery_verifies_bytes_reuses_nas_cache_and_detects_changes(tmp_path, monkeypatch):
    files = {"nested/tokens.bin": b"training tokens"}
    data = tar_bytes(files)
    item = input_item(data, files)
    fetches = []
    def fetch(url, token, stream, *args):
        fetches.append(token); stream.write(data); stream.seek(0)
    monkeypatch.setattr(worker, "fetch", fetch)
    destination = worker.deliver(item, "exact-capability", tmp_path / "cache")
    assert worker.deliver(item, "exact-capability", tmp_path / "cache") == destination and fetches == ["exact-capability"]
    assert (destination / "nested/tokens.bin").stat().st_mode & 0o777 == 0o444
    (destination / "nested/tokens.bin").chmod(0o644)
    (destination / "nested/tokens.bin").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"): worker.deliver(item, "exact-capability", tmp_path / "cache")
    item["asset_id"] = "asset." + "a" * 64
    with pytest.raises(ValueError, match="identity"): worker.deliver(item, "exact-capability", tmp_path / "cache")


def test_existing_bind_mount_is_accepted_without_replacing_directory(tmp_path):
    directory=tmp_path/"outputs"; directory.mkdir()
    worker.link_path(directory,directory)
    assert directory.is_dir() and not directory.is_symlink()
    other=tmp_path/"other";other.mkdir()
    link=tmp_path/"alias";link.symlink_to(other)
    with pytest.raises(ValueError,match="already bound"):
        worker.link_path(directory,link)


@pytest.mark.parametrize("case", ["missing", "unexpected", "size", "symlink", "duplicate"])
def test_delivery_rejects_tar_manifest_mismatches(tmp_path, monkeypatch, case):
    files = {"nested/x.bin": b"expected"}
    special = None
    archived = dict(files)
    if case == "missing": archived = {}
    if case == "unexpected": archived["extra"] = b"not declared"
    if case == "size": archived["nested/x.bin"] = b"different size"
    if case in {"symlink", "duplicate"}:
        special = tarfile.TarInfo("nested/x.bin")
        special.type = tarfile.SYMTYPE if case == "symlink" else tarfile.REGTYPE
        special.linkname = "/etc/passwd"
    data = tar_bytes(archived, special)
    item = input_item(data, files)
    def fetch(url, token, stream, *args): stream.write(data); stream.seek(0)
    monkeypatch.setattr(worker, "fetch", fetch)
    with pytest.raises(ValueError, match="archive"):
        worker.deliver(item, "capability", tmp_path / "cache")
    assert not list((tmp_path / "cache").glob(".input-*"))


@pytest.mark.parametrize("path", ["../x", "/x", "", "a\\b", ".secret", "a/.secret"])
def test_relative_paths_reject_escapes_and_hidden_files(path):
    with pytest.raises(ValueError): worker.relative_path(path)


@pytest.mark.parametrize("case", ["missing", "symlink", "parent-symlink", "extra"])
def test_cached_input_cannot_escape_or_add_files(tmp_path, case):
    tree = tmp_path / "cache"; tree.mkdir()
    outside = tmp_path / "outside"; outside.mkdir(); (outside / "x").write_bytes(b"abc")
    files = [{"path": "x", "bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}]
    if case == "symlink": (tree / "x").symlink_to(outside / "x")
    if case == "parent-symlink":
        (tree / "parent").symlink_to(outside, target_is_directory=True); files[0]["path"] = "parent/x"
    if case == "extra":
        (tree / "x").write_bytes(b"abc"); (tree / "extra").write_bytes(b"extra")
    with pytest.raises(ValueError): worker.verify_tree(tree, files)


def checkpoint(root, body=b"weights"):
    file = root / "checkpoints/step-1/model.pt"; file.parent.mkdir(parents=True, exist_ok=True); file.write_bytes(body)
    manifest = {"step": 1, "files": [{"path": "checkpoints/step-1/model.pt", "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}]}
    (root / "checkpoint.ready.json").write_text(json.dumps(manifest))
    return manifest, file


def test_checkpoint_archive_has_exact_declared_bytes_and_manifest(tmp_path):
    ready, _ = checkpoint(tmp_path)
    output = io.BytesIO()
    sha = worker.checkpoint_archive(tmp_path, output)
    assert sha == hashlib.sha256((tmp_path / "checkpoint.ready.json").read_bytes()).hexdigest()
    output.seek(0)
    with tarfile.open(fileobj=output) as archive:
        assert archive.extractfile("checkpoints/step-1/model.pt").read() == b"weights"
        assert json.load(archive.extractfile("checkpoint.ready.json")) == ready


def test_published_checkpoint_reuses_verified_cache_without_network(tmp_path, monkeypatch):
    root = tmp_path / "outputs"
    root.mkdir()
    checkpoint(root)
    stream = io.BytesIO()
    worker.checkpoint_archive(root, stream)
    data = stream.getvalue()
    cache = tmp_path / "cache"
    destination = worker.cache_checkpoint(stream, cache)
    files = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    item = input_item(data, files)
    def forbidden(*args):
        pytest.fail("same-backend checkpoint recovery must not download the archive")
    monkeypatch.setattr(worker, "fetch", forbidden)
    assert worker.deliver(item, "capability", cache) == destination
    assert worker.cache_checkpoint(stream, cache) == destination
    weights = destination / "checkpoints/step-1/model.pt"
    weights.chmod(0o644)
    weights.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        worker.deliver(item, "capability", cache)


def test_local_checkpoint_archive_mismatch_is_not_cached(tmp_path):
    data = tar_bytes({"x": b"bytes"})
    item = input_item(data, {"x": b"bytes"})
    with pytest.raises(ValueError, match="local checkpoint"):
        worker.deliver(item, None, tmp_path / "cache", archive_stream=io.BytesIO(b"changed archive"))
    assert not (tmp_path / "cache" / item["asset_id"]).exists()
    assert not list((tmp_path / "cache").glob(".input-*"))


@pytest.mark.parametrize("case", ["large-manifest", "empty", "many", "duplicate", "reserved", "wrong-sha", "wrong-size", "file-link", "parent-link", "fifo", "limit", "changing"])
def test_checkpoint_publication_rejects_incomplete_mutable_or_escaping_files(tmp_path, monkeypatch, case):
    ready, file = checkpoint(tmp_path)
    if case == "empty": ready["files"] = []
    if case == "many": ready["files"] *= 20001
    if case == "duplicate": ready["files"] *= 2
    if case == "reserved": ready["files"][0]["path"] = "checkpoint.ready.json"
    if case == "wrong-sha": ready["files"][0]["sha256"] = "a" * 64
    if case == "wrong-size": ready["files"][0]["bytes"] = 1
    if case == "file-link":
        file.unlink(); file.symlink_to(tmp_path / "checkpoint.ready.json")
    if case == "parent-link":
        ready["files"][0]["path"] = "escape/model.pt"; (tmp_path / "escape").symlink_to(file.parent, target_is_directory=True)
    if case == "fifo":
        file.unlink(); os.mkfifo(file)
    if case == "changing":
        original = worker.os.fstat
        calls = 0
        def fstat(fd):
            nonlocal calls
            calls += 1
            value = original(fd)
            if calls == 4:
                return SimpleNamespace(st_size=value.st_size, st_mtime_ns=value.st_mtime_ns + 1, st_ctime_ns=value.st_ctime_ns)
            return value
        monkeypatch.setattr(worker.os, "fstat", fstat)
    (tmp_path / "checkpoint.ready.json").write_text(json.dumps(ready))
    if case == "large-manifest": (tmp_path / "checkpoint.ready.json").write_bytes(b"x" * (256 * 1024 + 1))
    with pytest.raises((ValueError, OSError)):
        worker.checkpoint_archive(tmp_path, io.BytesIO(), 100 if case == "limit" else 2 * 1024 ** 3)


def test_container_alias_is_stable_and_rejects_conflicts(tmp_path):
    target = tmp_path / "target"; target.mkdir()
    link = tmp_path / "inputs/data"
    worker.link_path(target, link); worker.link_path(target, link)
    with pytest.raises(ValueError, match="already bound"): worker.link_path(tmp_path / "other", link)


@pytest.mark.parametrize("program_code,upload_fail,expected", [(0, False, 0), (3, False, 3), (-15, False, 143), (0, True, 74), (3, True, 3)])
def test_managed_worker_hides_capabilities_publishes_live_checkpoint_and_preserves_exit(tmp_path, monkeypatch, capsys, program_code, upload_fail, expected):
    root = tmp_path / "project/runs/trial/attempts/attempt-001/outputs"; root.mkdir(parents=True)
    checkpoint(root)
    for key, value in {"OUTPUT_DIR": str(root), "ML_EXPD_UPLOAD_URL": "https://api.example/final", "ML_EXPD_UPLOAD_TOKEN": "private-capability", "ML_EXPD_UPLOAD_LIMIT": str(2 * 1024 ** 2),
                       "ML_EXPD_INPUT_ASSETS": '[{"asset_id":"test","mount_path":"/inputs/data","archive_bytes":10240}]', "ML_EXPD_SNAPSHOT_URL": "https://api.example/snapshot", "ML_EXPD_SNAPSHOT_INTERVAL": "5"}.items():
        monkeypatch.setenv(key, value)
    linked, published, delivered = [], [], []
    monkeypatch.setattr(worker, "link_path", lambda target, path: linked.append(str(path)))
    deliver = worker.deliver
    def delivery(item, token, cache, **kwargs):
        if kwargs:
            return deliver(item, token, cache, **kwargs)
        delivered.append(token)
        return tmp_path
    monkeypatch.setattr(worker, "deliver", delivery)
    class Child:
        pid = 12345
        calls = 0
        def poll(self):
            self.calls += 1
            return None if self.calls == 1 else program_code
    child = Child()
    def popen(*args, **kwargs):
        assert not any(k.startswith("ML_EXPD_") for k in os.environ)
        return child
    monkeypatch.setattr(worker.subprocess, "Popen", popen)
    handlers = {}
    monkeypatch.setattr(worker.signal, "signal", lambda sig, fn: handlers.setdefault(sig, fn))
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: None)
    def upload(url, token, stream, length):
        if upload_fail: raise OSError("upload unavailable")
        assert token == "private-capability" and length > 0
        if url.endswith("snapshot"):
            stream.seek(0)
            digest, _ = worker.digest_stream(stream)
            assert (root.parents[4] / "data-assets" / ("asset." + digest)).is_dir()
        published.append(url)
    monkeypatch.setattr(worker, "upload", upload)
    assert worker.main(["python3", "train.py"]) == expected
    assert linked == ["/outputs", "/inputs/data"] and delivered == ["private-capability"]
    if not upload_fail:
        assert published == ["https://api.example/snapshot", "https://api.example/final"]
    assert "private-capability" not in capsys.readouterr().out
    child.calls = 0
    forwarded = []
    monkeypatch.setattr(worker.os, "killpg", lambda pid, sig: forwarded.append((pid, sig)))
    handlers[signal.SIGTERM](signal.SIGTERM, None)
    handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert forwarded == [(12345, signal.SIGTERM)]


def test_delivery_failure_prevents_training_and_worker_entrypoints(tmp_path, monkeypatch):
    for key, value in {"OUTPUT_DIR": str(tmp_path / "outputs"), "ML_EXPD_UPLOAD_URL": "https://api.example/final", "ML_EXPD_UPLOAD_TOKEN": "private", "ML_EXPD_UPLOAD_LIMIT": "1"}.items():
        monkeypatch.setenv(key, value)
    def bad_link(*args): raise OSError("cannot bind input")
    monkeypatch.setattr(worker, "link_path", bad_link)
    assert worker.main(["must-not-run"]) == 65
    monkeypatch.setitem(sys.modules, "legacy_worker", legacy)
    monkeypatch.setitem(sys.modules, "data_preparation", __import__("ml_exp_server.data_preparation", fromlist=["prepare"]))
    monkeypatch.setitem(sys.modules, "persistent_state", __import__("ml_exp_server.persistent_state", fromlist=["publish"]))
    runpy.run_path(worker.__file__, run_name="standalone-import")
    parsed = ast.parse(Path(worker.__file__).read_text())
    guard = ast.Module(body=[parsed.body[-1]], type_ignores=[])
    with pytest.raises(SystemExit) as result:
        exec(compile(guard, worker.__file__, "exec"), {"__name__": "__main__", "main": lambda: 0})
    assert result.value.code == 0


@pytest.mark.parametrize("mode", ["disabled", "not-ready", "later", "snapshot-overhead", "marker-race", "final-overhead", "cache-failure"])
def test_worker_checkpoint_options_and_archive_overhead_limits(tmp_path, monkeypatch, mode, capsys):
    root=tmp_path/"outputs"; root.mkdir()
    for key,value in {"OUTPUT_DIR":str(root),"ML_EXPD_UPLOAD_URL":"https://api.example/final","ML_EXPD_UPLOAD_TOKEN":"private","ML_EXPD_UPLOAD_LIMIT":str(2*1024**2)}.items(): monkeypatch.setenv(key,value)
    if mode != "disabled": monkeypatch.setenv("ML_EXPD_SNAPSHOT_URL","https://api.example/snapshot")
    else: monkeypatch.delenv("ML_EXPD_SNAPSHOT_URL",raising=False)
    if mode not in {"disabled","not-ready"}: checkpoint(root)
    monkeypatch.setattr(worker,"link_path",lambda *a:None)
    monkeypatch.setattr(worker.signal,"signal",lambda *a:None)
    monkeypatch.setattr(worker.time,"sleep",lambda *a:None)
    if mode == "later": monkeypatch.setattr(worker.time,"monotonic",lambda:0)
    class Child:
        calls=0
        def poll(self):
            self.calls+=1; return None if self.calls==1 else 0
    monkeypatch.setattr(worker.subprocess,"Popen",lambda *a,**k:Child())
    monkeypatch.setattr(worker,"upload",lambda *a:None)
    def cache(*args):
        if mode == "cache-failure": raise OSError("cache disk unavailable")
    monkeypatch.setattr(worker,"cache_checkpoint",cache)
    if mode == "snapshot-overhead":
        def oversized(root,stream,limit): stream.write(b"x"*(limit+1)); return hashlib.sha256(worker.read_checkpoint_ready(root)).hexdigest()
        monkeypatch.setattr(worker,"checkpoint_archive",oversized)
    if mode == "marker-race": monkeypatch.setattr(worker,"checkpoint_archive",lambda *a:"changed marker")
    if mode == "final-overhead":
        def oversized(root,stream,*a): stream.write(b"x"*(2*1024**2+1))
        monkeypatch.setattr(worker,"archive_outputs",oversized)
    assert worker.main(["train"]) == (74 if mode=="final-overhead" else 0)
    if mode in {"snapshot-overhead","marker-race"}: assert "CHECKPOINT_UPLOAD=FAILED" in capsys.readouterr().err
    if mode == "cache-failure":
        captured = capsys.readouterr()
        assert "CHECKPOINT_CACHE=FAILED" in captured.err and "CHECKPOINT_UPLOAD=COMPLETE" in captured.out
