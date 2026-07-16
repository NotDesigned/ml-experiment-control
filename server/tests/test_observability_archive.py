from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_exp_server.observability_archive import (
    ArchiveSource,
    ObservabilityArchive,
    SourceCursor,
    ArchiveRecord,
    _atomic_publish,
    _fsync_directory,
    _sanitize_url,
    _validate_record,
    _has_secret_key,
    sanitize_log_text,
)


def source(path: Path, kind: str = "log", name: str = "stdout") -> ArchiveSource:
    return ArchiveSource(
        workspace_id="workspace", project="elf", run_id="run-1",
        attempt_id="attempt-001", name=name, path=path, kind=kind,  # type: ignore[arg-type]
    )


def test_append_waits_for_partial_line_and_uses_byte_offsets(tmp_path: Path):
    log = tmp_path / "stdout.log"
    log.write_bytes("first 世界\npartial".encode())
    archive = ObservabilityArchive(tmp_path / "archive")

    first = archive.scan(source(log))
    assert [record.payload["text"] for record in first.records] == ["first 世界"]
    assert first.cursor.offset == len("first 世界\n".encode())

    log.write_bytes("first 世界\npartial line\nnext\n".encode())
    second = archive.scan(source(log), first.cursor)
    assert [record.payload["text"] for record in second.records] == [
        "partial line", "next",
    ]
    assert second.records[0].byte_start == first.cursor.offset
    assert not second.generation_changed


def test_truncate_replace_and_in_place_rewrite_start_new_generations(tmp_path: Path):
    log = tmp_path / "stdout.log"
    log.write_text("old one\nold two\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    first = archive.scan(source(log))

    log.write_text("new\n")
    truncated = archive.scan(source(log), first.cursor)
    assert truncated.generation_changed
    assert truncated.cursor.generation != first.cursor.generation
    assert truncated.records[0].byte_start == 0

    replacement = tmp_path / "replacement"
    replacement.write_text("replacement\n")
    replacement.replace(log)
    replaced = archive.scan(source(log), truncated.cursor)
    assert replaced.generation_changed
    assert replaced.cursor.generation != truncated.cursor.generation

    rewritten_cursor = replaced.cursor
    log.write_text("same-length!\n")
    rewritten = archive.scan(source(log), rewritten_cursor)
    assert rewritten.generation_changed
    assert rewritten.records[0].byte_start == 0


def test_structured_jsonl_rejects_invalid_secret_and_nonfinite_records(tmp_path: Path):
    metrics = tmp_path / "train_metrics.jsonl"
    metrics.write_text(
        '{"step":1,"loss":2.5}\n'
        '{broken}\n'
        '{"step":2,"nested":{"api_key":"do-not-store"}}\n'
        '{"step":3,"loss":NaN}\n'
        '[1,2]\n'
        '{"step":4,"loss":1.5}',
    )
    archive = ObservabilityArchive(tmp_path / "archive")
    batch = archive.scan(source(metrics, "metrics", "train"))

    assert [record.payload["step"] for record in batch.records] == [1]
    assert [issue.reason for issue in batch.issues] == [
        "invalid_json", "secret_key", "nonfinite", "invalid_payload",
    ]
    assert batch.cursor.offset == metrics.read_bytes().rfind(b"\n") + 1
    assert "do-not-store" not in repr(batch)


def test_camel_case_secret_keys_are_rejected_without_false_positive(tmp_path: Path):
    for key in ("refreshToken", "accessToken", "sessionCookie", "authToken"):
        assert _has_secret_key({key: "top-secret"})
    assert not _has_secret_key({"trainStepCount": 42})

    metrics = tmp_path / "train_metrics.jsonl"
    metrics.write_text(
        '{"step":1,"refreshToken":"top-secret"}\n'
        '{"step":2,"trainStepCount":42}\n'
    )
    batch = ObservabilityArchive(tmp_path / "archive").scan(
        source(metrics, "metrics", "train")
    )

    assert [issue.reason for issue in batch.issues] == ["secret_key"]
    assert [record.payload["step"] for record in batch.records] == [2]
    assert "top-secret" not in repr(batch)


def test_events_accept_nested_finite_json_and_have_stable_record_keys(tmp_path: Path):
    events = tmp_path / "events.jsonl"
    events.write_text('{"event":"metric","values":[1,2.0]}\n')
    archive = ObservabilityArchive(tmp_path / "archive")
    first = archive.scan(source(events, "events", "timeline"))
    replay = archive.scan(source(events, "events", "timeline"), None)
    assert first.records == replay.records
    assert len(first.records[0].idempotency_key) == 64


def test_structured_string_values_are_sanitized_before_persistence(tmp_path: Path):
    events = tmp_path / "events.jsonl"
    events.write_text(
        '{"event":"warning","message":"password=hunter2",'
        '"url":"https://user:pass@example.test/path?accessToken=query-secret"}\n'
    )
    archive = ObservabilityArchive(tmp_path / "archive")
    batch = archive.scan(source(events, "events", "timeline"))
    path = archive.persist(batch.records)[0]

    assert "hunter2" not in repr(batch)
    assert "query-secret" not in repr(batch)
    assert "user:pass" not in path.read_text()


def test_logs_are_sanitized_before_record_or_disk_persistence(tmp_path: Path):
    log = tmp_path / "stderr.log"
    secret = "cloud-secret-123"
    log.write_text(
        f"WANDB_API_KEY={secret} Authorization: Bearer token-value "
        "https://user:pass@example.test/path?api_key=query-secret&step=1\n"
    )
    archive = ObservabilityArchive(tmp_path / "archive")
    batch = archive.scan(source(log, "log", "stderr"))
    paths = archive.persist(batch.records)

    assert secret not in repr(batch)
    assert "query-secret" not in repr(batch)
    assert "user:pass" not in repr(batch)
    assert "[REDACTED]" in str(batch.records[0].payload["text"])
    assert secret not in paths[0].read_text()
    assert "query-secret" not in paths[0].read_text()
    index_text = paths[0].with_suffix(".idx").read_text()
    assert secret not in index_text
    assert "query-secret" not in index_text


@pytest.mark.parametrize("name", [
    "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY", "MY_PASSWORD",
])
def test_vendor_prefixed_environment_secrets_are_redacted(tmp_path: Path, name: str):
    log = tmp_path / "stdout.log"
    log.write_text(f"{name}=uniquely-sensitive-value\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    batch = archive.scan(source(log))
    persisted = archive.persist(batch.records)[0].read_text()
    assert "uniquely-sensitive-value" not in repr(batch)
    assert "uniquely-sensitive-value" not in persisted
    assert "[REDACTED]" in persisted


def test_oversized_unterminated_record_advances_cursor(tmp_path: Path):
    log = tmp_path / "stdout.log"
    secret = "OPENAI_API_KEY=" + "s" * 20
    log.write_text(secret + "\nsafe\n")
    archive = ObservabilityArchive(tmp_path / "archive", max_read_bytes=8)
    first = archive.scan(source(log))
    assert first.cursor.offset == 8
    assert [issue.reason for issue in first.issues] == ["oversized_record"]
    cursor = first.cursor
    records = []
    while cursor.offset < log.stat().st_size:
        batch = archive.scan(source(log), cursor)
        assert batch.cursor.offset > cursor.offset
        records.extend(batch.records)
        cursor = batch.cursor
    assert [item.payload["text"] for item in records] == ["safe"]
    assert secret not in repr(records)
    at_eof = archive.scan(source(log), cursor)
    assert at_eof.cursor.offset == cursor.offset
    assert at_eof.issues == ()


def test_persist_is_idempotent_and_fails_closed_on_collision(tmp_path: Path):
    log = tmp_path / "stdout.log"
    log.write_text("hello\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(log)).records[0]

    first = archive.persist([record])[0]
    second = archive.persist([record])[0]
    assert first == second
    payload = json.loads(first.read_text())
    assert payload["idempotency_key"] == record.idempotency_key

    # The segment's filename is content-addressed, so directly corrupting its
    # bytes on disk breaks that invariant and is caught as digest corruption
    # -- a stricter, earlier failure than a same-key/different-bytes
    # collision between two otherwise-valid writes (see the dedicated
    # collision tests below).
    first.write_text("different")
    with pytest.raises(ValueError, match="digest mismatch"):
        archive.persist([record])


def test_safe_roots_sources_and_cursor_binding(tmp_path: Path):
    real_root = tmp_path / "real"
    real_root.mkdir()
    link_root = tmp_path / "archive-link"
    link_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(ValueError, match="real directory"):
        ObservabilityArchive(link_root)

    log = tmp_path / "stdout.log"
    log.write_text("ok\n")
    link = tmp_path / "linked.log"
    link.symlink_to(log)
    archive = ObservabilityArchive(tmp_path / "archive")
    with pytest.raises(ValueError, match="symbolic"):
        archive.scan(source(link))

    wrong = SourceCursor("wrong", "g", "f", 0, 0, "x")
    with pytest.raises(ValueError, match="does not belong"):
        archive.scan(source(log), wrong)

    batch = archive.scan(source(log))
    shard = archive.archive_root / batch.records[0].source_id[:2]
    shard.symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        archive.persist(batch.records)


@pytest.mark.parametrize(
    ("raw", "leak"),
    [
        ("password: hunter2", "hunter2"),
        ("Bearer abc.def.ghi", "abc.def.ghi"),
        ("https://name:pass@example.test/x#private", "pass"),
    ],
)
def test_sanitize_log_text_common_secret_shapes(raw: str, leak: str):
    assert leak not in sanitize_log_text(raw)


def test_archive_source_cursor_and_read_limit_validate_inputs(tmp_path):
    with pytest.raises(ValueError, match="unsupported observability source kind"):
        source(tmp_path / "file", "invalid")
    with pytest.raises(ValueError, match="identity fields"):
        ArchiveSource("", "demo", "run", "attempt", "name", tmp_path, "log")
    with pytest.raises(ValueError, match="cursor offsets"):
        SourceCursor("source", "generation", "file", -1, 0, "anchor")
    with pytest.raises(ValueError, match="must be positive"):
        ObservabilityArchive(tmp_path / "archive", max_read_bytes=0)


def test_archive_root_chmod_failure_is_nonfatal(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "ml_exp_server.observability_archive.os.chmod",
        lambda *_args: (_ for _ in ()).throw(OSError("unsupported")),
    )
    assert ObservabilityArchive(tmp_path / "archive").archive_root.is_dir()


def test_scan_rejects_non_regular_fstat(monkeypatch, tmp_path):
    path = tmp_path / "source"
    path.write_text("line\n")
    real = path.stat()
    monkeypatch.setattr(
        "ml_exp_server.observability_archive.os.fstat",
        lambda _fd: SimpleNamespace(
            st_mode=stat.S_IFDIR, st_dev=real.st_dev, st_ino=real.st_ino,
            st_size=real.st_size,
        ),
    )
    with pytest.raises(ValueError, match="regular file"):
        ObservabilityArchive(tmp_path / "archive").scan(source(path))


def test_oversized_record_discard_continues_across_non_newline_chunk(tmp_path):
    path = tmp_path / "stdout.log"
    path.write_text("abcdefghijklmno\n")
    archive = ObservabilityArchive(tmp_path / "archive", max_read_bytes=4)
    first = archive.scan(source(path))
    second = archive.scan(source(path), first.cursor)
    assert second.records == ()
    assert second.issues[0].reason == "oversized_record"


def test_discard_cursor_at_eof_has_no_duplicate_issue(tmp_path):
    path = tmp_path / "stdout.log"
    path.write_text("unterminated")
    archive = ObservabilityArchive(tmp_path / "archive", max_read_bytes=4)
    cursor = archive.scan(source(path)).cursor
    while cursor.offset < path.stat().st_size:
        cursor = archive.scan(source(path), cursor).cursor
    batch = archive.scan(source(path), cursor)
    assert batch.records == ()
    assert batch.issues == ()


def test_persist_rejects_bad_key_and_segment_symlink(tmp_path):
    path = tmp_path / "stdout.log"
    path.write_text("line\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(path)).records[0]
    with pytest.raises(ValueError, match="invalid archive record key"):
        archive.persist([replace(record, idempotency_key="bad")])

    # A first persist commits a content-addressed segment + index. Removing
    # both and pre-empting the exact same content-addressed segment path
    # with a symlink must be rejected the next time the identical batch is
    # persisted (e.g. after a crash-retry that re-scans the same bytes).
    segment_path = archive.persist([record])[0]
    index_path = segment_path.with_suffix(".idx")
    segment_path.unlink()
    index_path.unlink()
    target = archive.archive_root / "safe-target"
    target.write_text("outside")
    segment_path.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        archive.persist([record])


def test_archive_load_skips_readable_and_missing_ids_then_rejects_bad_directory(
    tmp_path,
):
    archive = ObservabilityArchive(tmp_path / "archive")
    missing = "a" * 64
    assert archive.load(["legacy-source", missing]) == ()
    directory = archive.archive_root / "bb" / ("b" * 64)
    directory.parent.mkdir()
    directory.write_text("not a directory")
    with pytest.raises(ValueError, match="real directory"):
        archive.load(["b" * 64])


def test_archive_load_rejects_v1_record_symlink_invalid_json_and_identity(tmp_path):
    # persist() only ever writes the new v2 layout now; the legacy
    # ``<key>.json`` layout is still read, so these edge cases must be
    # exercised against a hand-authored legacy file, exactly as an older
    # daemon would have left on disk.
    path = tmp_path / "stdout.log"
    path.write_text("line\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(path)).records[0]
    directory = archive._safe_child(record.source_id[:2], record.source_id)
    directory.mkdir(parents=True)
    legacy = directory / f"{record.idempotency_key}.json"
    legacy.write_text(json.dumps(record.__dict__))

    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    legacy.unlink()
    legacy.symlink_to(outside)
    with pytest.raises(ValueError, match="regular file"):
        archive.load([record.source_id])

    legacy.unlink()
    legacy.write_text("not-json")
    with pytest.raises(ValueError, match="invalid archived"):
        archive.load([record.source_id])

    legacy.write_text(json.dumps(record.__dict__))
    renamed = legacy.with_name("d" * 64 + ".json")
    legacy.rename(renamed)
    with pytest.raises(ValueError, match="identity mismatch"):
        archive.load([record.source_id])


def test_safe_child_and_crlf_and_url_sanitization_edges(tmp_path):
    archive = ObservabilityArchive(tmp_path / "archive")
    with pytest.raises(ValueError, match="unsafe archive path"):
        archive._safe_child("../escape")

    path = tmp_path / "stdout.log"
    path.write_bytes(b"line\r\n")
    assert archive.scan(source(path)).records[0].payload["text"] == "line"
    assert _sanitize_url("https://[::1]:9443/path),") == (
        "https://[::1]:9443/path),"
    )
    assert _sanitize_url("http://[bad") == "[REDACTED URL]"


def test_validate_record_rejects_every_integrity_violation(tmp_path):
    path = tmp_path / "stdout.log"
    path.write_text("line\n")
    record = ObservabilityArchive(tmp_path / "archive").scan(source(path)).records[0]
    cases = [
        (replace(record, payload={"text": "password=secret"}), "unsanitized log"),
        (replace(record, kind="metrics", payload={"api_key": "secret"}), "secret-like"),
        (replace(record, kind="metrics", payload={"loss": math.nan}), "non-finite"),
        (replace(record, payload={"text": "different"}), "payload digest"),
        (replace(record, idempotency_key="a" * 64), "idempotency key"),
    ]
    for invalid, message in cases:
        with pytest.raises(ValueError, match=message):
            _validate_record(invalid)


def test_atomic_publish_swallows_concurrent_publish_race(tmp_path, monkeypatch):
    # A concurrent or previously crashed writer racing to hardlink the same
    # content-addressed path must not surface as a failure.
    directory = tmp_path / "segments"
    directory.mkdir()
    final_path = directory / "final.jsonl"
    monkeypatch.setattr(
        "ml_exp_server.observability_archive.os.link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileExistsError()),
    )
    _atomic_publish(directory, final_path, b"payload")
    assert not final_path.exists()


def test_atomic_publish_swallows_concurrent_temp_cleanup_race(tmp_path, monkeypatch):
    # A concurrent cleanup of the same random temp name must not surface as
    # a failure once the real content has already been published.
    directory = tmp_path / "segments"
    directory.mkdir()
    final_path = directory / "final.jsonl"
    monkeypatch.setattr(
        "ml_exp_server.observability_archive.os.unlink",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )
    _atomic_publish(directory, final_path, b"payload")
    assert final_path.read_bytes() == b"payload"


def test_fsync_directory_ignores_open_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "ml_exp_server.observability_archive.os.open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unsupported")),
    )
    _fsync_directory(tmp_path)


def test_persist_groups_one_segment_per_source_and_replays_without_duplication(
    tmp_path,
):
    metrics = tmp_path / "train_metrics.jsonl"
    metrics.write_text(
        '{"step":1,"loss":2.0}\n{"step":2,"loss":1.5}\n{"step":3,"loss":1.0}\n'
    )
    log = tmp_path / "stdout.log"
    log.write_text("hello\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    metrics_batch = archive.scan(source(metrics, "metrics", "train"))
    log_batch = archive.scan(source(log))
    assert len(metrics_batch.records) == 3

    all_records = metrics_batch.records + log_batch.records
    paths = archive.persist(all_records)
    assert paths[0] == paths[1] == paths[2]
    assert paths[3] != paths[0]

    metrics_dir = archive._safe_child(
        metrics_batch.records[0].source_id[:2], metrics_batch.records[0].source_id,
        "segments",
    )
    assert sorted(item.suffix for item in metrics_dir.iterdir()) == [".idx", ".jsonl"]

    replayed = archive.persist(all_records)
    assert replayed == paths
    assert sorted(item.suffix for item in metrics_dir.iterdir()) == [".idx", ".jsonl"]

    loaded = archive.load([
        metrics_batch.records[0].source_id, log_batch.records[0].source_id,
    ])
    assert loaded == tuple(
        sorted(all_records, key=lambda item: (item.source_id, item.idempotency_key))
    )


def test_v2_index_records_offsets_and_digests_slice_the_segment(tmp_path):
    metrics = tmp_path / "train_metrics.jsonl"
    metrics.write_text('{"step":1,"loss":2.0}\n{"step":2,"loss":1.5}\n')
    archive = ObservabilityArchive(tmp_path / "archive")
    batch = archive.scan(source(metrics, "metrics", "train"))
    segment_path = archive.persist(batch.records)[0]
    index_path = segment_path.with_suffix(".idx")

    segment_bytes = segment_path.read_bytes()
    index = json.loads(index_path.read_text())
    assert index["version"] == 2
    assert index["source_id"] == batch.records[0].source_id
    assert index["segment_size"] == len(segment_bytes)
    assert hashlib.sha256(segment_bytes).hexdigest() == index["segment_sha256"]
    assert index_path.suffix != ".json"
    for record, entry in zip(batch.records, index["records"]):
        chunk = segment_bytes[entry["byte_start"]:entry["byte_end"]]
        assert hashlib.sha256(chunk).hexdigest() == entry["sha256"]
        assert json.loads(chunk.decode())["idempotency_key"] == record.idempotency_key
        assert entry["idempotency_key"] == record.idempotency_key


def test_persist_same_key_different_full_bytes_fails_closed(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("hello\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(log)).records[0]
    # Same source_id/generation/byte_start/byte_end/payload_digest (so the
    # idempotency key matches), but a different kind -- kind and name are
    # deliberately excluded from the key material, so this is a legitimate
    # same-key-different-bytes collision, not a hash collision.
    twin = replace(record, kind="metrics", name="alt")
    assert twin.idempotency_key == record.idempotency_key

    with pytest.raises(RuntimeError, match="collision"):
        archive.persist([record, twin])

    archive.persist([record])
    with pytest.raises(RuntimeError, match="collision"):
        archive.persist([twin])


def test_persist_fails_closed_against_existing_legacy_v1_record(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("hello\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(log)).records[0]
    directory = archive._safe_child(record.source_id[:2], record.source_id)
    directory.mkdir(parents=True)
    legacy = directory / f"{record.idempotency_key}.json"
    legacy.write_text(json.dumps({**record.__dict__, "name": "different-name"}))

    with pytest.raises(RuntimeError, match="collision"):
        archive.persist([record])


def test_load_reads_mixed_v1_and_v2_records_in_stable_order(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("first\nsecond\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    batch = archive.scan(source(log))
    first, second = batch.records

    directory = archive._safe_child(first.source_id[:2], first.source_id)
    directory.mkdir(parents=True)
    (directory / f"{first.idempotency_key}.json").write_text(
        json.dumps(first.__dict__)
    )
    archive.persist([second])

    loaded = archive.load([first.source_id])
    assert loaded == tuple(
        sorted([first, second], key=lambda item: item.idempotency_key)
    )


def test_persist_backfills_missing_index_without_creating_second_segment(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("hello\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(log)).records[0]
    segment_path = archive.persist([record])[0]
    index_path = segment_path.with_suffix(".idx")
    segments_dir = segment_path.parent
    index_path.unlink()

    # A crash between publishing the segment and publishing its index marker
    # leaves the segment orphaned -- load() must not surface it.
    assert archive.load([record.source_id]) == ()

    result = archive.persist([record])[0]
    assert result == segment_path
    assert index_path.exists()
    assert sorted(item.name for item in segments_dir.iterdir()) == sorted(
        [segment_path.name, index_path.name]
    )
    assert archive.load([record.source_id]) == (record,)


def test_index_referencing_missing_segment_fails_closed(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("hello\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(log)).records[0]
    segment_path = archive.persist([record])[0]
    segment_path.unlink()

    with pytest.raises(ValueError, match="missing or invalid"):
        archive.load([record.source_id])
    with pytest.raises(ValueError, match="missing or invalid"):
        archive.persist([record])


def test_tampered_segment_bytes_fail_closed_on_load(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("hello\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(log)).records[0]
    segment_path = archive.persist([record])[0]
    segment_path.write_bytes(segment_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="segment digest mismatch"):
        archive.load([record.source_id])


def test_tampered_index_offsets_fail_closed_on_load(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("hello\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(log)).records[0]
    segment_path = archive.persist([record])[0]
    index_path = segment_path.with_suffix(".idx")
    payload = json.loads(index_path.read_text())
    payload["records"][0]["byte_start"] = 5
    index_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="index offsets"):
        archive.load([record.source_id])


def test_tampered_index_record_digest_fails_closed_on_load(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("hello\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(log)).records[0]
    segment_path = archive.persist([record])[0]
    index_path = segment_path.with_suffix(".idx")
    payload = json.loads(index_path.read_text())
    payload["records"][0]["sha256"] = "0" * 64
    index_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="record digest mismatch"):
        archive.load([record.source_id])


def test_symlinked_segment_or_index_fails_closed_on_load(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("hello\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(log)).records[0]
    segment_path = archive.persist([record])[0]
    index_path = segment_path.with_suffix(".idx")

    segment_bytes = segment_path.read_bytes()
    outside_segment = tmp_path / "outside-segment"
    outside_segment.write_bytes(segment_bytes)
    segment_path.unlink()
    segment_path.symlink_to(outside_segment)
    with pytest.raises(ValueError, match="missing or invalid"):
        archive.load([record.source_id])

    segment_path.unlink()
    segment_path.write_bytes(segment_bytes)
    outside_index = tmp_path / "outside-index"
    outside_index.write_bytes(index_path.read_bytes())
    index_path.unlink()
    index_path.symlink_to(outside_index)
    with pytest.raises(ValueError, match="invalid archive segment index"):
        archive.load([record.source_id])


def test_symlinked_segments_directory_fails_closed(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("hello\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(log)).records[0]
    archive.persist([record])
    directory = archive._safe_child(record.source_id[:2], record.source_id)
    segments_dir = directory / "segments"
    # Keep the real target inside archive_root so this specifically exercises
    # the "segments must be a real directory" check, distinct from the
    # archive_root-escape check covered by test_safe_roots_sources_and_cursor_binding.
    outside = archive.archive_root / "outside-segments"
    shutil.move(str(segments_dir), str(outside))
    segments_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="real directory"):
        archive.load([record.source_id])


def test_persist_dedupes_the_exact_same_record_within_one_batch(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("hello\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(log)).records[0]

    # The identical record (same key, byte-identical encoding) appearing
    # twice in one persist() call must dedupe silently, not raise.
    paths = archive.persist([record, record])
    assert paths[0] == paths[1]
    assert archive.load([record.source_id]) == (record,)


def test_persist_fails_closed_when_orphan_segment_bytes_are_corrupted(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("hello\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(log)).records[0]
    segment_path = archive.persist([record])[0]
    index_path = segment_path.with_suffix(".idx")
    # Orphan the segment (crash before the index marker was published), then
    # corrupt its bytes -- distinct from test_index_referencing_missing_segment
    # (missing entirely) and from load-time digest tampering (caught earlier,
    # via _v2_committed_records, once an index exists).
    index_path.unlink()
    segment_path.write_bytes(b"corrupted-orphan-content")

    with pytest.raises(RuntimeError, match="segment collision"):
        archive.persist([record])


def test_load_fails_closed_on_v1_v2_cross_format_collision(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("hello\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(log)).records[0]
    twin = replace(record, kind="metrics", name="alt")
    assert twin.idempotency_key == record.idempotency_key
    archive.persist([twin])

    directory = archive._safe_child(record.source_id[:2], record.source_id)
    (directory / f"{record.idempotency_key}.json").write_text(
        json.dumps(record.__dict__)
    )

    with pytest.raises(RuntimeError, match="collision"):
        archive.load([record.source_id])


def test_load_rejects_key_duplicated_across_two_segments(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("hello\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(log)).records[0]
    archive.persist([record])

    segments_dir = archive._safe_child(
        record.source_id[:2], record.source_id, "segments",
    )
    # Hand-craft a second, differently-encoded segment (as if from a buggy
    # writer or a multi-writer race) that commits the exact same key --
    # differently indented JSON gives it a different digest/filename than
    # the segment persist() already published.
    line = (json.dumps(record.__dict__, indent=0) + "\n").encode("utf-8")
    digest2 = hashlib.sha256(line).hexdigest()
    (segments_dir / f"segment-{digest2}.jsonl").write_bytes(line)
    index2 = {
        "version": 2, "source_id": record.source_id, "segment_sha256": digest2,
        "segment_size": len(line),
        "records": [{
            "idempotency_key": record.idempotency_key, "byte_start": 0,
            "byte_end": len(line), "sha256": hashlib.sha256(line).hexdigest(),
        }],
    }
    (segments_dir / f"segment-{digest2}.idx").write_text(json.dumps(index2))

    with pytest.raises(ValueError, match="duplicated across segments"):
        archive.load([record.source_id])


def test_tampered_index_invalid_json_fails_closed_on_load(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("hello\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(log)).records[0]
    segment_path = archive.persist([record])[0]
    index_path = segment_path.with_suffix(".idx")
    index_path.write_text("not-json")

    with pytest.raises(ValueError, match="invalid archive segment index"):
        archive.load([record.source_id])


def test_tampered_index_wrong_version_fails_closed_on_load(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("hello\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(log)).records[0]
    segment_path = archive.persist([record])[0]
    index_path = segment_path.with_suffix(".idx")
    payload = json.loads(index_path.read_text())
    payload["version"] = 1
    index_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="invalid archive segment index"):
        archive.load([record.source_id])


def test_index_entry_missing_trailing_newline_fails_closed_on_load(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("hello\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(log)).records[0]
    segment_path = archive.persist([record])[0]
    index_path = segment_path.with_suffix(".idx")
    segment_bytes = segment_path.read_bytes()
    payload = json.loads(index_path.read_text())
    entry = payload["records"][0]
    # Shrink the declared span by one byte so it stops just short of the
    # line's trailing "\n" -- offsets and the recomputed digest still
    # validate, but the terminator check must still fail closed.
    new_end = entry["byte_end"] - 1
    chunk = segment_bytes[entry["byte_start"]:new_end]
    entry["byte_end"] = new_end
    entry["sha256"] = hashlib.sha256(chunk).hexdigest()
    index_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="index offsets"):
        archive.load([record.source_id])


def test_index_missing_trailing_entry_reports_incomplete_segment(tmp_path):
    metrics = tmp_path / "train_metrics.jsonl"
    metrics.write_text('{"step":1}\n{"step":2}\n')
    archive = ObservabilityArchive(tmp_path / "archive")
    batch = archive.scan(source(metrics, "metrics", "train"))
    segment_path = archive.persist(batch.records)[0]
    index_path = segment_path.with_suffix(".idx")
    payload = json.loads(index_path.read_text())
    payload["records"] = payload["records"][:1]
    index_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="index incomplete"):
        archive.load([batch.records[0].source_id])


def test_segment_record_invalid_json_fails_closed_on_load(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("hello\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(log)).records[0]
    directory = archive._safe_child(record.source_id[:2], record.source_id)
    segments_dir = directory / "segments"
    segments_dir.mkdir(parents=True)

    line = b"not-json\n"
    seg_digest = hashlib.sha256(line).hexdigest()
    (segments_dir / f"segment-{seg_digest}.jsonl").write_bytes(line)
    index_payload = {
        "version": 2, "source_id": record.source_id, "segment_sha256": seg_digest,
        "segment_size": len(line),
        "records": [{
            "idempotency_key": record.idempotency_key, "byte_start": 0,
            "byte_end": len(line), "sha256": hashlib.sha256(line).hexdigest(),
        }],
    }
    (segments_dir / f"segment-{seg_digest}.idx").write_text(json.dumps(index_payload))

    with pytest.raises(ValueError, match="invalid archived observability record"):
        archive.load([record.source_id])


def test_segment_record_identity_mismatch_fails_closed_on_load(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("hello\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(log)).records[0]
    directory = archive._safe_child(record.source_id[:2], record.source_id)
    segments_dir = directory / "segments"
    segments_dir.mkdir(parents=True)

    line = (json.dumps(record.__dict__) + "\n").encode("utf-8")
    seg_digest = hashlib.sha256(line).hexdigest()
    (segments_dir / f"segment-{seg_digest}.jsonl").write_bytes(line)
    wrong_key = "0" * 64
    index_payload = {
        "version": 2, "source_id": record.source_id, "segment_sha256": seg_digest,
        "segment_size": len(line),
        "records": [{
            "idempotency_key": wrong_key, "byte_start": 0,
            "byte_end": len(line), "sha256": hashlib.sha256(line).hexdigest(),
        }],
    }
    (segments_dir / f"segment-{seg_digest}.idx").write_text(json.dumps(index_payload))

    with pytest.raises(ValueError, match="identity mismatch"):
        archive.load([record.source_id])


def test_segment_duplicate_key_within_same_segment_fails_closed_on_load(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("hello\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(log)).records[0]
    directory = archive._safe_child(record.source_id[:2], record.source_id)
    segments_dir = directory / "segments"
    segments_dir.mkdir(parents=True)

    line = (json.dumps(record.__dict__) + "\n").encode("utf-8")
    segment_bytes = line + line
    seg_digest = hashlib.sha256(segment_bytes).hexdigest()
    (segments_dir / f"segment-{seg_digest}.jsonl").write_bytes(segment_bytes)
    entry = {
        "idempotency_key": record.idempotency_key, "byte_start": 0,
        "byte_end": len(line), "sha256": hashlib.sha256(line).hexdigest(),
    }
    entry2 = {**entry, "byte_start": len(line), "byte_end": len(line) * 2}
    index_payload = {
        "version": 2, "source_id": record.source_id, "segment_sha256": seg_digest,
        "segment_size": len(segment_bytes), "records": [entry, entry2],
    }
    (segments_dir / f"segment-{seg_digest}.idx").write_text(json.dumps(index_payload))

    with pytest.raises(ValueError, match="duplicate record key within segment"):
        archive.load([record.source_id])


def test_index_entry_not_a_dict_fails_closed_on_load(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("hello\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(log)).records[0]
    segment_path = archive.persist([record])[0]
    index_path = segment_path.with_suffix(".idx")
    payload = json.loads(index_path.read_text())
    payload["records"][0] = "not-a-dict"
    index_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="invalid archive segment index"):
        archive.load([record.source_id])


def test_leftover_temp_files_are_ignored_by_scans(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("hello\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(log)).records[0]
    segment_path = archive.persist([record])[0]
    segments_dir = segment_path.parent
    (segments_dir / ".tmp-deadbeefdeadbeef").write_bytes(b"partial-segment-write")
    (segments_dir.parent / ".tmp-leftover").write_bytes(b"partial-v1-write")

    assert archive.load([record.source_id]) == (record,)
    assert archive.persist([record])[0] == segment_path
