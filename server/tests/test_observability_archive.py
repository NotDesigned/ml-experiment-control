from __future__ import annotations

import json
import math
import os
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

    first.write_text("different")
    with pytest.raises(RuntimeError, match="collision"):
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


def test_persist_rejects_bad_key_and_record_symlink(tmp_path):
    path = tmp_path / "stdout.log"
    path.write_text("line\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(path)).records[0]
    with pytest.raises(ValueError, match="invalid archive record key"):
        archive.persist([replace(record, idempotency_key="bad")])

    persisted = archive.persist([record])[0]
    persisted.unlink()
    target = archive.archive_root / "safe-target"
    target.write_text("outside")
    persisted.symlink_to(target)
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


def test_archive_load_rejects_record_symlink_invalid_json_and_identity(tmp_path):
    path = tmp_path / "stdout.log"
    path.write_text("line\n")
    archive = ObservabilityArchive(tmp_path / "archive")
    record = archive.scan(source(path)).records[0]
    persisted = archive.persist([record])[0]
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    persisted.unlink()
    persisted.symlink_to(outside)
    with pytest.raises(ValueError, match="regular file"):
        archive.load([record.source_id])

    persisted.unlink()
    persisted.write_text("not-json")
    with pytest.raises(ValueError, match="invalid archived"):
        archive.load([record.source_id])

    persisted.write_text(json.dumps(record.__dict__))
    renamed = persisted.with_name("d" * 64 + ".json")
    persisted.rename(renamed)
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


def test_fsync_directory_ignores_open_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "ml_exp_server.observability_archive.os.open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unsupported")),
    )
    _fsync_directory(tmp_path)
