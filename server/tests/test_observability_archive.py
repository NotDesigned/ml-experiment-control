from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_exp_server.observability_archive import (
    ArchiveSource,
    ObservabilityArchive,
    SourceCursor,
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
