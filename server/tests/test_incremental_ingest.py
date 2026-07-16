"""Direct tests for the incremental append-file cursor primitive.

These exercise ``incremental_io`` in isolation from ``observability_archive``'s
domain semantics (payload parsing, redaction, persistence) -- only the raw
byte-cursor mechanics: offsets, generations, and truncate/replace/rewrite
detection.
"""

from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path

import pytest

from ml_exp_server.incremental_io import (
    SourceCursor,
    digest,
    open_regular_nofollow,
    scan_increment,
)

SOURCE = "a" * 64


def test_digest_matches_sha256_hexdigest():
    assert digest(b"hello") == hashlib.sha256(b"hello").hexdigest()
    assert digest(b"") != digest(b"x")


def test_first_scan_reads_from_zero_and_completes_only_full_lines(tmp_path: Path):
    path = tmp_path / "source.log"
    path.write_bytes(b"first\npartial")
    read = scan_increment(SOURCE, path, None, max_read_bytes=1024)

    assert read.data == b"first\n"
    assert read.record_start == 0
    assert read.generation_changed is True
    assert read.cursor.source_id == SOURCE
    assert read.cursor.offset == len(b"first\n")
    assert read.cursor.anchor_start == 0
    assert read.issues == ()


def test_append_advances_from_prior_cursor_using_byte_offsets(tmp_path: Path):
    path = tmp_path / "source.log"
    path.write_bytes(b"first\npartial")
    first = scan_increment(SOURCE, path, None, max_read_bytes=1024)

    path.write_bytes(b"first\npartial line\nnext\n")
    second = scan_increment(SOURCE, path, first.cursor, max_read_bytes=1024)

    assert second.data == b"partial line\nnext\n"
    assert second.record_start == first.cursor.offset
    assert second.generation_changed is False
    assert second.cursor.generation == first.cursor.generation
    assert second.cursor.offset == len(b"first\npartial line\nnext\n")


def test_no_new_bytes_returns_empty_data_and_unchanged_cursor(tmp_path: Path):
    path = tmp_path / "source.log"
    path.write_bytes(b"line\n")
    first = scan_increment(SOURCE, path, None, max_read_bytes=1024)
    second = scan_increment(SOURCE, path, first.cursor, max_read_bytes=1024)

    assert second.data == b""
    assert second.generation_changed is False
    assert second.cursor == first.cursor


def test_truncate_starts_a_new_generation_from_zero(tmp_path: Path):
    path = tmp_path / "source.log"
    path.write_text("old one\nold two\n")
    first = scan_increment(SOURCE, path, None, max_read_bytes=1024)

    path.write_text("new\n")
    truncated = scan_increment(SOURCE, path, first.cursor, max_read_bytes=1024)

    assert truncated.generation_changed is True
    assert truncated.cursor.generation != first.cursor.generation
    assert truncated.record_start == 0
    assert truncated.data == b"new\n"


def test_replace_with_new_inode_starts_a_new_generation(tmp_path: Path):
    path = tmp_path / "source.log"
    path.write_text("old\n")
    first = scan_increment(SOURCE, path, None, max_read_bytes=1024)

    replacement = tmp_path / "replacement"
    replacement.write_text("replacement\n")
    replacement.replace(path)
    replaced = scan_increment(SOURCE, path, first.cursor, max_read_bytes=1024)

    assert replaced.generation_changed is True
    assert replaced.cursor.generation != first.cursor.generation
    assert replaced.cursor.file_identity != first.cursor.file_identity
    assert replaced.data == b"replacement\n"


def test_in_place_rewrite_of_same_length_content_is_detected_via_anchor(
    tmp_path: Path,
):
    path = tmp_path / "source.log"
    path.write_text("original one\n")
    first = scan_increment(SOURCE, path, None, max_read_bytes=1024)

    # Same byte length, same inode (in-place overwrite), different content:
    # only the trailing anchor digest can catch this.
    path.write_text("same-length!\n")
    rewritten = scan_increment(SOURCE, path, first.cursor, max_read_bytes=1024)

    assert rewritten.generation_changed is True
    assert rewritten.cursor.generation != first.cursor.generation
    assert rewritten.cursor.file_identity == first.cursor.file_identity
    assert rewritten.data == b"same-length!\n"


def test_anchor_start_never_goes_negative_for_short_files(tmp_path: Path):
    path = tmp_path / "source.log"
    path.write_bytes(b"x\n")
    read = scan_increment(SOURCE, path, None, max_read_bytes=1024)
    assert read.cursor.anchor_start == 0


def test_oversized_unterminated_record_is_discarded_across_chunks(tmp_path: Path):
    path = tmp_path / "source.log"
    path.write_text("abcdefghijklmno\n")
    first = scan_increment(SOURCE, path, None, max_read_bytes=4)
    assert first.data == b""
    assert [item.reason for item in first.issues] == ["oversized_record"]

    cursor = first.cursor
    collected = b""
    while cursor.offset < path.stat().st_size:
        step = scan_increment(SOURCE, path, cursor, max_read_bytes=4)
        collected += step.data
        cursor = step.cursor
    # The whole oversized logical line is dropped, never re-emitted in parts.
    assert collected == b""


def test_discard_completes_as_soon_as_the_newline_arrives(tmp_path: Path):
    path = tmp_path / "source.log"
    path.write_text("toolongline\nshort\n")
    # A chunk size wide enough that the discarded record's terminator and
    # the following short line are not split across three-plus chunks, so
    # this specifically exercises "discard ends and a complete trailing
    # line is returned from the same increment".
    first = scan_increment(SOURCE, path, None, max_read_bytes=8)
    assert first.data == b""
    assert [item.reason for item in first.issues] == ["oversized_record"]

    cursor = first.cursor
    total = b""
    while cursor.offset < path.stat().st_size:
        step = scan_increment(SOURCE, path, cursor, max_read_bytes=8)
        assert step.issues == ()
        total += step.data
        cursor = step.cursor
    assert total == b"short\n"


def test_discard_at_eof_reports_no_duplicate_issue(tmp_path: Path):
    path = tmp_path / "source.log"
    path.write_text("unterminated")
    cursor = scan_increment(SOURCE, path, None, max_read_bytes=4).cursor
    while cursor.offset < path.stat().st_size:
        cursor = scan_increment(SOURCE, path, cursor, max_read_bytes=4).cursor
    at_eof = scan_increment(SOURCE, path, cursor, max_read_bytes=4)
    assert at_eof.data == b""
    assert at_eof.issues == ()
    assert at_eof.cursor.offset == cursor.offset


def test_scan_increment_rejects_symlink_source(tmp_path: Path):
    real = tmp_path / "real.log"
    real.write_text("line\n")
    link = tmp_path / "link.log"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="symbolic"):
        scan_increment(SOURCE, link, None, max_read_bytes=1024)


def test_scan_increment_rejects_non_regular_file(tmp_path: Path):
    directory = tmp_path / "a-directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        scan_increment(SOURCE, directory, None, max_read_bytes=1024)


def test_open_regular_nofollow_rejects_symlink_and_accepts_regular_file(
    tmp_path: Path,
):
    real = tmp_path / "real.log"
    real.write_text("line\n")
    link = tmp_path / "link.log"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="symbolic"):
        open_regular_nofollow(link)

    descriptor, metadata = open_regular_nofollow(real)
    try:
        assert metadata.st_size == real.stat().st_size
    finally:
        os.close(descriptor)


def test_open_regular_nofollow_propagates_non_symlink_oserror(
    tmp_path: Path, monkeypatch,
):
    path = tmp_path / "source.log"
    path.write_text("line\n")

    def fake_open(_path, _flags):
        raise OSError(errno.EACCES, "permission denied")

    monkeypatch.setattr(os, "open", fake_open)
    with pytest.raises(OSError, match="permission denied"):
        open_regular_nofollow(path)


def test_scan_increment_validates_max_read_bytes(tmp_path: Path):
    path = tmp_path / "source.log"
    path.write_text("line\n")
    with pytest.raises(ValueError, match="must be positive"):
        scan_increment(SOURCE, path, None, max_read_bytes=0)


def test_scan_increment_rejects_cursor_for_a_different_source(tmp_path: Path):
    path = tmp_path / "source.log"
    path.write_text("line\n")
    wrong = SourceCursor(
        source_id="b" * 64, generation="g", file_identity="f",
        offset=0, anchor_start=0, anchor_digest="x",
    )
    with pytest.raises(ValueError, match="does not belong"):
        scan_increment(SOURCE, path, wrong, max_read_bytes=1024)


def test_source_cursor_rejects_invalid_offsets():
    with pytest.raises(ValueError, match="cursor offsets"):
        SourceCursor("s", "g", "f", -1, 0, "x")
    with pytest.raises(ValueError, match="cursor offsets"):
        SourceCursor("s", "g", "f", 5, -1, "x")
    with pytest.raises(ValueError, match="cursor offsets"):
        SourceCursor("s", "g", "f", 5, 6, "x")


def test_first_scan_of_the_same_source_and_file_is_a_stable_replay(tmp_path: Path):
    path = tmp_path / "source.log"
    path.write_text('{"event":"metric"}\n')
    first = scan_increment(SOURCE, path, None, max_read_bytes=1024)
    replay = scan_increment(SOURCE, path, None, max_read_bytes=1024)
    assert first.cursor == replay.cursor
    assert first.data == replay.data
