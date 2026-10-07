"""Actual producer exit semantics and deterministic HTTPS transport failures."""

import http.client
import io
import json
import os
import runpy
import signal
import sys
import tarfile
from types import SimpleNamespace

import pytest

from ml_exp_server import worker_artifacts as worker


@pytest.mark.parametrize("url", ["http://example/upload", "https://user:pass@example/upload", "https://example/upload?key=value", "https://example/upload#fragment"])
def test_worker_rejects_nonfixed_https_upload_destination(url):
    with pytest.raises(ValueError, match="fixed HTTPS"):
        worker.upload_parts(url, "test-capability", io.BytesIO(b"payload"), 7)












def test_archiver_ignores_special_files_and_a_regular_to_fifo_race(monkeypatch, tmp_path):
    normal = tmp_path / "race.txt"
    normal.write_text("content")
    os.mkfifo(tmp_path / "fifo")
    (tmp_path / "link").symlink_to(normal)
    original_open = os.open
    def replace_before_open(path, flags, *args, **kwargs):
        if path == "race.txt" and not flags & os.O_DIRECTORY:
            os.unlink(path, dir_fd=kwargs["dir_fd"])
            os.mkfifo(path, dir_fd=kwargs["dir_fd"])
        return original_open(path, flags, *args, **kwargs)
    monkeypatch.setattr(worker.os, "open", replace_before_open)
    assert worker.archive_outputs(tmp_path, io.BytesIO(), 100, ["**/*"]) == 0
