"""Fail-closed HTTP bearer credential loading."""

from __future__ import annotations

import os

import pytest

from ml_exp_server.http_auth import HttpAuthError, load_bearer_token


def test_bearer_token_requires_private_regular_owned_file(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(HttpAuthError, match="unavailable"):
        load_bearer_token(missing)

    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(HttpAuthError, match="regular file"):
        load_bearer_token(directory)

    target = tmp_path / "target"
    target.write_text("x" * 40, encoding="utf-8")
    target.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(HttpAuthError, match="not a symlink"):
        load_bearer_token(link)

    target.chmod(0o640)
    with pytest.raises(HttpAuthError, match="0600"):
        load_bearer_token(target)


def test_bearer_token_rejects_unsafe_content_and_size(tmp_path):
    token = tmp_path / "token"
    token.write_text("too-short", encoding="utf-8")
    token.chmod(0o600)
    with pytest.raises(HttpAuthError, match="at least 32"):
        load_bearer_token(token)

    token.write_text("x" * 32 + " internal-space", encoding="utf-8")
    with pytest.raises(HttpAuthError, match="non-whitespace"):
        load_bearer_token(token)

    token.write_bytes(b"x" * 4097)
    with pytest.raises(HttpAuthError, match="unexpectedly large"):
        load_bearer_token(token)

    token.write_bytes(b"\xff" * 40)
    with pytest.raises(HttpAuthError, match="unreadable"):
        load_bearer_token(token)


def test_bearer_token_rejects_foreign_owner_when_supported(monkeypatch, tmp_path):
    if not hasattr(os, "getuid"):
        pytest.skip("POSIX ownership is unavailable")
    token = tmp_path / "token"
    token.write_text("x" * 40, encoding="utf-8")
    token.chmod(0o600)
    monkeypatch.setattr(os, "getuid", lambda: token.stat().st_uid + 1)
    with pytest.raises(HttpAuthError, match="owned by the daemon user"):
        load_bearer_token(token)
