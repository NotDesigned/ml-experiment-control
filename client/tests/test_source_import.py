"""Source import is useful without a Dockerfile, builder or scheduler."""
import hashlib
import io
import json
import tarfile
from urllib.parse import parse_qs, urlsplit

import pytest

from ml_exp_client.cli import main


def test_source_import_needs_only_source_capability(tmp_path, monkeypatch, capsys):
    source = tmp_path / "src"
    source.mkdir()
    (source / "train.py").write_text("print('hello')\n")
    calls = []
    class API:
        def negotiate(self):return {"capabilities": ["source-import.v1"]}
        def call(self, path, **kwargs):
            calls.append(path)
            url = urlsplit(path)
            assert url.path == "/api/source-imports/archive"
            assert parse_qs(url.query) == {"project": ["demo"], "sha256": [hashlib.sha256(kwargs["raw"]).hexdigest()]}
            with tarfile.open(fileobj=io.BytesIO(kwargs["raw"])) as archive:
                assert archive.getnames() == ["train.py"]
                assert archive.extractfile("train.py").read() == b"print('hello')\n"
            return {"project": "demo", "source_id": "source." + "a" * 64, "tree_digest": "sha256:" + "a" * 64}
    monkeypatch.setattr("ml_exp_client.cli.Client", lambda *a, **k: API())
    assert main(["--url", "https://api.example", "source-import", "--project", "demo", "--directory", str(source)]) == 0
    assert json.loads(capsys.readouterr().out)["source_id"] == "source." + "a" * 64
    assert len(calls) == 1 and sorted(p.name for p in tmp_path.iterdir()) == ["src"]


@pytest.mark.parametrize("case", ["unsupported", "disabled", "bad-project", "secret-path", "symlink"])
def test_invalid_source_import_never_uploads(tmp_path, monkeypatch, capsys, case):
    class API:
        def negotiate(self):
            return {"capabilities": [] if case == "unsupported" else ["source-import.v1"],
                    "source_imports": case != "disabled"}
        def call(self, *a, **kw):pytest.fail("invalid source must not be uploaded")
    monkeypatch.setattr("ml_exp_client.cli.Client", lambda *a, **k: API())
    if case == "secret-path":(tmp_path / ".env").write_text("PRIVATE_TEST_VALUE")
    if case == "symlink":(tmp_path / "link").symlink_to("/etc/passwd")
    assert main(["--url", "https://api.example", "source-import", "--project",
                 "bad/project" if case == "bad-project" else "demo", "--directory", str(tmp_path)]) == 2
    output = capsys.readouterr()
    assert not output.out and "PRIVATE_TEST_VALUE" not in output.err
