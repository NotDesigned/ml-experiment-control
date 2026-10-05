"""Offline client behavior, packaged resources and dependency boundaries."""
import importlib.metadata
import json
import os
import subprocess
import sys
import tarfile
import io

from ml_exp_client import source_archive, __version__
from ml_exp_client.cli import main


def test_client_distribution_has_no_runtime_dependencies():
    distribution = importlib.metadata.distribution("ml-experiment-client")
    assert not distribution.requires
    assert distribution.version == __version__
    assert any(entry.name == "ml-exp" and entry.value == "ml_exp_client.cli:main"
               for entry in distribution.entry_points)


def test_offline_init_creates_runnable_source_and_preserves_existing_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ML_EXPD_API_TOKEN", raising=False)
    monkeypatch.delenv("ML_EXPD_API_TOKEN_FILE", raising=False)
    monkeypatch.delenv("ML_EXPD_API_URL", raising=False)
    source = tmp_path / "study"
    assert main(["init", str(source)]) == 0
    assert json.loads(capsys.readouterr().out)["entrypoint"] == ["python", "train.py"]
    program = (source / "train.py").read_bytes()
    assert main(["init", str(source)]) == 2
    assert "File exists" in capsys.readouterr().err
    assert (source / "train.py").read_bytes() == program
    output = tmp_path / "results"
    env = dict(os.environ, OUTPUT_DIR=str(output), PROJECT_NAME="study", RUN_ID="trial",
               ATTEMPT_ID="attempt-001", SOURCE_ID="source." + "a" * 64)
    executed = subprocess.run([sys.executable, str(source / "train.py"), "--steps", "4"],
                              env=env, capture_output=True, text=True, check=True)
    assert json.loads(executed.stdout)["run_id"] == "trial"
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [item["step"] for item in metrics] == [1, 2, 3, 4]
    with tarfile.open(fileobj=io.BytesIO(source_archive(source)), mode="r:gz") as archive:
        assert archive.getnames() == ["train.py"]
        assert archive.extractfile("train.py").read() == program


def test_module_entry_point_needs_no_configuration_for_help_or_version(tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith("ML_EXPD_")}
    for arguments, expected in ((["--version"], __version__), (["--help"], "download")):
        result = subprocess.run([sys.executable, "-m", "ml_exp_client", *arguments],
                                cwd=tmp_path, env=env, capture_output=True, text=True)
        assert result.returncode == 0 and expected in result.stdout
    code = "import ml_exp_client,sys; assert not ({'ml_exp_server','experiment_control','fastapi','boto3','yaml'} & set(sys.modules))"
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_client_configuration_errors_are_local_and_do_not_print_tokens(monkeypatch, capsys):
    monkeypatch.setenv("ML_EXPD_API_URL", "http://remote.example")
    monkeypatch.setenv("ML_EXPD_API_TOKEN", "private-test-token")
    assert main(["check"]) == 2
    assert "HTTPS" in capsys.readouterr().err
    monkeypatch.setenv("ML_EXPD_API_URL", "https://api.example")
    monkeypatch.delenv("ML_EXPD_API_TOKEN")
    monkeypatch.delenv("ML_EXPD_API_TOKEN_FILE", raising=False)
    assert main(["check"]) == 2
    assert "set ML_EXPD_API_TOKEN" in capsys.readouterr().err
