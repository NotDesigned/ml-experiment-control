"""Metrics are available during training and survive a reporting restart."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from ml_exp_client import MetricWriter
from ml_exp_client.cli import main
from ml_exp_client.api import ClientError
from ml_exp_client.status import read_attempt


def test_append_is_immediately_readable_and_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    writer = MetricWriter()
    writer.log("train_loss", 4.1, unit="nats/token", step=100,
               dataset_id="training-v1", variant_id="baseline")
    first = writer.path.read_bytes()
    assert first.endswith(b"\n")
    MetricWriter(tmp_path).log("validation_loss", 3.9, unit="nats/token", step=100,
                              checkpoint_id="sha256:checkpoint", dataset_id="validation-v1",
                              numerator=390, denominator=100)
    assert writer.path.read_bytes().startswith(first)
    records = [json.loads(line) for line in writer.path.read_text().splitlines()]
    assert [r["name"] for r in records] == ["train_loss", "validation_loss"]
    assert records[1]["numerator"] == 390 and records[1]["denominator"] == 100


@pytest.mark.parametrize("value,spelling", [(float("nan"), "nan"), (float("inf"), "inf"), (-float("inf"), "-inf")])
def test_nonfinite_value_is_preserved_for_server_failure_classification(tmp_path, value, spelling):
    writer = MetricWriter(tmp_path)
    writer.log("loss", value, unit="nats/token", step=1)
    assert json.loads(writer.path.read_text())["value"] == spelling
    assert "NaN" not in writer.path.read_text() and "Infinity" not in writer.path.read_text()


@pytest.mark.parametrize("name,unit", [("", "nats/token"), ("loss", " "), ("bad\nname", "unit"),
                                      ("loss", "bad\x00unit"), ("x" * 129, "unit"), (None, "unit")])
def test_invalid_name_or_unit_does_not_append(tmp_path, name, unit):
    writer = MetricWriter(tmp_path)
    with pytest.raises(ValueError, match="name and unit"):
        writer.log(name, 1, unit=unit)
    assert not writer.path.exists()


def test_no_default_unit_or_implicit_output_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    with pytest.raises(ValueError, match="OUTPUT_DIR"):
        MetricWriter()
    writer = MetricWriter(tmp_path)
    with pytest.raises(TypeError):
        writer.log("loss", 1)
    with pytest.raises(TypeError, match="scalar"):
        writer.log("loss", True, unit="fraction")
    with pytest.raises(ValueError, match="context"):
        writer.log("loss", 1, unit="nats/token", arbitrary="not a metric field")
    writer.log("metric", None, unit="fraction", status="MISSING", error="NOT_EVALUATED", epoch=None)
    record = json.loads(writer.path.read_text())
    assert record["value"] is None and record["status"] == "MISSING" and "epoch" not in record


def test_symlink_and_oversize_records_are_rejected_without_damaging_history(tmp_path):
    writer = MetricWriter(tmp_path)
    writer.log("loss", 1, unit="nats/token", step=1)
    original = writer.path.read_bytes()
    with pytest.raises(ValueError, match="64 KiB"):
        writer.log("loss", 2, unit="nats/token", error="x" * 65536)
    assert writer.path.read_bytes() == original
    target = tmp_path / "original.jsonl"
    writer.path.rename(target)
    writer.path.symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        writer.log("loss", 2, unit="nats/token")
    assert target.read_bytes() == original


def test_generated_training_source_publishes_before_process_exit(tmp_path, capsys):
    source = tmp_path / "source"
    assert main(["init", str(source)]) == 0
    capsys.readouterr()
    output = tmp_path / "outputs"
    env = dict(os.environ, OUTPUT_DIR=str(output), PROJECT_NAME="study", RUN_ID="trial",
               ATTEMPT_ID="attempt-001", SOURCE_ID="source.test")
    child = subprocess.Popen([sys.executable, str(source / "train.py"), "--steps", "2", "--step-delay", "0.5"],
                             env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        path = output / "metrics.jsonl"
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert path.exists() and child.poll() is None
        first = json.loads(path.read_text().splitlines()[0])
        assert first["step"] == 1 and first["unit"] == "dimensionless"
        stdout, stderr = child.communicate(timeout=10)
        assert child.returncode == 0, stderr
        assert len(path.read_text().splitlines()) == 2
        assert json.loads(stdout)["steps"] == 2
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()


def test_summary_is_an_explicit_result_in_the_same_append_only_stream(tmp_path):
    writer = MetricWriter(tmp_path)
    writer.log("loss", 4, unit="nats/token", step=10)
    writer.summary("loss", 3, unit="nats/token", step=5, checkpoint_id="selected")
    records = [json.loads(line) for line in writer.path.read_text().splitlines()]
    assert "summary" not in records[0]
    assert records[1] == {"name": "loss", "value": 3, "unit": "nats/token", "step": 5,
                          "checkpoint_id": "selected", "summary": True}
    for invalid in (None, 1, "true"):
        with pytest.raises(ValueError, match="boolean"):
            writer.log("loss", 0, unit="nats/token", summary=invalid)
    assert len(writer.path.read_text().splitlines()) == 2


def test_named_metric_reads_negotiate_compact_format_and_preserve_names(monkeypatch, capsys):
    from urllib.parse import parse_qs, urlsplit
    calls = []
    payload = {"project": "demo", "run_id": "trial", "attempt_id": "attempt-001", "records": []}
    class Reader:
        capabilities = ["metric-records.v1"]
        def negotiate(self): return {"capabilities": self.capabilities}
        def call(self, path):
            calls.append(path)
            return payload
    reader = Reader()
    assert read_attempt(reader, "metrics", "demo", "trial", "attempt-001", names=["loss,validation"],
                        keys="train_loss, accuracy") is payload
    assert parse_qs(urlsplit(calls[-1]).query) == {"max_points": ["2000"], "format": ["records"],
        "name": ["loss,validation", "train_loss", "accuracy"]}
    monkeypatch.setattr("ml_exp_client.cli.Client", lambda *args: reader)
    assert main(["metrics", "--project", "demo", "--run", "trial", "--attempt", "attempt-001",
                 "--name", "loss,validation", "--name", "accuracy"]) == 0
    assert json.loads(capsys.readouterr().out) == payload
    assert parse_qs(urlsplit(calls[-1]).query)["name"] == ["loss,validation", "accuracy"]
    read_attempt(reader, "metrics", "demo", "trial", "attempt-001")
    assert "name=" not in calls[-1] and "format=records" in calls[-1]
    for invalid in ([""], ["bad\nname"], [True], ["x" * 129], ["loss"] * 257):
        with pytest.raises(ClientError, match="name filters"):
            read_attempt(reader, "metrics", "demo", "trial", "attempt-001", names=invalid)
    reader.capabilities = []
    with pytest.raises(ClientError, match="metric-records.v1"):
        read_attempt(reader, "metrics", "demo", "trial", "attempt-001", names=["loss"])
