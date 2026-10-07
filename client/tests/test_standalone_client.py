"""Offline client behavior, packaged resources and dependency boundaries."""
import importlib.metadata
import json
import os
import subprocess
import sys
import tarfile
import io
from pathlib import Path

from ml_exp_client import source_archive, __version__
from ml_exp_client.cli import main
import pytest


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
    assert all(item["name"] == "loss" and item["unit"] == "dimensionless" for item in metrics)
    import ml_exp_client.metrics
    assert (source / "ml_exp_metrics.py").read_bytes() == Path(ml_exp_client.metrics.__file__).read_bytes()
    with tarfile.open(fileobj=io.BytesIO(source_archive(source)), mode="r:gz") as archive:
        assert archive.getnames() == ["Dockerfile", "ml_exp_metrics.py", "train.py"]
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


def test_pack_uses_only_dockerfile_and_checks_it_before_upload(tmp_path, monkeypatch, capsys):
    calls = []
    class API:
        def __init__(self, *args): pass
        def negotiate(self): return {"capabilities": ["dockerfile-build.v1"]}
        def call(self, path, **kwargs):
            calls.append((path, kwargs))
            if "source-imports" in path: return {"source_id": "source." + "a" * 64}
            if path.endswith("prepare"): return {"project": "demo", "runtime_id": "runtime." + "b" * 64, "status": "PREPARED", "confirmation": "BUILD example"}
            return {}
        def wait(self, *args, **kwargs): return {"project": "demo", "runtime_id": "runtime." + "b" * 64, "status": "READY"}
    monkeypatch.setattr("ml_exp_client.cli.Client", API)
    (tmp_path / "Dockerfile").write_text("FROM registry.example/python@sha256:" + "a" * 64 + "\n")
    state = tmp_path / "runtime.json"
    assert main(["pack", "--project", "demo", "--source", str(tmp_path), "--state", str(state)]) == 0
    prepared = next(kwargs["data"] for path, kwargs in calls if path.endswith("prepare"))
    assert prepared["dockerfile"] == "Dockerfile" and "image" not in prepared
    assert "READY" in capsys.readouterr().out


def test_new_build_options_require_server_capabilities_before_upload(tmp_path, monkeypatch, capsys):
    class API:
        def __init__(self, *args):
            pass
        def negotiate(self):
            return {"capabilities": []}
    monkeypatch.setattr("ml_exp_client.cli.Client", API)
    assert main(["pack", "--project", "demo", "--source", str(tmp_path), "--state", str(tmp_path / "runtime.json"),
                 "--dockerfile", "Dockerfile"]) == 2
    assert "capability" in capsys.readouterr().err and not (tmp_path / "runtime.json").exists()
    saved = tmp_path / "runtime.json"
    saved.write_text(json.dumps({"project": "demo", "runtime_id": "runtime." + "a" * 64}))
    assert main(["create", "--runtime-state", str(saved), "--run", "trial", "--executor", "gpu",
                 "--data-preparation", '{"script":"download.py"}']) == 2
    assert "data-preparation.v1" in capsys.readouterr().err


def test_dockerfile_data_upload_and_input_binding_use_only_http(tmp_path,monkeypatch,capsys):
    calls=[]
    class API:
        def __init__(self,*a):pass
        def negotiate(self):return {"capabilities":["dockerfile-build.v1","data-assets.v1","data-preparation.v1"]}
        def call(self,path,**kwargs):
            calls.append((path,kwargs))
            if path=="/api/storage-limits":return {"asset_archive_bytes":1000000}
            if path.startswith("/api/assets/archive"):
                assert hasattr(kwargs["raw"],"read") and kwargs["length"]>0
                return {"project":"demo","asset_id":"asset."+"a"*64,"status":"READY"}
            if "source-imports" in path:return {"project":"demo","source_id":"source."+"b"*64}
            if path.endswith("prepare"):return {"project":"demo","runtime_id":"runtime."+"c"*64,"status":"PREPARED","confirmation":"BUILD test"}
            return {}
        def wait(self,*a,**k):return {"project":"demo","runtime_id":"runtime."+"c"*64,"status":"READY"}
    monkeypatch.setattr("ml_exp_client.cli.Client",API)
    source=tmp_path/"source";source.mkdir();(source/"Dockerfile").write_text("FROM registry.example/python@sha256:" + "a" * 64)
    runtime=tmp_path/"runtime.json"
    assert main(["pack","--project","demo","--source",str(source),"--dockerfile","Dockerfile","--state",str(runtime)])==0
    prepared=next(kwargs["data"] for path,kwargs in calls if path.endswith("prepare"))
    assert prepared["dockerfile"]=="Dockerfile" and "image" not in prepared
    data=tmp_path/"data";data.mkdir();(data/"tokens.bin").write_bytes(b"tokens")
    asset=tmp_path/"data.json"
    assert main(["asset-upload","--project","demo","--directory",str(data),"--state",str(asset)])==0
    assert json.loads(asset.read_text())["status"]=="READY"
    bindings=json.dumps([{"asset_id":"asset."+"a"*64,"mount_path":"/inputs/data"}])
    assert main(["create","--runtime-state",str(runtime),"--run","trial","--executor","sensecore-1gpu","--inputs",bindings,"--checkpoint-interval","5","--data-preparation",'{"script":"download.py"}'])==0
    definition=next(kwargs["data"] for path,kwargs in calls if path.endswith("/runs"))
    assert definition["inputs"]==json.loads(bindings) and definition["checkpoint_upload"]=={"interval_seconds":5}
    assert definition["data_preparation"] == {"script": "download.py"}
    assert main(["assets","--project","demo"])==0
    assert main(["snapshots","--project","demo","--run","trial","--attempt","attempt-001"])==0
    assert main(["asset-upload","--project","demo","--directory",str(data),"--state",str(asset)])==2
    assert "state file exists" in capsys.readouterr().err


def test_data_archives_are_compressed_reproducible_and_preserve_file_hashes(tmp_path):
    from ml_exp_client.api import data_archive
    data=tmp_path/"data";data.mkdir();body=b"training data\n"*10000
    (data/"tokens.txt").write_bytes(body)
    first=io.BytesIO();second=io.BytesIO()
    one=data_archive(data,first);two=data_archive(data,second)
    assert one==two and first.getvalue()==second.getvalue() and one[1]<len(body)/10
    with tarfile.open(fileobj=first,mode="r:gz") as archive:
        assert archive.getnames()==["tokens.txt"] and archive.extractfile("tokens.txt").read()==body
