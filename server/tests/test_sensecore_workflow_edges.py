"""Build provenance, capability scope, limits and interrupted filesystem writes."""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_exp_server import container_controller
from ml_exp_server.data_assets import AssetStore
from ml_exp_server.dockerfile_build import DOCKERFILE_RECIPE, INTERNAL, inspect_dockerfile
from ml_exp_server.image_builder import BUILD_LOG, ImageBuilder, MANIFEST_TYPE, bundle_id
from ml_exp_server.container_execution import DockerfileRuntimeSpec
from ml_exp_server.source_imports import seal_tree
from ml_exp_server.source_revisions import _tree_digest
from tests.test_container_api import archive, client, import_source
from tests.test_sensecore_data_workflow import stored, custom_runtime, controller, put_asset, BASE


def builder(tmp_path, ignore=True):
    pending = tmp_path / "sources/demo/pending"; tree = pending / "tree"; tree.mkdir(parents=True)
    (tree / "Dockerfile").write_text(f"FROM {BASE}\nRUN echo user-dockerfile\nCOPY train.py /custom/train.py\n")
    (tree / "train.py").write_text("print('training')")
    if ignore:
        (tree / "Dockerfile.dockerignore").write_text("*\n!train.py\n")
    digest = _tree_digest(tree); source_id = "source." + digest[7:]; pending.rename(pending.with_name(source_id))
    source = pending.with_name(source_id); (source / "source.json").write_text(json.dumps({"project":"demo","tree_digest":digest})); seal_tree(source)
    value = ImageBuilder({"state_root": str(tmp_path / "state"), "source_root": str(tmp_path / "sources"), "repository": "registry.example/runtime", "publisher": "buildkit", "allow_dockerfile_builds": True})
    request = {"operation":"build","project":"demo","source_id":source_id,"base_image":BASE,"dockerfile":"Dockerfile","packaging_revision":DOCKERFILE_RECIPE}
    return value, request, source


@pytest.mark.parametrize("ignore", [True, False])
def test_custom_dockerfile_is_built_with_managed_source_and_immutable_receipt(tmp_path, ignore):
    value, request, source = builder(tmp_path, ignore)
    raw = json.dumps({"mediaType":MANIFEST_TYPE,"config":{"digest":"sha256:"+"c"*64}})
    digest = "sha256:" + hashlib.sha256(raw.encode()).hexdigest()
    def docker(args, **kwargs):
        context = Path(args[-1]); recipe = (context / "Dockerfile").read_text()
        assert "RUN echo user-dockerfile" in recipe and "--network=default" in args
        assert (context / INTERNAL / "source/train.py").read_text() == "print('training')"
        assert (context / INTERNAL / "worker.py").is_file() and (context / INTERNAL / "artifacts.py").is_file()
        assert "!ml-expd-build-internal/**" in (context / ".dockerignore").read_text()
        Path(args[args.index("--metadata-file")+1]).write_text(json.dumps({"containerimage.digest":digest,"containerimage.config.digest":"sha256:"+"c"*64}))
        return ""
    value._docker = docker; value._skopeo = lambda *a, **k: raw
    result = value.request(request)
    from ml_exp_server.worker_contract import CAPABILITIES
    assert result["image"].endswith(digest) and result["capabilities"] == list(CAPABILITIES)
    assert value.request({**request,"operation":"get"}) == result
    logs = value.request({**request,"operation":"logs"}); assert logs["lines"] == [] and not logs["truncated"]
    path = value.root / (result["bundle_id"] + ".log"); path.write_text("x"*9000+"\nlast build line\n")
    logs = value.request({**request,"operation":"logs"}); assert logs["truncated"] and logs["lines"][-1] == "last build line"


@pytest.mark.parametrize("case", ["ambiguous", "disabled", "archive", "wrong-base", "disallowed-other-base"])
def test_builder_requires_opt_in_and_checks_all_external_bases(tmp_path, case):
    value, request, source = builder(tmp_path)
    if case == "ambiguous": request["requirements"] = "requirements.txt"
    if case == "disabled": value.config["allow_dockerfile_builds"] = False
    if case == "archive": value.config["publisher"] = "archive"
    if case == "wrong-base": request["base_image"] = BASE.replace("a"*64,"b"*64)
    if case == "disallowed-other-base":
        tree=source/"tree"; tree.chmod(0o700)
        (tree/"Dockerfile").chmod(0o600); (tree/"Dockerfile").write_text("FROM other.example/base@sha256:"+"b"*64+" AS build\nFROM "+BASE+"\n")
        digest=_tree_digest(tree); new_id="source."+digest[7:]; (source/"source.json").chmod(0o600)
        (source/"source.json").write_text(json.dumps({"project":"demo","tree_digest":digest})); seal_tree(source)
        source.chmod(0o700); source.rename(source.with_name(new_id)); source.with_name(new_id).chmod(0o500)
        request["source_id"]=new_id; value.config["base_image_prefixes"]=["registry.example/"]
    with pytest.raises(ValueError): value.request(request)


def test_command_log_is_thread_scoped_and_preserves_errors(tmp_path):
    value, _, _ = builder(tmp_path)
    log = tmp_path / "build.log"; token = BUILD_LOG.set(log)
    try:
        assert value._command(["/bin/sh","-c","echo build-output"]) == ""
        assert "build-output" in log.read_text()
        with pytest.raises(ValueError,match="OCI packaging"):
            value._command(["/bin/sh","-c","echo build-failed; exit 3"])
    finally: BUILD_LOG.reset(token)
    assert "build-failed" in log.read_text()


@pytest.mark.parametrize("case", ["missing", "link", "large", "reserved"])
def test_dockerfile_source_boundaries(tmp_path, case):
    if case != "missing": (tmp_path/"Dockerfile").write_text("FROM "+BASE)
    if case == "link": (tmp_path/"Dockerfile").unlink(); (tmp_path/"Dockerfile").symlink_to(tmp_path.parent/"elsewhere")
    if case == "large": (tmp_path/"Dockerfile").write_bytes(b"x"*(256*1024+1))
    if case == "reserved": (tmp_path/INTERNAL).mkdir()
    with pytest.raises(ValueError): inspect_dockerfile(tmp_path,"Dockerfile")


def test_runtime_recipe_without_dockerfile_and_duplicate_mounts_fail(client, monkeypatch):
    with pytest.raises(ValueError): DockerfileRuntimeSpec(source_id="source."+"a"*64,image=BASE,entrypoint=["python3"],packaging_revision=DOCKERFILE_RECIPE)
    bundle=custom_runtime(client,monkeypatch)
    binding={"asset_id":"asset."+"a"*64,"mount_path":"/inputs/data"}
    response=client.post("/api/projects/demo/runs",json={"run_id":"trial","runtime_id":bundle["runtime_id"],"executor":"cloud","inputs":[binding,binding]})
    assert response.status_code == 422
    response=client.post("/api/projects/demo/runs",json={"run_id":"managed-wyd","runtime_id":bundle["runtime_id"],"executor":"gpu"})
    assert response.status_code==200
    root=Path(client.app.state.runtime.project("demo").base_dir)
    import yaml
    campaign=yaml.safe_load((root/"experiments/campaigns/run-managed-wyd.yaml").read_text())
    campaign["local_root"]=str(client.app.state.runtime.config.project_run_root_path("demo"))
    ctl=container_controller.Controller(campaign,"managed-wyd","attempt-001")
    ctl.prepare()
    assert ctl.store.load_manifest()["execution"]["managed_io"] is True


def test_data_metadata_and_limits_fail_closed(client,stored,tmp_path):
    store=AssetStore(stored[0],client.app.state.runtime.config.project_registry_root_path())
    with pytest.raises(ValueError): store.list("../invalid")
    with pytest.raises(ValueError): store.receive("demo",io.BytesIO(b"x"),"invalid",1)
    with pytest.raises(ValueError): store.receive("demo",io.BytesIO(),"a"*64,0)
    data=archive({"nested/data":b"abc"}); result=put_asset(client,data).json()
    path=store.root/"demo"/result["asset_id"]/"asset.json"; original=path.read_text(); path.chmod(0o600)
    changed=json.loads(original); changed["project"]="other"; path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match="metadata identity"): store.read("demo",result["asset_id"])
    changed=json.loads(original); changed["archive_bytes"]+=1; path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match="already bound"): store.receive("demo",io.BytesIO(data),result["sha256"],len(data))
    path.write_text(original); path.chmod(0o400)
    (store.root/"demo"/"asset.invalid").mkdir(); (store.root/"demo"/("asset."+"c"*64+".lock")).touch()
    assert len(store.list("demo")["assets"])==1


def test_upload_headers_stream_limits_and_snapshot_capabilities(client,stored,monkeypatch):
    data=archive({"data":b"abc"}); params={"project":"demo","sha256":hashlib.sha256(data).hexdigest()}
    for length in ("invalid","-1"):
        assert client.post("/api/assets/archive",params=params,content=data,headers={"Content-Length":length}).status_code==400
    assert client.post("/api/assets/archive",params=params,content=data,headers={"Content-Length":str(3*1024**2)}).status_code==413
    monkeypatch.setattr("ml_exp_server.api.asset_routes.shutil.disk_usage",lambda p:SimpleNamespace(free=1))
    assert client.post("/api/assets/archive",params=params,content=data).status_code==507
    monkeypatch.undo()
    service=client.app.state.runtime.config.container_execution
    assert client.put("/api/snapshot-transfers/demo/trial/attempt-001",content=data).status_code==401
    assert client.put("/api/snapshot-transfers/demo/trial/attempt-001",content=data,headers={"Authorization":"Bearer invalid"}).status_code==401
    service.artifact_store_file=None
    assert client.get("/api/storage-limits").status_code==404


def test_backend_asset_preflight_verifies_frozen_data(client,stored,monkeypatch,capsys,tmp_path):
    asset=put_asset(client).json(); bundle=custom_runtime(client,monkeypatch)
    ctl=controller(client,bundle,inputs=[{"asset_id":asset["asset_id"],"mount_path":"/inputs/data"}])
    path=tmp_path/"campaign.yaml"
    import yaml
    path.write_text(yaml.safe_dump(ctl.campaign))
    args=[str(path),"assets-verify","--run","trial"]
    container_controller.cli(args)
    assert json.loads(capsys.readouterr().out)[0]["inputs"]==1
    ctl.campaign["runs"][0]["inputs"][0]["sha256"]="c"*64
    path.write_text(yaml.safe_dump(ctl.campaign))
    with pytest.raises(SystemExit): container_controller.cli(args)
    assert "controller failed" in capsys.readouterr().err.lower()


def test_snapshots_require_explicit_attempt_policy_and_repeat_issue_is_stable(client,stored,monkeypatch):
    bundle=custom_runtime(client,monkeypatch); ctl=controller(client,bundle)
    store=AssetStore(stored[0],client.app.state.runtime.config.project_registry_root_path())
    url,token,limit=store.objects.issue("demo","trial","attempt-001",ctl.root,["**/*"])
    data=archive({"checkpoint.pt":b"weights"})
    with pytest.raises(ValueError,match="not enabled"):
        store.snapshot("demo","trial","attempt-001",token,io.BytesIO(data),len(data))
    assert client.put("/api/snapshot-transfers/demo/trial/attempt-001",content=data,headers={"Authorization":"Bearer "+token}).status_code==401
    assert store.objects.issue("demo","trial","attempt-001",ctl.root,["**/*"],checkpoint_upload=True)==(url,token,limit)
    assert store.objects.issue("demo","trial","attempt-001",ctl.root,["**/*"],checkpoint_upload=True)==(url,token,limit)


def test_streamed_upload_is_bounded_even_without_content_length(client,stored):
    chunks=iter([b"a"*(1024**2),b"b"*(1024**2),b"c"])
    response=client.post("/api/assets/archive?project=demo&sha256="+"a"*64,content=chunks)
    assert response.status_code==413


def test_build_logs_and_oversized_input_manifest(client,stored,monkeypatch):
    bundle=custom_runtime(client,monkeypatch)
    def logs(socket,payload):
        assert payload["operation"]=="logs" and payload["dockerfile"]=="Dockerfile"
        return {"lines":["client Dockerfile RUN completed"],"truncated":False}
    monkeypatch.setattr("ml_exp_server.api.container_routes.builder_request",logs)
    endpoint="/api/projects/demo/runtimes/"+bundle["runtime_id"]+"/logs"
    assert client.get(endpoint).json()["lines"]==["client Dockerfile RUN completed"]
    files={f"shards/{i:04d}-"+"x"*90:b"token" for i in range(250)}
    asset=put_asset(client,archive(files)).json()
    response=client.post("/api/projects/demo/runs",json={"run_id":"too-many","runtime_id":bundle["runtime_id"],"executor":"cloud","inputs":[{"asset_id":asset["asset_id"],"mount_path":"/inputs/data"}]})
    assert response.status_code==409
