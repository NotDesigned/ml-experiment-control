"""Environment selection, dependency isolation and immutable build receipts."""
import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import yaml

from ml_exp_server import dependency_install as installer
from ml_exp_server.container_execution import RuntimeSpec
from ml_exp_server.environment_build import DEPENDENCY_RECIPE, dockerfile, inspect_requirements, installer_digest
from ml_exp_server.image_builder import ImageBuilder, MANIFEST_TYPE, bundle_id
from ml_exp_server.source_imports import seal_tree
from ml_exp_server.source_revisions import _tree_digest
from tests.test_container_api import archive, client, import_source


BASE = "registry.example/base@sha256:" + "a" * 64


def catalogue(client, tmp_path, payload=None):
    path = tmp_path / "environments.yaml"
    path.write_text(yaml.safe_dump(payload if payload is not None else {"environments": {
        "torch": {"image": BASE, "versions": {"torch": "2.10.0+cu126"}, "private_field": "not-public"}}}))
    client.app.state.runtime.config.container_execution.environments_file = str(path)
    return path


def prepare(client, content=b"colorama==0.4.6\n", **changes):
    source = import_source(client, archive({"train.py": b"print('training')\n", "requirements.txt": content}))
    definition = {"source_id": source["source_id"], "image": BASE,
                  "requirements": "requirements.txt", "entrypoint": ["python3", "train.py"], **changes}
    return client.post("/api/projects/demo/runtimes/prepare", json=definition)


def test_catalogue_and_environment_resolution_freeze_actual_image(client, tmp_path):
    assert client.get("/api/environments").json() == {"environments": []}
    path = catalogue(client, tmp_path)
    entries = client.get("/api/environments").json()["environments"]
    assert entries == [{"id": "torch", "image": BASE, "versions": {"torch": "2.10.0+cu126"}}]
    source = import_source(client)
    request = {"source_id": source["source_id"], "environment_id": "torch", "entrypoint": ["python3", "train.py"]}
    response = client.post("/api/projects/demo/runtimes/prepare", json=request)
    assert response.status_code == 200, response.text
    value = response.json()
    assert value["spec"]["image"] == BASE and "environment_id" not in value["spec"]
    assert value["environment"] == entries[0] and "RUN " not in value["dockerfile"]
    path.write_text(yaml.safe_dump({"environments": {"torch": {"image": BASE.replace("a" * 64, "c" * 64)}}}))
    endpoint = "/api/projects/demo/runtimes/" + value["runtime_id"]
    client.post(endpoint + "/execute", json={"confirmation": value["confirmation"]})
    assert client.get(endpoint).json()["spec"]["image"] == BASE
    assert client.post("/api/projects/demo/runtimes/prepare", json={**request, "environment_id": "missing"}).status_code == 404


@pytest.mark.parametrize("payload", [[], {"environments": []}, {"environments": {"../bad": {"image": BASE}}},
                                    {"environments": {"bad": "string"}}, {"environments": {"bad": {"image": "latest"}}}])
def test_invalid_catalogue_cannot_publish_unvalidated_entries(client, tmp_path, payload):
    catalogue(client, tmp_path, payload)
    response = client.get("/api/environments")
    assert response.status_code == (200 if payload == [] else 409)


@pytest.mark.parametrize("changes", [{"image": None}, {"environment_id": "torch"},
                                     {"requirements": "../requirements.txt"}, {"requirements": "/etc/passwd"},
                                     {"requirements": "a\\b"}, {"requirements": "a\x00b"},
                                     {"requirements": ".private/requirements.txt"}, {"requirements": ""},
                                     {"requirements": "a/./requirements.txt"}])
def test_invalid_runtime_build_definition_fails_before_execution(client, changes):
    assert prepare(client, **changes).status_code == 422


def test_explicit_null_fields_and_dependency_recipe_requirements():
    spec = RuntimeSpec(source_id="source." + "a" * 64, image=None, environment_id="torch", requirements=None, entrypoint=["python3"])
    assert spec.environment_id == "torch"
    with pytest.raises(ValueError, match="requires a requirements"):
        RuntimeSpec(source_id=spec.source_id, image=BASE, entrypoint=["python3"], packaging_revision=DEPENDENCY_RECIPE)


@pytest.mark.parametrize("text", [b"", b"# comment\n", b"colorama>=0.4\n", b"colorama==*\n", b"--index-url https://private\n",
                                b"-r secret.txt\n", b"pkg @ https://remote/wheel\n", b"pkg==1; python_version>'3'\n",
                                b"x==1\n" * 1001, b"x" * (256 * 1024 + 1), b"\xff\n",
                                b"x==1 --hash=sha256:" + b"a" * 64 + b"\ny==2\n"])
def test_dependency_file_rejects_dynamic_and_unbounded_inputs(client, text):
    assert prepare(client, text).status_code == 409


def test_missing_and_escaping_requirements_files(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        inspect_requirements(tmp_path, "missing.txt")
    outside = tmp_path.parent / "outside-requirements.txt"
    outside.write_text("x==1\n")
    (tmp_path / "link.txt").symlink_to(outside)
    with pytest.raises(ValueError, match="missing"):
        inspect_requirements(tmp_path, "link.txt")
    (tmp_path / "directory").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="missing"):
        inspect_requirements(tmp_path, "directory/outside-requirements.txt")


@pytest.mark.parametrize("tamper", [None, "dependencies", "dockerfile_sha256", "installer_sha256", "bundle_id"])
def test_dependency_runtime_accepts_only_exact_build_receipt(client, monkeypatch, tamper):
    response = prepare(client)
    assert response.status_code == 200, response.text
    value = response.json()
    assert value["spec"]["packaging_revision"] == DEPENDENCY_RECIPE
    assert value["dependencies"]["sha256"] == hashlib.sha256(b"colorama==0.4.6\n").hexdigest()
    assert 'RUN ["python3"' in value["dockerfile"]
    def build(socket, request):
        assert request["requirements"] == "requirements.txt"
        result = {"project": request["project"], "source_id": request["source_id"], "base_image": BASE,
                  "image": "registry.example/results@sha256:" + "b" * 64,
                  "bundle_id": bundle_id("demo", request["source_id"], BASE, request["requirements"]),
                  "dependencies": value["dependencies"], "installer_sha256": installer_digest(),
                  "dockerfile_sha256": hashlib.sha256(value["dockerfile"].encode()).hexdigest()}
        if tamper:
            result[tamper] = "wrong"
        return result
    monkeypatch.setattr("ml_exp_server.container_execution.builder_request", build)
    endpoint = "/api/projects/demo/runtimes/" + value["runtime_id"]
    client.post(endpoint + "/execute", json={"confirmation": value["confirmation"]})
    completed = client.get(endpoint).json()
    assert completed["status"] == ("RECONCILE_REQUIRED" if tamper else "READY")
    if not tamper:
        for executor in ("gpu", "cloud"):
            frozen = client.post("/api/projects/demo/runs", json={"run_id": executor, "runtime_id": value["runtime_id"], "executor": executor}).json()
            assert frozen["image"] == completed["image"]


def test_builder_installs_dependencies_before_copying_source_and_verifies_registry(tmp_path):
    source = tmp_path / "sources/demo/pending"
    tree = source / "tree"
    tree.mkdir(parents=True)
    content = "# locked dependency\ncolorama==0.4.6 \\\n    --hash=sha256:" + "a" * 64 + "\n"
    (tree / "requirements.lock").write_text(content)
    (tree / "Dockerfile").write_text("RUN execute-untrusted-project-file\n")
    digest = _tree_digest(tree)
    source_id = "source." + digest[7:]
    source.rename(source.with_name(source_id))
    source = source.with_name(source_id)
    (source / "source.json").write_text(json.dumps({"project": "demo", "tree_digest": digest}))
    seal_tree(source)
    builder = ImageBuilder({"state_root": str(tmp_path / "state"), "source_root": str(tmp_path / "sources"),
                            "repository": "registry.example/results", "publisher": "buildkit", "allow_dependency_builds": True})
    raw = json.dumps({"mediaType": MANIFEST_TYPE, "config": {"digest": "sha256:" + "c" * 64}})
    published = "sha256:" + hashlib.sha256(raw.encode()).hexdigest()
    def build(args, **kwargs):
        context = Path(args[-1])
        recipe = (context / "Dockerfile").read_text()
        assert "--network=default" in args and "execute-untrusted" not in recipe
        assert recipe.index("RUN ") < recipe.index("COPY source/")
        assert (context / "requirements.txt").read_text() == content
        assert hashlib.sha256((context / "dependency_install.py").read_bytes()).hexdigest() == installer_digest()
        Path(args[args.index("--metadata-file") + 1]).write_text(json.dumps({"containerimage.digest": published, "containerimage.config.digest": "sha256:" + "c" * 64}))
        return ""
    builder._docker = build
    builder._skopeo = lambda *args, **kwargs: raw
    request = {"operation": "build", "project": "demo", "source_id": source_id, "base_image": BASE, "requirements": "requirements.lock"}
    receipt = builder.request(request)
    assert receipt["image"].endswith(published) and receipt["dependencies"]["hash_locked"]
    assert receipt["bundle_id"] != bundle_id("demo", source_id, BASE)
    assert builder.request({**request, "operation": "get"}) == receipt
    for settings in ({"allow_dependency_builds": False}, {"publisher": "archive"}):
        builder.config.update(settings)
        request["requirements"] = "different.lock"
        with pytest.raises(ValueError, match="enabled BuildKit"):
            builder.request(request)


@pytest.mark.parametrize("case", ["success", "framework-change", "install-failure", "entrypoint"])
def test_installer_runs_only_inside_container_and_preserves_gpu_framework(tmp_path, monkeypatch, case):
    calls = []
    distributions = [SimpleNamespace(metadata={"Name": name}, version=version) for name, version in
                     (("torch", "2.10.0+cu126"), ("nvidia-cublas-cu12", "12.6"), ("", "ignored"), ("other_pkg", "1"), ("other_pkg", "0"))]
    count = 0
    def installed():
        nonlocal count
        count += 1
        result = list(distributions)
        if count > 1 and case == "framework-change":
            result[0] = SimpleNamespace(metadata={"Name": "torch"}, version="cpu")
        return result
    monkeypatch.setattr(installer.importlib.metadata, "distributions", installed)
    original_path = Path
    monkeypatch.setattr(installer, "Path", lambda value: original_path(tmp_path / value.lstrip("/")))
    (tmp_path / "tmp").mkdir()
    def run(command, **kwargs):
        calls.append(command)
        if case == "install-failure":
            raise RuntimeError("installation rejected")
    monkeypatch.setattr(installer.subprocess, "run", run)
    if case == "framework-change":
        with pytest.raises(ValueError, match="base GPU framework"):
            installer.main()
    elif case == "install-failure":
        with pytest.raises(RuntimeError):
            installer.main()
    elif case == "entrypoint":
        # Execute the CLI guard without touching host paths or invoking pip.
        import ast
        source = Path(installer.__file__).read_text()
        tree = ast.parse(source)
        guard = ast.Module(body=[tree.body[-1]], type_ignores=[])
        exec(compile(guard, installer.__file__, "exec"), {"__name__": "__main__", "main": installer.main})
    else:
        installer.main()
    assert "--only-binary=:all:" in calls[0] and "--constraint" in calls[0]
    if case in {"success", "entrypoint"}:
        metadata = json.loads((tmp_path / "usr/local/share/ml-expd/environment.json").read_text())
        assert metadata["packages"]["torch"] == "2.10.0+cu126"
        assert "other-pkg" in metadata["packages"]
        assert metadata["packages"]["other-pkg"] == "1"
