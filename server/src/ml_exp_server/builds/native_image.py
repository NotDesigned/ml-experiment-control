"""Small image effect adapter; native Runtime identities belong to Rust.

This adapter requires its own explicitly configured native worker executable
and source namespace. It never imports a training environment on the API host.
"""
import fcntl
import hashlib
import os
import json
from pathlib import Path
import re
import shutil
import stat
import tempfile

from .builder_client import IMAGE, PROJECT, BuildReceiptMissingError
from .image_build_context import BUILD_LOG, BUILD_PROGRESS, BUILD_CACHE
from ..dockerfile_build import inspect_dockerfile, INTERNAL
from ..projects.source_revisions import _tree_digest
from ..runs.execution_progress import record_progress, progress_view
from ..storage import atomic_json

RECIPE = "native-image-recipe.v1"


def published_image(builder, tag, labels):
    """Observe one authenticated tag and validate recipe labels, never build."""
    from ..data.registry_transport import Registry, MANIFEST, OCI_MANIFEST, digest
    registry = Registry.from_config(builder.config, timeout=15)
    status, _, raw = registry.request(
        "GET", registry.prefix + "/manifests/" + tag,
        headers={"Accept": MANIFEST + ", " + OCI_MANIFEST}, expected=(200, 404))
    if status == 404:
        return None
    manifest = json.loads(raw)
    if manifest.get("mediaType") not in (MANIFEST, OCI_MANIFEST):
        raise ValueError("native publication must contain one image manifest")
    config = registry.json_blob(manifest["config"]["digest"])
    actual = config.get("config", {}).get("Labels", {})
    if (config.get("os") != "linux" or config.get("architecture") != "amd64"
            or any(actual.get(key) != value for key, value in labels.items())):
        raise ValueError("native publication recipe, source, worker or platform differs")
    return builder.config["repository"] + "@" + digest(raw)


def recipe_labels(value):
    return {"org.ml-expd.source": value["source_id"],
            "org.ml-expd.recipe": RECIPE,
            "org.ml-expd.runtime": value["runtime_id"],
            "org.ml-expd.worker": value["worker_sha256"]}


def _active(lock):
    """Observe an existing build lock without creating or changing a file."""
    try:
        descriptor = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return False
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def _same_definition(saved, frozen):
    """The same Runtime ID with another worker is a build retried after an upgrade.

    An ID that names its worker cannot arrive with another one. An ID that does not is
    built with the current worker, and a retry is allowed only once the first attempt is
    proven to have published nothing. A receipt still has to match exactly.
    """
    return {**saved, "worker_sha256": None} == {**frozen, "worker_sha256": None}


def request(builder, value):
    allowed = {"operation", "runtime_id", "project", "source_id", "base_image", "dockerfile", "worker_sha256", "mirror"}
    if (set(value) != allowed or value["operation"] not in {
            "native-build", "native-get", "native-reconcile", "native-logs", "native-progress"}
            or not re.fullmatch(r"runtime\.[0-9a-f]{64}", value["runtime_id"])
            or not PROJECT.fullmatch(value["project"])
            or not re.fullmatch(r"source\.[0-9a-f]{64}", value["source_id"])
            or not IMAGE.fullmatch(value["base_image"])
            or not (value["mirror"] is False or value["mirror"] == "slurm")):
        raise ValueError("invalid native build request")
    identifier = value["runtime_id"].removeprefix("runtime.")
    frozen = {k: v for k, v in value.items() if k != "operation"}
    definition = builder.root / (identifier + ".native-definition.json")
    receipt = builder.root / (identifier + ".native.json")
    progress = builder.root / (identifier + ".progress.json")
    log = builder.root / (identifier + ".log")
    if definition.exists() and not _same_definition(json.loads(definition.read_text()), frozen):
        raise ValueError("native build frozen identity differs")
    if value["operation"] in {"native-logs", "native-progress"}:
        text = ""
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 8192))
                text = stream.read(8192).decode("utf-8", errors="replace")
        result = {"runtime_id": value["runtime_id"], "lines": text.splitlines(), "truncated": log.exists() and log.stat().st_size > 8192}
        if value["operation"] == "native-progress":
            active = _active(builder.root / (identifier + ".lock"))
            status = "READY" if receipt.exists() else "EXECUTING" if active else "RECONCILE_REQUIRED"
            result["progress"] = progress_view(progress, status, active=active and not receipt.exists())
        return result
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if result.get("request") != frozen:
            raise ValueError("native build receipt differs")
        return result
    from .build_recovery import publication_builder
    builder = publication_builder(builder, identifier)
    # A publisher can finish after a lost response. Observe the original tag
    # and recipe, without starting another build.
    if value["operation"] in {"native-get", "native-reconcile"}:
        if definition.exists():
            image = published_image(builder, "bundle-" + identifier, recipe_labels(value))
            if image:
                return _finish(builder, value, image, frozen, receipt)
        from .build_recovery import publication_absence
        proof = (publication_absence(builder, identifier)
                 if value["operation"] == "native-reconcile" else None)
        raise BuildReceiptMissingError("native image has not been published", retry_proof=proof)
    if not builder.config.get("allow_dockerfile_builds"):
        raise ValueError("native image builds are disabled")
    # Found out before the paid build rather than after it.
    if value["mirror"] and "target_repository" not in builder.config.get(mirror_policy(value), {}):
        raise ValueError(mirror_policy(value) + " is not configured")
    binary = Path(builder.config["native_worker_bin"])
    source = Path(builder.config["native_source_root"]) / value["project"] / value["source_id"]
    tree = source / "tree"
    if (binary.is_symlink() or not binary.is_file()
            or "sha256:" + hashlib.sha256(binary.read_bytes()).hexdigest() != value["worker_sha256"]
            or source.is_symlink() or tree.is_symlink()):
        raise ValueError("native worker or source binding differs")
    expected = "sha256:" + value["source_id"].removeprefix("source.")
    metadata = json.loads((source / "source.json").read_text())
    if (metadata.get("project") != value["project"] or metadata.get("tree_digest") != expected
            or _tree_digest(tree, require_read_only=True) != expected):
        raise ValueError("native source tree changed")
    inspection = inspect_dockerfile(tree, value["dockerfile"])
    if inspection["base_images"][-1] != value["base_image"]:
        raise ValueError("native Dockerfile base differs from the frozen request")
    with (builder.root / (identifier + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists():
            result = json.loads(receipt.read_text())
            if result.get("request") != frozen:
                raise ValueError("native build receipt differs")
            return result
        atomic_json(definition, frozen)
        record_progress(progress, "PREPARING_CONTEXT", "Preparing the sealed native worker and source")
        with tempfile.TemporaryDirectory(prefix="native-build-", dir=builder.root) as directory:
            context = Path(directory)
            shutil.copytree(tree, context, dirs_exist_ok=True, symlinks=False)
            # copytree preserves the sealed source modes. Only our temporary
            # control files need to be writable; the source stays unchanged.
            context.chmod(0o700)
            for filename in ("Dockerfile", ".dockerignore", "Dockerfile.dockerignore"):
                control = context / filename
                if control.exists():
                    control.chmod(0o600)
            internal = context / INTERNAL
            internal.mkdir(mode=0o700)
            shutil.copytree(tree, internal / "source", symlinks=False)
            shutil.copyfile(binary, internal / "worker-rs")
            if "sha256:" + hashlib.sha256((internal / "worker-rs").read_bytes()).hexdigest() != value["worker_sha256"]:
                raise ValueError("native worker changed while packaging")
            if _tree_digest(internal / "source") != expected:
                raise ValueError("native source changed while packaging")
            labels = " ".join(key + "=" + json.dumps(data) for key, data in recipe_labels(value).items())
            text = inspection["text"] + f"\nUSER root\nCOPY {INTERNAL}/worker-rs /usr/local/bin/ml-exp-worker-rs\nRUN chmod 0555 /usr/local/bin/ml-exp-worker-rs\nCOPY {INTERNAL}/source /ml-expd/source\nWORKDIR /ml-expd/source\nLABEL {labels}\nENTRYPOINT []\nCMD []\n"
            (context / "Dockerfile").write_text(text)
            for filename in (".dockerignore", "Dockerfile.dockerignore"):
                with (context / filename).open("a") as stream:
                    stream.write(f"\n!{INTERNAL}/\n!{INTERNAL}/**\n")
            tokens = [(var, var.set(data)) for var, data in (
                (BUILD_LOG, log), (BUILD_PROGRESS, progress),
                (BUILD_CACHE, builder.config["repository"] + ":native-cache-" + hashlib.sha256(value["project"].encode()).hexdigest()[:24]))]
            try:
                tag = builder.config["repository"] + ":bundle-" + identifier
                if builder.config.get("publisher") == "pai-dsw":
                    from .pai_dsw import DswPublisher
                    image = DswPublisher(builder).publish(tag, context)
                elif builder.config.get("publisher") == "buildkit":
                    image = builder._publish_buildkit(tag, context)
                else:
                    raise ValueError("native image publisher is not configured")
            finally:
                for var, token in reversed(tokens):
                    var.reset(token)
        return _finish(builder, value, image, frozen, receipt)


def mirror_policy(value):
    """Only Slurm gateways need a copy; SenseCore's `sensecore_mirror` was retired."""
    return "slurm_mirror"


def _finish(builder, value, image, frozen, receipt):
    if not IMAGE.fullmatch(image):
        raise ValueError("native published image is not digest-pinned")
    if value["mirror"]:
        from .image_mirror import mirror
        policy = mirror_policy(value)
        target = builder.config[policy]["target_repository"] + ":bundle-" + value["runtime_id"].removeprefix("runtime.")
        image = mirror(builder, {"source_image": image, "target_tag": target}, policy)["image"]
    result = {"runtime_id": value["runtime_id"], "project": value["project"], "source_id": value["source_id"], "recipe": RECIPE, "request": frozen,
              "runtime": {"runtime_id": value["runtime_id"], "source_id": value["source_id"], "image": image, "source_dir": "/ml-expd/source", "worker_path": "/usr/local/bin/ml-exp-worker-rs", "worker_sha256": value["worker_sha256"]}}
    atomic_json(receipt, result)
    return result
