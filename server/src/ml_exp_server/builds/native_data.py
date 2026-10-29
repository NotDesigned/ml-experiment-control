"""Data-image effects only: dataset bytes are assembled on the CPU DSW.

The native service owns recipe/copy identity and admission. Download capability
files stay outside Docker contexts and published receipts. No dataset, archive
or image layer is materialized on the API/builder host.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import time
import textwrap

from .builder_client import IMAGE, PROJECT, BuildReceiptMissingError
from .image_build_context import BUILD_LOG, BUILD_PROGRESS
from ..storage import atomic_json



def stage_script(remote, worker_sha256, archive_bytes):
    """A small isolated program; transfer capabilities never enter the tar."""
    return textwrap.dedent(f"""\
        from pathlib import Path
        import hashlib, json, shutil, subprocess, tarfile
        root = Path({remote!r})
        capability = root / "download.json"
        status = {{"status": "FAILED"}}
        try:
            request = json.loads((root / "request.json").read_text())
            original = request["context_sha256"]
            with (root / "context.tar").open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != original:
                    raise ValueError("native data context changed")
            if shutil.disk_usage(root).free <= {2 * archive_bytes + 1024**3}:
                raise ValueError("native data staging storage insufficient")
            seed = root / "seed"
            seed.mkdir(mode=0o700)
            with tarfile.open(root / "context.tar") as archive:
                archive.extractall(seed, filter="data")
            worker = seed / "worker-rs"
            worker.chmod(0o500)
            if "sha256:" + hashlib.sha256(worker.read_bytes()).hexdigest() != {worker_sha256!r}:
                raise ValueError("native data worker changed")
            with (root / "data-stage.log").open("wb") as log:
                result = subprocess.run([str(worker), "data-import", "--config", str(capability),
                                         "--bundle-output", str(root / "archive.tar")],
                                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                raise ValueError("native data verification failed")
            with tarfile.open(root / "context.tar", "a") as archive:
                archive.add(root / "archive.tar", arcname="archive.tar", recursive=False)
            with (root / "context.tar").open("rb") as stream:
                materialized = hashlib.file_digest(stream, "sha256").hexdigest()
            request["context_sha256"] = materialized
            temporary = root / "request.partial.json"
            temporary.write_text(json.dumps(request))
            temporary.replace(root / "request.json")
            status = {{"status": "SUCCEEDED", "control": original, "materialized": materialized}}
        finally:
            capability.unlink(missing_ok=True)
            temporary = root / "data-stage-status.partial.json"
            temporary.write_text(json.dumps(status))
            temporary.replace(root / "data-stage-status.json")
        """)


def request(builder, value):
    if set(value) != {"operation", "data_image_id", "recipe", "download"}:
        raise ValueError("invalid native data request")
    identifier = value["data_image_id"]
    recipe, download = value["recipe"], value["download"]
    if (value["operation"] not in {"native-data-build", "native-data-get"}
            or not re.fullmatch(r"data-image\.[0-9a-f]{64}", identifier)
            or set(recipe) != {"project", "executor", "definition", "base_image", "worker_sha256"}
            or not PROJECT.fullmatch(recipe["project"])
            or not PROJECT.fullmatch(recipe["executor"])
            or not IMAGE.fullmatch(recipe["base_image"])
            or download.get("project") != recipe["project"]
            or download.get("definition") != recipe["definition"]
            or download.get("target", {}).get("backend_id") != recipe["executor"]
            or download.get("source", {}).get("kind") != "dataset_parts"):
        raise ValueError("native data recipe or scope differs")
    binary = Path(builder.config["native_worker_bin"])
    if (binary.is_symlink() or not binary.is_file()
            or "sha256:" + hashlib.sha256(binary.read_bytes()).hexdigest() != recipe["worker_sha256"]):
        raise ValueError("native data worker differs")
    prefixes = builder.config.get("base_image_prefixes", [])
    if prefixes and not any(recipe["base_image"].startswith(p) for p in prefixes):
        raise ValueError("native data base is outside policy")
    key = identifier.removeprefix("data-image.")
    definition = builder.root / (key + ".native-data-definition.json")
    receipt = builder.root / (key + ".native-data.json")
    tag = builder.config["repository"] + ":bundle-" + key
    with (builder.root / (key + ".native-data.lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BuildReceiptMissingError("native data build is still in progress") from None
        if definition.exists() and json.loads(definition.read_text()) != recipe:
            raise ValueError("native data frozen recipe differs")
        if receipt.exists():
            result = json.loads(receipt.read_text())
            if result.get("recipe") != recipe or result.get("data_image_id") != identifier:
                raise ValueError("native data receipt differs")
            return result
        if value["operation"] == "native-data-get":
            if not definition.exists():
                raise BuildReceiptMissingError("native data image has not been published")
            from .native_image import published_image
            image = published_image(builder, tag.rsplit(":", 1)[1],
                                    {"org.ml-expd.data-image": identifier,
                                     "org.ml-expd.worker": recipe["worker_sha256"]})
            if image is None:
                raise BuildReceiptMissingError("native data image has not been published")
        else:
            if definition.exists():
                raise BuildReceiptMissingError("native data build already has a dispatch intent; reconcile it")
            if builder.config.get("publisher") != "pai-dsw":
                raise ValueError("native data image requires the isolated CPU DSW publisher")
            atomic_json(definition, recipe)
            from .pai_dsw import DswPublisher

            class DataPublisher(DswPublisher):
                def shutdown_timer(self):
                    # Both data materialization and image build consume this
                    # lease. The cloud timer still cannot exceed four hours.
                    original = self.timeout
                    self.timeout = min(original + download["timeout_seconds"], 13200)
                    try:
                        super().shutdown_timer()
                    finally:
                        self.timeout = original

                def build(self, remote, private):
                    # The inherited publisher transferred only control files and
                    # the small native importer, never the dataset itself.
                    self.secrets.append(download["source"]["channel"]["token"])
                    with tempfile.TemporaryDirectory(prefix="native-data-control-", dir=builder.root) as directory:
                        source = Path(directory) / "download.json"
                        atomic_json(source, download)
                        self.upload(source, remote + "/download.json", private)
                    self.phase("STAGING_DATA_ON_DSW", "Downloading and verifying dataset on the CPU builder")
                    # Start once, then observe a durable result with short RPCs.
                    wrapper = stage_script(remote, recipe["worker_sha256"], recipe["definition"]["archive_bytes"])
                    with tempfile.TemporaryDirectory(prefix="native-data-stage-", dir=builder.root) as directory:
                        script = Path(directory) / "stage.py"
                        script.write_text(wrapper)
                        self.upload(script, remote + "/data-stage.py", private)
                    start = ("from pathlib import Path; import subprocess; "
                             f"p=Path({remote!r}); (p/'data-stage-started').open('x').close(); "
                             "subprocess.Popen(['python3',str(p/'data-stage.py')],stdin=subprocess.DEVNULL,"
                             "stdout=(p/'data-stage-control.log').open('wb'),stderr=subprocess.STDOUT,start_new_session=True)")
                    self.cli(["dsw", "instance", "exec", self.instance, "--code", start], private, timeout=45)
                    deadline = time.monotonic() + download["timeout_seconds"] + 120
                    while True:
                        probe = ("from pathlib import Path; import json; "
                                 f"p=Path({remote!r}); s=p/'data-stage-status.json'; "
                                 "v=json.loads(s.read_text()) if s.exists() else {'status':'RUNNING'}; "
                                 "a=p/'archive.tar'; v['bytes']=a.stat().st_size if a.exists() else 0; "
                                 "print('ML_EXPD_NATIVE_DATA_CONTEXT='+json.dumps(v))")
                        output = self.cli(["dsw", "instance", "exec", self.instance, "--code", probe], private, timeout=45)
                        rows = [line.split("=", 1)[1] for line in output.splitlines()
                                if line.startswith("ML_EXPD_NATIVE_DATA_CONTEXT=")]
                        if len(rows) != 1:
                            raise ValueError("native data context verification is unavailable")
                        proof = json.loads(rows[0])
                        if proof["status"] == "SUCCEEDED":
                            break
                        if proof["status"] != "RUNNING" or time.monotonic() >= deadline:
                            raise ValueError("native data verification failed or exceeded its timeout")
                        self.phase("STAGING_DATA_ON_DSW", "Dataset verification running; archive bytes " + str(proof["bytes"]))
                        time.sleep(3)
                    result = super().build(remote, private)
                    if result["context_sha256"] != proof["materialized"]:
                        raise ValueError("materialized data context differs")
                    # Distinguish the source control context from the remotely
                    # materialized archive. No scientific/data hash is replaced.
                    result["materialized_context_sha256"] = proof["materialized"]
                    result["context_sha256"] = proof["control"]
                    return result

            tokens = [(var, var.set(data)) for var, data in (
                (BUILD_LOG, builder.root / (key + ".log")),
                (BUILD_PROGRESS, builder.root / (key + ".progress.json")))]
            try:
                with tempfile.TemporaryDirectory(prefix="native-data-recipe-", dir=builder.root) as directory:
                    context = Path(directory)
                    shutil.copyfile(binary, context / "worker-rs")
                    (context / "Dockerfile").write_text(
                        f"FROM {recipe['base_image']}\nUSER root\n"
                        "COPY worker-rs /usr/local/bin/ml-exp-worker-rs\n"
                        "RUN chmod 0555 /usr/local/bin/ml-exp-worker-rs\n"
                        "COPY archive.tar /ml-expd/data/archive.tar\n"
                        f"LABEL org.ml-expd.data-image={json.dumps(identifier)} "
                        f"org.ml-expd.worker={json.dumps(recipe['worker_sha256'])}\n"
                        "ENTRYPOINT []\nCMD []\n")
                    image = DataPublisher(builder).publish(tag, context)
            finally:
                for var, token in reversed(tokens):
                    var.reset(token)
        from .image_mirror import mirror
        target = builder.config["sensecore_mirror"]["target_repository"] + ":data-" + key
        image = mirror(builder, {"source_image": image, "target_tag": target})["image"]
        result = {"data_image_id": identifier, "recipe": recipe, "image": image}
        atomic_json(receipt, result)
        return result
