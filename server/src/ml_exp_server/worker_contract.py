"""One managed launcher and receipt contract for every source build recipe."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

WORKER_CONTRACT = "managed-worker.v1"
CAPABILITIES = ["data-assets.v1", "checkpoint-upload.v1", "data-preparation.v1"]
WORKERS = (("managed_worker.py", "worker.py"), ("container_worker.py", "legacy_worker.py"),
           ("data_preparation.py", "data_preparation.py"))


def worker_digest() -> str:
    root = Path(__file__).parent
    return hashlib.sha256(b"".join((root / source).read_bytes() for source, _ in WORKERS)).hexdigest()


def recipe_digest() -> str:
    root = Path(__file__).parent
    return hashlib.sha256(b"".join((root / name).read_bytes() for name in
                                  ("worker_contract.py", "environment_build.py", "dockerfile_build.py"))).hexdigest()


def install_workers(directory: Path) -> None:
    for source, target in WORKERS:
        path = directory / target
        shutil.copyfile(Path(__file__).with_name(source), path)
        path.chmod(0o444)


def worker_dockerfile(source_id: str, prefix: str = "") -> str:
    return (f"COPY {prefix}worker.py /usr/local/lib/ml-expd/worker.py\n"
            f"COPY {prefix}legacy_worker.py /usr/local/lib/ml-expd/legacy_worker.py\n"
            f"COPY {prefix}data_preparation.py /usr/local/lib/ml-expd/data_preparation.py\n"
            "ENV ML_EXPD_MULTIPART_UPLOAD=1\n"
            f"LABEL org.ml-expd.source={source_id}\n"
            "ENTRYPOINT []\nCMD [\"/bin/true\"]\n")


def managed_io(container: dict) -> bool:
    # Frozen Dockerfile runs predate the common contract. Old FROM/COPY images
    # must keep their original command and mounts when inspected or retried.
    return bool(container.get("dockerfile") or container.get("worker_contract") == WORKER_CONTRACT)
