"""Shared, fixed recipe for ACP-compatible data images."""
from __future__ import annotations

import hashlib
from pathlib import Path

RECIPE = "acp-data-image.v3"
DATA_WORKERS = ("data_copy_worker.py", "data_input.py")


def data_worker_digest(directory: Path | None = None) -> str:
    root = directory if directory is not None else Path(__file__).parent
    return hashlib.sha256(b"".join((root / name).read_bytes() for name in DATA_WORKERS)).hexdigest()


def recipe(base):
    return (f"FROM {base}\n"
            "RUN python -c \"import platform, shutil; assert platform.libc_ver()[0] == 'glibc' and shutil.which('bash'), 'ACP_DATA_BASE_REQUIRES_GLIBC_AND_BASH'\"\n"
            "COPY dataset.tar /payload/dataset.tar\n"
            "COPY asset.json /payload/asset.json\nCOPY workers/ /usr/local/lib/ml-expd/\n"
            "ENTRYPOINT []\nCMD [\"/bin/true\"]\n")
