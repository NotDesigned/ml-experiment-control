"""Shared, fixed recipe for ACP-compatible data images."""
from __future__ import annotations

RECIPE = "acp-data-image.v2"


def recipe(base):
    return (f"FROM {base}\n"
            "RUN python -c \"import platform, shutil; assert platform.libc_ver()[0] == 'glibc' and shutil.which('bash'), 'ACP_DATA_BASE_REQUIRES_GLIBC_AND_BASH'\"\n"
            "COPY dataset.tar /payload/dataset.tar\n"
            "COPY asset.json /payload/asset.json\nCOPY workers/ /usr/local/lib/ml-expd/\n"
            "ENTRYPOINT []\nCMD [\"/bin/true\"]\n")
