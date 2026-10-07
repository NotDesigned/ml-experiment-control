"""Client Dockerfile provenance and the managed data/checkpoint launcher."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import re
import shlex

from dockerfile_parse import DockerfileParser

from .source_paths import relative_source_path
from .worker_contract import worker_digest, worker_dockerfile

DOCKERFILE_RECIPE = "source-dockerfile-assets-v1"
PINNED_IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}$")
INTERNAL = "ml-expd-build-internal"


def inspect_dockerfile(tree: Path, name: str) -> dict:
    name = relative_source_path(name)
    path = tree / name
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(tree.resolve()):
        raise ValueError("Dockerfile is missing from the frozen source")
    raw = path.read_bytes()
    if len(raw) > 256 * 1024 or (tree / INTERNAL).exists():
        raise ValueError("Dockerfile exceeds 256 KiB or uses the reserved build directory")
    text = raw.decode("utf-8")
    # Pin the frontend as well as external stages. Heredoc/syntax extensions can
    # be added with a parser supporting their complete grammar in a later recipe.
    if re.search(r"(?im)^\s*#\s*(syntax|escape)\s*=", text) or "<<" in text:
        raise ValueError("custom frontend/escape directives and heredocs are not supported by this recipe")
    parser = DockerfileParser(fileobj=io.BytesIO(raw))
    stages, images = set(), []
    for instruction in parser.structure:
        value, operation = instruction["value"], instruction["instruction"]
        if operation == "FROM":
            parts = shlex.split(value)
            if parts and parts[0].startswith("--platform="):
                if parts.pop(0) != "--platform=linux/amd64":
                    raise ValueError("only linux/amd64 runtime images are supported")
            if len(parts) not in {1, 3} or len(parts) == 3 and parts[1].upper() != "AS":
                raise ValueError("invalid FROM instruction")
            image = parts[0]
            if image.lower() not in stages and image != "scratch":
                if not PINNED_IMAGE.fullmatch(image):
                    raise ValueError("every external FROM image must be pinned by sha256; build-argument bases are unsupported")
                images.append(image)
            if len(parts) == 3:
                stages.add(parts[2].lower())
        elif operation in {"ONBUILD", "ADD"}:
            raise ValueError("use COPY and explicit RUN commands instead of ADD/ONBUILD")
        elif operation == "COPY":
            match = re.search(r"(?:^|\s)--from=([^\s]+)", value)
            if match and match[1].lower() not in stages and not match[1].isdigit():
                raise ValueError("COPY --from must reference an earlier build stage")
    if not images:
        raise ValueError("Dockerfile must include a pinned external base and provide Python 3")
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "base_images": images,
            "text": text}


def managed_dockerfile(inspection: dict, source_id: str) -> str:
    return (inspection["text"].rstrip() + "\n\n# ML-Expd managed execution contract\n"
            'RUN ["python3", "-c", "import sys; assert sys.version_info >= (3, 10)"]\n'
            f"COPY {INTERNAL}/source/ /workspace/\n"
            "WORKDIR /workspace\n"
            + worker_dockerfile(source_id, INTERNAL + "/"))
