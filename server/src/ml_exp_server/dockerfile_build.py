"""Client Dockerfile provenance."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import re
import shlex

from dockerfile_parse import DockerfileParser

from .source_paths import relative_source_path

PINNED_IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}$")
INTERNAL = "ml-expd-build-internal"


def _has_heredoc(value: str) -> bool:
    # Quoted/escaped strings and shell comments can contain literal <<. Only
    # an unquoted operator changes how the following Dockerfile lines parse.
    tokens = re.finditer(r"""'[^']*'|"(?:\\.|[^"\\])*"|\\.|(?<!\S)\#[^\n]*|(<<-?)""", value, re.DOTALL)
    return any(token[1] for token in tokens)


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
    if re.search(r"(?im)^\s*#\s*(syntax|escape)\s*=", text):
        raise ValueError("custom frontend/escape directives are not supported by the Dockerfile parser")
    parser = DockerfileParser(fileobj=io.BytesIO(raw))
    stages, images = set(), []
    for instruction in parser.structure:
        value, operation = instruction["value"], instruction["instruction"]
        if operation in {"RUN", "COPY", "ADD"} and _has_heredoc(value):
            raise ValueError("heredocs are not supported by the Dockerfile parser")
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
        elif operation == "COPY":
            match = re.search(r"(?:^|\s)--from=([^\s]+)", value)
            if match and match[1].lower() not in stages and not match[1].isdigit():
                raise ValueError("COPY --from must reference an earlier build stage")
    if not images:
        raise ValueError("Dockerfile must include a digest-pinned external base")
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "base_images": images,
            "text": text}
