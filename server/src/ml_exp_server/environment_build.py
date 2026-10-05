"""Reviewed dependency recipe; project Dockerfiles are never executed."""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import re

DEPENDENCY_RECIPE = "source-python-dependencies-v3"
PIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(?:\[[A-Za-z0-9_,.-]+\])?==[A-Za-z0-9][A-Za-z0-9.!+_-]*(?:\s+--hash=sha256:[0-9a-f]{64})*")


def requirements_path(value: str) -> str:
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or "\\" in value or "\x00" in value
            or any(p in {".", ".."} or p.startswith(".") for p in value.split("/"))):
        raise ValueError("requirements must be a relative source file path")
    return value


def inspect_requirements(tree: Path, name: str) -> dict:
    name = requirements_path(name)
    path = tree / name
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(tree.resolve()):
        raise ValueError("requirements file is missing from the frozen source")
    content = path.read_bytes()
    if len(content) > 256 * 1024:
        raise ValueError("requirements file exceeds 256 KiB")
    text = content.decode("utf-8").replace("\\\r\n", "").replace("\\\n", "")
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines or len(lines) > 1000 or any(not PIN.fullmatch(line) for line in lines):
        raise ValueError("requirements must use exact package==version pins; URLs, includes and pip options are unsupported")
    hashed = ["--hash=" in line for line in lines]
    if any(hashed) and not all(hashed):
        raise ValueError("hash-locked requirements must hash every declared package")
    return {"path": name, "sha256": hashlib.sha256(content).hexdigest(),
            "packages": len(lines), "hash_locked": all(hashed)}


def dockerfile(image: str, source_id: str, requirements: str | None = None) -> str:
    installation = ""
    if requirements is not None:
        installation = ("COPY requirements.txt /tmp/ml-expd-requirements.txt\n"
                        "COPY dependency_install.py /tmp/ml-expd-install.py\n"
                        'RUN ["python3", "-m", "venv", "--without-pip", "--system-site-packages", "/opt/ml-expd-venv"]\n'
                        'ENV PATH="/opt/ml-expd-venv/bin:${PATH}"\n'
                        'RUN ["python3", "/tmp/ml-expd-install.py"]\n')
    return (f"FROM {image}\n" + installation + "COPY source/ /workspace/\nWORKDIR /workspace\n"
            "COPY worker.py /usr/local/lib/ml-expd/worker.py\nENV ML_EXPD_MULTIPART_UPLOAD=1\n"
            f"LABEL org.ml-expd.source={source_id}\nENTRYPOINT []\nCMD [\"/bin/true\"]\n")


def installer_digest() -> str:
    return hashlib.sha256(Path(__file__).with_name("dependency_install.py").read_bytes()).hexdigest()
