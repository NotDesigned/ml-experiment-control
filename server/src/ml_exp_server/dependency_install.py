"""Runs inside the build container, never on the control-plane host."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def installed() -> dict[str, str]:
    return {d.metadata["Name"].lower().replace("_", "-"): d.version
            for d in importlib.metadata.distributions() if d.metadata["Name"]}


def main():
    before = installed()
    protected = {name: version for name, version in before.items()
                 if name in {"torch", "torchvision", "torchaudio", "triton"} or name.startswith("nvidia-")}
    constraints = Path("/tmp/ml-expd-base-constraints.txt")
    constraints.write_text("".join(f"{name}=={version}\n" for name, version in sorted(protected.items())))
    environment = {k: v for k, v in os.environ.items() if not k.startswith("PIP_")}
    environment["PIP_CONFIG_FILE"] = os.devnull
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-input", "--no-cache-dir",
                    "--disable-pip-version-check", "--only-binary=:all:",
                    "--index-url", "https://pypi.org/simple", "--constraint", str(constraints),
                    "--requirement", "/tmp/ml-expd-requirements.txt"], check=True, env=environment)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True, env=environment)
    after = installed()
    if any(after.get(name) != version for name, version in protected.items()):
        raise ValueError("dependency installation changed the base GPU framework")
    path = Path("/usr/local/share/ml-expd/environment.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"python": platform.python_version(), "packages": after,
                               "protected_base_packages": protected}, sort_keys=True, indent=2) + "\n")
    path.chmod(0o444)


if __name__ == "__main__":
    main()
