"""Verify the client wheel alone from outside the repository (stdlib only)."""
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    assert not importlib.metadata.requires("ml-experiment-client")
    for module in ("ml_exp_server", "experiment_control", "fastapi", "boto3", "yaml", "pytest"):
        assert importlib.util.find_spec(module) is None, f"unexpected dependency: {module}"
    executable = Path(sys.executable).parent / "ml-exp"
    env = {key: value for key, value in os.environ.items()
           if key != "PYTHONPATH" and not key.startswith("ML_EXPD_")}
    with tempfile.TemporaryDirectory(prefix="ml-exp-client-wheel-") as directory:
        root = Path(directory)
        for argv in ([str(executable), "--help"], [str(executable), "--version"],
                     [sys.executable, "-m", "ml_exp_client", "--help"]):
            result = subprocess.run(argv, cwd=root, env=env, capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
        result = subprocess.run([str(executable), "init", "study"], cwd=root, env=env,
                                capture_output=True, text=True, check=True)
        assert json.loads(result.stdout)["entrypoint"] == ["python", "train.py"]
        env.update(OUTPUT_DIR=str(root / "outputs"), PROJECT_NAME="wheel-smoke", RUN_ID="trial",
                   ATTEMPT_ID="attempt-001", SOURCE_ID="source." + "a" * 64)
        subprocess.run([sys.executable, "study/train.py"], cwd=root, env=env,
                       capture_output=True, text=True, check=True)
        assert json.loads((root / "outputs/summary.json").read_text())["run_id"] == "trial"
        assert len((root / "outputs/metrics.jsonl").read_text().splitlines()) == 3
    print("Standalone client wheel: CLI/module/template verified outside checkout; no runtime dependencies")


if __name__ == "__main__":
    main()
