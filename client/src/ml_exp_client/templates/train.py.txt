"""Dependency-free output/metrics smoke; use your own training code for research."""
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=3)
args = parser.parse_args()
output = Path(os.environ["OUTPUT_DIR"])
output.mkdir(parents=True, exist_ok=True)
with (output / "metrics.jsonl").open("w") as metrics:
    for step in range(1, args.steps + 1):
        metrics.write(json.dumps({"step": step, "loss": 1 / step}) + "\n")
summary = {key.lower(): os.environ[key] for key in
           ("PROJECT_NAME", "RUN_ID", "ATTEMPT_ID", "SOURCE_ID")}
summary["steps"] = args.steps
(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary), flush=True)
