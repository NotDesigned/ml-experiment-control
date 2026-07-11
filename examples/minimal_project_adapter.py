"""Copyable adapter showing the complete project boundary.

This example assumes a training repository with a JSON config and a ``train.py``
entrypoint. Keep the adapter in that repository; only scheduler-neutral value
objects and protocols come from :mod:`experiment_control`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from experiment_control.project import AssetProbe, AssetRequirement, SourceBundle


class MinimalProjectAdapter:
    """Small, complete ``ProjectAdapter`` implementation."""

    name = "minimal"
    safe_env_keys = frozenset({"DATA_ROOT", "LOG_EVERY", "OUTPUT_DIR"})

    def validate_run(self, run: dict[str, Any]) -> None:
        config = str(run.get("config", ""))
        if not config.endswith(".json"):
            raise ValueError("minimal project config must be JSON")

    def operational_overrides(
        self, env: Mapping[str, str], output_dir: str
    ) -> list[str]:
        # Only allowlisted, non-secret controller values cross this boundary.
        return [f"output_dir={output_dir}"]

    def resolve_config(self, config_path: str, overrides: list[str]) -> dict[str, Any]:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        for item in overrides:
            key, separator, value = item.partition("=")
            if not separator:
                raise ValueError(f"invalid override: {item}")
            config[key] = value
        return config

    def environment(
        self, campaign: dict[str, Any], run: dict[str, Any]
    ) -> dict[str, str]:
        return {"OUTPUT_DIR": str(run["storage"]["run_dir"])}

    def command(self, run: dict[str, Any]) -> list[str]:
        command = ["python", "train.py", "--config", str(run["config"])]
        for override in run.get("config_overrides", []):
            command.extend(["--override", str(override)])
        return command

    def plan_assets(
        self, config_path: str, overrides: list[str]
    ) -> list[AssetRequirement]:
        config = self.resolve_config(config_path, overrides)
        dataset = str(config["dataset"])
        return [AssetRequirement("dataset", dataset, "training input")]

    def asset_probes(
        self,
        requirements: list[AssetRequirement],
        environment: Mapping[str, str],
    ) -> list[AssetProbe]:
        root = Path(environment["DATA_ROOT"])
        return [AssetProbe(item, str(root / item.identity)) for item in requirements]

    def parse_metric(self, line: str) -> dict[str, Any] | None:
        match = re.search(r"step=(\d+)\s+val_loss=([-+0-9.eE]+)", line)
        if not match:
            return None
        return {"step": int(match.group(1)), "val_loss": float(match.group(2))}

    def parse_checkpoint(self, line: str) -> dict[str, Any] | None:
        match = re.search(r"checkpoint=(\S+)\s+step=(\d+)\s+bytes=(\d+)", line)
        if not match:
            return None
        return {
            "path": match.group(1),
            "step": int(match.group(2)),
            "bytes": int(match.group(3)),
        }

    def summarize(self, run_dir: Path) -> dict[str, Any]:
        records = [
            json.loads(line)
            for line in (run_dir / "eval_metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]
        return {"latest_eval": records[-1] if records else None}

    def source_bundle(self, repo_root: Path) -> SourceBundle:
        return SourceBundle(
            root=repo_root,
            excludes=(".git/", ".env", "outputs/", "checkpoints/", "*.pt"),
            container_path="/workspace",
            required_paths=("train.py",),
        )
