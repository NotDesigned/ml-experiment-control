"""Discover only atomically completed checkpoint payloads."""

from __future__ import annotations

import json
import re
from pathlib import Path


CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)$")


def checkpoint_step(path: str | Path) -> int | None:
    match = CHECKPOINT_RE.fullmatch(Path(path).name)
    return int(match.group(1)) if match else None


def discover_latest_completed_checkpoint(run_dir: Path) -> dict[str, object] | None:
    """Return the newest payload whose JSON completion marker matches its size."""
    completed: list[tuple[int, Path, dict[str, object]]] = []
    for marker in run_dir.glob("checkpoint_*.complete"):
        payload = marker.with_suffix("")
        step = checkpoint_step(payload)
        if step is None or not payload.is_file():
            continue
        try:
            metadata = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict):
            continue
        if metadata.get("step") != step or metadata.get("bytes") != payload.stat().st_size:
            continue
        completed.append((step, payload, metadata))
    if not completed:
        return None
    step, payload, metadata = max(completed, key=lambda item: item[0])
    return {
        "path": str(payload), "step": step, "bytes": metadata["bytes"],
        "completed_at": metadata.get("completed_at"),
    }


def select_latest_checkpoint_name(names: list[str]) -> tuple[str, int] | None:
    """Select the largest valid checkpoint basename returned by a remote probe."""
    candidates = [(step, name) for name in names if (step := checkpoint_step(name)) is not None]
    if not candidates:
        return None
    step, name = max(candidates)
    return name, step
