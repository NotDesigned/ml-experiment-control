"""Narrow dependency bundle injected into platform adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..runner import CommandResult


@dataclass(frozen=True)
class BackendServices:
    run_command: Callable[..., CommandResult]
    local_run_dir: Callable[[dict[str, Any], dict[str, Any]], Path]
    backend_record: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
    summarize_run: Callable[[dict[str, Any], Path], dict[str, Any]]
    parse_metric: Callable[[dict[str, Any], str], dict[str, Any] | None]
    parse_checkpoint: Callable[[dict[str, Any], str], dict[str, Any] | None]
    atomic_write: Callable[..., None]
    utc_now: Callable[[], str]
