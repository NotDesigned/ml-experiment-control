"""Narrow dependency bundle injected into platform adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

from ..contracts import (
    BackendRecord,
    Campaign,
    CheckpointRecord,
    JsonValue,
    MetricRecord,
    RunSpec,
    RunSummary,
)
from ..runner import CommandResult


class RunCommand(Protocol):
    def __call__(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        check: bool = True,
        input_text: str | None = None,
        timeout_seconds: float | None = None,
    ) -> CommandResult: ...


class AtomicWriter(Protocol):
    def __call__(self, path: Path, payload: Mapping[str, JsonValue]) -> None: ...


@dataclass(frozen=True)
class BackendServices:
    run_command: RunCommand
    local_run_dir: Callable[[Campaign, RunSpec], Path]
    backend_record: Callable[[Campaign, RunSpec], BackendRecord]
    summarize_run: Callable[[Campaign, Path], RunSummary]
    parse_metric: Callable[[Campaign, str], MetricRecord | None]
    parse_checkpoint: Callable[[Campaign, str], CheckpointRecord | None]
    atomic_write: AtomicWriter
    utc_now: Callable[[], str]
