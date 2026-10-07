"""Append immediately visible metric records, using only Python's stdlib.

This file can also be copied unchanged into a training project. It needs no API
token, W&B key, network connection, or installed client package.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import stat

CONTEXT_FIELDS = frozenset({
    "step", "epoch", "timestamp", "dataset_id", "variant_id", "checkpoint_id",
    "protocol_id", "numerator", "denominator", "status", "error",
})


def _json_numbers(value):
    if isinstance(value, float) and not math.isfinite(value):
        # The server retains the original spelling and marks nonfinite evidence
        # FAILED. Never replace an invalid measurement with zero.
        return str(value)
    if isinstance(value, dict):
        return {key: _json_numbers(item) for key, item in value.items()}
    return value


class MetricWriter:
    """Write one newline-terminated record per call to OUTPUT_DIR/metrics.jsonl.

    Calls append, flush, and close before returning. Existing records survive a
    restart. In distributed training, the experiment's reporting rank should
    write the aggregated measurement. Scientific aggregation belongs to code.
    """

    def __init__(self, output_dir: str | Path | None = None):
        directory = output_dir if output_dir is not None else os.environ.get("OUTPUT_DIR")
        if not directory:
            raise ValueError("provide output_dir or set OUTPUT_DIR")
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "metrics.jsonl"

    def log(self, name: str, value, *, unit: str, **context) -> None:
        """Append a scalar with explicit units and optional scientific context.

        Pass ordinary Python scalars (e.g. loss.item()), not tensors. Unit/schema
        agreement and protocol identity are checked by the server.
        """
        for field in (name, unit):
            if (not isinstance(field, str) or not field.strip() or len(field) > 128
                    or any(ord(char) < 32 for char in field)):
                raise ValueError("metric name and unit must be nonempty printable strings")
        if context.keys() - CONTEXT_FIELDS:
            raise ValueError("unsupported metric context field")
        if value is not None and type(value) not in {int, float}:
            raise TypeError("metric value must be a Python scalar or None")
        record = _json_numbers({"name": name, "value": value, "unit": unit,
                                **{key: item for key, item in context.items() if item is not None}})
        line = (json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        if len(line) > 65536:
            raise ValueError("metric record exceeds the worker's 64 KiB line limit")
        if self.path.is_symlink():
            raise ValueError("metric file must be a regular file")
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self.path, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("metric file must be a regular file")
            # Closing flushes each complete line while training is still
            # running, handling short writes without per-step fsync on NAS.
            with os.fdopen(fd, "wb", closefd=False) as stream:
                stream.write(line)
        finally:
            os.close(fd)
