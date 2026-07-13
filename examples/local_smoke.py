"""Run one real command through LocalBackend without a host repository."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

from experiment_control.backends import build_registry
from experiment_control.backends.services import BackendServices
from experiment_control.contracts import (
    AttemptManifest,
    BackendRecord,
    Campaign,
    RunSpec,
    RunSummary,
)
from experiment_control.manifest import atomic_write, utc_now
from experiment_control.runner import SubprocessRunner


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="experiment-control-local-") as temporary:
        root = Path(temporary)
        run_dir = root / "run"
        record: BackendRecord = {"attempt_id": "attempt-001", "backend_job_id": None}

        def summarize(_campaign: Campaign, path: Path) -> RunSummary:
            return {"run_dir": str(path)}

        def backend_record(_campaign: Campaign, _run: RunSpec) -> BackendRecord:
            return record.copy()

        services = BackendServices(
            run_command=SubprocessRunner().run,
            local_run_dir=lambda _campaign, _run: run_dir,
            backend_record=backend_record,
            summarize_run=summarize,
            parse_metric=lambda _campaign, _line: None,
            parse_checkpoint=lambda _campaign, _line: None,
            atomic_write=atomic_write,
            utc_now=utc_now,
        )
        backend = build_registry(services).get("local")
        run: RunSpec = {
            "run_id": "local-smoke",
            "storage": {"run_dir": str(run_dir)},
            "backend": {"kind": "local", "workdir": str(Path.cwd())},
        }
        backend.preflight(run, scope="submit").require_ready()
        manifest: AttemptManifest = {
            "attempt_id": "attempt-001",
            "source_id": "source-local",
            "command": [sys.executable, "-c", "print('local backend is running')"],
        }
        job_id = backend.submit({}, run, manifest, dry_run=False)
        record["backend_job_id"] = job_id
        while True:
            status = backend.status({}, run)
            if status["state"] in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                break
            time.sleep(0.02)
        print(json.dumps({"status": status, "logs": backend.logs({}, run, tail=20)}, indent=2))


if __name__ == "__main__":
    main()
