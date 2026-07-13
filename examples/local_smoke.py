"""Run one real command through LocalBackend without a host repository."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import cast

from experiment_control.backends import build_registry
from experiment_control.backends.services import BackendServices
from experiment_control.contracts import (
    AttemptManifest,
    BackendRecord,
    Campaign,
    LocalBackendConfig,
    RunSpec,
    RunSummary,
    StorageConfig,
    SubmissionIntent,
)
from experiment_control.manifest import ExperimentStateStore, atomic_write, utc_now
from experiment_control.run_manifest import build_run_manifest
from experiment_control.runner import SubprocessRunner


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="experiment-control-local-") as temporary:
        root = Path(temporary)
        run_dir = root / "run"
        store = ExperimentStateStore(run_dir)
        backend_config: LocalBackendConfig = {
            "kind": "local", "workdir": str(Path.cwd()),
        }
        storage: StorageConfig = {
            "run_dir": str(run_dir), "project_data_root": str(root),
        }
        command = [sys.executable, "-c", "print('local backend is running')"]
        run: RunSpec = {
            "run_id": "local-smoke",
            "storage": storage,
            "backend": backend_config,
        }
        created_at = utc_now()
        store.ensure_manifest(build_run_manifest(
            project="local-smoke",
            run_id="local-smoke",
            created_at=created_at,
            config_path="inline",
            resolved_config={},
            source_id="source-local-smoke",
            runtime_tree_id="runtime-local-smoke",
            git_commit=None,
            campaign_id=None,
            campaign="local-smoke",
            image_id="sha256:" + "0" * 64,
            run_dir=str(run_dir),
            max_infra_retries=0,
            backend=dict(backend_config),
            resources={},
            storage=dict(storage),
            command=command,
            execution={},
        ))
        store.create_attempt({
            "schema_version": 1,
            "project": "local-smoke",
            "run_id": "local-smoke",
            "attempt_id": "attempt-001",
            "created_at": created_at,
            "backend": backend_config,
            "backend_job_id": None,
            "source_id": "source-local-smoke",
            "image_id": "sha256:" + "0" * 64,
            "command": command,
            "resources": {},
            "resume_from": None,
        })
        store.initialize_attempt_records("attempt-001")

        def summarize(_campaign: Campaign, path: Path) -> RunSummary:
            return {"run_dir": str(path)}

        def backend_record(_campaign: Campaign, _run: RunSpec) -> BackendRecord:
            value = store.load_backend("attempt-001")
            if value is None:
                raise RuntimeError("local smoke backend record is unavailable")
            return {
                "attempt_id": str(value["attempt_id"]),
                "backend_job_id": (
                    str(value["backend_job_id"])
                    if value.get("backend_job_id") is not None else None
                ),
            }

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
        backend.preflight(run, scope="submit").require_ready()
        manifest: AttemptManifest = {
            "attempt_id": "attempt-001",
            "source_id": "source-local",
            "command": command,
        }
        intent = cast(SubmissionIntent, store.begin_submission(
            project="local-smoke",
            run_id="local-smoke",
            attempt_id="attempt-001",
            backend="local",
            request=backend.submission_request({}, run, "attempt-001"),
        ))
        job_id = backend.submit(
            {}, run, manifest, dry_run=False, intent=intent,
        )
        store.reconcile_submission(
            project="local-smoke",
            run_id="local-smoke",
            attempt_id="attempt-001",
            backend_job_id=job_id,
        )
        while True:
            status = backend.status({}, run)
            if status["state"] in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                break
            time.sleep(0.02)
        print(json.dumps({"status": status, "logs": backend.logs({}, run, tail=20)}, indent=2))


if __name__ == "__main__":
    main()
