from __future__ import annotations

import json
from pathlib import Path

from experiment_control.backends.services import BackendServices
from experiment_control.runner import CommandResult
from experiment_control.backends.sensecore_rest import RESTError


SUBMISSION_TOKEN = "a" * 32


class QueueRunner:
    def __init__(self, results: list[CommandResult]):
        self.results = list(results)
        self.commands: list[tuple[str, ...]] = []
        self.command_kwargs: list[dict] = []

    def run(self, command, **kwargs):
        self.commands.append(tuple(command))
        self.command_kwargs.append(dict(kwargs))
        if not self.results:
            raise AssertionError(f"unexpected command: {command!r}")
        result = self.results.pop(0)
        if kwargs.get("check", True):
            result.check_returncode()
        return result


class RESTQueue:
    """Adapter tests inject REST responses; no cloud operations are performed."""
    def __init__(self, runner):
        self.runner = runner

    def result(self, operation, *arguments):
        return self.runner.run(["REST", operation, *arguments], check=False)

    def query(self, operation, *arguments):
        result = self.result(operation, *arguments)
        if result.returncode:
            raise RESTError(operation, status=503, uncertain=operation in {"create", "stop"})
        return json.loads(result.stdout or "{}")

    def identity(self):
        return "11111111-1111-4111-8111-111111111111"

    def resources(self, kind):
        self.query("resources", kind)
        return []

    def specs(self, backend):
        return [{"name": backend["worker_spec"]}]

    def find(self, backend, name):
        value = self.query("find", backend["workspace"], name)
        if not isinstance(value, list):
            raise ValueError("invalid job list")
        return [r for r in value if r.get("name") == name]

    def describe(self, backend, name):
        value = self.query("describe", backend["workspace"], name)
        if not isinstance(value, dict):
            raise ValueError("invalid job summary")
        return value

    def create(self, backend, document, **kwargs):
        return self.query("create", json.dumps(document))

    def stop(self, backend, name):
        return self.query("stop", backend["workspace"], name)

    def workers(self, backend, name):
        value = self.query("workers", backend["workspace"], name)
        if not isinstance(value, list):
            raise ValueError("invalid worker list")
        return value

    def logs(self, backend, name, tail):
        result = self.result("logs", backend["workspace"], name)
        text = result.stdout + "\n" + result.stderr
        return {"text": text, "expired": result.returncode not in (0, 124) and "expired" in text,
                "exit_code": result.returncode}


def services(
    tmp_path: Path,
    runner: QueueRunner,
    *,
    record: dict | None = None,
    summary: dict | None = None,
) -> BackendServices:
    backend_record = record or {
        "attempt_id": "attempt-001",
        "backend_job_id": "1234",
    }

    def atomic_write(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    return BackendServices(
        run_command=runner.run,
        local_run_dir=lambda _campaign, _run: tmp_path,
        backend_record=lambda _campaign, _run: dict(backend_record),
        summarize_run=lambda _campaign, _path: dict(summary or {}),
        parse_metric=lambda _campaign, _line: None,
        parse_checkpoint=lambda _campaign, _line: None,
        atomic_write=atomic_write,
        utc_now=lambda: "2026-07-12T00:00:00Z",
        sensecore_rest=lambda: RESTQueue(runner),
    )


def slurm_run() -> dict:
    return {
        "run_id": "backend-run",
        "image_id": "sha256:" + "a" * 64,
        "resources": {"gpus": 1, "cpus": 8},
        "storage": {
            "run_dir": "/shared/project/runs/backend-run",
            "data_root": "/shared",
            "project_data_root": "/shared/project",
            "hf_home": "/shared/project/cache/huggingface",
            "hf_datasets_cache": "/shared/project/cache/huggingface/datasets",
        },
        "backend": {
            "kind": "slurm",
            "ssh_alias": "test-login",
            "partition": "accelerator",
            "account": "lab",
            "qos": "normal",
            "gres": "gpu:accelerator:1",
            "time": "00:10:00",
            "mount_root": "/shared",
            "source_dir": "/shared/project/sources/source-fixed",
            "sif_path": "/shared/project/images/test.sif",
        },
    }


def sensecore_run() -> dict:
    return {
        "run_id": "sensecore-run",
        "image_id": "sha256:" + "b" * 64,
        "backend": {
            "kind": "sensecore",
            "workspace": "workspace",
            "aec2": "cluster",
            "job_name": "sensecore-run",
            "display_name": "render test",
            "image": "registry.example/project/image:source-fixed",
            "worker_spec": "gpu.4",
            "quota_type": "spot",
            "storage_mount": "volume/subdir:/shared",
        },
    }


def submission_intent(backend, run: dict, attempt_id: str = "attempt-001", *, token: str = SUBMISSION_TOKEN) -> dict:
    return {
        "submission_token": token,
        "request": backend.submission_request({}, run, attempt_id),
    }
