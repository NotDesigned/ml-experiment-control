"""Exact-Attempt metadata registration for backend-resident training state."""
from __future__ import annotations

import json
from pathlib import Path

from .artifact_store import ArtifactStore, ATTEMPT
from ..application_errors import ApplicationError
from ..workers.persistent_state import CHECKPOINT_ID, document, digest
from ..projects.source_imports import IDENTITY
from ..storage import atomic_json, utc_now


def storage_scope(backend, project_root):
    # Executor names and GPU sizes do not define a storage namespace.
    origin = backend.get("storage_mount") if backend["kind"] == "sensecore" else backend["ssh_alias"]
    return "storage." + digest({"kind": backend["kind"], "origin": origin, "project_root": project_root})


class CheckpointRegistry:
    def __init__(self, config: Path, root: Path):
        self.objects = ArtifactStore(config, root)
        self.root = root / "persistent-checkpoints"

    def directory(self, project, run, attempt):
        if not IDENTITY.fullmatch(project) or not IDENTITY.fullmatch(run) or not ATTEMPT.fullmatch(attempt):
            raise ValueError("invalid checkpoint provenance")
        return self.root / project / run / attempt

    def read(self, project, run, attempt, checkpoint_id):
        if not CHECKPOINT_ID.fullmatch(checkpoint_id):
            raise ValueError("invalid persistent checkpoint identity")
        path = self.directory(project, run, attempt) / (checkpoint_id + ".json")
        if not path.is_file():
            raise ApplicationError("unknown persistent checkpoint", status_code=404, code="UNKNOWN_CHECKPOINT")
        value = json.loads(path.read_text())
        context = {key: value[key] for key in ("project", "run_id", "attempt_id", "source_id", "image_id", "storage_scope", "state_root")}
        if (context["project"], context["run_id"], context["attempt_id"]) != (project, run, attempt) or document(context, {key: value[key] for key in ("step", "files")})["checkpoint_id"] != checkpoint_id:
            raise ValueError("registered checkpoint identity differs")
        return value

    def list(self, project, run, attempt):
        directory = self.directory(project, run, attempt)
        return {"checkpoints": [self.read(project, run, attempt, path.stem) for path in sorted(directory.glob("checkpoint.*.json"))]}

    def receive(self, project, run, attempt, token, ready):
        self.objects.authorize(project, run, attempt, token)
        with self.objects.record(project, run, attempt) as (_, transfer):
            context = transfer.get("checkpoint_state")
            if not context:
                raise ValueError("persistent checkpoint registration is not enabled")
            value = document(context, ready)
            directory = self.directory(project, run, attempt)
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            path = directory / (value["checkpoint_id"] + ".json")
            if path.exists():
                return self.read(project, run, attempt, value["checkpoint_id"])
            generation = value["files"][0]["path"].split("/")[1]
            for existing in self.list(project, run, attempt)["checkpoints"]:
                if existing["files"][0]["path"].split("/")[1] == generation:
                    raise ValueError("checkpoint generation is already bound to another manifest")
            value.update(status="REGISTERED", verification="worker-sha256", registered_at=utc_now())
            atomic_json(path, value)
            return value
