"""Immutable OCI staging and bounded evidence contracts; no remote calls."""

from dataclasses import replace
import json
import os
import subprocess

import pytest

from backend_harness import QueueRunner, sensecore_run, services, slurm_run, submission_intent
from experiment_control.backends.sensecore import SenseCoreBackend
from experiment_control.backends.wyd import WydSlurmBackend
from experiment_control.runner import CommandResult


@pytest.mark.parametrize("encoded,message", [
    ("", "empty or oversized"),
    ("x" * (1024 * 1024 + 1), "empty or oversized"),
    ("{broken", "malformed"),
    ("[]", "must be an object"),
])
def test_cloud_evidence_rejects_unbounded_or_nonobject_payloads(tmp_path, encoded, message):
    line = "EXPERIMENT_EVIDENCE_JSON=" + encoded + "\n"
    runner = QueueRunner([
        CommandResult(("logs",), 0, line),
        CommandResult(("redact",), 0, line),
    ])
    with pytest.raises(RuntimeError, match=message):
        SenseCoreBackend(services(tmp_path, runner)).collect({}, sensecore_run())


@pytest.mark.parametrize("image", ["registry.example/image:latest", "registry.example/image@sha256:" + "b" * 64])
def test_slurm_rejects_mutable_or_conflicting_oci_image(tmp_path, image):
    run = slurm_run()
    run["backend"]["oci_image"] = image
    with pytest.raises(ValueError, match="frozen image digest"):
        WydSlurmBackend(services(tmp_path, QueueRunner([]))).validate(run)


def test_missing_canonical_manifest_prevents_dispatch_and_remote_claim(tmp_path):
    run = slurm_run()
    runner = QueueRunner([CommandResult(("live",), 0,
        "accelerator|up|3-00:00:00|gpu:accelerator:8\nuser|lab||normal|normal\n")])
    backend = WydSlurmBackend(replace(services(tmp_path, runner),
                                     run_manifest_path=lambda *_: tmp_path / "missing.yaml"))
    manifest = {**run, "attempt_id": "attempt-001", "command": ["true"],
                "execution": {"source_mount": "/workspace", "workdir": "/workspace"}}
    with pytest.raises(FileNotFoundError, match="before dispatch"):
        backend.submit({}, run, manifest, dry_run=False, intent=submission_intent(backend, run))
    assert len(runner.commands) == 1


@pytest.mark.parametrize("mode", ["public", "private", "legacy-unbound"])
def test_oci_conversion_reuses_verified_cache_and_repairs_tampering(tmp_path, mode):
    private = mode == "private"
    binpath = tmp_path / "bin"
    binpath.mkdir()
    calls = tmp_path / "builds"
    tool = binpath / "apptainer"
    tool.write_text("#!/usr/bin/python3\nimport os,sys\nfrom pathlib import Path\n"
                    + ("assert os.environ['APPTAINER_DOCKER_PASSWORD']=='test-pull-password'\n" if private else "")
                    + "Path(sys.argv[3]).write_text('verified-image')\n"
                    + f"with Path({str(calls)!r}).open('a') as f:f.write('build\\n')\n")
    tool.chmod(0o700)
    commands = []
    class ShellRunner:
        def run(self, command, **kwargs):
            commands.append(tuple(command))
            proc = subprocess.run(["sh", "-c", command[-1]], text=True,
                                  input=kwargs.get("input_text"), capture_output=True,
                                  env={**os.environ, "PATH": str(binpath) + ":" + os.environ["PATH"]})
            result = CommandResult(tuple(command), proc.returncode, proc.stdout, proc.stderr)
            if kwargs.get("check", True):
                result.check_returncode()
            return result
    runner = ShellRunner()
    if mode == "legacy-unbound":
        backend = object.__new__(WydSlurmBackend)
        backend.remote_exec = lambda alias, command: runner.run(["ssh", alias, command])
    else:
        backend = WydSlurmBackend(replace(services(tmp_path, runner),
            oci_pull_environment=lambda: {"APPTAINER_DOCKER_USERNAME": "test-user", "APPTAINER_DOCKER_PASSWORD": "test-pull-password"} if private else {}))
    sif = tmp_path / "images" / "image.sif"
    run = slurm_run()
    run["backend"].update(sif_path=str(sif), oci_image="registry.example/image@sha256:" + "a" * 64)
    backend._stage_oci_image(run)
    backend._stage_oci_image(run)
    assert calls.read_text() == "build\n"
    sif.write_text("tampered")
    backend._stage_oci_image(run)
    assert calls.read_text() == "build\nbuild\n"
    assert sif.read_text() == "verified-image"
    assert "test-pull-password" not in json.dumps(commands)
    assert not list(sif.parent.glob("*.tmp.*"))
