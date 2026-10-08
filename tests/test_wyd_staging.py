"""CPU-only OCI staging checks; no registry, SSH connection or scheduler."""
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest

from backend_harness import QueueRunner, services, slurm_run
from experiment_control.backends.wyd import WydSlurmBackend, _safe_stage_failure_log
from experiment_control.runner import CommandResult


def stage_command(tmp_path, *, credentials=None):
    run = slurm_run()
    run["backend"].update(oci_image="registry/image@" + run["image_id"],
                          sif_path=str(tmp_path / "images" / "image.sif"),
                          apptainer_tmp_dir=str(tmp_path / "scratch"))
    fake = QueueRunner([CommandResult((), 0), CommandResult((), 0)])
    injected = services(tmp_path, fake)
    if credentials:
        injected = replace(injected, oci_pull_environment=lambda: credentials)
    backend = WydSlurmBackend(injected)
    backend._stage_oci_image(run)
    remote = shlex.split(fake.commands[-1][-1])
    if credentials:
        index = remote.index("timeout")
        remote = remote[index:]
        assert json.loads(fake.command_kwargs[-1]["input_text"]) == credentials
        assert credentials["APPTAINER_DOCKER_PASSWORD"] not in " ".join(fake.commands[-1])
    return run, backend, remote


def environment(tmp_path, *, fail=False, sleep=False):
    binaries = tmp_path / "bin"; binaries.mkdir()
    apptainer = binaries / "apptainer"
    apptainer.write_text("#!/bin/sh\n" + (
        "sleep 5\n" if sleep else "exit 42\n" if fail else
        'test -d "$APPTAINER_TMPDIR" || exit 43\n'
        'printf "%s\\n" "$APPTAINER_TMPDIR" > "$STAGING_TEST_SCRATCH"\n'
        'printf sealed-image > "$3"\n'))
    apptainer.chmod(0o755)
    return {**os.environ, "PATH": str(binaries) + ":" + os.environ["PATH"],
            "STAGING_TEST_SCRATCH": str(tmp_path / "scratch-used")}


def execute(tmp_path, command, **kwargs):
    (tmp_path / "images").mkdir(exist_ok=True)
    return subprocess.run(command, text=True, capture_output=True, timeout=5, **kwargs)


def test_conversion_uses_local_scratch_publishes_sealed_receipt_and_reuses(tmp_path):
    run, backend, command = stage_command(tmp_path)
    env = environment(tmp_path)
    result = execute(tmp_path, command, env=env)
    assert result.returncode == 0, result.stderr
    sif = Path(run["backend"]["sif_path"])
    image, digest = Path(str(sif) + ".oci").read_text().split()
    assert image == run["backend"]["oci_image"]
    assert digest == hashlib.sha256(sif.read_bytes()).hexdigest()
    assert (tmp_path / "scratch-used").read_text().startswith(str(tmp_path / "scratch"))
    assert list((tmp_path / "scratch").iterdir()) == []
    assert list((tmp_path / "images").glob("*.tmp.*")) == []
    record = json.loads(Path(backend.stage_progress_path(run)).read_text())
    assert record["phase"] == "SIF_PUBLISH" and record["status"] == "READY"
    # A corrupt receipt must rebuild. A valid receipt verifies the SIF SHA and
    # reuses the image even when the registry/converter is unavailable.
    Path(str(sif) + ".oci").write_text(image + " wrong-sha\n")
    assert execute(tmp_path, command, env=env).returncode == 0
    (tmp_path / "bin" / "apptainer").write_text("#!/bin/sh\nexit 55\n")
    result = execute(tmp_path, command, env=env)
    assert result.returncode == 0
    record = json.loads(Path(backend.stage_progress_path(run)).read_text())
    assert record["phase"] == "CACHE_VERIFY" and record["status"] == "READY"


def test_conversion_failure_keeps_true_exit_code_and_no_partial_receipt(tmp_path):
    run, backend, command = stage_command(tmp_path)
    result = execute(tmp_path, command, env=environment(tmp_path, fail=True))
    assert result.returncode == 42
    record = json.loads(Path(backend.stage_progress_path(run)).read_text())
    assert record["phase"] == "IMAGE_PULL_CONVERT" and record["exit_code"] == 42
    assert record["error_code"] == "STAGE_COMMAND_FAILED"
    assert not Path(run["backend"]["sif_path"] + ".oci").exists()
    assert "OOM" not in result.stderr


def test_lock_wait_is_bounded_without_starting_converter(tmp_path):
    run, backend, command = stage_command(tmp_path)
    (tmp_path / "images").mkdir()
    command[-1] = "0"
    with Path(run["backend"]["sif_path"] + ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = execute(tmp_path, command, env=environment(tmp_path))
    assert result.returncode == 75
    record = json.loads(Path(backend.stage_progress_path(run)).read_text())
    assert record["error_code"] == "LOCK_TIMEOUT"
    assert not (tmp_path / "scratch-used").exists()


def test_remote_budget_terminates_converter_and_records_timeout(tmp_path):
    run, backend, command = stage_command(tmp_path)
    command[3] = "0.2s"
    result = execute(tmp_path, command, env=environment(tmp_path, sleep=True))
    assert result.returncode == 124
    record = json.loads(Path(backend.stage_progress_path(run)).read_text())
    assert record["status"] == "FAILED" and record["exit_code"] == 124
    assert record["error_code"] == "STAGE_TIMEOUT"


def test_private_credentials_stay_in_stdin_and_default_scratch_and_budget(tmp_path):
    run, backend, command = stage_command(tmp_path, credentials={
        "APPTAINER_DOCKER_USERNAME": "example", "APPTAINER_DOCKER_PASSWORD": "private-value",
    })
    assert command[:4] == ["timeout", "--signal=TERM", "--kill-after=15s", "1080s"]
    run["backend"].pop("apptainer_tmp_dir")
    assert backend.image_stage_timeout_seconds(run) == 1080
    run["backend"]["image_stage_timeout_seconds"] = 30
    assert backend.image_stage_timeout_seconds(run) == 30


@pytest.mark.parametrize("budget", [True, "900", 0, 29, 1081, None])
def test_stage_budget_rejects_unbounded_or_malformed_configuration(budget):
    run = slurm_run(); run["backend"]["image_stage_timeout_seconds"] = budget
    with pytest.raises(ValueError, match="30 to 1080"):
        WydSlurmBackend.image_stage_timeout_seconds(run)


@pytest.mark.parametrize("payload", [None, {}, {"phase": "unsafe"},
    {"phase": "CACHE_VERIFY", "status": "unsafe"},
    {"phase": "CACHE_VERIFY", "status": "START", "timestamp": 0},
    {"phase": "CACHE_VERIFY", "status": "START", "timestamp": "invalid"}])
def test_stage_progress_rejects_untrusted_fields(payload):
    assert WydSlurmBackend._safe_stage_event(payload) is None


@pytest.mark.parametrize("code", [None, True, -1, 256, "42", 42])
def test_stage_progress_preserves_only_whitelisted_diagnostics(tmp_path, code):
    event = {"phase": "IMAGE_PULL_CONVERT", "status": "FAILED",
             "timestamp": "2026-10-08T12:00:00Z", "exit_code": code,
             "error_code": "STAGE_COMMAND_FAILED", "token": "private", "path": "/private"}
    fake = QueueRunner([CommandResult((), 0, json.dumps(event))])
    result = WydSlurmBackend(services(tmp_path, fake)).stage_progress(slurm_run())
    assert result["error_code"] == "STAGE_COMMAND_FAILED"
    assert ("exit_code" in result) == (code == 42)
    assert "token" not in result and "path" not in result


@pytest.mark.parametrize("result", [CommandResult((), 1), CommandResult((), 0, "broken")])
def test_unavailable_stage_progress_is_unknown(tmp_path, result):
    backend = WydSlurmBackend(services(tmp_path, QueueRunner([result])))
    assert backend.stage_progress(slurm_run()) is None


@pytest.mark.parametrize("error", [OSError("unavailable"), subprocess.TimeoutExpired("private argv", 15)])
def test_progress_transport_exceptions_are_unknown_without_leaking_arguments(tmp_path, error):
    def unavailable(*args, **kwargs):
        raise error
    backend = WydSlurmBackend(replace(services(tmp_path, QueueRunner([])), run_command=unavailable))
    assert backend.stage_progress(slurm_run()) is None


def test_remote_failure_does_not_forward_raw_registry_stderr(tmp_path, capsys):
    good = {"phase": "IMAGE_PULL_CONVERT", "status": "FAILED",
            "timestamp": "2026-10-08T12:00:00Z", "exit_code": 42}
    fake = QueueRunner([CommandResult((), 0), CommandResult((), 42, stderr=
        "password=private\nML_EXPD_STAGE_EVENT=bad\nML_EXPD_STAGE_EVENT=null\n"
        + "ML_EXPD_STAGE_EVENT=" + json.dumps(good))])
    backend = WydSlurmBackend(services(tmp_path, fake))
    run = slurm_run(); run["backend"]["oci_image"] = "registry@" + run["image_id"]
    with pytest.raises(RuntimeError, match="exit code 42"):
        backend._stage_oci_image(run)
    stderr = capsys.readouterr().err
    assert "private" not in stderr and "IMAGE_PULL_CONVERT" in stderr
    run["backend"]["apptainer_tmp_dir"] = "relative"
    backend = WydSlurmBackend(services(tmp_path, QueueRunner([CommandResult((), 0)])))
    with pytest.raises(ValueError, match="absolute"):
        backend._stage_oci_image(run)


@pytest.mark.parametrize(("event", "code"), [(None, 255),
    ({"phase": "IMAGE_PULL_CONVERT", "status": "START"}, 124),
    ({"phase": "CACHE_VERIFY", "status": "FAILED", "exit_code": 42}, 2)])
def test_transport_or_hard_timeout_emits_actual_failure_code(tmp_path, capsys, event, code):
    stderr = "private registry error"
    if event:
        stderr = "ML_EXPD_STAGE_EVENT=" + json.dumps({**event, "timestamp": "2026-10-08T12:00:00Z"})
    fake = QueueRunner([CommandResult((), 0), CommandResult((), code, stderr=stderr)])
    run = slurm_run(); run["backend"]["oci_image"] = "registry@" + run["image_id"]
    with pytest.raises(RuntimeError, match=f"exit code {code}"):
        WydSlurmBackend(services(tmp_path, fake))._stage_oci_image(run)
    last = json.loads(capsys.readouterr().err.strip().splitlines()[-1].split("=", 1)[1])
    assert last["exit_code"] == code and last["status"] == "FAILED"
    assert last["error_code"] == ("STAGE_TIMEOUT" if code == 124 else "STAGE_COMMAND_FAILED")


def test_real_converter_error_is_preserved_after_shell_failure(tmp_path, capsys):
    run, backend, command = stage_command(tmp_path)
    capsys.readouterr()
    env = environment(tmp_path)
    (tmp_path / "bin" / "apptainer").write_text(
        "#!/bin/sh\nprintf 'FATAL: creating SIF: no space left on device (errno 28)\\n' >&2\nexit 42\n")
    def remote(alias, command, **kwargs):
        result = subprocess.run(shlex.split(command), text=True, capture_output=True, env=env,
                                check=kwargs.get("check", True))
        return CommandResult((), result.returncode, result.stdout, result.stderr)
    backend.remote_exec = remote
    with pytest.raises(RuntimeError, match="exit code 42"):
        backend._stage_oci_image(run)
    stderr = capsys.readouterr().err
    assert "no space left on device (errno 28)" in stderr
    assert '"phase": "IMAGE_PULL_CONVERT"' in stderr and '"exit_code": 42' in stderr
    assert "ML_EXPD_STAGE_LOG_BEGIN" in stderr


def test_failure_log_redacts_pull_secrets_signed_urls_encoded_payloads_and_argv():
    password = "unlabelled-pull-password"
    encoded = "Z" * 100
    signed = "https://storage.example/private-object?X-Amz-Credential=private-id&X-Amz-Signature=private-sign"
    raw = ("FATAL: authentication refused " + password + "\n"
           "WANDB_API_KEY=private-key Authorization: Bearer private-bearer\n"
           "download " + signed + "\n"
           "payload " + encoded + "\n"
           "Command ['ssh', '-o', 'BatchMode=yes', 'private-host', 'private-code'] timed out\n"
           "argv=['python3', '-c', 'private-code']\n"
           "stdin=private-payload\n"
           "+ apptainer build --force /private/image.sif registry/private-image\n"
           "subprocess.TimeoutExpired(['private-code'])\n"
           "FATAL: unsquash failed: HTTP 401\n")
    safe = _safe_stage_failure_log(raw, {"APPTAINER_DOCKER_PASSWORD": password})
    for secret in (password, "private-key", "private-bearer", "private-object", "private-id",
                   "private-sign", encoded, "private-host", "private-code", "private-payload", "registry/private-image"):
        assert secret not in safe
    assert "authentication refused" in safe and "unsquash failed: HTTP 401" in safe
    assert "<redacted-signed-url>" in safe and "<redacted-encoded>" in safe


def test_failure_log_is_utf8_bounded_and_removes_only_internal_markers():
    assert _safe_stage_failure_log("ML_EXPD_STAGE_EVENT={}\n", {}) == ""
    safe = _safe_stage_failure_log("old failure\n" + "错误\n" * 10000 + "FATAL: HTTP 503\x00\n", {"ignored": ""})
    assert len(safe.encode("utf-8")) <= 16384
    assert "old failure" not in safe and safe.endswith("FATAL: HTTP 503")
    assert "\x00" not in safe


def test_private_failure_log_uses_current_pull_credentials(tmp_path, capsys):
    secret = "unlabelled-current-pull-secret"
    fake = QueueRunner([CommandResult((), 0), CommandResult((), 2, stderr="FATAL: registry refused " + secret)])
    injected = replace(services(tmp_path, fake), oci_pull_environment=lambda: {
        "APPTAINER_DOCKER_PASSWORD": secret,
    })
    run = slurm_run(); run["backend"]["oci_image"] = "registry@" + run["image_id"]
    with pytest.raises(RuntimeError, match="exit code 2"):
        WydSlurmBackend(injected)._stage_oci_image(run)
    stderr = capsys.readouterr().err
    assert secret not in stderr and "FATAL: registry refused <redacted>" in stderr
    assert secret not in " ".join(fake.commands[-1])
