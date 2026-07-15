from __future__ import annotations

import json
import shutil
import subprocess

import pytest


def run_safe(
    mode: str, payload: str = "", value: str | None = None,
) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("experiment-safe-sco")
    assert executable is not None, "uv sync must install the Rust sanitizer binary"
    command = [executable, mode]
    if value is not None:
        command.append(value)
    return subprocess.run(
        command, input=payload, text=True, capture_output=True, check=False,
    )


def test_job_summary_is_allowlisted_and_normalized() -> None:
    payload = {
        "name": "job-1",
        "display_name": "trial",
        "state": "SUSPENDED",
        "create_time": "today",
        "resource_pool": {"name": "pool", "secret": "do-not-print"},
        "roles": [{
            "image_path": "https://user:password@registry/example@sha256:abc",
            "command": "echo access_key_secret=do-not-print",
            "env": {"TOKEN": "do-not-print"},
            "resource_spec": [{
                "name": "accelerator-pool", "password": "do-not-print",
            }],
        }],
        "mount": [{
            "id": "vol", "subdir": "project", "mount_path": "/shared",
            "token": "no",
        }],
    }
    result = run_safe("job-summary", json.dumps(payload))
    assert result.returncode == 0
    summary = json.loads(result.stdout)
    assert summary["normalized_state"] == "PREEMPTED"
    assert summary["spec"] == "accelerator-pool"
    assert summary["mounts"] == [{
        "id": "vol", "mount_path": "/shared", "subdir": "project",
    }]
    assert "do-not-print" not in result.stdout
    assert "user:password" not in result.stdout
    assert "command" not in summary


def test_job_list_ignores_non_objects_and_accepts_fixed_empty_sentinels() -> None:
    result = run_safe(
        "job-list", json.dumps([{"name": "one", "state": "RUNNING"}, "bad"]),
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)[0]["normalized_state"] == "RUNNING"

    for payload in ("   \n", "No jobs found\n"):
        result = run_safe("job-list", payload)
        assert result.returncode == 0
        assert json.loads(result.stdout) == []

    unexpected = run_safe("job-list", "No resources available\n")
    assert unexpected.returncode != 0
    assert "raw response suppressed" in unexpected.stderr


def test_worker_list_keeps_phase_but_drops_addresses_and_continuations() -> None:
    table = """\
+----------+----------+---------+--------+---------+
| WORKER_NAME | RESOURCE | HOST_IP | POD_IP | PHASE |
+----------+----------+---------+--------+---------+
| job-worker-0 | 4 accelerators | 10.0.0.1 | 10.0.0.2 | Pending |
|              | 56 CPUs        |          |          |         |
+----------+----------+---------+--------+---------+
"""
    result = run_safe("worker-list", table)
    assert result.returncode == 0
    assert json.loads(result.stdout) == [{
        "phase": "Pending", "worker_name": "job-worker-0",
    }]
    assert "10.0.0" not in result.stdout


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ("| NAME | TOKEN |\n| worker | secret |\n", "unexpected worker table schema"),
        (
            "| WORKER_NAME | RESOURCE | HOST_IP | POD_IP | PHASE |\n"
            "| too | few | cells |\n",
            "malformed worker table",
        ),
        (
            "| WORKER_NAME | RESOURCE | HOST_IP | POD_IP | PHASE |\n"
            "| | gpu | address | | |\n",
            "malformed worker continuation",
        ),
        (
            "| WORKER_NAME | RESOURCE | HOST_IP | POD_IP | PHASE |\n"
            "| unsafe/name | gpu | | | Running |\n",
            "unsafe worker identity",
        ),
    ],
)
def test_worker_list_fails_closed_without_echoing_input(table, message) -> None:
    result = run_safe("worker-list", table)
    assert result.returncode != 0
    assert message in result.stderr
    assert "secret" not in result.stderr


def test_malformed_json_and_wrong_shapes_fail_without_echoing_input() -> None:
    secret = "must-not-echo"
    malformed = run_safe("job-summary", f"secret={secret} {{")
    assert malformed.returncode != 0
    assert secret not in malformed.stderr
    assert "raw response suppressed" in malformed.stderr

    summary_list = run_safe("job-summary", "[]")
    assert summary_list.returncode != 0
    assert "expected one JSON job object" in summary_list.stderr

    list_object = run_safe("job-list", "{}")
    assert list_object.returncode != 0
    assert "expected a JSON job array" in list_object.stderr


def test_redact_lines_handles_assignments_urls_bearer_and_query() -> None:
    raw = (
        'access_key_secret="alpha" token=bravo Authorization: Bearer charlie\n'
        "proxy=https://user:pass@example.test key=https://x.test/?token=delta&ok=1\n"
    )
    result = run_safe("redact-lines", raw)
    assert result.returncode == 0
    for secret in ("alpha", "bravo", "charlie", "user:pass", "delta"):
        assert secret not in result.stdout
    assert result.stdout.count("<redacted>") >= 5


def test_redact_lines_preserves_structured_scientific_token_fields() -> None:
    payload = {
        "run_id": "run-1",
        "attempt_id": "attempt-001",
        "image_id": "sha256:abc",
        "token_recon_ppl": 23.3,
        "oracle_plan_token_denoising_l2": 2.1,
        "sampled_plan_num_samples": 16,
        "tokenizer_path": "/data/tokenizer",
        "proxy_loss": 0.4,
    }
    prefix = "EXPERIMENT_EVIDENCE_JSON="
    result = run_safe(
        "redact-lines", "2026-07-16T12:00:00Z " + prefix + json.dumps(payload) + "\n",
    )
    assert result.returncode == 0
    evidence = json.loads(result.stdout.split(prefix, 1)[1])
    assert evidence == payload


def test_redact_lines_structurally_redacts_evidence_secrets() -> None:
    payload = {
        "submission_token": "alpha",
        "refreshToken": "bravo",
        "WANDB_API_KEY": "echo",
        "nested": {
            "access_key_secret": "foxtrot",
            "metric_url": "https://user:pass@example.test/?token=charlie",
        },
        "message": "Authorization: Bearer delta",
    }
    prefix = "EXPERIMENT_EVIDENCE_JSON="
    result = run_safe("redact-lines", prefix + json.dumps(payload) + "\n")
    assert result.returncode == 0
    for secret in (
        "alpha", "bravo", "echo", "foxtrot", "user:pass", "charlie", "delta",
    ):
        assert secret not in result.stdout
    evidence = json.loads(result.stdout.removeprefix(prefix))
    assert evidence["submission_token"] == "<redacted>"
    assert evidence["refreshToken"] == "<redacted>"
    assert evidence["WANDB_API_KEY"] == "<redacted>"
    assert evidence["nested"]["access_key_secret"] == "<redacted>"


def test_redact_lines_suppresses_malformed_structured_evidence() -> None:
    secret = "must-not-echo"
    prefix = "EXPERIMENT_EVIDENCE_JSON="
    result = run_safe("redact-lines", f'{prefix}{{"token":"{secret}"\n')
    assert result.returncode == 0
    assert result.stdout == f"{prefix}<redacted-malformed>\n"
    assert secret not in result.stdout


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("WAITING", "QUEUED"),
        ("completed", "SUCCEEDED"),
        ("SUSPENDING", "PREEMPTED"),
        ("CANCELED", "CANCELLED"),
        ("future-state", "UNKNOWN"),
    ],
)
def test_normalize_state_covers_scheduler_terminals(raw, expected) -> None:
    result = run_safe("normalize-state", value=raw)
    assert result.returncode == 0
    assert result.stdout.strip() == expected


def test_normalize_state_exits_while_inherited_stdin_remains_open() -> None:
    executable = shutil.which("experiment-safe-sco")
    assert executable is not None
    process = subprocess.Popen(
        [executable, "normalize-state", "RUNNING"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.wait(timeout=1) == 0
        assert process.stdout is not None
        assert process.stdout.read() == "RUNNING\n"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def test_help_and_argument_errors_are_stable() -> None:
    executable = shutil.which("experiment-safe-sco")
    assert executable is not None
    help_result = subprocess.run(
        [executable, "--help"], text=True, capture_output=True, check=False,
    )
    assert help_result.returncode == 0
    assert "Usage: experiment-safe-sco <MODE> [VALUE]" in help_result.stdout

    missing = run_safe("normalize-state")
    assert missing.returncode != 0
    assert "requires a state value" in missing.stderr
