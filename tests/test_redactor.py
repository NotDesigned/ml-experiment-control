from __future__ import annotations

import json
import shutil
import subprocess


def run_redactor(payload: str = "", *arguments: str) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("experiment-redact")
    assert executable is not None, "uv sync must install the Rust redactor"
    return subprocess.run([executable, *arguments], input=payload, text=True, capture_output=True, check=False)


def test_redact_lines_handles_assignments_urls_bearer_and_query() -> None:
    raw = (
        'access_key_secret="alpha" token=bravo Authorization: Bearer charlie\n'
        "proxy=https://user:pass@example.test key=https://x.test/?token=delta&ok=1\n"
    )
    result = run_redactor(raw)
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
    result = run_redactor(
        "2026-07-16T12:00:00Z " + prefix + json.dumps(payload) + "\n",
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
    result = run_redactor(prefix + json.dumps(payload) + "\n")
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
    result = run_redactor(f'{prefix}{{"token":"{secret}"\n')
    assert result.returncode == 0
    assert result.stdout == f"{prefix}<redacted-malformed>\n"
    assert secret not in result.stdout


def test_redact_lines_reassembles_transport_fragmented_evidence() -> None:
    prefix = "EXPERIMENT_EVIDENCE_JSON="
    raw = (
        "before\n"
        f'{prefix}{{"run_id":"long-\n'
        'run","token_recon_ppl":\n'
        "23.3}\n"
        "after token=alpha\n"
    )
    result = run_redactor(raw)
    assert result.returncode == 0
    lines = result.stdout.splitlines()
    assert lines[0] == "before"
    assert json.loads(lines[1].removeprefix(prefix)) == {
        "run_id": "long-run", "token_recon_ppl": 23.3,
    }
    assert lines[2] == "after token=<redacted>"


def test_help_and_removed_modes():
    result = run_redactor("", "--help")
    assert result.returncode == 0
    assert "Usage: experiment-redact" in result.stdout
    for mode in ("normalize-state", "worker-list", "job-summary", "job-list", "redact-lines"):
        result = run_redactor("token=never-echo", mode)
        assert result.returncode == 2
        assert "never-echo" not in result.stderr + result.stdout


def test_prefixed_environment_credentials_and_signed_downloads():
    result = run_redactor(
        "ML_EXPD_UPLOAD_TOKEN=private-capability ML_EXPD_BOOTSTRAP_TOKEN=private-bootstrap SOME_PRIVATE_KEY=private-key "
        "https://example.test/?X-Amz-Credential=private-id&X-Amz-Signature=private-sign&ok=1\n"
    )
    assert result.returncode == 0
    for secret in ("private-capability", "private-bootstrap", "private-key", "private-id", "private-sign"):
        assert secret not in result.stdout
