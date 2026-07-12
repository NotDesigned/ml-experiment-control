from __future__ import annotations

import json
import io
import runpy
import subprocess
import sys

import pytest

from experiment_control import safe_sco


def run_safe(
    mode: str, payload: str = "", value: str | None = None
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", "experiment_control.safe_sco", mode]
    if value is not None:
        command.append(value)
    return subprocess.run(
        command, input=payload, text=True, capture_output=True, check=False
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
                "name": "accelerator-pool", "password": "do-not-print"
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
        "id": "vol", "mount_path": "/shared", "subdir": "project"
    }]
    assert "do-not-print" not in result.stdout
    assert "user:password" not in result.stdout
    assert "command" not in summary


def test_job_list_ignores_non_objects() -> None:
    result = run_safe(
        "job-list", json.dumps([{"name": "one", "state": "RUNNING"}, "bad"])
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)[0]["normalized_state"] == "RUNNING"


def test_job_list_normalizes_successful_empty_stdout() -> None:
    result = run_safe("job-list", "   \n")
    assert result.returncode == 0
    assert json.loads(result.stdout) == []


def test_job_list_normalizes_only_the_sco_v1_2_no_match_sentinel() -> None:
    result = run_safe("job-list", "No jobs found\n")
    assert result.returncode == 0
    assert json.loads(result.stdout) == []

    unexpected = run_safe("job-list", "No resources available\n")
    assert unexpected.returncode != 0
    assert "raw response suppressed" in unexpected.stderr


def test_job_summary_tolerates_missing_external_state() -> None:
    result = run_safe("job-summary", json.dumps({"name": "job-without-state"}))
    assert result.returncode == 0
    assert json.loads(result.stdout)["normalized_state"] == "UNKNOWN"


def test_worker_list_keeps_phase_but_drops_network_addresses() -> None:
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
    workers = json.loads(result.stdout)
    assert workers == [{
        "phase": "Pending", "worker_name": "job-worker-0",
    }]
    assert "10.0.0" not in result.stdout


def test_worker_list_fails_closed_on_unknown_table_schema() -> None:
    result = run_safe("worker-list", "| NAME | TOKEN |\n| worker | secret |\n")
    assert result.returncode != 0
    assert "secret" not in result.stderr
    assert "raw response suppressed" in result.stderr


def test_malformed_json_is_not_echoed() -> None:
    secret = "must-not-echo"
    result = run_safe("job-summary", f"secret={secret} {{")
    assert result.returncode != 0
    assert secret not in result.stderr
    assert "raw response suppressed" in result.stderr


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


def test_normalize_state_covers_scheduler_terminals() -> None:
    cases = {
        "WAITING": "QUEUED",
        "completed": "SUCCEEDED",
        "SUSPENDING": "PREEMPTED",
        "CANCELED": "CANCELLED",
        "future-state": "UNKNOWN",
    }
    for raw, expected in cases.items():
        result = run_safe("normalize-state", value=raw)
        assert result.returncode == 0
        assert result.stdout.strip() == expected


def direct_main(monkeypatch, mode: str, payload: str = "", value: str | None = None):
    stdin = io.StringIO(payload)
    stdout = io.StringIO()
    monkeypatch.setattr(safe_sco.sys, "stdin", stdin)
    monkeypatch.setattr(safe_sco.sys, "stdout", stdout)
    argv = [mode] + ([value] if value is not None else [])
    code = safe_sco.main(argv)
    return code, stdout.getvalue()


def test_direct_safe_sco_modes_cover_in_process_cli_contract(monkeypatch):
    code, output = direct_main(monkeypatch, "normalize-state", value="RUNNING")
    assert code == 0 and output.strip() == "RUNNING"

    code, output = direct_main(monkeypatch, "redact-lines", "token=secret\nplain\n")
    assert code == 0 and "secret" not in output and "plain" in output

    table = "| WORKER_NAME | RESOURCE | HOST_IP | POD_IP | PHASE |\n| w-0 | gpu | 1 | 2 | Running |\n"
    code, output = direct_main(monkeypatch, "worker-list", table)
    assert code == 0 and json.loads(output) == [{"phase": "Running", "worker_name": "w-0"}]

    code, output = direct_main(monkeypatch, "job-summary", '{"name":"job","state":"FAILED"}')
    assert code == 0 and json.loads(output)["normalized_state"] == "FAILED"

    code, output = direct_main(monkeypatch, "job-list", '[{"name":"job"},null]')
    assert code == 0 and [item["name"] for item in json.loads(output)] == ["job"]


def test_direct_safe_sco_rejects_invalid_cli_payloads(monkeypatch):
    with pytest.raises(SystemExit) as missing:
        direct_main(monkeypatch, "normalize-state")
    assert missing.value.code == 2
    with pytest.raises(SystemExit, match="expected one JSON job object"):
        direct_main(monkeypatch, "job-summary", "[]")
    with pytest.raises(SystemExit, match="expected a JSON job array"):
        direct_main(monkeypatch, "job-list", "{}")
    with pytest.raises(SystemExit, match="raw response suppressed"):
        safe_sco.read_json(io.StringIO("secret=not-json"))


def test_safe_sco_helpers_fail_closed_on_malformed_worker_tables():
    assert safe_sco.safe_text(3) == 3
    assert safe_sco.worker_list("no table") == []
    with pytest.raises(SystemExit, match="unexpected worker table schema"):
        safe_sco.worker_list("| NAME | TOKEN |\n| worker | secret |\n")
    with pytest.raises(SystemExit, match="malformed worker table"):
        safe_sco.worker_list(
            "| WORKER_NAME | RESOURCE | HOST_IP | POD_IP | PHASE |\n| too | few | cells |\n"
        )
    with pytest.raises(SystemExit, match="malformed worker continuation"):
        safe_sco.worker_list(
            "| WORKER_NAME | RESOURCE | HOST_IP | POD_IP | PHASE |\n"
            "| | gpu | address | | |\n"
        )
    with pytest.raises(SystemExit, match="unsafe worker identity"):
        safe_sco.worker_list(
            "| WORKER_NAME | RESOURCE | HOST_IP | POD_IP | PHASE |\n"
            "| unsafe/name | gpu | | | Running |\n"
        )


def test_job_summary_tolerates_non_mapping_optional_sections():
    summary = safe_sco.job_summary({
        "name": "job", "roles": ["bad"], "resource_pool": "bad",
        "mount": ["bad"],
    })
    assert summary["pool"] is None
    assert summary["spec"] is None
    assert summary["mounts"] == []


def test_direct_empty_job_list_and_valid_worker_continuation():
    assert safe_sco.read_json(io.StringIO("No jobs found\n"), empty_list=True) == []
    workers = safe_sco.worker_list(
        "| WORKER_NAME | RESOURCE | HOST_IP | POD_IP | PHASE |\n"
        "| worker-0 | 4 GPUs | | | Running |\n"
        "| | 56 CPUs | | | |\n"
    )
    assert workers == [{"worker_name": "worker-0", "phase": "Running"}]


def test_module_entrypoint_delegates_to_main(monkeypatch):
    monkeypatch.setattr(sys, "argv", [safe_sco.__file__, "normalize-state", "RUNNING"])
    with pytest.raises(SystemExit) as completed:
        runpy.run_path(safe_sco.__file__, run_name="__main__")
    assert completed.value.code == 0
