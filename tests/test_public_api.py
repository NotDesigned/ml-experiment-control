from experiment_control.backends.base import BackendRegistry
from experiment_control.backends.sensecore import digest_pinned_image, scheduler_job_name
from experiment_control.preflight import PreflightCheck, PreflightReport
from experiment_control.identity import IdentityReport
from experiment_control.project import ProjectRegistry, SourceBundle
from experiment_control.runner import CommandResult
from experiment_control.states import FailureClass
import pytest


def test_package_public_primitives_have_no_host_dependency(tmp_path):
    report = PreflightReport(
        "fake", "submit", (PreflightCheck("tool", "tool", "PASS"),)
    )
    assert report.ready is True
    assert BackendRegistry().kinds == frozenset()
    assert ProjectRegistry().names == frozenset()
    assert SourceBundle(tmp_path).container_path == "/workspace"
    assert SourceBundle(tmp_path).required_paths == ()
    assert CommandResult(("true",), 0).returncode == 0
    assert FailureClass.TRANSPORT.value == "transport"
    assert IdentityReport(True, False).to_dict() == {
        "available": True,
        "ambiguous": False,
        "scheduler_job_ids": [],
        "remote_manifest_exists": None,
        "remote_manifest_matches": None,
    }


def test_sensecore_attempt_and_image_identities_are_deterministic():
    digest = "sha256:" + "d" * 64
    assert scheduler_job_name("run", "attempt-003") == "run--attempt-003"
    assert scheduler_job_name("r" * 80, "attempt-003") == scheduler_job_name(
        "r" * 80, "attempt-003"
    )
    assert len(scheduler_job_name("r" * 80, "attempt-003")) <= 63
    assert digest_pinned_image("registry.example/ns/image:source-abc", digest) == (
        f"registry.example/ns/image@{digest}"
    )


def test_preflight_report_serialization_and_fail_closed_requirement():
    report = PreflightReport(
        "fake", "submit", (
            PreflightCheck("tool", "tool", "PASS"),
            PreflightCheck("access", "authorization", "FAIL", "login required"),
        ),
    )
    assert report.ready is False
    assert report.to_dict()["checks"][0] == {
        "name": "tool", "category": "tool", "status": "PASS",
    }
    with pytest.raises(RuntimeError, match="access"):
        report.require_ready()
