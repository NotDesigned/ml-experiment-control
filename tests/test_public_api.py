from experiment_control.backends.base import BackendRegistry
from experiment_control.preflight import PreflightCheck, PreflightReport
from experiment_control.identity import IdentityReport
from experiment_control.project import ProjectRegistry, SourceBundle
from experiment_control.runner import CommandResult
from experiment_control.states import FailureClass


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
    }
