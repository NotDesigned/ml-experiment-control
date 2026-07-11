from experiment_control.backends.base import BackendRegistry
from experiment_control.preflight import PreflightCheck, PreflightReport
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
    assert CommandResult(("true",), 0).returncode == 0
    assert FailureClass.TRANSPORT.value == "transport"
