"""Transport-neutral Project service error mapping edges."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_exp_server import project_service as module
from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.project_registry import ProjectRegistryError
from ml_exp_server.project_service import ProjectApplicationService
from ml_exp_server.schemas import ProjectLifecycleState


class Dump(SimpleNamespace):
    def model_dump(self, **_kwargs):
        return dict(self.__dict__)


def test_register_maps_runtime_validation_failure(tmp_path):
    runtime = SimpleNamespace(
        register_project=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("invalid project")
        ),
    )
    with pytest.raises(ApplicationError) as caught:
        ProjectApplicationService(runtime).register((tmp_path / "project.yml").resolve())
    assert caught.value.code == "PROJECT_REGISTRATION_BLOCKED"


def test_register_reports_many_unavailable_roots_and_empty_index_error(
    monkeypatch, tmp_path,
):
    roots = [tmp_path / f"missing-{index}" for index in range(7)]
    project = SimpleNamespace(
        project="demo", resolved_run_roots=lambda: roots,
    )
    record = Dump(project="demo", state="ACTIVE")
    runtime = SimpleNamespace(
        register_project=lambda *_args, **_kwargs: project,
        index=object(), project_records=lambda: [record],
    )
    monkeypatch.setattr(
        module,
        "index_project",
        lambda *_args: (_ for _ in ()).throw(RuntimeError()),
    )
    result = ProjectApplicationService(runtime).register(
        (tmp_path / "project.yml").resolve(),
    )
    assert result["initial_index"]["error"] == "RuntimeError"

    monkeypatch.setattr(module, "index_project", lambda *_args: 0)
    result = ProjectApplicationService(runtime).register(
        (tmp_path / "project.yml").resolve(),
    )
    assert "(+2 more)" in result["initial_index"]["error"]


@pytest.mark.parametrize(("message", "status"), [
    ("unknown registered project", 404),
    ("cannot transition", 409),
])
def test_transition_maps_registry_status(message, status):
    runtime = SimpleNamespace(
        project_records=lambda: [SimpleNamespace(
            project="demo", state=ProjectLifecycleState.ACTIVE,
        )],
        transition_project=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ProjectRegistryError(message)
        ),
    )
    with pytest.raises(ApplicationError) as caught:
        ProjectApplicationService(runtime).transition(
            "demo", "pause", ProjectLifecycleState.PAUSED,
        )
    assert caught.value.status_code == status


def test_unregister_maps_unknown_project():
    runtime = SimpleNamespace(
        unregister_project=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ProjectRegistryError("unknown")
        ),
    )
    with pytest.raises(ApplicationError) as caught:
        ProjectApplicationService(runtime).unregister("missing")
    assert caught.value.code == "UNKNOWN_PROJECT"


def test_unregister_returns_non_destructive_effect():
    runtime = SimpleNamespace(
        unregister_project=lambda *_args, **_kwargs: Dump(
            project="demo", state="ACTIVE",
        ),
    )
    result = ProjectApplicationService(runtime).unregister("demo")
    assert result["unregistered"]["project"] == "demo"
    assert "were not changed" in result["effect"]
