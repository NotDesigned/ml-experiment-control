"""A server upgrade updates managed controllers without rewriting frozen science."""
from pathlib import Path
import sys

import pytest
import yaml

from ml_exp_server.runtime import _bind_daemon_run_root
from ml_exp_server.schemas import ControllerConfig, ResearchProject, ServerConfig
from tests.test_container_api import client, import_source


def test_existing_managed_project_follows_current_runtime_on_reregistration(client):
    import_source(client)
    runtime=client.app.state.runtime
    project=runtime.project('demo')
    path=Path(project.authored_file)
    data=yaml.safe_load(path.read_text());data['controller']['python']='/previous-release/bin/python'
    path.write_text(yaml.safe_dump(data))
    raw=path.read_bytes()
    reloaded=runtime.register_project(path)
    assert reloaded.controller.python == sys.executable
    assert runtime.project('demo').controller.python == sys.executable
    assert path.read_bytes() == raw


@pytest.mark.parametrize('case',['no-base','external','no-controller','no-capability','custom-tool'])
def test_user_controllers_are_not_rebound(tmp_path,case):
    config=ServerConfig(project_registry_root=str(tmp_path/'registry'),run_root=str(tmp_path/'runs'))
    project=ResearchProject(project='study',title='Study',run_roots=[],
        base_dir=config.project_registry_root_path()/'managed/study',
        controller=ControllerConfig(python='/custom/bin/python',experimentctl='tools/experimentctl.py',capabilities={'container_execution':True}))
    if case=='no-base':project.base_dir=None
    if case=='external':project.base_dir=tmp_path/'external'
    if case=='no-controller':project.controller=None
    if case=='no-capability':project.controller.capabilities={}
    if case=='custom-tool':project.controller.experimentctl='tools/custom.py'
    bound=_bind_daemon_run_root(config,project)
    assert bound.controller is None or bound.controller.python=='/custom/bin/python'
