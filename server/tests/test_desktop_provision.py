"""The staging helper and ACP payload must retain independent base images."""
import json
from pathlib import Path
import runpy
import subprocess
import sys
import os
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('separate', [False, True])
def test_helper_image_does_not_change_payload_from(tmp_path, monkeypatch, separate):
    payload = 'registry.example/copy@sha256:' + 'a' * 64
    helper = 'registry.example/helper@sha256:' + 'b' * 64 if separate else payload
    config = {'data_upload_container': 'ml-expd-data-stage', 'data_base_image': payload,
              'docker_host': 'unix:///run/private.sock', 'buildkit_network': 'private'}
    if separate:
        config['data_upload_helper_image'] = helper
    file = tmp_path / 'builder.json'
    file.write_text(json.dumps(config))
    commands, copied = [], {}
    real_run = subprocess.run
    def run(command, **kwargs):
        commands.append(command)
        operation = command[3:]
        if operation[:2] in (['volume', 'inspect'], ['container', 'inspect']):
            return SimpleNamespace(returncode=1, stdout='', stderr='Error: No such ' + operation[0] + ': ' + operation[-1])
        if operation[0] == 'cp':
            tree = Path(operation[1])
            copied.update(json.loads((tree / 'upload-config.json').read_text()))
            assert (tree / 'ml_exp_server/data_image_recipe.py').is_file()
            # The provisioned helper must import with only the stdlib and its
            # copied files, independent of the developer's installed daemon.
            imported = real_run([sys.executable, '-S', '-c', 'import ml_exp_server.desktop_upload'],
                                cwd=tree, env={**os.environ, 'PYTHONPATH': str(tree)},
                                capture_output=True, text=True, timeout=10)
            assert imported.returncode == 0, imported.stderr
        return SimpleNamespace(returncode=0, stdout='{"ok":true,"result":{}}' if operation[0] == 'exec' else '')
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['provision_desktop_data.py', '--builder-config', str(file)])
    tool = Path(__file__).parents[2] / 'tools/provision_desktop_data.py'
    runpy.run_path(str(tool), run_name='__main__')
    creates = [x for x in commands if x[3] in {'create', 'run'}]
    assert len(creates) == 2 and all(helper in x for x in creates)
    assert copied['data_base_image'] == payload


def invoke_tool(tmp_path, monkeypatch, callback, **overrides):
    config = {'data_upload_container': 'ml-expd-data-stage',
              'data_base_image': 'registry.example/copy@sha256:' + 'a' * 64,
              'docker_host': 'unix:///run/private.sock', 'buildkit_network': 'private',
              **overrides}
    file = tmp_path / 'builder.json'
    file.write_text(json.dumps(config))
    monkeypatch.setattr(subprocess, 'run', callback)
    monkeypatch.setattr(sys, 'argv', ['provision_desktop_data.py', '--builder-config', str(file)])
    tool = Path(__file__).parents[2] / 'tools/provision_desktop_data.py'
    # Run with optimization as well: operational guards must not be assertions.
    namespace = {'__name__': 'provision_guard_trial'}
    exec(compile(tool.read_text(), str(tool), 'exec', optimize=1), namespace)
    namespace['main']()


@pytest.mark.parametrize('field,value', [('data_upload_container', 'someone-else'),
    ('data_base_image', 'registry.example/copy:latest'), ('docker_host', 'tcp://public:2375')])
def test_invalid_configuration_fails_even_with_python_optimization(tmp_path, monkeypatch, field, value):
    with pytest.raises(ValueError):
        invoke_tool(tmp_path, monkeypatch, lambda *_args, **_kwargs: pytest.fail('Docker must not run'), **{field: value})


@pytest.mark.parametrize('error', ['Cannot connect to the Docker daemon',
    'permission denied while trying to connect to the Docker daemon socket',
    'dial unix /run/private.sock: connect: no such file or directory'])
def test_inspect_transport_errors_never_authorize_creating_or_replacing_objects(tmp_path, monkeypatch, error):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1, stdout='', stderr=error)
    with pytest.raises(RuntimeError, match='operation failed'):
        invoke_tool(tmp_path, monkeypatch, run)
    assert len(calls) == 1 and calls[0][3:5] == ['volume', 'inspect']


@pytest.mark.parametrize('kind', ['volume', 'container', 'setup'])
def test_all_ownership_and_collision_checks_precede_mutations(tmp_path, monkeypatch, kind):
    calls = []
    def run(command, **kwargs):
        args = command[3:]
        calls.append(args)
        if args[1] != 'inspect':
            pytest.fail('ownership checks must finish before mutations')
        if (kind == 'volume' and args[0] == 'volume'
                or kind == 'container' and args[-1] == 'ml-expd-data-stage'
                or kind == 'setup' and args[-1].endswith('-setup')):
            metadata = {'Labels': {'com.ml-expd.owner': 'other-application'}}
            return SimpleNamespace(returncode=0, stdout=json.dumps([{'Config': metadata, **metadata}]), stderr='')
        return SimpleNamespace(returncode=1, stdout='', stderr='Error: No such ' + args[0] + ': ' + args[-1])
    with pytest.raises(RuntimeError, match='unowned|existing helper setup'):
        invoke_tool(tmp_path, monkeypatch, run)
    assert calls
