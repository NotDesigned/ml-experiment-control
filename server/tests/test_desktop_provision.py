"""The staging helper and ACP payload must retain independent base images."""
import json
from pathlib import Path
import runpy
import subprocess
import sys
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
    def run(command, **kwargs):
        commands.append(command)
        operation = command[3:]
        if operation[:2] in (['volume', 'inspect'], ['container', 'inspect']):
            return SimpleNamespace(returncode=1, stdout='')
        if operation[0] == 'cp':
            tree = Path(operation[1])
            copied.update(json.loads((tree / 'upload-config.json').read_text()))
            assert (tree / 'ml_exp_server/data_image_recipe.py').is_file()
        return SimpleNamespace(returncode=0, stdout='{"ok":true,"result":{}}' if operation[0] == 'exec' else '')
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['provision_desktop_data.py', '--builder-config', str(file)])
    tool = Path(__file__).parents[2] / 'tools/provision_desktop_data.py'
    runpy.run_path(str(tool), run_name='__main__')
    creates = [x for x in commands if x[3] in {'create', 'run'}]
    assert len(creates) == 2 and all(helper in x for x in creates)
    assert copied['data_base_image'] == payload
