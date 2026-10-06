#!/usr/bin/env python3
"""Install/update only ML-Expd's owned desktop data helper; run while it is idle."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile

from ml_exp_server.worker_contract import WORKERS
import ml_exp_server.desktop_upload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--builder-config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.builder_config.read_text())
    name = config['data_upload_container']
    image = config.get('data_upload_helper_image', config['data_base_image'])
    assert re.fullmatch(r'ml-expd-data-stage[-A-Za-z0-9]*', name)
    assert re.fullmatch(r'[-A-Za-z0-9._:/]+@sha256:[0-9a-f]{64}', image)
    assert re.fullmatch(r'unix:///[-A-Za-z0-9_./]+', config['docker_host'])
    docker = [config.get('docker_bin', 'docker'), '--host', config['docker_host']]
    def run(*arguments, missing=False):
        result = subprocess.run([*docker, *arguments], capture_output=True, text=True, timeout=120)
        if result.returncode and not missing:
            raise RuntimeError('owned desktop helper operation failed')
        return result.stdout.strip() if not result.returncode else None
    # Refuse to adopt or remove a same-named workload/volume owned by another app.
    for volume in (name+'-code', name+'-data'):
        existing = run('volume', 'inspect', volume, missing=True)
        if existing:
            assert json.loads(existing)[0]['Labels'].get('com.ml-expd.owner') == 'data-staging'
        else:
            run('volume', 'create', '--label', 'com.ml-expd.owner=data-staging', volume)
    existing = run('container', 'inspect', name, missing=True)
    if existing:
        assert json.loads(existing)[0]['Config']['Labels'].get('com.ml-expd.owner') == 'data-staging'
        run('stop', name)
        run('rm', name)
    source = Path(ml_exp_server.desktop_upload.__file__).parent
    setup = name+'-setup'
    assert run('container', 'inspect', setup, missing=True) is None
    run('create','--pull','never','--name',setup,'--label','com.ml-expd.owner=data-staging',
        '--mount','type=volume,src='+name+'-code,dst=/app',image,'true')
    try:
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);package=root/'ml_exp_server';package.mkdir();(package/'__init__.py').write_text('')
            for file in ('desktop_upload.py','multipart_upload.py','storage.py','application_errors.py','archive_limits.py','data_image_recipe.py','build_contexts.py'):
                (package/file).write_bytes((source/file).read_bytes())
            workers=root/'workers';workers.mkdir()
            for origin,target in (*WORKERS,('data_copy_worker.py','data_copy_worker.py')):
                (workers/target).write_bytes((source/origin).read_bytes())
            (root/'upload-config.json').write_text(json.dumps({'max_asset_archive_bytes':None,'max_asset_bytes':None,
                'max_asset_files':20000,'upload_part_bytes':config.get('data_upload_part_bytes',16*1024**2),'upload_max_parts':65536,
                'upload_session_seconds':86400,'data_base_image':config['data_base_image']}))
            run('cp',str(root)+'/.',setup+':/app')
    finally:
        run('rm',setup)
    run('run','-d','--pull','never','--name',name,'--restart','unless-stopped','--label','com.ml-expd.owner=data-staging',
        '--network',config['buildkit_network'],'--read-only','--cap-drop=ALL','--security-opt=no-new-privileges',
        '--memory','512m','--cpus','2','--pids-limit','64','--tmpfs','/tmp:rw,noexec,nosuid,size=64m',
        '--mount','type=volume,src='+name+'-code,dst=/app,readonly','--mount','type=volume,src='+name+'-data,dst=/stage',
        '-e','PYTHONPATH=/app','-e','PYTHONDONTWRITEBYTECODE=1',image,'python','-m','ml_exp_server.desktop_upload','serve')
    value=json.loads(run('exec',name,'python','-m','ml_exp_server.desktop_upload','info','{}'))
    assert value['ok']
    print(json.dumps(value['result']))


if __name__=='__main__':
    main()
