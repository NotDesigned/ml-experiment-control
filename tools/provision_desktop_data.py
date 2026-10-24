#!/usr/bin/env python3
"""Install/update only ML-Expd's owned desktop data helper; run while it is idle."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile

from ml_exp_worker import source_path
from ml_exp_server.data_image_recipe import DATA_WORKERS
import ml_exp_server.desktop_upload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--builder-config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.builder_config.read_text())
    name = config['data_upload_container']
    image = config.get('data_upload_helper_image', config['data_base_image'])
    if not re.fullmatch(r'ml-expd-data-stage[-A-Za-z0-9]*', name):
        raise ValueError('invalid owned helper name')
    if not re.fullmatch(r'[-A-Za-z0-9._:/]+@sha256:[0-9a-f]{64}', image):
        raise ValueError('helper image must have an immutable digest')
    if not re.fullmatch(r'unix:///[-A-Za-z0-9_./]+', config['docker_host']):
        raise ValueError('helper requires a private Unix Docker socket')
    docker = [config.get('docker_bin', 'docker'), '--host', config['docker_host']]
    def run(*arguments, missing=False):
        result = subprocess.run([*docker, *arguments], capture_output=True, text=True, timeout=120)
        if result.returncode:
            # A disconnected Docker tunnel is not an absent volume/container.
            # Accept only Docker's exact not-found responses for this object.
            kind, _, identity = arguments if missing else ('', '', '')
            identity = identity.lower()
            expected = {
                f'error: no such {kind}: {identity}',
                f'error response from daemon: no such {kind}: {identity}',
            }
            if kind == 'volume':
                expected.add(f'error response from daemon: get {identity}: no such volume')
            if kind == 'container':
                expected.add(f'error: no such object: {identity}')
            if not (missing and result.returncode == 1
                    and result.stderr.strip().lower() in expected):
                raise RuntimeError('owned desktop helper operation failed')
            return None
        return result.stdout.strip()

    def owned(existing, *, container=False):
        value = json.loads(existing)
        metadata = value[0]['Config'] if container else value[0]
        labels = metadata.get('Labels') or {}
        if labels.get('com.ml-expd.owner') != 'data-staging':
            raise RuntimeError('refusing to modify an unowned desktop object')

    # Verify every existing object before changing any workload or volume.
    volumes = {volume: run('volume', 'inspect', volume, missing=True)
               for volume in (name+'-code', name+'-data')}
    for existing in volumes.values():
        if existing is not None:
            owned(existing)
    existing = run('container', 'inspect', name, missing=True)
    if existing is not None:
        owned(existing, container=True)
    setup = name+'-setup'
    if run('container', 'inspect', setup, missing=True) is not None:
        raise RuntimeError('refusing to overwrite an existing helper setup container')
    for volume, metadata in volumes.items():
        if metadata is None:
            run('volume', 'create', '--label', 'com.ml-expd.owner=data-staging', volume)
    if existing is not None:
        run('stop', name)
        run('rm', name)
    source = Path(ml_exp_server.desktop_upload.__file__).parent
    run('create','--pull','never','--name',setup,'--label','com.ml-expd.owner=data-staging',
         '--mount','type=volume,src='+name+'-code,dst=/app',image,'python','-c',
        "import shutil;shutil.rmtree('/app/ml_exp_server',ignore_errors=True);shutil.rmtree('/app/workers',ignore_errors=True);shutil.rmtree('/app/ml_exp_worker',ignore_errors=True)")
    run('start','--attach',setup)
    try:
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);package=root/'ml_exp_server';package.mkdir();(package/'__init__.py').write_text('')
            for file in ('desktop_upload.py','data/multipart_upload.py','storage.py','storage_capacity.py','application_errors.py','archive_limits.py','data_image_recipe.py','builds/build_contexts.py'):
                destination = package/file
                destination.parent.mkdir(exist_ok=True)
                if destination.parent != package:
                    (destination.parent/'__init__.py').write_text('')
                destination.write_bytes((source/file).read_bytes())
            workers=root/'ml_exp_worker';workers.mkdir()
            for file in ('__init__.py', *DATA_WORKERS):
                (workers/file).write_bytes(source_path(file).read_bytes())
            (root/'upload-config.json').write_text(json.dumps({'max_asset_archive_bytes':None,'max_asset_bytes':None,
                'max_asset_files':20000,'upload_part_bytes':config.get('data_upload_part_bytes',16*1024**2),'upload_max_parts':65536,
                'upload_session_seconds':86400,'data_base_image':config['data_base_image'],
                'worker_directory':'/app/ml_exp_worker'}))
            run('cp',str(root)+'/.',setup+':/app')
    finally:
        run('rm',setup)
    run('run','-d','--pull','never','--name',name,'--restart','unless-stopped','--label','com.ml-expd.owner=data-staging',
        '--network',config['buildkit_network'],'--read-only','--cap-drop=ALL','--security-opt=no-new-privileges',
        '--memory','512m','--cpus','2','--pids-limit','64','--tmpfs','/tmp:rw,noexec,nosuid,size=64m',
        '--mount','type=volume,src='+name+'-code,dst=/app,readonly','--mount','type=volume,src='+name+'-data,dst=/stage',
        '-e','PYTHONPATH=/app','-e','PYTHONDONTWRITEBYTECODE=1',image,'python','-m','ml_exp_server.desktop_upload','serve')
    value=json.loads(run('exec',name,'python','-m','ml_exp_server.desktop_upload','info','{}'))
    if value.get('ok') is not True:
        raise RuntimeError('desktop helper health check failed')
    print(json.dumps(value['result']))


if __name__=='__main__':
    main()
