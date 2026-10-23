"""On-demand immutable OCI delivery, without local image-layer storage."""
import fcntl
import hashlib
import json
import re
import subprocess

from .builder_client import IMAGE, BuildTransportError, builder_request
from ..data.registry_transport import Registry, MANIFEST, OCI_MANIFEST, digest
from ..storage import atomic_json


def target_image(delivery):
    source, tag = delivery['source_image'], delivery['target_tag']
    if (not IMAGE.fullmatch(source)
            or not re.fullmatch(r'[A-Za-z0-9.-]+/[a-z0-9._/-]+:bundle-[0-9a-f]{64}', tag)):
        raise ValueError('invalid frozen image delivery')
    return tag.rsplit(':', 1)[0] + '@' + source.split('@', 1)[1]


def ensure_delivery(delivery):
    target = target_image(delivery)
    result = builder_request(delivery['builder_socket'], {
        'operation': 'mirror', 'source_image': delivery['source_image'], 'target_tag': delivery['target_tag']})
    if result.get('image') != target:
        raise ValueError('image delivery receipt differs from frozen digest')
    return target


def delivery_for_run(run):
    value = run['backend'].get('image_delivery')
    if value is not None and (run['backend']['kind'] != 'sensecore'
            or value['source_image'] != run['container']['image'] or value['target_tag'] != run['backend']['image']):
        raise ValueError('frozen backend image delivery differs from Runtime')
    return value


def mirror(builder, request):
    target = target_image(request)
    source, tag = request['source_image'], request['target_tag']
    policy = builder.config.get('sensecore_mirror', {})
    source_repo, target_repo = source.split('@', 1)[0], tag.rsplit(':', 1)[0]
    if (source_repo != policy.get('source_repository') or target_repo != policy.get('target_repository')
            or source_repo == target_repo):
        raise ValueError('image delivery is outside the exact operator registry policy')
    scope = {**builder.config, 'repository': target_repo,
             'registry_auth_hosts': policy.get('target_auth_hosts', []),
             'registry_blob_hosts': policy.get('target_blob_hosts', [])}
    registry = Registry.from_config(scope, timeout=15)
    root = builder.root / 'mirrors';root.mkdir(mode=0o700, exist_ok=True)
    identity = hashlib.sha256(tag.encode()).hexdigest()
    path = root / (identity + '.json')

    def observed():
        status, _, raw = registry.request('GET', registry.prefix + '/manifests/' + tag.rsplit(':', 1)[1],
                                         headers={'Accept': MANIFEST + ', ' + OCI_MANIFEST}, expected=(200, 404))
        if status == 404:
            return False
        if digest(raw) != source.split('@', 1)[1]:
            raise ValueError('existing mirror tag has a different immutable digest')
        atomic_json(path, {'source_image': source, 'target_tag': tag, 'image': target, 'status': 'VERIFIED'})
        return True

    with (root / (identity + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            previous = json.loads(path.read_text())
            if previous.get('source_image') != source or previous.get('target_tag') != tag:
                raise ValueError('frozen mirror target belongs to another source')
        if not observed():
            atomic_json(path, {'source_image': source, 'target_tag': tag, 'status': 'COPYING'})
            try:
                result = subprocess.run([builder.config.get('skopeo', '/usr/bin/skopeo'), 'copy', '--preserve-digests',
                    '--src-authfile', builder.config['registry_auth_file'], '--dest-authfile', builder.config['registry_auth_file'],
                    'docker://' + source, 'docker://' + tag], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=600, check=False)
                success = result.returncode == 0
            except (OSError, subprocess.SubprocessError):
                success = False
            # A lost copy response is harmless if the exact published manifest
            # can be verified. No GPU/CPU scheduler job is created by delivery.
            if not observed():
                atomic_json(path, {'source_image': source, 'target_tag': tag, 'status': 'RECONCILE_REQUIRED'})
                raise BuildTransportError('IMAGE_MIRROR_UNAVAILABLE', {
                    'copy_reported_success': success, 'publication_uncertain': True, 'scheduler_submitted': False})
        return {'image': target, 'source_image': source, 'verified': True}
