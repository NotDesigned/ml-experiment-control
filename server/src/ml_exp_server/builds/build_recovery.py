"""Read-only proof for retrying an unpublished build; absence alone is not proof."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import http.client
import json
from urllib.parse import urlencode

from .builder_client import BuildTransportError, UnixConnection
from ..data.registry_transport import Registry, MANIFEST
from ..storage import utc_now


def docker_read(config, path, maximum=65536):
    connection = UnixConnection(config['docker_host'].removeprefix('unix://'), timeout=5)
    try:
        connection.request('GET', path, headers={'Host': 'localhost'})
        response = connection.getresponse()
        raw = response.read(maximum + 1)
        if response.status != 200 or len(raw) > maximum:
            raise ValueError('invalid Docker observation')
        return raw
    except (OSError, ValueError, http.client.HTTPException) as exc:
        raise BuildTransportError('BUILDER_DOCKER_UNAVAILABLE', {
            'publication_uncertain': False, 'scheduler_submitted': False,
            'phase': 'CONNECTING_BUILDER', 'error_class': type(exc).__name__,
        }) from None
    finally:
        connection.close()


def desktop_ready(config):
    if config.get('docker_host') and docker_read(config, '/_ping') != b'OK':
        raise BuildTransportError('BUILDER_DOCKER_UNAVAILABLE', {
            'publication_uncertain': False, 'scheduler_submitted': False,
            'phase': 'CONNECTING_BUILDER', 'error_class': 'InvalidPingResponse',
        })


def publication_absence(builder, identity):
    """Require both private locks, no worker/leases and an authenticated 404.

    This never starts/stops a container, builds an image or deletes a receipt.
    A missing local receipt, an offline engine or registry error grants nothing.
    """
    config, root = builder.config, builder.root
    if config.get('publisher') == 'pai-dsw':
        from .pai_build_recovery import observe_dsw_absence
        with (root / (identity + '.lock')).open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return None
            return observe_dsw_absence(builder, identity)
    if not config.get('docker_host') or not config.get('ephemeral_buildkit'):
        return None
    with (root / (identity + '.lock')).open('a') as bundle_lock:
        with (root / 'ephemeral-buildkit.lock').open('a') as worker_lock:
            try:
                fcntl.flock(bundle_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return None
            return observe_absence(builder, identity)


def observe_absence(builder, identity):
    """Caller holds both locks, including when rechecking immediately before build."""
    config, root = builder.config, builder.root
    if any((root / name).exists() for name in (
            identity + '.json', 'ephemeral-builder.json', 'source-context.json', 'storage-probe.json')):
        return None
    try:
        query = urlencode({'all': '1', 'filters': json.dumps({'name': ['buildx_buildkit_ml-expd-']})})
        containers = json.loads(docker_read(config, '/containers/json?' + query))
        if containers != []:
            return None
        registry = Registry(config['repository'], config.get('registry_auth_file', '/root/.docker/config.json'), timeout=15)
        status, _, _ = registry.request('GET', registry.prefix + '/manifests/bundle-' + identity,
                                        headers={'Accept': MANIFEST}, expected=(200, 404))
    except BuildTransportError:
        raise
    except Exception as exc:
        raise BuildTransportError('BUILD_RECOVERY_UNCHECKED', {
            'publication_uncertain': True, 'scheduler_submitted': False,
            'error_class': type(exc).__name__,
        }) from None
    if status != 404:
        raise BuildTransportError('BUILD_PUBLICATION_UNRESOLVED', {
            'publication_uncertain': True, 'scheduler_submitted': False,
            'registry_manifest_status': status,
        })
    return {'source': 'builder-reconcile.v1', 'bundle_id': identity,
            'observed_at': utc_now(), 'registry_manifest_status': 404,
            'active_builders': 0, 'leases_clear': True}


def valid_absence(proof, identity):
    if not isinstance(proof, dict) or set(proof) != {
            'source', 'bundle_id', 'observed_at', 'registry_manifest_status', 'active_builders', 'leases_clear'}:
        return False
    if (proof['source'] not in ('builder-reconcile.v1', 'pai-dsw-reconcile.v1') or proof['bundle_id'] != identity
            or type(proof['registry_manifest_status']) is not int or proof['registry_manifest_status'] != 404
            or type(proof['active_builders']) is not int
            or proof['active_builders'] != 0 or proof['leases_clear'] is not True):
        return False
    try:
        observed = datetime.fromisoformat(proof['observed_at'].replace('Z', '+00:00'))
        age = (datetime.now(timezone.utc) - observed).total_seconds()
        return 0 <= age <= 60
    except (AttributeError, TypeError, ValueError):
        return False
