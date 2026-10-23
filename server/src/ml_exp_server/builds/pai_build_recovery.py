"""Read-only proof before replacing a deleted, unpublished CPU DSW builder."""
import json
import uuid

from experiment_control.backends.pai_api import PaiAPI, PaiError, component

from .builder_client import BuildTransportError
from ..data.registry_transport import Registry, MANIFEST, OCI_MANIFEST
from ..storage import utc_now


def observe_dsw_absence(builder, identity):
    """Caller holds the original bundle lock. An uncertain create grants nothing."""
    root, config = builder.root, builder.config
    lease = root / (identity + '.dsw.json')
    if not lease.is_file() or (root / (identity + '.json')).exists():
        return None
    try:
        value = json.loads(lease.read_text())
        if not isinstance(value, dict):
            raise ValueError('invalid DSW recovery lease')
        scope = config['pai_dsw']
        api = PaiAPI.from_environment()
        if (value.get('phase') != 'DELETE_REQUESTED' or value.get('bundle_id') != identity
                or any(value.get(k) != scope[k] for k in ('workspace_id', 'quota_id'))
                or value.get('region') != api.config['region'] or type(value.get('gpu')) is not int or value['gpu'] != 0):
            return None
        identifier, name = component(value.get('instance_id')), component(value.get('name'))
        try:
            api.request('pai-dsw', 'GetInstance', '/api/v2/instances/' + identifier)
        except PaiError as exc:
            if not (exc.status == 404 or exc.status == 400 and exc.code == 'InstanceNotExistError'):
                raise
        else:
            return None
        seen = set()
        for page in range(1, 101):
            result = api.request('pai-dsw', 'ListInstances', '/api/v2/instances', query={
                'WorkspaceId': scope['workspace_id'], 'InstanceName': name, 'PageSize': 100, 'PageNumber': page})
            rows, total = result.get('Instances'), result.get('TotalCount')
            if not isinstance(rows, list) or type(total) is not int or total < 0:
                raise ValueError('invalid DSW recovery inventory')
            if any(not isinstance(row, dict) or row.get('InstanceName') == name
                   or row.get('InstanceId') == identifier for row in rows):
                return None
            for row in rows:
                key = component(row.get('InstanceId'))
                if key in seen:
                    raise ValueError('duplicate DSW recovery inventory identity')
                seen.add(key)
            if len(seen) == total:
                break
            if not rows or len(seen) > total:
                raise ValueError('incomplete DSW recovery inventory')
        else:
            raise ValueError('DSW recovery inventory exceeds bound')
        registry = Registry.from_config(config, timeout=15)
        status, _, _ = registry.request('GET', registry.prefix + '/manifests/bundle-' + identity,
                                        headers={'Accept': MANIFEST + ', ' + OCI_MANIFEST}, expected=(200, 404))
        if status != 404:
            raise BuildTransportError('BUILD_PUBLICATION_UNRESOLVED', {
                'publication_uncertain': True, 'scheduler_submitted': False, 'registry_manifest_status': status})
        return {'source': 'pai-dsw-reconcile.v1', 'bundle_id': identity, 'observed_at': utc_now(),
                'registry_manifest_status': 404, 'active_builders': 0, 'leases_clear': True}
    except BuildTransportError:
        raise
    except (OSError, ValueError, KeyError, TypeError, PaiError) as exc:
        raise BuildTransportError('BUILD_RECOVERY_UNCHECKED', {
            'publication_uncertain': True, 'scheduler_submitted': False, 'error_class': type(exc).__name__}) from None


def preserve_deleted_lease(builder, identity):
    """Freshly recheck before CREATE_INTENT; preserve the complete old lease."""
    proof = observe_dsw_absence(builder, identity)
    if proof is None:
        raise BuildTransportError('PAI_DSW_RECONCILE_REQUIRED', {
            'publication_uncertain': True, 'scheduler_submitted': False})
    root = builder.root
    source = root / (identity + '.dsw.json')
    history = root / (identity + '.dsw-history')
    history.mkdir(mode=0o700, exist_ok=True)
    # Keep the current lease until the publisher atomically records its next
    # CREATE_INTENT. A crash here leaves the same observable deleted builder.
    target = history / (uuid.uuid4().hex + '.json')
    from experiment_control.manifest import atomic_create
    atomic_create(target, json.loads(source.read_text()))
    target.chmod(0o600)
    return proof
