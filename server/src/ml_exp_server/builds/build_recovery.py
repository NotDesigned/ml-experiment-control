"""Read-only proof for retrying an unpublished build; absence alone is not proof."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
from copy import copy


def publication_builder(builder, identity):
    """A changed default registry only applies to builds without an intent."""
    for suffix in ('.acr.json', '.dsw.json'):
        path = builder.root / (identity + suffix)
        if path.exists():
            repository = json.loads(path.read_text()).get('repository')
            if repository:
                scoped = copy(builder)
                scoped.config = {**builder.config, 'repository': repository}
                return scoped
    return builder


def publication_absence(builder, identity):
    """Require both private locks, no worker/leases and an authenticated 404.

    This never starts/stops a container, builds an image or deletes a receipt.
    A missing local receipt, an offline engine or registry error grants nothing.
    """
    builder = publication_builder(builder, identity)
    config, root = builder.config, builder.root
    if (root / (identity + '.acr.json')).exists():
        # ACR has no native build-status API. An absent image cannot grant a
        # second builder or another tag push after the first webhook intent.
        return None
    if config.get('publisher') == 'pai-dsw':
        from .pai_build_recovery import observe_dsw_absence
        with (root / (identity + '.lock')).open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return None
            return observe_dsw_absence(builder, identity)
    return None


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
