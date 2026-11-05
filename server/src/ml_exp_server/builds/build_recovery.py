"""Read-only proof for retrying an unpublished build; absence alone is not proof."""
from __future__ import annotations

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
