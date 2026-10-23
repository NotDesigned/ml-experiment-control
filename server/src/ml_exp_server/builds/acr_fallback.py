"""One immutable Git tag for ACR Personal's native build fallback.

Personal ACR has no public build-status API. Observe the authenticated registry;
an absent image never proves a Git-triggered build has stopped or may be replayed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

from .builder_client import BuildTransportError
from .image_build_context import BUILD_PROGRESS
from ..data.registry_transport import Registry, MANIFEST, OCI_MANIFEST, digest
from ..projects.source_revisions import _tree_digest
from ..runs.execution_progress import record_progress
from ..storage import atomic_json


def phase(name, message, *, timeout_seconds=None):
    path = BUILD_PROGRESS.get()
    if path:
        record_progress(path, name, message, timeout_seconds=timeout_seconds)


def git(directory, args, *, data=None, stream=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_TERMINAL_PROMPT='0', GIT_AUTHOR_NAME='ML-Expd',
               GIT_AUTHOR_EMAIL='build@ml-expd.invalid', GIT_COMMITTER_NAME='ML-Expd',
               GIT_COMMITTER_EMAIL='build@ml-expd.invalid',
               GIT_AUTHOR_DATE='2000-01-01T00:00:00Z', GIT_COMMITTER_DATE='2000-01-01T00:00:00Z')
    result = subprocess.run(['git', '-c', 'core.hooksPath=/dev/null',
                             '-c', 'credential.helper=', '-c', 'credential.helper=!gh auth git-credential',
                             '--git-dir=' + str(directory), *args],
                            input=data, stdin=stream, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=env, timeout=120, check=False)
    if result.returncode or len(result.stdout) > 1024**2:
        # Git errors can contain credential helper output; never publish it.
        raise ValueError('ACR fallback Git operation failed')
    return result.stdout


def commit_context(directory, context, identity):
    """Plumbing bypasses client .gitignore, filters, hooks and Git configuration."""
    git(directory, ['init', '--bare', '--quiet', str(directory)])
    size = 0

    def tree(root):
        nonlocal size
        rows = []
        for path in sorted(root.iterdir()):
            if path.is_symlink() or path.name.lower() == '.git':
                raise ValueError('ACR fallback context contains an unsupported path')
            if path.is_dir():
                if not any(path.iterdir()):
                    raise ValueError('ACR Git fallback cannot preserve empty directories')
                mode, kind, sha = '040000', 'tree', tree(path)
            elif path.is_file():
                length = path.stat().st_size
                size += length
                if length > 95 * 1024**2 or size > 300 * 1024**2:
                    raise ValueError('ACR Git fallback exceeds its source-only context limit')
                with path.open('rb') as stream:
                    sha = git(directory, ['hash-object', '-w', '--stdin'], stream=stream).strip()
                mode, kind = ('100755' if path.stat().st_mode & 0o111 else '100644'), 'blob'
            else:
                raise ValueError('ACR fallback context contains a special file')
            rows.append(mode.encode() + b' ' + kind.encode() + b' ' + sha + b'\t' + os.fsencode(path.name) + b'\0')
        return git(directory, ['mktree', '-z'], data=b''.join(rows)).strip()

    return git(directory, ['commit-tree', tree(context).decode()], data=(identity + '\n').encode()).decode().strip()


def journal(builder, identity):
    return builder.root / (identity + '.acr.json')


def private_repository(remote):
    result = subprocess.run(['gh', 'api', 'repos/' + remote.removeprefix('https://github.com/').removesuffix('.git'),
                             '--jq', '.private'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=30, check=False)
    if result.returncode or result.stdout.strip() != b'true':
        raise ValueError('ACR fallback Git repository must be verifiably private')


def publication(builder, value):
    identity = value['bundle_id']
    if (value['repository'] != builder.config['repository'] or value['tag'] != 'bundle-' + identity
            or value['receipt']['bundle_id'] != identity):
        raise ValueError('ACR fallback publication scope differs')
    registry = Registry.from_config(builder.config, timeout=15)
    status, _, raw = registry.request('GET', registry.prefix + '/manifests/' + value['tag'],
                                     headers={'Accept': MANIFEST + ', ' + OCI_MANIFEST}, expected=(200, 404))
    if status == 404:
        return None
    manifest = json.loads(raw)
    if manifest.get('mediaType') not in (MANIFEST, OCI_MANIFEST):
        raise ValueError('ACR fallback must publish one image manifest')
    config = registry.json_blob(manifest['config']['digest'])
    if (config.get('os') != 'linux' or config.get('architecture') != 'amd64'
            or config.get('config', {}).get('Labels', {}).get('org.ml-expd.source') != value['receipt']['source_id']):
        raise ValueError('ACR fallback image source or platform differs')
    image = value['repository'] + '@' + digest(raw)
    if value.get('image') not in (None, image):
        raise ValueError('ACR fallback published digest changed')
    return image


def reconcile(builder, definition):
    """Recover a late publication without pushing a tag or starting a DSW."""
    path = journal(builder, definition['bundle_id'])
    value = json.loads(path.read_text())
    if any(value['receipt'].get(k) != v for k, v in definition.items()):
        raise ValueError('ACR fallback frozen receipt differs')
    image = publication(builder, value)
    if image is None:
        return None
    value.update(phase='PUBLISHED', image=image)
    atomic_json(path, value)
    receipt = {**value['receipt'], 'image': image}
    atomic_json(builder.root / (definition['bundle_id'] + '.json'), receipt)
    return receipt


def publish(builder, context, receipt):
    try:
        return _publish(builder, context, receipt)
    except BuildTransportError:
        raise
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        raise BuildTransportError('ACR_NATIVE_BUILD_UNCERTAIN', {
            'publication_uncertain': journal(builder, receipt['bundle_id']).exists(),
            'scheduler_submitted': False, 'error_class': type(error).__name__}) from None


def _publish(builder, context, receipt):
    identity = receipt['bundle_id']
    scope = builder.config['acr_fallback']
    remote = scope.get('git_repository')
    seconds = scope.get('wait_seconds', 300)
    if (not isinstance(remote, str) or not re.fullmatch(r'https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', remote)
            or type(seconds) is not int or not 0 <= seconds <= 2700):
        raise ValueError('ACR fallback requires an explicit private GitHub source and bounded wait')
    private_repository(remote)
    path = journal(builder, identity)
    with tempfile.TemporaryDirectory(prefix='acr-git-', dir=builder.root) as directory:
        context_digest = _tree_digest(context)
        commit = commit_context(Path(directory), context, identity)
        if _tree_digest(context) != context_digest:
            raise ValueError('ACR fallback context changed during Git packaging')
        expected = {'bundle_id': identity, 'repository': builder.config['repository'],
                    'tag': 'bundle-' + identity, 'git_repository': remote, 'git_commit': commit,
                    'context_digest': context_digest, 'receipt': receipt}
        existing = path.exists()
        value = json.loads(path.read_text()) if existing else expected
        if any(value.get(k) != v for k, v in expected.items()):
            raise ValueError('ACR fallback immutable context or target changed')
        ref = 'refs/tags/release-vbundle-' + identity
        rows = git(directory, ['ls-remote', '--refs', remote, ref]).decode().splitlines()
        if rows and rows != [commit + '\t' + ref]:
            raise ValueError('ACR fallback Git tag is owned by another context')
        if not rows:
            if existing:
                raise BuildTransportError('ACR_NATIVE_BUILD_UNCERTAIN', {
                    'publication_uncertain': True, 'scheduler_submitted': False, 'tag': value['tag']})
            value = {**expected, 'phase': 'PUSH_INTENT'}
            atomic_json(path, value)
            phase('TRIGGERING_ACR_BUILD', 'Publishing one immutable source tag to the private ACR build repository')
            # Never replay this push after an uncertain response, even if the
            # registry is still empty: ACR may already have accepted its webhook.
            git(directory, ['push', '--porcelain', remote, commit + ':' + ref])
        value['phase'] = 'OBSERVING'
        atomic_json(path, value)
    end = time.monotonic() + seconds
    while True:
        image = publication(builder, value)
        if image is not None:
            value.update(phase='PUBLISHED', image=image)
            atomic_json(path, value)
            return image
        phase('WAITING_ACR_BUILD', 'ACR native build triggered; waiting for authenticated image publication',
              timeout_seconds=seconds)
        if time.monotonic() >= end:
            raise BuildTransportError('ACR_NATIVE_BUILD_PENDING', {
                'publication_uncertain': True, 'scheduler_submitted': False,
                'tag': value['tag'], 'reconcile_required': True})
        time.sleep(5)


def publish_runtime(builder, tag, context, receipt):
    from .pai_dsw import DswPublisher
    from .pai_build_recovery import observe_dsw_absence
    identity = receipt['bundle_id']
    path = journal(builder, identity)
    if path.exists():
        return publish(builder, context, receipt)
    lease = builder.root / (identity + '.dsw.json')
    if builder.config.get('acr_fallback') and lease.exists() and observe_dsw_absence(builder, identity) is not None:
        return publish(builder, context, receipt)
    try:
        return DswPublisher(builder).publish(tag, context)
    except BuildTransportError:
        # Cleanup + native Get/List absence + authenticated manifest absence
        # must all succeed under the caller's bundle lock before switching.
        if not builder.config.get('acr_fallback'):
            raise
        deadline = time.monotonic() + 60
        while observe_dsw_absence(builder, identity) is None:
            # DeleteInstance is asynchronous. Only a recorded deletion request
            # warrants bounded read-only waiting; uncertain creates never do.
            if (not lease.exists() or json.loads(lease.read_text()).get('phase') != 'DELETE_REQUESTED'
                    or time.monotonic() >= deadline):
                raise
            phase('WAITING_DSW_DELETION', 'Waiting for native DSW absence before switching builders')
            time.sleep(3)
    phase('FALLBACK_TO_ACR', 'DSW removed and publication absent; switching this same context to native ACR build')
    return publish(builder, context, receipt)
