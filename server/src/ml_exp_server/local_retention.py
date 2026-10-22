"""Opt-in local payload expiry; backend checkpoints and records are not deleted."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat

from .backends.states import TERMINAL
from .data.data_assets import AssetStore, ASSET_ID
from .projects.source_imports import IDENTITY, SHA256
from .results.artifact_cache import directory, writable, evict_file, prune_empty, signature
from .results.artifact_store import stream_digest
from .storage import atomic_json, utc_now

PAYLOAD_SUFFIXES = {'.pt', '.pth', '.ckpt', '.safetensors', '.onnx', '.bin', '.npy', '.npz'}


def retention_seconds(config):
    value = config.get('local_payload_retention_seconds')
    if value is not None and (type(value) is not int or value < 60):
        raise ValueError('local payload retention must be at least 60 seconds')
    return value


def expired_timestamp(value, seconds, now):
    return (now - datetime.fromisoformat(value.replace('Z', '+00:00'))).total_seconds() >= seconds


def remove_payloads(root, *, seconds, now, all_files=False):
    """No-follow, hash/stable-inode unlink; skip young files and unsupported links."""
    removed = 0
    with directory(root) as descriptor:
        if descriptor is None:
            return 0
        for current, _, files, parent in os.fwalk('.', topdown=False, dir_fd=descriptor, follow_symlinks=False):
            for name in files:
                path = root / current / name
                info = os.stat(name, dir_fd=parent, follow_symlinks=False)
                if (not stat.S_ISREG(info.st_mode) or not (all_files or path.suffix.lower() in PAYLOAD_SUFFIXES)
                        or now.timestamp() - max(info.st_mtime, info.st_ctime) < seconds):
                    continue
                opened = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
                with os.fdopen(opened, 'rb') as stream:
                    if signature(os.fstat(stream.fileno())) != signature(info):
                        continue
                    checksum = stream_digest(stream)
                size = evict_file(path, {'bytes': info.st_size, 'sha256': checksum}, {}, '',
                                  expected_signature=signature(info))
                removed += size or 0
            prune_empty(root / current)
    return removed


class LocalRetention:
    def __init__(self, objects, index=None):
        self.objects, self.index = objects, index
        self.seconds = retention_seconds(objects.config)

    def maintain(self):
        if self.seconds is None:
            return
        now = datetime.now(timezone.utc)
        assets = AssetStore(self.objects.config_file, self.objects.root.parent)
        for metadata in assets.root.glob('*/*/asset.json'):
            project, asset = metadata.parent.parent.name, metadata.parent.name
            if IDENTITY.fullmatch(project) and ASSET_ID.fullmatch(asset):
                self.expire_asset(assets, project, asset, now)
        if self.index is not None:
            for row in self.index.list_runs():
                for attempt in row.attempts:
                    if attempt.state in TERMINAL:
                        self.expire_attempt(row, attempt.attempt_id, now)

    def expire_asset(self, assets, project, asset, now):
        with assets.record(project, asset) as path:
            metadata = path / 'asset.json'
            with directory(path) as descriptor:
                if descriptor is None or not metadata.is_file():
                    return
                if metadata.is_symlink():
                    raise ValueError('local asset metadata is a symlink')
                value = json.loads(metadata.read_text())
                if (value['project'] != project or value['asset_id'] != asset
                        or asset != 'asset.' + value['sha256']):
                    raise ValueError('local asset expiry identity differs')
                if value.get('remote_storage'):
                    return
                marker = path / 'local-retention.json'
                if marker.is_symlink():
                    raise ValueError('local retention marker is a symlink')
                previous = json.loads(marker.read_text()) if marker.exists() else {}
                if previous.get('status') == 'EXPIRED':
                    remove_payloads(path / 'tree', seconds=self.seconds, now=now, all_files=True)
                    return
                if not expired_timestamp(value['created_at'], self.seconds, now):
                    return
                key = '/'.join(['data-assets', project, asset + '.tar'])
                record = {'status': 'EXPIRING', 'reason': 'LOCAL_RETENTION_EXPIRED', 'expired_at': utc_now(),
                          'sha256': value['sha256'], 'object_key': key}
                with writable(descriptor):
                    atomic_json(marker, record)
                self.objects.client().delete_object(Bucket=self.objects.config['bucket'], Key=key)
                remove_payloads(path / 'tree', seconds=self.seconds, now=now, all_files=True)
                with writable(descriptor):
                    atomic_json(marker, {**record, 'status': 'EXPIRED'})

    def expire_attempt(self, row, attempt, now):
        project, run = row.project, row.run_id
        run_root = Path(row.run_dir)
        if (not run_root.is_relative_to(self.objects.root.parent.parent / 'runs')
                or run_root != run_root.resolve()):
            return
        with self.objects.record(project, run, attempt) as (path, value):
            if not value or value['run_dir'] != row.run_dir:
                return
            collection = self.objects.root.parent / 'result-collections' / project / run / (attempt + '.json')
            if collection.is_file() and json.loads(collection.read_text()).get('status') in {
                    'QUEUED', 'COLLECTING', 'SUBMITTING', 'WAITING', 'RECONCILE_REQUIRED'}:
                return
            receipt = value.get('receipt')
            if receipt and value.get('retention', {}).get('status') not in {'RELEASED', 'RELEASING', 'EXPIRED'}:
                if expired_timestamp(receipt['received_at'], self.seconds, now):
                    receipts = [receipt, *value.get('receipt_history', [])]
                    for item in receipts:
                        if (not SHA256.fullmatch(item['sha256']) or
                                item['object_key'] != '/'.join([project, run, attempt, item['sha256'] + '.tar'])):
                            raise ValueError('local archive expiry identity differs')
                    if value.get('retention') and value['retention'].get('status') != 'EXPIRING':
                        value.setdefault('retention_history', []).append(value['retention'])
                    value['retention'] = {'status': 'EXPIRING', 'reason': 'LOCAL_RETENTION_EXPIRED',
                                          'sha256': receipt['sha256'], 'expired_at': utc_now()}
                    atomic_json(path, value)
                    for item in receipts:
                        self.objects.client().delete_object(Bucket=self.objects.config['bucket'], Key=item['object_key'])
                    value['retention']['status'] = 'EXPIRED'
                    atomic_json(path, value)
            parent = Path(value['run_dir']) / 'attempts' / attempt
            removed = 0
            collected = parent / 'collected_run' / 'attempts' / attempt
            for root in [parent / 'uploaded_outputs', parent / 'outputs', parent / 'recovered_outputs',
                         collected / 'outputs', collected / 'state']:
                removed += remove_payloads(root, seconds=self.seconds, now=now)
            if removed:
                value['local_payload_cleanup'] = {'bytes_removed': value.get('local_payload_cleanup', {}).get('bytes_removed', 0) + removed,
                                                   'reason': 'LOCAL_RETENTION_EXPIRED', 'updated_at': utc_now()}
                atomic_json(path, value)
