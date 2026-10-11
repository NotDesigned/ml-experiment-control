"""Remove only receipt-verified files from disposable delivery caches."""
from __future__ import annotations

from contextlib import contextmanager
import errno
import os
from pathlib import Path
import stat

from .artifact_store import stream_digest
from .artifacts import open_directory, safe_parts

_ABSENT = {errno.ENOENT, errno.ENOTDIR, errno.ELOOP}


@contextmanager
def directory(path):
    try:
        descriptor = open_directory(path)
    except OSError as error:
        if error.errno not in _ABSENT:
            raise
        yield None
        return
    try:
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def writable(descriptor):
    mode = stat.S_IMODE(os.fstat(descriptor).st_mode)
    try:
        os.fchmod(descriptor, mode | stat.S_IWUSR)
        yield
    finally:
        os.fchmod(descriptor, mode)


def signature(metadata):
    return [metadata.st_dev, metadata.st_ino, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns]


def evict_file(path, item, retained, key):
    with directory(path.parent) as parent:
        if parent is None:
            return None
        try:
            descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        except OSError as error:
            if error.errno not in _ABSENT:
                raise
            return None
        with os.fdopen(descriptor, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size != item['bytes']:
                return None
            # Do not rehash unchanged mismatches on every collector cycle.
            if retained.get(key) == signature(before):
                return None
            digest = stream_digest(stream)
            after = os.fstat(stream.fileno())
            current = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if signature(before) != signature(after) or signature(after) != signature(current):
                return None
            if digest != item['sha256']:
                retained[key] = signature(after)
                return None
            with writable(parent):
                os.unlink(path.name, dir_fd=parent)
            retained.pop(key, None)
            return before.st_size


def prune_empty(path):
    with directory(path.parent) as parent:
        if parent is None:
            return
        try:
            with writable(parent):
                os.rmdir(path.name, dir_fd=parent)
        except OSError as error:
            if error.errno not in _ABSENT | {errno.ENOTEMPTY, errno.EEXIST}:
                raise


def evict_caches(value):
    attempt = Path(value['run_dir']) / 'attempts' / value['attempt_id']
    caches = [attempt / 'uploaded_outputs', attempt / 'recovered_outputs',
              attempt / 'collected_run' / 'attempts' / value['attempt_id'] / 'outputs']
    previous = value.get('cache_cleanup', {})
    report = {**previous, 'files_removed': previous.get('files_removed', 0),
              'bytes_removed': previous.get('bytes_removed', 0),
              'retained': dict(previous.get('retained', {}))}
    # Legacy receipts without file hashes cannot authorize local file deletion.
    for cache in caches:
        empty = {cache}
        for item in value['receipt']['files']:
            if not item.get('sha256'):
                continue
            try:
                parts = safe_parts(item['path'])
            except ValueError:
                # Preserve hidden/unsupported paths without blocking other releases.
                continue
            path = cache.joinpath(*parts)
            key = path.relative_to(attempt).as_posix()
            size = evict_file(path, item, report['retained'], key)
            if size is not None:
                report['files_removed'] += 1
                report['bytes_removed'] += size
                empty.update(path.parents[:len(parts) - 1])
        for path in sorted(empty, key=lambda p: len(p.parts), reverse=True):
            prune_empty(path)
    report.pop('diagnostic', None)
    return report
