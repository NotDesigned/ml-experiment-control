"""Private OSS copies of new, sealed result archives; no bucket-wide synchronization."""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import io
import json
from pathlib import Path
import re
import time

from ..application_errors import ApplicationError
from ..execution_queue import QueueUnavailable
from ..storage import atomic_json, utc_now
from .artifact_store import artifact_released

PART_BYTES = 8 * 1024**2


def target(config, identity, receipt):
    """Freeze a destination for this receipt, without reading cloud credentials."""
    settings = config.get('oss_archive')
    if settings is None:
        return None
    if (not isinstance(settings, dict) or not isinstance(settings.get('bucket'), str)
            or not re.fullmatch(r'[a-z0-9][a-z0-9-]{1,61}[a-z0-9]', settings['bucket'])
            or not isinstance(settings.get('region'), str)
            or not re.fullmatch(r'[a-z][a-z0-9-]{1,63}', settings['region'])
            or not isinstance(settings.get('user'), str)
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', settings['user'])
            or not isinstance(settings.get('credentials_file'), str)
            or not Path(settings['credentials_file']).is_absolute()):
        raise ValueError('invalid personal OSS archive configuration')
    key = '/'.join(['users', settings['user'], 'results', 'ml-expd', *identity, receipt['sha256'] + '.tar'])
    return {'status': 'PENDING', 'bucket': settings['bucket'], 'region': settings['region'],
            'user': settings['user'], 'object_key': key, 'sha256': receipt['sha256'],
            'bytes': receipt['bytes'], 'uri': 'oss://' + settings['bucket'] + '/' + key,
            'attempts': 0, 'bytes_copied': 0, 'updated_at': utc_now()}


def pending(value):
    archive = value.get('oss_archive')
    return bool(archive and archive['status'] != 'VERIFIED')


class DigestReader(io.RawIOBase):
    """A non-seekable, bounded source; SDK multipart handles transmission/retries."""
    def __init__(self, body, progress, expected_bytes, expected_sha256):
        self.body, self.progress = body, progress
        self.digest, self.size = hashlib.sha256(), 0
        self.expected_bytes, self.expected_sha256 = expected_bytes, expected_sha256

    def readable(self):
        return True

    def read(self, size=-1):
        if size == 0:
            return b''
        data = self.body.read(min(size, PART_BYTES) if size >= 0 else PART_BYTES)
        self.digest.update(data)
        self.size += len(data)
        if (self.size > self.expected_bytes or (not data or self.size == self.expected_bytes)
                and (self.size, self.digest.hexdigest()) != (self.expected_bytes, self.expected_sha256)):
            raise ValueError('source result archive differs from its receipt')
        self.progress(self.size)
        return data


def client(settings):
    import alibabacloud_oss_v2 as oss
    credentials = json.loads(Path(settings['credentials_file']).read_text())
    if (credentials.get('workspace_user') != settings['user']
            or credentials.get('region') != settings['region']):
        raise ValueError('archive credentials do not belong to the configured user/region')
    return oss.Client(oss.Config(region=settings['region'],
        endpoint='https://oss-' + settings['region'] + '.aliyuncs.com', signature_version='v1',
        credentials_provider=oss.credentials.StaticCredentialsProvider(
            credentials['access_key_id'], credentials['access_key_secret'], credentials.get('security_token')),
        retry_max_attempts=1, connect_timeout=5, readwrite_timeout=30, insecure_skip_verify=False))


def verify(destination, archive):
    """Verify actual object bytes; ETag/metadata alone is not a SHA256 proof."""
    import alibabacloud_oss_v2 as oss
    try:
        response = destination.get_object(oss.GetObjectRequest(bucket=archive['bucket'], key=archive['object_key']))
    except oss.exceptions.OperationError as error:
        inner = error.unwrap()
        if isinstance(inner, oss.exceptions.ServiceError) and inner.code == 'NoSuchKey':
            return False
        raise
    digest, size = hashlib.sha256(), 0
    with closing(response.body) as body:
        for data in body.iter_bytes(block_size=1024**2):
            digest.update(data)
            size += len(data)
            if size > archive['bytes']:
                raise ValueError('OSS archive object differs from its immutable identity')
    if (size, digest.hexdigest()) != (archive['bytes'], archive['sha256']):
        raise ValueError('OSS archive object differs from its immutable identity')
    return True


def failure(error):
    """Expose only typed provider evidence, never exception strings/URLs/headers."""
    import alibabacloud_oss_v2 as oss
    details = {'error_class': type(error).__name__}
    for _ in range(6):
        if isinstance(error, oss.exceptions.ServiceError):
            code = error.code
            details['provider_code'] = code if re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', code) else 'OSS_REQUEST_FAILED'
            details['http_status'] = error.status_code
            break
        unwrap = getattr(error, 'unwrap', None)
        if not callable(unwrap):
            break
        error = unwrap()
    return details


class OSSArchiveService:
    def __init__(self, store):
        self.store, self.future = store, None

    def automatic(self, submit):
        if self.future is not None and not self.future.done():
            return
        for path in self.store.root.glob('*/*/attempt-*.json'):
            value = json.loads(path.read_text())
            archive = value.get('oss_archive')
            if (not archive or archive['status'] in {'VERIFIED', 'SOURCE_EXPIRED'}
                    or archive.get('retry_after', 0) > time.time()):
                continue
            identity = (path.parent.parent.name, path.parent.name, path.stem)
            try:
                self.future = submit(self.finish, identity)
            except QueueUnavailable:
                return
            return

    def update(self, identity, archive, **fields):
        with self.store.record(*identity) as (path, value):
            current = value.get('oss_archive') if value else None
            if current is None or any(current.get(k) != archive[k]
                                      for k in ('sha256', 'object_key', 'bucket', 'region', 'user')):
                raise ValueError('archive target identity advanced')
            current.update(fields, updated_at=utc_now())
            atomic_json(path, value)

    def ticket(self, project, run, attempt):
        import alibabacloud_oss_v2 as oss
        identity = (project, run, attempt)
        with self.store.record(*identity) as (_, value):
            archive = value.get('oss_archive') if value else None
            if not archive or archive['status'] != 'VERIFIED':
                raise ApplicationError('OSS archive has not been verified', status_code=404, code='OSS_ARCHIVE_UNAVAILABLE')
            receipt = value['receipt']
            expected = target(self.store.config, identity, receipt)
            if any(archive[k] != expected[k] for k in ('object_key', 'bucket', 'region', 'user', 'sha256', 'bytes')):
                raise ValueError('OSS archive identity differs')
        signed = client(self.store.config['oss_archive']).presign(
            oss.GetObjectRequest(bucket=archive['bucket'], key=archive['object_key']), expires=timedelta(minutes=5))
        return {'transport': 'oss-presigned-get', 'url': signed.url, 'sha256': archive['sha256'],
                'bytes': archive['bytes'], 'files': receipt['files'], 'acknowledgement_required': False,
                'expires_at': (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()}

    def finish(self, identity):
        lock = self.store.root.joinpath(*identity[:2], identity[2] + '.oss.lock')
        with lock.open('a') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            archive, phase = None, 'VALIDATE'
            try:
                with self.store.record(*identity) as (_, value):
                    if not value or not value.get('oss_archive'):
                        return
                    archive = dict(value['oss_archive'])
                    if archive['status'] == 'VERIFIED':
                        return
                    released = artifact_released(value)
                    receipt = value['receipt']
                    expected = target(self.store.config, identity, receipt)
                    if any(archive[k] != expected[k] for k in ('sha256', 'object_key', 'bucket', 'region', 'user', 'bytes')):
                        raise ValueError('configured OSS archive target differs from frozen receipt')
                self.update(identity, archive, status='COPYING', attempts=archive['attempts'] + 1, bytes_copied=0, diagnostic=None)
                destination = client(self.store.config['oss_archive'])
                phase = 'VERIFY_EXISTING'
                if not verify(destination, archive):
                    if released:
                        self.expired(identity, archive)
                        return
                    phase = 'COPY'
                    self.copy(identity, archive, receipt, destination)
                    phase = 'VERIFY_PUBLICATION'
                    if not verify(destination, archive):
                        raise ValueError('OSS archive publication is unavailable')
                self.update(identity, archive, status='VERIFIED', bytes_copied=archive['bytes'],
                            verified_at=utc_now(), diagnostic=None)
            except ApplicationError as error:
                if error.code != 'RUN_DELETED':
                    raise
            except Exception as error:
                if archive is not None:
                    self.update(identity, archive, status='RETRY_REQUIRED', diagnostic={'phase': phase, **failure(error)},
                                retry_after=time.time() + 60)

    def expired(self, identity, archive):
        # Called outside the transfer record lock; the 24h local expiry remains explicit.
        self.update(identity, archive, status='SOURCE_EXPIRED', diagnostic='LOCAL_ARCHIVE_UNAVAILABLE')

    def copy(self, identity, archive, receipt, destination):
        import alibabacloud_oss_v2 as oss
        expected = '/'.join([*identity, archive['sha256'] + '.tar'])
        if receipt['object_key'] != expected:
            raise ValueError('archive source identity differs')
        previous = [0]
        def progress(size):
            if size - previous[0] >= PART_BYTES:
                self.update(identity, archive, bytes_read=size)
                previous[0] = size
        source = self.store.client(verified_stream=True).get_object(Bucket=self.store.config['bucket'], Key=expected)['Body']
        with closing(source) as body:
            reader = DigestReader(body, progress, archive['bytes'], archive['sha256'])
            oss.Uploader(destination, part_size=PART_BYTES, parallel_num=1, leave_parts_on_error=False).upload_from(
                oss.PutObjectRequest(bucket=archive['bucket'], key=archive['object_key'],
                                     content_type='application/x-tar', forbid_overwrite=True,
                                     metadata={'sha256': archive['sha256']}), reader)
            if (reader.size, reader.digest.hexdigest()) != (archive['bytes'], archive['sha256']):
                raise ValueError('source result archive differs from its receipt')
