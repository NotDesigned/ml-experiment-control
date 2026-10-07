"""S3 storage and write-only, exact-Attempt capabilities for worker transfers."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import hmac
import json
from pathlib import Path
import re
import secrets
import tempfile
from types import SimpleNamespace
from urllib.parse import urlsplit

from .source_imports import IDENTITY, unpack_source, seal_tree, remove_staging
from .storage import atomic_json, utc_now
from .archive_limits import byte_limit, exceeds, wire_limit
from .worker_launcher import LAUNCH_PATH, encode, endpoint

ATTEMPT = re.compile(r'^attempt-[0-9]{3,}$')
TRANSFER_PATH = re.compile(r'^/api/artifact-transfers/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/attempt-[0-9]{3,}$')


def stream_digest(stream):
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


class ArtifactStore:
    def __init__(self, config_file: Path, registry_root: Path):
        self.config = json.loads(config_file.read_text())
        self.root = registry_root / 'artifact-transfers'
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.limit = byte_limit(self.config.get('max_archive_bytes'))

    @contextmanager
    def record(self, project, run, attempt):
        if not IDENTITY.fullmatch(project) or not IDENTITY.fullmatch(run) or not ATTEMPT.fullmatch(attempt):
            raise ValueError('invalid transfer identity')
        directory = self.root / project / run
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (directory / (attempt + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = directory / (attempt + '.json')
            yield path, json.loads(path.read_text()) if path.exists() else None

    def issue(self, project, run, attempt, run_dir: Path, outputs: list[str], *, checkpoint_upload=False, checkpoint_state=None, record_stream=False):
        with self.record(project, run, attempt) as (path, value):
            if value is None:
                value = {'project': project, 'run_id': run, 'attempt_id': attempt,
                         'run_dir': str(run_dir), 'outputs': outputs, 'token': secrets.token_urlsafe(32),
                         'expires_at': (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
                         'receipt': None}
                atomic_json(path, value)
            if checkpoint_upload:
                if not value.get('checkpoint_upload'):
                    value['checkpoint_upload'] = True
                    atomic_json(path, value)
            if record_stream:
                value['record_stream'] = True
                atomic_json(path, value)
            if value['run_dir'] != str(run_dir) or value['outputs'] != outputs:
                raise ValueError('transfer identity is already bound')
            if checkpoint_state is not None:
                if value.get('checkpoint_state', checkpoint_state) != checkpoint_state:
                    raise ValueError('persistent checkpoint transfer identity is already bound')
                value['checkpoint_state'] = checkpoint_state
                atomic_json(path, value)
            url = self.config['public_transfer_base'].rstrip('/') + '/' + '/'.join([project, run, attempt])
            return url, value['token'], wire_limit(self.limit)

    def authorize(self, project, run, attempt, token):
        with self.record(project, run, attempt) as (_, value):
            if not value or not hmac.compare_digest(value['token'], token):
                raise ValueError('invalid transfer capability')
            if datetime.fromisoformat(value['expires_at']) < datetime.now(timezone.utc):
                raise ValueError('expired transfer capability')
            return value

    def seal_launch(self, project, run, attempt, document):
        body = encode(document)
        if document['identity'] != {'project': project, 'run_id': run, 'attempt_id': attempt}:
            raise ValueError('launch manifest identity differs')
        digest = hashlib.sha256(body).hexdigest()
        prefix = self.config['public_transfer_base'].rstrip('/').rsplit('/', 1)[0]
        url = prefix + '/launch-transfers/' + '/'.join([project, run, attempt, digest])
        endpoint(url)
        with self.record(project, run, attempt) as (path, value):
            if not value:
                raise ValueError('Attempt transfer capability has not been issued')
            if value.get('launch_manifest', document) != document:
                raise ValueError('Attempt launch manifest is already sealed')
            value['launch_manifest'] = document
            atomic_json(path, value)
        return url

    def launch(self, project, run, attempt, digest, token):
        value = self.authorize(project, run, attempt, token)
        if 'launch_manifest' not in value:
            raise ValueError('launch manifest is not available')
        body = encode(value['launch_manifest'])
        if not hmac.compare_digest(hashlib.sha256(body).hexdigest(), digest):
            raise ValueError('launch manifest digest differs')
        return body

    def client(self, *, public=False):
        import boto3
        from botocore.config import Config
        return boto3.client('s3', endpoint_url=self.config['public_endpoint'] if public else self.config['endpoint'], region_name=self.config.get('region', 'garage'),
                            aws_access_key_id=self.config['access_key'], aws_secret_access_key=self.config['secret_key'],
                            config=Config(signature_version='s3v4', connect_timeout=10, read_timeout=120,
                                          retries={'max_attempts': 3}, s3={'addressing_style': 'path'}))

    def download(self, key, digest, size, *, files=None):
        endpoint = urlsplit(self.config.get('public_endpoint', ''))
        if (endpoint.scheme != 'https' or not endpoint.hostname or endpoint.username or endpoint.password
                or endpoint.path not in {'', '/'} or endpoint.query or endpoint.fragment):
            raise ValueError('direct downloads require a public HTTPS object storage endpoint')
        seconds = int(self.config.get('download_url_seconds', 300))
        if not 30 <= seconds <= 3600:
            raise ValueError('invalid download URL lifetime')
        url = self.client(public=True).generate_presigned_url('get_object',
            Params={'Bucket': self.config['bucket'], 'Key': key}, ExpiresIn=seconds)
        return {'transport': 's3-presigned-get', 'url': url, 'sha256': digest, 'bytes': size,
                'expires_at': (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat(),
                'files': files}

    def artifact_download(self, project, run, attempt):
        with self.record(project, run, attempt) as (_, value):
            if not value or not value.get('receipt'):
                from .application_errors import ApplicationError
                raise ApplicationError('Attempt artifacts have not been uploaded', status_code=404, code='ARTIFACT_UNAVAILABLE')
            receipt = value['receipt']
        return self.download(receipt['object_key'], receipt['sha256'], receipt['bytes'], files=receipt['files'])

    def restore_cache(self, project, run, attempt):
        with self.record(project, run, attempt) as (_, value):
            if not value or not value.get('receipt'):
                return
            parent = Path(value['run_dir']) / 'attempts' / attempt
            destination = parent / 'uploaded_outputs'
            if destination.exists():
                return
            receipt = value['receipt']
            temporary = Path(tempfile.mkdtemp(prefix='.restore-', dir=parent))
            try:
                with tempfile.TemporaryFile() as stream:
                    self.client().download_fileobj(self.config['bucket'], receipt['object_key'], stream,
                                                   Config=self._transfer_config())
                    if stream.tell() != receipt['bytes']:
                        raise ValueError('stored artifact size mismatch')
                    stream.seek(0)
                    if stream_digest(stream) != receipt['sha256']:
                        raise ValueError('stored artifact digest mismatch')
                    stream.seek(0)
                    unpack_source(stream, temporary, SimpleNamespace(max_source_bytes=self.limit, max_source_files=20000), allow_empty=True)
                seal_tree(temporary)
                temporary.chmod(0o700)
                temporary.rename(destination)
                destination.chmod(0o500)
            finally:
                remove_staging(temporary)

    def receive(self, project, run, attempt, token, stream, size):
        if size <= 0 or exceeds(size, self.limit):
            raise ValueError('invalid artifact size')
        with self.record(project, run, attempt) as (path, value):
            if not value or not hmac.compare_digest(value['token'], token) or datetime.fromisoformat(value['expires_at']) < datetime.now(timezone.utc):
                raise ValueError('invalid transfer capability')
            run_dir = Path(value['run_dir'])
            manifest = json.loads((run_dir / 'manifest.json').read_text()) if (run_dir / 'manifest.json').exists() else None
            if manifest is None:
                import yaml
                manifest = yaml.safe_load((run_dir / 'manifest.yaml').read_text())
            import yaml
            identity = yaml.safe_load((run_dir / 'attempts' / attempt / 'attempt.yaml').read_text())
            if any(manifest.get(k) != v or identity.get(k) != v for k, v in [('project', project), ('run_id', run)]):
                raise ValueError('Run/Attempt transfer identity mismatch')
            digest = stream_digest(stream)
            stream.seek(0)
            if value['receipt']:
                if value['receipt']['sha256'] != digest:
                    raise ValueError('Attempt artifacts are already sealed with another digest')
                return value['receipt']
            parent = run_dir / 'attempts' / attempt
            temporary = Path(tempfile.mkdtemp(prefix='.upload-', dir=parent))
            try:
                unpack_source(stream, temporary, SimpleNamespace(max_source_bytes=self.limit, max_source_files=20000), allow_empty=True)
                # A project may only upload its declared output paths.
                import fnmatch
                files = []
                for file in temporary.rglob('*'):
                    if file.is_file():
                        relative = file.relative_to(temporary).as_posix()
                        if not any(fnmatch.fnmatch(relative, p) or (p.startswith('**/') and fnmatch.fnmatch(relative, p[3:])) for p in value['outputs']):
                            raise ValueError('artifact is outside declared outputs')
                        with file.open('rb') as body:
                            file_sha = stream_digest(body)
                        files.append({'path': relative, 'bytes': file.stat().st_size, 'sha256': file_sha})
                stream.seek(0)
                key = '/'.join([project, run, attempt, digest + '.tar'])
                self.client().upload_fileobj(stream, self.config['bucket'], key,
                                             ExtraArgs={'ContentType': 'application/x-tar', 'Metadata': {'sha256': digest}},
                                             Config=self._transfer_config())
                receipt = {'sha256': digest, 'bytes': size, 'files': sorted(files, key=lambda f: f['path']),
                           'object_key': key, 'received_at': utc_now()}
                seal_tree(temporary)
                temporary.chmod(0o700)
                destination = parent / 'uploaded_outputs'
                if destination.exists():
                    # Recover the S3-write/local-rename/receipt-write interruption window.
                    remove_staging(destination)
                temporary.rename(destination)
                destination.chmod(0o500)
                value['receipt'] = receipt
                atomic_json(path, value)
                return receipt
            finally:
                remove_staging(temporary)

    @staticmethod
    def _transfer_config():
        from boto3.s3.transfer import TransferConfig
        return TransferConfig(multipart_threshold=16 * 1024 ** 2, multipart_chunksize=16 * 1024 ** 2, max_concurrency=2)
