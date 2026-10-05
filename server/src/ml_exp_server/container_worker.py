"""Stdlib-only image launcher: run frozen argv, then upload this Attempt's outputs."""
from __future__ import annotations

import fnmatch
import hashlib
import http.client
import json
import os
import re
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.parse import urlsplit

MULTIPART_UPLOAD = os.environ.pop('ML_EXPD_MULTIPART_UPLOAD', '0') == '1'


def archive_outputs(root: Path, stream, limit: int, patterns: list[str]) -> int:
    total = 0
    count = 0
    # Descriptor-relative traversal prevents project-created symlinks escaping outputs.
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for directory, _, names, fd in os.fwalk(root, follow_symlinks=False):
            for name in sorted(names):
                path = Path(directory, name).relative_to(root).as_posix()
                if any(part.startswith('.') for part in Path(path).parts):
                    continue
                if not any(fnmatch.fnmatch(path, p) or (p.startswith('**/') and fnmatch.fnmatch(path, p[3:])) for p in patterns):
                    continue
                if not stat.S_ISREG(os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode):
                    continue
                opened = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
                try:
                    info = os.fstat(opened)
                    if not stat.S_ISREG(info.st_mode):
                        continue
                    count += 1
                    total += info.st_size
                    if count > 20000 or total > limit:
                        raise ValueError('artifact limit exceeded')
                    member = tarfile.TarInfo(path)
                    member.size, member.mode = info.st_size, 0o400
                    with os.fdopen(os.dup(opened), 'rb') as source:
                        archive.addfile(member, source)
                finally:
                    os.close(opened)
    return count


def upload(url: str, token: str, stream, length: int):
    if MULTIPART_UPLOAD:
        return upload_parts(url, token, stream, length)
    return legacy_upload(url, token, stream, length)


def upload_request(target, method, path, token, data=b''):
    connection = http.client.HTTPSConnection(target.hostname, target.port or 443, timeout=300)
    try:
        connection.request(method, path, body=data, headers={'Authorization': 'Bearer ' + token,
            'Content-Type': 'application/json' if method == 'POST' else 'application/octet-stream'})
        response = connection.getresponse()
        body = response.read(4 * 1024 ** 2)
        if response.status >= 500 or response.status == 429:
            raise OSError('multipart archive transfer temporarily unavailable')
        if response.status != 200:
            raise ValueError('multipart archive transfer rejected (HTTP ' + str(response.status) + ')')
        return json.loads(body)
    finally:
        connection.close()


def upload_parts(url, token, stream, length):
    target = urlsplit(url)
    if target.scheme != 'https' or target.username or target.password or target.query or target.fragment:
        raise ValueError('artifact transfer requires a fixed HTTPS endpoint')
    marker = '/snapshot-transfers/' if '/snapshot-transfers/' in target.path else '/artifact-transfers/'
    kind = 'checkpoint' if marker == '/snapshot-transfers/' else 'artifacts'
    if marker not in target.path:
        raise ValueError('invalid worker upload endpoint')
    endpoint = target.path.replace(marker, '/attempt-uploads/', 1) + '/' + kind
    digest = hashlib.sha256()
    stream.seek(0)
    while data := stream.read(1024 ** 2):
        digest.update(data)
    identity = json.dumps({'sha256': digest.hexdigest(), 'bytes': length}).encode()
    # Reissuing create returns the same durable session. On a lost response or
    # transient connection failure, only missing parts are resent.
    attempt = 0
    while True:
        try:
            value = upload_request(target, 'POST', endpoint, token, identity)
            if (not 1024 <= value['part_bytes'] <= 64 * 1024 ** 2 or value['sha256'] != digest.hexdigest() or value['bytes'] != length
                    or value['part_count'] != (length + value['part_bytes'] - 1) // value['part_bytes']
                    or not re.fullmatch(r'upload\.[0-9a-f]{64}', value['upload_id'])):
                raise ValueError('invalid multipart upload receipt')
            if value['status'] == 'COMPLETED':
                return
            root = endpoint + '/' + value['upload_id']
            stream.seek(0)
            for number in range(value['part_count']):
                data = stream.read(value['part_bytes'])
                part_sha = hashlib.sha256(data).hexdigest()
                previous = value['parts'].get(str(number))
                if previous:
                    if previous != {'sha256': part_sha, 'bytes': len(data)}:
                        raise ValueError('sealed upload part differs')
                    continue
                upload_request(target, 'PUT', root + '/parts/' + str(number) + '?sha256=' + part_sha, token, data)
            upload_request(target, 'POST', root + '/complete', token)
            return
        except (OSError, http.client.HTTPException):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
            attempt += 1


def legacy_upload(url: str, token: str, stream, length: int):
    target = urlsplit(url)
    if target.scheme != 'https' or target.username or target.password or target.query or target.fragment:
        raise ValueError('artifact transfer requires a fixed HTTPS endpoint')
    for attempt in range(3):
        connection = http.client.HTTPSConnection(target.hostname, target.port or 443, timeout=300)
        try:
            connection.putrequest('PUT', target.path)
            connection.putheader('Authorization', 'Bearer ' + token)
            connection.putheader('Content-Type', 'application/x-tar')
            connection.putheader('Content-Length', str(length))
            connection.endheaders()
            stream.seek(0)
            while chunk := stream.read(1024 * 1024):
                connection.send(chunk)
            response = connection.getresponse()
            response.read(4096)
            if response.status == 200:
                return
            if response.status < 500:
                raise ValueError('artifact transfer was rejected')
        except (OSError, http.client.HTTPException):
            if attempt == 2:
                raise
        finally:
            connection.close()
        time.sleep(2 ** attempt)
    raise ValueError('artifact transfer failed')


def main(argv=None):
    url = os.environ.pop('ML_EXPD_UPLOAD_URL')
    token = os.environ.pop('ML_EXPD_UPLOAD_TOKEN')
    limit = int(os.environ.pop('ML_EXPD_UPLOAD_LIMIT'))
    patterns = json.loads(os.environ.pop('ML_EXPD_OUTPUT_PATTERNS', '["**/*"]'))
    root = Path(os.environ['OUTPUT_DIR'])
    root.mkdir(parents=True, exist_ok=True)
    child = subprocess.Popen(argv or sys.argv[1:], start_new_session=True)
    def forward(signum, _frame):
        if child.poll() is None:
            os.killpg(child.pid, signum)
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    code = child.wait()
    try:
        with tempfile.TemporaryFile(dir=root.parent) as stream:
            archive_outputs(root, stream, limit - 1024 * 1024, patterns)
            length = stream.tell()
            if length > limit:
                raise ValueError('artifact archive limit exceeded')
            upload(url, token, stream, length)
        print('ML_EXPD_ARTIFACT_UPLOAD=COMPLETE', flush=True)
    except Exception:
        print('ML_EXPD_ARTIFACT_UPLOAD=FAILED', file=sys.stderr, flush=True)
        return code if code > 0 else 74
    return code if code >= 0 else 128 - code


if __name__ == '__main__':
    raise SystemExit(main())
