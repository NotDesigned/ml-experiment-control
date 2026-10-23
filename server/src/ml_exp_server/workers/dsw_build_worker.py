"""Small stdlib Docker Engine client on one owned zero-GPU DSW instance."""
from __future__ import annotations

import base64
import hashlib
import http.client
import json
from pathlib import Path
import re
import shutil
import socket
import sys
import time
from urllib.parse import quote, urlencode


class Engine(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect('/var/docker/proxy/docker-proxy.sock')


def request(method, path, body=None, headers=None, *, stream=False):
    connection = Engine('localhost', timeout=600)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError('DSW Docker API HTTP ' + str(response.status))
        if not stream:
            raw = response.read(65537)
            if len(raw) > 65536:
                raise ValueError('DSW Docker metadata exceeds its bound')
            return json.loads(raw)
        size = 0
        while line := response.readline(65537):
            size += len(line)
            if len(line) > 65536 or size > 2 * 1024**2:
                raise ValueError('DSW build log exceeds its bound')
            event = json.loads(line)
            print(json.dumps(event), flush=True)
            if event.get('error') or event.get('errorDetail'):
                raise ValueError('DSW Docker build or publication failed')
    finally:
        connection.close()


def encoded(value):
    return base64.urlsafe_b64encode(json.dumps(value).encode()).decode()


def pull_bases(images, credentials):
    """Retry only immutable image downloads, before executing any Dockerfile RUN."""
    if not isinstance(images, list) or len(images) > 64:
        raise ValueError('invalid pinned base image list')
    for image in images:
        if not isinstance(image, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}', image):
            raise ValueError('DSW base image must be pinned')
        host = image.split('/', 1)[0]
        attempt = 0
        while True:
            print(json.dumps({'phase': 'PULLING_BASE_IMAGE', 'attempt': attempt + 1, 'image': image}), flush=True)
            try:
                request('POST', '/images/create?' + urlencode({'fromImage': image, 'platform': 'linux/amd64'}),
                        headers={'X-Registry-Auth': encoded(credentials.get(host, {})), 'Content-Length': '0'}, stream=True)
                break
            except (OSError, ValueError, http.client.HTTPException):
                attempt += 1
                if attempt >= 3:
                    raise
                time.sleep(2 ** (attempt - 1))


def run(root, spec):
    if not re.fullmatch(r'/tmp/ml-expd-build-[0-9a-f]{32}', str(root)):
        raise ValueError('invalid owned DSW build directory')
    archive = root / 'context.tar'
    with archive.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != spec['context_sha256']:
            raise ValueError('DSW source context checksum differs')
    tag = spec['tag']
    if not re.fullmatch(r'[A-Za-z0-9.-]+/[a-z0-9._/-]+:bundle-[0-9a-f]{64}', tag):
        raise ValueError('invalid DSW build publication tag')
    host = tag.split('/', 1)[0]
    auths = json.loads((root / 'auth/config.json').read_text())['auths']
    credentials = {}
    for registry, item in auths.items():
        user, password = base64.b64decode(item['auth'], validate=True).decode().split(':', 1)
        credentials[registry] = {'username': user, 'password': password, 'serveraddress': registry}
    pull_bases(spec.get('base_images', []), credentials)
    with archive.open('rb') as stream:
        request('POST', '/build?' + urlencode({'t': tag, 'rm': '1', 'version': '1', 'platform': 'linux/amd64'}),
                body=stream, headers={'Content-Type': 'application/x-tar', 'Content-Length': str(archive.stat().st_size),
                                      'X-Registry-Config': encoded(credentials)}, stream=True)
    repository, name = tag.rsplit(':', 1)
    request('POST', '/images/' + quote(repository, safe='') + '/push?' + urlencode({'tag': name}),
            headers={'X-Registry-Auth': encoded(credentials[host]), 'Content-Length': '0'}, stream=True)
    value = request('GET', '/images/' + quote(tag, safe='') + '/json')
    matches = [d for d in value.get('RepoDigests', [])
               if re.fullmatch(re.escape(repository) + r'@sha256:[0-9a-f]{64}', d)]
    if len(matches) != 1:
        raise ValueError('DSW registry digest is unavailable or ambiguous')
    return {'image': matches[0], 'context_sha256': spec['context_sha256']}


def main(argv=None):
    root = Path((argv or sys.argv[1:])[0])
    try:
        receipt = run(root, json.loads((root / 'request.json').read_text()))
        value = {'status': 'SUCCEEDED', 'receipt': receipt}
    except Exception as error:
        if re.fullmatch(r'/tmp/ml-expd-build-[0-9a-f]{32}', str(root)):
            (root / 'status.tmp').write_text(json.dumps({'status': 'FAILED', 'error_class': type(error).__name__}))
            (root / 'status.tmp').replace(root / 'status.json')
        raise
    else:
        (root / 'status.tmp').write_text(json.dumps(value))
        (root / 'status.tmp').replace(root / 'status.json')
    finally:
        if re.fullmatch(r'/tmp/ml-expd-build-[0-9a-f]{32}', str(root)):
            shutil.rmtree(root / 'auth', ignore_errors=True)


if __name__ == '__main__':
    main()
