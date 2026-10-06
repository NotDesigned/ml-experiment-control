"""Only verified pinned blobs survive; no untrusted request populates the cache."""
import hashlib
import http.client
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import socket
import ssl
import threading
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tests.test_blob_https_adapter import adapter


def module(name):
    path = Path(__file__).parents[1] / ('examples/ssh-builder/' + name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize('method', ['GET', 'HEAD'])
def test_verified_cache_serves_without_upstream_or_signed_url_logs(adapter, tmp_path, monkeypatch, method, capsys):
    helper, port, requests, _ = adapter
    body = b'cached layer' * 300000
    digest = hashlib.sha256(body).hexdigest()
    (tmp_path / digest).write_bytes(body)
    monkeypatch.setattr(helper, 'CACHE', tmp_path)
    url = 'http://' + helper.HOST + '/registry/docker/registry/v2/blobs/sha256/' + digest[:2] + '/' + digest + '/data?secret=test-secret'
    connection = http.client.HTTPConnection('127.0.0.1', port)
    connection.request(method, url)
    response = connection.getresponse()
    assert response.status == 200 and response.getheader('Content-Length') == str(len(body))
    assert response.read() == (body if method == 'GET' else b'')
    connection.close()
    assert requests == [] and 'test-secret' not in capsys.readouterr().err


@pytest.mark.parametrize('kind', ['missing', 'symlink', 'fifo', 'prefix-mismatch', 'traversal', 'conditional', 'range'])
def test_unavailable_cache_or_http_preconditions_use_original_tls(adapter, tmp_path, monkeypatch, kind):
    helper, port, requests, body = adapter
    digest = hashlib.sha256(b'cached').hexdigest()
    file = tmp_path / digest
    path = '/registry/docker/registry/v2/blobs/sha256/' + digest[:2] + '/' + digest + '/data'
    headers = {}
    if kind == 'symlink':
        target = tmp_path / 'secret'; target.write_bytes(b'private'); file.symlink_to(target)
    elif kind == 'fifo':
        os.mkfifo(file)
    elif kind == 'prefix-mismatch':
        file.write_bytes(b'cached'); path = path.replace('/' + digest[:2] + '/', '/ff/')
    elif kind == 'traversal':
        path = path.replace(digest + '/data', '../secret/data')
    elif kind in {'conditional', 'range'}:
        file.write_bytes(b'cached'); headers = {'If-Match': 'expected'} if kind == 'conditional' else {'Range': 'bytes=0-2'}
    monkeypatch.setattr(helper, 'CACHE', tmp_path)
    connection = http.client.HTTPConnection('127.0.0.1', port)
    connection.request('GET', 'http://' + helper.HOST + path, headers=headers)
    response = connection.getresponse(); assert response.read() == body; connection.close()
    assert requests == [('GET', path, headers)]


@pytest.mark.parametrize('case', ['valid', 'wrong-hash', 'short', 'budget'])
def test_writer_verifies_before_atomic_visibility_and_removes_incomplete_files(tmp_path, case):
    tool = module('prewarm_ccr')
    data = b'model layer' * 100000
    digest = hashlib.sha256(data).hexdigest()
    expected = digest if case != 'wrong-hash' else 'a' * 64
    limit = 1 if case == 'budget' else 16 * 1024 ** 3
    payload = data[:20] if case == 'short' else data
    child = subprocess.run([sys.executable, '-c', tool.WRITER, expected, str(len(data)), str(limit)],
                           env={**os.environ, 'CCR_BLOB_CACHE': str(tmp_path)}, input=payload,
                           capture_output=True, timeout=10)
    if case == 'valid':
        assert child.returncode == 0
        assert json.loads(child.stdout) == {'sha256': digest, 'bytes': len(data)}
        assert (tmp_path / digest).read_bytes() == data
        assert (tmp_path / digest).stat().st_mode & 0o777 == 0o444
    else:
        assert child.returncode != 0 and not (tmp_path / expected).exists()
    assert not list(tmp_path.glob('.partial-*'))


@pytest.mark.parametrize('case', ['valid', 'corrupt', 'short-once'])
def test_remote_downloader_keeps_large_bytes_off_ssh_and_checks_whole_layer(tmp_path, case):
    tool = module('prewarm_ccr')
    body = b'layer-content' * 700000  # More than two ranges.
    digest = hashlib.sha256(body).hexdigest(); ranges = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            assert self.headers['Authorization'] == 'Bearer test-private-token'
            value = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            start, end = value['start'], value['end']; ranges.append((start, end))
            output = body[start:end + 1]
            if case == 'corrupt' and start == 0: output = b'x' + output[1:]
            if case == 'short-once' and start == 0 and ranges.count((start, end)) == 1:
                output = output[:-1]
            self.send_response(206)
            self.send_header('Content-Length', str(len(output)))
            self.send_header('Content-Range', f'bytes {start}-{end}/{len(body)}')
            self.end_headers(); self.wfile.write(output)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever); thread.start()
    try:
        control = {'digest': digest, 'size': len(body), 'limit': 16 * 1024 ** 3,
                   'url': 'signed-upstream-test-secret', 'token': 'test-private-token',
                   'relay': f'http://127.0.0.1:{server.server_port}/ml-expd-builder-relay'}
        payload = json.dumps(control).encode() + b'\n'
        assert len(payload) < 1024
        result = subprocess.run([sys.executable, '-c', tool.DOWNLOADER + tool.WRITER], input=payload,
                                env={**os.environ, 'CCR_BLOB_CACHE': str(tmp_path)}, capture_output=True, timeout=15)
        if case != 'corrupt':
            assert result.returncode == 0 and (tmp_path / digest).read_bytes() == body
            assert json.loads(result.stdout.splitlines()[-1]) == {'sha256': digest, 'bytes': len(body)}
        else:
            assert result.returncode != 0 and not (tmp_path / digest).exists()
        expected = [(start, min(len(body) - 1, start + 2097151)) for start in range(0, len(body), 2097152)]
        if case == 'short-once': expected.append((0, 2097151))
        assert sorted(ranges) == sorted(expected)
        assert b'test-private-token' not in result.stdout and b'test-secret' not in result.stdout
        assert not list(tmp_path.glob('.partial-*'))
    finally:
        server.shutdown(); thread.join(); server.server_close()


@pytest.mark.parametrize('change', [{'url': 'https://evil/secret'}, {'start': True}, {'end': 4 * 1024 ** 2},
                                  {'start': -1}, {'size': 3}, {'extra': 1}, {'url': None}])
def test_relay_scope_and_ranges_fail_closed(change):
    helper = module('ccr_relay')
    value = {'url': 'https://' + helper.HOST + '/registry/docker/registry/v2/blobs/sha256/aa/' + 'a' * 64 + '/data?signature=private',
             'start': 0, 'end': 3, 'size': 100}
    assert helper.range_spec(value)[1:] == (0, 3, 100)
    with pytest.raises(ValueError): helper.range_spec({**value, **change})


@pytest.mark.parametrize('case', ['valid', 'auth', 'host', 'fields', 'length', 'busy', 'entire', 'offset', 'short', 'redirect'])
def test_authenticated_relay_checks_complete_range_and_never_exposes_private_urls(monkeypatch, case, capsys):
    helper = module('ccr_relay')
    body = b'abcdefgh'; called = []
    class Response(io.BytesIO):
        status = 206
        def getheader(self, key):
            if key == 'Content-Range': return 'bytes 0-7/100' if case != 'offset' else 'bytes 1-8/100'
            if key == 'Content-Length': return '8'
    class Connection:
        def __init__(self, host, **kw): assert host == helper.HOST and kw['timeout'] == 30
        def request(self, method, path, headers): called.append((method, path, headers))
        def getresponse(self):
            result = Response(body if case != 'short' else b'a')
            result.status = 200 if case == 'entire' else 307 if case == 'redirect' else 206
            return result
        def close(self): pass
    monkeypatch.setattr(helper.http.client, 'HTTPSConnection', Connection)
    server = helper.ThreadingHTTPServer(('127.0.0.1', 0), helper.Handler)
    server.token = 'x' * 64; server.slots = threading.BoundedSemaphore(0 if case == 'busy' else 4)
    thread = threading.Thread(target=server.serve_forever); thread.start()
    value = {'url': 'https://' + helper.HOST + '/registry/docker/registry/v2/blobs/sha256/aa/' + 'a' * 64 + '/data?signature=test-secret',
             'start': 0, 'end': 7, 'size': 100}
    if case == 'host': value['url'] = 'https://evil/secret'
    if case == 'fields': value['unexpected'] = 1
    data = json.dumps(value).encode()
    try:
        connection = http.client.HTTPConnection(*server.server_address, timeout=3)
        headers = {'Authorization': 'Bearer ' + ('bad' if case == 'auth' else server.token)}
        if case == 'length': headers['Content-Length'] = '0'
        connection.request('POST', '/ml-expd-builder-relay', body=data, headers=headers)
        response = connection.getresponse(); output = response.read(); connection.close()
        expected = 206 if case == 'valid' else 401 if case == 'auth' else 429 if case == 'busy' else 422 if case in {'host', 'fields', 'length'} else 502
        assert response.status == expected
        assert (output == body) if case == 'valid' else (b'test-secret' not in output)
        if case in {'auth', 'host', 'fields', 'length', 'busy'}: assert not called
        assert 'test-secret' not in capsys.readouterr().err
    finally:
        server.shutdown(); thread.join(); server.server_close()


@pytest.mark.parametrize('case', ['changed-manifest', 'index', 'oversize', 'bad-layer', 'unpinned', 'bad-socket'])
def test_prewarm_rejects_unknown_identity_or_capacity_before_remote_write(tmp_path, monkeypatch, case):
    tool = module('prewarm_ccr')
    manifest = {'layers': [{'digest': 'sha256:' + 'a' * 64, 'size': 10}]}
    if case == 'index': manifest = {'manifests': []}
    if case == 'bad-layer': manifest['layers'][0]['size'] = -1
    raw = json.dumps(manifest).encode(); digest = hashlib.sha256(raw).hexdigest()
    image = tool.REGISTRY + '/ccr-zhicheng-02/elf@sha256:' + digest
    if case == 'changed-manifest': raw += b' '
    if case == 'unpinned': image = image.split('@')[0] + ':latest'
    config = {'docker_host': 'tcp://public:2375' if case == 'bad-socket' else 'unix:///run/private/docker.sock'}
    monkeypatch.setattr(tool.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=raw))
    monkeypatch.setattr(tool.subprocess, 'Popen', lambda *a, **kw: pytest.fail('must not write before identity checks'))
    with pytest.raises(ValueError): tool.prewarm(config, image, 1 if case == 'oversize' else 100,
                                               'https://api.example/ml-expd-builder-relay', 'x' * 64)


def test_build_request_wait_does_not_expire_before_private_900_second_builder(monkeypatch):
    from ml_exp_server import image_builder
    seen = []
    class Connection:
        def __init__(self, path, *, timeout): seen.append(timeout)
        def request(self, *a, **kw): pass
        def getresponse(self): return SimpleNamespace(status=200, read=lambda *a: b'{"image":"verified"}')
        def close(self): pass
    monkeypatch.setattr(image_builder, 'UnixConnection', Connection)
    assert image_builder.builder_request('/private.sock', {'operation': 'build'}) == {'image': 'verified'}
    assert seen == [1200]
    image_builder.builder_request('/private.sock', {'operation': 'logs'}, timeout=15)
    assert seen == [1200, 15]


@pytest.mark.parametrize('host,cert,key', [('0.0.0.0', None, None), ('::', None, None),
                                         ('127.0.0.1', 'cert', None), ('127.0.0.1', None, 'key')])
def test_public_relay_cannot_start_without_tls(host, cert, key):
    with pytest.raises(ValueError): module('ccr_relay').create_server(host, 0, 'x' * 64, cert, key)


def test_tls_relay_verifies_certificates_and_idle_handshake_does_not_block(tmp_path):
    helper = module('ccr_relay')
    key, cert = tmp_path / 'key.pem', tmp_path / 'cert.pem'
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'ec', '-pkeyopt', 'ec_paramgen_curve:P-256',
                    '-nodes', '-keyout', str(key), '-out', str(cert), '-days', '1', '-subj', '/CN=localhost',
                    '-addext', 'subjectAltName=DNS:localhost'], check=True, capture_output=True)
    server = helper.create_server('127.0.0.1', 0, 'x' * 64, cert, key)
    thread = threading.Thread(target=server.serve_forever); thread.start()
    stalled = socket.create_connection(server.server_address, timeout=3)
    try:
        rejected = http.client.HTTPSConnection('localhost', server.server_port, timeout=3)
        with pytest.raises(ssl.SSLCertVerificationError):
            rejected.request('GET', '/health')
        rejected.close()
        trusted = http.client.HTTPSConnection('localhost', server.server_port, timeout=3,
                                             context=ssl.create_default_context(cafile=str(cert)))
        trusted.request('POST', '/ml-expd-builder-relay', body=b'{}')
        response = trusted.getresponse(); assert response.status == 401; response.read(); trusted.close()
        trusted = http.client.HTTPSConnection('localhost', server.server_port, timeout=3,
                                             context=ssl.create_default_context(cafile=str(cert)))
        trusted.request('GET', '/health'); response = trusted.getresponse(); assert response.read() == b'ok'; trusted.close()
    finally:
        stalled.close(); server.shutdown(); thread.join(); server.server_close()
