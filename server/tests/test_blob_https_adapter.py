"""The optional desktop adapter streams only the CCR blob endpoint over TLS."""
import http.client
import importlib.util
import io
from pathlib import Path
import threading

import pytest


@pytest.fixture
def adapter(monkeypatch):
    path = Path(__file__).parents[1]/'examples/ssh-builder/https_blobs.py'
    spec = importlib.util.spec_from_file_location('blob_adapter', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    requests = []
    body = b'blob-bytes' * 20000
    class Connection:
        def __init__(self, host, **kw):
            assert host == module.HOST and kw['timeout'] == 120
            self.stream = io.BytesIO(body)
        def request(self, method, path, headers): requests.append((method, path, headers))
        def getresponse(self): return self
        status = 200
        def getheader(self, key): return str(len(body)) if key == 'Content-Length' else None
        def read(self, size):
            assert size == 65536
            return self.stream.read(size)
        def close(self): pass
    monkeypatch.setattr(module.http.client, 'HTTPSConnection', Connection)
    server = module.ThreadingHTTPServer(('127.0.0.1', 0), module.Handler)
    thread = threading.Thread(target=server.serve_forever); thread.start()
    try: yield module, server.server_address[1], requests, body
    finally: server.shutdown(); thread.join(); server.server_close()


@pytest.mark.parametrize('method', ['GET', 'HEAD'])
def test_signed_path_and_range_are_preserved_without_logging(adapter, method, capsys):
    module, port, requests, body = adapter
    path = '/registry/docker/registry/v2/blobs/sha256/aa/data?X-Amz-test=test-secret'
    connection = http.client.HTTPConnection('127.0.0.1', port)
    connection.request(method, 'http://'+module.HOST+path, headers={'Range': 'bytes=0-9', 'Authorization': 'must-not-forward'})
    response = connection.getresponse()
    assert response.status == 200
    assert response.read() == (body if method == 'GET' else b'')
    connection.close()
    assert requests == [(method, path, {'Range': 'bytes=0-9'})]
    assert capsys.readouterr().err == ''


@pytest.mark.parametrize('target,method,status', [
    ('http://evil.test/registry/docker/registry/v2/blobs/sha256/aa', 'GET', 403),
    ('http://user:pass@aoss.cn-sh-01b.sensecoreapi-oss.cn/registry/docker/registry/v2/blobs/sha256/aa', 'GET', 403),
    ('http://aoss.cn-sh-01b.sensecoreapi-oss.cn/other', 'GET', 403),
    ('http://[invalid/', 'GET', 403),
    ('example.test:443', 'CONNECT', 501),
])
def test_adapter_cannot_become_a_general_proxy(adapter, target, method, status):
    module, port, requests, _ = adapter
    connection = http.client.HTTPConnection('127.0.0.1', port)
    connection.putrequest(method, target, skip_host=True)
    connection.putheader('Host', 'adapter')
    connection.endheaders()
    response = connection.getresponse(); assert response.status == status; response.read(); connection.close()
    assert not requests


def test_adapter_reports_upstream_failure_without_private_url(adapter, monkeypatch):
    module, port, _, _ = adapter
    def unavailable(*a, **kw): raise OSError('private upstream evidence')
    monkeypatch.setattr(module.http.client.HTTPSConnection, 'request', unavailable)
    connection = http.client.HTTPConnection('127.0.0.1', port)
    connection.request('GET', 'http://'+module.HOST+'/registry/docker/registry/v2/blobs/sha256/aa?secret=value')
    response = connection.getresponse(); body = response.read(); connection.close()
    assert response.status == 502 and b'secret' not in body and b'private' not in body
