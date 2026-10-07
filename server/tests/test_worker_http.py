"""Origin TLS is preserved while TCP goes to a fixed private gateway."""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
import ssl
import subprocess
import threading
from urllib.parse import urlsplit

import pytest

from ml_exp_server import worker_http as transport

RELAY = {'endpoint': 'tcp://127.0.0.1:18443', 'origin': 'https://example.test'}


@pytest.mark.parametrize('value', [[], {}, {'endpoint': 'x', 'origin': 'https://example.test'},
    {'endpoint': 'tcp://8.8.8.8:443', 'origin': 'https://example.test'},
    {'endpoint': 'tcp://0.0.0.0:443', 'origin': 'https://example.test'},
    {'endpoint': 'tcp://224.0.0.1:443', 'origin': 'https://example.test'},
    {'endpoint': 'tcp://127.0.0.1', 'origin': 'https://example.test'},
    {'endpoint': 'tcp://127.0.0.1:0', 'origin': 'https://example.test'},
    {'endpoint': 'tcp://127.0.0.1:99999', 'origin': 'https://example.test'},
    {'endpoint': 'tcp://127.0.0.1:443/', 'origin': 'https://example.test'},
    {'endpoint': 'tcp://user:secret@127.0.0.1:443', 'origin': 'https://example.test'},
    {'endpoint': 'tcp://127.0.0.1:443?private', 'origin': 'https://example.test'},
    {'endpoint': 'tcp://127.0.0.1:443#private', 'origin': 'https://example.test'},
    {'endpoint': 'http://127.0.0.1:443', 'origin': 'https://example.test'},
    {**RELAY, 'origin': 'http://example.test'}, {**RELAY, 'origin': 'https://example.test:0'},
    {**RELAY, 'origin': 'https://example.test:99999'}, {**RELAY, 'origin': 'https://'},
    {**RELAY, 'origin': 'https://example.test/path'}, {**RELAY, 'origin': 1},
    {**RELAY, 'endpoint': None}, {**RELAY, 'extra': 'secret'}])
def test_relay_config_rejects_unsafe_or_ambiguous_endpoints(value):
    with pytest.raises(transport.ApiTransportError, match='API_RELAY_CONFIG_INVALID'):
        transport.relay_environment(value)


def test_direct_transport_ignores_unrelated_proxy_and_relay_is_origin_scoped(monkeypatch):
    assert transport.relay_environment(None) == {}
    assert transport.relay_environment({**RELAY, 'endpoint': 'tcp://[::1]:18443'})
    direct = transport.https_connection(urlsplit('https://example.test'), timeout=5, environment={})
    assert direct.host == 'example.test' and direct.port == 443
    env = transport.relay_environment(RELAY)
    routed = transport.https_connection(urlsplit('https://example.test'), timeout=5, environment=env)
    with pytest.raises(transport.ApiTransportError, match='API_RELAY_TARGET_MISMATCH'):
        transport.https_connection(urlsplit('https://other.test'), timeout=5, environment=env)
    with pytest.raises(transport.ApiTransportError, match='API_RELAY_TARGET_MISMATCH'):
        transport.https_connection(urlsplit('https://example.test:444'), timeout=5, environment=env)
    for key in env:
        with pytest.raises(transport.ApiTransportError, match='CONFIG_INVALID'):
            transport.https_connection(urlsplit('https://example.test'), timeout=5, environment={key: env[key]})
    reached = []
    monkeypatch.setattr(transport.socket, 'create_connection', lambda *args: reached.append(args) or 'socket')
    assert routed._create_connection(('example.test', 443), 5, None) == 'socket'
    assert reached == [(('127.0.0.1', 18443), 5, None)]
    def fail(*args): raise OSError('private credential must not escape')
    monkeypatch.setattr(transport.socket, 'create_connection', fail)
    with pytest.raises(transport.ApiTransportError, match='^API_RELAY_UNREACHABLE$'):
        routed.connect()
    for key, value in env.items(): monkeypatch.setenv(key, value)
    assert transport.https_connection(urlsplit('https://example.test'), timeout=1).host == 'example.test'


def test_real_tls_uses_origin_sni_certificate_and_authorization(tmp_path, monkeypatch):
    cert, key = tmp_path/'cert.pem', tmp_path/'key.pem'
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1',
        '-keyout', str(key), '-out', str(cert), '-subj', '/CN=example.test',
        '-addext', 'subjectAltName=DNS:example.test'], check=True, capture_output=True)
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.headers['Host'], self.headers['Authorization']))
            self.send_response(200); self.end_headers(); self.wfile.write(b'origin')
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); context.load_cert_chain(cert, key)
    context.set_servername_callback(lambda sock, name, context: seen.append(name))
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    trusted = ssl.create_default_context(cafile=str(cert))
    monkeypatch.setattr(ssl, '_create_default_https_context', lambda: trusted)
    try:
        env = transport.relay_environment({**RELAY, 'endpoint': f'tcp://127.0.0.1:{server.server_port}'})
        connection = transport.https_connection(urlsplit('https://example.test'), timeout=3, environment=env)
        connection.request('GET', '/', headers={'Authorization': 'Bearer scoped-capability'})
        assert connection.getresponse().read() == b'origin'; connection.close()
        assert seen == ['example.test', ('example.test', 'Bearer scoped-capability')]
        env['ML_EXPD_API_ORIGIN'] = 'https://wrong.test'
        bad = transport.https_connection(urlsplit('https://wrong.test'), timeout=3, environment=env)
        with pytest.raises(ssl.SSLCertVerificationError): bad.connect()
        bad.close()
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_network_diagnostics_preserve_codes_without_raw_details():
    from ml_exp_server.data_input import DeliveryTrace
    from ml_exp_server import worker_launcher
    assert transport.error_code(OSError('private')) == 'OSError'
    assert transport.error_code(transport.ApiTransportError('private')) == 'ApiTransportError'
    error = transport.ApiTransportError('API_RELAY_UNREACHABLE')
    assert transport.error_code(error) == 'API_RELAY_UNREACHABLE'
    trace = DeliveryTrace(); trace.failed(error)
    assert trace.value['code'] == 'API_RELAY_UNREACHABLE'


def test_injected_libraries_import_shared_transport_without_server_dependencies(monkeypatch):
    import runpy, sys
    from pathlib import Path
    directory = Path(transport.__file__).parent
    monkeypatch.syspath_prepend(str(directory))
    for name in ('worker_artifacts.py', 'persistent_state.py', 'data_input.py'):
        assert callable(runpy.run_path(str(directory/name))['https_connection'])
