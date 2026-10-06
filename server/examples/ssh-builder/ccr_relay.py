"""Restricted, authenticated CCR range relay behind the existing HTTPS ingress.

Bind to loopback by default; public binding requires TLS. POST bodies contain signed URLs so ingress paths never
contain them. No credentials, request bodies or upstream errors are logged.
This service writes no image data to disk and never submits builds or jobs.
"""
import argparse
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import socket
import ssl
import threading
from urllib.parse import urlsplit

HOST = 'aoss.cn-sh-01b.sensecoreapi-oss.cn'
CHUNK_LIMIT = 4 * 1024 * 1024
MAX_REQUESTS = 16


def range_spec(value):
    if not isinstance(value, dict) or set(value) != {'url', 'start', 'end', 'size'}:
        raise ValueError('invalid range fields')
    if not isinstance(value['url'], str):
        raise ValueError('invalid CCR URL')
    url = urlsplit(value['url'])
    match = re.fullmatch(r'/registry/docker/registry/v2/blobs/sha256/([0-9a-f]{2})/([0-9a-f]{64})/data', url.path)
    if (url.scheme != 'https' or url.netloc != HOST or url.fragment or match is None
            or not match[2].startswith(match[1]) or not url.query):
        raise ValueError('only signed CCR blob URLs are allowed')
    start, end, size = (value[key] for key in ('start', 'end', 'size'))
    if (any(type(x) is not int for x in (start, end, size)) or not 0 <= start <= end < size <= 8 * 1024 ** 3
            or end - start + 1 > CHUNK_LIMIT):
        raise ValueError('invalid CCR range bounds')
    return url.path + '?' + url.query, start, end, size


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == '/health':
            self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
        else:
            self.send_error(405, 'POST required')

    def do_POST(self):
        if self.path != '/ml-expd-builder-relay':
            self.send_error(404, 'Unknown relay route')
            return
        if not hmac.compare_digest(self.headers.get('Authorization', '').encode(), ('Bearer ' + self.server.token).encode()):
            self.send_error(401, 'Builder relay authorization required')
            return
        try:
            self.connection.settimeout(15)
            count = int(self.headers.get('Content-Length', '0'))
            if not 0 < count <= 16384:
                raise ValueError('invalid request length')
            path, start, end, size = range_spec(json.loads(self.rfile.read(count)))
        except (OSError, ValueError, TypeError, KeyError):
            self.send_error(422, 'Invalid CCR range request')
            return
        if not self.server.slots.acquire(blocking=False):
            self.send_error(429, 'Builder relay busy')
            return
        upstream = http.client.HTTPSConnection(HOST, timeout=30)
        try:
            upstream.request('GET', path, headers={'Range': f'bytes={start}-{end}'})
            response = upstream.getresponse()
            if (response.status != 206 or response.getheader('Content-Range') != f'bytes {start}-{end}/{size}'
                    or response.getheader('Content-Length') != str(end - start + 1)):
                self.send_error(502, 'CCR range unavailable or mismatched')
                return
            # Buffer at most one bounded range before responding. A short or
            # redirected upstream can never masquerade as a complete chunk.
            body = response.read(end - start + 2)
            if len(body) != end - start + 1:
                self.send_error(502, 'Incomplete CCR range')
                return
            self.connection.settimeout(60)  # Authorized slow receivers; TLS/header idleness remains 15 s.
            self.send_response(206)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
        except (OSError, http.client.HTTPException):
            self.close_connection = True
        finally:
            upstream.close()
            self.server.slots.release()


def create_server(host, port, token, cert=None, key=None):
    if bool(cert) != bool(key) or (host != '127.0.0.1' and not cert):
        raise ValueError('non-loopback relay requires a TLS certificate and key')
    context = None
    if cert:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(cert, key)
        context.set_alpn_protocols(['http/1.1'])

    class Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6 if ':' in host else socket.AF_INET
        def server_bind(self):
            if self.address_family == socket.AF_INET6:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            super().server_bind()
        def get_request(self):
            connection, address = super().get_request()
            connection.settimeout(15)
            if context:
                # Defer the handshake to the request thread. An idle TLS client
                # cannot block the main accept loop or another authenticated client.
                connection = context.wrap_socket(connection, server_side=True, do_handshake_on_connect=False)
            return connection, address
        def handle_error(self, request, client_address):
            # Expected rejected/expired TLS sessions never produce traceback logs.
            pass

    server = Server((host, port), Handler)
    server.token = token
    server.slots = threading.BoundedSemaphore(MAX_REQUESTS)
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token-file', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8878)
    parser.add_argument('--host', choices=['127.0.0.1', '0.0.0.0', '::'], default='127.0.0.1')
    parser.add_argument('--cert-file', type=Path)
    parser.add_argument('--key-file', type=Path)
    args = parser.parse_args()
    token = args.token_file.read_text().strip()
    if not re.fullmatch(r'[A-Za-z0-9_-]{32,128}', token):
        raise SystemExit('Invalid private relay credential')
    server = create_server(args.host, args.port, token, args.cert_file, args.key_file)
    with server:
        server.serve_forever()


if __name__ == '__main__':
    main()
