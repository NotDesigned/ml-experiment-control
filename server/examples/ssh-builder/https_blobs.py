"""Restricted HTTP-to-HTTPS adapter for CCR's signed blob redirects.

Run on the private BuildKit Docker network without publishing any host port.
This is an operator workaround for an HTTP-only CCR redirect, not a general
forward proxy. Query strings and headers are never logged or persisted.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import os
from pathlib import Path
import re
import stat
from urllib.parse import urlsplit

HOST = 'aoss.cn-sh-01b.sensecoreapi-oss.cn'
REQUEST_HEADERS = ('Range', 'If-Match', 'If-None-Match', 'If-Modified-Since', 'If-Unmodified-Since')
RESPONSE_HEADERS = ('Content-Length', 'Content-Type', 'Content-Range', 'Accept-Ranges', 'ETag', 'Last-Modified')
CACHE = Path(os.environ['CCR_BLOB_CACHE']) if 'CCR_BLOB_CACHE' in os.environ else None


def cached_blob(path):
    """Only administrator-prewarmed, content-addressed regular files are served.

    The operator verifies the complete SHA256 before atomic publication; BuildKit
    independently verifies the requested digest. The adapter never caches an
    arbitrary request or changes the upstream image identity.
    """
    match = re.fullmatch(r'/registry/docker/registry/v2/blobs/sha256/([0-9a-f]{2})/([0-9a-f]{64})/data', path)
    if CACHE is None or match is None or not match[2].startswith(match[1]):
        return None
    try:
        fd = os.open(CACHE / match[2], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            return None
        return os.fdopen(fd, 'rb')
    except OSError:
        return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.forward()

    def do_HEAD(self):
        self.forward()

    def forward(self):
        try:
            url = urlsplit(self.path)
        except ValueError:
            self.send_error(403, 'Invalid CCR blob endpoint')
            return
        if (url.scheme != 'http' or url.netloc != HOST or url.fragment
                or not url.path.startswith('/registry/docker/registry/v2/blobs/sha256/')):
            self.send_error(403, 'Only the configured CCR blob endpoint is allowed')
            return
        blob = cached_blob(url.path)
        if blob is not None:
            with blob:
                size = os.fstat(blob.fileno()).st_size
                # Conditional and range requests go upstream. BuildKit normally
                # downloads a full blob; do not weaken HTTP preconditions.
                if not any(key in self.headers for key in REQUEST_HEADERS):
                    self.send_response(200)
                    self.send_header('Content-Length', str(size))
                    self.send_header('Content-Type', 'application/octet-stream')
                    self.end_headers()
                    try:
                        if self.command == 'GET':
                            while data := blob.read(1048576):
                                self.wfile.write(data)
                    except OSError:
                        self.close_connection = True
                    return
        connection = http.client.HTTPSConnection(HOST, timeout=120)
        try:
            headers = {key: self.headers[key] for key in REQUEST_HEADERS if key in self.headers}
            path = url.path + ('?' + url.query if url.query else '')
            connection.request(self.command, path, headers=headers)
            response = connection.getresponse()
        except (OSError, http.client.HTTPException):
            connection.close()
            self.send_error(502, 'CCR HTTPS connection unavailable')
            return
        try:
            self.send_response(response.status)
            for key in RESPONSE_HEADERS:
                value = response.getheader(key)
                if value is not None:
                    self.send_header(key, value)
            self.end_headers()
            if self.command == 'GET':
                while data := response.read(65536):
                    self.wfile.write(data)
        except (OSError, http.client.HTTPException):
            self.close_connection = True
        finally:
            connection.close()


if __name__ == '__main__':
    ThreadingHTTPServer(('0.0.0.0', 8080), Handler).serve_forever()
