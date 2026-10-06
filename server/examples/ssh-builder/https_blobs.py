"""Restricted HTTP-to-HTTPS adapter for CCR's signed blob redirects.

Run on the private BuildKit Docker network without publishing any host port.
This is an operator workaround for an HTTP-only CCR redirect, not a general
forward proxy. Query strings and headers are never logged or persisted.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
from urllib.parse import urlsplit

HOST = 'aoss.cn-sh-01b.sensecoreapi-oss.cn'
REQUEST_HEADERS = ('Range', 'If-Match', 'If-None-Match', 'If-Modified-Since', 'If-Unmodified-Since')
RESPONSE_HEADERS = ('Content-Length', 'Content-Type', 'Content-Range', 'Accept-Ranges', 'ETag', 'Last-Modified')


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
