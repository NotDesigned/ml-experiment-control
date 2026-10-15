"""A byte-preserving OCI registry oracle; never opens a network connection."""
import hashlib
import json
from urllib.parse import parse_qs, urlsplit

from ml_exp_server.data.registry_transport import MANIFEST, digest, encoded


class Response:
    def __init__(self, status, headers=None, body=b''):
        self.status, self.headers, self.body = status, headers or {}, body
    def read(self, size):
        return self.body[:size]
    def getheaders(self):
        return list(self.headers.items())


class MemoryRegistry:
    host = 'registry.example'
    repository = host + '/test/data'
    prefix = '/v2/test/data'
    secret = 'fake-registry-password'
    def __init__(self):
        self.blobs, self.manifests, self.uploads, self.calls = {}, {}, {}, []
        self.patch_failure = self.finish_failure = self.publish_failure = None
        self.overrides = {}
        self.challenge = None
        self.token_payload, self.token_status = {'token': 'fake-bearer-secret'}, 200
        self.oss_response = None
        self.head_digest = None
        self.closed, self.next_upload = 0, 0
        layer = b'base-layer'
        self.blobs[digest(layer)] = layer
        self.base_config = {'architecture': 'amd64', 'os': 'linux',
            'rootfs': {'type': 'layers', 'diff_ids': [digest(layer)]}}
        raw = encoded(self.base_config)
        self.blobs[digest(raw)] = raw
        self.base = {'schemaVersion': 2, 'mediaType': MANIFEST,
            'config': {'digest': digest(raw), 'size': len(raw),
                       'mediaType': 'application/vnd.docker.container.image.v1+json'},
            'layers': [{'digest': digest(layer), 'size': len(layer),
                        'mediaType': 'application/vnd.docker.image.rootfs.diff.tar'}]}
        self.base_digest = digest(encoded(self.base))
        self.manifests[self.base_digest] = encoded(self.base)

    def connection(self, host, **kwargs):
        owner = self
        class Connection:
            def request(self, method, path, body=None, headers=None):
                self.reply = owner.request(host, method, path, body, dict(headers or {}))
            def getresponse(self):
                return self.reply
            def close(self):
                owner.closed += 1
        return Connection()

    def request(self, host, method, path, body, headers):
        self.calls.append((host, method, path, headers, body))
        if host != self.host:
            assert host == 'objects.example', 'an unapproved network host was contacted'
            assert 'Authorization' not in headers
            response = self.oss_response
            if isinstance(response, BaseException):
                raise response
            return response
        key = (method, path)
        if key in self.overrides:
            result = self.overrides[key]
            if isinstance(result, BaseException):
                raise result
            return result
        if path.startswith('/token'):
            return Response(self.token_status, body=encoded(self.token_payload))
        if self.challenge and not headers.get('Authorization', '').startswith('Bearer '):
            return Response(401, {'WWW-Authenticate': self.challenge})
        plain = urlsplit(path).path
        if method == 'POST' and plain == self.prefix + '/blobs/uploads/':
            self.next_upload += 1
            identity = 'session-' + str(self.next_upload)
            location = self.prefix + '/blobs/uploads/' + identity + '?state=private-location-secret'
            self.uploads[identity] = bytearray()
            return Response(202, {'Location': location})
        if '/blobs/uploads/' in plain:
            identity = plain.rsplit('/', 1)[-1]
            if method == 'DELETE':
                return Response(204 if self.uploads.pop(identity, None) is not None else 404)
            data = self.uploads[identity]
            if method == 'GET':
                return Response(204, {'Range': f'0-{max(0, len(data)-1)}', 'Location': path})
            if method == 'PATCH':
                assert headers['Content-Range'] == f'{len(data)}-{len(data)+len(body)-1}'
                failure, self.patch_failure = self.patch_failure, None
                if isinstance(failure, int):
                    data.extend(body[:failure])
                    raise OSError('fake partial write and lost response')
                data.extend(body)
                if failure == 'lost-ack':
                    raise OSError('fake committed write and lost response')
                return Response(202, {'Range': f'0-{len(data)-1}', 'Location': path})
            if method == 'PUT':
                sha = parse_qs(urlsplit(path).query)['digest'][0]
                if digest(data) != sha:
                    return Response(400)
                failure, self.finish_failure = self.finish_failure, None
                if failure == 'before-commit':
                    raise OSError('fake disconnected before blob finalization')
                self.blobs[sha] = bytes(data)
                if failure:
                    raise OSError('fake finalized blob and lost response')
                return Response(201, {'Docker-Content-Digest': sha})
        if '/blobs/' in plain:
            sha = plain.rsplit('/', 1)[-1]
            if sha not in self.blobs:
                return Response(404)
            return Response(200, {'Docker-Content-Digest': self.head_digest or sha},
                            b'' if method == 'HEAD' else self.blobs[sha])
        if '/manifests/' in plain:
            identity = plain.rsplit('/', 1)[-1]
            if method == 'PUT':
                self.manifests[identity] = body
                self.manifests[digest(body)] = body
                failure, self.publish_failure = self.publish_failure, None
                if failure:
                    raise OSError('fake published manifest and lost response')
                return Response(201)
            return Response(200, body=self.manifests[identity])
        raise AssertionError((method, path))
