"""OCI trust, credentials and immutable identity checks without real networking."""
import base64
import http.client
import json

import pytest

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.data import registry_transport as module
from ml_exp_server.data.registry_transport import Registry, digest, encoded, INDEX, MANIFEST
from tests.fake_registry import MemoryRegistry, Response


@pytest.fixture
def registry(tmp_path, monkeypatch):
    memory = MemoryRegistry()
    auth = tmp_path / 'auth.json'
    auth.write_text(json.dumps({'auths': {memory.host: {'auth': base64.b64encode(
        ('fake-user:' + memory.secret).encode()).decode()}}}))
    monkeypatch.setattr(module.http.client, 'HTTPSConnection', memory.connection)
    return Registry(memory.repository, auth, blob_hosts=['objects.example']), memory, auth


def test_registry_blob_upload_manifest_and_reuse_verify_exact_bytes(registry):
    registry, memory, _ = registry
    body = b'arbitrary upload data' * 100
    descriptor = registry.blob(body)
    assert memory.blobs[descriptor['digest']] == body and descriptor['size'] == len(body)
    requests = len(memory.calls)
    assert registry.blob(body) == descriptor
    assert all(call[1] == 'HEAD' for call in memory.calls[requests:])
    assert registry.json_blob(memory.base['config']['digest']) == memory.base_config
    assert registry.manifest(memory.base_digest) == memory.base
    reference = registry.publish('data-' + 'a' * 64, memory.base)
    assert reference == memory.repository + '@' + memory.base_digest
    assert memory.closed == len(memory.calls)


def test_bearer_token_is_requested_from_same_origin_and_never_returned(registry):
    registry, memory, _ = registry
    memory.challenge = 'Bearer realm="https://registry.example/token?account=fake",service="registry-service"'
    assert registry.manifest(memory.base_digest) == memory.base
    token_call = next(call for call in memory.calls if call[2].startswith('/token'))
    assert 'scope=repository%3Atest%2Fdata%3Apull%2Cpush' in token_call[2]
    assert token_call[3]['Authorization'].startswith('Basic ')
    assert memory.calls[-1][3]['Authorization'] == 'Bearer fake-bearer-secret'


@pytest.mark.parametrize('location', ['https://other.example/v2/test/data/blobs/uploads/id',
    'http://registry.example/v2/test/data/blobs/uploads/id', '//other.example/upload',
    '/v2/other/data/blobs/uploads/id', '/v2/test/data/blobs/uploads/id#fragment',
    'https://fake:password@registry.example/v2/test/data/blobs/uploads/id'])
def test_upload_location_cannot_escape_trusted_origin_and_repository(registry, location):
    registry, memory, _ = registry
    with pytest.raises(ValueError, match='escaped'):
        registry.location(location)
    assert not memory.calls


@pytest.mark.parametrize('challenge', ['Basic realm="registry"',
    'Bearer realm="http://registry.example/token"', 'Bearer realm="https://other.example/token"',
    'Bearer realm="https://user:secret@registry.example/token"',
    'Bearer realm="https://registry.example/token#fragment"'])
def test_authentication_realm_is_never_followed_outside_trust_boundary(registry, challenge):
    registry, memory, _ = registry
    memory.challenge = challenge
    with pytest.raises(ValueError, match='realm'):
        registry.manifest(memory.base_digest)
    assert len(memory.calls) == 1


@pytest.mark.parametrize('payload,status,error', [({'token': ''}, 200, ValueError),
    ({'token': 'invalid\r\nheader'}, 200, ValueError), ({'token': 17}, 200, ValueError),
    ({'access_token': 'valid'}, 403, ApplicationError)])
def test_authentication_failure_is_bounded_and_has_no_credential_echo(registry, payload, status, error):
    registry, memory, _ = registry
    memory.challenge = 'Bearer realm="https://registry.example/token"'
    memory.token_payload, memory.token_status = payload, status
    with pytest.raises(error) as failure:
        registry.manifest(memory.base_digest)
    assert memory.secret not in str(failure.value)


@pytest.mark.parametrize('path', ['relative', '//other.example/path', '/path\r\nHeader: bad'])
def test_request_rejects_nonlocal_or_header_injection_paths(registry, path):
    registry, memory, _ = registry
    with pytest.raises(ValueError):
        registry.request('GET', path)
    assert not memory.calls


@pytest.mark.parametrize('error', [OSError('fake socket failure'), http.client.HTTPException('fake protocol failure')])
def test_transport_failures_are_safe_typed_errors(registry, error):
    registry, memory, _ = registry
    path = '/v2/test/data/blobs/' + 'sha256:' + 'a' * 64
    memory.overrides['HEAD', path] = error
    with pytest.raises(ApplicationError) as failure:
        registry.check_blob('sha256:' + 'a' * 64)
    assert failure.value.code == 'CCR_TRANSPORT' and failure.value.status_code == 503
    assert memory.closed == 1


def test_metadata_size_limit_and_http_errors(registry):
    registry, memory, _ = registry
    path = '/v2/test/data/metadata'
    memory.overrides['GET', path] = Response(200, body=b'12345')
    with pytest.raises(ValueError, match='bound'):
        registry.request('GET', path, maximum=4)
    for status, expected in [(403, 409), (507, 503)]:
        memory.overrides['GET', path] = Response(status)
        with pytest.raises(ApplicationError) as failure:
            registry.request('GET', path)
        assert failure.value.status_code == expected and failure.value.code == 'CCR_HTTP_' + str(status)


def test_blob_head_and_config_and_manifest_identities_are_enforced(registry):
    registry, memory, _ = registry
    memory.head_digest = 'sha256:' + 'f' * 64
    with pytest.raises(ValueError, match='identity'):
        registry.check_blob(memory.base['config']['digest'])
    memory.head_digest = None
    config_path = registry.prefix + '/blobs/' + memory.base['config']['digest']
    memory.overrides['GET', config_path] = Response(200, body=b'{}')
    with pytest.raises(ValueError, match='identity'):
        registry.json_blob(memory.base['config']['digest'])
    memory.overrides['GET', registry.prefix + '/manifests/' + memory.base_digest] = Response(200, body=b'{}')
    with pytest.raises(ValueError, match='identity'):
        registry.manifest(memory.base_digest)


@pytest.mark.parametrize('location', ['http://other.example/private?signature=fake',
    'https://objects.example:1234/private', 'https://user:secret@objects.example/private',
    'https://objects.example/private#fragment', 'file://objects.example/private'])
def test_signed_blob_redirect_is_exact_allowlist_only(registry, location):
    registry, memory, _ = registry
    path = registry.prefix + '/blobs/' + memory.base['config']['digest']
    memory.overrides['GET', path] = Response(307, {'Location': location})
    with pytest.raises(ValueError, match='redirect'):
        registry.json_blob(memory.base['config']['digest'])
    assert len(memory.calls) == 1


def test_approved_http_blob_url_is_upgraded_to_tls_without_registry_auth(registry):
    registry, memory, _ = registry
    sha = memory.base['config']['digest']
    memory.overrides['GET', registry.prefix + '/blobs/' + sha] = Response(307,
        {'Location': 'http://objects.example/private?signature=fake-oss-secret'})
    memory.oss_response = Response(200, body=memory.blobs[sha])
    assert registry.json_blob(sha) == memory.base_config
    assert memory.calls[-1][0] == 'objects.example'
    assert memory.calls[-1][3] == {}


def test_multiarch_base_selects_exact_linux_amd64_manifest(registry):
    registry, memory, _ = registry
    index = {'schemaVersion': 2, 'mediaType': INDEX, 'manifests': [
        {'digest': memory.base_digest, 'platform': {'os': 'linux', 'architecture': 'amd64'}},
        {'digest': 'sha256:' + 'f' * 64, 'platform': {'os': 'linux', 'architecture': 'arm64'}}]}
    sha = digest(encoded(index))
    memory.manifests[sha] = encoded(index)
    assert registry.manifest(sha) == memory.base
    for rows in [[], [index['manifests'][0], index['manifests'][0]]]:
        invalid = {**index, 'manifests': rows}
        sha = digest(encoded(invalid)); memory.manifests[sha] = encoded(invalid)
        with pytest.raises(ValueError, match='one linux/amd64'):
            registry.manifest(sha)


@pytest.mark.parametrize('repository', [None, 'registry.example', 'https://registry.example/test', 'registry.example/a/../b'])
def test_repository_is_an_explicit_registry_repository(registry, repository):
    _, _, auth = registry
    with pytest.raises(ValueError, match='repository'):
        Registry(repository, auth)


@pytest.mark.parametrize('credential', [None, {}, {'auth': 17},
    {'auth': base64.b64encode(b'no-colon').decode()},
    {'auth': base64.b64encode(b'user:pass\n').decode()},
    {'auth': base64.b64encode(b'user:pass\r').decode()}])
def test_missing_and_invalid_docker_credentials_fail_before_network(registry, credential):
    _, memory, auth = registry
    auth.write_text(json.dumps({'auths': {memory.host: credential}}))
    with pytest.raises(ValueError):
        Registry(memory.repository, auth)
    assert memory.calls == []


def test_origin_key_credentials_and_access_token_alias_are_supported(registry):
    _, memory, auth = registry
    doc = json.loads(auth.read_text())
    auth.write_text(json.dumps({'auths': {'https://' + memory.host: doc['auths'][memory.host]}}))
    transport = Registry(memory.repository, auth)
    memory.challenge = 'Bearer realm="https://registry.example/token"'
    memory.token_payload = {'access_token': 'valid-access-token'}
    assert transport.manifest(memory.base_digest) == memory.base
    assert transport.manifest(memory.base_digest) == memory.base


def test_upload_offset_and_ack_must_match_exact_bytes(registry):
    registry, memory, _ = registry
    location = registry.begin()
    assert registry.offset(location)[0] == 0
    memory.overrides['GET', location] = Response(204)
    with pytest.raises(ValueError, match='offset'):
        registry.offset(location)
    with pytest.raises(ValueError, match='empty'):
        registry.append(location, 0, b'')
    memory.overrides['PATCH', location] = Response(202, {'Range': '0-99'})
    with pytest.raises(ValueError, match='offset'):
        registry.append(location, 0, b'abc')
    assert registry.location('/v2/test/data/blobs/uploads/test') == '/v2/test/data/blobs/uploads/test'
    del memory.overrides['PATCH', location]
    noquery = location.split('?', 1)[0]
    registry.append(noquery, 0, b'abc')
    registry.finish(noquery, digest(b'abc'))


@pytest.mark.parametrize('operation', ['finish', 'check_blob', 'json_blob', 'manifest'])
def test_blob_and_manifest_references_must_be_sha256_digests(registry, operation):
    registry, memory, _ = registry
    with pytest.raises(ValueError):
        getattr(registry, operation)('/v2/test/data/blobs/uploads/test', 'latest') if operation == 'finish' else getattr(registry, operation)('latest')
    assert not memory.calls


@pytest.mark.parametrize('response', [Response(403), Response(200, body=b'x' * (8*1024**2+1)),
    OSError('fake oss failure'), http.client.HTTPException('fake oss protocol failure')])
def test_redirect_download_failures_are_bounded_and_safe(registry, response):
    registry, memory, _ = registry
    sha = memory.base['config']['digest']
    memory.overrides['GET', registry.prefix + '/blobs/' + sha] = Response(307,
        {'Location': 'https://objects.example/private'})
    memory.oss_response = response
    with pytest.raises((ValueError, ApplicationError)):
        registry.json_blob(sha)
    assert memory.closed == len(memory.calls)
    assert memory.calls[-1][3] == {}


def test_unsupported_manifest_and_bad_publication_contract_are_rejected(registry, monkeypatch):
    registry, memory, _ = registry
    bad = {'mediaType': 'application/unsupported'}
    sha = digest(encoded(bad)); memory.manifests[sha] = encoded(bad)
    with pytest.raises(ValueError, match='unsupported'):
        registry.manifest(sha)
    with pytest.raises(ValueError, match='tag'):
        registry.publish('latest', memory.base)
    monkeypatch.setattr(registry, 'manifest', lambda *_args: {'another': 'manifest'})
    with pytest.raises(ValueError, match='published data image differs'):
        registry.publish('data-' + 'a' * 64, memory.base)


@pytest.mark.parametrize('path', [
    '/v2/test/data/blobs/uploads/%2e%2e/%2e%2e/%2e%2e/%2e%2e/another',
    '/v2/test/data/blobs/uploads/%2F..%2Fanother',
    'https://registry.example/v2/test/data/blobs/uploads/../another',
    'https://registry.example/v2/test/data/blobs/uploads/./session'])
def test_upload_location_rejects_encoded_and_literal_path_traversal(registry, path):
    registry, memory, _ = registry
    with pytest.raises(ValueError, match='escaped'):
        registry.location(path)
    assert not memory.calls


def test_authenticated_head_redirect_confirms_blob_existence_over_tls_without_auth_forwarding(registry):
    registry, memory, _ = registry
    value = memory.blobs[memory.base['config']['digest']]
    sha = digest(value)
    memory.overrides['HEAD', registry.prefix + '/blobs/' + sha] = Response(307,
        {'Location': 'http://objects.example/layer?signature=fake-oss-secret'})
    memory.oss_response = Response(200, {'Content-Length': str(len(value))})
    registry.check_blob(sha)
    assert registry.blob(value) == {'digest': sha, 'size': len(value)}
    oss = [call for call in memory.calls if call[0] == 'objects.example']
    assert oss and all(call[1] == 'HEAD' and call[3] == {} for call in oss)
    assert not any(call[1] in {'POST', 'PATCH', 'PUT'} for call in memory.calls)


@pytest.mark.parametrize('location', ['https://other.example/private',
    'https://objects.example:1234/private', 'https://user:secret@objects.example/private'])
def test_head_redirect_never_contacts_an_unapproved_blob_target(registry, location):
    registry, memory, _ = registry
    sha = memory.base['config']['digest']
    memory.overrides['HEAD', registry.prefix + '/blobs/' + sha] = Response(307, {'Location': location})
    with pytest.raises(ValueError, match='redirect'):
        registry.check_blob(sha)
    assert len(memory.calls) == 1


@pytest.mark.parametrize('response', [Response(404), Response(200, body=b'unexpected-head-body'),
    OSError('fake HEAD disconnected'), http.client.HTTPException('fake HEAD protocol failure')])
def test_head_redirect_requires_success_and_empty_body_with_safe_transport_errors(registry, response):
    registry, memory, _ = registry
    sha = memory.base['config']['digest']
    memory.overrides['HEAD', registry.prefix + '/blobs/' + sha] = Response(307,
        {'Location': 'https://objects.example/private'})
    memory.oss_response = response
    with pytest.raises((ValueError, ApplicationError)):
        registry.check_blob(sha)
    assert memory.calls[-1][1] == 'HEAD' and memory.calls[-1][3] == {}
    assert memory.closed == len(memory.calls)
