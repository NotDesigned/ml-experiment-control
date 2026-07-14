# Versioned daemon HTTP contract

`GET /api/health` is the compatibility handshake for independently released
clients. It returns `api_protocol_version`, `min_client_protocol_version`, the
daemon package version, authentication/transport modes, a capability list, and
the versioned OpenAPI path (`/api/v1/openapi.json`). Clients send
`X-ML-Expd-Client-Protocol`; unsupported versions receive HTTP 426 with
`INCOMPATIBLE_API_PROTOCOL` before any operation is dispatched.
The header is mandatory for every API resource and mutation. A headerless
`GET /api/health` is the only bootstrap exception, so an operator can discover
the supported range before selecting a client; supplying an invalid header to
that endpoint still fails with HTTP 426.

Each Action execution envelope also carries a monotonic `revision`. The daemon
uses that revision as a compare-and-set token, so a late execute or reconcile
response cannot overwrite a newer durable transition even when the status text
itself has not changed.

Protocol version 1 currently guarantees these capability families:

- `terminal-snapshot.v1`
- `terminal-snapshot-limits.v1`
- `project-lifecycle.v1`
- `actions.v1`
- `submissions.v1`
- `observability.v1`
- `bearer-auth.v1` and `tls-bind.v1`

Adding an optional response field or capability does not require a protocol
bump. Removing or changing a required field, route meaning, identity rule, or
mutation lifecycle does. A client pin must pass the real subprocess
compatibility smoke before its submodule gitlink advances.

Terminal snapshots remain a complete run read model in protocol v1. Their
`scale` object reports Project and Run counts plus the bounded observability
target page's `returned`, `total`, `limit`, and `truncated` fields. A client
must not interpret a 500-target projection as complete when `truncated=true`.
The same target-page metadata is returned by `/api/observability`.
Callers that need one Project can pass `project=<id>` to the terminal snapshot;
the daemon then avoids loading and serializing unrelated Projects and target
statuses. An unknown filter returns 404 rather than an ambiguous empty view.

Health also reports the daemon-owned publisher loop separately from individual
outbox targets. `publisher.last_error`, `last_success_at`, and
`consecutive_failures` expose systemic loop failure; reviewed project writes
that could not be rolled forward during startup appear in
`project_write_recovery_errors`.

## Authentication and binding

Local loopback remains the default. To protect a shared multi-user host, create
an owner-only token and configure it without placing the secret in YAML:

```bash
umask 077
python -c 'import secrets; print(secrets.token_urlsafe(48))' \
  > ~/.config/ml-expd/http.token
```

```yaml
http_auth:
  bearer_token_file: ~/.config/ml-expd/http.token
```

Clients provide the value through `ML_EXPD_API_TOKEN`. Token files must be
regular, owned by the daemon user, mode `0600` or stricter, and contain at least
32 non-whitespace characters. A non-loopback bind additionally requires both
`--ssl-certfile` and `--ssl-keyfile`; the daemon refuses to send bearer tokens
over a remotely reachable plaintext listener. An authenticated local tunnel or
reverse proxy bound to loopback remains a valid deployment boundary.

The API never returns the token, its path, hashes, or authorization header.
