# HTTP contract

## Negotiate and authenticate

The current protocol is 2. `GET /api/health` returns `server_version`,
`api_protocol_version`, `min_client_protocol_version`, `capabilities`, enabled
policy and `openapi_path` (`/api/v2/openapi.json`). Package versions alone do not
prove a capability is available; frozen worker images have their own capabilities.

Every normal API request sends:

```text
Authorization: Bearer <private token>
X-ML-Expd-Client-Protocol: 2
```

Only headerless health GET may bootstrap protocol discovery; it still requires
configured authentication. Invalid/unsupported protocol receives HTTP 426 before
dispatch. A proxy must retain Authorization, protocol headers and the API prefix.

The daemon token file must be regular, daemon-owned, mode 0600 or stricter, with
at least 32 non-whitespace characters. This is shared-token authentication, not
per-user/project RBAC. The client supports `ML_EXPD_API_TOKEN_FILE` or
`ML_EXPD_API_TOKEN`; prefer the file. Cloud and registry credentials stay server-side.
Non-loopback daemon binding requires a token and native TLS certificate/key.
A TLS proxy may front a loopback HTTP daemon. Remote clients use HTTPS.

Retrieve the schema with `ml-exp check --schema openapi.json`, or an authenticated
protocol-aware HTTP client. Auth/protocol middleware is not represented as
OpenAPI security/parameter definitions. Stock `/docs` also needs authentication
and does not reliably inject headers or preserve a proxy prefix. There is no
`/api/guide`; use the [quickstart](api-quickstart.md).

## Async state and retries

With `async-actions.v1`, execute durably claims an Action and returns EXECUTING
while a daemon-owned thread continues. Runtime/data delivery have their own
durable async resources. Poll the returned ID; disconnecting does not cancel.
After daemon interruption, in-flight operations require reconciliation, not replay.

Action execution includes a monotonic `revision` for compare-and-set updates.
Use `resolution`, `safe_to_retry` and `next_action`, not HTTP timeout text, to
decide recovery. Important resolutions include SUBMITTED, STILL_IN_PROGRESS,
UNKNOWN_DO_NOT_RETRY, FAILED_BEFORE_SUBMISSION and NOT_SUBMITTED_SAFE_TO_RETRY.
Only an operation's read-only reconciliation can prove an uncertain effect absent.
Prepare a fresh reviewed operation when instructed; do not reuse stale gates.

Archive part retries are different: identical target/hash/size/part bytes are
idempotent. Those transport rules do not authorize retrying Run or scheduler create.

## SSE invalidations

`GET /api/stream` supplies best-effort index invalidations via `sse-starlette`,
with 15-second heartbeat/send timeout and bounded queues of 128 updates:

```text
data: {"type":"index_updated","project":"demo","run_id":"trial"}

```

Fetch an authoritative HTTP snapshot on initial connection and every reconnect.
There are no replay IDs; `Last-Event-ID` cannot recover missed updates. Queue
overflow emits `{"type":"resync_required"}` and closes the stream when possible.
Reload even if a timeout prevented receiving that message. Unknown event types
are allowed; a slow consumer never blocks others. See [API reads](source-api.md).
