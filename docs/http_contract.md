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

## Structured errors and execution capacity

Background execution has separate bounded action, preparation, build, data and
collection lanes. Capacity counts running work, queued work and reservations
before durable execution claims. A saturated lane rejects new execution admission
with HTTP 503 before that request claims or enqueues work:

```json
{"detail": "execution capacity is full or stopping; retry this request later", "code": "EXECUTION_CAPACITY_UNAVAILABLE", "lane": "action", "retry_after_seconds": 2}
```

The response includes `X-ML-Expd-Error-Code: EXECUTION_CAPACITY_UNAVAILABLE` and
`Retry-After: 2`. Observation GETs remain available while execution is saturated.
After an accepted claim, disconnecting does not abandon its enqueue or background
work; shutdown keeps the workspace lease until admitted handoffs and workers
finish. A queue 503 does not authorize replaying a different, earlier request whose
outcome is unknown. Query its saved resource first; the client never automatically
repeats scheduler POSTs.

The Python SDK raises `ClientError` with `status`, `retryable`, `error_code`,
`retry_after` (seconds or null), `body_truncated`, `response_body` and `details`.
`response_body` preserves the complete sanitized JSON shape, including top-level
queue lane/retry hints or storage required/available/reserved byte counts.
`details` remains the JSON `detail` value when present, otherwise the complete
body. HTTP failures preserve their status without becoming automatically
retryable; connection failures set `retryable=true` but still do not replay calls.

Only complete JSON bodies up to 16 KiB are retained. Strings, nesting and entry
counts are bounded; credentials, request inputs and URLs are redacted. Oversized
or malformed bodies are discarded, and the HTTP response is closed. `str(error)`
includes only method, sanitized API path, status and safe error code, excluding
query strings and JSON details. The CLI writes a JSON error object to stderr with
that summary and the sanitized SDK fields, so queue and capacity evidence remains
visible without raw bodies, headers or signed URLs. Inspect these fields or the
original resource's diagnostics:

```python
from ml_exp_client import ClientError

try:
    result = client.call(endpoint, data=request)
except ClientError as error:
    status, code = error.status, error.error_code
    diagnostics = error.response_body
    retry_after_seconds = error.retry_after
    # Observe saved IDs before deciding on an explicit retry.
    raise
```

## Client observation and explicit training operations

`ml-exp logs` and `ml-exp metrics` use
`GET /api/attempts/P/R::A/logs` and `GET /api/attempts/P/R::A/metrics`. When no
Attempt is supplied, the client first reads `GET /api/runs/P/R/attempts` and binds
the request to its current Attempt. These reads do not submit work.

`ml-exp cancel` and `ml-exp retry` first POST
`/api/attempts/P/R::A/cancel` or `/api/attempts/P/R::A/retry` to prepare an Action.
The client saves the request before HTTP and saves the returned Action's identity,
scope and review digests. A separate `--execute` invocation rechecks those values,
authorizes `/api/actions/authorize` and executes `/api/actions/execute`. Saved
dispatch markers prevent automatic replay after a lost reply. Observation uses
`GET /api/actions/ID`; explicit `--reconcile` calls `/api/actions/reconcile` and
never resubmits the operation. Retry requires a finite positive GPU-hour budget
and creates a new Attempt in the same frozen Run, without selecting a checkpoint.

`experiment --download-to` observes the original Attempt's collection after
training becomes terminal. It requires `AVAILABLE`, `SHA256_VERIFIED` and an
available download before retrieval, and never POSTs collection while waiting.
See [recovery](recovery.md#resume-publication-or-a-download-acknowledgement) for
timeouts, terminal collection failures and ACK-only resume.

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
