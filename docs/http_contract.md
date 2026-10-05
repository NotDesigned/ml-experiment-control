# Versioned daemon HTTP contract

`GET /api/health` is the compatibility handshake for independently released
clients. It returns `api_protocol_version`, `min_client_protocol_version`, the
daemon package version, authentication/transport modes, a capability list, and
the versioned OpenAPI path (`/api/v2/openapi.json`). Clients send
`X-ML-Expd-Client-Protocol`; unsupported versions receive HTTP 426 with
`INCOMPATIBLE_API_PROTOCOL` before any operation is dispatched.
The header is mandatory for every API resource and mutation. A headerless
`GET /api/health` is the only bootstrap exception, so an operator can discover
the supported range before selecting a client; supplying an invalid header to
that endpoint still fails with HTTP 426. This exception concerns the protocol
header only: health still requires the configured Bearer authentication.

Each Action execution envelope also carries a monotonic `revision`. The daemon
uses that revision as a compare-and-set token, so a late execute or reconcile
response cannot overwrite a newer durable transition even when the status text
itself has not changed. Reconcile is not a generic mutation retry endpoint:
operation-specific reconciliation must observe durable effects without
reissuing them, or fail closed as `RECONCILE_REQUIRED` when no read-only proof
exists.

Protocol version 2 provides source/container execution and artifact downloads.
The current implementation advertises these capability families:

- `source-import.v1`, `container-execution.v1`, `artifact-download.v1`
- `terminal-snapshot.v1`, `terminal-snapshot-limits.v1`
- `project-lifecycle.v1`, `project-import.v1`, `project-source-locator.v1`
- `source-revision-import.v1`
- `actions.v1`, `submissions.v1`
- `action-resolution.v1`, `async-actions.v1`, `run-clone.v1`
- `bearer-auth.v1`, `tls-bind.v1`

The last three Action capabilities are optional on older protocol-2 deployments.
Read the live health response. A compatible synchronous submission may return
`VERIFIED` immediately; an async submission returns `EXECUTING` and needs polling.
Protocol 1 is rejected. W&B tracking/observability routes have been removed.

Adding an optional response field or capability does not require a protocol
bump. Removing or changing a required field, route meaning, identity rule, or
mutation lifecycle does. A client pin must pass the real subprocess
compatibility smoke before its submodule gitlink advances.

Terminal snapshots expose Project/Run read models and a `scale` object with
`projects`, `runs`, and `runs_by_project`. There is no observability target page.
Callers can pass `project=<id>` to avoid loading unrelated Projects. An unknown
filter returns 404 rather than an ambiguous empty view. Preserve per-layer
freshness fields; a terminal response does not make every observation current.

Generic Submit/Retry operations use daemon-owned scheduler resource policy from
`action_runtime.scheduler_resource_approval` and `max_gpu_hours_per_action`.
Clients receive only a read-only policy summary in operation metadata and cannot
override those values through the generic operation endpoint. The prepared
Action freezes requested resources, policy cap/mode, and the resulting budget
gate for human approval. Explicit legacy submission endpoints retain their
operator-facing parameters for protocol compatibility.

## Asynchronous Action execution

On servers advertising `async-actions.v1`, `POST /api/actions/execute` and the
submission execute endpoint durably claim
the exact Action, return an `EXECUTING` snapshot immediately, and continue the
controller call in a daemon-owned executor. The HTTP client disconnecting or
stopping its poll does not cancel the mutation. Clients inspect the durable
state with `GET /api/actions/{action_id}`. A daemon restart converts an
interrupted `EXECUTING` record to `RECONCILE_REQUIRED`; it never blindly
reissues the mutation.

On servers advertising `action-resolution.v1`, execution snapshots make retry
semantics explicit:

- `SUBMITTED`: the exact scheduler identity was observed; do not retry;
- `FAILED_BEFORE_SUBMISSION`: the scheduler was not contacted; prepare a new
  Action;
- `NOT_SUBMITTED_SAFE_TO_RETRY`: read-only reconciliation proved no submission;
- `STILL_IN_PROGRESS`: the daemon currently owns execution; wait;
- `UNKNOWN_DO_NOT_RETRY`: the effect is uncertain; reconcile without replay.
- `APPLIED`: a non-submission effect was verified;
- `FAILED_BEFORE_EFFECT` or `NOT_APPLIED_SAFE_TO_RETRY`: a non-submission
  effect was proved absent, so a new Action may be prepared.

`safe_to_retry` and `next_action` are normative companions to `resolution`.
They prevent clients from deriving safety from HTTP timeouts or generic failure
text.

## Deterministic Run derivation

On servers advertising `run-clone.v1`, the direct `run.clone` operation prepares
one campaign update Action. It copies
an authored Run, applies explicit `KEY=VALUE` overrides, and optionally assigns
one profile. Multiple ordered profiles require a `{profile}` placeholder in the
new Run ID and produce an ordered fallback family in the same atomic diff.
This is authoring support, not queue-aware automatic scheduling: selecting and
submitting a fallback remains an explicit reviewed operation.

Reviewed project writes that could not be rolled forward during startup appear
in health's `project_write_recovery_errors`. There is no W&B publisher loop.

## Authentication and binding

Local loopback remains the default. To protect a shared multi-user host, create
an owner-only token and configure it without placing the secret in YAML:

```bash
umask 077
mkdir -p ~/.config/ml-expd
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

## Retrieve the schema from the API

With a base URL that may include a reverse-proxy prefix:

```bash
curl --fail --silent --show-error \
  -H "Authorization: Bearer $ML_EXPD_API_TOKEN" \
  -H 'X-ML-Expd-Client-Protocol: 2' \
  "$ML_EXPD_API_URL/api/v2/openapi.json" > openapi.json
```

The schema describes route bodies and responses. Authentication and protocol
headers are enforced by middleware and are not currently represented as
OpenAPI security schemes/parameters. `/docs` also requires Bearer auth; its
stock Swagger page neither supplies these headers nor preserves every reverse
proxy prefix. Use the authenticated schema and the [client package](../client/README.md)
for a working external workflow. This UI limitation does not weaken API auth.
No `/api/guide` endpoint is currently provided; [the quickstart](api-quickstart.md)
is the usage guide.
