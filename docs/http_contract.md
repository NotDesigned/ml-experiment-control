# HTTP contract

## Negotiate and authenticate

The current protocol is 2. `GET /api/health` returns `server_version`,
`api_protocol_version`, `min_client_protocol_version`, `capabilities`, enabled
policy and `openapi_path` (`/api/v2/openapi.json`). Package versions alone do not
prove a capability is available; frozen worker images have their own capabilities.
Platform metrics are accepted and queried independently of W&B. Without the
optional SDK, publication reports `WANDB_SDK_NOT_INSTALLED`; it does not rewrite
the frozen Run route or discard accepted records.

Health also returns `supported_capabilities`
(implemented protocols) and `deployment` (the readiness document). Mutation
permissions and executor capabilities remain separate checks; these lists do not
grant authorization. Authenticated `GET /api/readiness` returns the same document
and sends no provider request. `training=INCOMPLETE` is always reported today with
`training_status_scope=acceptance_evidence`: it describes recorded acceptance, not
whether submission is available. Read `submission_available_backends` and each
backend's `submission_blockers`. Clients should use the normal experiment preflight
before submitting a job.

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
The daemon binds loopback only and has no native TLS. A TLS proxy fronts it, and
remote clients use HTTPS.

Retrieve the schema from `/api/v2/openapi.json` with `ml-exp get /api/v2/openapi.json`
or an authenticated protocol-aware HTTP client. It may not list every native route.
Use the [quickstart](api-quickstart.md) for the workflow.

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

## Backend-independent progress

`GET /api/submissions/{id}/progress` reports the exact Run/Attempt for Slurm,
SenseCore ACP and PAI DLC through the same optional evidence fields:

| Field | Meaning |
| --- | --- |
| `scheduler_state`, `queue` | Normalized allocation state and available native blocker, position, observation time and stale flag |
| `worker_phase` | Last supported worker stage, original timestamp, age, event ID and sequence; a transition is not a heartbeat |
| `last_metric` | Last received metric observation and journal provenance, retaining unit, protocol, variant and checkpoint context |
| `input_delivery` | Bounded exact-Attempt data transfer evidence |
| `result_collection` | Independent training result, publication and verification status |

Missing fields on older servers and null evidence mean unknown. A scheduler
terminal state is not overwritten by an earlier worker stage. The last received
metric is not the largest step or best model, and an absent metric timestamp
does not prove recent progress. These GETs do not refresh providers, create
record scopes, retry training or change publication routes.

The client combines these fields with indexed Run evidence and collection
receipts for `ml-exp run status`. Its presentation does not branch on backend type;
unsupported diagnostics remain unavailable. See [recovery](recovery.md) for
saved-state observation and the distinct meanings of retry and checkpoint resume.

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

HTTP failures preserve their status and sanitized code without becoming
automatically retried. The client prints one sanitized line to stderr and exits
nonzero, excluding query strings, raw bodies, headers and signed URLs. Inspect the
original resource's diagnostics, and observe saved IDs before deciding on an
explicit retry.

## Client observation and explicit training operations

`ml-exp run logs` and `ml-exp run metrics` use
`GET /api/attempts/P/R::A/logs` and `GET /api/attempts/P/R::A/metrics`. When no
Attempt is supplied, the client first reads `GET /api/runs/P/R/attempts` and binds
the request to its current Attempt. These reads do not submit work.

With `metric-records.v1`, the client requests `format=records` and repeats `name`
to filter metric names. The compact response contains normalized `records`, full
history `validation`, `total_records`, `downsampled`, `missing_names`, frozen
`protocol_id` and exact Run/Attempt identity. Validation runs before filtering or
sampling. It omits the duplicate legacy points/keys representation and filesystem
paths. Omitting `format` keeps the old response; `keys` remains a legacy field
filter. Mixing `format=records` with `keys` is rejected. See [metrics](metrics.md)
for explicit result summaries and scoped requirements.

`ml-exp run cancel` and `ml-exp run retry` POST
`/api/submissions/ID/cancel` and `/api/submissions/ID/retry`. Authorization and
dispatch use `/api/submissions/ID/authorize` and `/api/submissions/ID/execute`, with
the exact returned confirmation. Observation uses `GET /api/submissions/ID`; explicit
`/api/submissions/ID/reconcile` never resubmits the operation. Retry creates a new
Attempt in the same frozen Run from an explicit checkpoint or `--from-start`; it does
not select a checkpoint or submit. See [recovery](recovery.md).

The native API has no SSE stream. Poll the HTTP resources.

The client also requests `GET /api/submissions/{id}/progress` and tolerates its
absence. The current server registers no GET handler for that path, so treat the
field table above as optional evidence that may be unavailable.

Preparation checks actual provider resources and referenced inputs, without estimating worker-package size from dataset inventories. The actual compiled package retains its transport-size bound. A rejected request has no server preparation slot; retrying the exact original POST after correcting the cause preserves the deterministic request identity. GET never creates it.

For a known cloud Submission whose archive is not published, `GET /api/submissions/{id}/artifacts` returns `409 ARTIFACT_NOT_PUBLISHED`; a missing Submission remains 404. This response does not request collection or replay training. Saved client status continues reading the other evidence when optional progress is unavailable, and still fails on authentication errors.
