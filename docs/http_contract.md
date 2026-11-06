# HTTP contract

## Negotiate and authenticate

The current protocol is 2. `GET /api/health` returns `server_version`,
`api_protocol_version`, `min_client_protocol_version`, `capabilities`, enabled
policy and `openapi_path` (`/api/v2/openapi.json`). Package versions alone do not
prove a capability is available.
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

`GET /api/submissions/{id}/progress` reports retained evidence for the exact Run and
Attempt, without contacting a provider. It carries `submission_id`, `project`,
`run_id`, `attempt_id` and `availability`, which is `AVAILABLE` once a cloud worker has
reported and `NOT_REPORTED` otherwise, including for all Slurm Submissions.

| Field | Meaning |
| --- | --- |
| `worker_phase` | Last reported worker stage and its observation time; a transition is not a heartbeat |
| `input_delivery` | The most informative input record: a failure first, then unfinished work |
| `inputs` | All retained input delivery records for this Attempt |

`GET /api/experiments/{project}/{run}/submissions` lists the current Attempt's cloud
Submission for clients without saved state.

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

Repeat `name` to filter metric names. The compact response contains normalized
`records`, full history `validation`, `total_records`, `downsampled`,
`missing_names`, frozen `protocol_id` and exact Run/Attempt identity. Validation
runs before filtering or sampling. Unknown query parameters are refused. See
[metrics](metrics.md) for explicit result summaries and scoped requirements.

`ml-exp run cancel` and `ml-exp run retry` POST
`/api/submissions/ID/cancel` and `/api/submissions/ID/retry`. Authorization and
dispatch use `/api/submissions/ID/authorize` and `/api/submissions/ID/execute`, with
the exact returned confirmation. Observation uses `GET /api/submissions/ID`; explicit
`/api/submissions/ID/reconcile` never resubmits the operation. Retry creates a new
Attempt in the same frozen Run from an explicit checkpoint or `--from-start`; it does
not select a checkpoint or submit. See [recovery](recovery.md).

The native API has no SSE stream. Poll the HTTP resources.

Preparation checks actual provider resources and referenced inputs, without estimating worker-package size from dataset inventories. The actual compiled package retains its transport-size bound. A rejected request has no server preparation slot; retrying the exact original POST after correcting the cause preserves the deterministic request identity. GET never creates it.

For a known cloud Submission whose archive is not published, `GET /api/submissions/{id}/artifacts` returns `409 ARTIFACT_NOT_PUBLISHED`; a missing Submission remains 404. This response does not request collection or replay training. Saved client status continues reading the other evidence when optional progress is unavailable, and still fails on authentication errors.

## API resource reference

For a complete install-to-results example use the [client quickstart](api-quickstart.md).
This section is a lookup table for custom clients, Runtime reuse and diagnosis;
request/response bodies are defined by the installed `GET /api/v2/openapi.json`.
Paths are relative to the API base. `P`, `R`, `A`, `S` denote returned project,
Run, Attempt and Submission IDs, not filesystem paths. `GET /api/health` is the
protocol-discovery exception to the headers above.

### Discover, import and build

| Method/path | Resource or purpose |
|---|---|
| `GET /api/health` | Installed version, protocol, capabilities and policy |
| `GET /api/readiness` | Per-backend submission availability and evidence; no provider request |
| `GET /api/executors` | Configured resource/backend choices |
| `GET /api/backends/EXECUTOR_ID/capacity` | Cached capacity with its original observation time |
| `POST /api/backends/EXECUTOR_ID/capacity/refresh` | Read provider evidence into the cache |
| `GET /api/projects` | Registered projects |
| `POST /api/source-imports/archive` | Raw source archive |
| `GET /api/projects/P/source-imports/ARCHIVE_SHA256` | Reconcile a lost publication response |
| `GET /api/projects/P/sources/SOURCE_ID` | Immutable source identity |
| `POST /api/projects/P/runtimes` / `GET` | Freeze source and Dockerfile; list Runtimes |
| `GET /api/projects/P/runtimes/RUNTIME_ID` | Runtime state/receipt |
| `POST …/runtimes/RUNTIME_ID/execute` | Build using returned confirmation |
| `GET …/runtimes/RUNTIME_ID/logs` / `/progress` | Build diagnostics |
| `POST …/runtimes/RUNTIME_ID/reconcile` / `/materialize` | Observe publication without rebuilding; produce a Slurm SIF |

New builds use Dockerfile only, with pinned external bases. Old READY images
retain their workers; an obsolete unbuilt recipe requires new preparation.
Source admission rejects links, unsafe paths and credential-looking filenames;
it is not proof that file contents contain no secrets. See [builds](builds.md).

### Data and server preparation

| Method/path | Resource or purpose |
|---|---|
| `GET /api/projects/P/datasets/E` | Retained dataset identities and readiness |
| `POST /api/projects/P/datasets/E` | Begin one target-backed upload |
| `GET /api/projects/P/datasets/E/A` | Read one asset's status |
| `POST /api/projects/P/datasets/E/import-oss` | Import an approved plain tar from OSS |
| `POST /api/projects/P/experiment-preparations` | Combined Runtime/data/Run/Submission preparation |
| `GET /api/projects/P/experiment-preparations/PREPARATION_ID` | Dependencies, phase and next action |
| `POST …/experiment-preparations/PREPARATION_ID/continue` | Explicitly advance inspected dependencies |

Use the exact [multipart paths and resume rules](data.md#native-http-endpoints),
including zero-based part numbers. Completion establishes destination durability,
not GPU training. SenseCore data-image publication and NAS delivery are distinct.
Slurm/SenseCore target-backed assets do not provide client archive downloads;
approved OSS datasets have a separate signed-download route. Do not use output
routes to retrieve datasets.

### Freeze, review and submit

| Method/path | Resource or purpose |
|---|---|
| `POST /api/projects/P/runs` | Freeze a Run referencing a READY Runtime |
| `POST /api/runs/P/R/submissions` | Preflight and GPU-hour budget review |
| `GET /api/runs/P/R/preparation` | Find the original preparation after a lost reply |
| `GET /api/runs/P/R/submission` | Find the original Submission after a lost reply |
| `POST /api/submissions/S/authorize` | Authorize current prepared gates |
| `POST /api/submissions/S/execute` | Submit using returned confirmation |
| `GET /api/submissions/S` | Durable state and next action |
| `POST /api/submissions/S/reconcile` | Resolve an uncertain original submission |
| `POST /api/submissions/S/refresh` / `/history` | Sample evidence; ingest the next metrics page |
| `POST /api/submissions/S/cancel` / `/retry` / `/renew` | Explicit cancel, new Attempt or replacement gate |

Prepare/build/execute may return HTTP 202: poll the returned identity. An expired
review requires reprepare, not bypassing gates. Unknown effects require
reconciliation, not resubmission. Retry creates a new Attempt of the same Run;
changing source, parameters or restore reference requires a new Run.

### Read evidence and retrieve results

| Method/path | Resource or purpose |
|---|---|
| `GET /api/runs/P/R` / `/attempts` | Run evidence and exact executions |
| `GET /api/attempts/P/R::A/logs` | Stored bounded stdout/stderr |
| `GET /api/attempts/P/R::A/metrics?name=loss` | Compact normalized [metrics](metrics.md); repeat `name` |
| `GET /api/runs/P/R/attempts/A/checkpoints` / `/checkpoints/CHECKPOINT_ID` | Registered checkpoint metadata |
| `POST /api/submissions/S/checkpoint` | Verify and register a complete backend checkpoint |
| `GET /api/runs/P/R/attempts/A/collection` | Training, collection and verification states |
| `POST /api/submissions/S/collect` / `/sync` | Explicit collection; a cloud sync may use a zero-GPU job |
| `GET /api/submissions/S/artifacts` / `/artifacts/SHA256` | Result bundle metadata and download |
| `POST /api/submissions/S/download-ticket` | Short-lived signed part URL |
| `POST /api/submissions/S/acknowledge` | Confirm verified local files and release |
| `GET /api/runs/P/R/tracking` | Frozen [W&B](wandb.md) destination and publication status |

The signed object GET carries no API token. Validate archive and file hashes,
then follow the ticket's acknowledgement contract. Signed URLs are temporary;
do not treat them as artifact identities. Download ACK can release the server
archive; it does not delete checkpoints or the metric journal.

A scheduler terminal state does not prove training/result success. Empty delayed
logs, missing files or a 404 are not proof that backend storage is empty.
Provider diagnostics are bound to the original job; refreshing submits no training.
For incomplete archives, repair and other recovery options see [recovery](recovery.md).
Internal launch/data/checkpoint/record transfer routes use narrow worker
capabilities, not the administrative client token. Scientific evaluation uses an
ordinary Run with checkpoint/data/protocol inputs, not a project-specific metric
or research-question prerequisite.
