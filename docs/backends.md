# Executor capabilities and server preparation

A client uploads code/data and describes a Run. The server matches an executor,
prepares its immutable Runtime and required backend data, freezes the Run and
prepares submission checks. The client needs no Docker, SSH or cloud credentials.
GPU execution still requires explicit `--execute` and a GPU-hour budget.

## What executors declare

`GET /api/executors` includes a nested `capabilities` contract:

| Group | Fields | Meaning |
|---|---|---|
| runtime | platforms, materialization | Supported target platform and OCI/OCI-to-SIF path |
| resources | allocation_mode, inventory_query | Flexible requests or a fixed worker spec; whether this contract implements live inventory |
| network | compute_internet, api_transport | Configured compute egress and direct HTTPS/private TCP relay |
| storage | persistent, storage_id, checkpoint_registration | Configured shared filesystem and metadata registration |
| data | assets, asset_preparation, download_script, preparation | Asset delivery before/inside a job; optional preparation script before/inside a job |
| observability | logs, queue_reason, exit_code, preemption_reason | Evidence the adapter can actually retrieve |
| jobs | exact_submission_lookup, walltime_enforcement | Exact request reconciliation and which layer enforces duration |
| outputs | transfer, offline_export | Adapter result transfer and a generic adapter offline-export hook |

This is a configured contract, not a health probe or a promise of free GPUs.
`storage_id` fingerprints the configured namespace; it does not independently
prove two mounts share a filesystem. Checkpoint matching additionally checks the
registered project's storage scope. Runtime worker features are validated
separately: old images do not gain new behavior from a server upgrade.

PAI DLC is an opt-in single-node adapter; see [PAI configuration and limits](pai.md).
It supports quota queries, approved persistent Dataset mounts and worker
transfers. Uploaded assets require configured personal OSS storage plus CPFS
caching; CPU result recovery remains unavailable. Native OSS filesystem mounts
are provider-blocked and stay disabled.

The existing adapters declare:

| Detail | Slurm | SenseCore ACP | PAI DLC |
|---|---|---|---|
| Runtime | OCI → Apptainer SIF | OCI | OCI |
| Allocation | Flexible, bounded by configured capacity | Exact GPU count of fixed worker spec | Flexible within configured quota/capacity; single node, 1–8 GPUs |
| Uploaded input | Parts over SSH to shared storage | CCR data image → zero-GPU debug ACP → NAS | Approved OSS asset → inside-job CPFS cache |
| Script download | Gateway before allocation when relay is configured | Zero-GPU debug ACP before training when CPU preparation is configured | Inside allocated job |
| Queue/exit/preemption reasons | Supported | Not declared as reliable structured fields | Queue blockers when available; exit/preemption not guaranteed |
| Walltime | Scheduler and worker | Worker | Scheduler and worker; 24-hour cap |
| Server resource status | Slurm partition allocation | Read-only workspace cluster list | Quota allocation snapshot |
| Generic adapter offline-export hook | Not implemented | Not implemented | Not implemented |

The server separately advertises `result-collection.v1` for recovering outputs
from an exact original Attempt. It publishes Slurm shared-directory outputs or
uses a bounded zero-GPU SenseCore CPU job to read NAS and resume the original
archive transfer. This server operation does not require the generic adapter
`outputs.offline_export` hook. Query or start it with `ml-exp collect`; see
[result recovery](recovery.md). It preserves the original scheduler outcome and
does not rerun training or prepare input data.

A separate resource endpoint may expose backend quota/catalogue data. That is
not an admission guarantee. Queue ETA remains null without reliable backend data.

### Slurm allocation status

`GET /api/executors/{id}/status` reports a read-only partition snapshot for Slurm.
`resources` contains `gpus`, `cpus`, and `memory_bytes`, each with `total`,
`allocated`, and `unallocated`. Memory uses Slurm MiB converted to bytes.
`nodes` includes each unique node's state and allocation; normal/debug partitions
sharing nodes are separate views of the same resources, and must not be added together.
`pending_queue_entries` counts visible pending `squeue` rows (an array may be
compressed into one row); it exposes no users, job names or commands.

The server advertises `resources.inventory_query=true` for Slurm. This server
endpoint does not change the generic core adapter's capabilities or executor
matching: configured limits remain distinct from observed allocations.
Snapshots are cached for 25 seconds with one bounded SSH observation at a time.
`observed_at` is the last successful sample and `checked_at` the last completed
query. On failure, prior good data is explicitly `STALE`; without prior evidence
the response is `UNKNOWN` and resource/queue counts are null. A concurrent query
can return `UNKNOWN` / `QUERY_IN_PROGRESS`. Unsupported backends return
`UNSUPPORTED`; unknown executor IDs return 404. Authentication is unchanged.

`unallocated` is arithmetic scheduler allocation headroom, including down/drained
nodes; it is not physical GPU utilization or immediately usable capacity.
Reservations, priorities, node flags and resource combinations can prevent
allocation. `admission_guaranteed` is always false; no ETA, submission or provider
mutation is performed by this GET.

### SenseCore cluster list

`GET /api/backends/sensecore/clusters` reuses the existing workspace-binding
REST query. Optionally pass `executor_id` to choose a configured SenseCore
executor's workspace; by default the first SenseCore executor in sorted ID order
is used, and its ID is returned. Only active bindings visible to that workspace
are listed. No subscription, VPC, credentials or raw provider payload is exposed.

Each item contains `name`, `zone`, `state`, `is_debug`,
`reported_remaining_cards` (provider `reserved_number`) and
`spot_allowance_cards` (complete sum of `spot_status[].spot_quota.device`).
Absent/invalid remaining counts or any unreadable SPOT share yield null, not 0.
These are distinct provider figures; they do not establish physical GPU
utilization, a reliable occupied/total ratio, or availability of a particular
worker spec. Normal and debug pools are labelled; listing a debug pool does not
make it eligible for GPU ACP training. Existing highest-SPOT selection is unchanged.

The same 25-second cache and OBSERVED/STALE/UNKNOWN semantics apply, with a
`clusters: null` payload if no successful evidence exists. Missing configuration
or unknown executor returns 404; a non-SenseCore selector returns 422. This GET
does not create/stop a workload, change pool selection or refresh a frozen Run.

## Matching

Use either an explicit `executor`, or an `executor_selector`:

```json
{
  "executor_selector": {"backend": "slurm", "candidates": ["slurm-large", "slurm-gpu"]},
  "requirements": {"persistent_checkpoints": true, "exit_code": true}
}
```

Candidate order is preference. With only a backend, matching chooses the smallest
configured allocation that fits, then a stable ID. It never silently switches
clouds or ranks unknown free capacity. `POST /api/executors/match` validates before
uploads/builds and returns selected/compatible executors, rejection fields and
`live_availability_checked=false`. Unsupported requirements fail before work.

## Server-owned preparation

`POST /api/projects/{project}/experiment-preparations` accepts:

```json
{
  "runtime": {"source_id": "source.64_HEX", "dockerfile": "Dockerfile", "entrypoint": ["python3", "train.py"]},
  "run": {
    "run_id": "trial-001", "executor": "slurm-gpu",
    "resources": {"gpus": 1, "cpus": 8, "memory_gb": 32, "max_time": "00:10:00"}
  },
  "max_gpu_hours": 0.2
}
```

Replace placeholder identities with API responses. Instead of `runtime`, supply
`run.runtime_id` for an existing image. Uploaded assets use `run.inputs`; selection
and requirements are sibling fields. The API returns 202 and a durable preparation
ID. GET the same resource or its `/progress` endpoint to observe:
`ACCEPTED → BUILDING_RUNTIME → PREPARING_DATA → FREEZING_RUN → PREPARING_DATA_SCRIPT → CHECKING_SUBMISSION`.
The script stage is optional; progress includes its exact operation/job identity and safe diagnostic.
A completed preparation returns the frozen Run and submission ID. `READY` denotes
completed preparation; check `submission.ready` before authorizing GPU execution.

One project/run ID binds one request hash. Repeating an identical POST observes
the original operation; a changed intent returns 409. Concurrent preparations
share a Runtime build and data receipt. Effect markers are persisted before each
request. An interruption or uncertain dependency becomes `RECONCILE_REQUIRED`;
restarting the daemon never replays a build, copy or scheduler request.

Inspect saved dependency IDs first. Once they are independently READY, an explicit
POST to `/experiment-preparations/{id}/continue` can advance the original request.
It retains all effect markers and never retries an uncertain effect. Client:

```bash
ml-exp experiment experiment.json --state trial.state.json --resume --continue-preparation
```

Low-level source, Runtime, Run and submission APIs remain available. The new
one-config workflow requires `server-experiment-preparation.v1`; older clients
continue to use the existing low-level protocol-2 endpoints.

## Data delivery diagnosis

`GET /api/submissions/{id}/progress` includes `input_delivery` from bounded,
exact-Attempt worker logs. New images with `input-delivery-progress.v1` report
waiting for the cache lock, downloading, extracting, verifying, publishing,
mounting and READY/FAILED. Evidence includes HTTP status, received/expected bytes,
elapsed time, observed throughput, timestamp, failed phase and a safe error code.
This is observed log evidence, not a scheduler heartbeat or execution authority.
Logs are collected on polling; timestamps are not refreshed merely by API reads.

URLs, tokens, response bodies, exception messages and private paths are excluded.
Examples distinguish timeout, TLS, HTTP/length rejection, truncated download,
archive/file SHA256 mismatch, archive-manifest mismatch and storage/network errors
(with errno when available). Training does not start after delivery failure.

Historical images retain their old worker. A generic failure produces
`INPUT_DELIVERY_FAILED_DETAILS_UNAVAILABLE`; the server cannot reconstruct an
exception that was never recorded. Rebuild a Runtime for detailed diagnostics;
do not resubmit historical GPU jobs automatically.

### Slurm shared-data staging

New target-backed uploads stream parts through the shared 1 GiB API disk buffer
to a pinned Slurm receiver over SSH. Each part is removed locally after matching
durable ACK; the whole archive and expanded files live only on shared storage.
The backend validates archive/file hashes and atomically publishes the cache.
A bounded read-only check permits reuse before GPU submission. GPU nodes need no
external data access. New frozen inputs require the verified cache and fail closed
if it disappears. Historical desktop-backed assets retain their original staging
path, and old low-level Runs retain their frozen worker download behavior.

### Slurm compute nodes without Internet access

Configure `backend.api_relay` on Slurm executors whose workers must use a connected
private-network gateway. The gateway uses systemd's existing
`systemd-socket-proxyd` with one fixed API destination; example user units are in
`server/examples/slurm-api-relay/`. Bind the socket to the gateway's private IP,
install both units in its `~/.config/systemd/user/`, and enable the socket:
`systemctl --user enable --now ml-expd-api-relay.socket`. The user manager must
remain running after SSH disconnect (linger).

```yaml
compute_internet: false
backend:
  # Existing Slurm settings remain here.
  api_relay:
    endpoint: tcp://192.0.2.10:18443
    origin: https://api.example.org
```

The origin must match the configured artifact API host and port. The worker
connects TCP to the gateway while retaining origin TLS certificate verification,
SNI, Host and exact-Attempt authorization. No general HTTP/SOCKS proxy, TLS
termination or public listener is needed. Bootstrap, data transfers, checkpoint
registration and multipart result uploads share this transport. Worker requests
to other origins fail closed. Network declarations expose configured
`compute_internet` (null if unknown) and `api_transport`; these are not live
reachability claims and do not expose the private endpoint.

A relay-enabled executor requires a newly built Runtime declaring
`api-tcp-relay.v1`; old immutable images and Runs remain unchanged. Stage and
submission preflight verify the gateway's origin TLS path through the configured
SSH gateway before scheduler submission. Unreachable relay errors carry the
fixed `API_RELAY_UNREACHABLE` code; origin mismatch is
`API_RELAY_TARGET_MISMATCH`. TLS and fixed-destination upstream failures can be
investigated with the gateway user service journal.

For these executors, download scripts run in the same SIF on the SSH gateway,
with no GPU allocation, during submission staging. Existing script/parameter
identity, SHA256 file inventory, cache lock and atomic READY receipt are reused.
The allocated worker only verifies the sealed cache; missing or corrupt data
fails before training rather than downloading on an offline GPU node. The script
retains its own timeout; a 60-second launcher allowance covers container startup
and teardown. A cache verification also runs before scheduler submission. An
uncertain stage remains subject to the existing explicit reconciliation rules.


### SenseCore script preparation

With the operator's `data_delivery` CPU profile configured, new SenseCore Runs
freeze that profile and prepare their download script in a RESERVED debug ACP
job with 2 CPUs, 4 GiB and no GPU. The exact training image supplies the helper,
source and script. Already staged inputs are verified and linked at their frozen
mount paths before the script starts. The script writes into NAS; its hash-based
cache and READY receipt are reused by the training worker.
Inputs combined with CPU script preparation must already be staged on NAS
through the CCR data-delivery path (including historical pre-staged assets). API-hosted inputs without that transport are rejected
before building or creating the CPU job (`DATA_SCRIPT_INPUT_TRANSPORT_UNSUPPORTED`);
the platform does not attempt an unauthenticated download in the helper.

`data.preparation` reports `phase=before_job`, `budget=preparation`,
`transport=cpu-acp-shared-storage` and `max_timeout_seconds=3600`. Queue and script
limits are bounded independently; this work does not consume the training GPU
allocation or its walltime. Set `requirements.data_preparation_before_job=true`
to reject an executor that cannot provide this. Matching rejects an excessive
script timeout before building. It is not a free-resource or startup-time guarantee.

The CPU intent is persisted before its single REST create. Confirmed UID, image,
script and profile are checked when observing the original job. A lost reply or
missing receipt requires reconciliation; continue never creates a replacement.
New Runs freeze a callback wrapper identity. After the script, that wrapper
rereads the sealed NAS receipt and verifies actual file SHA256 before sending an
independent expiring capability callback. Provider stdout indexing is not required
for readiness. A fast callback waits for the original confirmed UID; wrong or
replacement UIDs are rejected. Verified data may be READY even if callback ACK
failure makes the CPU job fail; its original job status is preserved. The token
is removed before the user's script runs. Config changes do not rebind old Runs;
historical inside-job/stdout receipt modes remain unchanged. See [data](data.md).

## Internal boundary for another backend

Scheduler capabilities, image staging, submission, observation and cancellation
continue to use the existing core `Backend`/`BackendRegistry`. The server's
`backends/operations.py` contains the additional preparation and independent
collection operations: `prepare_asset`, `prepare_script`, `script_progress`,
`confirm_terminal` and `collect`. Experiment preparation and result recovery
select an installed adapter through one explicit mapping. Adding an adapter does
not require new client provider branches, plugin discovery or another journal.
A third fake adapter exercises this server boundary in tests; it is not a claim
that a third production scheduler is installed.

SenseCore data copy, download-script preparation and result recovery share the
CPU profile/specification/document checks and exact name/UID observation helpers
in `backends/sensecore_cpu.py`. They retain separate durable intents, timeouts,
retry rules and receipts. A confirmed UID is persisted even when a data READY
callback arrives first. An old interrupted operation without a UID may bind only
the single exact, owned job found under its original name. Subsequent observations
and timeout cancellation reject another UID. Missing or ambiguous jobs never
cause create to be replayed. The provider's batch-stop API is still name based;
checking the UID first does not make it an atomic compare-and-stop operation.
Historical valid profiles keep their original cluster/quota choices and frozen
manifests; sharing validation does not move old CPU operations to another pool.
